import numpy as np
import torch
from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn

from mmdet3d.core import (PseudoSampler, box3d_multiclass_nms_custom, limit_period,
                          xywhr2xyxyr)
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models import HEADS
from ..builder import build_loss
from .custom_train_mixins import CustomAnchorTrainMixin
from ...utils import GradReverse

@HEADS.register_module()
class CustomAnchor3DHead(BaseModule, CustomAnchorTrainMixin):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
        loss_custom (dict): Config of custom attribute loss.
    """

    def __init__(self,
                 num_classes,
                 num_custom,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2),
                 loss_custom=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 custom_cfg=dict(
                     type='density', 
                     divide=torch.Tensor([5, 25, 50])),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_custom = num_custom
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.custom_cfg = custom_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.fp16_enabled = False

        # build anchor generator
        self.anchor_generator = build_anchor_generator(anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors
        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.use_sigmoid_custom = loss_custom.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        if not self.use_sigmoid_cls:
            self.num_custom += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)
        self.loss_custom = build_loss(loss_custom)
        self.fp16_enabled = False

        # decide whether to use GRL
        self.grad_reverse = self.custom_cfg.get('grad_reverse', False)

        # decide whether to stop gradient of dets in certain range
        self.stop_gradient = self.custom_cfg.get('stop_gradient', False)

        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.custom_out_channels = self.num_anchors * self.num_custom
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        self.conv_custom = nn.Conv2d(self.feat_channels, self.custom_out_channels, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        if self.grad_reverse:
            reverse_weight = self.custom_cfg.get('reverse_weight', 0.1)
            if self.stop_gradient:
                x_detach = x.detach()
                custom_pred = self.conv_custom(GradReverse.apply(x, reverse_weight))
                custom_pred_detach = self.conv_custom(x_detach)
                custom_pred = {'attach': custom_pred, 'detach': custom_pred_detach}
            else:
                reverse_weight = self.custom_cfg.get('reverse_weight', 0.1)
                custom_pred = self.conv_custom(GradReverse.apply(x, reverse_weight))
        else:
            custom_pred = self.conv_custom(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds, custom_pred

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, input_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[torch.Tensor]]: Anchors of each image, valid flags \
                of each image.
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def loss_single(self, cls_score, bbox_pred, dir_cls_preds, custom_preds, labels,
                    label_weights, bbox_targets, bbox_weights, dir_targets,
                    dir_weights, custom_targets, custom_weights, num_total_samples):
        """Calculate loss of Single-level results.

        Args:
            cls_score (torch.Tensor): Class score in single-level.
            bbox_pred (torch.Tensor): Bbox prediction in single-level.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single-level.
            custom_preds (torch.Tensor): Class score of custom attributes.
            labels (torch.Tensor): Labels of class.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_targets (torch.Tensor): Targets of bbox predictions.
            bbox_weights (torch.Tensor): Weights of bbox loss.
            dir_targets (torch.Tensor): Targets of direction predictions.
            dir_weights (torch.Tensor): Weights of direction loss.
            custom_targets (torch.Tensor): Taragets of custom attribute.
            custom_weights (torch.Tensor): Weights of custom predictions.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox \
                and direction, respectively.
        """
    
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        
        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # custom loss
        # build custom classification label
        divide = self.custom_cfg.get('divide', None)
        custom_targets = custom_targets.reshape(-1)
        custom_weights = custom_weights.reshape(-1)
        if self.stop_gradient:
            custom_preds_attach = custom_preds['attach']
            custom_preds_detach = custom_preds['detach']
            custom_preds_attach = custom_preds_attach.permute(0, 2, 3, 1).reshape(-1, self.num_custom)
            custom_preds_detach = custom_preds_detach.permute(0, 2, 3, 1).reshape(-1, self.num_custom)
            custom_targets_ = custom_targets.clone() # copy for division
            stop_range = self.custom_cfg.get('stop_range', [0, 1])
            custom_weights_attach = custom_weights.clone()
            custom_weights_detach = custom_weights.clone()
            for label_d in range(len(divide)-1):
                custom_targets[(divide[label_d]<custom_targets_) & (custom_targets_ <= divide[label_d+1])] = label_d
                if label_d in stop_range:
                    custom_weights_attach[(divide[label_d]<custom_targets_) & (custom_targets_ <= divide[label_d+1])] = 0
                else:
                    custom_weights_detach[(divide[label_d]<custom_targets_) & (custom_targets_ <= divide[label_d+1])] = 0
            # select pos targets
            pos_custom_targets = custom_targets[pos_inds]
            pos_custom_weights_attach = custom_weights_attach[pos_inds]
            pos_custom_weights_detach = custom_weights_detach[pos_inds]
            pos_custom_pred_attach = custom_preds_attach[pos_inds] 
            pos_custom_pred_detach = custom_preds_detach[pos_inds] 
        else:
            custom_preds = custom_preds.permute(0, 2, 3, 1).reshape(-1, self.num_custom)

            custom_targets_ = custom_targets.clone() # copy for division
            for label_d in range(len(divide)-1):
                custom_targets[(divide[label_d]<custom_targets_) & (custom_targets_ <= divide[label_d+1])] = label_d
            # select pos targets
            pos_custom_targets = custom_targets[pos_inds]
            pos_custom_weights = custom_weights[pos_inds]
            pos_custom_pred = custom_preds[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_preds = dir_cls_preds[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)
            
            if self.stop_gradient:
                loss_custom_attach = self.loss_custom(
                pos_custom_pred_attach, pos_custom_targets.long(), pos_custom_weights_attach, avg_factor=num_total_samples)
                loss_custom_detach = self.loss_custom(
                pos_custom_pred_detach, pos_custom_targets.long(), pos_custom_weights_detach, avg_factor=num_total_samples)
                loss_custom = loss_custom_detach + loss_custom_attach
            else:
                loss_custom = self.loss_custom(
                pos_custom_pred, pos_custom_targets.long(), pos_custom_weights, avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            loss_custom = pos_custom_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()

        return loss_cls, loss_bbox, loss_dir, loss_custom

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             custom_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore=None):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.
            gt_custom (list[torch.Tensor]): Gt custom attritube labels of each sample.
            input_metas (list[dict]): Contain pcd and img's meta info and custom attritube info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss custom (list[torch.Tensor]): Custom attribute loss.
                - loss_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        custom_channels = self.custom_out_channels if self.use_sigmoid_custom else 1
        
        '''build gt custom
        '''
        custom_attribute = self.custom_cfg.get('type', 'distance')
        if custom_attribute == 'distance':
            centers = [t.tensor[:, :3] for t in gt_bboxes]
            gt_custom = [torch.norm(c, p=2, dim=1) for c in centers]
        elif custom_attribute == 'density':
            gt_num_points = [info['gt_num_points'] for info in input_metas]
            gt_num_points = [torch.from_numpy(g).float() for g in gt_num_points]
            dims = [t.tensor[:, 3:6] for t in gt_bboxes]
            surf_area = [(d[:, 0]+d[:, 1])*d[:, 2] for d in dims]
            gt_custom = [num/s for num, s in zip(gt_num_points, surf_area)]

        gt_custom = [t.cuda() for t in gt_custom]
        
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            gt_bboxes,
            input_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            gt_custom_list=gt_custom,
            num_classes=self.num_classes,
            num_custom=self.num_custom,
            label_channels=label_channels,
            custom_channels=custom_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, custom_targets_list, custom_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # num_total_samples = None
        losses_cls, losses_bbox, losses_dir, losses_custom = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            custom_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            custom_targets_list,
            custom_weights_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir, loss_custom=losses_custom)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   custom_scores,
                   input_metas,
                   cfg=None,
                   rescale=False):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        mlvl_anchors = [
            anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
        ]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]
            if self.stop_gradient:
                custom_pred_list = [
                    custom_scores[i]['attach'][img_id].detach() for i in range(num_levels)
                ]
            else:
                custom_pred_list = [
                    custom_scores[i][img_id].detach() for i in range(num_levels)
                ]
            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, custom_pred_list, mlvl_anchors,
                                               input_meta, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          custom_scores,
                          mlvl_anchors,
                          input_meta,
                          cfg=None,
                          rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_custom_scores = []
        for cls_score, bbox_pred, dir_cls_pred, custom_score, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, custom_scores, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            custom_score = custom_score.permute(1,2,0).reshape(-1, self.num_custom)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            
            if self.use_sigmoid_custom:
                custom_scores = custom_score.sigmoid()
            else:
                custom_scores = custom_score.softmax(-1)
            
            custom_score = torch.max(custom_score, dim=-1)[1]

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                custom_scores = custom_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_custom_scores.append(custom_scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        mlvl_custom_scores = torch.cat(mlvl_custom_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms_custom(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, cfg.max_num,
                                       cfg, mlvl_dir_scores, mlvl_custom_scores=mlvl_custom_scores)
        bboxes, scores, labels, custom_scores, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels, custom_scores # actually, custom scores means custom category
