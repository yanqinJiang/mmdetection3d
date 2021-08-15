import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module()
class SECONDFPN_EMBEDDING(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 custom_type='distance',
                 map_enabled=False,
                 point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN_EMBEDDING, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.out_channels = out_channels
        self.custom_type = custom_type
        self.map_enabled = map_enabled
        self.point_cloud_range = point_cloud_range
        self.fp16_enabled = False

        if self.map_enabled:
            in_channels = [c+1 for c in in_channels] # extra dim for distance/density map
        deblocks = []
        self.in_channels = in_channels

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x, points=None):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        # build distance map
        x = list(x) # change it to list, cause tuple is immutable
        if self.map_enabled:
            input_sizes = [feat.shape[2:] for feat in x]

        if self.custom_type == 'distance':
            for i in range(len(x)):
                xi = torch.linspace(self.point_cloud_range[0], self.point_cloud_range[3], input_sizes[i][1], device='cuda')
                yi = torch.linspace(self.point_cloud_range[1], self.point_cloud_range[4], input_sizes[i][0], device='cuda')
                xiyi = torch.meshgrid(yi, xi)
                map = torch.sqrt(xiyi[0]**2 + xiyi[1]**2)

                map = map.expand(x[i].shape[0], 1, -1, -1)
                x[i] = torch.cat([x[i], map], dim=1)

        elif self.custom_type == 'density':
            for i in range(len(x)):
                delta_x = (self.point_cloud_range[3] - self.point_cloud_range[0]) / input_sizes[i][0]
                delta_y = (self.point_cloud_range[4] - self.point_cloud_range[1]) / input_sizes[i][1]
                map_list = []
                # map = torch.zeros(len(x[0]), 1, input_sizes[i][0], input_sizes[i][1], device='cuda')
                for j in range(len(x[0])):
                    map = torch.zeros(input_sizes[i], device='cuda')
                    pt_x = points[j][:, 0].clone()
                    pt_y = points[j][:, 1].clone()
                    pt_x /= delta_x
                    pt_y /= delta_y
                    pt_x = torch.ceil(pt_x).long()
                    pt_y = torch.ceil(pt_y).long()
                    pt_x = torch.clamp(pt_x, min=0, max=map.shape[0]-1)
                    pt_y = torch.clamp(pt_y, min=0, max=map.shape[1]-1)
                    index = torch.stack([pt_x, pt_y], dim=0)
                    uni_index, count = torch.unique(index, dim=1, return_counts=True)
                    map[tuple(uni_index)] += count
                    map_list.append(map)
                    # for k in range(len(pt_x)):
                    #     map[j][0][pt_x[k]][pt_y[k]] += 1
                map = torch.stack(map_list, dim=0)
                map = map.unsqueeze(1)
                x[i] = torch.cat([x[i], map], dim=1)

        x = tuple(x)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]
