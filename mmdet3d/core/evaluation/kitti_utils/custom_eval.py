import gc
import io as sysio
import numba
import numpy as np


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def custom_clean_data(gt_anno, dt_anno, current_class, difficulty, cus):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno['bbox'][i]
        gt_name = gt_anno['name'][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == 'Pedestrian'.lower()
              and 'Person_sitting'.lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno['truncated'][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])
                or (gt_anno['custom'][i] != cus)):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
    # for i in range(num_gt):
        if gt_anno['name'][i] == 'DontCare':
            dc_bboxes.append(gt_anno['bbox'][i])
    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno['bbox'][i, 3] - dt_anno['bbox'][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=False)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]) + qbox_area -
                              iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (
                    min(boxes[i, 1], qboxes[j, 1]) -
                    max(boxes[i, 1] - boxes[i, 4],
                        qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=False)
def custom_compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False,
                           cur_cus=0):
                           
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -3]
    dt_custom_pred = dt_datas[:, -1]
    dt_custom = dt_datas[:, -2]
    gt_custom = gt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size

    # matched_det = [False] * det_size
    # matched_gt_idx = [0] * det_size

    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity, correct_pred = 0, 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        # if (valid_detection != NO_DETECTION) and ignored_gt[i] == 0:
        #     matched_det[det_idx] = True
        #     matched_gt_idx[det_idx] = i

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            if (gt_custom[i]==cur_cus) and dt_custom_pred[det_idx]==cur_cus:
                correct_pred += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    
    # '''add more code to set unmatched dt out of range as ignored'''
    # ignored_det = [1 if ((not matched_det[i]) and (dt_custom[i] != cur_cus) and (sign != -1))
    #                             else sign for i, sign in enumerate(ignored_det)]
    # ignored_det = [ignored_gt[matched_gt_idx[i]] if (matched_det[i] and (sign != 1))
    #                             else sign for i, sign in enumerate(ignored_det)]
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i] 
                     or (dt_custom[i] != cur_cus))):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (dt_custom[j] != cur_cus):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx], correct_pred


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=False)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False,
                             cur_cus=0):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _, correct_pred = custom_compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos,
                cur_cus=cur_cus)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            pr[t, 4] += correct_pred
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a['bbox'] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a['bbox'] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a['location'][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a['dimensions'][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate(
                [a['location'][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a['dimensions'][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = bev_box_overlap(gt_boxes,
                                           dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = d3_box_overlap(gt_boxes,
                                          dt_boxes).astype(np.float64)
        else:
            raise ValueError('unknown metric')
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _custom_prepare_data(gt_annos, dt_annos, current_class, difficulty, cus):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = custom_clean_data(gt_annos[i], dt_annos[i], current_class, difficulty, cus)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]['bbox'], gt_annos[i]['alpha'][..., np.newaxis], gt_annos[i]['custom'][..., np.newaxis]
        ], 1)
        dt_datas = np.concatenate([
            dt_annos[i]['bbox'], dt_annos[i]['alpha'][..., np.newaxis],
            dt_annos[i]['score'][..., np.newaxis], dt_annos[i]['custom'][..., np.newaxis], dt_annos[i]['custom_pred'][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def custom_eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=200,
               num_custom=3):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        current_classes (list[int]): 0: car, 1: pedestrian, 2: cyclist.
        difficultys (list[int]): Eval difficulty, 0: easy, 1: normal, 2: hard
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps (float): Min overlap. format:
            [num_overlap, metric, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_custom, num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_custom, num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    custom = np.zeros(
        [num_custom, num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_custom, num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for cus in range(0,num_custom):  
        for m, current_class in enumerate(current_classes):
            for idx_l, difficulty in enumerate(difficultys):
                rets = _custom_prepare_data(gt_annos, dt_annos, current_class, difficulty, cus)
                (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
                dontcares, total_dc_num, total_num_valid_gt) = rets
                for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                    thresholdss = []
                    for i in range(len(gt_annos)):
                        rets = custom_compute_statistics_jit(
                            overlaps[i],
                            gt_datas_list[i],
                            dt_datas_list[i],
                            ignored_gts[i],
                            ignored_dets[i],
                            dontcares[i],
                            metric,
                            min_overlap=min_overlap,
                            thresh=0.0,
                            compute_fp=False,
                            cur_cus=cus)
                        tp, fp, fn, similarity, thresholds, correct_pred = rets
                        thresholdss += thresholds.tolist()
                    thresholdss = np.array(thresholdss)
                    thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                    thresholds = np.array(thresholds)
                    pr = np.zeros([len(thresholds), 5])
                    idx = 0
                    for j, num_part in enumerate(split_parts):
                        gt_datas_part = np.concatenate(
                            gt_datas_list[idx:idx + num_part], 0)
                        dt_datas_part = np.concatenate(
                            dt_datas_list[idx:idx + num_part], 0)
                        dc_datas_part = np.concatenate(
                            dontcares[idx:idx + num_part], 0)
                        ignored_dets_part = np.concatenate(
                            ignored_dets[idx:idx + num_part], 0)
                        ignored_gts_part = np.concatenate(
                            ignored_gts[idx:idx + num_part], 0)
                        fused_compute_statistics(
                            parted_overlaps[j],
                            pr,
                            total_gt_num[idx:idx + num_part],
                            total_dt_num[idx:idx + num_part],
                            total_dc_num[idx:idx + num_part],
                            gt_datas_part,
                            dt_datas_part,
                            dc_datas_part,
                            ignored_gts_part,
                            ignored_dets_part,
                            metric,
                            min_overlap=min_overlap,
                            thresholds=thresholds,
                            compute_aos=compute_aos,
                            cur_cus=cus)
                        idx += num_part
                    for i in range(len(thresholds)):
                        recall[cus, m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[cus, m, idx_l, k, i] = pr[i, 0] / (
                            pr[i, 0] + pr[i, 1])
                        custom[cus, m, idx_l, k, i] = pr[i,-1] / pr[i, 0]
                        if compute_aos:
                            aos[cus, m, idx_l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                    for i in range(len(thresholds)):
                        precision[cus, m, idx_l, k, i] = np.max(
                            precision[cus, m, idx_l, k, i:], axis=-1)
                        recall[cus, m, idx_l, k, i] = np.max(
                            recall[cus, m, idx_l, k, i:], axis=-1)
                        custom[cus, m, idx_l, k, i] = np.max(
                            custom[cus, m, idx_l, k, i:], axis=-1
                        )
                        if compute_aos:
                            aos[cus, m, idx_l, k, i] = np.max(
                                aos[cus, m, idx_l, k, i:], axis=-1)
    ret_dict = {
        'recall': recall,
        'precision': precision,
        'orientation': aos,
        'custom': custom
    }
    # clean temp variables
    del overlaps
    del parted_overlaps

    gc.collect()
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def custom_do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            eval_types=['bbox', 'bev', '3d'],
            num_custom=3):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    mAP_bbox = None
    mAP_aos = None
    if 'bbox' in eval_types:
        ret = custom_eval_class(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            0,
            min_overlaps,
            compute_aos=('aos' in eval_types),
            num_custom=num_custom)
        # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
        mAP_bbox = get_mAP(ret['precision'])
        if 'aos' in eval_types:
            mAP_aos = get_mAP(ret['orientation'])

    mAP_bev = None
    if 'bev' in eval_types:
        ret = custom_eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                         min_overlaps, num_custom=num_custom)
        mAP_bev = get_mAP(ret['precision'])

    mAP_3d = None
    if '3d' in eval_types:
        ret = custom_eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                         min_overlaps, num_custom=num_custom)
        mAP_3d = get_mAP(ret['precision'])
    
    mAP_custom = get_mAP(ret['custom'])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_custom


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos, num_custom=3):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = custom_do_eval(gt_annos, dt_annos,
                                                 current_classes, min_overlaps,
                                                 compute_aos, num_custom=num_custom)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def custom_kitti_eval(gt_annos,
               dt_annos,
               current_classes,
               eval_types=['bbox', 'bev', '3d'],
               num_custom=3):
    """KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, 'must contain at least one evaluation type'
    if 'aos' in eval_types:
        assert 'bbox' in eval_types, 'must evaluate bbox when evaluating aos'
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5], [0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.7, 0.5, 0.5, 0.7, 0.5]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25],
                            [0.5, 0.25, 0.25, 0.5, 0.25]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    pred_alpha = False
    valid_alpha_gt = False
    for anno in dt_annos:
        mask = (anno['alpha'] != -10)
        if anno['alpha'][mask].shape[0] != 0:
            pred_alpha = True
            break
    for anno in gt_annos:
        if anno['alpha'][0] != -10:
            valid_alpha_gt = True
            break
    compute_aos = (pred_alpha and valid_alpha_gt)
    if compute_aos:
        eval_types.append('aos')

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPcustom = custom_do_eval(gt_annos, dt_annos,
                                             current_classes, min_overlaps,
                                             eval_types, num_custom=num_custom)

    ret_dict = {}
    custom_idx = ['\nrange0\n','\nrange1\n', '\nrange2\n']
    difficulty = ['easy', 'moderate', 'hard']
    for cur in range(num_custom):
        result += custom_idx[cur]
        for j, curcls in enumerate(current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            curcls_name = class_to_name[curcls]
            for i in range(min_overlaps.shape[0]):
                # prepare results for print
                result += ('{} AP@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                    curcls_name, *min_overlaps[i, :, j]))
                if mAPbbox is not None:
                    result += 'bbox AP:{:.4f}, {:.4f}, {:.4f}\n'.format(
                        *mAPbbox[cur, j, :, i])
                if mAPbev is not None:
                    result += 'bev  AP:{:.4f}, {:.4f}, {:.4f}\n'.format(
                        *mAPbev[cur, j, :, i])
                if mAP3d is not None:
                    result += '3d   AP:{:.4f}, {:.4f}, {:.4f}\n'.format(
                        *mAP3d[cur, j, :, i])

                if compute_aos:
                    result += 'aos  AP:{:.2f}, {:.2f}, {:.2f}\n'.format(
                        *mAPaos[cur, j, :, i])
                if mAPcustom is not None:
                    result += 'custom AP:{:.4f}, {:.4f}, {:.4f}\n'.format(
                        *mAPcustom[cur, j, :, i])
                # prepare results for logger
                for idx in range(3):
                    if i == 0:
                        postfix = f'{difficulty[idx]}_strict'
                    else:
                        postfix = f'{difficulty[idx]}_loose'
                    prefix = f'KITTI/{curcls_name}'
                    if mAP3d is not None:
                        ret_dict[f'range_{cur}_{prefix}_3D_{postfix}'] = mAP3d[cur, j, idx, i]
                    if mAPbev is not None:
                        ret_dict[f'range_{cur}_{prefix}_BEV_{postfix}'] = mAPbev[cur, j, idx, i]
                    if mAPbbox is not None:
                        ret_dict[f'range_{cur}_{prefix}_2D_{postfix}'] = mAPbbox[cur, j, idx, i]
                    if mAPcustom is not None:
                        ret_dict[f'range_{cur}_{prefix}_custom_{postfix}'] = mAPcustom[cur, j, idx, i]

        # calculate mAP over all classes if there are multiple classes
        if len(current_classes) > 1:
            # prepare results for print
            result += ('\nOverall AP@{}, {}, {}:\n'.format(*difficulty))
            if mAPbbox is not None:
                mAPbbox_mean = mAPbbox.mean(axis=1)
                result += 'bbox AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAPbbox_mean[cur, :, 0])
            if mAPbev is not None:
                mAPbev_mean = mAPbev.mean(axis=1)
                result += 'bev  AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAPbev_mean[cur, :, 0])
            if mAP3d is not None:
                mAP3d_mean = mAP3d.mean(axis=1)
                result += '3d   AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP3d_mean[cur, :, 0])
            if compute_aos:
                mAPaos_mean = mAPaos.mean(axis=1)
                result += 'aos  AP:{:.2f}, {:.2f}, {:.2f}\n'.format(*mAPaos_mean[cur, :, 0])
            if mAPcustom is not None:
                mAPcustom_mean = mAPcustom.mean(axis=1)
                result += 'custom AP:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAPcustom_mean[cur, :, 0])
            # prepare results for logger
            for idx in range(3):
                postfix = f'{difficulty[idx]}'
                if mAP3d is not None:
                    ret_dict[f'range_{cur}_KITTI/Overall_3D_{postfix}'] = mAP3d_mean[cur, idx, 0]
                if mAPbev is not None:
                    ret_dict[f'range_{cur}_KITTI/Overall_BEV_{postfix}'] = mAPbev_mean[cur, idx, 0]
                if mAPbbox is not None:
                    ret_dict[f'range_{cur}_KITTI/Overall_2D_{postfix}'] = mAPbbox_mean[cur, idx, 0]
                if mAPcustom is not None:
                    ret_dict[f'range_{cur}_KITTI/Overall_custom_{postfix}'] = mAPcustom_mean[cur, idx, 0]

    return result, ret_dict

