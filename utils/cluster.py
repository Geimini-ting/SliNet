import torch


def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / (union + 0.0000001)

    return out if use_batch else out.squeeze(0)


def diou(box_a, box_b, beta=1.0, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)
    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-7))
    out = inter / area_a if iscrowd else inter / union - D ** beta
    return out if use_batch else out.squeeze(0)


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=True):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.
    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """
    iou_thr = nms_cfg
    num_classes = 1

    bboxes = multi_bboxes[:, None].expand(
        multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device='cuda')
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels
    scores, idx = scores.sort(0, descending=True)
    bboxes = bboxes[idx]
    labels = labels[idx]
    box = bboxes + labels.unsqueeze(1).expand_as(bboxes) * 40

    iouu = diou(box, box, 1)
    iou = (iouu + 0).triu_(diagonal=1)
    B = iou
    for i in range(999):
        A = B
        maxA = A.max(dim=0)[0]
        E = (maxA <= iou_thr).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    # Now just filter out the ones higher than the threshold
    B = torch.triu(iouu).mul(E)
    keep = []
    for i in range(len(idx)):
        if maxA[i] <= iou_thr:
            keep.append(idx[i])
    keep = torch.tensor(keep)

    weights = (torch.exp(-(1 - (B * (B > 1).float())) ** 2 / 0.025)) * (scores.reshape((1, len(scores))))
    bboxes = torch.mm(weights, bboxes).float() / weights.sum(1, keepdim=True)
    # Only keep the top max_num highest scores across all classes
    if max_num > 0:
        scores = scores[keep][:max_num]
        labels = labels[keep][:max_num]
        bboxes = bboxes[keep][:max_num]
        keep = keep[:max_num]

    dets = torch.cat([bboxes, scores[:, None]], dim=1)
    if return_inds:
        return keep
    else:
        return dets, labels


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.
    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.
    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
