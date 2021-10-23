from typing import Counter
import torch
import numpy as np


def _midpoint_to_corners(box):
    return torch.Tensor([
        box[..., 0:1] - box[..., 2:3] / 2,  # x1
        box[..., 1:2] - box[..., 3:4] / 2,  # y1
        box[..., 0:1] + box[..., 2:3] / 2,  # x2
        box[..., 1:2] + box[..., 3:4] / 2,  # y2
    ])


def intersection_over_union(box_preds, ground_boxes, box_format):
    """
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    """

    # convert midpoint box to corners
    if box_format == 'midpoint':
        box1 = _midpoint_to_corners(box_preds)
        box2 = _midpoint_to_corners(ground_boxes)

    if box_format == 'corners':
        box1 = box_preds
        box2 = ground_boxes

    x1, y1 = torch.max(box1[..., 0:1], box1[..., 0:1]), torch.max(
        box1[..., 1:2], box2[..., 1:2])
    x2, y2 = torch.min(box1[..., 2:3], box1[..., 2:3]), torch.min(
        box1[..., 3:4], box2[..., 3:4])

    intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1[..., 2:3] - box1[..., 0:1])
                    * (box1[..., 1:2] - box1[..., 3:4]))
    box2_area = abs((box2[..., 2:3] - box2[..., 0:1])
                    * (box2[..., 1:2] - box2[..., 3:4]))

    return intersection_area / (box1_area + box2_area - intersection_area + 1e-7)


def non_max_suppresion(
    bboxes,
    iou_threshold=0.1,
    threshold=0.5,
    box_format='corners'
):
    """
        bboxes (tensor): [[class, pred, x1, y1, x2, y2]]
    """

    bboxes = [
        bbox if box_format == 'corners' else _midpoint_to_corners(bbox)
        for bbox in bboxes if bbox[1] >= threshold
    ]

    # sort by pred in desc order
    bboxes = sorted(bboxes,
                    key=lambda x: x[1],  # by pred score
                    reverse=True)

    rtn_bboxes = []

    while bboxes:
        curr = bboxes.pop(0)

        bboxes = [
            bbox
            for bbox in bboxes
            if bbox[0] != curr[0]  # class
            or intersection_over_union(bbox[2:].unsqueeze(0), curr[2:].unsqueeze(0), box_format=box_format)[0].item() <= iou_threshold
        ]

        rtn_bboxes.append(curr.tolist())

    return torch.Tensor(
        np.asarray(rtn_bboxes)
    )


def _find_best_iou_ind(img_ground_truths, class_detection, box_format):
    best_iou = 0
    best_gt_ind = -1
    for ind, gt in enumerate(img_ground_truths):
        iou = intersection_over_union(
            class_detection[3:],
            gt[3:],
            box_format=box_format
        ).item()

        if iou > best_iou:
            best_iou = iou
            best_gt_ind = ind

    return round(best_iou, 4), best_gt_ind


def _call_avg_precision(TP, FP, total_ground_truth_bboxes):
    epsilon = 1e-6
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = FP.cumsum(dim=0)
    # print('cumsum:', TP_cumsum, FP_cumsum)

    # find precision - out of all prediction how many were right
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = torch.cat((torch.tensor([1]), precisions))
    # print('precisions:', precisions)

    # find recall- out of all truths how many we found
    recalls = TP_cumsum / (total_ground_truth_bboxes + epsilon)
    recalls = torch.cat((torch.tensor([0]), recalls))
    # print('recalls:', recalls)

    return torch.trapz(precisions, recalls)


def _average_precision(class_detections, class_ground_truths, iou_threshold: float = 0.5, box_format: str = 'corners'):
    # amt_bboxes = {0:2, 1:3}
    amt_bboxes = Counter([int(g[0].item()) for g in class_ground_truths])
    for key, val in amt_bboxes.items():
        amt_bboxes[key] = torch.zeros(val)

    # sort by prob
    class_detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros(len(class_detections))
    FP = TP.clone()

    for class_detection_ind, class_detection in enumerate(class_detections):
        img_ground_truths = [
            bbox for bbox in class_ground_truths if bbox[0] == class_detection[0]
        ]

        # print(img_ground_truths, class_detection)

        best_iou, best_gt_ind = _find_best_iou_ind(
            img_ground_truths, class_detection, box_format)

        # print('best:', best_iou, best_gt_ind)

        class_detection_img_ind = class_detection[0].item()
        if best_iou >= iou_threshold:
            if amt_bboxes[class_detection_img_ind][best_gt_ind] == 0:
                TP[best_gt_ind] = 1
                amt_bboxes[class_detection_img_ind][best_gt_ind] = 1
            else:
                FP[class_detection_ind] = 1
        else:
            FP[class_detection_ind] = 1

    avg_precision = _call_avg_precision(TP, FP, len(class_ground_truths))

    return avg_precision


def mean_average_precision(bb_preds, bb_truths,
                           num_classes: int, iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9], box_format: str = "corners"):
    """
        bb_preds: [[img_ind, class, prob, x1, y1, x2, y2], ...]
        bb_truths: [[img_ind, class, prob, x1, y1, x2, y2], ...]
    """

    bb_preds_is_empty = len(bb_preds) == 0
    bb_truths_is_empty = len(bb_truths) == 0
    if bb_preds_is_empty and bb_truths_is_empty:
        return 1

    if bb_preds_is_empty and not bb_truths_is_empty:
        return 0

    assert bb_preds.ndim > 1
    assert bb_truths.ndim > 1
    assert bb_preds.ndim == bb_truths.ndim

    avg_precisions = []

    for iou_threshold in iou_thresholds:
        iou_avg = []
        for c in range(num_classes):
            avg_precision = _average_precision(
                [detection for detection in bb_preds if detection[1] == c],
                [gt_truth for gt_truth in bb_truths if gt_truth[1] == c],
                iou_threshold=iou_threshold,
                box_format=box_format
            )

            iou_avg.append(avg_precision)

        avg_precisions.append(
            round((sum(iou_avg) / len(iou_avg)).item(), 4)
        )

    return sum(avg_precisions) / len(avg_precisions)


if __name__ == '__main__':
    preds = torch.tensor([
        [1, 0, 0.8, 1, 1, 2, 2],
        [1, 0, 0.8, 2, 2, 3, 3],
    ])

    ground_truths = torch.tensor([
        [1, 0, 1, 1, 1, 2, 2],
        [1, 0, 1, 3, 3, 4, 4],
    ])

    print(mean_average_precision(preds, ground_truths, 1))
