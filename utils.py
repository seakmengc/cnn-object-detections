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


if __name__ == '__main__':
    print(non_max_suppresion(
        torch.Tensor([
            [1, 0.9, 1, 1, 2, 2],
            [1, 0.8, 1, 1, 2, 2],
            [1, 0.8, 1.90, 1.90, 3, 3],
            [2, 0.9, 1, 1, 2, 2],
            [2, 0.5, 1, 1, 2, 2],
        ])
    ))
