import torch


def _midpoint_to_corners(box):
    return torch.Tensor([
        box[..., 0:1] - box[..., 2:3] / 2,  # x1
        box[..., 1:2] - box[..., 3:4] / 2,  # y1
        box[..., 0:1] + box[..., 2:3] / 2,  # x2
        box[..., 1:2] + box[..., 3:4] / 2,  # y2
    ])


def intersection_over_union(box_pred, ground_box, format):
    """
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    """

    # convert midpoint box to corners
    if format == 'midpoint':
        box1 = _midpoint_to_corners(box_pred)
        box2 = _midpoint_to_corners(ground_box)

    if format == 'corners':
        box1 = box_pred
        box2 = ground_box

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
