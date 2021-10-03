import torch
import unittest

from utils import intersection_over_union


class TestIOUFunction(unittest.TestCase):
    def test_corners_format(self):
        box_pred = torch.Tensor([[1, 1, 2, 2]])
        ground_box = torch.Tensor([[1.5, 1.5, 2.5, 2.5]])
        format = 'corners'

        res = intersection_over_union(box_pred, ground_box, format)

        self.assertAlmostEqual(res[0].item(), 0.3333, 4)

    def test_corners_format_with_full_overlap(self):
        box_pred = torch.Tensor([[1, 1, 2, 2]])
        ground_box = torch.Tensor([[1, 1, 2, 2]])
        format = 'corners'

        res = intersection_over_union(box_pred, ground_box, format)

        self.assertAlmostEqual(res[0].item(), 1, 4)

    def test_midpoint_format(self):
        box_pred = torch.Tensor([[1.5, 1.5, 1, 1]])
        ground_box = torch.Tensor([[2, 2, 1, 1]])
        format = 'midpoint'

        res = intersection_over_union(box_pred, ground_box, format)

        self.assertAlmostEqual(res[0].item(), 0.3333, 4)

    def test_midpoint_format_with_full_overlap(self):
        box_pred = torch.Tensor([[1.5, 1.5, 1, 1]])
        ground_box = torch.Tensor([[1.5, 1.5, 1, 1]])
        format = 'midpoint'

        res = intersection_over_union(box_pred, ground_box, format)

        self.assertAlmostEqual(res[0].item(), 1, 4)


if __name__ == '__main__':
    unittest.main()
