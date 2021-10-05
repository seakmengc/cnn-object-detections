import torch
import unittest

from utils import intersection_over_union, non_max_suppresion


class TestIOUFunction(unittest.TestCase):
    def test_corners_format(self):
        box_pred = torch.Tensor([[1, 1, 2, 2]])
        ground_box = torch.Tensor([[1.5, 1.5, 2.5, 2.5]])
        box_format = 'corners'

        res = intersection_over_union(box_pred, ground_box, box_format)

        self.assertAlmostEqual(res[0].item(), 0.3333, 4)

    def test_corners_format_with_full_overlap(self):
        box_pred = torch.Tensor([[1, 1, 2, 2]])
        ground_box = torch.Tensor([[1, 1, 2, 2]])
        box_format = 'corners'

        res = intersection_over_union(box_pred, ground_box, box_format)

        self.assertAlmostEqual(res[0].item(), 1, 4)

    def test_midpoint_format(self):
        box_pred = torch.Tensor([[1.5, 1.5, 1, 1]])
        ground_box = torch.Tensor([[2, 2, 1, 1]])
        box_format = 'midpoint'

        res = intersection_over_union(box_pred, ground_box, box_format)

        self.assertAlmostEqual(res[0].item(), 0.3333, 4)

    def test_midpoint_format_with_full_overlap(self):
        box_pred = torch.Tensor([[1.5, 1.5, 1, 1]])
        ground_box = torch.Tensor([[1.5, 1.5, 1, 1]])
        box_format = 'midpoint'

        res = intersection_over_union(box_pred, ground_box, box_format)

        self.assertAlmostEqual(res[0].item(), 1, 4)


class TestNMSFunction(unittest.TestCase):
    def test_it_should_remove_under_threshold(self):
        bboxes = torch.Tensor([
            [1, 0.3, 1, 1, 2, 2]
        ])

        res = non_max_suppresion(bboxes=bboxes, threshold=0.5)

        self.assertEqual(len(res), 0)

    def test_it_should_work_with_midpoint(self):
        bboxes = torch.Tensor([
            [1, 0.3, 1, 1, 2, 2]
        ])

        res = non_max_suppresion(
            bboxes=bboxes, threshold=0.5, box_format='midpoint')

        self.assertEqual(len(res), 0)

    def test_it_should_return_one_given_one_input(self):
        bboxes = torch.Tensor([
            [1, 0.6, 1, 1, 2, 2]
        ])

        res = non_max_suppresion(bboxes=bboxes)

        self.assertEqual(len(res), 1)
        self.assertListEqual(res.tolist(), bboxes.tolist())

    def test_it_should_suppress(self):
        bboxes = torch.Tensor([
            [1, 0.9, 1, 1, 2, 2],
            [1, 0.8, 1, 1, 2, 2],
            [1, 0.8, 1.90, 1.90, 3, 3],
            [2, 0.9, 1, 1, 2, 2],
            [2, 0.5, 1, 1, 2, 2],
            [2, 0.4, 1, 1, 2, 2],
            [3, 0.4, 1, 1, 2, 2],
        ])

        res = non_max_suppresion(bboxes, threshold=0.5)

        expected = torch.Tensor([
            [1, 0.9, 1, 1, 2, 2],
            [1, 0.8, 1.90, 1.90, 3, 3],
            [2, 0.9, 1, 1, 2, 2],
        ])

        self.assertEqual(len(res), len(expected))
        self.assertListEqual(
            sorted(res.tolist(), key=lambda x: x[0]),
            sorted(expected.tolist(), key=lambda x: x[0]),
        )


if __name__ == '__main__':
    unittest.main()
