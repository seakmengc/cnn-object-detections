import torch
import unittest

from src.utils import intersection_over_union, mean_average_precision, non_max_suppresion


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


class TestMAPFunction(unittest.TestCase):
    def test_with_all_correct_predictions_one_class(self):
        preds = torch.tensor([
            [1, 0, 0.8, 1, 1, 2, 2],
            [1, 0, 0.8, 3, 3, 4, 4],
        ])

        ground_truths = torch.tensor([
            [1, 0, 1, 1, 1, 2, 2],
            [1, 0, 1, 3, 3, 4, 4],
        ])

        res = mean_average_precision(
            preds, ground_truths, 1, iou_thresholds=[0.5])

        self.assertEqual(res, 1)

    def test_with_all_correct_predictions_but_missing_target_one_class(self):
        preds = torch.tensor([
            [1, 0, 0.8, 1, 1, 2, 2],
        ])

        ground_truths = torch.tensor([
            [1, 0, 1, 1, 1, 2, 2],
            [1, 0, 1, 3, 3, 4, 4],
        ])

        res = mean_average_precision(
            preds, ground_truths, 1, iou_thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        self.assertEqual(res, 0.5)

    def test_with_no_correct_predictions_one_class(self):
        preds = torch.tensor([])

        ground_truths = torch.tensor([
            [1, 0, 1, 1, 1, 2, 2],
            [1, 0, 1, 3, 3, 4, 4],
        ])

        res = mean_average_precision(
            preds, ground_truths, 1, iou_thresholds=[0.5])

        self.assertEqual(res, 0)

    def test_with_no_correct_predictions_and_truths_one_class(self):
        preds = torch.tensor([])

        ground_truths = torch.tensor([])

        res = mean_average_precision(
            preds, ground_truths, 1, iou_thresholds=[0.5])

        self.assertEqual(res, 1)

    def test_with_all_correct_predictions_multi_classes(self):
        preds = torch.tensor([
            [1, 0, 0.8, 1, 1, 2, 2],
            [1, 1, 0.8, 3, 3, 4, 4],
        ])

        ground_truths = torch.tensor([
            [1, 0, 1, 1, 1, 2, 2],
            [1, 1, 1, 3, 3, 4, 4],
        ])

        res = mean_average_precision(
            preds, ground_truths, 2, iou_thresholds=[0.5])

        self.assertEqual(res, 1)

    def test_with_all_correct_predictions_but_missing_target_multi_classes(self):
        preds = torch.tensor([
            [1, 0, 0.8, 1, 1, 2, 2],
        ])

        ground_truths = torch.tensor([
            [1, 0, 1, 1, 1, 2, 2],
            [1, 1, 1, 3, 3, 4, 4],
        ])

        res = mean_average_precision(
            preds, ground_truths, 2, iou_thresholds=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        self.assertEqual(res, 0.5)

    def test_with_no_correct_predictions_multi_classes(self):
        preds = torch.tensor([])

        ground_truths = torch.tensor([
            [1, 0, 1, 1, 1, 2, 2],
            [1, 1, 1, 3, 3, 4, 4],
        ])

        res = mean_average_precision(
            preds, ground_truths, 2, iou_thresholds=[0.5])

        self.assertEqual(res, 0)


if __name__ == '__main__':
    unittest.main()
