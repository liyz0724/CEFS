import unittest

import numpy as np

from changedetection.evaluation import BinaryChangeEvaluator, DamageClassificationEvaluator
from changedetection.evaluation.event_bda import EventBDAMetricsAccumulator


class EventInferenceMetricsTest(unittest.TestCase):
    def test_metrics_are_grouped_by_xbd_event(self):
        labels_loc = np.array(
            [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ],
            dtype=np.int64,
        )
        pred_loc = np.array(
            [
                [[0, 1], [0, 1]],
                [[0, 0], [0, 0]],
            ],
            dtype=np.int64,
        )
        labels_clf = np.array(
            [
                [[0, 1], [0, 2]],
                [[0, 3], [0, 4]],
            ],
            dtype=np.int64,
        )
        pred_clf = labels_clf.copy()
        names = [
            "guatemala-volcano_00000001",
            "nepal-flooding_00000002",
        ]

        accumulator = EventBDAMetricsAccumulator()
        accumulator.add_batch(labels_loc, pred_loc, labels_clf, pred_clf, names)
        results = {result["event"]: result for result in accumulator.results()}

        self.assertEqual(set(results), {"guatemala-volcano", "nepal-flooding"})
        self.assertEqual(results["guatemala-volcano"]["samples"], 1)
        self.assertEqual(results["nepal-flooding"]["samples"], 1)
        self.assertAlmostEqual(results["guatemala-volcano"]["loc_F1"], 1.0, places=6)
        self.assertAlmostEqual(results["nepal-flooding"]["loc_F1"], 0.0)

        global_loc = BinaryChangeEvaluator()
        global_clf = DamageClassificationEvaluator(num_classes=5)
        global_loc.add_batch(labels_loc, pred_loc)
        global_clf.add_batch(labels_clf[labels_loc > 0], pred_clf[labels_loc > 0])

        grouped_loc = sum(
            evaluator_loc.confusion_matrix
            for evaluator_loc, _ in accumulator._evaluators.values()
        )
        grouped_clf = sum(
            evaluator_clf.confusion_matrix
            for _, evaluator_clf in accumulator._evaluators.values()
        )
        np.testing.assert_array_equal(grouped_loc, global_loc.confusion_matrix)
        np.testing.assert_array_equal(grouped_clf, global_clf.confusion_matrix)

    def test_rejects_name_count_mismatch(self):
        labels = np.zeros((2, 2, 2), dtype=np.int64)
        accumulator = EventBDAMetricsAccumulator()

        with self.assertRaisesRegex(ValueError, "one item name per batch sample"):
            accumulator.add_batch(labels, labels, labels, labels, ["nepal-flooding_00000001"])


if __name__ == "__main__":
    unittest.main()
