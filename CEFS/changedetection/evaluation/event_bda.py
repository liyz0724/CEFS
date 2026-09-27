"""Per-disaster xBD metrics for grouped building-damage inference."""

from changedetection.utils_func.xbd_event import extract_xbd_event

from .binary import BinaryChangeEvaluator
from .damage import DamageClassificationEvaluator


class EventBDAMetricsAccumulator:
    """Accumulate localization and damage metrics independently by xBD event."""

    def __init__(self):
        self._evaluators = {}
        self._sample_counts = {}

    def add_batch(self, labels_loc, pred_loc, labels_clf, pred_clf, names):
        batch_size = int(labels_loc.shape[0])
        if len(names) != batch_size:
            raise ValueError(
                "Per-event inference metrics require one item name per batch sample: "
                f"got {len(names)} names for batch size {batch_size}."
            )

        for index, item_name in enumerate(names):
            event_name = extract_xbd_event(item_name)
            if event_name not in self._evaluators:
                self._evaluators[event_name] = (
                    BinaryChangeEvaluator(),
                    DamageClassificationEvaluator(num_classes=5),
                )
                self._sample_counts[event_name] = 0

            evaluator_loc, evaluator_clf = self._evaluators[event_name]
            sample_labels_loc = labels_loc[index]
            sample_pred_loc = pred_loc[index]
            sample_labels_clf = labels_clf[index]
            sample_pred_clf = pred_clf[index]

            evaluator_loc.add_batch(sample_labels_loc, sample_pred_loc)
            building_mask = sample_labels_loc > 0
            evaluator_clf.add_batch(
                sample_labels_clf[building_mask],
                sample_pred_clf[building_mask],
            )
            self._sample_counts[event_name] += 1

    def results(self):
        for event_name in sorted(self._evaluators):
            evaluator_loc, evaluator_clf = self._evaluators[event_name]
            loc_metrics = evaluator_loc.compute()
            damage_metrics = evaluator_clf.compute()
            yield {
                "event": event_name,
                "samples": self._sample_counts[event_name],
                "loc_F1": loc_metrics.f1,
                "clf_F1": damage_metrics.harmonic_mean_f1,
                "oa_F1": 0.3 * loc_metrics.f1 + 0.7 * damage_metrics.harmonic_mean_f1,
                "sub_F1": damage_metrics.per_class_f1,
            }
