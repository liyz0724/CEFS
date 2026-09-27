import unittest
from types import SimpleNamespace

from changedetection.engine.base import BaseTrainer


class ResumeResolutionTest(unittest.TestCase):
    def _trainer(self, *, model_checkpoint=None, resume_checkpoint=None):
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.args = SimpleNamespace(
            model_checkpoint_path=model_checkpoint,
            resume_training_path=resume_checkpoint,
        )
        trainer.config = SimpleNamespace(
            MODEL=SimpleNamespace(
                CHECKPOINT=model_checkpoint or "",
                RESUME=resume_checkpoint or "",
            )
        )
        return trainer

    def test_resume_is_not_also_treated_as_weight_only_checkpoint(self):
        trainer = self._trainer(resume_checkpoint="/tmp/latest.pth")
        self.assertIsNone(trainer._resolve_model_checkpoint_path())
        self.assertEqual(
            trainer._resolve_resume_training_path(),
            "/tmp/latest.pth",
        )

    def test_weight_only_checkpoint_remains_supported(self):
        trainer = self._trainer(model_checkpoint="/tmp/model.pth")
        self.assertEqual(
            trainer._resolve_model_checkpoint_path(),
            "/tmp/model.pth",
        )
        self.assertIsNone(trainer._resolve_resume_training_path())


if __name__ == "__main__":
    unittest.main()
