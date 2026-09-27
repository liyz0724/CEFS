import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from changedetection.checkpoints import (
    resume_training_state,
    save_training_checkpoint,
    validate_checkpoint_load,
)
from changedetection.engine.base import BaseTrainer


class _ResumeHarness(BaseTrainer):
    def emit_log(self, message):
        self.messages.append(message)

    def load_checkpoint_extra_state(self, extra_state):
        self.restored_extra_state = extra_state


class CheckpointResumeTest(unittest.TestCase):
    def test_checkpoint_validation_rejects_zero_match(self):
        with self.assertRaises(RuntimeError):
            validate_checkpoint_load(
                {
                    "path": "/tmp/wrong.pth",
                    "loaded_keys": 0,
                    "missing_keys": ["weight"],
                    "unexpected_keys": ["other.weight"],
                    "mismatched_keys": [],
                },
                context="test checkpoint",
            )

    def test_complete_checkpoint_validation_rejects_partial_match(self):
        with self.assertRaises(RuntimeError):
            validate_checkpoint_load(
                {
                    "path": "/tmp/partial.pth",
                    "loaded_keys": 1,
                    "missing_keys": ["bias"],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                },
                context="test checkpoint",
                require_complete=True,
            )

    def test_full_resume_state_includes_report_path_and_optimizer(self):
        source_model = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-4)

        source_optimizer.zero_grad()
        source_model(torch.ones(2, 3)).sum().backward()
        source_optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory) / "latest.pth")
            save_training_checkpoint(
                checkpoint_path,
                model=source_model,
                optimizer=source_optimizer,
                iteration=1500,
                best_score=0.75,
                best_record={"iteration": 1500},
                task_name="bda",
                extra_state={
                    "auxiliary_state": {
                        "update_counts": torch.tensor([[3, 2, 1, 0]])
                    }
                },
            )

            restored_model = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
            state = resume_training_state(
                checkpoint_path,
                model=restored_model,
                optimizer=restored_optimizer,
            )

        self.assertEqual(state["iteration"], 1500)
        self.assertEqual(state["load_info"]["path"], checkpoint_path)
        self.assertTrue(state["optimizer_loaded"])
        self.assertTrue(state["has_optimizer_state"])
        self.assertEqual(
            int(state["extra_state"]["auxiliary_state"]["update_counts"][0, 0]),
            3,
        )
        for source, restored in zip(source_model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(source, restored))

    def test_checkpoint_save_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "latest.pth"
            save_training_checkpoint(
                str(checkpoint_path),
                model=torch.nn.Linear(2, 1),
                iteration=10,
            )
            self.assertTrue(checkpoint_path.is_file())
            temporary_files = [
                path for path in Path(directory).iterdir() if path.suffix == ".tmp"
            ]
            self.assertEqual(temporary_files, [])

    def test_failed_checkpoint_save_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "latest.pth"
            original_model = torch.nn.Linear(2, 1)
            save_training_checkpoint(
                str(checkpoint_path),
                model=original_model,
                iteration=10,
            )

            with mock.patch(
                "changedetection.checkpoints.torch.save",
                side_effect=RuntimeError("simulated write failure"),
            ):
                with self.assertRaises(RuntimeError):
                    save_training_checkpoint(
                        str(checkpoint_path),
                        model=torch.nn.Linear(2, 1),
                        iteration=20,
                    )

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["iteration"], 10)
            temporary_files = [
                path for path in Path(directory).iterdir() if path.suffix == ".tmp"
            ]
            self.assertEqual(temporary_files, [])

    def test_base_trainer_resume_flow_formats_report_and_restores_extra_state(self):
        source_model = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory) / "latest.pth")
            save_training_checkpoint(
                checkpoint_path,
                model=source_model,
                optimizer=source_optimizer,
                iteration=1500,
                best_score=0.75,
                best_record={"iteration": 1500},
                task_name="bda",
                extra_state={"auxiliary_state": {"sentinel": torch.tensor(7)}},
            )

            harness = _ResumeHarness.__new__(_ResumeHarness)
            harness.resume_training_path = checkpoint_path
            harness.model = torch.nn.Linear(3, 2)
            harness.optimizer = torch.optim.AdamW(harness.model.parameters(), lr=1e-4)
            harness.scheduler = None
            harness.start_iteration = 0
            harness.best_score = None
            harness.best_record = None
            harness.messages = []
            harness.restored_extra_state = None
            harness._resume_from_checkpoint()

        self.assertEqual(harness.start_iteration, 1500)
        self.assertEqual(int(harness.restored_extra_state["auxiliary_state"]["sentinel"]), 7)
        self.assertTrue(any("RESUME Model" in message for message in harness.messages))
        self.assertTrue(any("RESUME State" in message for message in harness.messages))


if __name__ == "__main__":
    unittest.main()
