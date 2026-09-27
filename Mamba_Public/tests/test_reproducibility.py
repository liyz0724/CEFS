import random
import unittest

import numpy as np
import torch

from changedetection.script.script_utils import seed_everything
from changedetection.datasets.builder import seed_data_loader_worker


class ReproducibilityTest(unittest.TestCase):
    def test_seed_everything_repeats_random_streams(self):
        seed_everything(17)
        first = (random.random(), float(np.random.rand()), float(torch.rand(())))
        seed_everything(17)
        second = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(first, second)

    def test_negative_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            seed_everything(-1)

    def test_worker_seed_function_is_callable(self):
        seed_everything(23)
        seed_data_loader_worker(0)
        first = (random.random(), float(np.random.rand()))
        seed_everything(23)
        seed_data_loader_worker(0)
        second = (random.random(), float(np.random.rand()))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
