import unittest

import torch

from changedetection.utils_func.lovasz_loss import flatten_probas, lovasz_softmax


class LovaszLossEdgeCaseTest(unittest.TestCase):
    def test_single_valid_pixel_preserves_class_dimension(self):
        logits = torch.randn(1, 5, 4, 4, requires_grad=True)
        probas = torch.softmax(logits, dim=1)
        labels = torch.full((1, 4, 4), 255, dtype=torch.long)
        labels[0, 2, 3] = 2

        flat_probas, flat_labels = flatten_probas(probas, labels, ignore=255)
        self.assertEqual(tuple(flat_probas.shape), (1, 5))
        self.assertEqual(tuple(flat_labels.shape), (1,))

        loss = lovasz_softmax(probas, labels, ignore=255)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_zero_valid_pixels_returns_scalar_zero(self):
        logits = torch.randn(1, 5, 4, 4, requires_grad=True)
        probas = torch.softmax(logits, dim=1)
        labels = torch.full((1, 4, 4), 255, dtype=torch.long)

        flat_probas, flat_labels = flatten_probas(probas, labels, ignore=255)
        self.assertEqual(tuple(flat_probas.shape), (0, 5))
        self.assertEqual(tuple(flat_labels.shape), (0,))

        loss = lovasz_softmax(probas, labels, ignore=255)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
