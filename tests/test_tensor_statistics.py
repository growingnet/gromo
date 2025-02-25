from unittest import TestCase, main

import torch

from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.utils import reset_device, set_device


class TestTensorStatistic(TestCase):
    def test_mean(self):
        set_device("cpu")
        x = None
        n_samples = 0
        f = lambda: (x.sum(dim=0), x.size(0))
        tensor_statistic = TensorStatistic(
            shape=(2, 3), update_function=f, name="Average"
        )
        tensor_statistic_unshaped = TensorStatistic(
            shape=None, update_function=f, name="Average-unshaped"
        )

        for t in [tensor_statistic, tensor_statistic_unshaped]:
            self.assertRaises(ValueError, t)

        tensor_statistic.init()
        tensor_statistic_unshaped.init()
        mean_x = torch.zeros((2, 3))
        for n in [1, 5, 8, 15]:
            x = torch.randn(n, 2, 3)
            n_samples += x.size(0)
            mean_x += x.sum(dim=0)
            for t in [tensor_statistic, tensor_statistic_unshaped]:
                t.updated = False
                t.update()
                self.assertTrue(torch.allclose(t(), mean_x / n_samples))
                self.assertEqual(t.samples, n_samples)

                t.update()
                self.assertTrue(torch.allclose(t(), mean_x / n_samples))
                self.assertEqual(t.samples, n_samples)

        x = torch.zeros(1, 3, 4)
        for t in [tensor_statistic, tensor_statistic_unshaped]:
            t.updated = False
            self.assertRaises(AssertionError, t.update)

            t.reset()
            self.assertIsNone(t._tensor)
            self.assertEqual(t.samples, 0)

    def tearDown(self) -> None:
        reset_device()


if __name__ == "__main__":
    main()
