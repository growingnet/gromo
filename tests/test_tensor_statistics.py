from unittest import TestCase, main

import torch
from torch.utils.data import DataLoader, TensorDataset

from gromo.utils.tensor_statistic import (
    TensorStatistic,
    TensorStatiticWithEstimationError,
)
from gromo.utils.utils import reset_device, set_device


class TestTensorStatistic(TestCase):
    _tested_class = TensorStatistic

    def test_mean(self):
        set_device("cpu")
        x = None
        n_samples = 0
        f = lambda: (x.sum(dim=0), x.size(0))  # type: ignore  # noqa: E731
        tensor_statistic = self._tested_class(
            shape=(2, 3), update_function=f, name="Average"
        )
        tensor_statistic_unshaped = self._tested_class(
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


class TestTensorStatiticWithEstimationError(TestTensorStatistic):
    _tested_class = TensorStatiticWithEstimationError

    def setUp(self) -> None:
        set_device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

    def test_error(self):
        num_batches = 10
        batch_size = 10
        total_samples = torch.Size((num_batches * batch_size,))
        mean = torch.tensor([3.0, 4.0])
        cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample(total_samples)
        dataloader = DataLoader(
            TensorDataset(samples), batch_size=batch_size, shuffle=False
        )

        mean_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (x.sum(dim=0), x.size(0)),
            name="Mean with Error",
        )

        self.assertRaises(AssertionError, mean_statistic.error)
        for i, (batch,) in enumerate(dataloader):
            mean_statistic.updated = False
            mean_statistic.update(x=batch)
            if i == 0:
                self.assertEqual(mean_statistic.error(), float("inf"))
        self.assertEqual(mean_statistic.samples, num_batches * batch_size)
        true_error = torch.norm(mean_statistic() - mean).item() ** 2
        self.assertLessEqual(
            true_error, mean_statistic.error() * 3
        )  # this test pass most of the time, but can fail due to randomness
        # (if no seed is set)

        cov_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (
                (x - mean_statistic()).T @ (x - mean_statistic()),
                x.size(0),
            ),
            name="Covariance with Error",
        )

        for (batch,) in dataloader:
            cov_statistic.updated = False
            cov_statistic.update(x=batch)

        true_error = torch.norm(cov_statistic() - cov).item() ** 2
        self.assertLessEqual(
            true_error, cov_statistic.error() * 2
        )  # this test pass most of the time, but can fail due to randomness
        # (if no seed is set)

    def test_stop_trace_computation(self):
        num_batches = 3
        batch_size = 10
        total_samples = torch.Size((num_batches * batch_size,))
        mean = torch.tensor([3.0, 4.0])
        cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        samples = dist.sample(total_samples)
        dataloader = DataLoader(
            TensorDataset(samples), batch_size=batch_size, shuffle=False
        )

        mean_statistic = TensorStatiticWithEstimationError(
            shape=None,
            update_function=lambda x: (x.sum(dim=0), x.size(0)),
            name="Mean with Error",
            trace_precision=5.0,
        )

        self.assertRaises(AssertionError, mean_statistic.error)
        for i, (batch,) in enumerate(dataloader):
            mean_statistic.updated = False
            mean_statistic.update(x=batch)

        self.assertFalse(mean_statistic._compute_trace)


if __name__ == "__main__":
    main()
