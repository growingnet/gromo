import torch
import torch.utils.data
from torch import nn
from torchmetrics import Metric

from gromo.containers.growing_container import GrowingContainer, GrowingModel
from gromo.utils.trainning_utils import (
    AverageMeter,
    enumerate_dataloader,
    evaluate_model,
    gradient_descent,
)
from tests.torch_unittest import TorchTestCase


# ---------------------------------------------------------------------------
# Minimal test doubles for evaluate_model
# ---------------------------------------------------------------------------
class _SimpleModel(nn.Module):
    """A trivial linear model for testing evaluate_model."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)


class _SimpleGrowingModel(GrowingModel):
    """Minimal GrowingModel whose extended_forward returns a Tensor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features=in_features, out_features=out_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)

    def extended_forward(self, x: torch.Tensor, mask: dict | None = None) -> torch.Tensor:
        """Extended forward returns a plain Tensor."""
        return self.forward(x)


class _SimpleGrowingContainer(GrowingContainer):
    """Minimal GrowingContainer whose extended_forward returns a tuple."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features=in_features, out_features=out_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)

    def extended_forward(
        self, x: torch.Tensor, mask: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extended forward returns (output, None)."""
        return self.forward(x), None


class _SumMetric(Metric):
    """Accumulates the sum of first predictions — just enough to test the metrics path."""

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, _target: torch.Tensor):
        """Accumulate prediction sums."""
        self.total += preds.sum()

    def compute(self) -> torch.Tensor:
        """Return accumulated total."""
        return self.total


class TestAverageMeter(TorchTestCase):
    """Tests for AverageMeter."""

    def test_empty_meter_returns_zero(self):
        """Empty meter returns 0.0."""
        meter = AverageMeter()
        self.assertEqual(meter.compute().item(), 0.0)

    def test_float_updates(self):
        """Average of float updates is correct."""
        meter = AverageMeter()
        meter.update(torch.tensor(4.0), n=2)
        meter.update(torch.tensor(6.0), n=3)
        # sum = 4*2 + 6*3 = 26, count = 5
        self.assertAlmostEqual(meter.compute().item(), 26.0 / 5, places=6)

    def test_inf_is_skipped(self):
        """Inf values are ignored."""
        meter = AverageMeter()
        meter.update(torch.tensor(3.0))
        meter.update(torch.tensor(float("inf")))
        self.assertEqual(meter.compute().item(), 3.0)

    def test_reset(self):
        """Reset brings meter back to initial state."""
        meter = AverageMeter()
        meter.update(torch.tensor(10.0))
        meter.reset()
        self.assertEqual(meter.compute().item(), 0.0)


class TestEnumerateDataloader(TorchTestCase):
    """Tests for enumerate_dataloader."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 10,
        batch_size: int = 2,
        with_generator: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Create a simple dataloader for testing."""
        x = torch.randn(n_samples, 3)
        y = torch.randint(0, 2, (n_samples,))
        dataset = torch.utils.data.TensorDataset(x, y)
        gen = torch.Generator() if with_generator else None
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, generator=gen)

    def test_default_yields_all_batches(self):
        """Without limits, all batches are yielded."""
        dl = self._make_dataloader(n_samples=6, batch_size=2)
        batches = list(enumerate_dataloader(dl))
        self.assertEqual(len(batches), 3)

    def test_batch_limit(self):
        """Batch limit truncates output."""
        dl = self._make_dataloader(n_samples=10, batch_size=2)
        batches = list(enumerate_dataloader(dl, batch_limit=2))
        self.assertEqual(len(batches), 2)

    def test_epochs_fraction(self):
        """Fractional epochs limits batches proportionally."""
        dl = self._make_dataloader(n_samples=10, batch_size=2)  # 5 batches
        batches = list(enumerate_dataloader(dl, epochs=0.5))
        self.assertEqual(len(batches), 2)  # int(5 * 0.5) = 2

    def test_epochs_and_batch_limit_raises(self):
        """Providing both epochs and batch_limit raises TypeError."""
        dl = self._make_dataloader()
        with self.assertRaises(TypeError):
            list(enumerate_dataloader(dl, epochs=1.0, batch_limit=5))

    def test_seed_with_generator(self):
        """Seed is applied when dataloader has a Generator."""
        dl = self._make_dataloader(with_generator=True)
        batches = list(enumerate_dataloader(dl, dataloader_seed=0))
        self.assertGreater(len(batches), 0)
        batches_again = list(enumerate_dataloader(dl, dataloader_seed=0))
        for (_, (x_1, y_1)), (_, (x_2, y_2)) in zip(batches, batches_again):
            self.assertTrue(torch.equal(x_1, x_2))
            self.assertTrue(torch.equal(y_1, y_2))

    def test_seed_without_generator_raises(self):
        """AttributeError when seed given but no Generator."""
        dl = self._make_dataloader(with_generator=False)
        with self.assertRaises(AttributeError):
            list(enumerate_dataloader(dl, dataloader_seed=42))


class TestEvaluateModel(TorchTestCase):
    """Tests for evaluate_model."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8,
        in_features: int = 4,
        out_features: int = 2,
        batch_size: int = 4,
    ) -> torch.utils.data.DataLoader:
        """Create a simple regression dataloader."""
        x = torch.randn(n_samples, in_features)
        y = torch.randn(n_samples, out_features)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_basic_evaluation(self):
        """Evaluate a plain nn.Module without metrics."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        loss, metric_val = evaluate_model(model, dl, nn.MSELoss(reduction="mean"))
        self.assertIsInstance(loss, float)
        self.assertEqual(metric_val, 0.0)  # DummyMetric

    def test_with_metrics(self):
        """Evaluate with a custom metric (exercises the metrics branch)."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        metric = _SumMetric()
        loss, metric_val = evaluate_model(
            model, dl, nn.MSELoss(reduction="mean"), metrics=metric
        )
        self.assertIsInstance(loss, float)
        self.assertIsInstance(metric_val, float)

    def test_extended_growing_model(self):
        """use_extended_model=True with a GrowingModel."""
        model = _SimpleGrowingModel(4, 2)
        dl = self._make_dataloader()
        loss, _ = evaluate_model(
            model,
            dl,
            nn.MSELoss(reduction="mean"),
            use_extended_model=True,
        )
        self.assertIsInstance(loss, float)

    def test_extended_growing_container(self):
        """use_extended_model=True with a GrowingContainer."""
        model = _SimpleGrowingContainer(4, 2)
        dl = self._make_dataloader()
        loss, _ = evaluate_model(
            model,
            dl,
            nn.MSELoss(reduction="mean"),
            use_extended_model=True,
        )
        self.assertIsInstance(loss, float)

    def test_extended_invalid_model_raises(self):
        """use_extended_model=True with a plain nn.Module raises TypeError."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        with self.assertRaises(TypeError):
            evaluate_model(
                model,
                dl,
                nn.MSELoss(reduction="mean"),
                use_extended_model=True,
            )


class _FakeScheduler:
    """Minimal scheduler double that records step/epoch_step calls."""

    def __init__(self):
        self.step_count = 0
        self.epoch_step_count = 0

    def step(self):
        """Record a step call."""
        self.step_count += 1

    def epoch_step(self):
        """Record an epoch_step call."""
        self.epoch_step_count += 1


class TestGradientDescent(TorchTestCase):
    """Tests for gradient_descent."""

    @staticmethod
    def _make_dataloader(
        n_samples: int = 8,
        in_features: int = 4,
        out_features: int = 2,
        batch_size: int = 4,
    ) -> torch.utils.data.DataLoader:
        """Create a simple regression dataloader."""
        x = torch.randn(n_samples, in_features)
        y = torch.randn(n_samples, out_features)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )

    def test_basic_training(self):
        """One round of gradient descent without scheduler or metrics."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss, metric_val = gradient_descent(
            model,
            dl,
            optimizer,
            scheduler=None,
            loss_function=nn.MSELoss(reduction="mean"),
        )
        self.assertIsInstance(loss, float)
        self.assertEqual(metric_val, 0.0)  # DummyMetric

    def test_with_metrics(self):
        """Gradient descent with a custom metric."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metric = _SumMetric()
        loss, metric_val = gradient_descent(
            model,
            dl,
            optimizer,
            scheduler=None,
            loss_function=nn.MSELoss(reduction="mean"),
            metrics=metric,
        )
        self.assertIsInstance(loss, float)
        self.assertIsInstance(metric_val, float)

    def test_with_scheduler(self):
        """Scheduler.step() called per batch, epoch_step() called once."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader(n_samples=8, batch_size=4)  # 2 batches
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = _FakeScheduler()
        gradient_descent(
            model,
            dl,
            optimizer,
            scheduler=scheduler,
            loss_function=nn.MSELoss(reduction="mean"),
        )
        self.assertEqual(scheduler.step_count, 2)
        self.assertEqual(scheduler.epoch_step_count, 1)

    def test_loss_decreases(self):
        """Loss after training is lower than before."""
        model = _SimpleModel(4, 2)
        dl = self._make_dataloader(n_samples=16, batch_size=4)
        loss_fn = nn.MSELoss(reduction="mean")
        loss_before, _ = evaluate_model(model, dl, loss_fn)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(5):
            gradient_descent(
                model,
                dl,
                optimizer,
                scheduler=None,
                loss_function=loss_fn,
            )
        loss_after, _ = evaluate_model(model, dl, loss_fn)
        self.assertLess(loss_after, loss_before)
