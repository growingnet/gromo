import torch
import torch.utils.data

from gromo.utils.trainning_utils import AverageMeter, enumerate_dataloader
from tests.torch_unittest import TorchTestCase


class TestAverageMeter(TorchTestCase):
    """Tests for AverageMeter."""

    def test_empty_meter_returns_zero(self):
        """Empty meter returns 0.0."""
        meter = AverageMeter()
        self.assertEqual(meter.compute(), 0.0)

    def test_float_updates(self):
        """Average of float updates is correct."""
        meter = AverageMeter()
        meter.update(4.0, n=2)
        meter.update(6.0, n=3)
        # sum = 4*2 + 6*3 = 26, count = 5
        self.assertAlmostEqual(meter.compute(), 26.0 / 5)

    def test_inf_is_skipped(self):
        """Inf values are ignored."""
        meter = AverageMeter()
        meter.update(3.0)
        meter.update(float("inf"))
        self.assertEqual(meter.compute(), 3.0)

    def test_tensor_sum(self):
        """Tensor sum is handled via .item()."""
        meter = AverageMeter()
        meter.update(torch.tensor(10.0))  # pyright: ignore[reportArgumentType]
        self.assertAlmostEqual(meter.compute(), 10.0)

    def test_unsupported_sum_type_raises(self):
        """TypeError when sum has unsupported type."""
        meter = AverageMeter()
        meter.sum = "bad"  # type: ignore[assignment]
        meter.count = 1
        with self.assertRaises(TypeError):
            meter.compute()

    def test_reset(self):
        """Reset brings meter back to initial state."""
        meter = AverageMeter()
        meter.update(10.0)
        meter.reset()
        self.assertEqual(meter.compute(), 0.0)


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
