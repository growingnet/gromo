"""Tests of the K-FAC variants and of the Kronecker factor statistic."""

import dataclasses
from typing import get_args

import torch

from gromo.utils.kfac import KFACVariant, Placement, Setting
from gromo.utils.tensor_statistic import KroneckerFactorStatistic, TensorStatistic
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


REDUCE = KFACVariant("reduce", "count_weighted")


class TestKFACVariant(TorchTestCase):
    def test_count_weighted_reduce(self) -> None:
        """Count-weighted K-FAC-reduce: rho_n = R_n^2 (the number of pairs of
        positions), omega_n = rho_n for both factors, D_A = sum_n omega_n."""
        lengths = torch.tensor([2.0, 5.0, 1.0, 0.0])
        rho = REDUCE.rho(lengths)
        self.assertAllClose(rho, torch.tensor([4.0, 25.0, 1.0, 0.0]))
        self.assertAllClose(REDUCE.input_weights(rho), rho)
        self.assertAllClose(REDUCE.gradient_weights(rho), rho)
        sum_weights = torch.tensor(30.0)
        self.assertAllClose(REDUCE.input_denominator(sum_weights, 4), sum_weights)

    def test_other_combinations_not_implemented(self) -> None:
        """Every known (setting, placement) pair other than ("reduce",
        "count_weighted") raises NotImplementedError."""
        for setting in get_args(Setting):
            for placement in get_args(Placement):
                if (setting, placement) == ("reduce", "count_weighted"):
                    continue
                with self.subTest(setting=setting, placement=placement):
                    with self.assertRaises(NotImplementedError):
                        KFACVariant(setting, placement)

    def test_unknown_setting_or_placement(self) -> None:
        """An unknown setting or placement raises ValueError naming it."""
        for setting, placement, unknown in (
            ("unknown", "count_weighted", "unknown"),
            ("reduce", "unknown", "unknown"),
            ("expand", "other", "other"),
        ):
            with self.subTest(setting=setting, placement=placement):
                with self.assertRaises(ValueError) as context:
                    KFACVariant(setting, placement)  # type: ignore[arg-type]
                self.assertIn(unknown, str(context.exception))

    def test_frozen_and_comparable(self) -> None:
        """Variants are immutable values: equal fields give equal variants."""
        self.assertEqual(REDUCE, KFACVariant("reduce", "count_weighted"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            REDUCE.setting = "expand"  # type: ignore[misc]


class TestKroneckerFactorStatistic(TorchTestCase):
    """Factors on fixed tensors returned by the positions function.

    The references are computed in float64 from the per-sequence sums over the
    valid positions u_n = sum_m x_{n,m} and the lengths R_n:
    A = sum_n u_n u_n^T / sum_n R_n^2 and B = sum_n u_n u_n^T / N.
    """

    LENGTHS = (2, 5, 1, 4)

    def setUp(self) -> None:
        super().setUp()
        self.device = global_device()
        self.current: torch.Tensor | None = None

    def _positions(self) -> torch.Tensor:
        assert self.current is not None
        return self.current

    def _statistic(
        self, side: str, assume_centered: bool = False
    ) -> KroneckerFactorStatistic:
        return KroneckerFactorStatistic(
            REDUCE,
            side,
            self._positions,
            assume_centered=assume_centered,
            device=self.device,
            name=side,
        )

    def _feed(
        self,
        statistic: KroneckerFactorStatistic,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        count_samples: bool = True,
    ) -> None:
        """Make x the current batch and add it to the statistic."""
        self.current = x
        statistic.updated = False
        statistic.update(count_samples=count_samples, position_mask=mask)

    def _padded(
        self, lengths: tuple[int, ...], max_len: int, d: int = 3, offset: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random float64 batch (N, max_len, d) with random padding, and its mask."""
        x = offset + torch.randn(
            len(lengths), max_len, d, dtype=torch.float64, device=self.device
        )
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=self.device)
        for n, r in enumerate(lengths):
            mask[n, :r] = True
        return x, mask

    @staticmethod
    def _sums(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sums u_n over the valid positions and lengths R_n, by hand."""
        u = torch.stack(
            [x[n, mask[n]].sum(dim=0) for n in range(x.shape[0])]
        )  # empty sum: 0
        return u, mask.sum(dim=1).to(x.dtype)

    def _reference(
        self, batches: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A = sum_n u_n u_n^T / sum_n R_n^2 and B = sum_n u_n u_n^T / N."""
        u, lengths = zip(*(self._sums(x, mask) for x, mask in batches), strict=True)
        u, lengths = torch.cat(u), torch.cat(lengths)
        return u.T @ u / lengths.square().sum(), u.T @ u / u.shape[0]

    def test_variable_lengths_with_mask(self) -> None:
        """Variable lengths, padding with random values and a mask: A and B equal the
        formulas computed from u_n and g_n. A boolean and a 0/1 integer mask give
        the same factors."""
        x, mask = self._padded(self.LENGTHS, 5)
        a_ref, b_ref = self._reference([(x, mask)])
        for mask_used in (mask, mask.to(torch.int64)):
            with self.subTest(mask_dtype=mask_used.dtype):
                input_stat = self._statistic("input")
                gradient_stat = self._statistic("gradient")
                self._feed(input_stat, x, mask_used)
                self._feed(gradient_stat, x, mask_used)
                self.assertAllClose(input_stat(), a_ref, atol=1e-12, rtol=1e-12)
                self.assertAllClose(gradient_stat(), b_ref, atol=1e-12, rtol=1e-12)

    def test_equal_lengths(self) -> None:
        """Equal lengths R, no mask: A = (1/N) sum_n ā_n ā_n^T with the means
        ā_n = u_n / R, since sum_n R^2 = N R^2; B = (1/N) sum_n g_n g_n^T."""
        x = torch.randn(4, 6, 3, dtype=torch.float64, device=self.device)
        means, sums = x.mean(dim=1), x.sum(dim=1)
        input_stat = self._statistic("input")
        gradient_stat = self._statistic("gradient")
        self._feed(input_stat, x)
        self._feed(gradient_stat, x)
        self.assertAllClose(input_stat(), means.T @ means / 4, atol=1e-12, rtol=1e-12)
        self.assertAllClose(gradient_stat(), sums.T @ sums / 4, atol=1e-12, rtol=1e-12)

    def test_2d_input_is_plain_statistic(self) -> None:
        """2-D input (one position per sample, R_n = 1): A and B both equal the plain
        S = E = (1/N) sum_n x_n x_n^T of a TensorStatistic, over several batches,
        whatever the mask."""
        batches = [
            torch.randn(n, 3, dtype=torch.float64, device=self.device) for n in (5, 3)
        ]
        plain = TensorStatistic(
            (3, 3),
            update_function=lambda: (
                self._positions().T @ self._positions(),
                self._positions().shape[0],
            ),
        )
        input_stat = self._statistic("input")
        gradient_stat = self._statistic("gradient")
        for x in batches:
            ignored_mask = torch.zeros(x.shape[0], 7, device=self.device)
            self._feed(input_stat, x, ignored_mask)
            self._feed(gradient_stat, x, ignored_mask)
            plain.updated = False
            plain.update()
        self.assertAllClose(input_stat(), plain(), atol=1e-12, rtol=1e-12)
        self.assertAllClose(gradient_stat(), plain(), atol=1e-12, rtol=1e-12)

    def _shift_batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Three padded batches with a large mean (100), where the shift matters."""
        return [
            self._padded((2, 5, 1), 5, offset=100.0),
            self._padded((3, 4), 5, offset=100.0),
            self._padded((5, 1, 1, 2), 5, offset=100.0),
        ]

    def test_shift_batches_equal_concatenation(self) -> None:
        """Shift. The numerator W_2 + w_1 s^T + s w_1^T + W_0 s s^T equals
        sum_n omega_n x̄_n x̄_n^T for any shift s, so several batches give the same
        factors as their concatenation, whose shift is different, on both sides and
        with or without the shift (assume_centered)."""
        batches = self._shift_batches()
        whole = (
            torch.cat([x for x, _ in batches]),
            torch.cat([mask for _, mask in batches]),
        )
        for side in ("input", "gradient"):
            for assume_centered in (False, True):
                with self.subTest(side=side, assume_centered=assume_centered):
                    by_batch = self._statistic(side, assume_centered)
                    for x, mask in batches:
                        self._feed(by_batch, x, mask)
                    concatenated = self._statistic(side, assume_centered)
                    self._feed(concatenated, *whole)
                    self.assertEqual(by_batch.samples, concatenated.samples)
                    self.assertAllClose(
                        by_batch(), concatenated(), atol=1e-12, rtol=1e-12
                    )

    def test_assume_centered(self) -> None:
        """assume_centered is independent of the side. When False, the shift is the
        weighted mean of the first batch,
        s = sum_n R_n^2 x̄_n / sum_n R_n^2 = sum_n R_n u_n / sum_n R_n^2, on both
        sides. When True, no shift is set (s = 0). In both cases the factor is the
        uncentred one, A or B of the formulas."""
        batches = self._shift_batches()
        u, lengths = self._sums(*batches[0])
        shift_ref = (lengths @ u) / lengths.square().sum()
        a_ref, b_ref = self._reference(batches)
        for side, reference in (("input", a_ref), ("gradient", b_ref)):
            for assume_centered in (False, True):
                with self.subTest(side=side, assume_centered=assume_centered):
                    statistic = self._statistic(side, assume_centered)
                    for x, mask in batches:
                        self._feed(statistic, x, mask)
                    if assume_centered:
                        self.assertIsNone(statistic._shift)
                    else:
                        self.assertAllClose(
                            statistic._shift, shift_ref, atol=1e-12, rtol=1e-12
                        )
                    self.assertAllClose(statistic(), reference, atol=1e-10, rtol=1e-12)

    def test_count_samples_false(self) -> None:
        """count_samples=False adds the batch to the sums but not to N:
        B = (sum_{n in batch 1} g_n g_n^T + sum_{n in batch 2} g_n g_n^T) / N_1."""
        first, mask_first = self._padded(self.LENGTHS, 5)
        second, mask_second = self._padded(self.LENGTHS, 5)
        statistic = self._statistic("gradient")
        self._feed(statistic, first, mask_first)
        self._feed(statistic, second, mask_second, count_samples=False)
        u_first, _ = self._sums(first, mask_first)
        u_second, _ = self._sums(second, mask_second)
        expected = (u_first.T @ u_first + u_second.T @ u_second) / len(self.LENGTHS)
        self.assertEqual(statistic.samples, len(self.LENGTHS))
        self.assertAllClose(statistic(), expected, atol=1e-12, rtol=1e-12)

    def test_updated_flag(self) -> None:
        """update does nothing while `updated` is True, and sets it otherwise."""
        x, mask = self._padded(self.LENGTHS, 5)
        statistic = self._statistic("input")
        self.assertTrue(statistic.updated)
        self.current = x
        statistic.update(position_mask=mask)
        self.assertEqual(statistic.samples, 0)
        self._feed(statistic, x, mask)
        self.assertTrue(statistic.updated)
        before = statistic()
        statistic.update(position_mask=mask)
        self.assertEqual(statistic.samples, len(self.LENGTHS))
        self.assertAllClose(statistic(), before)

    def test_accumulation_dtype(self) -> None:
        """init(torch.float64) with float32 updates: the sums are kept in float64
        (`dtype`) and the factor is returned in float32, close to the float64
        reference."""
        x, mask = self._padded(self.LENGTHS, 5)
        a_ref, b_ref = self._reference([(x, mask)])
        for side, reference in (("input", a_ref), ("gradient", b_ref)):
            with self.subTest(side=side):
                statistic = self._statistic(side)
                statistic.init(torch.float64)
                self._feed(statistic, x.float(), mask)
                self.assertEqual(statistic.dtype, torch.float64)
                factor = statistic()
                self.assertEqual(factor.dtype, torch.float32)
                self.assertAllClose(factor.double(), reference, atol=1e-5, rtol=1e-5)

    def test_reset(self) -> None:
        """reset clears the sums, the count and the shift, and keeps the accumulation
        dtype: after a reset, the factor is the one of the new batches only."""
        old, mask_old = self._padded(self.LENGTHS, 5, offset=10.0)
        new, mask_new = self._padded((3, 1), 5)
        statistic = self._statistic("input")
        statistic.init(torch.float64)
        self._feed(statistic, old.float(), mask_old)
        statistic.reset()
        self.assertEqual(statistic.samples, 0)
        self.assertIsNone(statistic.dtype)
        self.assertIsNone(statistic._shift)
        with self.assertRaises(ValueError):
            statistic()
        self._feed(statistic, new.float(), mask_new)
        self.assertEqual(statistic.dtype, torch.float64)
        a_ref, _ = self._reference([(new, mask_new)])
        self.assertAllClose(statistic().double(), a_ref, atol=1e-5, rtol=1e-5)

    def test_value_error_before_update(self) -> None:
        """Calling a factor before any counted update raises ValueError."""
        for side in ("input", "gradient"):
            with self.subTest(side=side):
                with self.assertRaises(ValueError):
                    self._statistic(side)()

    def test_empty_sequence(self) -> None:
        """A sequence with R_n = 0 adds nothing to the sums but counts in N: A is
        unchanged (its denominator sum_n R_n^2 is unchanged), B is multiplied by
        N / (N + 1), `samples` is N + 1, and there is no NaN."""
        x, mask = self._padded(self.LENGTHS, 5)
        x_empty = torch.cat([x, torch.randn_like(x[:1])])
        mask_empty = torch.cat([mask, torch.zeros_like(mask[:1])])
        n = len(self.LENGTHS)
        for side, scale in (("input", 1.0), ("gradient", n / (n + 1))):
            with self.subTest(side=side):
                without = self._statistic(side)
                self._feed(without, x, mask)
                with_empty = self._statistic(side)
                self._feed(with_empty, x_empty, mask_empty)
                self.assertEqual(with_empty.samples, n + 1)
                self.assertFalse(with_empty().isnan().any())
                self.assertAllClose(
                    with_empty(), scale * without(), atol=1e-12, rtol=1e-12
                )

    def test_first_batch_without_valid_position(self) -> None:
        """A first batch without valid position counts in N but does not set the
        shift: A is undefined (sum_n R_n^2 = 0) and raises ValueError, and after a
        valid batch A is finite and equals the formula over both batches."""
        empty = torch.full((2, 5, 3), float("nan"), dtype=torch.float64)
        empty = empty.to(self.device)
        empty_mask = torch.zeros(2, 5, dtype=torch.bool, device=self.device)
        x, mask = self._padded(self.LENGTHS, 5)
        statistic = self._statistic("input")
        self._feed(statistic, empty, empty_mask)
        self.assertIsNone(statistic._shift)
        self.assertEqual(statistic.samples, 2)
        with self.assertRaises(ValueError):
            statistic()
        self._feed(statistic, x, mask)
        self.assertIsNotNone(statistic._shift)
        self.assertEqual(statistic.samples, 2 + len(self.LENGTHS))
        a_ref, _ = self._reference([(x, mask)])
        self.assertAllClose(statistic(), a_ref, atol=1e-12, rtol=1e-12)

    def test_unexpected_keyword(self) -> None:
        """update accepts only position_mask, even when `updated` is True."""
        statistic = self._statistic("input")
        for updated in (True, False):
            with self.subTest(updated=updated):
                statistic.updated = updated
                with self.assertRaises(TypeError):
                    statistic.update(mask=None)

    def test_constructor(self) -> None:
        """The side must be "input" or "gradient"; the statistic is a
        TensorStatistic, with the default name and device of TensorStatistic."""
        with self.assertRaises(ValueError):
            KroneckerFactorStatistic(REDUCE, "output", self._positions)  # type: ignore[arg-type]
        statistic = KroneckerFactorStatistic(REDUCE, "gradient", self._positions)
        self.assertIsInstance(statistic, TensorStatistic)
        self.assertEqual(statistic.name, "KroneckerFactorStatistic")
        self.assertFalse(statistic.assume_centered)
        self.assertEqual(statistic.device, global_device())
        self.assertIsNone(statistic.dtype)
        self.assertIn("gradient", str(statistic))


if __name__ == "__main__":
    from unittest import main

    main()
