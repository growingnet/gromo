"""Tests of gromo.utils.shrinkage."""

import math
import unittest

import torch

from gromo.utils.shrinkage import (
    LedoitWolfEstimate,
    ReduceMoments,
    fixed_shrinkage,
    reduce_fourth_moment,
    weighted_ledoit_wolf,
)
from tests.torch_unittest import TorchTestCase


DTYPE = torch.float64


def outer_products(y: torch.Tensor) -> torch.Tensor:
    """Matrices y_n y_n^T, of shape (N, d, d), of vectors y of shape (N, d)."""
    return y[:, :, None] * y[:, None, :]


def weighted_estimate(z: torch.Tensor, nu: torch.Tensor) -> LedoitWolfEstimate:
    """Weighted Ledoit-Wolf estimate computed from the matrices Z_n themselves.

    S = sum_n nu_n Z_n / sum_n nu_n and the fourth moment sum_n nu_n^2 ||Z_n||_F^2,
    for z of shape (N, d, d) and nu of shape (N,).
    """
    s = torch.einsum("n,nij->ij", nu, z) / nu.sum()
    return weighted_ledoit_wolf(
        s,
        sum_weights=float(nu.sum()),
        sum_squared_weights=float(nu.square().sum()),
        weighted_fourth_moment=float(nu.square() @ z.square().sum(dim=(1, 2))),
    )


def random_weights(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Weights nu_n uniform in [1, 10]."""
    return 1 + 9 * torch.rand(n, generator=generator, dtype=DTYPE)


class TestFixedShrinkage(TorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.matrix = torch.randn(5, 5, dtype=DTYPE)

    def test_zero_intensity(self) -> None:
        """Intensity 0 returns the matrix."""
        self.assertAllClose(fixed_shrinkage(self.matrix, 0.0), self.matrix)

    def test_full_intensity(self) -> None:
        """Intensity 1 returns tr(M) / d * I."""
        expected = self.matrix.trace() / 5 * torch.eye(5, dtype=DTYPE)
        self.assertAllClose(fixed_shrinkage(self.matrix, 1.0), expected)

    def test_out_of_range(self) -> None:
        """An intensity outside [0, 1] raises ValueError."""
        for intensity in (-0.1, 1.5, math.nan):
            with self.subTest(intensity=intensity):
                with self.assertRaises(ValueError):
                    fixed_shrinkage(self.matrix, intensity)


class TestWeightedLedoitWolf(TorchTestCase):
    def test_classical_ledoit_wolf(self) -> None:
        """With nu_n = 1 and Z_n = y_n y_n^T, the estimate is the textbook one.

        Ledoit and Wolf (2004): S = sum_n y_n y_n^T / N, lambda = tr(S) / d,
        delta^2 = ||S - lambda I||_F^2,
        beta^2 = min(sum_n ||y_n y_n^T - S||_F^2 / N^2, delta^2),
        shrunk matrix (beta^2 / delta^2) lambda I + (1 - beta^2 / delta^2) S.
        """
        n, d = 15, 6
        y = torch.randn(n, d, dtype=DTYPE) * torch.linspace(0.5, 2.0, d, dtype=DTYPE)
        z = outer_products(y)
        estimate = weighted_estimate(z, torch.ones(n, dtype=DTYPE))

        s = y.T @ y / n
        mean_eigenvalue = s.trace() / d
        identity = torch.eye(d, dtype=DTYPE)
        delta2 = float((s - mean_eigenvalue * identity).square().sum())
        beta2 = min(float((z - s).square().sum()) / n**2, delta2)
        intensity = beta2 / delta2
        expected = intensity * mean_eigenvalue * identity + (1 - intensity) * s

        # The data must give a non-trivial intensity, below the clipping.
        self.assertTrue(0 < intensity < 1)
        self.assertAlmostEqual(estimate.delta2, delta2, places=10)
        self.assertAlmostEqual(estimate.beta2, beta2, places=10)
        self.assertAlmostEqual(estimate.intensity, intensity, places=10)
        self.assertAllClose(estimate.matrix, expected)

    def test_scale_homogeneity(self) -> None:
        """Scaling all the Z_n by s > 0 scales the matrix by s, and leaves tau."""
        n, d, scale = 12, 4, 3.7
        z = outer_products(torch.randn(n, d, dtype=DTYPE))
        nu = random_weights(n)
        estimate = weighted_estimate(z, nu)
        scaled = weighted_estimate(scale * z, nu)

        self.assertAllClose(scaled.matrix, scale * estimate.matrix)
        self.assertAlmostEqual(scaled.intensity, estimate.intensity, places=10)

    def test_weight_homogeneity(self) -> None:
        """Scaling all the weights by s > 0 changes nothing.

        At the level of `weighted_ledoit_wolf`, the sum of weights is multiplied by
        s, and the sum of squared weights and the fourth moment by s^2.
        """
        n, d, scale = 12, 4, 0.3
        z = outer_products(torch.randn(n, d, dtype=DTYPE))
        nu = random_weights(n)
        estimate = weighted_estimate(z, nu)
        scaled = weighted_estimate(z, scale * nu)

        self.assertAllClose(scaled.matrix, estimate.matrix)
        for field in (
            "intensity",
            "mean_eigenvalue",
            "beta2",
            "delta2",
            "effective_samples",
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(scaled, field), getattr(estimate, field), places=10
                )

    def test_identical_matrices(self) -> None:
        """Identical Z_n give beta^2 = 0, tau = 0 and the matrix S (up to rounding)."""
        n, d = 6, 4
        b = torch.randn(d, d, dtype=DTYPE)
        z = (b @ b.T).expand(n, d, d)
        estimate = weighted_estimate(z, random_weights(n))

        self.assertAlmostEqual(estimate.beta2, 0.0, places=10)
        self.assertAlmostEqual(estimate.intensity, 0.0, places=10)
        self.assertAllClose(estimate.matrix, b @ b.T)

    def test_clipped_to_full_shrinkage(self) -> None:
        """beta^2 is clipped to delta^2, which gives tau = 1 and lambda I.

        d = 3, nu = (1, 1), Z_1 = e_1 e_1^T, Z_2 = e_2 e_2^T: S = diag(1/2, 1/2, 0),
        lambda = 1/3, delta^2 = 1/2 - 3/9 = 1/6, and the unclipped beta^2 is
        (2 - 2 * 1/2) / 4 = 1/4 > delta^2.
        """
        z = torch.zeros(2, 3, 3, dtype=DTYPE)
        z[0, 0, 0] = z[1, 1, 1] = 1.0
        estimate = weighted_estimate(z, torch.ones(2, dtype=DTYPE))

        self.assertAlmostEqual(estimate.delta2, 1 / 6)
        self.assertAlmostEqual(estimate.beta2, 1 / 6)
        self.assertEqual(estimate.intensity, 1.0)
        self.assertAllClose(estimate.matrix, torch.eye(3, dtype=DTYPE) / 3)

    def test_zero_matrices(self) -> None:
        """Z_n = 0 give delta^2 = 0, tau = 0 and the zero matrix, without NaN."""
        estimate = weighted_estimate(torch.zeros(4, 3, 3, dtype=DTYPE), random_weights(4))

        self.assertEqual(estimate.delta2, 0.0)
        self.assertEqual(estimate.intensity, 0.0)
        self.assertTrue(torch.equal(estimate.matrix, torch.zeros(3, 3, dtype=DTYPE)))

    def test_multiple_of_identity(self) -> None:
        """For S = lambda I, delta^2 is 0 up to rounding: the matrix is S, no NaN.

        Z_k = d lambda e_k e_k^T (k = 1..d, nu_k = 1) has mean lambda I and gives a
        positive unclipped beta^2, so tau is 1 when delta^2 rounds above 0 and 0
        when it is 0. Among these cases, the raw delta^2 rounds below 0 for
        (d, lambda) = (3, 0.1) and above 0 for (3, 0.7), (7, 2.9).
        """
        for d, lam in ((3, 0.1), (3, 0.7), (7, 0.7), (7, 2.9)):
            with self.subTest(d=d, lam=lam):
                s = lam * torch.eye(d, dtype=DTYPE)
                estimate = weighted_ledoit_wolf(
                    s,
                    sum_weights=d,
                    sum_squared_weights=d,
                    weighted_fourth_moment=d * (d * lam) ** 2,
                )
                self.assertGreaterEqual(estimate.delta2, 0.0)
                self.assertAlmostEqual(estimate.delta2, 0.0, places=10)
                self.assertTrue(0.0 <= estimate.intensity <= 1.0)
                self.assertTrue(torch.isfinite(estimate.matrix).all())
                self.assertAllClose(estimate.matrix, s, atol=1e-12, rtol=1e-12)

    def test_effective_samples(self) -> None:
        """n_eff = (sum_n nu_n)^2 / sum_n nu_n^2: 100 / 30 for nu = (1, 2, 3, 4)."""
        nu = torch.arange(1, 5, dtype=DTYPE)
        z = outer_products(torch.randn(4, 3, dtype=DTYPE))
        self.assertAlmostEqual(weighted_estimate(z, nu).effective_samples, 100 / 30)

    def test_spectrum_and_trace(self) -> None:
        """For a positive semi-definite S, lambda_min(LW) >= tau lambda, and
        tr(LW) = tr(S).

        S is singular (N < d), so the bound is attained.
        """
        n, d = 4, 8
        z = outer_products(torch.randn(n, d, dtype=DTYPE))
        nu = random_weights(n)
        estimate = weighted_estimate(z, nu)
        s = torch.einsum("n,nij->ij", nu, z) / nu.sum()
        bound = estimate.intensity * estimate.mean_eigenvalue

        self.assertGreater(estimate.intensity, 0.0)
        self.assertGreaterEqual(
            float(torch.linalg.eigvalsh(estimate.matrix).min()), bound - 1e-12
        )
        self.assertAlmostEqual(
            float(estimate.matrix.trace()), float(s.trace()), places=10
        )

    def test_dtype_and_device(self) -> None:
        """The matrix keeps the dtype and device of S, the scalars are floats."""
        s = torch.randn(3, 3).square()
        estimate = weighted_ledoit_wolf(s, 2.0, 2.0, float(s.square().sum()) * 4)
        self.assertEqual(estimate.matrix.dtype, s.dtype)
        self.assertEqual(estimate.matrix.device, s.device)
        for value in (
            estimate.intensity,
            estimate.mean_eigenvalue,
            estimate.beta2,
            estimate.delta2,
            estimate.effective_samples,
        ):
            self.assertIsInstance(value, float)

    def test_invalid_inputs(self) -> None:
        """A non-square or empty matrix, or a non-positive sum of weights, raises
        ValueError."""
        for shape in ((3, 4), (3,), (2, 3, 3), (0, 0)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    weighted_ledoit_wolf(torch.zeros(shape), 1.0, 1.0, 0.0)
        for sum_weights, sum_squared_weights in ((0.0, 1.0), (1.0, -1.0)):
            with self.subTest(sums=(sum_weights, sum_squared_weights)):
                with self.assertRaises(ValueError):
                    weighted_ledoit_wolf(
                        torch.eye(3), sum_weights, sum_squared_weights, 0.0
                    )


class TestCalibration(unittest.TestCase):
    def test_beta2_estimates_squared_error(self) -> None:
        """Averaged over 200 seeds, beta^2 is within 10% of ||S - Sigma_0||_F^2.

        y_n ~ N(0, Sigma_0), d = 10, N = 50,
        Sigma_0 = diag(linspace(0.2, 3, d)), nu_n uniform in [1, 10] and independent
        of y_n, Z_n = y_n y_n^T. beta^2 is biased low by about 1 - 1 / n_eff, a few
        percent here (n_eff is about 41).
        """
        n, d, seeds = 50, 10, 200
        sigma_0 = torch.diag(torch.linspace(0.2, 3.0, d, dtype=DTYPE))
        total_beta2 = 0.0
        total_squared_error = 0.0
        for seed in range(seeds):
            generator = torch.Generator().manual_seed(seed)
            y = torch.randn(n, d, generator=generator, dtype=DTYPE) @ sigma_0.sqrt()
            nu = random_weights(n, generator)
            z = outer_products(y)
            s = torch.einsum("n,nij->ij", nu, z) / nu.sum()
            total_beta2 += weighted_estimate(z, nu).beta2
            total_squared_error += float((s - sigma_0).square().sum())

        self.assertLess(abs(total_beta2 / total_squared_error - 1), 0.1)


class TestReduceFourthMoment(TorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.n, self.d = 30, 5
        self.x = 3 * torch.randn(self.n, self.d, dtype=DTYPE) + 1
        self.nu = random_weights(self.n)

    def accumulate(
        self,
        x: torch.Tensor,
        nu: torch.Tensor,
        dtype: torch.dtype = DTYPE,
        sizes: tuple[int, ...] = (7, 10, 13),
    ) -> ReduceMoments:
        """Accumulate the vectors in several `add` calls of the given sizes."""
        moments = ReduceMoments.zeros(self.d, dtype=dtype)
        for x_batch, nu_batch in zip(
            x.split(list(sizes)), nu.split(list(sizes)), strict=True
        ):
            moments.add(x_batch, nu_batch)
        return moments

    @staticmethod
    def direct(x: torch.Tensor, nu: torch.Tensor, centre: torch.Tensor) -> float:
        """sum_n nu_n^2 ||x_n - c||^4, computed directly in float64."""
        x, nu, centre = x.to(DTYPE), nu.to(DTYPE), centre.to(DTYPE)
        return float(nu.square() @ (x - centre).square().sum(dim=1).square())

    def centres(self) -> list[torch.Tensor]:
        """The origin, the mean, the weighted mean and a random point."""
        return [
            torch.zeros(self.d, dtype=DTYPE),
            self.x.mean(dim=0),
            self.nu @ self.x / self.nu.sum(),
            torch.randn(self.d, dtype=DTYPE),
        ]

    def test_zeros(self) -> None:
        """The empty accumulators have the shapes (), (d,) or (d, d) and the dtype."""
        moments = ReduceMoments.zeros(self.d, dtype=torch.float32, device="cpu")
        self.assertShapeEqual(moments.sum_weights, ())
        self.assertShapeEqual(moments.sum_vectors, (self.d,))
        self.assertShapeEqual(moments.sum_outer_products, (self.d, self.d))
        self.assertEqual(moments.sum_fourth_powers.dtype, torch.float32)
        self.assertEqual(reduce_fourth_moment(moments, torch.zeros(self.d)), 0.0)

    def test_matches_direct_computation(self) -> None:
        """The one-pass formula equals sum_n nu_n^2 ||x_n - c||^4 for several c, and
        the sums of weights are sum_n nu_n and sum_n nu_n^2."""
        moments = self.accumulate(self.x, self.nu)
        self.assertAlmostEqual(float(moments.sum_weights), float(self.nu.sum()))
        self.assertAlmostEqual(
            float(moments.sum_squared_weights), float(self.nu.square().sum())
        )
        for centre in self.centres():
            with self.subTest(centre=centre):
                expected = self.direct(self.x, self.nu, centre)
                self.assertAlmostEqual(
                    reduce_fourth_moment(moments, centre),
                    expected,
                    delta=1e-10 * expected,
                )

    def test_origin_invariance(self) -> None:
        """Vectors x_n - o and centre c - o give the same result for any origin o."""
        origin = 10 * torch.randn(self.d, dtype=DTYPE)
        moments = self.accumulate(self.x, self.nu)
        shifted_moments = self.accumulate(self.x - origin, self.nu)
        for centre in self.centres():
            with self.subTest(centre=centre):
                expected = reduce_fourth_moment(moments, centre)
                self.assertAlmostEqual(
                    reduce_fourth_moment(shifted_moments, centre - origin),
                    expected,
                    delta=1e-10 * expected,
                )

    def test_mixed_dtypes(self) -> None:
        """float32 vectors with integer weights are summed in float64 accumulators,
        and the result matches the direct computation up to float32 rounding."""
        x = self.x.to(torch.float32)
        nu = torch.randint(1, 10, (self.n,))
        moments = self.accumulate(x, nu, dtype=DTYPE)
        self.assertEqual(moments.sum_outer_products.dtype, DTYPE)
        self.assertEqual(moments.sum_weights.dtype, DTYPE)
        centre = x.mean(dim=0)
        expected = self.direct(x, nu, centre)
        self.assertAlmostEqual(
            reduce_fourth_moment(moments, centre), expected, delta=1e-5 * expected
        )

    def test_invalid_shapes(self) -> None:
        """Inconsistent shapes of the vectors, weights or centre raise ValueError."""
        moments = ReduceMoments.zeros(self.d, dtype=DTYPE)
        for vectors, weights in (
            (torch.randn(self.d), torch.ones(1)),
            (torch.randn(4, self.d + 1), torch.ones(4)),
            (torch.randn(4, self.d), torch.ones(3)),
            (torch.randn(4, self.d), torch.ones(4, 1)),
        ):
            with self.subTest(vectors=vectors.shape, weights=weights.shape):
                with self.assertRaises(ValueError):
                    moments.add(vectors.to(DTYPE), weights)
        with self.assertRaises(ValueError):
            reduce_fourth_moment(moments, torch.zeros(self.d + 1))


if __name__ == "__main__":
    unittest.main()
