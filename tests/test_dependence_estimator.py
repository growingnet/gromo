import unittest

import torch

from gromo.utils.dependence_estimator import *


class TestDependenceEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.X = torch.rand((10, 500))

    def test_gaussian_kernel(self) -> None:
        K = gaussian_kernel(self.X)

        # Check if the kernel matrix is symmetric
        self.assertTrue(torch.allclose(K, K.T, atol=1e-6))

        # Check the shape of the kernel matrix
        self.assertEqual(K.shape, (self.X.shape[0], self.X.shape[0]))

        # Check if the fast Gaussian kernel matches the standard one
        K_slow = slow_gaussian_kernel(self.X)
        self.assertEqual(K.shape, K_slow.shape)
        self.assertTrue(torch.allclose(K, K_slow, atol=1e-6))

        # Check the values of the Gaussian kernel
        sigma = 1.0
        dist = torch.cdist(self.X, self.X) ** 2
        expected_K = torch.exp(-dist / (2 * sigma**2))
        K = gaussian_kernel(self.X, sigma=sigma)
        self.assertTrue(torch.allclose(K, expected_K, atol=1e-6))

    def test_center_kernel_matrix(self) -> None:
        # Create a random kernel matrix
        K = gaussian_kernel(self.X)

        # Center the kernel matrix
        K_centered = center_kernel_matrix(K)

        # Check if the centered kernel matrix is symmetric
        self.assertTrue(torch.allclose(K_centered, K_centered.T, atol=1e-6))

        # Check if the row and column sums of the centered kernel matrix are approximately zero
        self.assertTrue(
            torch.allclose(K_centered.sum(dim=0), torch.zeros(K.shape[0]), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(K_centered.sum(dim=1), torch.zeros(K.shape[0]), atol=1e-6)
        )

        # Check if centering a centered kernel matrix results in the same matrix
        K_recentered = center_kernel_matrix(K_centered)
        self.assertTrue(torch.allclose(K_centered, K_recentered, atol=1e-6))

    def test_HSIC(self) -> None:
        K = gaussian_kernel(self.X)
        L = gaussian_kernel(torch.rand((10, 20)))

        # Center the kernel matrices
        K_centered = center_kernel_matrix(K)
        L_centered = center_kernel_matrix(L)

        # Compute HSIC
        hsic_value = HSIC(K_centered, L_centered)

        # Check if the HSIC value is a scalar
        self.assertTrue(torch.is_tensor(hsic_value))
        self.assertEqual(hsic_value.dim(), 0)

        # Check if HSIC is non-negative
        self.assertGreaterEqual(hsic_value.item(), 0)

        # Check HSIC with identical kernel matrices (should be positive and non-zero)
        hsic_self = HSIC(K_centered, K_centered)
        self.assertGreater(hsic_self.item(), 0)

    def test_calculate_dependency(self) -> None:
        # Create random input data
        X_inputs = {
            "feature1": torch.rand((100, 20)),
            "feature2": torch.rand((100, 30)),
        }
        Y = torch.rand((100, 10))
        n_samples = 10

        # Call the calculate_dependency function
        result = calculate_dependency(X_inputs, Y, n_samples)

        # Check if the result is a dictionary
        self.assertIsInstance(result, dict)

        # Check if the keys in the result match the input keys
        self.assertSetEqual(set(result.keys()), set(X_inputs.keys()))

        # Check if the values in the result are scalars add non-negative
        for key, value in result.items():
            self.assertTrue(torch.is_tensor(value))
            self.assertEqual(value.dim(), 0)
            self.assertGreaterEqual(value.item(), 0)

        # Test with normalization disabled
        result_no_norm = calculate_dependency(X_inputs, Y, n_samples, normalize=False)
        for key in result.keys():
            self.assertNotEqual(result[key].item(), result_no_norm[key].item())

        # More samples
        result_more_samples = calculate_dependency(X_inputs, Y, n_samples=20)
        for key in result.keys():
            self.assertNotEqual(result[key], result_more_samples[key])


if __name__ == "__main__":
    unittest.main()
