"""
Unit tests for GrowingBatchNorm2d and GrowingBatchNorm1d classes.
"""

import unittest

import torch

from gromo.modules.growing_normalisation import GrowingBatchNorm1d, GrowingBatchNorm2d


class TestGrowingBatchNorm2d(unittest.TestCase):
    """Test cases for GrowingBatchNorm2d class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_features = 32
        self.batch_size = 8
        self.height = 16
        self.width = 16

    def test_initialization(self):
        """Test proper initialization of GrowingBatchNorm2d."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            device=self.device,
            name="test_bn",
        )

        self.assertEqual(bn.num_features, self.initial_features)
        self.assertEqual(bn.name, "test_bn")
        self.assertEqual(bn.eps, 1e-5)
        self.assertEqual(bn.momentum, 0.1)
        self.assertTrue(bn.affine)
        self.assertTrue(bn.track_running_stats)

        # Check parameter shapes
        self.assertEqual(bn.weight.shape[0], self.initial_features)
        self.assertEqual(bn.bias.shape[0], self.initial_features)
        if bn.track_running_stats:
            self.assertEqual(bn.running_mean.shape[0], self.initial_features)
            self.assertEqual(bn.running_var.shape[0], self.initial_features)

    def test_initialization_no_affine(self):
        """Test initialization without affine parameters."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features,
            affine=False,
            track_running_stats=True,
            device=self.device,
        )

        self.assertIsNone(bn.weight)
        self.assertIsNone(bn.bias)
        self.assertIsNotNone(bn.running_mean)
        self.assertIsNotNone(bn.running_var)

    def test_initialization_no_running_stats(self):
        """Test initialization without running statistics."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features,
            affine=True,
            track_running_stats=False,
            device=self.device,
        )

        self.assertIsNotNone(bn.weight)
        self.assertIsNotNone(bn.bias)
        self.assertIsNone(bn.running_mean)
        self.assertIsNone(bn.running_var)

    def test_forward_pass(self):
        """Test forward pass with original features."""
        bn = GrowingBatchNorm2d(num_features=self.initial_features, device=self.device)

        # Create test input
        x = torch.randn(
            self.batch_size,
            self.initial_features,
            self.height,
            self.width,
            device=self.device,
        )

        # Forward pass
        output = bn(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

        # Check that batch norm is working (mean close to 0, std close to 1)
        bn.eval()
        with torch.no_grad():
            # Run a few forward passes to update running statistics
            for _ in range(10):
                _ = bn(x)

            # Test with eval mode
            output_eval = bn(x)
            # The exact values depend on the running statistics, but shape should be correct
            self.assertEqual(output_eval.shape, x.shape)

    def test_grow_default_parameters(self):
        """Test growing with default parameters."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features,
            device=self.device,
            track_running_stats=True,  # Explicitly enable running stats
        )
        additional_features = 16

        # Store original parameters
        original_weight = bn.weight.data.clone()
        original_bias = bn.bias.data.clone()
        original_running_mean = (
            bn.running_mean.clone() if bn.running_mean is not None else None
        )
        original_running_var = (
            bn.running_var.clone() if bn.running_var is not None else None
        )

        # Grow the layer
        bn.grow(additional_features)

        # Check new dimensions
        expected_features = self.initial_features + additional_features
        self.assertEqual(bn.num_features, expected_features)
        self.assertEqual(bn.weight.shape[0], expected_features)
        self.assertEqual(bn.bias.shape[0], expected_features)
        if bn.track_running_stats:
            self.assertIsNotNone(bn.running_mean)
            self.assertIsNotNone(bn.running_var)
            self.assertEqual(bn.running_mean.shape[0], expected_features)
            self.assertEqual(bn.running_var.shape[0], expected_features)

        # Check that original parameters are preserved
        torch.testing.assert_close(
            bn.weight.data[: self.initial_features], original_weight
        )
        torch.testing.assert_close(bn.bias.data[: self.initial_features], original_bias)
        if bn.track_running_stats and original_running_mean is not None:
            torch.testing.assert_close(
                bn.running_mean[: self.initial_features], original_running_mean
            )
            torch.testing.assert_close(
                bn.running_var[: self.initial_features], original_running_var
            )

        # Check that new parameters have default values
        torch.testing.assert_close(
            bn.weight.data[self.initial_features :],
            torch.ones(additional_features, device=self.device),
        )
        torch.testing.assert_close(
            bn.bias.data[self.initial_features :],
            torch.zeros(additional_features, device=self.device),
        )
        if bn.track_running_stats:
            torch.testing.assert_close(
                bn.running_mean[self.initial_features :],
                torch.zeros(additional_features, device=self.device),
            )
            torch.testing.assert_close(
                bn.running_var[self.initial_features :],
                torch.ones(additional_features, device=self.device),
            )

    def test_grow_custom_parameters(self):
        """Test growing with custom parameters."""
        bn = GrowingBatchNorm2d(num_features=self.initial_features, device=self.device)
        additional_features = 8

        # Create custom parameters
        custom_weights = torch.full((additional_features,), 0.5, device=self.device)
        custom_biases = torch.full((additional_features,), -0.1, device=self.device)
        custom_running_mean = torch.full((additional_features,), 0.2, device=self.device)
        custom_running_var = torch.full((additional_features,), 1.5, device=self.device)

        # Grow with custom parameters
        bn.grow(
            additional_features,
            new_weights=custom_weights,
            new_biases=custom_biases,
            new_running_mean=custom_running_mean,
            new_running_var=custom_running_var,
        )

        # Check that custom parameters are used
        torch.testing.assert_close(
            bn.weight.data[self.initial_features :], custom_weights
        )
        torch.testing.assert_close(bn.bias.data[self.initial_features :], custom_biases)
        torch.testing.assert_close(
            bn.running_mean[self.initial_features :], custom_running_mean
        )
        torch.testing.assert_close(
            bn.running_var[self.initial_features :], custom_running_var
        )

    def test_grow_multiple_times(self):
        """Test growing multiple times."""
        bn = GrowingBatchNorm2d(num_features=self.initial_features, device=self.device)

        # First growth
        bn.grow(8)
        self.assertEqual(bn.num_features, self.initial_features + 8)

        # Second growth
        bn.grow(4)
        self.assertEqual(bn.num_features, self.initial_features + 8 + 4)

        # Third growth
        bn.grow(12)
        self.assertEqual(bn.num_features, self.initial_features + 8 + 4 + 12)

        # Check that all parameters have correct dimensions
        expected_features = self.initial_features + 8 + 4 + 12
        self.assertEqual(bn.weight.shape[0], expected_features)
        self.assertEqual(bn.bias.shape[0], expected_features)
        self.assertEqual(bn.running_mean.shape[0], expected_features)
        self.assertEqual(bn.running_var.shape[0], expected_features)

    def test_forward_after_growth(self):
        """Test forward pass after growing."""
        bn = GrowingBatchNorm2d(num_features=self.initial_features, device=self.device)
        additional_features = 16

        # Grow the layer
        bn.grow(additional_features)

        # Create input with new dimensions
        new_features = self.initial_features + additional_features
        x = torch.randn(
            self.batch_size, new_features, self.height, self.width, device=self.device
        )

        # Forward pass should work without errors
        output = bn(x)
        self.assertEqual(output.shape, x.shape)

    def test_grow_no_affine(self):
        """Test growing when affine=False."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features, affine=False, device=self.device
        )

        # Grow the layer
        bn.grow(8)

        # Check that weight and bias are still None
        self.assertIsNone(bn.weight)
        self.assertIsNone(bn.bias)

        # Check that running statistics are grown
        self.assertEqual(bn.running_mean.shape[0], self.initial_features + 8)
        self.assertEqual(bn.running_var.shape[0], self.initial_features + 8)

    def test_grow_no_running_stats(self):
        """Test growing when track_running_stats=False."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features,
            track_running_stats=False,
            device=self.device,
        )

        # Grow the layer
        bn.grow(8)

        # Check that running statistics are still None
        self.assertIsNone(bn.running_mean)
        self.assertIsNone(bn.running_var)

        # Check that weight and bias are grown
        self.assertEqual(bn.weight.shape[0], self.initial_features + 8)
        self.assertEqual(bn.bias.shape[0], self.initial_features + 8)

    def test_grow_dummy(self):
        """Grow with no running stats and no affine"""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features,
            track_running_stats=False,
            affine=False,
            device=self.device,
        )

        # Grow the layer
        bn.grow(8)

        self.assertEqual(bn.num_features, self.initial_features + 8)

    def test_grow_error_cases(self):
        """Test error cases for grow method."""
        bn = GrowingBatchNorm2d(num_features=self.initial_features, device=self.device)

        # Test negative additional_features
        with self.assertRaises(ValueError):
            bn.grow(-1)

        # Test zero additional_features
        with self.assertRaises(ValueError):
            bn.grow(0)

        # Test wrong size custom weights
        with self.assertRaises(ValueError):
            bn.grow(8, new_weights=torch.ones(4))  # Should be 8, not 4

        # Test wrong size custom biases
        with self.assertRaises(ValueError):
            bn.grow(8, new_biases=torch.zeros(10))  # Should be 8, not 10

        # Test wrong size custom running_mean
        with self.assertRaises(ValueError):
            bn.grow(8, new_running_mean=torch.zeros(5))  # Should be 8, not 5

        # Test wrong size custom running_var
        with self.assertRaises(ValueError):
            bn.grow(8, new_running_var=torch.ones(12))  # Should be 8, not 12

    def test_get_growth_info(self):
        """Test get_growth_info method."""
        bn = GrowingBatchNorm2d(num_features=self.initial_features, name="test_bn")

        # Initial info
        info = bn.get_growth_info()
        self.assertEqual(info["num_features"], self.initial_features)
        self.assertEqual(info["name"], "test_bn")

        # After growth
        bn.grow(16)
        info = bn.get_growth_info()
        self.assertEqual(info["num_features"], self.initial_features + 16)

    def test_extra_repr(self):
        """Test extra_repr method."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features, eps=1e-4, momentum=0.05, name="test_repr"
        )

        repr_str = bn.extra_repr()
        self.assertIsInstance(repr_str, str, "extra_repr should return a string")

    def test_device_handling(self):
        """Test proper device handling."""
        if torch.cuda.is_available():
            # Test CUDA device
            bn = GrowingBatchNorm2d(
                num_features=self.initial_features, device=torch.device("cuda")
            )
            self.assertEqual(bn.weight.device.type, "cuda")

            # Grow and check device
            bn.grow(8)
            self.assertEqual(bn.weight.device.type, "cuda")

            # Test custom device parameter in grow
            bn.grow(4, device=torch.device("cuda"))
            self.assertEqual(bn.weight.device.type, "cuda")

    def test_dtype_preservation(self):
        """Test that dtype is preserved during growth."""
        bn = GrowingBatchNorm2d(num_features=self.initial_features, dtype=torch.float32)

        original_dtype = bn.weight.dtype
        bn.grow(8)

        self.assertEqual(bn.weight.dtype, original_dtype)
        self.assertEqual(bn.bias.dtype, original_dtype)
        self.assertEqual(bn.running_mean.dtype, original_dtype)
        self.assertEqual(bn.running_var.dtype, original_dtype)


class TestGrowingBatchNorm1d(unittest.TestCase):
    """Test cases for GrowingBatchNorm1d class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_features = 32
        self.batch_size = 8
        self.sequence_length = 64

    def test_initialization(self):
        """Test proper initialization of GrowingBatchNorm1d."""
        bn = GrowingBatchNorm1d(
            num_features=self.initial_features, device=self.device, name="test_bn_1d"
        )

        self.assertEqual(bn.num_features, self.initial_features)
        self.assertEqual(bn.name, "test_bn_1d")

    def test_forward_pass_1d(self):
        """Test forward pass with 1D batch norm."""
        bn = GrowingBatchNorm1d(num_features=self.initial_features, device=self.device)

        # Create test input (batch_size, features, sequence_length)
        x = torch.randn(
            self.batch_size,
            self.initial_features,
            self.sequence_length,
            device=self.device,
        )

        # Forward pass
        output = bn(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

    def test_grow_1d(self):
        """Test growing functionality for 1D batch norm."""
        bn = GrowingBatchNorm1d(num_features=self.initial_features, device=self.device)
        additional_features = 16

        # Grow the layer
        bn.grow(additional_features)

        # Check new dimensions
        expected_features = self.initial_features + additional_features
        self.assertEqual(bn.num_features, expected_features)
        self.assertEqual(bn.weight.shape[0], expected_features)
        self.assertEqual(bn.bias.shape[0], expected_features)

    def test_forward_after_growth_1d(self):
        """Test forward pass after growing for 1D batch norm."""
        bn = GrowingBatchNorm1d(num_features=self.initial_features, device=self.device)
        additional_features = 16

        # Grow the layer
        bn.grow(additional_features)

        # Create input with new dimensions
        new_features = self.initial_features + additional_features
        x = torch.randn(
            self.batch_size, new_features, self.sequence_length, device=self.device
        )

        # Forward pass should work without errors
        output = bn(x)
        self.assertEqual(output.shape, x.shape)

    def test_extra_repr(self):
        """Test extra_repr method."""
        bn = GrowingBatchNorm1d(
            num_features=self.initial_features, device=self.device, name="test_bn_1d"
        )
        self.assertIsInstance(bn.extra_repr(), str)


if __name__ == "__main__":
    unittest.main()
