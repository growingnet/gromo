"""
Unit tests for GrowingBatchNorm2d and GrowingBatchNorm1d classes.
"""

import unittest

import torch

from gromo.modules.growing_module import SupportsExtendedForward
from gromo.modules.growing_normalisation import (
    GrowingBatchNorm1d,
    GrowingBatchNorm2d,
    GrowingGroupNorm,
    GrowingLayerNorm,
)


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

            # Provide CPU tensors; _extend_parameter should .to(cuda) them transparently.
            cpu_weights = torch.ones(8)  # intentionally on CPU
            bn.grow(8, new_weights=cpu_weights)
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

    def test_extend_parameter_none_param(self):
        """Test _extend_parameter returns early when the attribute is None."""
        bn = GrowingBatchNorm2d(
            num_features=self.initial_features, affine=False, device=self.device
        )
        # weight is None when affine=False; calling _extend_parameter directly
        # should silently return without raising or modifying anything.
        bn._extend_parameter(
            "weight", 8, None, torch.ones, torch.device("cpu"), as_parameter=True
        )
        self.assertIsNone(bn.weight)

    def test_extended_forward(self):
        """Test extended_forward applies BN to x and passes x_ext unchanged."""
        extension_size = 8
        bn = GrowingBatchNorm2d(num_features=self.initial_features, device=self.device)
        bn.eval()

        x = torch.randn(
            self.batch_size,
            self.initial_features,
            self.height,
            self.width,
            device=self.device,
        )
        x_ext = torch.randn(
            self.batch_size,
            extension_size,
            self.height,
            self.width,
            device=self.device,
        )

        self.assertIsInstance(bn, SupportsExtendedForward)

        processed_x, processed_x_ext = bn.extended_forward(x, x_ext)

        self.assertEqual(processed_x.shape, x.shape)
        self.assertEqual(processed_x_ext.shape, x_ext.shape)
        torch.testing.assert_close(processed_x, bn(x))
        torch.testing.assert_close(processed_x_ext, x_ext)


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

    def test_extended_forward(self):
        """Test extended_forward applies BN to x and passes x_ext unchanged."""
        extension_size = 8
        bn = GrowingBatchNorm1d(num_features=self.initial_features, device=self.device)
        bn.eval()

        x = torch.randn(self.batch_size, self.initial_features, device=self.device)
        x_ext = torch.randn(self.batch_size, extension_size, device=self.device)

        self.assertIsInstance(bn, SupportsExtendedForward)

        processed_x, processed_x_ext = bn.extended_forward(x, x_ext)

        self.assertEqual(processed_x.shape, x.shape)
        self.assertEqual(processed_x_ext.shape, x_ext.shape)
        torch.testing.assert_close(processed_x, bn(x))
        torch.testing.assert_close(processed_x_ext, x_ext)


class TestGrowingLayerNorm(unittest.TestCase):
    """Test cases for GrowingLayerNorm class."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_features = 64
        self.batch_size = 4
        self.seq_len = 20

    def test_initialization(self):
        """Test proper initialization of GrowingLayerNorm."""
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features,
            eps=1e-5,
            elementwise_affine=True,
            bias=True,
            device=self.device,
            name="test_ln",
        )
        self.assertEqual(ln.normalized_shape, (self.initial_features,))
        self.assertEqual(ln.name, "test_ln")
        self.assertEqual(ln.eps, 1e-5)
        self.assertEqual(ln.weight.shape[0], self.initial_features)
        self.assertEqual(ln.bias.shape[0], self.initial_features)

    def test_initialization_no_bias(self):
        """Test initialization with bias=False."""
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features,
            elementwise_affine=True,
            bias=False,
            device=self.device,
        )
        self.assertIsNotNone(ln.weight)
        self.assertIsNone(ln.bias)

    def test_initialization_no_affine(self):
        """Test initialization with elementwise_affine=False."""
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features,
            elementwise_affine=False,
            device=self.device,
        )
        self.assertIsNone(ln.weight)
        self.assertIsNone(ln.bias)

    def test_forward_pass(self):
        """Test forward pass with original shape."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)
        x = torch.randn(
            self.batch_size, self.seq_len, self.initial_features, device=self.device
        )
        output = ln(x)
        self.assertEqual(output.shape, x.shape)

    def test_grow_default_parameters(self):
        """Test growing with default (ones/zeros) parameters."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)
        additional = 16
        orig_weight = ln.weight.data.clone()
        orig_bias = ln.bias.data.clone()

        ln.grow(additional)

        expected = self.initial_features + additional
        self.assertEqual(ln.normalized_shape, (expected,))
        self.assertEqual(ln.weight.shape[0], expected)
        self.assertEqual(ln.bias.shape[0], expected)

        # Original params preserved
        torch.testing.assert_close(ln.weight.data[: self.initial_features], orig_weight)
        torch.testing.assert_close(ln.bias.data[: self.initial_features], orig_bias)

        # New params are ones / zeros
        torch.testing.assert_close(
            ln.weight.data[self.initial_features :],
            torch.ones(additional, device=self.device),
        )
        torch.testing.assert_close(
            ln.bias.data[self.initial_features :],
            torch.zeros(additional, device=self.device),
        )

    def test_grow_custom_parameters(self):
        """Test growing with custom weights and biases."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)
        additional = 8
        custom_weights = torch.full((additional,), 2.0, device=self.device)
        custom_biases = torch.full((additional,), -1.0, device=self.device)

        ln.grow(additional, new_weights=custom_weights, new_biases=custom_biases)

        torch.testing.assert_close(
            ln.weight.data[self.initial_features :], custom_weights
        )
        torch.testing.assert_close(ln.bias.data[self.initial_features :], custom_biases)

    def test_grow_no_bias(self):
        """Test growing when bias=False: only weight is extended."""
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features, bias=False, device=self.device
        )
        additional = 8
        ln.grow(additional)

        expected = self.initial_features + additional
        self.assertEqual(ln.normalized_shape, (expected,))
        self.assertEqual(ln.weight.shape[0], expected)
        self.assertIsNone(ln.bias)

    def test_grow_no_affine(self):
        """Test growing when elementwise_affine=False: no params extended."""
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features,
            elementwise_affine=False,
            device=self.device,
        )
        additional = 8
        ln.grow(additional)

        self.assertEqual(ln.normalized_shape, (self.initial_features + additional,))
        self.assertIsNone(ln.weight)
        self.assertIsNone(ln.bias)

    def test_grow_multiple_times(self):
        """Test growing multiple times accumulates correctly."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)
        ln.grow(8)
        ln.grow(16)
        ln.grow(4)

        expected = self.initial_features + 8 + 16 + 4
        self.assertEqual(ln.normalized_shape, (expected,))
        self.assertEqual(ln.weight.shape[0], expected)
        self.assertEqual(ln.bias.shape[0], expected)

    def test_forward_after_growth(self):
        """Test forward pass after growing."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)
        ln.grow(16)
        new_features = self.initial_features + 16
        x = torch.randn(self.batch_size, self.seq_len, new_features, device=self.device)
        output = ln(x)
        self.assertEqual(output.shape, x.shape)

    def test_grow_multi_dim_normalized_shape(self):
        """Test growing the first (channel) dim of a 2-D normalized_shape."""
        H, W = 8, 16
        ln = GrowingLayerNorm(normalized_shape=[H, W], device=self.device)
        additional = 8

        ln.grow(additional)

        self.assertEqual(ln.normalized_shape, (H + additional, W))
        self.assertEqual(ln.weight.shape, (H + additional, W))
        self.assertEqual(ln.bias.shape, (H + additional, W))

        # Forward pass with correctly-grown input
        x = torch.randn(self.batch_size, H + additional, W, device=self.device)
        out = ln(x)
        self.assertEqual(out.shape, x.shape)

    def test_grow_error_cases(self):
        """Test ValueError for non-positive additional_last_dim and wrong custom shapes."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)

        with self.assertRaises(ValueError):
            ln.grow(0)

        with self.assertRaises(ValueError):
            ln.grow(-1)

        # Wrong-size custom weights (must match additional)
        with self.assertRaises(ValueError):
            ln.grow(8, new_weights=torch.ones(5))

        # Wrong-size custom biases
        with self.assertRaises(ValueError):
            ln.grow(8, new_biases=torch.zeros(10))

    def test_get_growth_info(self):
        """Test get_growth_info returns correct dict."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, name="info_ln")
        info = ln.get_growth_info()
        self.assertEqual(info["normalized_shape"], (self.initial_features,))
        self.assertEqual(info["name"], "info_ln")

        ln.grow(8)
        info = ln.get_growth_info()
        self.assertEqual(info["normalized_shape"], (self.initial_features + 8,))

    def test_extra_repr(self):
        """Test extra_repr returns a string containing the name."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, name="repr_ln")
        repr_str = ln.extra_repr()
        self.assertIsInstance(repr_str, str)
        self.assertIn("repr_ln", repr_str)

    def test_extend_parameter_none_param(self):
        """Test _extend_parameter returns early when the attribute is None."""
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features,
            elementwise_affine=False,
            device=self.device,
        )
        # weight is None; _extend_parameter should silently return without error.
        ln._extend_parameter(
            "weight", 8, None, torch.ones, torch.device("cpu"), as_parameter=True
        )
        self.assertIsNone(ln.weight)

    def test_extend_parameter_as_buffer(self):
        """Test as_parameter=False path stores result as a buffer."""
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)
        device = ln.weight.device
        # Register a fresh buffer (not an nn.Parameter) so register_buffer succeeds.
        ln.register_buffer(
            "running_stat",
            torch.zeros(self.initial_features, device=device),
        )
        ln._extend_parameter(
            "running_stat", 8, None, torch.zeros, device, as_parameter=False
        )
        buffers = dict(ln.named_buffers())
        self.assertIn("running_stat", buffers)
        self.assertEqual(buffers["running_stat"].shape[0], self.initial_features + 8)

    def test_grow_device_inferred_from_weight(self):
        """Test that device is inferred from self.weight when not passed."""
        # Construct on CPU explicitly and call grow() without the device kwarg.
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features, device=torch.device("cpu")
        )
        ln.grow(8)  # device is None
        self.assertEqual(ln.weight.device.type, "cpu")
        self.assertEqual(ln.weight.shape[0], self.initial_features + 8)

    def test_grow_custom_params_device_transfer(self):
        """Test that custom params on a different device are moved automatically."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        ln = GrowingLayerNorm(
            normalized_shape=self.initial_features, device=torch.device("cuda")
        )
        cpu_weights = torch.ones(8)  # intentionally on CPU
        ln.grow(8, new_weights=cpu_weights)
        self.assertEqual(ln.weight.device.type, "cuda")

    def test_extended_forward(self):
        """Test extended_forward applies LayerNorm to x and passes x_ext unchanged."""
        extension_size = 8
        ln = GrowingLayerNorm(normalized_shape=self.initial_features, device=self.device)

        x = torch.randn(
            self.batch_size, self.seq_len, self.initial_features, device=self.device
        )
        x_ext = torch.randn(
            self.batch_size, self.seq_len, extension_size, device=self.device
        )

        self.assertIsInstance(ln, SupportsExtendedForward)

        processed_x, processed_x_ext = ln.extended_forward(x, x_ext)

        self.assertEqual(processed_x.shape, x.shape)
        self.assertEqual(processed_x_ext.shape, x_ext.shape)
        torch.testing.assert_close(processed_x, ln(x))
        torch.testing.assert_close(processed_x_ext, x_ext)


class TestGrowingGroupNorm(unittest.TestCase):
    """Test cases for GrowingGroupNorm class."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_groups = 4
        self.initial_channels = 32
        self.batch_size = 4
        self.height = 8
        self.width = 8

    def test_initialization(self):
        """Test proper initialization of GrowingGroupNorm."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            eps=1e-5,
            affine=True,
            device=self.device,
            name="test_gn",
        )
        self.assertEqual(gn.num_groups, self.num_groups)
        self.assertEqual(gn.num_channels, self.initial_channels)
        self.assertEqual(gn.name, "test_gn")
        self.assertEqual(gn.eps, 1e-5)
        self.assertEqual(gn.weight.shape[0], self.initial_channels)
        self.assertEqual(gn.bias.shape[0], self.initial_channels)

    def test_initialization_no_affine(self):
        """Test initialization with affine=False."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            affine=False,
            device=self.device,
        )
        self.assertIsNone(gn.weight)
        self.assertIsNone(gn.bias)

    def test_forward_pass(self):
        """Test forward pass with original channel count."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        x = torch.randn(
            self.batch_size,
            self.initial_channels,
            self.height,
            self.width,
            device=self.device,
        )
        output = gn(x)
        self.assertEqual(output.shape, x.shape)

    def test_grow_default_parameters(self):
        """Test growing with default (ones/zeros) parameters."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        additional = 8  # 32 + 8 = 40; 40 % 4 == 0
        orig_weight = gn.weight.data.clone()
        orig_bias = gn.bias.data.clone()

        gn.grow(additional)

        expected = self.initial_channels + additional
        self.assertEqual(gn.num_channels, expected)
        self.assertEqual(gn.weight.shape[0], expected)
        self.assertEqual(gn.bias.shape[0], expected)

        # Original params preserved
        torch.testing.assert_close(gn.weight.data[: self.initial_channels], orig_weight)
        torch.testing.assert_close(gn.bias.data[: self.initial_channels], orig_bias)

        # New params are ones / zeros
        torch.testing.assert_close(
            gn.weight.data[self.initial_channels :],
            torch.ones(additional, device=self.device),
        )
        torch.testing.assert_close(
            gn.bias.data[self.initial_channels :],
            torch.zeros(additional, device=self.device),
        )

    def test_grow_custom_parameters(self):
        """Test growing with custom weights and biases."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        additional = 8
        custom_weights = torch.full((additional,), 0.5, device=self.device)
        custom_biases = torch.full((additional,), -0.5, device=self.device)

        gn.grow(additional, new_weights=custom_weights, new_biases=custom_biases)

        torch.testing.assert_close(
            gn.weight.data[self.initial_channels :], custom_weights
        )
        torch.testing.assert_close(gn.bias.data[self.initial_channels :], custom_biases)

    def test_grow_no_affine(self):
        """Test growing when affine=False: no params extended."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            affine=False,
            device=self.device,
        )
        gn.grow(8)  # 32 + 8 = 40; 40 % 4 == 0
        self.assertEqual(gn.num_channels, self.initial_channels + 8)
        self.assertIsNone(gn.weight)
        self.assertIsNone(gn.bias)

    def test_grow_with_new_num_groups(self):
        """Test that new_num_groups updates num_groups during growth."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        # 32 + 8 = 40; 40 % 8 == 0
        gn.grow(8, new_num_groups=8)
        self.assertEqual(gn.num_groups, 8)
        self.assertEqual(gn.num_channels, self.initial_channels + 8)

    def test_grow_multiple_times(self):
        """Test growing multiple times accumulates correctly."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        gn.grow(8)  # 40 % 4 == 0
        gn.grow(4)  # 44 % 4 == 0
        gn.grow(12)  # 56 % 4 == 0

        expected = self.initial_channels + 8 + 4 + 12
        self.assertEqual(gn.num_channels, expected)
        self.assertEqual(gn.weight.shape[0], expected)
        self.assertEqual(gn.bias.shape[0], expected)

    def test_forward_after_growth(self):
        """Test forward pass after growing."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        gn.grow(8)
        new_channels = self.initial_channels + 8
        x = torch.randn(
            self.batch_size, new_channels, self.height, self.width, device=self.device
        )
        output = gn(x)
        self.assertEqual(output.shape, x.shape)

    def test_grow_error_non_positive(self):
        """Test ValueError for non-positive additional_channels."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        with self.assertRaises(ValueError):
            gn.grow(0)
        with self.assertRaises(ValueError):
            gn.grow(-4)

    def test_grow_error_not_divisible(self):
        """Test ValueError when result is not divisible by num_groups."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        # 32 + 3 = 35; 35 % 4 != 0
        with self.assertRaises(ValueError):
            gn.grow(3)

    def test_grow_wrong_shape_custom_params(self):
        """Test ValueError for wrong-shape custom weights/biases."""
        # Wrong size
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        with self.assertRaises(ValueError):
            gn.grow(8, new_weights=torch.ones(5))

        gn2 = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        with self.assertRaises(ValueError):
            gn2.grow(8, new_biases=torch.zeros(10))

        # Wrong ndim (2-D tensor instead of 1-D)
        gn3 = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        with self.assertRaises(ValueError):
            gn3.grow(8, new_weights=torch.ones(8, 2))

    def test_get_growth_info(self):
        """Test get_growth_info returns correct dict."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            name="info_gn",
        )
        info = gn.get_growth_info()
        self.assertEqual(info["num_channels"], self.initial_channels)
        self.assertEqual(info["num_groups"], self.num_groups)
        self.assertEqual(info["name"], "info_gn")

        gn.grow(8)
        info = gn.get_growth_info()
        self.assertEqual(info["num_channels"], self.initial_channels + 8)

    def test_extra_repr(self):
        """Test extra_repr returns a string containing the name."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            name="repr_gn",
        )
        repr_str = gn.extra_repr()
        self.assertIsInstance(repr_str, str)
        self.assertIn("repr_gn", repr_str)

    def test_extend_parameter_none_param(self):
        """Test _extend_parameter returns early when the attribute is None."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            affine=False,
            device=self.device,
        )
        # weight is None when affine=False; should silently return.
        gn._extend_parameter(
            "weight", 8, None, torch.ones, torch.device("cpu"), as_parameter=True
        )
        self.assertIsNone(gn.weight)

    def test_extend_parameter_as_buffer(self):
        """Test as_parameter=False path stores result as a buffer."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )
        device = gn.weight.device
        # Register a fresh buffer (not an nn.Parameter) so register_buffer succeeds.
        gn.register_buffer(
            "running_stat",
            torch.zeros(self.initial_channels, device=device),
        )
        gn._extend_parameter(
            "running_stat", 8, None, torch.zeros, device, as_parameter=False
        )
        buffers = dict(gn.named_buffers())
        self.assertIn("running_stat", buffers)
        self.assertEqual(buffers["running_stat"].shape[0], self.initial_channels + 8)

    def test_grow_device_inferred_from_weight(self):
        """Test that device is inferred from self.weight when not passed."""
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=torch.device("cpu"),
        )
        gn.grow(8)  # device is None
        self.assertEqual(gn.weight.device.type, "cpu")
        self.assertEqual(gn.weight.shape[0], self.initial_channels + 8)

    def test_grow_custom_params_device_transfer(self):
        """Test that custom params on a different device are moved automatically."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=torch.device("cuda"),
        )
        cpu_weights = torch.ones(8)  # intentionally on CPU
        gn.grow(8, new_weights=cpu_weights)
        self.assertEqual(gn.weight.device.type, "cuda")

    def test_extended_forward(self):
        """Test extended_forward applies GroupNorm to x and passes x_ext unchanged."""
        extension_size = 4  # must be divisible by num_groups if tested standalone
        gn = GrowingGroupNorm(
            num_groups=self.num_groups,
            num_channels=self.initial_channels,
            device=self.device,
        )

        x = torch.randn(
            self.batch_size,
            self.initial_channels,
            self.height,
            self.width,
            device=self.device,
        )
        x_ext = torch.randn(
            self.batch_size,
            extension_size,
            self.height,
            self.width,
            device=self.device,
        )

        self.assertIsInstance(gn, SupportsExtendedForward)

        processed_x, processed_x_ext = gn.extended_forward(x, x_ext)

        self.assertEqual(processed_x.shape, x.shape)
        self.assertEqual(processed_x_ext.shape, x_ext.shape)
        torch.testing.assert_close(processed_x, gn(x))
        torch.testing.assert_close(processed_x_ext, x_ext)


if __name__ == "__main__":
    unittest.main()
