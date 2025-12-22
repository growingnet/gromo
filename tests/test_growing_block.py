import torch

from gromo.containers.growing_block import GrowingBlock, LinearGrowingBlock
from gromo.utils.utils import global_device


try:
    from tests.torch_unittest import TorchTestCase, indicator_batch
    from tests.unittest_tools import unittest_parametrize
except ImportError:
    from torch_unittest import TorchTestCase, indicator_batch
    from unittest_tools import unittest_parametrize


class TestGrowingBlock(TorchTestCase):
    """Test GrowingBlock base class functionality."""

    def test_set_default_values(self):
        """Test the static method set_default_values."""
        activation = torch.nn.ReLU()

        # Test with all None values
        pre_act, mid_act, kwargs_first, kwargs_second = GrowingBlock.set_default_values(
            activation=activation
        )
        self.assertEqual(pre_act, activation)
        self.assertEqual(mid_act, activation)
        self.assertEqual(kwargs_first, dict())
        self.assertEqual(kwargs_second, dict())

        # Test with some values provided
        pre_activation = torch.nn.Sigmoid()
        kwargs_layer = {"use_bias": False}

        pre_act, mid_act, kwargs_first, kwargs_second = GrowingBlock.set_default_values(
            activation=activation,
            pre_activation=pre_activation,
            kwargs_layer=kwargs_layer,
        )
        self.assertEqual(pre_act, pre_activation)
        self.assertEqual(mid_act, activation)
        self.assertEqual(kwargs_first, kwargs_layer)
        self.assertEqual(kwargs_second, kwargs_layer)

        # Test with explicit kwargs_first_layer and kwargs_second_layer
        # (covers lines 116 and 126)
        kwargs_first_explicit = {"use_bias": True, "device": "cpu"}
        kwargs_second_explicit = {"use_bias": False, "device": "cuda"}

        pre_act, mid_act, kwargs_first, kwargs_second = GrowingBlock.set_default_values(
            activation=activation,
            pre_activation=pre_activation,
            kwargs_layer=kwargs_layer,
            kwargs_first_layer=kwargs_first_explicit,
            kwargs_second_layer=kwargs_second_explicit,
        )
        self.assertEqual(pre_act, pre_activation)
        self.assertEqual(mid_act, activation)
        self.assertEqual(
            kwargs_first, kwargs_first_explicit
        )  # Should use explicit value, not kwargs_layer
        self.assertEqual(
            kwargs_second, kwargs_second_explicit
        )  # Should use explicit value, not kwargs_layer

        # Test with only kwargs_first_layer provided (covers line 116)
        pre_act, mid_act, kwargs_first, kwargs_second = GrowingBlock.set_default_values(
            activation=None,
            kwargs_layer=kwargs_layer,
            kwargs_first_layer=kwargs_first_explicit,
        )
        self.assertIsInstance(pre_act, torch.nn.Identity)  # Should fall back to identity
        self.assertIsInstance(mid_act, torch.nn.Identity)  # Should fall back to identity
        self.assertEqual(kwargs_first, kwargs_first_explicit)  # Should use explicit value
        self.assertEqual(kwargs_second, kwargs_layer)  # Should fallback to kwargs_layer

        # Test with only kwargs_second_layer provided (covers line 126)
        pre_act, mid_act, kwargs_first, kwargs_second = GrowingBlock.set_default_values(
            activation=activation,
            kwargs_layer=kwargs_layer,
            kwargs_second_layer=kwargs_second_explicit,
        )
        self.assertEqual(kwargs_first, kwargs_layer)  # Should fallback to kwargs_layer
        self.assertEqual(
            kwargs_second, kwargs_second_explicit
        )  # Should use explicit value


class ScalingModule(torch.nn.Module):
    def __init__(self, scaling_factor: float = 1.0):
        super(ScalingModule, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling_factor * x


class TestLinearGrowingBlock(TorchTestCase):
    """Test LinearGrowingBlock functionality."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = global_device()
        self.batch_size = 4
        self.in_features = 3
        self.out_features = 5
        self.hidden_features = 2
        self.added_features = 7
        self.scaling_factor = 0.9
        self.downsample = torch.nn.Linear(
            self.in_features, self.out_features, device=self.device
        )
        self.first_layer_extension = torch.nn.Linear(
            self.in_features, self.added_features, device=self.device
        )
        self.second_layer_extension = torch.nn.Linear(
            self.added_features, self.out_features, device=self.device
        )
        self.second_layer_extension_no_downsample = torch.nn.Linear(
            self.added_features, self.in_features, device=self.device
        )

    def test_init_with_zero_features(self):
        """Test initialization with 0 hidden features."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=0,
            device=self.device,
            name="zero_block",
        )

        # Check basic properties
        self.assertEqual(block.in_features, self.in_features)
        self.assertEqual(block.out_features, self.in_features)
        self.assertEqual(block.hidden_features, 0)
        self.assertEqual(block.name, "zero_block")

        # Check layer configurations
        self.assertEqual(block.first_layer.in_features, self.in_features)
        self.assertEqual(block.first_layer.out_features, 0)  # hidden_features = 0
        self.assertEqual(block.second_layer.in_features, 0)
        self.assertEqual(block.second_layer.out_features, self.in_features)

        # Check that layers are connected
        self.assertIs(block.second_layer.previous_module, block.first_layer)

        with self.subTest("Test __str__ method"):
            self.assertIsInstance(str(block), str)
            self.assertIsInstance(block.__str__(verbose=1), str)
            self.assertIsInstance(block.__str__(verbose=2), str)
            with self.assertRaises(ValueError):
                block.__str__(verbose=-1)

    def test_init_with_positive_features(self):
        """Test initialization with >0 hidden features."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
            name="positive_block",
        )

        # Check basic properties
        self.assertEqual(block.in_features, self.in_features)
        self.assertEqual(block.out_features, self.in_features)
        self.assertEqual(block.hidden_features, self.hidden_features)

        # Check layer configurations
        self.assertEqual(block.first_layer.in_features, self.in_features)
        self.assertEqual(block.first_layer.out_features, self.hidden_features)
        self.assertEqual(block.second_layer.in_features, self.hidden_features)
        self.assertEqual(block.second_layer.out_features, self.in_features)

    def test_init_with_custom_activations(self):
        """Test initialization with custom activation functions."""
        activation = torch.nn.ReLU()
        pre_activation = torch.nn.Sigmoid()
        mid_activation = torch.nn.Tanh()

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            activation=activation,
            pre_activation=pre_activation,
            mid_activation=mid_activation,
            device=self.device,
        )

        self.assertEqual(block.pre_activation, pre_activation)
        # mid_activation should be used as post_layer_function for first_layer
        self.assertEqual(block.first_layer.post_layer_function, mid_activation)

    def test_forward_zero_features_no_downsample(self):
        """Test forward pass with 0 hidden features and no downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=0,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        output = block(x)

        # With 0 hidden features and identity downsample, output should equal input
        self.assertShapeEqual(output, x.shape)
        self.assertAllClose(output, x)

    def test_forward_zero_features_with_downsample(self):
        """Test forward pass with 0 hidden features and downsample."""
        # Create a downsample that changes dimensions
        downsample = torch.nn.Linear(
            self.in_features, self.out_features, device=self.device
        )

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=0,
            downsample=downsample,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        output = block(x)
        expected_output = downsample(x)

        # With 0 hidden features, forward should return downsample(x)
        self.assertShapeEqual(output, (self.batch_size, self.out_features))
        self.assertAllClose(output, expected_output)

    def test_forward_positive_features_no_downsample(self):
        """Test forward pass with >0 hidden features and no downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        output = block(x)

        # Output should be processed through both layers plus identity
        # output = second_layer(first_layer(pre_activation(x))) + identity(x)
        expected_identity = x  # identity downsample

        # Manual forward pass
        pre_activated = block.pre_activation(x)
        first_out = block.first_layer(pre_activated)
        second_out = block.second_layer(first_out)
        expected_output = second_out + expected_identity

        self.assertShapeEqual(output, (self.batch_size, self.in_features))
        self.assertAllClose(output, expected_output)

    def test_forward_positive_features_with_downsample(self):
        """Test forward pass with >0 hidden features and downsample."""
        downsample = torch.nn.Linear(
            self.in_features, self.out_features, device=self.device
        )

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            downsample=downsample,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        output = block(x)

        # Manual forward pass
        identity = downsample(x)
        pre_activated = block.pre_activation(x)
        first_out = block.first_layer(pre_activated)
        second_out = block.second_layer(first_out)
        expected_output = second_out + identity

        self.assertShapeEqual(output, (self.batch_size, self.out_features))
        self.assertAllClose(output, expected_output)

    def test_input_storage_zero_features_no_downsample(self):
        """Test input and pre-activity storage with 0 hidden features and no
        downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=0,
            device=self.device,
        )

        # Enable storage
        block.first_layer.store_input = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        block(x)

        # For zero features, the block stores pre_activation(x) in first_layer._input
        expected_stored_input = block.pre_activation(x).detach()

        # Check that input is stored correctly
        self.assertAllClose(block.first_layer.input, expected_stored_input)

    def test_input_storage_zero_features_with_downsample(self):
        """Test input and pre-activity storage with 0 hidden features and downsample."""
        downsample = torch.nn.Linear(
            self.in_features, self.out_features, device=self.device
        )

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=0,
            downsample=downsample,
            device=self.device,
        )

        # Enable storage directly
        block.first_layer.store_input = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        block(x)

        # Check storage behavior
        expected_stored_input = block.pre_activation(x).detach()
        self.assertAllClose(block.first_layer.input, expected_stored_input)

    def test_input_storage_positive_features_no_downsample(self):
        """Test input and pre-activity storage with >0 hidden features and no
        downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Enable storage directly
        block.first_layer.store_input = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        block(x)

        # Check that first layer input is stored
        expected_stored_input = block.pre_activation(x).detach()
        self.assertAllClose(block.first_layer.input, expected_stored_input)

        # Check that first layer has processed the input correctly
        self.assertIsNotNone(block.first_layer.input)

    def test_input_storage_positive_features_with_downsample(self):
        """Test input and pre-activity storage with >0 hidden features and downsample."""
        downsample = torch.nn.Linear(
            self.in_features, self.out_features, device=self.device
        )

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            downsample=downsample,
            device=self.device,
        )

        # Enable storage directly
        block.first_layer.store_input = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        block(x)

        # Check storage
        expected_stored_input = block.pre_activation(x).detach()
        self.assertAllClose(block.first_layer.input, expected_stored_input)

    def test_extended_forward_zero_features_no_downsample(self):
        """Test extended_forward with zero hidden features."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=0,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        with self.subTest("No extension"):
            # With zero features and no extension, should return identity
            output = block.extended_forward(x)

            self.assertAllClose(output, x)

        with self.subTest("With extension"):
            # With zero features and an extension
            block.scaling_factor = self.scaling_factor
            block.first_layer.extended_output_layer = self.first_layer_extension
            block.second_layer.extended_input_layer = (
                self.second_layer_extension_no_downsample
            )
            output = block.extended_forward(x)
            expected_output = (
                torch.nn.Sequential(
                    block.pre_activation,
                    self.first_layer_extension,
                    ScalingModule(self.scaling_factor),
                    block.first_layer.extended_post_layer_function,
                    self.second_layer_extension_no_downsample,
                    ScalingModule(self.scaling_factor),
                )(x)
                + x
            )  # identity downsample

            self.assertAllClose(output, expected_output)

    def test_extended_forward_zero_features_with_downsample(self):
        """Test extended_forward with zero hidden features and downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=0,
            downsample=self.downsample,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        with self.subTest("No extension"):
            # With zero features and no extension, should return downsample(x)
            output = block.extended_forward(x)
            expected_output = self.downsample(x)

            self.assertAllClose(output, expected_output)

        with self.subTest("With extension"):
            # With zero features and an extension
            block.scaling_factor = self.scaling_factor
            block.first_layer.extended_output_layer = self.first_layer_extension
            block.second_layer.extended_input_layer = self.second_layer_extension
            output = block.extended_forward(x)
            expected_output = torch.nn.Sequential(
                block.pre_activation,
                self.first_layer_extension,
                ScalingModule(self.scaling_factor),
                block.first_layer.extended_post_layer_function,
                self.second_layer_extension,
                ScalingModule(self.scaling_factor),
            )(x) + self.downsample(x)

            self.assertAllClose(output, expected_output)

    def test_extended_forward_positive_features_no_downsample(self):
        """Test extended_forward with positive hidden features and no downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        with self.subTest("No extension"):
            # With positive features and no extension, should use normal forward
            # through layers
            output = block.extended_forward(x)
            # Use the already tested forward method
            expected_output = block(x)

            self.assertAllClose(output, expected_output)

        with self.subTest("With extension"):
            # With positive features and extension, should use extended_forward
            # method
            block.scaling_factor = self.scaling_factor
            # Set up extensions for both layers
            block.first_layer.extended_output_layer = self.first_layer_extension
            block.second_layer.extended_input_layer = (
                self.second_layer_extension_no_downsample
            )

            # The extended forward for positive features should call the
            # layers' extended_forward methods
            output = block.extended_forward(x)

            # For positive features, the block should call
            # first_layer.extended_forward and second_layer.extended_forward
            # This is complex to replicate exactly, so we'll just verify the
            # shape and that it runs without error
            self.assertShapeEqual(output, (self.batch_size, self.in_features))

            # Now test exact computation by manually calling the layers'
            # extended_forward methods
            identity = x  # identity downsample
            pre_activated = block.pre_activation(x)
            first_out, first_ext = block.first_layer.extended_forward(pre_activated)
            second_out, second_ext = block.second_layer.extended_forward(
                first_out, first_ext
            )
            expected_output = second_out + identity

            self.assertAllClose(output, expected_output)
            self.assertIsNone(second_ext)

    def test_extended_forward_positive_features_with_downsample(self):
        """Test extended_forward with positive hidden features and downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            downsample=self.downsample,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        with self.subTest("No extension"):
            # With positive features and no extension, should use normal forward
            # through layers
            output = block.extended_forward(x)
            # Use the already tested forward method
            expected_output = block(x)

            self.assertAllClose(output, expected_output)

        with self.subTest("With extension"):
            # With positive features and extension, should use extended_forward
            # method
            block.scaling_factor = self.scaling_factor
            # Set up extensions for both layers
            block.first_layer.extended_output_layer = self.first_layer_extension
            block.second_layer.extended_input_layer = self.second_layer_extension

            # The extended forward for positive features should call the
            # layers' extended_forward methods
            output = block.extended_forward(x)

            # For positive features, the block should call
            # first_layer.extended_forward and second_layer.extended_forward
            # This is complex to replicate exactly, so we'll just verify the
            # shape and that it runs without error
            self.assertShapeEqual(output, (self.batch_size, self.out_features))

            # Now test exact computation by manually calling the layers'
            # extended_forward methods
            identity = self.downsample(x)
            pre_activated = block.pre_activation(x)
            first_out, first_ext = block.first_layer.extended_forward(pre_activated)
            second_out, second_ext = block.second_layer.extended_forward(
                first_out, first_ext
            )
            expected_output = second_out + identity

            self.assertAllClose(output, expected_output)
            self.assertIsNone(second_ext)

    def test_pre_activity_storage_zero_features_no_downsample(self):
        """Test pre-activity storage with 0 hidden features and no downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=0,
            device=self.device,
        )

        # Enable pre-activity storage directly
        block.second_layer.store_pre_activity = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        output = block(x)

        # For zero features, the block stores identity (downsample(x)) as pre_activity
        # Since downsample is Identity, it should be zeros with same shape as x
        expected_stored_pre_activity = torch.zeros_like(x)

        # Check that pre-activity is stored correctly
        self.assertAllClose(block.second_layer.pre_activity, expected_stored_pre_activity)

        # Backward pass to compute gradients
        loss = torch.norm(output)
        loss.backward()

        # Check that pre-activity gradient can be accessed
        pre_activity_grad = block.second_layer.pre_activity.grad
        self.assertIsNotNone(pre_activity_grad)
        assert pre_activity_grad is not None  # to avoid type warning
        self.assertShapeEqual(pre_activity_grad, block.second_layer.pre_activity.shape)

    def test_pre_activity_storage_zero_features_with_downsample(self):
        """Test pre-activity storage with 0 hidden features and downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=0,
            downsample=self.downsample,
            device=self.device,
        )

        # Enable pre-activity storage directly
        block.second_layer.store_pre_activity = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        output = block(x)

        # For zero features with downsample, pre_activity should be zeros
        # with same shape as downsample(x)
        expected_stored_pre_activity = torch.zeros_like(self.downsample(x))

        # Check that pre-activity is stored correctly
        self.assertAllClose(block.second_layer.pre_activity, expected_stored_pre_activity)

        # Backward pass to compute gradients
        loss = torch.norm(output)
        loss.backward()

        # Check that pre-activity gradient can be accessed
        pre_activity_grad = block.second_layer.pre_activity.grad
        self.assertIsNotNone(pre_activity_grad)
        assert pre_activity_grad is not None  # to avoid type warning
        self.assertShapeEqual(pre_activity_grad, block.second_layer.pre_activity.shape)

    def test_pre_activity_storage_positive_features_no_downsample(self):
        """Test pre-activity storage with >0 hidden features and no downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Enable pre-activity storage directly
        block.second_layer.store_pre_activity = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        output = block(x)

        # For positive features, pre_activity should be the output of first_layer
        pre_activated = block.pre_activation(x)
        expected_stored_pre_activity = block.second_layer.layer(
            block.first_layer(pre_activated)
        )

        # Check that pre-activity is stored correctly
        self.assertAllClose(block.second_layer.pre_activity, expected_stored_pre_activity)

        # Check that pre-activity has been processed correctly
        self.assertIsNotNone(block.second_layer.pre_activity)

        # Backward pass to compute gradients
        loss = torch.norm(output)
        loss.backward()

        # Check that pre-activity gradient can be accessed
        pre_activity_grad = block.second_layer.pre_activity.grad
        self.assertIsNotNone(pre_activity_grad)
        assert pre_activity_grad is not None  # to avoid type warning
        self.assertShapeEqual(pre_activity_grad, block.second_layer.pre_activity.shape)

        # Verify gradient shape matches the output of first_layer
        expected_shape = (self.batch_size, self.in_features)
        self.assertShapeEqual(pre_activity_grad, expected_shape)

    def test_pre_activity_storage_positive_features_with_downsample(self):
        """Test pre-activity storage with >0 hidden features and downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            downsample=self.downsample,
            device=self.device,
        )

        # Enable pre-activity storage directly
        block.second_layer.store_pre_activity = True

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Forward pass
        output = block(x)

        # For positive features with downsample, pre_activity should be
        # output of first_layer
        pre_activated = block.pre_activation(x)
        expected_stored_pre_activity = block.second_layer.layer(
            block.first_layer(pre_activated)
        )

        # Check that pre-activity is stored correctly
        self.assertAllClose(block.second_layer.pre_activity, expected_stored_pre_activity)

        # Backward pass to compute gradients
        loss = torch.norm(output)
        loss.backward()

        # Check that pre-activity gradient can be accessed
        pre_activity_grad = block.second_layer.pre_activity.grad
        self.assertIsNotNone(pre_activity_grad)
        assert pre_activity_grad is not None  # to avoid type warning
        self.assertShapeEqual(pre_activity_grad, block.second_layer.pre_activity.shape)

        # Verify gradient shape matches the output of first_layer
        expected_shape = (self.batch_size, self.out_features)
        self.assertShapeEqual(pre_activity_grad, expected_shape)

    def test_scaling_factor_property(self):
        """Test scaling factor property getter and setter."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Test getter
        original_scaling_factor = block.scaling_factor
        self.assertEqual(original_scaling_factor, block.second_layer.scaling_factor)

        # Test setter
        new_scaling_factor = 0.5
        block.scaling_factor = new_scaling_factor
        self.assertEqual(block.scaling_factor, new_scaling_factor)
        self.assertEqual(block.second_layer.scaling_factor, new_scaling_factor)

    def test_parameter_update_decrease_setter(self):
        """Test parameter_update_decrease setter with Tensor and float."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Test setter with Tensor
        tensor_value = torch.tensor(0.5, device=self.device)
        block.parameter_update_decrease = tensor_value
        self.assertIsInstance(block.parameter_update_decrease, torch.Tensor)
        assert isinstance(block.second_layer.parameter_update_decrease, torch.Tensor)
        self.assertEqual(
            block.second_layer.parameter_update_decrease.item(),
            tensor_value.item(),
        )

        # Test setter with float (should convert to tensor)
        float_value = 0.3
        block.parameter_update_decrease = float_value
        self.assertAlmostEqual(
            block.second_layer.parameter_update_decrease.item(),
            float_value,
        )

        # Test setter with invalid type
        with self.assertRaises(TypeError):
            block.parameter_update_decrease = "invalid"  # type: ignore

    def test_set_scaling_factor_method(self):
        """Test set_scaling_factor method."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Test set_scaling_factor method
        new_factor = 0.7
        block.set_scaling_factor(new_factor)
        self.assertEqual(block.second_layer.scaling_factor, new_factor)

    @unittest_parametrize(({"hidden_features": 0}, {"hidden_features": 3}))
    def test_init_computation(self, hidden_features: int = 0):
        """Test initialization of computation."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=hidden_features,
            device=self.device,
        )

        # Initialize computation
        block.init_computation()

        # Check that required storage flags are set
        self.assertTrue(block.first_layer.store_input)
        self.assertTrue(block.second_layer.store_pre_activity)

        # Check that tensor statistics are initialized
        self.assertIsNotNone(block.second_layer.tensor_m_prev)
        self.assertIsNotNone(block.second_layer.tensor_s_growth)

        # For hidden_features > 0, additional statistics should be initialized
        if hidden_features > 0:
            self.assertTrue(block.second_layer.store_input)
            self.assertIsNotNone(block.second_layer.cross_covariance)
            self.assertIsNotNone(block.second_layer.tensor_s)
            self.assertIsNotNone(block.second_layer.tensor_m)

    def test_with_custom_kwargs(self):
        """Test initialization with custom layer kwargs."""
        kwargs_layer = {"use_bias": False}
        kwargs_first_layer = {"use_bias": True}

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            kwargs_layer=kwargs_layer,
            kwargs_first_layer=kwargs_first_layer,
            device=self.device,
        )

        # First layer should use kwargs_first_layer
        self.assertTrue(block.first_layer.use_bias)
        # Second layer should use kwargs_layer
        self.assertFalse(block.second_layer.use_bias)

    def test_reset_computation(self):
        """Test reset of computation."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Initialize and then reset
        block.init_computation()
        block.reset_computation()

        # Check that storage flags are reset
        self.assertFalse(block.first_layer.store_input)
        self.assertFalse(block.second_layer.store_input)
        self.assertFalse(block.second_layer.store_pre_activity)

    def test_forward_backward_compatibility(self):
        """Test that forward and backward passes work correctly."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
            downsample=self.downsample,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        output = block(x)
        loss = torch.norm(output)

        # Should be able to backward without errors
        loss.backward()

        # Check that gradients were computed
        for param in block.parameters():
            self.assertIsNotNone(param.grad)

    def test_full_addition_loop_with_indicator_batch(self):
        """Test complete addition loop starting with 0 features using
        indicator batch data.

        We start with 0 hidden features and no downsampling, and train the
        block to learn the zero function. As the residual stream adds the
        identity, the optimal extension should learn the negative identity.
        """

        # Step 1: Create the block with no downsampling, no activation
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,  # Same dimensions for identity mapping
            hidden_features=0,  # Start with 0 features
            activation=torch.nn.Identity(),
            device=self.device,
            name="test_block",
        )

        # Step 2: Init the computation
        block.init_computation()

        # Step 3: Forward/backward with loss ||.||^2 / 2 using indicator batch
        x_batch = indicator_batch((self.in_features,), device=self.device)

        block.zero_grad()
        output = block(x_batch)

        # Loss: ||output||^2 / 2
        # to ensure that the gradient will be equal to x_batch
        loss = (output**2).sum() / 2
        loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(block.second_layer.pre_activity.grad)

        # Update computation
        block.update_computation()

        # Step 4: Compute updates (with max neurons = input features)
        block.compute_optimal_updates(maximum_added_neurons=self.in_features)

        # Verify updates were computed
        self.assertIsNotNone(block.first_layer.extended_output_layer)
        self.assertIsNotNone(block.second_layer.extended_input_layer)
        self.assertIsNotNone(block.eigenvalues_extension)
        self.assertIsNotNone(block.parameter_update_decrease)
        assert isinstance(block.parameter_update_decrease, torch.Tensor)
        self.assertAlmostEqual(block.parameter_update_decrease.item(), 0.0, places=5)
        assert isinstance(block.first_layer.extended_output_layer, torch.nn.Linear)
        assert isinstance(block.second_layer.extended_input_layer, torch.nn.Linear)
        assert isinstance(block.eigenvalues_extension, torch.Tensor)

        # Step 5: Reset computation
        block.reset_computation()

        # Verify reset
        self.assertFalse(block.first_layer.store_input)
        self.assertFalse(block.second_layer.store_pre_activity)

        # Step 6: Set scaling factor to 1
        block.scaling_factor = 1

        # Step 7: Check that the negative identity mapping was correctly learned
        # The extended forward should approximate the negative identity mapping
        with torch.no_grad():
            extended_output = block.extended_forward(x_batch)
            self.assertShapeEqual(extended_output, x_batch.shape)
            self.assertAllClose(extended_output, torch.zeros_like(x_batch), atol=1e-5)

        # Step 8: Check that `first_order_improvement` returns a value
        original_improvement = block.first_order_improvement
        self.assertIsInstance(original_improvement, torch.Tensor)
        self.assertTrue(
            original_improvement.item() >= 0
        )  # Should be positive improvement

        # Step 9: Sub select new neurons
        num_neurons_to_keep = min(1, block.eigenvalues_extension.shape[0])

        block.sub_select_optimal_added_parameters(num_neurons_to_keep)

        # Verify sub-selection
        self.assertEqual(block.eigenvalues_extension.shape[0], num_neurons_to_keep)
        self.assertEqual(
            block.first_layer.extended_output_layer.out_features,
            num_neurons_to_keep,
        )
        self.assertEqual(
            block.second_layer.extended_input_layer.in_features, num_neurons_to_keep
        )

        # Step 10: Check that `first_order_improvement` returns lower or equal value
        reduced_improvement = block.first_order_improvement
        self.assertIsInstance(reduced_improvement, torch.Tensor)
        self.assertLessEqual(
            reduced_improvement.item(),
            original_improvement.item(),
            "Reduced improvement should be <= original improvement",
        )

        # Step 11: Apply change
        # Store original weights for comparison
        expected_first_weight = (
            block.first_layer.extended_output_layer.weight.data.clone()
        )
        expected_second_weight = (
            block.second_layer.extended_input_layer.weight.data.clone()
        )

        original_first_out_features = block.first_layer.out_features
        original_second_in_features = block.second_layer.in_features

        block.apply_change()

        # Step 12: Check that the change was correctly done
        self.assertEqual(
            block.first_layer.out_features,
            original_first_out_features + num_neurons_to_keep,
        )
        self.assertEqual(
            block.second_layer.in_features,
            original_second_in_features + num_neurons_to_keep,
        )
        self.assertEqual(block.hidden_features, num_neurons_to_keep)  # Was 0 before

        # Verify weights were extended properly
        self.assertShapeEqual(
            block.first_layer.weight,
            (original_first_out_features + num_neurons_to_keep, self.in_features),
        )
        # Easy to check for equality as there was no neurons before
        self.assertAllClose(
            block.first_layer.weight,
            expected_first_weight,
        )
        self.assertShapeEqual(
            block.second_layer.weight,
            (self.in_features, original_second_in_features + num_neurons_to_keep),
        )
        # Easy to check for equality as there was no neurons before
        self.assertAllClose(
            block.second_layer.weight,
            expected_second_weight,
        )

        # Test that the extended layer behaves as expected
        with torch.no_grad():
            block(x_batch)

        # Step 13: Delete update
        block.delete_update()

        # Step 14: Check that the update was deleted
        deleted_objects = [
            block.second_layer.optimal_delta_layer,
            block.second_layer.extended_input_layer,
            block.second_layer.extended_output_layer,
            block.first_layer.optimal_delta_layer,
            block.first_layer.extended_output_layer,
            block.first_layer.extended_input_layer,
            block.eigenvalues_extension,
        ]
        for obj in deleted_objects:
            self.assertIsNone(obj)

    def test_full_addition_loop_with_features_identity_initialization(
        self, bias: bool = False
    ):
        """
        Test complete addition loop starting with features and identity initialization.

        We start with the following setup:
        x -Id-> x -0-> 0 + x
        and we aim to get zero output.
        The optimal solution is to change the parameters to:
        x -Id-> x -(-Id)-> -x + x = 0
        which leads to a zero bottleneck.
        We verify that the optimal update is indeed the negative identity and that any new
        neuron proposed has a very small eigenvalues (no improvement possible).
        """

        # Step 1: Create the block with features, no downsampling, no activation
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,  # Same dimensions for identity mapping
            hidden_features=self.in_features,  # Start with features
            activation=torch.nn.Identity(),
            device=self.device,
            name="test_block_with_features",
            kwargs_layer={"use_bias": bias},
        )

        # Step 2: Initialize first layer with identity and second layer with zeros
        with torch.no_grad():
            # First layer: identity transformation
            block.first_layer.weight.data = torch.eye(
                self.in_features, self.in_features, device=self.device
            )
            if block.first_layer.use_bias:
                block.first_layer.bias.data.zero_()

            # Second layer: zero transformation (to make the whole block
            # identity when added to residual)
            block.second_layer.weight.data.zero_()
            if block.second_layer.use_bias:
                block.second_layer.bias.data.zero_()

        # Verify the block performs identity mapping
        x_test = torch.randn(self.batch_size, self.in_features, device=self.device)
        with torch.no_grad():
            output_test = block(x_test)
            self.assertAllClose(output_test, x_test, atol=1e-6)

        # Step 3: Init the computation
        block.init_computation()

        # Verify initialization
        self.assertTrue(block.first_layer.store_input)
        self.assertTrue(block.second_layer.store_pre_activity)
        self.assertTrue(
            block.second_layer.store_input
        )  # Should be True for positive features

        # Step 4: Forward/backward with loss ||output||^2 / 2 using indicator batch
        x_batch = indicator_batch((self.in_features,), device=self.device)

        block.zero_grad()
        output = block(x_batch)

        # Loss: ||output||^2 / 2
        loss = (output**2).sum() / 2
        loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(block.second_layer.pre_activity.grad)

        block.update_computation()

        block.compute_optimal_updates(maximum_added_neurons=self.in_features)

        block.reset_computation()

        # Step 6: Check that the optimal delta layer is exactly the identity (negative)
        self.assertIsNotNone(block.second_layer.optimal_delta_layer)
        assert isinstance(block.second_layer.optimal_delta_layer, torch.nn.Linear)

        if not bias:
            expected_delta_weight = torch.eye(
                self.in_features, self.in_features, device=self.device
            )
            self.assertAllClose(
                block.second_layer.optimal_delta_layer.weight,
                expected_delta_weight,
                atol=1e-5,
                msg=(
                    "Optimal delta weight should be approximately the "
                    "identity matrix for already optimal layer"
                ),
            )

        # Step 7: Check that no new neurons are proposed
        # Since the block is already optimal (identity), eigenvalues should
        # be very small or zero
        self.assertIsNotNone(block.eigenvalues_extension)
        assert isinstance(block.eigenvalues_extension, torch.Tensor)

        # All eigenvalues should be very small (ideally zero) since no
        # improvement is possible
        self.assertTrue(
            torch.all(torch.abs(block.eigenvalues_extension) < 1e-3),
            f"Eigenvalues should be very small for optimal block, got "
            f"{block.eigenvalues_extension}",
        )

        self.assertIsNotNone(block.parameter_update_decrease)
        assert isinstance(block.parameter_update_decrease, torch.Tensor)
        self.assertAlmostEqual(
            block.parameter_update_decrease.item(),
            2 * loss.item() / x_batch.size(0),
            places=3,
        )

        # Step 9: Set scaling factor to 1
        block.scaling_factor = 1.0

        # Step 10: Check that first_order_improvement is correct
        improvement = block.first_order_improvement
        self.assertIsInstance(improvement, torch.Tensor)
        self.assertAlmostEqual(
            improvement.item(), 1, msg="First order improvement should be 1"
        )

        # Step 13: Delete update
        block.delete_update()

        # Step 14: Check that the update was deleted
        deleted_objects = [
            block.second_layer.optimal_delta_layer,
            block.second_layer.extended_input_layer,
            block.second_layer.extended_output_layer,
            block.first_layer.optimal_delta_layer,
            block.first_layer.extended_output_layer,
            block.first_layer.extended_input_layer,
            block.eigenvalues_extension,
        ]
        for obj in deleted_objects:
            self.assertIsNone(obj)

    def test_apply_change(self):
        """Test apply_change method with different scenarios.

        Tests three cases:
        1. Providing explicit extension_size parameter (overrides eigenvalues
           shape)
        2. Not providing extension_size but having eigenvalues_extension set
           (uses eigenvalues shape)
        3. Neither extension_size nor eigenvalues_extension (should raise
           assertion)
        """
        with self.subTest("Case 1: Explicit extension_size parameter"):
            # Setup: Create a block with some initial hidden features
            initial_hidden_features = 2
            block = LinearGrowingBlock(
                in_features=self.in_features,
                out_features=self.in_features,
                hidden_features=initial_hidden_features,
                device=self.device,
            )

            # Store original dimensions
            original_first_out = block.first_layer.out_features
            original_second_in = block.second_layer.in_features

            # Create second extension without bias as required by apply_change
            second_extension_no_bias = torch.nn.Linear(
                self.added_features,
                self.in_features,
                bias=False,
                device=self.device,
            )

            # Manually set up the extensions using setUp-defined layers
            block.first_layer.extended_output_layer = self.first_layer_extension
            block.second_layer.extended_input_layer = second_extension_no_bias
            block.second_layer.optimal_delta_layer = torch.nn.Linear(
                initial_hidden_features,
                self.in_features,
                device=self.device,
            )
            # Set scaling factor to avoid warnings
            block.scaling_factor = 1.0

            # Apply change with explicit size
            # This should add self.added_features (7) to the layers, and
            # update hidden_features by the same amount
            explicit_size = block.second_layer.extended_input_layer.in_features
            block.apply_change(extension_size=explicit_size)

            # Verify dimensions increased by the actual number of neurons
            # in the extension layers
            self.assertEqual(
                block.first_layer.out_features,
                original_first_out + self.added_features,
            )
            self.assertEqual(
                block.second_layer.in_features,
                original_second_in + self.added_features,
            )
            # But hidden_features is updated by extension_size
            self.assertEqual(
                block.hidden_features,
                initial_hidden_features + explicit_size,
            )

            # Clean up for next test
            block.delete_update()

        with self.subTest("Case 2: Using eigenvalues_extension"):
            # Reset to initial state for this test
            block = LinearGrowingBlock(
                in_features=self.in_features,
                out_features=self.in_features,
                hidden_features=initial_hidden_features,
                device=self.device,
            )
            original_first_out = block.first_layer.out_features
            original_second_in = block.second_layer.in_features

            # Create second extension without bias
            second_extension_no_bias = torch.nn.Linear(
                self.added_features,
                self.in_features,
                bias=False,
                device=self.device,
            )

            # Manually set up the extensions using setUp-defined layers
            block.first_layer.extended_output_layer = self.first_layer_extension
            block.second_layer.extended_input_layer = second_extension_no_bias
            block.second_layer.optimal_delta_layer = torch.nn.Linear(
                initial_hidden_features,
                self.in_features,
                device=self.device,
            )

            # Set eigenvalues_extension
            block.second_layer.eigenvalues_extension = torch.empty(
                (self.added_features,), device=self.device
            )
            # Set scaling factor
            block.scaling_factor = 1.0

            # Apply change without extension_size (should use eigenvalues)
            block.apply_change(extension_size=None)

            # Verify dimensions increased by number of neurons in extension
            self.assertEqual(
                block.first_layer.out_features,
                original_first_out + self.added_features,
            )
            self.assertEqual(
                block.second_layer.in_features,
                original_second_in + self.added_features,
            )
            # And hidden_features increased by eigenvalues shape
            self.assertEqual(
                block.hidden_features,
                initial_hidden_features + self.added_features,
            )

            # Clean up for next test
            block.delete_update()

        with self.subTest("Case 3: Neither extension_size nor eigenvalues"):
            # Reset to initial state for this test
            block = LinearGrowingBlock(
                in_features=self.in_features,
                out_features=self.in_features,
                hidden_features=initial_hidden_features,
                device=self.device,
            )

            # Create second extension without bias
            second_extension_no_bias = torch.nn.Linear(
                self.added_features,
                self.in_features,
                bias=False,
                device=self.device,
            )

            # Manually set up the extensions WITHOUT eigenvalues_extension
            block.first_layer.extended_output_layer = self.first_layer_extension
            block.second_layer.extended_input_layer = second_extension_no_bias
            block.second_layer.optimal_delta_layer = torch.nn.Linear(
                initial_hidden_features,
                self.in_features,
                device=self.device,
            )
            # Explicitly ensure eigenvalues_extension is None
            block.second_layer.eigenvalues_extension = None
            # Set scaling factor
            block.scaling_factor = 1.0

            # Should raise AssertionError
            with self.assertRaises(AssertionError):
                block.apply_change(extension_size=None)

    def test_create_layer_extensions(self):
        """
        Test `create_layer_extensions` method.

        Create some layer extensions and check that they have an
        effect when used in extended_forward.
        """
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Input tensor
        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # Extended forward without extensions should match normal forward
        block.second_layer.weight.data.fill_(0.0)
        block.second_layer.bias.data.fill_(0.0)

        output_no_ext = block.extended_forward(x)
        self.assertAllClose(output_no_ext, x)

        # Now use the extensions in extended forward
        block.scaling_factor = self.scaling_factor
        block.create_layer_extensions(self.added_features)

        output_with_ext = block.extended_forward(x)
        self.assertFalse(torch.allclose(output_with_ext, output_no_ext))

    def test_normalize_optimal_updates(self):
        """
        Test `normalize_optimal_updates` method.

        Create a block with extensions, normalize the updates,
        check that the std of the extension weights is close
        to the one of the original weights.
        """
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )
        block.first_layer.weight.data *= 2.0
        block.second_layer.weight.data *= 0.1

        # Create layer extensions
        block.create_layer_extensions(self.added_features)
        assert isinstance(block.first_layer.extended_output_layer, torch.nn.Linear)
        assert isinstance(block.second_layer.extended_input_layer, torch.nn.Linear)

        # Normalize optimal updates
        block.normalize_optimal_updates()

        # Check std of extension weights
        self.assertAlmostEqual(
            block.second_layer.extended_input_layer.weight.std().item(),
            block.second_layer.weight.std().item(),
            places=2,
            msg="Second layer extension weights std should match original weights std",
        )

    def test_weights_statistics(self):
        """Test that weights_statistics method runs and returns a dictionary."""

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        stats = block.weights_statistics()

        # Check that the result is a dictionary
        self.assertIsInstance(
            stats, dict, "weights_statistics should return a dictionary"
        )

    def test_optimal_delta_layer_property(self):
        """Test optimal_delta_layer getter and setter."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.in_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Test getter - initially should be None
        self.assertIsNone(block.optimal_delta_layer)

        # Test setter by calling the property setter directly
        # (nn.Module.__setattr__ intercepts normal assignments)
        delta_layer = torch.nn.Linear(self.hidden_features, self.in_features)
        block.optimal_delta_layer = delta_layer
        self.assertIs(block.optimal_delta_layer, delta_layer)
        self.assertIs(block.second_layer.optimal_delta_layer, delta_layer)

        # Test setter with None
        block.optimal_delta_layer = None
        self.assertIsNone(block.optimal_delta_layer)
