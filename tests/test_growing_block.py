import torch
from matplotlib.pylab import block

from gromo.containers.growing_block import GrowingBlock, LinearGrowingBlock
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase, indicator_batch


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


class TestLinearGrowingBlock(TorchTestCase):
    """Test LinearGrowingBlock functionality."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = global_device()
        self.batch_size = 4
        self.in_features = 6
        self.out_features = 8
        self.hidden_features = 5
        self.downsample = torch.nn.Linear(
            self.in_features, self.out_features, device=self.device
        )

    def test_init_with_zero_features(self):
        """Test initialization with 0 hidden features."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=0,
            device=self.device,
            name="zero_block",
        )

        # Check basic properties
        self.assertEqual(block.in_features, self.in_features)
        self.assertEqual(block.out_features, self.out_features)
        self.assertEqual(block.hidden_features, 0)
        self.assertEqual(block.name, "zero_block")

        # Check layer configurations
        self.assertEqual(block.first_layer.in_features, self.in_features)
        self.assertEqual(block.first_layer.out_features, 0)  # hidden_features = 0
        self.assertEqual(block.second_layer.in_features, 0)
        self.assertEqual(block.second_layer.out_features, self.out_features)

        # Check that layers are connected
        self.assertIs(block.second_layer.previous_module, block.first_layer)

    def test_init_with_positive_features(self):
        """Test initialization with >0 hidden features."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
            name="positive_block",
        )

        # Check basic properties
        self.assertEqual(block.in_features, self.in_features)
        self.assertEqual(block.out_features, self.out_features)
        self.assertEqual(block.hidden_features, self.hidden_features)

        # Check layer configurations
        self.assertEqual(block.first_layer.in_features, self.in_features)
        self.assertEqual(block.first_layer.out_features, self.hidden_features)
        self.assertEqual(block.second_layer.in_features, self.hidden_features)
        self.assertEqual(block.second_layer.out_features, self.out_features)

    def test_init_with_custom_activations(self):
        """Test initialization with custom activation functions."""
        activation = torch.nn.ReLU()
        pre_activation = torch.nn.Sigmoid()
        mid_activation = torch.nn.Tanh()

        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
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
            out_features=self.out_features,
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
        """Test input and pre-activity storage with 0 hidden features and no downsample."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
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
        """Test input and pre-activity storage with >0 hidden features and no downsample."""
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

    def test_scaling_factor_property(self):
        """Test scaling factor property getter and setter."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
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

    def test_init_computation(self):
        """Test initialization of computation."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
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
        if block.hidden_features > 0:
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
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            kwargs_layer=kwargs_layer,
            kwargs_first_layer=kwargs_first_layer,
            device=self.device,
        )

        # First layer should use kwargs_first_layer
        self.assertTrue(block.first_layer.use_bias)
        # Second layer should use kwargs_layer
        self.assertFalse(block.second_layer.use_bias)

    def test_extended_forward_zero_features(self):
        """Test extended_forward with zero hidden features."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=0,
            device=self.device,
        )

        x = torch.randn(self.batch_size, self.in_features, device=self.device)

        # With zero features and no extended_output_layer, should return identity
        output = block.extended_forward(x)
        expected_output = block.downsample(x)

        self.assertAllClose(output, expected_output)

    def test_reset_computation(self):
        """Test reset of computation."""
        block = LinearGrowingBlock(
            in_features=self.in_features,
            out_features=self.out_features,
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
        x.requires_grad_(True)

        output = block(x)
        loss = torch.norm(output)

        # Should be able to backward without errors
        loss.backward()

        # Check that gradients were computed
        self.assertIsNotNone(x.grad)
        for param in block.parameters():
            self.assertIsNotNone(param.grad)
        self.assertIsNotNone(x.grad)
        for param in block.parameters():
            self.assertIsNotNone(param.grad)
