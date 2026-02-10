import torch

from gromo.containers.growing_block import RestrictedConv2dGrowingBlock
from gromo.utils.utils import global_device


try:
    from tests.torch_unittest import GrowableIdentity, SizedIdentity, TorchTestCase
except ImportError:
    from torch_unittest import GrowableIdentity, SizedIdentity, TorchTestCase


class TestConv2dGrowingBlock(TorchTestCase):
    """Test RestrictedConv2dGrowingBlock functionality."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = global_device()
        self.batch_size = 2
        self.in_channels = 3
        self.out_channels = 3
        self.hidden_channels = 5
        self.input_height = 7
        self.input_width = 11

    def test_init(self):
        """Test initialization of RestrictedConv2dGrowingBlock."""
        with self.assertRaises(ValueError):
            # kernel_size must be specified
            RestrictedConv2dGrowingBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=None,
                hidden_channels=0,
                device=self.device,
            )

        # Init with kwargs dictionaries
        with self.assertWarns(UserWarning):
            # Initializing zero-element tensors is a no-op
            block = RestrictedConv2dGrowingBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                hidden_channels=0,
                device=self.device,
                kwargs_layer={"kernel_size": 3, "padding": 1},
            )
        self.assertEqual(block.first_layer.kernel_size, (3, 3))
        self.assertEqual(block.first_layer.padding, (1, 1))
        self.assertEqual(block.second_layer.kernel_size, (3, 3))
        self.assertEqual(block.second_layer.padding, (1, 1))

        block = RestrictedConv2dGrowingBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            hidden_channels=self.hidden_channels,
            device=self.device,
            name="test_conv_block",
        )

        # Check basic properties
        self.assertEqual(block.in_features, self.in_channels)
        self.assertEqual(block.out_features, self.out_channels)
        self.assertEqual(block.hidden_neurons, self.hidden_channels)
        self.assertEqual(block.name, "test_conv_block")

        # Check layer configurations
        self.assertEqual(block.first_layer.in_channels, self.in_channels)
        self.assertEqual(block.first_layer.out_channels, self.hidden_channels)
        self.assertEqual(block.second_layer.in_channels, self.hidden_channels)
        self.assertEqual(block.second_layer.out_channels, self.out_channels)

        # Check that layers are connected
        self.assertIs(block.second_layer.previous_module, block.first_layer)

    def test_forward_with_sized_activation(self):
        """Test forward pass with sized activation function."""
        # Create block with SizedIdentity activation
        sized_activation = SizedIdentity(self.hidden_channels)

        with self.assertWarns(UserWarning):
            # Warns because no extended_mid_activation is provided
            # and mid_activation is fixed size
            block = RestrictedConv2dGrowingBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                hidden_channels=self.hidden_channels,
                mid_activation=sized_activation,
                device=self.device,
                kwargs_layer={"padding": 1},
            )

        x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.input_height,
            self.input_width,
            device=self.device,
        )

        # Forward pass should work without errors
        output = block(x)

        # Check output shape
        self.assertShapeEqual(
            output,
            (self.batch_size, self.out_channels, self.input_height, self.input_width),
        )

    def test_extended_forward_with_growable_activation_and_apply_change(self):
        """Test extended forward with growable activation, then apply change and test forward."""
        # Create block with GrowableIdentity activation
        growable_activation = GrowableIdentity(self.hidden_channels)

        block = RestrictedConv2dGrowingBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            hidden_channels=self.hidden_channels,
            mid_activation=growable_activation,
            extended_mid_activation=torch.nn.Identity(),
            device=self.device,
            kwargs_layer={"padding": 1},
        )

        x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.input_height,
            self.input_width,
            device=self.device,
        )

        # Initialize computation for growing
        block.init_computation()

        # Forward pass to gather statistics
        block.zero_grad()
        output = block(x)
        loss = torch.norm(output)
        loss.backward()

        # Update computation
        block.update_computation()

        # Compute optimal updates
        added_neurons = 2
        block.compute_optimal_updates(maximum_added_neurons=added_neurons)

        # Reset computation
        block.reset_computation()

        # Set scaling factor
        block.scaling_factor = 1.0

        # Test extended forward before applying changes
        with torch.no_grad():
            extended_output = block.extended_forward(x)
            self.assertShapeEqual(
                extended_output,
                (self.batch_size, self.out_channels, self.input_height, self.input_width),
            )

        # Apply changes
        block.apply_change()

        # Verify the block grew
        self.assertEqual(block.hidden_neurons, self.hidden_channels + added_neurons)
        self.assertEqual(
            block.first_layer.out_channels, self.hidden_channels + added_neurons
        )
        self.assertEqual(
            block.second_layer.in_channels, self.hidden_channels + added_neurons
        )

        # Test forward pass after applying changes
        with torch.no_grad():
            final_output = block(x)
            self.assertShapeEqual(
                final_output,
                (self.batch_size, self.out_channels, self.input_height, self.input_width),
            )

        # Clean up
        block.delete_update()


if __name__ == "__main__":
    from unittest import main

    main()
