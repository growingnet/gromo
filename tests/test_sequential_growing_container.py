import torch

from gromo.containers.sequential_growing_container import SequentialGrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


class DummySequentialGrowingContainer(SequentialGrowingContainer):
    """
    Dummy implementation of SequentialGrowingContainer for testing purposes.
    Uses two simple linear layers in sequence.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    hidden_features : int
        Number of hidden features between the two layers
    device : torch.device | str | None
        Device to use for the layers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 4,
        device: torch.device | str | None = None,
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )

        # Create two linear growing modules
        self.layer1 = LinearGrowingModule(
            in_features=in_features,
            out_features=hidden_features,
            name="layer1",
            device=self.device,
        )
        self.layer2 = LinearGrowingModule(
            in_features=hidden_features,
            out_features=hidden_features,
            name="layer2",
            previous_module=self.layer1,
            device=self.device,
        )

        # Set up the growable layers list
        self._growable_layers = [self.layer1, self.layer2]

        # Initialize with the first scheduling method
        self.set_growing_layers(scheduling_method="all")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through both layers."""
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def number_of_neurons_to_add(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, growth_amount: int = 1
    ) -> int:
        """
        Implementation of the abstract method.
        Returns the number of neurons to add based on growth_amount.
        """
        return growth_amount

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """Property required by parent class."""
        return torch.tensor(0.0)


class TestSequentialGrowingContainer(TorchTestCase):
    """Test SequentialGrowingContainer implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 3
        self.out_features = 2
        self.hidden_features = 4
        self.device = torch.device("cpu")

    def test_set_growing_layers_all(self):
        """Test set_growing_layers with 'all' scheduling method."""
        container = DummySequentialGrowingContainer(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Test 'all' scheduling method
        container.set_growing_layers(scheduling_method="all")

        # All layers should be in _growing_layers
        self.assertEqual(
            len(container._growing_layers),
            2,
            "All layers should be in _growing_layers",
        )
        self.assertIn(container.layer1, container._growing_layers)
        self.assertIn(container.layer2, container._growing_layers)

    def test_set_growing_layers_sequential(self):
        """Test set_growing_layers with 'sequential' scheduling method."""
        container = DummySequentialGrowingContainer(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Start with layer_to_grow_index = -1
        self.assertEqual(container.layer_to_grow_index, -1)

        # First call to sequential should move to index 0
        container.set_growing_layers(scheduling_method="sequential")
        self.assertEqual(container.layer_to_grow_index, 0)
        self.assertEqual(len(container._growing_layers), 1)
        self.assertIs(container._growing_layers[0], container.layer1)

        # Second call should move to index 1
        container.set_growing_layers(scheduling_method="sequential")
        self.assertEqual(container.layer_to_grow_index, 1)
        self.assertEqual(len(container._growing_layers), 1)
        self.assertIs(container._growing_layers[0], container.layer2)

        # Third call should wrap around to index 0
        container.set_growing_layers(scheduling_method="sequential")
        self.assertEqual(container.layer_to_grow_index, 0)
        self.assertEqual(len(container._growing_layers), 1)
        self.assertIs(container._growing_layers[0], container.layer1)

    def test_set_growing_layers_with_index(self):
        """Test set_growing_layers with explicit index parameter."""
        container = DummySequentialGrowingContainer(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Set to index 0
        container.set_growing_layers(index=0)
        self.assertEqual(container.layer_to_grow_index, 0)
        self.assertEqual(len(container._growing_layers), 1)
        self.assertIs(container._growing_layers[0], container.layer1)

        # Set to index 1
        container.set_growing_layers(index=1)
        self.assertEqual(container.layer_to_grow_index, 1)
        self.assertEqual(len(container._growing_layers), 1)
        self.assertIs(container._growing_layers[0], container.layer2)

        # Test invalid indices
        with self.assertRaises(IndexError):
            container.set_growing_layers(index=10)

        with self.assertRaises(IndexError):
            container.set_growing_layers(index=-1)

    def test_set_growing_layers_invalid_method(self):
        """Test set_growing_layers with invalid scheduling method."""
        container = DummySequentialGrowingContainer(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        with self.assertRaises(ValueError):
            container.set_growing_layers(scheduling_method="invalid_method")

    def test_number_of_neurons_to_add(self):
        """Test that number_of_neurons_to_add returns an int."""
        container = DummySequentialGrowingContainer(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Test with default argument
        result = container.number_of_neurons_to_add()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 1)

        # Test with custom argument
        result = container.number_of_neurons_to_add(growth_amount=5)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 5)

    def test_update_information(self):
        """
        Test update_information returns expected dictionary structure with
        at least two growing layers.
        """
        container = DummySequentialGrowingContainer(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_features=self.hidden_features,
            device=self.device,
        )

        # Set up layer2 (which has layer1 as previous_module) with necessary attributes
        # parameter_update_decrease and eigenvalues_extension are needed
        # because first_order_improvement is computed from them
        container.layer2.parameter_update_decrease = torch.tensor(0.3)
        container.layer2.eigenvalues_extension = torch.tensor([0.6, 0.5])
        container.layer1.parameter_update_decrease = torch.tensor(0.5)

        # Set growing layers to only include layer2 (which has proper previous_module)
        container.set_growing_layers(index=1)

        # Get the information
        info = container.update_information()

        # Verify the structure
        self.assertIsInstance(info, dict)

        # Now set both layers to growing and test again
        container.set_growing_layers(scheduling_method="all")
        info = container.update_information()
        self.assertIsInstance(info, dict)
