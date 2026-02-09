import unittest

import torch
import torch.nn as nn

from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.utils.utils import compute_tensor_stats
from tests.test_growing_container import create_synthetic_data, gather_statistics
from tests.torch_unittest import TorchTestCase


class TestGrowingResidualMLP(TorchTestCase):
    def setUp(self):
        # Create synthetic data
        self.in_features = (3, 32, 32)
        self.out_features = 10
        self.num_samples = 20
        self.batch_size = 4
        self.dataloader = create_synthetic_data(
            self.num_samples, self.in_features, (self.out_features,), self.batch_size
        )

        # Create a simple MLP model
        self.num_features = 16
        self.hidden_features = 8
        self.num_blocks = 2

        self.model = GrowingResidualMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            num_features=self.num_features,
            hidden_features=self.hidden_features,
            num_blocks=self.num_blocks,
            activation=nn.ReLU(),
            device=torch.device("cpu"),
        )

        # Create a loss
        self.loss = nn.MSELoss()

        # Compute the optimal updates
        gather_statistics(self.dataloader, self.model, self.loss)
        with self.assertMaybeWarns(
            UserWarning,
            "Using the pseudo-inverse for the computation of the optimal delta",
        ):
            self.model.compute_optimal_updates()

    def test_init(self):
        l1 = GrowingResidualMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            num_features=self.num_features,
            hidden_features=self.hidden_features,
            num_blocks=self.num_blocks,
            activation=nn.ReLU(),
            device=torch.device("cpu"),
        )

        self.assertIsInstance(l1, GrowingResidualMLP)
        self.assertIsInstance(l1, torch.nn.Module)

    def test_forward(self):
        x = torch.randn(1, *self.in_features)
        y = self.model.forward(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_extended_forward(self):
        x = torch.randn(1, *self.in_features)
        y = self.model.extended_forward(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_set_growing_layers(self):
        self.model.set_growing_layers()
        self.assertEqual(len(self.model._growing_layers), self.num_blocks)

    def test_tensor_statistics(self):
        tensor = torch.randn(10)
        stats = compute_tensor_stats(tensor)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)

    def test_weights_statistics(self):
        stats = self.model.weights_statistics()
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)

    def test_update_information(self):
        info = self.model.update_information()
        self.assertIsInstance(info, dict)
        self.assertGreater(len(info), 0)

    def test_select_update(self):
        layer_index = 0
        selected_index = self.model.select_update(layer_index=layer_index)
        self.assertEqual(selected_index, layer_index)


if __name__ == "__main__":
    unittest.main()
