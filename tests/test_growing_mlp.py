import unittest

import torch
import torch.nn as nn

from gromo.containers.growing_mlp import GrowingMLP
from tests.test_growing_container import create_synthetic_data, gather_statistics


class TestGrowingMLP(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        self.in_features = 2
        self.out_features = 1
        self.num_samples = 20
        self.batch_size = 4
        self.dataloader = create_synthetic_data(
            self.num_samples, (self.in_features,), (self.out_features,), self.batch_size
        )

        # Create a simple MLP model
        self.hidden_size = 4
        self.number_hidden_layers = 2
        self.model = GrowingMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=self.number_hidden_layers,
            activation=nn.ReLU(),
        )

        # Create a loss
        self.loss = nn.MSELoss()

        # Compute the optimal updates
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()

    def test_forward(self):
        x = torch.randn(1, self.in_features)
        y = self.model.forward(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_extended_forward(self):
        x = torch.randn(1, self.in_features)
        y = self.model.extended_forward(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_tensor_statistics(self):
        tensor = torch.randn(10)
        stats = self.model.tensor_statistics(tensor)
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

    def test_normalise(self):
        y_pred_list = [self.model(x) for x, _ in self.dataloader]

        self.model.normalise()
        y_pred_normalised_list = [self.model(x) for x, _ in self.dataloader]
        for y_pred, y_pred_normalised in zip(y_pred_list, y_pred_normalised_list):
            self.assertTrue(torch.allclose(y_pred, y_pred_normalised))

    def test_normalisation_factor(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        factors = self.model.normalisation_factor(values)
        self.assertEqual(factors.shape, values.shape)


if __name__ == "__main__":
    unittest.main()
