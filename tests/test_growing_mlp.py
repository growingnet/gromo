"""
Test suite for the GrowingMLP container module.

This module contains comprehensive tests for the GrowingMLP and Perceptron classes,
ensuring proper functionality of growing multi-layer perceptron networks including
forward passes, statistics computation, normalization, and edge cases.
"""

import io
import unittest
from contextlib import redirect_stdout

import torch
import torch.nn as nn

from gromo.containers.growing_mlp import GrowingMLP, Perceptron
from tests.test_growing_container import create_synthetic_data, gather_statistics


class TestGrowingMLP(unittest.TestCase):
    """
    Test class for GrowingMLP functionality.

    This class tests all major functionality of the GrowingMLP container including:
    - Forward and extended forward passes
    - Statistics computation (tensor, weights, updates)
    - Normalization methods
    - Edge cases and error handling
    - Different input configurations (flattened vs non-flattened)
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.

        Creates synthetic data, initializes a GrowingMLP model, and prepares
        statistics for testing the growing functionality.
        """
        # Create synthetic data for testing
        self.in_features = 2
        self.out_features = 1
        self.num_samples = 20
        self.batch_size = 4
        self.dataloader = create_synthetic_data(
            self.num_samples, (self.in_features,), (self.out_features,), self.batch_size
        )

        # Create a simple MLP model for testing
        self.hidden_size = 4
        self.number_hidden_layers = 2
        self.model = GrowingMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=self.number_hidden_layers,
            activation=nn.ReLU(),
        )

        # Create a loss function for testing
        self.loss = nn.MSELoss()

        # Compute the optimal updates for growing functionality
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()

    def test_forward(self):
        """Test the forward pass of the GrowingMLP model."""
        x = torch.randn(1, self.in_features)
        y = self.model.forward(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_extended_forward(self):
        """Test the extended forward pass with current modifications."""
        x = torch.randn(1, self.in_features)
        y = self.model.extended_forward(x)
        self.assertEqual(y.shape, (1, self.out_features))

    def test_tensor_statistics(self):
        """Test computation of tensor statistics (min, max, mean, std)."""
        tensor = torch.randn(10)
        stats = self.model.tensor_statistics(tensor)

        # Check that all required statistics are present
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)

        # Test edge case with single element tensor (std should be -1)
        single_tensor = torch.tensor([5.0])
        single_stats = self.model.tensor_statistics(single_tensor)
        self.assertEqual(single_stats["std"], -1)

    def test_weights_statistics(self):
        """Test computation of weight statistics for all layers."""
        stats = self.model.weights_statistics()
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)

        # Check that each layer has the required statistics
        for layer_idx, layer_stats in stats.items():
            self.assertIn("weight", layer_stats)
            self.assertIn("input_shape", layer_stats)
            self.assertIn("output_shape", layer_stats)
            # Note: bias might or might not be present depending on use_bias

    def test_weights_statistics_without_bias(self):
        """Test weights statistics for model without bias terms."""
        # Create model without bias to test the bias=None branch
        model_no_bias = GrowingMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=self.number_hidden_layers,
            activation=nn.ReLU(),
            use_bias=False,  # This will test the bias=None branch
        )

        stats = model_no_bias.weights_statistics()
        self.assertIsInstance(stats, dict)

        # Check that bias is not present in statistics
        for layer_idx, layer_stats in stats.items():
            self.assertNotIn("bias", layer_stats)

    def test_update_information(self):
        """Test retrieval of update information for growing layers."""
        info = self.model.update_information()
        self.assertIsInstance(info, dict)
        self.assertGreater(len(info), 0)

        # Check that each growing layer has the required information
        for layer_idx, layer_info in info.items():
            self.assertIn("update_value", layer_info)
            self.assertIn("parameter_improvement", layer_info)
            self.assertIn("eigenvalues_extension", layer_info)

    def test_normalise(self):
        """Test the normalization method for layer weights."""
        # Get predictions before normalization
        y_pred_list = [self.model(x) for x, _ in self.dataloader]

        # Normalize the model
        self.model.normalise()

        # Get predictions after normalization
        y_pred_normalised_list = [self.model(x) for x, _ in self.dataloader]

        # Predictions should remain the same after normalization
        for y_pred, y_pred_normalised in zip(y_pred_list, y_pred_normalised_list):
            self.assertTrue(torch.allclose(y_pred, y_pred_normalised))

    def test_normalise_verbose(self):
        """Test the normalization method with verbose output."""
        # Test the verbose branch that prints normalization factors
        # Capture stdout to test verbose output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            self.model.normalise(verbose=True)

        # Check that something was printed
        output = captured_output.getvalue()
        self.assertIn("Normalisation:", output)

    def test_normalise_without_bias(self):
        """Test normalization for model without bias terms."""
        # Create model without bias to test the bias=None branch in normalise
        model_no_bias = GrowingMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=self.number_hidden_layers,
            activation=nn.ReLU(),
            use_bias=False,
        )

        # This should not raise an error even without bias terms
        model_no_bias.normalise()

    def test_normalisation_factor(self):
        """Test computation of normalization factors."""
        values = torch.tensor([1.0, 2.0, 3.0])
        factors = self.model.normalisation_factor(values)
        self.assertEqual(factors.shape, values.shape)

        # Test mathematical property: factors should normalize the geometric mean
        geometric_mean = values.prod().pow(1 / values.numel())
        normalized_values = values * factors
        self.assertTrue(
            torch.allclose(normalized_values, geometric_mean.repeat(values.shape))
        )

    def test_invalid_in_features_type(self):
        """Test error handling for invalid in_features type."""
        # Test the TypeError branch for invalid in_features type
        with self.assertRaises(TypeError) as context:
            # Use a type that's not int, list, or tuple to trigger the error
            invalid_features = set([1, 2, 3])  # set is not supported
            GrowingMLP(
                in_features=invalid_features,  # type: ignore  # This should trigger TypeError
                out_features=self.out_features,
                hidden_size=self.hidden_size,
                number_hidden_layers=self.number_hidden_layers,
            )

        self.assertIn(
            "Expected in_features to be int, list, or tuple", str(context.exception)
        )

    def test_getitem_method(self):
        """Test the __getitem__ method for accessing layers."""
        # Test valid indexing
        first_layer = self.model[0]
        self.assertIsNotNone(first_layer)

        last_layer = self.model[len(self.model.layers) - 1]
        self.assertIsNotNone(last_layer)

    def test_getitem_invalid_index(self):
        """Test __getitem__ method with invalid indices."""
        # Test negative index (should raise AssertionError)
        with self.assertRaises(AssertionError):
            _ = self.model[-1]

        # Test index too large (should raise AssertionError)
        with self.assertRaises(AssertionError):
            _ = self.model[len(self.model.layers)]

    def test_str_and_repr(self):
        """Test string representation methods."""
        model_str = str(self.model)
        model_repr = repr(self.model)

        self.assertIsInstance(model_str, str)
        self.assertIsInstance(model_repr, str)
        self.assertEqual(model_str, model_repr)  # __repr__ calls __str__

    def test_without_flatten(self):
        """Test the model without flattening the input."""
        # Test the model without flattening the input
        in_features = [10, 4]  # Use list instead of tuple to avoid type issues
        model = GrowingMLP(
            in_features=in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=self.number_hidden_layers,
            activation=nn.ReLU(),
            flatten=False,
        )
        x = torch.randn(1, *in_features)
        y = model.forward(x)
        self.assertEqual(y.shape, (1, *in_features[:-1], self.out_features))

    def test_tuple_in_features_with_flatten(self):
        """Test model with tuple in_features and flatten=True."""
        in_features = [2, 3, 4]  # Use list for compatibility
        model = GrowingMLP(
            in_features=in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=self.number_hidden_layers,
            activation=nn.ReLU(),
            flatten=True,  # This will flatten the input
        )

        # Test that the model correctly computes num_features as product
        expected_num_features = 2 * 3 * 4
        self.assertEqual(model.num_features, expected_num_features)

        # Test forward pass
        x = torch.randn(1, *in_features)
        y = model.forward(x)
        self.assertEqual(y.shape, (1, self.out_features))


class TestPerceptron(unittest.TestCase):
    """
    Test class for Perceptron functionality.

    Tests the Perceptron class which is a specialized single-hidden-layer
    version of GrowingMLP.
    """

    def setUp(self):
        """Set up test fixtures for Perceptron tests."""
        self.in_features = 3
        self.hidden_features = 5
        self.out_features = 2

    def test_perceptron_initialization(self):
        """Test Perceptron class initialization and basic functionality."""
        # Create a Perceptron instance
        perceptron = Perceptron(
            in_features=self.in_features,
            hidden_feature=self.hidden_features,
            out_features=self.out_features,
            activation=nn.Sigmoid(),
            use_bias=True,
            flatten=True,
        )

        # Test that it's properly initialized as a GrowingMLP
        self.assertIsInstance(perceptron, GrowingMLP)
        self.assertEqual(len(perceptron.layers), 2)  # Input->Hidden, Hidden->Output

        # Test forward pass
        x = torch.randn(1, self.in_features)
        y = perceptron.forward(x)
        self.assertEqual(y.shape, (1, self.out_features))


if __name__ == "__main__":
    unittest.main()
