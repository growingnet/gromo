"""
Unit tests for GrowingDropout2d
"""

import unittest

import torch

from gromo.modules.growing_dropout import GrowingDropout2d
from gromo.modules.growing_module import SupportsExtendedForward


class TestGrowingDropout2d(unittest.TestCase):
    """Test cases for GrowingDropout2d class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_features = 32
        self.batch_size = 8
        self.height = 16
        self.width = 16

    def test_initialization(self):
        """Test proper initialization of GrowingDropout2d."""
        dropout = GrowingDropout2d(
            dropout_rate=0.5,
            name="test_dropout",
        )

        # Check attributes
        self.assertEqual(dropout.dropout_rate, 0.5)
        self.assertEqual(dropout.name, "test_dropout")
        self.assertFalse(dropout.inplace)

    def test_default_initialization(self):
        """Test default initialization of GrowingDropout2d."""
        dropout = GrowingDropout2d()

        # Check default attributes
        self.assertEqual(dropout.dropout_rate, 0.0)
        self.assertEqual(dropout.name, "growing_dropout")
        self.assertFalse(dropout.inplace)

    def test_forward_pass(self):
        """Test forward pass with original features."""
        dropout = GrowingDropout2d(
            dropout_rate=0.5,
        )

        # Create test input
        x = torch.randn(
            self.batch_size,
            self.initial_features,
            self.height,
            self.width,
            device=self.device,
        )

        # Forward pass
        output = dropout(x)

        # Check output shape
        self.assertEqual(output.shape, x.shape)

    def test_extended_forward(self):
        """Test extended_forward applies dropout to x and passes x_ext unchanged."""
        extension_size = 8
        dropout = GrowingDropout2d(
            dropout_rate=0.5,
        )

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

        self.assertIsInstance(dropout, SupportsExtendedForward)

        processed_x, processed_x_ext = dropout.extended_forward(x, x_ext)

        self.assertEqual(processed_x.shape, x.shape)
        self.assertEqual(processed_x_ext.shape, x_ext.shape)
        torch.testing.assert_close(processed_x, dropout(x))
        torch.testing.assert_close(processed_x_ext, x_ext)

    def test_extra_repr(self):
        """Test extra_repr method."""
        dropout = GrowingDropout2d(
            name="test_dropout",
        )

        repr_str = dropout.extra_repr()
        self.assertIsInstance(repr_str, str, "extra_repr should return a string")

    def test_zero_dropout(self):
        """Test that setting dropout_rate to 0 results in no dropout."""
        dropout = GrowingDropout2d(
            dropout_rate=0.0,
        )

        x = torch.randn(
            self.batch_size,
            self.initial_features,
            self.height,
            self.width,
            device=self.device,
        )

        output = dropout(x)
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, x)

    def test_full_dropout(self):
        """Test that setting dropout_rate to 1 results in all zeros."""
        dropout = GrowingDropout2d(
            dropout_rate=1.0,
        )

        x = torch.randn(
            self.batch_size,
            self.initial_features,
            self.height,
            self.width,
            device=self.device,
        )

        output = dropout(x)
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(output, torch.zeros_like(x))
