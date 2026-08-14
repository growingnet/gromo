"""Unit tests for GrowingDropPath."""

import unittest
from unittest.mock import patch

import torch

from gromo.modules.growing_drop_path import DropPath, GrowingDropPath, drop_path
from gromo.modules.growing_module import SupportsExtendedForward


class TestGrowingDropPath(unittest.TestCase):
    """Test cases for stochastic depth helpers."""

    def test_drop_path_validation_and_training_modes(self):
        x = torch.ones(2, 3, 4)

        self.assertTrue(torch.equal(drop_path(x, drop_prob=0.0, training=True), x))
        self.assertTrue(torch.equal(drop_path(x, drop_prob=0.5, training=False), x))

        with patch(
            "gromo.modules.growing_drop_path.torch.rand",
            return_value=torch.ones((2, 1, 1)),
        ):
            dropped = drop_path(x, drop_prob=0.5, training=True)
            torch.testing.assert_close(dropped, torch.zeros_like(x))

        with self.assertRaisesRegex(ValueError, r"`drop_prob` must lie in \[0, 1\]"):
            drop_path(x, drop_prob=1.5, training=True)

    def test_extended_forward(self):
        x = torch.ones(2, 3, 4)
        x_ext = torch.randn(2, 3, 4)
        drop_module = DropPath(0.5)
        drop_module.train()

        self.assertIs(DropPath, GrowingDropPath)
        self.assertIsInstance(drop_module, GrowingDropPath)
        self.assertIsInstance(drop_module, SupportsExtendedForward)

        with patch(
            "gromo.modules.growing_drop_path.torch.rand",
            return_value=torch.ones((2, 1, 1)),
        ):
            dropped, forwarded_ext = drop_module.extended_forward(x, x_ext)
            torch.testing.assert_close(dropped, torch.zeros_like(x))
            self.assertIs(forwarded_ext, x_ext)

        dropped_none, forwarded_ext = drop_module.extended_forward(None, x_ext)
        self.assertIsNone(dropped_none)
        self.assertIs(forwarded_ext, x_ext)
