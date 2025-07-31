"""
Simple test to verify that the performance regression tests work correctly.
"""

import unittest
import torch
import torch.nn as nn

from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.utils.utils import set_device, global_device
from tests.torch_unittest import TorchTestCase


class TestSimplePerformance(TorchTestCase):
    """Simple performance test to verify the setup."""
    
    def test_basic_forward_pass(self):
        """Test that a basic forward pass works."""
        # Set device for testing
        set_device("cpu")
        
        # Create synthetic data for testing
        x = torch.randn(2, 3, 32, 32)
        
        # Initialize model
        model = GrowingResidualMLP(
            in_features=(3, 32, 32),
            out_features=10,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU(),
            device=global_device()
        )
        
        # Run forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()