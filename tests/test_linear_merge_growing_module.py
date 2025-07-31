from copy import deepcopy
from unittest import main

import torch

from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize


class TestLinearMergeGrowingModule(TorchTestCase):
    """
    Test suite for the LinearMergeGrowingModule class.
    
    This class contains unit tests that verify the functionality of the LinearMergeGrowingModule,
    which is responsible for merging multiple input streams in a growing neural network.
    
    Key aspects tested:
    - Module initialization and basic properties
    - Input and activity storage behavior
    - Scaling factor updates and propagation
    - Module connectivity and dimension management
    - Error handling for invalid configurations
    
    The test class sets up common test fixtures and provides helper methods for repeated
    test patterns. Each test method focuses on a specific aspect of the module's functionality.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method is run.
        
        Initializes:
        - Test parameters (in_features, out_features)
        - Common module configurations with and without bias
        - Test input data
        """
        torch.manual_seed(0)
        self.demo_modules = dict()
        for bias in (True, False):
            demo_merge = LinearMergeGrowingModule(
                in_features=3, name="merge", device=global_device()
            )
            demo_merge_prev = LinearGrowingModule(
                5,
                3,
                use_bias=bias,
                name="merge_prev",
                device=global_device(),
                next_module=demo_merge,
            )
            demo_merge_next = LinearGrowingModule(
                3,
                7,
                use_bias=bias,
                name="merge_next",
                device=global_device(),
                previous_module=demo_merge,
            )
            demo_merge.set_previous_modules([demo_merge_prev])
            demo_merge.set_next_modules([demo_merge_next])
            self.demo_modules[bias] = {
                "add": demo_merge,
                "prev": demo_merge_prev,
                "next": demo_merge_next,
                "seq": torch.nn.Sequential(demo_merge_prev, demo_merge, demo_merge_next),
            }
        self.input_x = torch.randn((11, 5), device=global_device())

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_init(self, bias: bool):
        """Test LinearMergeGrowingModule initialization.
        
        Verifies that the module is properly initialized and can be instantiated
        with the expected configuration.
        
        Args:
            bias: Whether to test with bias enabled or disabled
        """
        self.assertIsInstance(
            self.demo_modules[bias]["add"], 
            LinearMergeGrowingModule,
            "Module should be an instance of LinearMergeGrowingModule"
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_input_storage(self, bias: bool):
        """Test input storage functionality in the merge module.
        
        Verifies that:
        1. The merge module correctly forwards input storage requests
        2. Input tensors are properly stored during the forward pass
        3. The stored input has the expected shape
        
        Args:
            bias: Whether to test with bias enabled or disabled
        """
        demo_layers = self.demo_modules[bias]
        # Enable input storage in the next module
        demo_layers["next"].store_input = True
        
        # Verify initial state
        self.assertEqual(demo_layers["add"].store_activity, 1,
                        "Merge module should forward store_activity requests")
        self.assertFalse(demo_layers["next"]._internal_store_input,
                       "Internal store flag should be False when set via property")
        self.assertIsNone(demo_layers["next"].input,
                         "Input should be None before forward pass")

        # Perform forward pass
        _ = demo_layers["seq"](self.input_x)

        # Verify input was stored with correct shape
        self.assertShapeEqual(
            demo_layers["next"].input,
            (self.input_x.size(0), demo_layers["next"].in_features),
            "Stored input should match expected shape (batch_size, in_features)"
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_activity_storage(self, bias: bool):
        """Test pre-activity storage functionality in the merge module.
        
        Verifies that:
        1. The merge module correctly forwards activity storage requests
        2. Pre-activation values are properly stored during the forward pass
        3. The stored activity has the expected shape
        
        Args:
            bias: Whether to test with bias enabled or disabled
        """
        demo_layers = self.demo_modules[bias]
        # Enable pre-activity storage in the previous module
        demo_layers["prev"].store_pre_activity = True
        
        # Verify initial state
        self.assertEqual(demo_layers["add"].store_input, 1,
                        "Merge module should forward store_input requests")
        self.assertFalse(demo_layers["prev"]._internal_store_pre_activity,
                       "Internal store flag should be False when set via property")
        self.assertIsNone(demo_layers["prev"].pre_activity,
                         "Pre-activity should be None before forward pass")

        # Perform forward pass
        _ = demo_layers["seq"](self.input_x)

        # Verify pre-activity was stored with correct shape
        self.assertShapeEqual(
            demo_layers["prev"].pre_activity,
            (self.input_x.size(0), demo_layers["prev"].out_features),
            "Stored pre-activity should match expected shape (batch_size, out_features)"
        )

    def test_update_scaling_factor(self):
        """Test the update of scaling factors in the merge module.
        
        Verifies that:
        1. The merge module correctly updates scaling factors in connected modules
        2. The scaling factor is properly propagated to the next module
        3. The previous module's next_module scaling factor is updated
        4. The previous module's own scaling factor remains unchanged
        """
        demo_layers = self.demo_modules[True]
        test_scaling_factor = 0.5
        
        # Update scaling factor through the merge module
        demo_layers["add"].update_scaling_factor(scaling_factor=test_scaling_factor)
        
        # Verify scaling factors were updated correctly
        self.assertEqual(
            demo_layers["prev"]._scaling_factor_next_module.item(), 
            test_scaling_factor,
            "Previous module's next_module scaling factor should be updated"
        )
        self.assertEqual(
            demo_layers["prev"].scaling_factor.item(), 
            0.0,
            "Previous module's own scaling factor should remain unchanged"
        )
        self.assertEqual(
            demo_layers["next"].scaling_factor.item(), 
            test_scaling_factor,
            "Next module's scaling factor should be updated"
        )

    def test_update_scaling_factor_incorrect_input_module(self):
        """Test error handling when updating scaling factors with invalid input modules.
        
        Verifies that a TypeError is raised when the merge module's previous_modules
        contains an invalid module type (not a GrowingModule).
        """
        demo_layers = self.demo_modules[True]
        # Replace one of the previous modules with a regular Linear layer
        demo_layers["add"].previous_modules = [demo_layers["prev"], torch.nn.Linear(7, 3)]
        
        # Verify that updating scaling factor raises TypeError
        with self.assertRaises(
            TypeError,
            msg="Should raise TypeError when previous_modules contains invalid types"
        ):
            demo_layers["add"].update_scaling_factor(scaling_factor=0.5)

    def test_update_scaling_factor_incorrect_output_module(self):
        """Test error handling when updating scaling factors with invalid output modules.
        
        Verifies that a TypeError is raised when the merge module's next_modules
        contains an invalid module type (not a GrowingModule).
        """
        demo_layers = self.demo_modules[True]
        # Replace the next module with a regular Linear layer
        demo_layers["add"].set_next_modules([demo_layers["next"], torch.nn.Linear(3, 7)])
        
        # Verify that updating scaling factor raises TypeError
        with self.assertRaises(
            TypeError,
            msg="Should raise TypeError when next_modules contains invalid types"
        ):
            demo_layers["add"].update_scaling_factor(scaling_factor=0.5)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_set_previous_next_modules(self, bias: bool):
        """Test dynamic addition of input and output modules to the merge module.
        
        Verifies that:
        1. New input modules can be added to the merge module
        2. New output modules can be added to the merge module
        3. The input/output feature counts are updated correctly
        4. The bias terms are handled correctly when with_bias=True
        
        Args:
            bias: Whether to test with bias enabled or disabled
        """
        demo_layers = self.demo_modules[bias]
        
        # Create new input and output modules to add
        new_input_layer = LinearGrowingModule(
            2, 3,  # 2 input features, 3 output features
            use_bias=bias,
            name="new_prev",
            device=global_device(),
            next_module=demo_layers["add"],
        )
        new_output_layer = LinearGrowingModule(
            3, 2,  # 3 input features, 2 output features
            use_bias=bias,
            name="new_next",
            device=global_device(),
            previous_module=demo_layers["add"],
        )

        # Verify initial state before adding new modules
        self.assertEqual(
            demo_layers["add"].sum_in_features(), 
            demo_layers["prev"].in_features,
            "Initial input feature count should match previous layer's input features"
        )
        self.assertEqual(
            demo_layers["add"].sum_in_features(with_bias=True),
            demo_layers["prev"].in_features + bias,
            "Input feature count with bias should account for bias term"
        )
        self.assertEqual(
            demo_layers["add"].sum_out_features(), 
            demo_layers["next"].out_features,
            "Output feature count should match next layer's output features"
        )

        # Add new input and output modules
        demo_layers["add"].set_previous_modules([demo_layers["prev"], new_input_layer])
        demo_layers["add"].set_next_modules([demo_layers["next"], new_output_layer])

        # Verify updated input feature counts
        self.assertEqual(
            demo_layers["add"].sum_in_features(),
            demo_layers["prev"].in_features + new_input_layer.in_features,
            "Input feature count should be sum of all input modules' features"
        )

        # Verify updated input feature counts with bias
        self.assertEqual(
            demo_layers["add"].sum_in_features(with_bias=True),
            (demo_layers["prev"].in_features + bias + 
             new_input_layer.in_features + bias),
            "Input feature count with bias should include all bias terms"
        )

        # Verify updated output feature count
        self.assertEqual(
            demo_layers["add"].sum_out_features(),
            demo_layers["next"].out_features + new_output_layer.out_features,
            "Output feature count should be sum of all output modules' features"
        )


if __name__ == "__main__":
    main()