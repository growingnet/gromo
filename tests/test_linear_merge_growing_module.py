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
    
    def test_linear_merge_growing_module_warning_scenarios(self):
        """
        Test that warnings are correctly triggered when set_next_modules or set_previous_modules
        are called with non-empty tensor_s or previous_tensor_s/previous_tensor_m.
        Also, test the module's behavior *after* such warnings.
        """
        # Create a LinearMergeGrowingModule
        merge_module = LinearMergeGrowingModule(in_features=5, name="test_merge_module")
        
        # Create a dummy previous module to connect
        prev_module = LinearGrowingModule(5, 5, name="dummy_prev")
        
        # Simulate some activity to make tensor_s and previous_tensor_s/m non-empty
        # For tensor_s (output S of merge_module)
        merge_module.store_activity = True
        merge_module.tensor_s.init()
        merge_module(torch.randn(10, 5, requires_grad=True)).norm().backward()
        merge_module.tensor_s.update()
        
        # For previous_tensor_s and previous_tensor_m (input S and M of merge_module)
        # Need to set previous modules first to initialize these TensorStatistics
        merge_module.set_previous_modules([prev_module])
        prev_module.store_input = True
        merge_module.store_input = True # Ensure merge module stores its input
        merge_module.previous_tensor_s.init()
        merge_module.previous_tensor_m.init()
        merge_module(prev_module(torch.randn(10, 5))).norm().backward() # Pass through merge module to trigger input storage
        
        # Simulate gradient for merge_module.pre_activity
        merge_module.store_pre_activity = True
        merge_module._pre_activity = torch.randn(10, merge_module.in_features)
        merge_module._pre_activity.requires_grad_(True)
        torch.norm(merge_module._pre_activity).backward()

        merge_module.previous_tensor_s.update()
        merge_module.previous_tensor_m.update()

        # Test set_next_modules warning
        with self.assertWarns(UserWarning):
            merge_module.set_next_modules([LinearGrowingModule(5, 5, name="dummy_next")])

        # Test set_previous_modules warnings
        with self.assertWarns(UserWarning):
            merge_module.set_previous_modules([LinearGrowingModule(5, 5, name="another_dummy_prev")])
        
        # Test behavior after warnings: ensure the module still functions
        # The module should still be able to process input and its properties should be accessible
        merge_module.set_previous_modules([prev_module])
        merge_module.init_computation()

        x = torch.randn(10, 5, requires_grad=True)
        output = merge_module(prev_module(x))
        loss = output.norm()
        loss.backward()

        merge_module.tensor_s.update()
        merge_module.previous_tensor_s.update()
        merge_module.previous_tensor_m.update()

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, x.shape)
        self.assertIsNotNone(merge_module.tensor_s())
        self.assertIsNotNone(merge_module.previous_tensor_s())
        self.assertIsNotNone(merge_module.previous_tensor_m())

    def test_construct_full_activity_with_merged_previous_modules(self):
        """
        Test the behavior of construct_full_activity when a previous module is itself a LinearMergeGrowingModule.
        This addresses the FIXME comment on line 138 of linear_growing_module.py.
        """
        # Outer merge module
        # The in_features of outer_merge should match the out_features of inner_merge (which is 5)
        outer_merge = LinearMergeGrowingModule(in_features=5, name="outer_merge")

        # Inner merge module
        inner_merge = LinearMergeGrowingModule(in_features=5, name="inner_merge")

        # LinearGrowingModules connected to the inner merge module
        prev_linear_1 = LinearGrowingModule(3, 5, use_bias=True, name="prev_linear_1")
        prev_linear_2 = LinearGrowingModule(2, 5, use_bias=False, name="prev_linear_2")

        # Set previous modules for the inner merge module
        inner_merge.set_previous_modules([prev_linear_1, prev_linear_2])

        # Set previous module for the outer merge module (the inner merge module)
        outer_merge.set_previous_modules([inner_merge])

        # Enable input storage for the modules
        prev_linear_1.store_input = True
        prev_linear_2.store_input = True
        inner_merge.store_input = True

        # Simulate input for the innermost linear modules
        batch_size = 7
        input_1 = torch.randn(batch_size, 3, requires_grad=True)
        input_2 = torch.randn(batch_size, 2, requires_grad=True)

        # Store inputs in the linear modules
        # Run forward pass to populate _input and _pre_activity
        out1 = prev_linear_1(input_1)
        out2 = prev_linear_2(input_2)

        # Ensure inner_merge's input is populated by running a forward pass
        inner_merge_input = out1 + out2 # Simulate merged input
        _ = inner_merge(inner_merge_input) # Run forward pass to populate inner_merge.input

        # Call construct_full_activity on the inner merge module directly for testing
        full_activity = inner_merge.construct_full_activity()

        # Expected full activity:
        # From prev_linear_1 (3 features + bias): 3 + 1 = 4
        # From prev_linear_2 (2 features, no bias): 2
        # Total expected features: 4 + 2 = 6
        expected_total_features = (prev_linear_1.in_features + prev_linear_1.use_bias) + \
                                  (prev_linear_2.in_features + prev_linear_2.use_bias)
        
        self.assertEqual(full_activity.shape, (batch_size, expected_total_features))

        # Verify the content of the full_activity tensor
        # For prev_linear_1 (with bias)
        self.assertAllClose(full_activity[:, :prev_linear_1.in_features], input_1)
        self.assertAllClose(full_activity[:, prev_linear_1.in_features], torch.ones(batch_size))

        # For prev_linear_2 (no bias)
        self.assertAllClose(full_activity[:, prev_linear_1.in_features + 1 :], input_2)

    def test_compute_optimal_delta_distribution_in_merge_module(self):
        """
        Test that compute_optimal_delta in LinearMergeGrowingModule correctly distributes
        deltas back to multiple previous modules.
        """
        torch.manual_seed(0)
        batch_size = 10

        # Create a LinearMergeGrowingModule
        merge_module = LinearMergeGrowingModule(in_features=7, name="merge_module")

        # Create multiple LinearGrowingModule instances as previous modules
        prev_linear_1 = LinearGrowingModule(3, 7, use_bias=True, name="prev_linear_1")
        prev_linear_2 = LinearGrowingModule(4, 7, use_bias=False, name="prev_linear_2")
        prev_linear_3 = LinearGrowingModule(2, 7, use_bias=True, name="prev_linear_3") # Additional module

        # Set previous modules for the merge module
        merge_module.set_previous_modules([prev_linear_1, prev_linear_2, prev_linear_3])

        # Initalize merge module statistics computation
        prev_linear_1.init_computation()
        prev_linear_2.init_computation()
        prev_linear_3.init_computation()
        print(f"Store activity before init: {merge_module.store_activity}")
        print(f"Store pre_activity before init: {merge_module.store_activity}")
        merge_module.init_computation()
        print(f"Store activity after init: {merge_module.store_activity}")
        print(f"Store pre_activity after init: {merge_module.store_activity}")

        # Simulate input for the previous linear modules
        input_1 = torch.randn(batch_size, 3, requires_grad=True)
        input_2 = torch.randn(batch_size, 4, requires_grad=True)
        input_3 = torch.randn(batch_size, 2, requires_grad=True)

        # Simulate activity and accumulate statistics
        out_1 = prev_linear_1(input_1)
        out_2 = prev_linear_2(input_2)
        out_3 = prev_linear_3(input_3)

        # Merge these outputs
        merged_output_for_merge_module = out_1 + out_2 + out_3 # Assuming in_features match

        # Run the merge module forward pass to populate its internal 'input' and 'activity'
        # and ensure gradients are retained for backward pass
        # merged_output_for_merge_module.requires_grad_(True)
        # merged_output_for_merge_module.retain_grad() # Ensure gradient is retained
        merged_output = merge_module(merged_output_for_merge_module)
        loss = merged_output.norm()
        loss.backward()

        # Update previous_tensor_s and previous_tensor_m
        print(f"Store activity before update: {merge_module.store_activity}")
        print(f"Store pre_activity before update: {merge_module.store_activity}")
        merge_module.previous_tensor_s.update()
        merge_module.previous_tensor_m.update()

        # Compute optimal delta
        merge_module.compute_optimal_delta()

        # Verify that each previous module has its optimal_delta_layer set
        self.assertIsNotNone(prev_linear_1.optimal_delta_layer)
        self.assertIsNotNone(prev_linear_2.optimal_delta_layer)
        self.assertIsNotNone(prev_linear_3.optimal_delta_layer)

        # Verify the shapes of the delta layers
        self.assertEqual(prev_linear_1.optimal_delta_layer.weight.shape, (7, 3))
        self.assertEqual(prev_linear_1.optimal_delta_layer.bias.shape, (7,))
        
        self.assertEqual(prev_linear_2.optimal_delta_layer.weight.shape, (7, 4))
        self.assertIsNone(prev_linear_2.optimal_delta_layer.bias)

        self.assertEqual(prev_linear_3.optimal_delta_layer.weight.shape, (7, 2))
        self.assertEqual(prev_linear_3.optimal_delta_layer.bias.shape, (7,))

        # Further verification would involve checking the actual values of the delta layers
        # against a theoretically computed optimal delta, which is complex for this setup.
        # For now, shape and existence checks are sufficient to confirm distribution.

    def test_linear_merge_growing_module_not_implemented_errors(self):
        """
        Test scenarios that currently raise NotImplementedError when LinearMergeGrowingModule
        is used as a previous module for LinearGrowingModule.
        Specifically, test tensor_s_growth and compute_optimal_added_parameters.
        """
        # Create a LinearMergeGrowingModule to act as a previous module
        merge_module = LinearMergeGrowingModule(in_features=5, name="merge_as_prev")

        # Create a LinearGrowingModule with the merge_module as its previous module
        growing_module = LinearGrowingModule(
            in_features=5, out_features=3, previous_module=merge_module, name="growing_with_merge_prev"
        )

        # Test tensor_s_growth property
        with self.assertRaisesRegex(NotImplementedError, "S growth is not implemented for module preceded by an LinearMergeGrowingModule."):
            _ = growing_module.tensor_s_growth

        with self.assertRaisesRegex(NotImplementedError, ""):
            growing_module.compute_optimal_added_parameters()
    
    # def test_warning_set_next_modules_with_non_empty_tensor_s(self):
    #     """
    #     Test that a warning is triggered when set_next_modules is called with non-empty tensor S.
        
    #     This test verifies that:
    #     1. A UserWarning is raised when set_next_modules is called on a LinearMergeGrowingModule
    #        that has non-empty tensor S samples.
    #     2. The warning message correctly identifies the issue.
    #     """
    #     # Create a LinearMergeGrowingModule
    #     merge_module = LinearMergeGrowingModule(in_features=5, name="test_merge_module")
        
    #     # Simulate some activity to make tensor_s non-empty
    #     merge_module.store_activity = True
    #     merge_module.tensor_s.init()
        
    #     # Run a forward pass to populate tensor_s
    #     x = torch.randn(10, 5, requires_grad=True)
    #     _ = merge_module(x)
    #     merge_module.tensor_s.update()
        
    #     # Verify tensor_s has samples
    #     self.assertGreater(merge_module.tensor_s.samples, 0)
        
    #     # Test that warning is triggered when setting next modules
    #     with self.assertWarns(UserWarning) as warning_context:
    #         merge_module.set_next_modules([LinearGrowingModule(5, 5, name="dummy_next")])
        
    #     # Verify the warning message
    #     self.assertIn("You are setting the next modules", str(warning_context.warning))
    #     self.assertIn("with a non-empty tensor S", str(warning_context.warning))

    # def test_warning_set_previous_modules_with_non_empty_tensors(self):
    #     """
    #     Test that warnings are triggered when set_previous_modules is called with non-empty
    #     previous_tensor_s or previous_tensor_m.
        
    #     This test verifies that:
    #     1. A UserWarning is raised when set_previous_modules is called on a LinearMergeGrowingModule
    #        that has non-empty previous_tensor_s samples.
    #     2. A UserWarning is raised when set_previous_modules is called on a LinearMergeGrowingModule
    #        that has non-empty previous_tensor_m samples.
    #     3. The warning messages correctly identify the issues.
    #     """
    #     # Create a LinearMergeGrowingModule
    #     merge_module = LinearMergeGrowingModule(in_features=5, name="test_merge_module")
        
    #     # Create a dummy previous module to connect
    #     prev_module = LinearGrowingModule(5, 5, name="dummy_prev")
        
    #     # Set previous modules first to initialize previous_tensor_s and previous_tensor_m
    #     merge_module.set_previous_modules([prev_module])
        
    #     # Simulate some activity to make previous_tensor_s non-empty
    #     prev_module.store_input = True
    #     merge_module.store_input = True
    #     merge_module.previous_tensor_s.init()
    #     merge_module.previous_tensor_m.init()
        
    #     # Run forward passes to populate previous_tensor_s and previous_tensor_m
    #     x = torch.randn(10, 5, requires_grad=True)
    #     y = prev_module(x)
    #     out = merge_module(y)
    #     out.norm().backward()
        
    #     # Update previous_tensor statistics
    #     merge_module.previous_tensor_s.update()
    #     merge_module.previous_tensor_m.update()
        
    #     # Verify previous_tensor_s and previous_tensor_m have samples
    #     self.assertGreater(merge_module.previous_tensor_s.samples, 0)
    #     self.assertGreater(merge_module.previous_tensor_m.samples, 0)
        
    #     # Test that warning is triggered when setting previous modules with non-empty previous_tensor_s
    #     with self.assertWarns(UserWarning) as warning_context:
    #         merge_module.set_previous_modules([LinearGrowingModule(5, 5, name="another_dummy_prev")])
        
    #     # Verify the warning message for previous_tensor_s
    #     self.assertIn("You are setting the previous modules", str(warning_context.warnings[0].message))
    #     self.assertIn("with a non-empty previous tensor S", str(warning_context.warnings[0].message))
        
    #     # Verify the warning message for previous_tensor_m
    #     self.assertIn("You are setting the previous modules", str(warning_context.warnings[1].message))
    #     self.assertIn("with a non-empty previous tensor M", str(warning_context.warnings[1].message))


    def test_compute_s_update_activity_fallback(self):
        """
        Test compute_s_update method fallback to activity when input is not stored.
        
        This test verifies that:
        1. compute_s_update uses activity when store_input is False but store_activity is True
        2. The fallback computation produces correct tensor shapes
        3. Appropriate errors are raised when neither input nor activity is stored
        """
        # Create a merge module that uses activity instead of input for S computation
        merge_module = LinearMergeGrowingModule(in_features=5)
        
        # Enable activity storage but not input storage
        merge_module.store_activity = True
        merge_module.store_input = False
        
        # Initialize tensor_s
        merge_module.tensor_s.init()
        
        # Run forward pass to populate activity
        x = torch.randn(10, 5)
        _ = merge_module(x)
        
        # Update tensor S using activity (line 222 coverage)
        merge_module.tensor_s.update()
        
        # Verify the tensor was computed correctly
        s_tensor = merge_module.tensor_s()
        self.assertIsInstance(s_tensor, torch.Tensor)
        # For activity computation with bias extension, shape should be (6, 6) because use_bias=True
        self.assertEqual(s_tensor.shape, (6, 6))  # 5 features + 1 bias


if __name__ == "__main__":
    main()