from unittest import TestCase, main

import torch

from gromo.modules.growing_module import GrowingModule
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize


class TestGrowingModule(TorchTestCase):
    """
    Test suite for the GrowingModule class.
    
    This class contains unit tests that verify the core functionality of the GrowingModule,
    which serves as the base class for growing neural network modules. The tests cover:
    - Basic module properties and forward pass
    - Extended forward pass with input/output extensions
    - Module initialization and connection management
    - Input handling and storage
    - Update and deletion mechanisms
    
    The test class sets up common test fixtures including a base model and extension layers.
    Each test method focuses on a specific aspect of the module's functionality.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(0)  # Ensure reproducibility
        
        # Create test input tensors
        self.x = torch.randn(2, 3, device=global_device())  # Base input
        self.x_ext = torch.randn(2, 7, device=global_device())  # Extended input
        
        # Create base layer and extension layers
        self.layer = torch.nn.Linear(3, 5, bias=False, device=global_device())
        self.layer_in_extension = torch.nn.Linear(
            7, 5, bias=False, device=global_device()
        )
        self.layer_out_extension = torch.nn.Linear(
            3, 7, bias=False, device=global_device()
        )
        
        # Initialize the GrowingModule under test
        self.model = GrowingModule(
            layer=self.layer,
            tensor_s_shape=(3, 3),
            tensor_m_shape=(3, 5),
            allow_growing=False
        )

    def test_weight(self):
        """Test that the module's weight matches the underlying layer's weight."""
        self.assertTrue(
            torch.equal(self.model.weight, self.layer.weight),
            "Module weight should match the underlying layer's weight"
        )

    def test_bias(self):
        """
        Test that the module's bias is None when the underlying layer has no bias.
        
        The test layer is initialized with bias=False, so we expect the bias to be None.
        """
        self.assertIsNone(
            self.model.bias,
            "Bias should be None when the underlying layer has no bias"
        )

    def test_forward(self):
        """
        Test the basic forward pass of the GrowingModule.
        
        Verifies that the module's forward pass produces the same output as directly
        calling the underlying layer's forward pass with the same input.
        """
        expected_output = self.layer(self.x)
        actual_output = self.model(self.x)
        self.assertTrue(
            torch.equal(actual_output, expected_output),
            "Module forward should produce the same output as the underlying layer"
        )

    def test_extended_forward(self):
        """
        Test the extended forward pass with various input/output extension scenarios.
        
        Verifies:
        1. Basic extended forward pass with no extensions
        2. Extended forward with input extension layer
        3. Error handling for missing input extension
        4. Extended forward with output extension layer
        """
        # Test basic extended forward (no extensions)
        y_th = self.layer(self.x)
        y, y_sup = self.model.extended_forward(self.x)
        self.assertIsNone(
            y_sup,
            "Secondary output should be None when no output extension is configured"
        )
        self.assertTrue(
            torch.equal(y, y_th),
            "Primary output should match the base layer's output"
        )

        # ========== Test with input extension ==========
        # Configure input extension and test with extended input
        self.model.extended_input_layer = self.layer_in_extension
        self.model.scaling_factor = 1.0  # Set scaling factor for the extension
        
        y, y_sup = self.model.extended_forward(self.x, self.x_ext)
        expected_extension = self.layer_in_extension(self.x_ext)
        self.assertIsNone(
            y_sup,
            "Secondary output should still be None with only input extension"
        )
        self.assertTrue(
            torch.allclose(y, y_th + expected_extension),
            "Output should be the sum of base and extended input processing"
        )

        # Test warning when extended input is provided but no extension layer is configured
        self.model.extended_input_layer = None
        with self.assertWarns(
            UserWarning,
            msg="Should warn when extended input is provided without extension layer"
        ):
            # This should warn but not raise an error
            self.model.extended_forward(self.x, self.x_ext)

        # ========== Test with output extension ==========
        # Test warning when extended input is provided but no extension is configured
        with self.assertWarns(
            UserWarning,
            msg="Should warn when extended input is provided without extension layer"
        ):
            self.model.extended_forward(self.x, self.x_ext)

        # Configure output extension and test
        self.model.extended_output_layer = self.layer_out_extension
        self.model._scaling_factor_next_module = 1.0
        
        y, y_sup = self.model.extended_forward(self.x)
        expected_extension = self.layer_out_extension(self.x)
        
        self.assertTrue(
            torch.equal(y, y_th),
            "Primary output should remain unchanged with output extension"
        )
        self.assertTrue(
            torch.equal(y_sup, expected_extension),
            "Secondary output should match the output extension's processing"
        )

    def test_str(self):
        """
        Test the string representation of the GrowingModule.
        
        Verifies that the string representation is a non-empty string.
        """
        model_str = str(self.model)
        self.assertIsInstance(
            model_str, 
            str,
            "String representation should return a string"
        )
        self.assertGreater(
            len(model_str), 0,
            "String representation should not be empty"
        )

    def test_repr(self):
        """
        Test the official string representation of the GrowingModule.
        
        Verifies that the representation is a non-empty string containing
        relevant information about the module.
        Note: Only check the verbose=0 case
        """
        model_repr = repr(self.model)
        self.assertIsInstance(
            model_repr, 
            str,
            "Representation should return a string"
        )
        self.assertGreater(
            len(model_repr), 0,
            "Representation should not be empty"
        )
        
        # Check for key components in the representation
        self.assertIn(
            'GrowingModule', 
            model_repr,
            "Representation should include the class name"
        )
        self.assertIn(
            'parameters', 
            model_repr,
            "Representation should include parameter information"
        )

    def test_init(self):
        """
        Test the initialization of GrowingModule with various configurations.
        
        Verifies:
        1. Proper error when allow_growing=True without proper tensor shapes
        2. Successful creation with allow_growing=False
        3. Proper connection when initializing with a previous module
        """
        # Test that allow_growing=True requires proper tensor shapes
        with self.assertRaises(
            AssertionError,
            msg="Should raise AssertionError when allow_growing=True without proper shapes"
        ):
            GrowingModule(
                torch.nn.Linear(3, 5, bias=False, device=global_device()),
                tensor_s_shape=(3, 3),
                tensor_m_shape=(3, 5),
                allow_growing=True,  # Should fail without proper tensor shapes
            )

        # Test successful creation with allow_growing=False
        l1 = GrowingModule(
            torch.nn.Linear(3, 5, bias=False, device=global_device()),
            tensor_s_shape=(3, 3),
            tensor_m_shape=(3, 5),
            allow_growing=False,
        )
        self.assertIsInstance(
            l1, 
            GrowingModule,
            "Should successfully create a GrowingModule with allow_growing=False"
        )

        # Test connection with a previous module
        l2 = GrowingModule(
            torch.nn.Linear(5, 7, bias=False, device=global_device()),
            tensor_s_shape=(5, 5),
            tensor_m_shape=(5, 7),
            allow_growing=True,
            previous_module=l1,  # Connect to the first module
        )

        self.assertIsInstance(
            l2, 
            GrowingModule,
            "Should successfully create a connected GrowingModule"
        )
        self.assertIs(
            l2.previous_module, 
            l1,
            "Previous module reference should be properly set"
        )

    @staticmethod
    def _reset_layer(layer: 'GrowingModule', first: bool = True) -> None:
        """
        Helper method to reset layer attributes to a known state for testing.
        
        Args:
            layer: The layer to reset
            first: If True, skip resetting extended_input_layer
        """
        dummy_layer = torch.nn.Identity()
        layer.extended_output_layer = dummy_layer
        layer.optimal_delta_layer = dummy_layer
        if not first:
            layer.extended_input_layer = dummy_layer

    def _reset_all_layers(self, *layers: 'GrowingModule') -> None:
        """Reset all test layers to a known state."""
        for i, layer in enumerate(layers):
            self._reset_layer(layer, first=(i == 0))

    def test_delete_update(self):
        """
        Test the delete_update method with various configurations.
        
        Verifies the behavior of deleting update-related attributes with different
        combinations of parameters (include_previous, include_output).
        """
        # Create two connected layers for testing
        l1 = GrowingModule(
            torch.nn.Linear(3, 5, bias=False, device=global_device()),
            tensor_s_shape=(3, 3),
            tensor_m_shape=(3, 5),
            allow_growing=False,
        )
        l2 = GrowingModule(
            torch.nn.Linear(5, 7, bias=False, device=global_device()),
            tensor_s_shape=(5, 5),
            tensor_m_shape=(5, 7),
            allow_growing=True,
            previous_module=l1,  # l1 -> l2 connection
        )

        # Test 1: Delete update on first layer (l1)
        self._reset_all_layers(l1, l2)
        l1.delete_update()
        
        self.assertIsInstance(
            l1.extended_output_layer, 
            torch.nn.Identity,
            "extended_output_layer should be reset to Identity"
        )
        self.assertIsNone(
            l1.optimal_delta_layer,
            "optimal_delta_layer should be set to None"
        )

        # Test 2: Delete update on second layer without including previous
        self._reset_all_layers(l1, l2)
        with self.assertWarns(
            UserWarning,
            msg="Should warn when include_previous=False with a previous module"
        ):
            l2.delete_update(include_previous=False)
            
        self.assertIsNone(
            l2.extended_input_layer,
            "extended_input_layer should be set to None"
        )
        self.assertIsInstance(
            l1.extended_output_layer,
            torch.nn.Identity,
            "Previous layer's extended_output_layer should remain unchanged"
        )
        self.assertIsNone(
            l2.optimal_delta_layer,
            "optimal_delta_layer should be set to None"
        )

        # Test 3: Delete update on second layer (should include previous)
        self._reset_all_layers(l1, l2)
        l2.delete_update()
        
        self.assertIsNone(
            l2.extended_input_layer,
            "extended_input_layer should be set to None"
        )
        self.assertIsNone(
            l1.extended_output_layer,
            "Previous layer's extended_output_layer should be set to None"
        )
        self.assertIsNone(
            l2.optimal_delta_layer,
            "optimal_delta_layer should be set to None"
        )

        # Test 4: Delete update including output
        self._reset_all_layers(l1, l2)
        l2.delete_update(include_output=True)
        
        self.assertIsNone(
            l2.extended_input_layer,
            "extended_input_layer should be set to None"
        )
        self.assertIsNone(
            l1.extended_output_layer,
            "Previous layer's extended_output_layer should be set to None"
        )
        self.assertIsNone(
            l2.optimal_delta_layer,
            "optimal_delta_layer should be set to None"
        )
        self.assertIsNone(
            l2.extended_output_layer,
            "extended_output_layer should be set to None when include_output=True"
        )

        # Test incorrect behavior - delete update with invalid previous module
        self._reset_layer(l1, first=False)
        with self.assertWarns(UserWarning):
            l1.delete_update()

        # Test incorrect behavior - invalid previous module type
        self._reset_layer(l1, first=False)
        with self.assertRaises(TypeError):
            l1.previous_module = True  # type: ignore
            l1.delete_update()

        # Test incorrect behavior - invalid previous module type with include_previous=False
        self._reset_layer(l1, first=False)
        with self.assertRaises(TypeError):
            l1.previous_module = True  # type: ignore
            l1.delete_update(include_previous=False)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_input(self, bias: bool = True):
        """
        Test input storage functionality in the GrowingModule.
        
        Verifies that:
        1. Input tensors are properly stored during forward pass
        2. Stored input has the correct shape and device
        3. Input storage respects the module's bias configuration
        
        Args:
            bias: Whether to test with bias enabled or disabled
        """
        # Create a test layer and wrap it in a GrowingModule
        layer = torch.nn.Linear(3, 5, bias=bias, device=global_device())
        model = GrowingModule(
            layer, 
            tensor_s_shape=(3, 3), 
            tensor_m_shape=(3, 5), 
            allow_growing=False
        )
        
        # Enable input storage and perform forward pass
        model.store_input = True
        model(self.x)  # Perform forward pass
        
        # Verify input was stored correctly
        self.assertIsNotNone(
            model.input,
            "Input should be stored when store_input is True"
        )
        self.assertShapeEqual(
            model.input, 
            (2, 3),  # batch_size=2, input_features=3
            "Stored input should match expected shape"
        )
        self.assertEqual(
            model.input.device, 
            global_device(),
            "Stored input should be on the correct device"
        )
        self.assertEqual(
            model.input.dtype, 
            torch.float32,
            "Stored input should be float32"
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_input_extended(self, bias: bool = True):
        """
        Test extended input storage functionality in the GrowingModule.
        
        This test verifies the behavior of input storage when using extended inputs,
        with special handling for modules with and without bias terms.
        
        When bias is enabled (bias=True):
        - The input_extended property should raise NotImplementedError since
          extended inputs with bias terms are not implemented
          
        When bias is disabled (bias=False):
        - The input_extended property should return the same as the regular input
        
        Args:
            bias: Boolean flag to test both with and without bias terms
        """
        # Configure the model with the specified bias setting
        self.model.use_bias = bias
        self.model.store_input = True
        
        # Perform a forward pass to store the input
        self.model(self.x)
        
        # Test behavior based on bias configuration
        if bias:
            # With bias, extended input should raise NotImplementedError
            with self.assertRaises(
                NotImplementedError,
                msg="Extended inputs with bias terms should raise NotImplementedError"
            ):
                _ = self.model.input_extended
        else:
            # Without bias, extended input should match regular input
            self.assertAllClose(
                self.model.input_extended,
                self.x,
                msg="With bias disabled, extended input should match regular input"
            )


if __name__ == "__main__":
    main()
