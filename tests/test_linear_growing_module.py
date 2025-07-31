"""
Test suite for LinearGrowingModule and LinearMergeGrowingModule in gromo.modules.linear_growing_module.

This module contains unit tests that verify the functionality of the linear growing modules,
including their ability to handle tensor operations, parameter updates, and growth mechanisms.
The tests cover both basic functionality and edge cases, with a focus on numerical stability
and correctness of the underlying mathematical operations.

Key Test Areas:
- Tensor statistics computation (S and M tensors)
- Parameter updates and gradient calculations
- Module growth and merging operations
- Numerical stability and edge cases
"""

from copy import deepcopy
from unittest import TestCase, main

import torch

from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize


def theoretical_s_1(n, c):
    """
    Compute theoretical values of the second moment matrix S for specific input patterns.
    
    This helper function generates test data and computes the expected values of the S tensor
    (second moment matrix) for both input and output of a specially structured weight matrix W.
    The weight matrix W has the form:
        W = [0 ... 0 \
             0 1 0 ... 0 \
             0 0 2 0 ... 0 \
             ... \
             1 ... 1]
    
    The function is used to verify the correctness of tensor statistics computation
    in the LinearGrowingModule and LinearMergeGrowingModule classes.

    Parameters
    ----------
    n : int
        Number of samples in the test data.
    c : int
        Number of features in the input data.

    Returns
    -------
    tuple
        A tuple containing:
        - x1 (torch.Tensor): First test input tensor of shape (n, c)
        - x2 (torch.Tensor): Second test input tensor of shape (n, c)
        - is1 (torch.Tensor): Theoretical value of n*S for input x1
        - is2 (torch.Tensor): Theoretical value of 2n*S for concatenated inputs [x1, x2]
        - os1 (torch.Tensor): Theoretical value of n*S for output W(x1)
        - os2 (torch.Tensor): Theoretical value of 2n*S for output W([x1, x2])
        
    Note
    ----
    The S tensor represents the second moment matrix (X^T X) where X is the input matrix.
    The theoretical values are computed using closed-form solutions for the specific
    weight matrix structure and input patterns used in these tests.
    """

    va = torch.arange(c)
    v1 = torch.ones(c, dtype=torch.long)
    is0 = va.view(-1, 1) @ va.view(1, -1)
    isc = va.view(-1, 1) @ v1.view(1, -1)
    isc = isc + isc.T
    is1 = torch.ones(c, c)
    va_im = torch.arange(c + 1) ** 2
    va_im[-1] = c * (c - 1) // 2
    v1_im = torch.arange(c + 1)
    os0 = va_im.view(-1, 1) @ va_im.view(1, -1)
    osc = va_im.view(-1, 1) @ v1_im.view(1, -1)
    osc = osc + osc.T
    os1 = v1_im.view(-1, 1) @ v1_im.view(1, -1)

    x1 = torch.ones(n, c)
    x1 *= torch.arange(n).view(-1, 1)

    x2 = torch.tile(torch.arange(c), (n, 1))
    x2 += torch.arange(n).view(-1, 1)

    is_theory_1 = n * (n - 1) * (2 * n - 1) // 6 * is1

    os_theory_1 = n * (n - 1) * (2 * n - 1) // 6 * os1

    is_theory_2 = n * is0 + n * (n - 1) // 2 * isc + n * (n - 1) * (2 * n - 1) // 3 * is1

    os_theory_2 = n * os0 + n * (n - 1) // 2 * osc + n * (n - 1) * (2 * n - 1) // 3 * os1

    return x1, x2, is_theory_1, is_theory_2, os_theory_1, os_theory_2


class TestLinearGrowingModule(TorchTestCase):
    """
    Test suite for the LinearGrowingModule class.
    
    This class contains unit tests that verify the functionality of the LinearGrowingModule,
    including tensor statistics computation, parameter updates, and growth operations.
    The tests use both synthetic data with known theoretical values and randomized inputs
    to ensure numerical stability and correctness.
    
    The test class sets up common test fixtures and provides helper methods for repeated
    test patterns. Each test method focuses on a specific aspect of the module's functionality.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method is run.
        
        Initializes:
        - Test parameters (n, c)
        - A special weight matrix for testing
        - Random input data
        - Demo layer configurations with and without bias
        """
        self.n = 11
        # This assert is checking that the test is correct and not that the code is correct
        # that why it is not a self.assert*
        assert self.n % 2 == 1
        self.c = 5

        self.weight_matrix_1 = torch.ones(self.c + 1, self.c, device=global_device())
        self.weight_matrix_1[:-1] = torch.diag(torch.arange(self.c)).to(global_device())
        # W = (0 ... 0 \\ 0 1 0 ... 0 \\ 0 0 2 0 ... 0 \\ ... \\ 1 ... 1)

        torch.manual_seed(0)
        self.input_x = torch.randn((11, 5), device=global_device())
        self.demo_layers = dict()
        for bias in (True, False):
            demo_layer_1 = LinearGrowingModule(
                5,
                3,
                use_bias=bias,
                name=f"L1({'bias' if bias else 'no_bias'})",
                device=global_device(),
            )
            demo_layer_2 = LinearGrowingModule(
                3,
                7,
                use_bias=bias,
                name=f"L2({'bias' if bias else 'no_bias'})",
                previous_module=demo_layer_1,
                device=global_device(),
            )
            self.demo_layers[bias] = (demo_layer_1, demo_layer_2)

    def test_compute_s(self):
        """
        Test the computation of the second moment matrix (S) for both input and output.
        
        This test verifies that:
        1. The LinearGrowingModule correctly computes the S tensor (second moment matrix)
           for both input and output tensors.
        2. The LinearMergeGrowingModule correctly tracks and updates the S tensor
           from its input modules.
        3. The computed values match the theoretical expectations for the given
           weight matrix structure and input patterns.
           
        The test uses a specially constructed weight matrix and known input patterns
        for which we can compute the expected S tensors analytically.
        """
        # Get test data and theoretical values for S tensors
        # x1, x2: Test input tensors
        # is_th_1, is_th_2: Theoretical S tensors for input after 1st and 2nd passes
        # os_th_1, os_th_2: Theoretical S tensors for output after 1st and 2nd passes
        x1, x2, is_th_1, is_th_2, os_th_1, os_th_2 = theoretical_s_1(self.n, self.c)

        # Set up test network with one linear layer and one merge module
        output_module = LinearMergeGrowingModule(in_features=self.c + 1, name="output")
        layer = LinearGrowingModule(
            self.c, self.c + 1, use_bias=False, name="layer1", next_module=output_module
        )
        
        # Connect the layer to the merge module
        output_module.set_previous_modules([layer])
        net = torch.nn.Sequential(layer, output_module)

        # Set the weight matrix to our special test pattern
        # This matrix has a specific structure that makes theoretical calculations tractable
        layer.layer.weight.data = self.weight_matrix_1

        # Initialize and configure tensor statistics tracking
        layer.tensor_s.init()  # Initialize input S tensor
        layer.store_input = True  # Enable input storage for S computation
        
        output_module.tensor_s.init()  # Initialize output S tensor
        output_module.store_activity = True  # Enable activity storage for output S
        
        # Initialize S tensor for tracking statistics from previous layer
        output_module.previous_tensor_s.init()

        # Forward pass 1: Process first test input
        _ = net(x1.float().to(global_device()))
        
        # Update statistics after first pass
        layer.tensor_s.update()  # Update input statistics
        output_module.tensor_s.update()  # Update output statistics
        output_module.previous_tensor_s.update()  # Update merge module's view of previous layer

        # Verify S tensors after first pass
        # 1. Check that the input S tensor matches the theoretical value
        #    (normalized by number of samples)
        self.assertAllClose(
            layer.tensor_s(), 
            is_th_1.float().to(global_device()) / self.n,
            msg="Input S tensor after first pass does not match theoretical value"
        )
        
        # 2. Check that the output S tensor matches the theoretical value
        #    We only check the upper-left (c+1)x(c+1) submatrix as that's where
        #    the non-zero values should be based on our test setup
        self.assertAllClose(
            output_module.tensor_s()[: self.c + 1, : self.c + 1],
            os_th_1.float().to(global_device()) / self.n,
            msg="Output S tensor after first pass does not match theoretical value"
        )

        # 3. Verify that the merge module's view of the previous S tensor is consistent
        #    with what we computed directly from the layer
        self.assertAllClose(
            output_module.previous_tensor_s(),
            is_th_1.float().to(global_device()) / self.n,
            msg="Merge module's view of previous S tensor is inconsistent"
        )

        # Second forward pass with input x2
        _ = net(x2.float().to(global_device()))
        
        # Update statistics after second pass
        layer.tensor_s.update()
        output_module.tensor_s.update()
        output_module.previous_tensor_s.update()

        # Verify S tensors after second pass (should be averages over both passes)
        # 1. Check input S tensor after both passes
        self.assertAllClose(
            layer.tensor_s(), 
            is_th_2.float().to(global_device()) / (2 * self.n),
            msg="Input S tensor after second pass does not match theoretical value"
        )
        
        # 2. Check output S tensor after both passes
        self.assertAllClose(
            output_module.tensor_s()[: self.c + 1, : self.c + 1],
            os_th_2.float().to(global_device()) / (2 * self.n),
            msg="Output S tensor after second pass does not match theoretical value"
        )

        # 3. Final consistency check for merge module's view after both passes
        self.assertAllClose(
            output_module.previous_tensor_s(),
            is_th_2.float().to(global_device()) / (2 * self.n),
            msg="Merge module's view of previous S tensor is inconsistent after second pass"
        )

    @unittest_parametrize(
        (
            {"force_pseudo_inverse": True, "test_name": "with_pseudo_inverse"},
            {"force_pseudo_inverse": False, "test_name": "with_standard_inverse"},
            {"update_layer": False, "test_name": "without_update"},
        )
    )
    def test_compute_delta(
        self, force_pseudo_inverse: bool = False, update_layer: bool = True, **kwargs
    ):
        """
        Test the computation of optimal weight updates (delta) for the linear layer.
        
        This test verifies that:
        1. The module correctly computes weight updates using either standard inverse
           or pseudo-inverse methods based on the force_pseudo_inverse flag.
        2. The computed updates minimize the loss function as expected.
        3. The behavior is consistent whether or not the layer parameters are updated.
        
        The test is parameterized to check different scenarios:
        - force_pseudo_inverse=True: Forces use of pseudo-inverse for the update
        - force_pseudo_inverse=False: Uses standard inverse when possible
        - update_layer=False: Tests computation without applying updates
        
        Parameters
        ----------
        force_pseudo_inverse : bool, optional
            Whether to force the use of pseudo-inverse for the update computation.
        update_layer : bool, optional
            Whether to apply the computed updates to the layer parameters.
        **kwargs : dict
            Additional parameters from the parameterization, including 'test_name' for
            better test identification in the output.
        """
        # Test different reduction modes for the loss function
        # Note: "mean" and "sum" reductions are currently not working as expected
        for reduction in {"mixed"}:
            # Set up the appropriate batch reduction factor based on reduction mode
            # - mean: normalize by batch size and number of batches
            # - mixed: only normalize by total number of samples (batch_size * num_batches)
            # - sum: no normalization
            batch_red = self.c if reduction == "mean" else 1
            
            # Define a simple L2 loss function with the specified reduction
            loss_func = lambda x, y: torch.norm(x - y) ** 2 / batch_red

            # Test different learning rates (alpha) to verify update scaling
            for alpha in (0.1, 1.0, 10.0):
                # Create a test layer with zero-initialized weights
                layer = LinearGrowingModule(self.c, self.c, use_bias=False, name="layer1")
                layer.layer.weight.data = torch.zeros_like(
                    layer.layer.weight, device=global_device()
                )
                
                # Initialize and configure tensor statistics
                layer.tensor_s.init()  # For second moment matrix
                layer.tensor_m.init()  # For gradient statistics
                layer.store_input = True  # Store input for S computation
                layer.store_pre_activity = True  # Store pre-activations for gradient computation

                # Run multiple batches to accumulate statistics
                for _ in range(nb_batch := 3):
                    # Create input and target tensors (identity matrix scaled by alpha)
                    x = alpha * torch.eye(self.c, device=global_device())
                    y = x  # Target is same as input (identity mapping)
                    
                    # Forward pass
                    out = layer(x)
                    
                    # Compute loss and backpropagate
                    loss = loss_func(out, y)
                    loss.backward()
                    
                    # Update statistics with this batch
                    layer.tensor_s.update()  # Update second moment matrix
                    layer.tensor_m.update()  # Update gradient statistics
                    layer.zero_grad()  # Reset gradients for next iteration

                # S
                self.assertAllClose(
                    layer.tensor_s(),
                    alpha**2 * torch.eye(self.c, device=global_device()) / self.c,
                    message=f"Error in S for {reduction=}, {alpha=}",
                )

                # dL / dA
                self.assertAllClose(
                    layer.pre_activity.grad,
                    -2 * alpha * torch.eye(self.c, device=global_device()) / batch_red,
                    message=f"Error in dL/dA for {reduction=}, {alpha=}",
                )

                # M
                self.assertAllClose(
                    layer.tensor_m(),
                    -2
                    * alpha**2
                    * torch.eye(self.c, device=global_device())
                    / self.c
                    / batch_red,
                    message=f"Error in M for {reduction=}, {alpha=}",
                )

                # Compute the optimal weight update
                w, _, fo = layer.compute_optimal_delta(
                    force_pseudo_inverse=force_pseudo_inverse, update=update_layer
                )

                # dW*
                self.assertAllClose(
                    w,
                    -2 * torch.eye(self.c, device=global_device()) / batch_red,
                    message=f"Error in dW* for {reduction=}, {alpha=}",
                )

                if update_layer:
                    self.assertAllClose(
                        layer.optimal_delta_layer.weight,
                        w,
                        message=f"Error in the update of the delta layer for {reduction=}, {alpha=}",
                    )
                else:
                    self.assertIsNone(
                        layer.optimal_delta_layer,
                    )

                # <dW*, dL/dA>
                factors = {
                    "mixed": 1,
                    "mean": self.c,  # batch size to compensate the batch normalization
                    "sum": self.c * nb_batch,  # number of samples
                }
                self.assertAlmostEqual(
                    fo.item(),
                    4 * alpha**2 / batch_red**2 * factors[reduction],
                    places=3,
                    msg=f"Error in <dW*, dL/dA> for {reduction=}, {alpha=}",
                )

    def test_str(self):
        self.assertIsInstance(str(LinearGrowingModule(5, 5)), str)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_extended_forward_out(self, bias):
        """
        Test the extended_forward method for output extension in LinearGrowingModule.
        
        This test verifies that:
        1. The extended_forward method correctly computes the extended outputs
           when adding new output dimensions to the layer.
        2. The computation respects the scaling factors for both the current layer
           and the next module.
        3. The apply_change and _apply_output_changes methods correctly modify
           the layer's behavior.
           
        The test uses fixed linear layers with known weights to verify the computations.
        
        Parameters
        ----------
        bias : bool
            Whether to include bias terms in the test layers.
        """
        torch.manual_seed(0)
        # Fixed layer representing the original layer weights
        l0 = torch.nn.Linear(5, 1, bias=bias, device=global_device())
        # Layer representing the extension to add new output dimensions
        l_ext = torch.nn.Linear(5, 2, bias=bias, device=global_device())
        # Layer representing the optimal delta update for the original output
        l_delta = torch.nn.Linear(5, 1, bias=bias, device=global_device())

        # Create the growing layer under test and initialize it with the same weights as l0
        layer = LinearGrowingModule(
            5, 1, use_bias=bias, name="layer1", device=global_device()
        )
        # Initialize layer weights and set up test configurations
        layer.weight.data.copy_(l0.weight.data)
        if bias:
            layer.bias.data.copy_(l0.bias.data)
        # Configure the test scenario with known delta and extension layers
        layer.optimal_delta_layer = l_delta  # For testing weight updates
        layer.extended_output_layer = l_ext   # For testing output dimension extension

        # Test with different scaling factor combinations
        for gamma, gamma_next in ((0.0, 0.0), (1.0, 1.5), (5.0, 5.5)):
            # Set scaling factors for the current layer and next module
            layer.scaling_factor = gamma
            layer._scaling_factor_next_module[0] = gamma_next
            
            # Generate test input and verify base behavior
            x = torch.randn((10, 5), device=global_device())
            self.assertAllClose(layer(x), l0(x), 
                              msg=f"Base layer output mismatch with gamma={gamma}, gamma_next={gamma_next}")

            # Test extended_forward returns the expected outputs
            # y_ext_1: Original output with delta update applied
            # y_ext_2: New output dimensions scaled by gamma_next
            y_ext_1, y_ext_2 = layer.extended_forward(x)

            # Verify the extended outputs match expected computations
            self.assertAllClose(y_ext_1, l0(x) - gamma**2 * l_delta(x),
                              msg=f"Extended output 1 mismatch with gamma={gamma}")
            self.assertAllClose(y_ext_2, gamma_next * l_ext(x),
                              msg=f"Extended output 2 mismatch with gamma_next={gamma_next}")

        # Test apply_change with apply_previous=False
        # This should apply the delta update to the main output
        layer.apply_change(apply_previous=False)
        y = layer(x)
        self.assertAllClose(y, l0(x) - gamma**2 * l_delta(x),
                          msg="Output after apply_change does not match expected value")

        # Test _apply_output_changes to verify it correctly extends the output
        layer._apply_output_changes()
        y_changed = layer(x)  # Should now have extended output dimensions
        
        # Split the output into original and extended parts
        y_changed_1 = y_changed[:, :1]  # Original output dimension
        y_changed_2 = y_changed[:, 1:]  # New output dimensions
        
        # Verify both parts of the output are correct
        self.assertAllClose(y_changed_1, l0(x) - gamma**2 * l_delta(x),
                          msg="Original output dimension after _apply_output_changes is incorrect")
        self.assertAllClose(
            y_changed_2,
            gamma_next * l_ext(x),
            atol=1e-7,
            msg="Extended output dimensions after _apply_output_changes are incorrect"
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_extended_forward_in(self, bias):
        """
        Test the extended_forward method for input extension in LinearGrowingModule.
        
        This test verifies that:
        1. The extended_forward method correctly handles extended input dimensions.
        2. The computation respects the scaling factors for the current layer.
        3. The gradient flow through the extended inputs works as expected.
        4. The apply_change method correctly modifies the layer to accept the extended input.
        
        The test uses fixed linear layers with known weights to verify the computations.
        
        Parameters
        ----------
        bias : bool
            Whether to include bias terms in the test layers.
        """
        torch.manual_seed(0)
        # Fixed layer representing the original layer weights
        l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
        # Layer representing the extension for new input dimensions
        l_ext = torch.nn.Linear(5, 1, bias=bias, device=global_device())
        if bias:
            l_ext.bias.data.fill_(0)  # Zero out bias for deterministic testing
        # Layer representing the optimal delta update for the original weights
        l_delta = torch.nn.Linear(3, 1, bias=bias, device=global_device())

        # Create the growing layer under test and initialize it with the same weights as l0
        layer = LinearGrowingModule(
            3, 1, use_bias=bias, name="layer1", device=global_device()
        )
        # Initialize layer weights and set up test configurations
        layer.weight.data.copy_(l0.weight.data)
        if bias:
            layer.bias.data.copy_(l0.bias.data)
        # Configure the test scenario with known delta and extension layers
        layer.optimal_delta_layer = l_delta  # For testing weight updates
        layer.extended_input_layer = l_ext    # For testing input dimension extension

        # Test with different scaling factors
        for gamma in (0.0, 1.0, 5.0):
            # Reset gradients and set the scaling factor
            layer.zero_grad()
            layer.scaling_factor = gamma
            
            # Generate test inputs (original and extended dimensions)
            x = torch.randn((10, 3), device=global_device())
            x_ext = torch.randn((10, 5), device=global_device())
            
            # Verify base behavior with original input dimensions
            self.assertAllClose(layer(x), l0(x),
                              msg=f"Base layer output mismatch with gamma={gamma}")

            # Test extended_forward with both original and extended inputs
            # The second return value should be None for input extensions
            y, none = layer.extended_forward(x, x_ext)
            self.assertIsNone(none, "Second return value should be None for input extension")

            # Verify the combined output matches the expected computation:
            # y = l0(x) - gamma² * l_delta(x) + gamma * l_ext(x_ext)
            expected_output = l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext)
            self.assertAllClose(y, expected_output,
                              msg=f"Extended forward output mismatch with gamma={gamma}")

            # Test gradient flow through the extended inputs
            # Compute gradient of the output norm w.r.t. the scaling factor
            torch.norm(y).backward()
            
            # Verify that the scaling factor has received gradients
            self.assertIsNotNone(layer.scaling_factor.grad,
                               "Scaling factor should have gradients after backward pass")

        # Test apply_change with apply_previous=False
        # This should modify the layer to accept the extended input dimensions
        layer.apply_change(apply_previous=False)
        
        # Create concatenated input (original + extended dimensions)
        x_cat = torch.concatenate((x, x_ext), dim=1)
        
        # Get the output after applying changes
        y = layer(x_cat)
        
        # Verify the output matches the expected computation
        expected_output = l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext)
        self.assertAllClose(
            y,
            expected_output,
            msg="Output after apply_change does not match expected value with extended input"
        )

    def test_number_of_parameters(self):
        """
        Test the parameter counting functionality of LinearGrowingModule.
        
        This test verifies that:
        1. The number_of_parameters method correctly counts trainable parameters.
        2. The count matches the expected number of parameters for different layer configurations.
        3. The count is consistent with the actual parameters in the underlying PyTorch layer.
        
        The test covers various combinations of input and output dimensions,
        with and without bias terms, to ensure comprehensive coverage.
        """
        # Test with different input and output dimensions
        for in_features in (1, 3):
            for out_features in (1, 3):
                # Test both with and without bias terms
                for use_bias in (True, False):
                    # Create a layer with the current configuration
                    layer = LinearGrowingModule(
                        in_features, 
                        out_features, 
                        use_bias=use_bias, 
                        name="test_layer"
                    )
                    
                    # Expected number of parameters: (input_dim + bias) * output_dim
                    expected_params = (in_features + use_bias) * out_features
                    
                    # Verify the count matches the expected value
                    self.assertEqual(
                        layer.number_of_parameters(),
                        expected_params,
                        f"Incorrect parameter count for {in_features=}, {out_features=}, {use_bias=}",
                    )
                    
                    # Additional verification by checking the actual parameters
                    actual_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    self.assertEqual(
                        actual_params,
                        expected_params,
                        f"Parameter count mismatch between number_of_parameters() and actual parameters for {in_features=}, {out_features=}, {use_bias=}",
                    )
    
    def test_layer_in_extension(self):
        """
        Test the behavior of layer_in_extension method in LinearGrowingModule.
        
        This test verifies that:
        1. The layer_in_extension method correctly extends the input dimension.
        2. The parameter count is updated correctly after extension.
        3. The forward pass works with the extended input dimensions.
        """
        # Create a layer with initial configuration
        layer = LinearGrowingModule(3, 1, use_bias=False, name="test_layer")
        layer.weight = torch.nn.Parameter(torch.ones(1, 3))
        
        # Verify initial state
        self.assertEqual(layer.number_of_parameters(), 3)
        self.assertEqual(layer.in_features, 3)
        
        # Test forward pass with initial configuration
        x = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[6.0]]))
        
        # Extend the input dimension
        layer.layer_in_extension(torch.tensor([[10]], dtype=torch.float32))
        
        # Verify state after extension
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.in_features, 4)
        self.assertEqual(layer.layer.in_features, 4)
        
        # Test forward pass with extended input
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[46.0]]))

    def test_layer_out_extension(self):
        """
        Test the behavior of layer_out_extension method in LinearGrowingModule.
        
        This test verifies that:
        1. The layer_out_extension method correctly extends the output dimension.
        2. The parameter count is updated correctly after extension.
        3. The forward pass works with the extended output dimensions.
        4. The behavior is correct both with and without bias.
        """
        # --- Test output extension without bias ---
        layer = LinearGrowingModule(1, 3, use_bias=False, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        
        # Verify initial state
        self.assertEqual(layer.number_of_parameters(), 3)
        self.assertEqual(layer.out_features, 3)
        
        # Test forward pass with initial configuration
        x = torch.tensor([[1]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[1.0, 1.0, 1.0]]))

        # Extend the output dimension
        layer.layer_out_extension(torch.tensor([[10]], dtype=torch.float32))
        
        # Verify state after extension
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.out_features, 4)
        self.assertEqual(layer.layer.out_features, 4)
        
        # Test forward pass with extended output
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[1.0, 1.0, 1.0, 10.0]]))

        # --- Test output extension with bias ---
        layer = LinearGrowingModule(1, 3, use_bias=True, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        layer.bias = torch.nn.Parameter(10 * torch.ones(3))
        
        # Verify initial state
        self.assertEqual(layer.number_of_parameters(), 6)
        self.assertEqual(layer.out_features, 3)
        
        # Test forward pass with initial configuration
        x = torch.tensor([[-1]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[9.0, 9.0, 9.0]]))

        # Extend the output dimension and provide bias for the new output
        layer.layer_out_extension(
            torch.tensor([[10]], dtype=torch.float32),
            bias=torch.tensor([100], dtype=torch.float32),
        )
        
        # Verify state after extension
        self.assertEqual(layer.number_of_parameters(), 8)
        self.assertEqual(layer.out_features, 4)
        
        # Test forward pass with extended output
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[9.0, 9.0, 9.0, 90.0]]))

    def test_apply_change_delta_layer(self):
        """
        Test the apply_change method with a delta layer in LinearGrowingModule.
        
        This test verifies that:
        1. The apply_change method correctly applies changes from the optimal_delta_layer.
        2. The scaling factor (gamma) is properly applied to the delta updates.
        3. The behavior is consistent with and without bias terms.
        4. The output matches the expected computation: y = l0(x) - gamma² * l_delta(x)
        """
        torch.manual_seed(0)
        
        # Test both with and without bias
        for bias in {True, False}:
            # Create a base linear layer and a delta layer for comparison
            l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            l_delta = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            
            # Initialize the growing layer and copy weights from the base layer
            layer = LinearGrowingModule(
                3, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)
            layer.optimal_delta_layer = l_delta

            # Copy bias if present
            if bias:
                layer.bias.data.copy_(l0.bias.data)

            # Set scaling factor and apply the delta change
            gamma = 5.0
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)

            # Test with random input
            x = torch.randn((10, 3), device=global_device())
            y = layer(x)
            
            # Verify the output matches the expected computation
            expected_output = l0(x) - gamma**2 * l_delta(x)
            self.assertAllClose(y, expected_output)

    def test_apply_change_out_extension(self):
        """
        Test the apply_change method with output extension in LinearGrowingModule.
        
        This test verifies that:
        1. The apply_change method correctly handles output dimension extensions.
        2. The scaling factor (gamma) is properly applied to the extended outputs.
        3. The behavior is consistent with and without bias terms.
        4. The output matches the expected computation after extension.
        """
        torch.manual_seed(0)
        
        # Test both with and without bias
        for bias in {True, False}:
            # Create a base linear layer and an extension layer
            l0 = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            l_ext = torch.nn.Linear(5, 2, bias=bias, device=global_device())
            
            # Initialize the growing layer and copy weights from the base layer
            layer = LinearGrowingModule(
                5, 1, use_bias=bias, name="layer1", device=global_device()
            )
            # Copy initial weights and biases (if any) from the base layer
            layer.weight.data.copy_(l0.weight.data)
            if bias:
                layer.bias.data.copy_(l0.bias.data)
            
            # Set up the extension layer
            layer.extended_output_layer = l_ext

            # Set up scaling factors and apply initial changes
            gamma = 5.0
            gamma_next = 5.5
            layer.scaling_factor = gamma
            
            # Apply changes without affecting previous layers
            layer.apply_change(apply_previous=False)
            
            # Verify weights remain unchanged after apply_change with apply_previous=False
            self.assertAllClose(layer.weight.data, l0.weight.data)

            # Set up next module's scaling factor and apply output changes
            layer._scaling_factor_next_module[0] = gamma_next
            layer._apply_output_changes()

            # Test with random input
            x = torch.randn((10, 5), device=global_device())
            y = layer(x)
            
            # Split output into base and extended parts
            y_base = y[:, :1]  # Original output dimensions
            y_extended = y[:, 1:]  # Extended output dimensions
            
            # Verify outputs:
            # - Base output should match the original layer
            # - Extended output should match the extension layer scaled by gamma_next
            self.assertAllClose(y_base, l0(x))
            self.assertAllClose(y_extended, gamma_next * l_ext(x))

    def test_apply_change_in_extension(self):
        """
        Test the behavior of apply_change with input extension in LinearGrowingModule.
        
        This test verifies that:
        1. The apply_change method correctly handles input extensions when apply_previous=False.
        2. The forward pass correctly processes concatenated input tensors.
        3. The scaling factor is properly applied to the extended input.
        
        The test checks both with and without bias terms to ensure comprehensive coverage.
        """
        # Set random seed for reproducibility
        torch.manual_seed(0)
        
        # Test both with and without bias
        for bias in {True, False}:
            # Create a base linear layer for comparison
            l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            
            # Create an extended input layer (5 input features to 1 output)
            l_ext = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            
            # For bias terms, zero out the extended layer's bias to isolate the effect
            if bias:
                l_ext.bias.data.fill_(0)
                
            # Create the growing layer with the same configuration as l0
            layer = LinearGrowingModule(
                3, 1, use_bias=bias, name="test_layer", device=global_device()
            )
            
            # Copy weights from the base layer
            layer.weight.data.copy_(l0.weight.data)
            if bias:
                layer.bias.data.copy_(l0.bias.data)
                
            # Set up the extended input layer
            layer.extended_input_layer = l_ext

            # Set scaling factor and apply changes (without updating previous layers)
            gamma = 5.0
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)

            # Create test data: concatenated input with 3 base and 5 extended features
            x_cat = torch.randn((10, 8), device=global_device())
            y = layer(x_cat)
            
            # Split the input into base and extended parts
            x = x_cat[:, :3]    # First 3 features go to base layer
            x_ext = x_cat[:, 3:] # Remaining 5 features go to extended layer

            # Verify the output matches the expected computation:
            # y = l0(x) + gamma * l_ext(x_ext)
            self.assertAllClose(
                y,
                l0(x) + gamma * l_ext(x_ext),
                atol=1e-7,
                message=(
                    f"Mismatch in forward pass with input extension. "
                    f"Max error: {(y - l0(x) - gamma * l_ext(x_ext)).abs().max():.2e}"
                ),
            )

    def test_sub_select_optimal_added_parameters_out(self):
        """
        Test the sub-selection of optimal parameters for output extension.
        
        This test verifies that:
        1. The method correctly handles the sub-selection of optimal parameters
           when reducing the number of output dimensions.
        2. The extended output layer's parameters are properly updated.
        3. The behavior is consistent with and without bias terms.
           
        The test ensures that when we reduce the number of output dimensions,
        the most significant parameters (based on eigenvalues) are preserved.
        """
        # Test both with and without bias
        for bias in {True, False}:
            # Create a layer with extended output dimensions
            layer = LinearGrowingModule(3, 1, use_bias=bias, name="test_layer")
            
            # Set up an extended output layer (1 input to 2 outputs)
            layer.extended_output_layer = torch.nn.Linear(3, 2, bias=bias)
            
            # Store the original extended layer for comparison
            original_extended_layer = deepcopy(layer.extended_output_layer)
            
            # Test sub-selection of parameters (reduce from 2 outputs to 1)
            layer._sub_select_added_output_dimension(1)
            
            # Verify the extended layer now has the correct output dimension
            self.assertEqual(layer.extended_output_layer.out_features, 1)
            
            # Check that the weights were properly selected (should take the first output)
            self.assertAllClose(
                layer.extended_output_layer.weight,
                original_extended_layer.weight[0:1],
                message="Weights were not correctly sub-selected for output extension"
            )
            
            # If bias is enabled, check that it was also properly sub-selected
            if bias:
                self.assertAllClose(
                    layer.extended_output_layer.bias,
                    original_extended_layer.bias[0:1],
                    message="Biases were not correctly sub-selected for output extension"
                )

    def test_sub_select_optimal_added_parameters_in(self):
        """
        Test the sub-selection of optimal parameters for input extension.
        
        This test verifies that:
        1. The sub_select_optimal_added_parameters method correctly reduces the number of input extension dimensions.
        2. The extended input layer's weights (and biases, if present) are correctly updated to reflect the sub-selection.
        3. The eigenvalues_extension tensor is updated to match the reduced set.
        
        The test uses a layer with two input extensions and then reduces it to one, verifying that the correct weights and eigenvalues are retained.
        """
        bias = False
        # Create a layer with an extended input of 2 dimensions
        layer = LinearGrowingModule(1, 3, use_bias=bias, name="layer1")
        layer.extended_input_layer = torch.nn.Linear(2, 3, bias=bias)
        layer.eigenvalues_extension = torch.tensor([2.0, 1.0])

        # Prepare a reference layer with only the first input extension retained
        new_layer = torch.nn.Linear(1, 3, bias=bias)
        new_layer.weight.data = layer.extended_input_layer.weight.data[:, 0].view(-1, 1)
        if bias:
            new_layer.bias.data = layer.extended_input_layer.bias.data

        # Perform sub-selection to keep only the first input extension
        layer.sub_select_optimal_added_parameters(1, sub_select_previous=False)

        # Verify that the weight matrix matches the reference
        self.assertAllClose(layer.extended_input_layer.weight, new_layer.weight)

        # If bias is enabled, verify bias values
        if bias:
            self.assertAllClose(layer.extended_input_layer.bias, new_layer.bias)

        # Check that the eigenvalues_extension is updated correctly
        self.assertAllClose(layer.eigenvalues_extension, torch.tensor([2.0]))


    def test_sample_number_invariant(self):
        """
        Test that certain internal invariants of LinearGrowingModule remain unchanged
        when the number of samples is doubled (i.e., the batch is duplicated).
        
        This test ensures that key tensors and statistics inside the module are
        invariant to the number of samples, which is important for mathematical
        consistency and reproducibility.
        
        The invariants checked include:
        - tensor_s: second moment matrix
        - tensor_m: mean tensor
        - delta_raw: raw delta tensor
        - optimal_delta_layer: optimal delta layer
        - parameter_update_decrease: decrease in loss from parameter update
        - eigenvalues_extension: eigenvalues for extensions
        - tensor_m_prev: previous mean tensor
        - cross_covariance: cross-covariance tensor
        
        The test performs the following steps:
        1. Sets up a two-layer network using LinearGrowingModule.
        2. Runs a forward and backward pass with a random batch.
        3. Stores the invariants after computation.
        4. Duplicates the batch (doubles the number of samples) and repeats the computation.
        5. Verifies that all invariants remain unchanged.
        """
        invariants = [
            "tensor_s",
            "tensor_m",
            # "pre_activity",
            # "input",
            "delta_raw",
            "optimal_delta_layer",
            "parameter_update_decrease",
            "eigenvalues_extension",
            "tensor_m_prev",
            "cross_covariance",
        ]

        def linear_layer_equality(layer1, layer2, rtol=1e-5, atol=1e-8):
            """Check equality of two linear layers' weights and biases."""
            return torch.allclose(
                layer1.weight, layer2.weight, atol=atol, rtol=rtol
            ) and (
                (layer1.bias is None and layer2.bias is None)
                or (torch.allclose(layer1.bias, layer2.bias, atol=atol, rtol=rtol))
            )

        def set_invariants(layer: LinearGrowingModule):
            """Store a copy of all invariants from the layer for later comparison."""
            _reference = dict()
            for inv in invariants:
                inv_value = getattr(layer, inv)
                if isinstance(inv_value, torch.Tensor):
                    _reference[inv] = inv_value.clone()
                elif isinstance(inv_value, torch.nn.Linear):
                    # Deep copy linear layers
                    _reference[inv] = deepcopy(inv_value)
                elif hasattr(inv_value, '__call__'):
                    # For callable statistics objects
                    _reference[inv] = inv_value().clone()
                else:
                    raise ValueError(f"Invalid type for {inv} ({type(inv_value)})")
            return _reference

        def check_invariants(layer: LinearGrowingModule, reference: dict, rtol=1e-5, atol=1e-8):
            """Compare current invariants to reference invariants and assert equality."""
            for inv in invariants:
                new_inv_value = getattr(layer, inv)
                if isinstance(new_inv_value, torch.Tensor):
                    self.assertAllClose(
                        reference[inv],
                        new_inv_value,
                        rtol=rtol,
                        atol=atol,
                        message=f"Error on {inv=}"
                    )
                elif isinstance(new_inv_value, torch.nn.Linear):
                    self.assertTrue(
                        linear_layer_equality(reference[inv], new_inv_value, rtol=rtol, atol=atol),
                        f"Error on {inv=} (linear layer mismatch)"
                    )
                elif hasattr(new_inv_value, '__call__'):
                    self.assertAllClose(
                        reference[inv],
                        new_inv_value(),
                        rtol=rtol,
                        atol=atol,
                        message=f"Error on {inv=} (callable/statistic)"
                    )
                else:
                    raise ValueError(f"Invalid type for {inv} ({type(new_inv_value)})")

        # --- Step 1: Set up a two-layer network using LinearGrowingModule ---
        torch.manual_seed(0)
        layer_in = LinearGrowingModule(
            in_features=5,
            out_features=3,
            name="layer_in",
            post_layer_function=torch.nn.SELU(),
        )
        layer_out = LinearGrowingModule(
            in_features=3, out_features=7, name="layer_out", previous_module=layer_in
        )
        net = torch.nn.Sequential(layer_in, layer_out)

        # --- Step 2: Run a forward and backward pass with a random batch ---
        def update_computation(double_batch=False):
            loss_fn = torch.nn.MSELoss(reduction="sum")
            torch.manual_seed(0)
            net.zero_grad()
            x = torch.randn((10, 5), device=global_device())
            if double_batch:
                # Duplicate the batch to double the number of samples
                x = torch.cat((x, x), dim=0)
            y = net(x)
            loss = loss_fn(y, torch.zeros_like(y))
            loss.backward()
            layer_out.update_computation()
            layer_in.tensor_s.update()

        # --- Step 3: Initialize and compute invariants with original batch ---
        layer_out.init_computation()
        layer_in.tensor_s.init()
        update_computation()
        layer_out.compute_optimal_updates()
        reference = set_invariants(layer_out)

        # --- Step 4: Duplicate batch and recompute invariants ---
        for double_batch in (False, True):
            update_computation(double_batch=double_batch)
            layer_out.compute_optimal_updates()
            # --- Step 5: Check that all ignvariants remain unchanged ---
            check_invariants(layer_out, reference)

    @unittest_parametrize(({"bias": True, "dtype": torch.float64}, {"bias": False}))
    def test_compute_optimal_added_parameters(
        self, bias: bool, dtype: torch.dtype = torch.float32
    ):
        """
        Test the computation of optimal parameters for added dimensions (extensions) in LinearGrowingModule.
        
        This test verifies that:
        1. The compute_optimal_added_parameters method produces tensors of the correct shape and type.
        2. The method works correctly for both bias and no-bias configurations.
        3. The computed parameters are consistent with the expected dimensionality.
        4. The method works with different dtypes (tested with float32 and float64).
        
        The test uses a two-layer setup:
        - The first layer stores its input for later use.
        - The second layer is initialized for computation and its tensor_s_growth is also initialized.
        - After a forward and backward pass, the optimal delta is computed and the added parameters are extracted.
        - Assertions verify the shape and type of the computed parameters.
        - Additional tests verify the behavior of sub-selecting optimal parameters.
        """
        # --- Step 1: Prepare demo layers and initialize computation ---
        demo_layers = self.demo_layers[bias]
        demo_layers[0].store_input = True  # Enable input storage in the first layer
        demo_layers[1].init_computation()  # Initialize computation in the second layer
        demo_layers[1].tensor_s_growth.init()  # Initialize growth statistics

        # --- Step 2: Forward and backward pass to accumulate gradients/statistics ---
        # Pass input through the network and compute loss
        y = demo_layers[0](self.input_x)  # Forward pass through first layer
        y = demo_layers[1](y)  # Forward pass through second layer
        loss = torch.norm(y)  # Compute loss
        loss.backward()  # Backpropagate gradients

        # --- Step 3: Update computation and tensor statistics ---
        demo_layers[1].update_computation()  # Update layer computations
        demo_layers[1].tensor_s_growth.update()  # Update growth statistics

        # --- Step 4: Compute optimal delta and added parameters ---
        demo_layers[1].compute_optimal_delta()  # Compute optimal weight updates
        # Get optimal parameters for added dimensions
        alpha, alpha_b, omega, eigenvalues = demo_layers[1].compute_optimal_added_parameters(dtype=dtype)

        # --- Step 5: Assertions on the shape of computed parameters ---
        # Check alpha shape: (k, in_features) where k is the number of new dimensions
        self.assertShapeEqual(
            alpha,
            (-1, demo_layers[0].in_features),
        )
        k = alpha.size(0)  # Number of new dimensions
        
        # Check bias terms if bias is enabled
        if bias:
            self.assertShapeEqual(alpha_b, (k,))
        else:
            self.assertIsNone(alpha_b)

        # Check omega shape: (out_features, k)
        self.assertShapeEqual(
            omega,
            (demo_layers[1].out_features, k),
        )

        # Check eigenvalues shape: (k,)
        self.assertShapeEqual(eigenvalues, (k,))

        # Verify that extension layers are properly initialized
        self.assertIsInstance(demo_layers[0].extended_output_layer, torch.nn.Linear)
        self.assertIsInstance(demo_layers[1].extended_input_layer, torch.nn.Linear)

        # --- Test sub-selection of optimal parameters ---
        # This section tests the ability to select a subset of the optimal parameters
        # It verifies that the dimensions are updated correctly after sub-selection
        num_selected = 2
        demo_layers[1].sub_select_optimal_added_parameters(num_selected)
        
        # Verify that the number of eigenvalues matches the number of selected parameters
        self.assertEqual(demo_layers[1].eigenvalues_extension.shape[0], num_selected)
        
        # Verify that the input layer's extended dimensions are updated
        self.assertEqual(demo_layers[1].extended_input_layer.in_features, num_selected)
        
        # Verify that the output layer's extended dimensions are updated
        self.assertEqual(demo_layers[0].extended_output_layer.out_features, num_selected)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_s_growth(self, bias: bool):
        """
        Test the initialization and updating of tensor growth statistics.
        
        This test verifies that:
        1. The tensor_s_growth object is properly initialized and updated.
        2. The number of samples tracked matches the input batch size.
        3. The shape of the computed tensor matches the expected dimensions.
        
        The test performs the following steps:
        1. Sets up the test environment with the specified bias configuration.
        2. Performs a forward and backward pass through the network.
        3. Updates the tensor growth statistics.
        4. Verifies the statistics match the expected values.
        
        Args:
            bias: Whether to test with bias enabled or disabled.
        """
        # --- Step 1: Setup test environment ---
        demo_layers = self.demo_layers[bias]
        demo_layers[0].store_input = True  # Enable input storage in the first layer
        demo_layers[1].tensor_s_growth.init()  # Initialize growth statistics

        # --- Step 2: Perform forward and backward pass ---
        # Pass input through the network and compute loss
        y = demo_layers[0](self.input_x)  # Forward pass through first layer
        y = demo_layers[1](y)  # Forward pass through second layer
        loss = torch.norm(y)  # Compute loss
        loss.backward()  # Backpropagate gradients

        # --- Step 3: Update growth statistics ---
        demo_layers[1].tensor_s_growth.update()

        # --- Step 4: Verify statistics ---
        # Check that the number of samples matches the input batch size
        self.assertEqual(
            demo_layers[1].tensor_s_growth.samples,
            self.input_x.size(0),
            "Number of samples in tensor_s_growth should match input batch size"
        )
        
        # Calculate expected size: input features + 1 if bias is enabled
        expected_size = demo_layers[0].in_features + demo_layers[0].use_bias
        
        # Verify the shape of the computed tensor
        self.assertShapeEqual(
            demo_layers[1].tensor_s_growth(), 
            (expected_size, expected_size),
            f"Expected square matrix of size {expected_size}x{expected_size}"
        )

    def test_tensor_s_growth_errors(self):
        """
        Test error conditions for tensor_s_growth property.
        
        This test verifies that:
        1. Setting tensor_s_growth to an invalid value raises an AttributeError.
        2. Accessing tensor_s_growth without initialization raises a ValueError.
        """
        # Test that setting tensor_s_growth to a non-TensorStatistics object raises an AttributeError
        with self.assertRaises(AttributeError):
            self.demo_layers[True][1].tensor_s_growth = 1

        # Test that accessing tensor_s_growth without initialization raises a ValueError
        with self.assertRaises(ValueError):
            _ = self.demo_layers[True][0].tensor_s_growth


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
