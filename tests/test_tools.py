import contextlib
import io
import unittest.mock
from unittest import main, mock

import torch

from gromo.utils.tools import (
    apply_border_effect_on_unfolded,
    compute_mask_tensor_t,
    compute_optimal_added_parameters,
    compute_output_shape_conv,
    create_bordering_effect_convolution,
    optimal_delta,
    sqrt_inverse_matrix_semi_positive,
)
from tests.torch_unittest import TorchTestCase

from .unittest_tools import unittest_parametrize


test_input_shapes = [
    {"h": 4, "w": 4},
    {"h": 4, "w": 5},
    {"h": 5, "w": 4},
    {"h": 5, "w": 5},
]


class TestTools(TorchTestCase):
    def test_sqrt_inverse_matrix_semi_positive(self):
        matrix = 9 * torch.eye(5)
        sqrt_inverse_matrix = sqrt_inverse_matrix_semi_positive(matrix)
        self.assertAllClose(sqrt_inverse_matrix @ sqrt_inverse_matrix, matrix.inverse())

    def test_random_sqrt_inverse_matrix_semi_positive(self):
        """
        Test the sqrt_inverse_matrix_semi_positive on random X^T X matrice
        with X in (5, 3)
        Test the function on cpu and cuda if available.
        """
        if torch.cuda.is_available():
            devices = (torch.device("cuda"), torch.device("cpu"))
        else:
            devices = (torch.device("cpu"),)
            print("Warning: No cuda device available therefore only testing on cpu")
        for device in devices:
            matrix = torch.randn(5, 3, dtype=torch.float64, device=device)
            matrix = matrix.t() @ matrix
            sqrt_inverse_matrix = sqrt_inverse_matrix_semi_positive(
                matrix, threshold=1e-7
            )
            reconstructed_inverse = sqrt_inverse_matrix @ sqrt_inverse_matrix
            if torch.abs(torch.linalg.det(matrix)) > 1e-5:
                self.assertAllClose(
                    reconstructed_inverse,
                    torch.linalg.inv(matrix),
                    message=f"Error with device {device}",
                )

    @unittest_parametrize(
        (
            {"dtype": torch.float32},
            {"dtype": torch.float64},
        )
    )
    def test_sqrt_inverse_matrix_semi_positive_shrinkage(self, dtype=torch.float32):
        """
        Test that the sqrt_inverse_matrix_semi_positive function applies shrinkage correctly.
        """
        matrix = torch.zeros(5, 5, dtype=dtype)
        with mock.patch(
            "gromo.utils.tools.torch.linalg.eigh",
            side_effect=[
                torch.linalg.LinAlgError("forced"),
                torch.linalg.eigh(
                    torch.eye(5, dtype=dtype) * torch.finfo(dtype).resolution
                ),
            ],
        ):
            with self.assertWarns(RuntimeWarning):
                sqrt_inverse_matrix = sqrt_inverse_matrix_semi_positive(
                    matrix, threshold=torch.finfo(dtype).resolution
                )
            self.assertAllClose(
                sqrt_inverse_matrix,
                torch.zeros_like(matrix, dtype=dtype),
                message="Shrinkage not applied correctly",
            )

    def test_compute_output_shape_conv(self):
        """
        Test the compute_output_shape_conv function
        with various inputs shapes and conv kernel sizes.
        """
        kernel_sizes = [1, 2, 3, 5, 7]
        input_shapes = [2, 5, 11, 41]
        for k_h in kernel_sizes:
            for k_w in kernel_sizes:
                conv = torch.nn.Conv2d(1, 1, (k_h, k_w))
                for h in input_shapes:
                    if k_h <= h:
                        for w in input_shapes:
                            if k_w <= w:
                                with self.subTest(h=h, w=w, k_h=k_h, k_w=k_w):
                                    out_shape = conv(
                                        torch.empty(
                                            (1, conv.in_channels, h, w),
                                            device=conv.weight.device,
                                        )
                                    ).shape[2:]
                                    predicted_out_shape = compute_output_shape_conv(
                                        (h, w), conv
                                    )
                                    self.assertEqual(
                                        out_shape,
                                        predicted_out_shape,
                                        f"Error with {h=}, {w=}, {k_h=}, {k_w=}",
                                    )

    @unittest_parametrize(test_input_shapes)
    def test_compute_mask_tensor_t_without_bias(self, h, w):
        """
        Test the compute_mask_tensor_t function.
        Check that it respects its property.
        """
        for k_h in (1, 2, 3):
            for k_w in (1, 2, 3):
                with self.subTest(k_h=k_h, k_w=k_w):
                    conv = torch.nn.Conv2d(2, 3, (k_h, k_w), bias=False)
                    # TODO: add test for the case with bias activated
                    conv_kernel_flatten = conv.weight.data.flatten(start_dim=2)
                    mask = compute_mask_tensor_t((h, w), conv)
                    x_input = torch.randn(1, 2, h, w)
                    x_input_flatten = x_input.flatten(start_dim=2)
                    y_th = conv(x_input).flatten(start_dim=2)
                    y_via_mask = torch.einsum(
                        "cds, jsp, idp -> icj",
                        conv_kernel_flatten,
                        mask,
                        x_input_flatten,
                    )
                    self.assertAllClose(
                        y_th,
                        y_via_mask,
                        atol=1e-6,
                        message=f"Error with {h=}, {w=}, {k_h=}, {k_w=} ",
                    )

    @unittest_parametrize(test_input_shapes)
    def test_compute_mask_tensor_t_with_bias(self, h, w):
        """
        Test the compute_mask_tensor_t function with bias activated.
        Check that it respects its property.
        """
        for k_h in (1, 2, 3):
            for k_w in (1, 2, 3):
                conv = torch.nn.Conv2d(2, 3, (k_h, k_w), bias=True)
                conv_kernel_flatten = conv.weight.data.flatten(start_dim=2)
                with self.subTest(k_h=k_h, k_w=k_w):
                    mask = compute_mask_tensor_t((h, w), conv)
                    x_input = torch.randn(1, 2, h, w)
                    x_input_flatten = x_input.flatten(start_dim=2)
                    y_th = conv(x_input).flatten(start_dim=2)
                    y_via_mask = torch.einsum(
                        "cds, jsp, idp -> icj",
                        conv_kernel_flatten,
                        mask,
                        x_input_flatten,
                    )
                    self.assertTrue(conv.bias is not None, "Bias should be activated")
                    assert conv.bias is not None
                    y_via_mask += conv.bias.data.view(1, -1, 1)
                    self.assertAllClose(
                        y_th,
                        y_via_mask,
                        atol=1e-6,
                        message=f"Error with {h=}, {w=}, {k_h=}, {k_w=} ",
                    )

    def test_apply_border_effect_on_unfolded_typing(self, bias: bool = False):
        conv1 = torch.nn.Conv2d(2, 3, (3, 5), padding=(1, 2), bias=bias)
        conv2 = torch.nn.Conv2d(3, 4, (3, 5), padding=(1, 2), bias=False)
        x = torch.randn(11, 2, 13, 17)
        unfolded_x = torch.nn.functional.unfold(
            x,
            kernel_size=conv1.kernel_size,
            padding=conv1.padding,  # type: ignore
            stride=conv1.stride,
            dilation=conv1.dilation,
        )
        # everything is ok
        _ = apply_border_effect_on_unfolded(
            unfolded_x,
            (x.shape[2], x.shape[3]),
            border_effect_conv=conv2,
        )
        unfolded_x = None
        with self.assertRaises(TypeError):
            _ = apply_border_effect_on_unfolded(
                unfolded_x,  # type: ignore
                (x.shape[2], x.shape[3]),
                border_effect_conv=conv2,
            )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_apply_border_effect_on_unfolded(self, bias: bool):
        for kh in (1, 2, 3):
            for kw in (1, 2, 3):
                for ph in (0, 1, 2):
                    for pw in (0, 1, 2):
                        with self.subTest(kh=kh, kw=kw, ph=ph, pw=pw):
                            self._test_apply_border_effect_on_unfolded(
                                bias=bias, kh=kh, kw=kw, ph=ph, pw=pw
                            )

    def _test_apply_border_effect_on_unfolded(
        self, bias: bool = True, kh: int = 3, kw: int = 3, ph: int = 1, pw: int = 1
    ):
        # kh, kw = 3, 1
        # ph, pw = 0, 0
        conv1 = torch.nn.Conv2d(2, 3, (3, 5), padding=(1, 2), bias=bias)
        x = torch.randn(11, 2, 13, 17)
        unfolded_x = torch.nn.functional.unfold(
            x,
            kernel_size=conv1.kernel_size,
            padding=conv1.padding,  # type: ignore
            stride=conv1.stride,
            dilation=conv1.dilation,
        )
        if bias:
            unfolded_x = torch.cat(
                [unfolded_x, torch.ones_like(unfolded_x[:, :1])], dim=1
            )
        # kh, kw, ph, pw = torch.randint(low=0, high=4, size=(4,))
        # kh, kw, ph, pw = 3, 3, 2, 2
        conv2 = torch.nn.Conv2d(3, 4, (kh, kw), padding=(ph, pw))

        bordered_unfolded_x = apply_border_effect_on_unfolded(
            unfolded_x,
            (x.shape[2], x.shape[3]),
            border_effect_conv=conv2,
        )
        self.assertShapeEqual(
            bordered_unfolded_x,
            (
                x.shape[0],
                conv1.in_channels * conv1.kernel_size[0] * conv1.kernel_size[1] + bias,
                None,
            ),
        )  # None because we don't check the size of the last dimension

        conv2 = torch.nn.Conv2d(3, 4, (kh, kw), padding=(ph, pw), bias=False)
        # We are sure that conv2 has no bias as it represents an expansion
        new_kernel = torch.zeros_like(conv2.weight)
        new_kernel[:, :, kh // 2 : kh // 2 + 1, kw // 2 : kw // 2 + 1] = (
            conv2.weight[:, :, kh // 2, kw // 2].unsqueeze(-1).unsqueeze(-1)
        )
        conv2.weight = torch.nn.Parameter(new_kernel)

        y_th = conv1(x)
        z_th = conv2(y_th)
        self.assertShapeEqual(
            bordered_unfolded_x,
            (
                x.shape[0],
                conv1.in_channels * conv1.kernel_size[0] * conv1.kernel_size[1] + bias,
                z_th.shape[2] * z_th.shape[3],
            ),
        )
        # self.assertAllClose(
        #     bordered_unfolded_x,
        #     unfolded_x,
        # )
        w_c1 = conv1.weight.flatten(start_dim=1)
        if bias:
            assert conv1.bias is not None
            w_c1 = torch.cat([w_c1, conv1.bias[:, None]], dim=1)

        y_via_mask = torch.einsum(
            "iax, ca -> icx",
            bordered_unfolded_x,
            w_c1,
        )
        # self.assertAllClose(
        #     y_th.flatten(start_dim=2),
        #     y_via_mask,
        #     atol=1e-6,
        #     message=f"Error on y.",
        # )
        self.assertShapeEqual(
            y_via_mask, (x.shape[0], conv1.out_channels, z_th.shape[2] * z_th.shape[3])
        )

        z_via_mask = torch.einsum(
            "iax, ca -> icx",
            y_via_mask,
            conv2.weight[:, :, kh // 2, kw // 2],
        )

        self.assertShapeEqual(
            z_via_mask, (x.shape[0], conv2.out_channels, z_th.shape[2] * z_th.shape[3])
        )
        z_via_mask = z_via_mask.reshape(
            z_via_mask.shape[0], z_via_mask.shape[1], z_th.shape[2], z_th.shape[3]
        )

        self.assertAllClose(
            z_th,
            z_via_mask,
            atol=1e-6,
            message=f"Error: {torch.abs(z_th - z_via_mask).max().item():.2e}",
        )

    def test_compute_optimal_added_parameters(self):
        """Test compute_optimal_added_parameters function comprehensively"""
        torch.manual_seed(42)  # For reproducible tests

        # Test case 1: Simple case with known solution
        matrix_s = torch.eye(3) * 2.0  # Simple diagonal matrix
        matrix_n = torch.randn(3, 2)

        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, numerical_threshold=1e-10, statistical_threshold=1e-6
        )

        # Check output shapes
        self.assertEqual(alpha.shape[1], matrix_s.shape[0])  # alpha: (k, s)
        self.assertEqual(omega.shape[0], matrix_n.shape[1])  # omega: (t, k)
        self.assertEqual(eigenvalues.shape[0], alpha.shape[0])  # lambda: (k,)
        self.assertEqual(alpha.shape[0], omega.shape[1])  # k dimensions match

        # Test case 2: Symmetric positive definite matrix
        matrix_s = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        matrix_n = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, numerical_threshold=1e-12, statistical_threshold=1e-8
        )

        # Eigenvalues should be positive (since we're dealing with SVD)
        self.assertTrue(torch.all(eigenvalues >= 0))

        # Test case 3: With maximum_added_neurons constraint
        matrix_s = torch.eye(4) * 3.0
        matrix_n = torch.randn(4, 5)
        max_neurons = 2

        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, maximum_added_neurons=max_neurons
        )

        # Should respect the maximum constraint
        self.assertLessEqual(alpha.shape[0], max_neurons)
        self.assertLessEqual(omega.shape[1], max_neurons)
        self.assertLessEqual(eigenvalues.shape[0], max_neurons)

        # Test case 4: Non-symmetric matrix (should trigger warning)
        matrix_s_nonsym = torch.tensor([[1.0, 0.5], [0.3, 1.0]])
        matrix_n = torch.tensor([[1.0], [1.0]])

        with self.assertWarns(UserWarning):  # The input matrix S is not symmetric
            alpha, omega, eigenvalues = compute_optimal_added_parameters(
                matrix_s_nonsym, matrix_n
            )

        # Should still produce valid outputs
        self.assertEqual(alpha.shape[1], matrix_s_nonsym.shape[0])
        self.assertEqual(omega.shape[0], matrix_n.shape[1])

        # Test case 5: Edge case with very small singular values
        matrix_s = torch.eye(2) * 1e-8  # Very small values
        matrix_n = torch.randn(2, 2)

        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, numerical_threshold=1e-10, statistical_threshold=1e-6
        )

        # Should handle small values gracefully
        self.assertFalse(torch.any(torch.isnan(alpha)))
        self.assertFalse(torch.any(torch.isnan(omega)))
        self.assertFalse(torch.any(torch.isnan(eigenvalues)))

    def test_compute_optimal_added_parameters_error_cases(self):
        """Test error handling in compute_optimal_added_parameters"""

        # Test incompatible matrix shapes
        matrix_s = torch.eye(3)
        matrix_n = torch.randn(2, 4)  # Wrong shape

        with self.assertRaises(AssertionError):
            compute_optimal_added_parameters(matrix_s, matrix_n)

        # Test non-square S matrix
        matrix_s = torch.randn(3, 4)  # Not square
        matrix_n = torch.randn(4, 2)

        with self.assertRaises(AssertionError):
            compute_optimal_added_parameters(matrix_s, matrix_n)

    def test_create_bordering_effect_convolution(self):
        """Test create_bordering_effect_convolution function"""
        # Test basic functionality
        channels = 12  # 2 * 3 * 2 (in_channels * kernel_h * kernel_w)
        conv = torch.nn.Conv2d(2, 4, (3, 2), padding=(1, 0))

        identity_conv = create_bordering_effect_convolution(channels, conv)

        # Check properties
        self.assertEqual(identity_conv.in_channels, channels)
        self.assertEqual(identity_conv.out_channels, channels)
        self.assertEqual(identity_conv.groups, channels)
        self.assertEqual(identity_conv.kernel_size, conv.kernel_size)
        self.assertEqual(identity_conv.padding, conv.padding)
        self.assertEqual(identity_conv.stride, conv.stride)
        self.assertEqual(identity_conv.dilation, conv.dilation)
        self.assertIsNone(identity_conv.bias)

        # Check weight initialization (should be identity in center)
        mid_h, mid_w = conv.kernel_size[0] // 2, conv.kernel_size[1] // 2
        center_weights = identity_conv.weight.data[:, 0, mid_h, mid_w]
        self.assertTrue(torch.allclose(center_weights, torch.ones(channels)))

        # Test error cases
        with self.assertRaises(ValueError):
            create_bordering_effect_convolution(-1, conv)  # Invalid channels

        with self.assertRaises(ValueError):
            create_bordering_effect_convolution(0, conv)  # Invalid channels

        with self.assertRaises(TypeError):
            create_bordering_effect_convolution(
                12,
                "not_a_conv",  # type: ignore
            )  # Wrong type

    def test_sqrt_inverse_matrix_semi_positive_preferred_linalg(self):
        """Test sqrt_inverse_matrix_semi_positive with `magama` preferred_linalg_library"""
        matrix = 4 * torch.eye(3)

        expected = sqrt_inverse_matrix_semi_positive(matrix)

        # Test with preferred_linalg_library set to "magma"
        if torch.cuda.is_available():
            matrix_cuda = matrix.cuda()
            torch.backends.cuda.preferred_linalg_library(
                "magma"
            )  # Set preferred library to magma
            result_magma = sqrt_inverse_matrix_semi_positive(matrix_cuda)
            self.assertIsNotNone(result_magma)
            self.assertEqual(result_magma.device.type, "cuda")
            torch.backends.cuda.preferred_linalg_library(
                "default"
            )  # Reset to default after test
            self.assertAllClose(
                result_magma.cpu(),
                expected,
                atol=1e-6,
                message="Error with magma preferred_linalg_library",
            )

    def test_compute_optimal_added_parameters_svd_error_handling(self):
        """Test SVD LinAlgError handling and debug output"""
        matrix_s = torch.eye(3) * 2.0
        matrix_n = torch.randn(3, 2)

        # Mock the SVD to trigger LinAlgError on first call, succeed on second
        # Create a proper successful result with correct values by computing SVD on the actual matrix_p
        # Replicate the computation of matrix_p as in compute_optimal_added_parameters
        matrix_s_inverse_sqrt = torch.linalg.inv(torch.linalg.cholesky(matrix_s))
        matrix_p = matrix_s_inverse_sqrt @ matrix_n
        u_real, s_real, vt_real = torch.linalg.svd(matrix_p, full_matrices=False)
        successful_result = (u_real, s_real, vt_real)

        # Capture stdout to verify debug prints
        captured_output = io.StringIO()

        with unittest.mock.patch("torch.linalg.svd") as mock_svd:
            mock_svd.side_effect = [
                torch.linalg.LinAlgError("Mocked SVD error"),  # First call fails
                successful_result,  # Second call succeeds
            ]

            with unittest.mock.patch("sys.stdout", captured_output):
                alpha, omega, eigenvalues = compute_optimal_added_parameters(
                    matrix_s, matrix_n
                )

            # Verify debug output was printed
            output = captured_output.getvalue()
            self.assertIn("Warning: An error occurred during the SVD computation", output)
            self.assertIn("matrix_s:", output)
            self.assertIn("matrix_n:", output)
            self.assertIn("matrix_s_inverse_sqrt:", output)
            self.assertIn("matrix_p:", output)

            # Verify the function still succeeded after retry
            self.assertIsNotNone(alpha)
            self.assertIsNotNone(omega)
            self.assertIsNotNone(eigenvalues)

            # Verify SVD was called twice (first failed, second succeeded)
            self.assertEqual(mock_svd.call_count, 2)

    def test_compute_optimal_added_parameters_matrix_shapes_in_error(self):
        """Test that matrix information is correctly printed in SVD error scenario"""
        matrix_s = torch.eye(2) * 3.0
        matrix_n = torch.randn(2, 4)

        # Create successful result BEFORE mocking
        dummy_matrix = torch.randn(2, 4)  # Same shape as matrix_p would be
        u_real, s_real, vt_real = torch.linalg.svd(dummy_matrix, full_matrices=False)
        successful_result = (u_real, s_real, vt_real)

        # Capture stdout to verify matrix information is printed
        captured_output = io.StringIO()

        with unittest.mock.patch("torch.linalg.svd") as mock_svd:
            mock_svd.side_effect = [
                torch.linalg.LinAlgError("Test error"),
                successful_result,
            ]

            with contextlib.redirect_stdout(captured_output):
                compute_optimal_added_parameters(matrix_s, matrix_n)

            output = captured_output.getvalue()

            # Verify specific matrix information is printed
            self.assertIn("matrix_s.min()=", output)
            self.assertIn("matrix_s.max()=", output)
            self.assertIn("matrix_s.shape=", output)
            self.assertIn("matrix_n.min()=", output)
            self.assertIn("matrix_n.max()=", output)
            self.assertIn("matrix_n.shape=", output)
            self.assertIn("matrix_s_inverse_sqrt.min()=", output)
            self.assertIn("matrix_p.min()=", output)

    @unittest_parametrize(
        ({"force_pseudo_inverse": False}, {"force_pseudo_inverse": True})
    )
    def test_compute_optimal_delta_basic_functionality(
        self, force_pseudo_inverse: bool = False
    ):
        """Test basic functionality of compute_optimal_delta with normal and forced pseudo-inverse."""
        # Create test matrices
        tensor_s = torch.eye(3) * 2.0
        tensor_m = torch.randn(3, 2)

        # Test computation with either normal or forced pseudo-inverse
        delta, decrease = optimal_delta(
            tensor_s, tensor_m, force_pseudo_inverse=force_pseudo_inverse
        )

        # Verify output shapes
        self.assertEqual(delta.shape, (2, 3))  # M.T shape
        self.assertIsInstance(decrease, torch.Tensor)

        # Verify numerical properties
        self.assertFalse(torch.isnan(delta).any())
        if not isinstance(decrease, torch.Tensor):
            decrease = torch.tensor(decrease)
        self.assertFalse(torch.isnan(decrease).any())

    def test_compute_optimal_delta_dtype_conversion(self):
        """Test dtype conversion in compute_optimal_delta."""
        # Create test matrices with same dtype
        tensor_s = torch.eye(3, dtype=torch.float64) * 2.0
        tensor_m = torch.randn(3, 2, dtype=torch.float64)

        # Test with specified dtype conversion
        delta, decrease = optimal_delta(tensor_s, tensor_m, dtype=torch.float32)

        # Should preserve original dtype in output
        self.assertEqual(delta.dtype, torch.float64)  # Original tensor dtype
        # parameter_update_decrease should also be converted back to original dtype
        self.assertEqual(decrease.dtype, torch.float64)

    def test_compute_optimal_delta_dtype_assertion(self):
        """Test that compute_optimal_delta raises AssertionError for mismatched dtypes."""
        # Create test matrices with different dtypes
        tensor_s = torch.eye(3, dtype=torch.float32) * 2.0
        tensor_m = torch.randn(3, 2, dtype=torch.float64)

        # Should raise AssertionError
        with self.assertRaises(AssertionError) as context:
            optimal_delta(tensor_s, tensor_m)

        # Verify the error message mentions dtype mismatch
        self.assertIn("same dtype", str(context.exception))
        self.assertIn("tensor_s.dtype", str(context.exception))
        self.assertIn("tensor_m.dtype", str(context.exception))

    def test_compute_optimal_delta_decrease_dtype_preservation(self):
        """Test that parameter_update_decrease dtype is preserved."""
        # Test with float32
        tensor_s = torch.eye(3, dtype=torch.float32) * 2.0
        tensor_m = torch.randn(3, 2, dtype=torch.float32)

        delta, decrease = optimal_delta(tensor_s, tensor_m)

        self.assertEqual(delta.dtype, torch.float32)
        if isinstance(decrease, torch.Tensor):
            self.assertEqual(decrease.dtype, torch.float32)

        # Test with float64
        tensor_s = torch.eye(3, dtype=torch.float64) * 2.0
        tensor_m = torch.randn(3, 2, dtype=torch.float64)

        delta, decrease = optimal_delta(tensor_s, tensor_m)

        self.assertEqual(delta.dtype, torch.float64)
        if isinstance(decrease, torch.Tensor):
            self.assertEqual(decrease.dtype, torch.float64)

    def test_compute_optimal_delta_linalg_error_fallback(self):
        """Test LinAlgError fallback in compute_optimal_delta."""
        tensor_s = torch.eye(3) * 2.0
        tensor_m = torch.randn(3, 2)

        with (
            unittest.mock.patch(
                "torch.linalg.solve", side_effect=torch.linalg.LinAlgError("Mocked error")
            ),
            unittest.mock.patch("gromo.utils.tools.warn") as mock_warn,
        ):
            delta, _ = optimal_delta(tensor_s, tensor_m)

            # Should have called warning about pseudo-inverse
            mock_warn.assert_called()
            warn_calls = [str(call) for call in mock_warn.call_args_list]
            self.assertTrue(any("pseudo-inverse" in call for call in warn_calls))

        # Should still produce valid results
        self.assertEqual(delta.shape, (2, 3))
        self.assertFalse(torch.isnan(delta).any())

    def test_compute_optimal_delta_negative_decrease_warning(self):
        """Test warning when parameter_update_decrease is negative.

        Note: This is a defensive test for a theoretically rare case.
        Due to the mathematical properties of the computation, negative decrease
        should be very rare with well-conditioned positive definite matrices.
        """
        # Create matrices and force a scenario by mocking the trace computation
        tensor_s = torch.eye(2, dtype=torch.float32)
        tensor_m = torch.ones(2, 2, dtype=torch.float32)

        # Mock torch.trace to return a negative value
        with unittest.mock.patch("gromo.utils.tools.warn") as mock_warn:
            with unittest.mock.patch("torch.trace", return_value=torch.tensor(-1.0)):
                optimal_delta(tensor_s, tensor_m)

                # The warning should be called
                mock_warn.assert_called()

                # Check that the specific warning about negative decrease was called
                warning_calls = [
                    call
                    for call in mock_warn.call_args_list
                    if len(call[0]) > 0
                    and "parameter update decrease should be positive" in call[0][0]
                ]
                self.assertTrue(
                    len(warning_calls) > 0, "Should warn about negative decrease"
                )
                warn_calls = [str(call) for call in mock_warn.call_args_list]
                self.assertTrue(any("should be positive" in call for call in warn_calls))

    def test_compute_optimal_delta_negative_decrease_float64_retry(self):
        """Test retry with float64 when negative decrease occurs."""
        tensor_s = torch.eye(2, dtype=torch.float32)
        tensor_m = torch.ones(2, 2, dtype=torch.float32)

        # Mock torch.trace to return negative value, triggering the retry mechanism
        with unittest.mock.patch("torch.trace", return_value=torch.tensor(-1.0)):
            with unittest.mock.patch("gromo.utils.tools.warn") as mock_warn:
                optimal_delta(tensor_s, tensor_m)

                # Should warn about negative decrease and trying float64
                warn_calls = [str(call) for call in mock_warn.call_args_list]
                self.assertTrue(
                    any(
                        "parameter update decrease should be positive" in call
                        for call in warn_calls
                    )
                )
                self.assertTrue(
                    any(
                        "Trying to use the pseudo-inverse with torch.float64" in call
                        for call in warn_calls
                    )
                )

    def test_compute_optimal_delta_negative_decrease_zero_fallback(self):
        """Test zero fallback when pseudo-inverse also gives negative decrease."""
        tensor_s = torch.eye(2) * 0.1
        tensor_m = -torch.ones(2, 2) * 10.0

        with unittest.mock.patch("torch.trace", return_value=torch.tensor(-1.0)):
            with unittest.mock.patch("gromo.utils.tools.warn") as mock_warn:
                delta, _ = optimal_delta(tensor_s, tensor_m, force_pseudo_inverse=True)

                # Should warn about setting delta to zero
                warn_calls = [str(call) for call in mock_warn.call_args_list]
                self.assertTrue(
                    any("set" in call and "zero" in call for call in warn_calls)
                )

                # Delta should be all zeros
                self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))

    def test_compute_optimal_delta_assertion_checks(self):
        """Test assertion checks in compute_optimal_delta."""
        tensor_s = torch.eye(3)
        tensor_m = torch.randn(3, 2)

        # Test NaN assertion
        with unittest.mock.patch(
            "torch.linalg.solve", return_value=torch.full(tensor_m.shape, float("nan"))
        ):
            with self.assertRaises(AssertionError) as context:
                optimal_delta(tensor_s, tensor_m)
            self.assertIn("NaN values", str(context.exception))

    def test_compute_optimal_delta_matrix_dimensions(self):
        """Test compute_optimal_delta with various matrix dimensions."""
        test_cases = [
            (2, 3),  # Small matrices
            (5, 4),  # Medium matrices
            (1, 1),  # Minimal case
            (3, 1),  # Single output
        ]

        for s_dim, m_cols in test_cases:
            with self.subTest(s_dim=s_dim, m_cols=m_cols):
                tensor_s = torch.eye(s_dim) + 0.1 * torch.randn(s_dim, s_dim)
                tensor_s = tensor_s @ tensor_s.T  # Ensure positive definite
                tensor_m = torch.randn(s_dim, m_cols)

                delta, decrease = optimal_delta(tensor_s, tensor_m)

                self.assertEqual(delta.shape, (m_cols, s_dim))
                self.assertFalse(torch.isnan(delta).any())
                if isinstance(decrease, torch.Tensor):
                    self.assertFalse(torch.isnan(decrease).any())
                else:
                    self.assertFalse(torch.isnan(torch.tensor(decrease)))


if __name__ == "__main__":
    main()
