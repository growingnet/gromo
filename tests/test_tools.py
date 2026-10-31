import contextlib
import io
import unittest.mock
import warnings
from functools import partial
from unittest import main, mock

import torch

from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import (
    KNOWN_THRESHOLD_RULES,
    apply_border_effect_on_unfolded,
    compute_mask_tensor_t,
    compute_optimal_added_parameters,
    compute_output_shape_conv,
    create_bordering_effect_weight,
    numerical_floor_terms,
    optimal_delta,
    pseudo_inverse_matrix_semi_positive,
    pytorch_pinv_threshold,
    resolve_threshold,
    resolve_threshold_rule,
    spectrum_summary,
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
                sqrt_inverse_matrix = sqrt_inverse_matrix_semi_positive(matrix)
            # The eigendecomposition of the regularized matrix is the one used
            self.assertAllClose(
                sqrt_inverse_matrix,
                torch.eye(5, dtype=dtype) / torch.finfo(dtype).resolution ** 0.5,
                message="Shrinkage not applied correctly",
            )
            self.assertTrue(
                torch.equal(matrix, torch.zeros_like(matrix)),
                "The caller's matrix must not be modified",
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

        # Test case 6: Alpha zero
        matrix_s = torch.eye(3) * 2.0
        matrix_n = torch.randn(3, 2)
        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, statistical_threshold=0.0, alpha_zero=True
        )

        # Check that alpha is zero when forced via the alpha_zero=True flag
        self.assertTrue(torch.allclose(alpha, torch.zeros_like(alpha)))

        # Test case 7: Omega zero
        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, statistical_threshold=0.0, omega_zero=True
        )
        self.assertTrue(torch.allclose(omega, torch.zeros_like(omega)))

    def test_compute_optimal_added_parameters_e_numerical_threshold(self):
        """E's whitening threshold can differ from S's, and the floor is relative."""
        matrix_s = torch.eye(3) * 2.0
        matrix_n = torch.randn(3, 2)
        matrix_e = torch.diag(torch.tensor([1.0, 1e-3]))

        def singular_values(**kwargs):
            return compute_optimal_added_parameters(
                matrix_s, matrix_n, statistical_threshold=0.0, **kwargs
            )[2]

        whitened_n = matrix_n / torch.sqrt(torch.tensor(2.0))
        # Default: the floor keeps both directions of E
        self.assertAllClose(
            singular_values(matrix_covariance_loss_gradient=matrix_e),
            torch.linalg.svdvals(whitened_n @ torch.diag(torch.tensor([1.0, 1e3**0.5]))),
            rtol=1e-4,
        )
        # A threshold for E only cuts its second direction, and leaves S untouched
        cut = torch.linalg.svdvals(whitened_n @ torch.diag(torch.tensor([1.0, 0.0])))
        self.assertAllClose(
            singular_values(
                matrix_covariance_loss_gradient=matrix_e, e_numerical_threshold=1e-2
            )[: cut.shape[0]],
            cut,
            rtol=1e-4,
        )
        # e_numerical_threshold=None falls back to numerical_threshold
        self.assertAllClose(
            singular_values(
                matrix_covariance_loss_gradient=matrix_e, numerical_threshold=1e-2
            )[: cut.shape[0]],
            cut,
            rtol=1e-4,
        )
        # The floor is relative: an E of scale 1e-8 is not truncated, E^{-1/2} = 1e4 I
        self.assertAllClose(
            singular_values(matrix_covariance_loss_gradient=torch.eye(2) * 1e-8),
            torch.linalg.svdvals(whitened_n) * 1e4,
            rtol=1e-4,
        )

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

    def test_create_bordering_effect_weight(self):
        """Test create_bordering_effect_weight function"""
        # Test basic functionality
        channels = 12  # 2 * 3 * 2 (in_channels * kernel_h * kernel_w)
        conv = torch.nn.Conv2d(2, 4, (3, 2), padding=(1, 0))

        weight = create_bordering_effect_weight(channels, conv)

        # Check shape: depthwise kernel (channels, 1, kH, kW)
        self.assertShapeEqual(
            weight, (channels, 1, conv.kernel_size[0], conv.kernel_size[1])
        )

        # Check weight initialization (1.0 at the center, 0.0 elsewhere)
        mid_h, mid_w = conv.kernel_size[0] // 2, conv.kernel_size[1] // 2
        center_weights = weight[:, 0, mid_h, mid_w]
        self.assertAllClose(center_weights, torch.ones(channels))
        self.assertAllClose(weight.sum(), torch.tensor(float(channels)))

        # Test error cases
        with self.assertRaises(ValueError):
            create_bordering_effect_weight(-1, conv)  # Invalid channels

        with self.assertRaises(ValueError):
            create_bordering_effect_weight(0, conv)  # Invalid channels

        with self.assertRaises(TypeError):
            create_bordering_effect_weight(
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

        # Mock the SVD to trigger LinAlgError
        # The function now prints diagnostics and re-raises the error (no retry)
        captured_output = io.StringIO()

        with unittest.mock.patch("torch.linalg.svd") as mock_svd:
            mock_svd.side_effect = torch.linalg.LinAlgError("Mocked SVD error")
            with unittest.mock.patch("sys.stdout", captured_output):
                with self.assertRaises(torch.linalg.LinAlgError) as context:
                    # Function should raise the error after printing diagnostics
                    compute_optimal_added_parameters(matrix_s, matrix_n)

            # Verify debug output was printed
            output = captured_output.getvalue()
            self.assertIn("Warning: An error occurred during the SVD computation", output)
            self.assertIn("matrix_s:", output)
            self.assertIn("matrix_n:", output)
            self.assertIn("matrix_s_inverse_sqrt:", output)
            self.assertIn("matrix_p:", output)

            # Verify the error was re-raised
            self.assertIn("Mocked SVD error", str(context.exception))

            # Verify SVD was called once (no retry)
            self.assertEqual(mock_svd.call_count, 1)

    def test_compute_optimal_added_parameters_matrix_shapes_in_error(self):
        """Test that matrix information is correctly printed in SVD error scenario"""
        matrix_s = torch.eye(2) * 3.0
        matrix_n = torch.randn(2, 4)

        # Capture stdout to verify matrix information is printed
        captured_output = io.StringIO()

        with unittest.mock.patch("torch.linalg.svd") as mock_svd:
            mock_svd.side_effect = torch.linalg.LinAlgError("Test error")
            with contextlib.redirect_stdout(captured_output):
                with self.assertRaises(torch.linalg.LinAlgError):
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
        gradient_covariance = torch.eye(2) * 0.5
        tensor_m = torch.randn(3, 2)

        # Test computation with either normal or forced pseudo-inverse
        delta, decrease = optimal_delta(
            tensor_s,
            tensor_m,
            force_pseudo_inverse=force_pseudo_inverse,
            tensor_covariance_loss_gradient=gradient_covariance,
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
        gradient_covariance = torch.eye(2, dtype=torch.float64) * 0.5
        tensor_m = torch.randn(3, 2, dtype=torch.float64)

        # Test with specified dtype conversion
        delta, decrease = optimal_delta(
            tensor_s,
            tensor_m,
            dtype=torch.float32,
            tensor_covariance_loss_gradient=gradient_covariance,
        )

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


class TestComputeOptimalAddedParametersTheory(TorchTestCase):
    """
    Theoretical tests for compute_optimal_added_parameters using simple
    diagonal matrices.

    Setup:
        X = Diag(1, 2, 3, 4, 5)
        Y = Diag(4, 4, 4, 4, 4)
        N = X^T @ Y
        S = X^T @ X
    """

    def setUp(self) -> None:
        super().setUp()
        self.x = torch.diag(torch.arange(1.0, 6.0))
        self.y = 4.0 * torch.eye(5)
        self.n = self.x.T @ self.y
        self.s = self.x.T @ self.x

    def _reconstruction(self, alpha: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Compute X @ Alpha.T @ Omega.T, the reconstructed output."""
        return self.x @ alpha.T @ omega.T

    def test_s_xtx_exact_reconstruction(self) -> None:
        """S=X^T X, ignore_singular_values=False:
        X @ Alpha.T @ Omega.T == Y

        With the correct covariance S, the full SVD reconstruction
        exactly recovers Y.
        """
        alpha, omega, _ = compute_optimal_added_parameters(
            self.s,
            self.n,
            numerical_threshold=1e-10,
            statistical_threshold=1e-6,
        )
        result = self._reconstruction(alpha, omega)
        self.assertAllClose(
            result,
            self.y,
            atol=1e-5,
            message="With S=X^TX, reconstruction should exactly equal Y",
        )

    def test_s_xtx_ignore_sv_nonzero_pattern(self) -> None:
        """S=X^T X, ignore_singular_values=True:
        X @ Alpha.T @ Omega.T is non-zero iff Y is non-zero

        When singular values are ignored (treated as 1), the reconstruction
        preserves the non-zero structure of Y but not its exact values.
        """
        alpha, omega, _ = compute_optimal_added_parameters(
            self.s,
            self.n,
            numerical_threshold=1e-10,
            statistical_threshold=0,
            ignore_singular_values=True,
        )
        result = self._reconstruction(alpha, omega)
        self.assertTrue(
            torch.equal(result != 0, self.y != 0),
            "With ignore_singular_values, non-zero pattern should match Y",
        )

    def test_s_none_max_neurons_2_structure(self) -> None:
        """S=None, ignore_singular_values=False, maximum_added_neurons=2:
        X @ Alpha.T @ Omega.T is equal to Y on the last 2 features,
        and the first 3 features are zero.

        Without S (GradMax path), the top-2 singular values of N correspond
        to the last 2 features (largest entries of diagonal N). The first 3
        features are exactly zero and the last 2 preserve Y's non-zero pattern.
        """
        alpha, omega, _ = compute_optimal_added_parameters(
            None,
            self.n,
            statistical_threshold=0,
            maximum_added_neurons=2,
        )
        result = self._reconstruction(alpha, omega)

        # First 3 row-features and column-features are zero
        self.assertAllClose(
            result[:3],
            torch.zeros(3, 5),
            atol=1e-6,
            message="First 3 row-features should be zero",
        )
        self.assertAllClose(
            result[:, :3],
            torch.zeros(5, 3),
            atol=1e-6,
            message="First 3 column-features should be zero",
        )

        # Last 2 features have same non-zero pattern as Y
        self.assertTrue(
            torch.equal(result[3:] != 0, self.y[3:] != 0),
            "Last 2 features should be non-zero where Y is non-zero",
        )
        self.assertAllClose(
            result[3:],
            (self.y * self.x**2)[3:],
            atol=1e-5,
            message="Last 2 features should match Y scaled by X^2",
        )

    def test_s_none_ignore_sv_max_neurons_2_pattern(self) -> None:
        """S=None, ignore_singular_values=True, maximum_added_neurons=2:
        For the last 2 features: non-zero iff Y is non-zero.
        For the first 3 features: zero.

        Same truncation as above, but with unit singular values the
        reconstruction only preserves the directional (non-zero) pattern.
        """
        alpha, omega, _ = compute_optimal_added_parameters(
            None,
            self.n,
            statistical_threshold=0,
            maximum_added_neurons=2,
            ignore_singular_values=True,
        )
        result = self._reconstruction(alpha, omega)

        # First 3 features are zero
        self.assertAllClose(
            result[:3],
            torch.zeros(3, 5),
            atol=1e-6,
            message="First 3 row-features should be zero",
        )
        self.assertAllClose(
            result[:, :3],
            torch.zeros(5, 3),
            atol=1e-6,
            message="First 3 column-features should be zero",
        )

        # Last 2 features: non-zero iff Y is non-zero
        self.assertTrue(
            torch.equal(result[3:] != 0, self.y[3:] != 0),
            "Last 2 features: non-zero iff Y is non-zero",
        )


class TestGrowthSpectra(TorchTestCase):
    """Tests for the `spectra` out-parameter and the threshold rules."""

    def setUp(self):
        super().setUp()
        self.matrix_s = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
        self.matrix_n = torch.randn(3, 3)
        self.matrix_e = torch.diag(torch.tensor([1.0, 0.5, 0.25]))

    def test_spectra_does_not_change_results(self):
        """Collecting the spectra leaves the returned values untouched."""
        kwargs = {
            "matrix_s": self.matrix_s,
            "matrix_n": self.matrix_n,
            "matrix_covariance_loss_gradient": self.matrix_e,
        }
        reference = compute_optimal_added_parameters(**kwargs)  # type: ignore
        spectra = dict()
        collected = compute_optimal_added_parameters(**kwargs, spectra=spectra)  # type: ignore
        for expected, obtained in zip(reference, collected, strict=False):
            self.assertAllClose(expected, obtained)
        self.assertNotEqual(spectra, dict())

    def test_spectra_schema(self):
        """Every key is set, with None for the matrices that are not used."""
        spectra = {}
        compute_optimal_added_parameters(
            matrix_s=self.matrix_s,
            matrix_n=self.matrix_n,
            matrix_covariance_loss_gradient=self.matrix_e,
            spectra=spectra,
        )
        self.assertEqual(set(spectra.keys()), {"matrix_s", "matrix_e", "extension"})
        self.assertIsNotNone(spectra["matrix_s"])
        self.assertIsNotNone(spectra["matrix_e"])

        # GradMax path: no S, no E
        spectra = {}
        compute_optimal_added_parameters(
            matrix_s=None, matrix_n=self.matrix_n, spectra=spectra
        )
        self.assertIsNone(spectra["matrix_s"])
        self.assertIsNone(spectra["matrix_e"])
        self.assertIsNotNone(spectra["extension"])

    def test_spectra_counts(self):
        """The recorded counts describe the selection that was applied."""
        spectra = {}
        _, _, eigenvalues = compute_optimal_added_parameters(
            matrix_s=self.matrix_s,
            matrix_n=self.matrix_n,
            statistical_threshold=0.0,
            spectra=spectra,
        )
        extension = spectra["extension"]
        self.assertEqual(extension["singular_values"].shape[0], 3)
        self.assertEqual(extension["kept"], eigenvalues.shape[0])
        self.assertEqual(extension["kept_by_threshold"], 3)
        self.assertIsNone(extension["maximum_added_neurons"])

        matrix_s_spectra = spectra["matrix_s"]
        self.assertEqual(matrix_s_spectra["total"], 3)
        self.assertEqual(matrix_s_spectra["kept"], 3)
        self.assertFalse(matrix_s_spectra["regularized"])
        self.assertAllClose(
            matrix_s_spectra["eigenvalues"],
            torch.tensor([1.0, 2.0, 4.0]),
        )

    def test_spectra_maximum_added_neurons(self):
        """The threshold cut and the maximum cut are recorded separately."""
        spectra = {}
        compute_optimal_added_parameters(
            matrix_s=self.matrix_s,
            matrix_n=self.matrix_n,
            statistical_threshold=0.0,
            maximum_added_neurons=1,
            spectra=spectra,
        )
        self.assertEqual(spectra["extension"]["kept_by_threshold"], 3)
        self.assertEqual(spectra["extension"]["kept"], 1)
        self.assertEqual(spectra["extension"]["maximum_added_neurons"], 1)

    def test_optimal_delta_spectrum(self):
        """The recorded singular values are those of the returned delta."""
        tensor_s = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
        tensor_m = torch.randn(3, 2)
        spectra = {}
        delta, _ = optimal_delta(tensor_s, tensor_m, spectra=spectra)
        self.assertAllClose(spectra["singular_values"], torch.linalg.svdvals(delta))

    def test_resolve_threshold(self):
        """A value passes through, a rule is evaluated, degenerate cases fall back."""
        spectrum = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(resolve_threshold(1e-3, spectrum), 1e-3)
        self.assertEqual(resolve_threshold(lambda s: s.mean().item(), spectrum), 2.0)
        self.assertEqual(
            resolve_threshold(lambda s: 1.0, torch.empty(0)),
            0.0,
        )
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(
                resolve_threshold(lambda s: float("inf"), spectrum, fallback=1e-6),
                1e-6,
            )

    def test_resolve_threshold_rule(self):
        """A name resolves to its rule, a rule to itself, an unknown name raises."""
        rule = resolve_threshold_rule("mean_over_sqrt_n")
        self.assertIs(rule, KNOWN_THRESHOLD_RULES["mean_over_sqrt_n"])

        def custom(statistic, spectrum):
            return 0.0

        self.assertIs(resolve_threshold_rule(custom), custom)
        with self.assertRaises(ValueError):
            resolve_threshold_rule("not_a_rule")  # type: ignore

    def test_mean_over_sqrt_n_rule(self):
        """The shipped rule scales the mean of the spectrum by 1 / sqrt(n)."""
        statistic = TensorStatistic(shape=None, update_function=lambda: (None, 0))  # type: ignore
        statistic.samples = 4
        rule = KNOWN_THRESHOLD_RULES["mean_over_sqrt_n"]
        self.assertAlmostEqual(rule(statistic, torch.tensor([1.0, 2.0, 3.0])), 1.0)
        statistic.samples = 0  # no sample yet: no division by zero
        self.assertAlmostEqual(rule(statistic, torch.tensor([1.0, 2.0, 3.0])), 2.0)

    def test_lethal_threshold_is_guarded(self):
        """A threshold that would select nothing never leaves an empty selection."""
        # A finite but huge value: the min(threshold, s.max()) guard keeps one neuron
        _, _, eigenvalues = compute_optimal_added_parameters(
            matrix_s=self.matrix_s,
            matrix_n=self.matrix_n,
            statistical_threshold=1e12,
        )
        self.assertEqual(eigenvalues.shape[0], 1)

        # A non-finite rule value: falls back to keeping the whole spectrum
        with self.assertWarns(RuntimeWarning):
            _, _, eigenvalues = compute_optimal_added_parameters(
                matrix_s=self.matrix_s,
                matrix_n=self.matrix_n,
                statistical_threshold=lambda s: float("inf"),
            )
        self.assertEqual(eigenvalues.shape[0], 3)

        # Whitening side: falling back rather than returning a zero matrix
        with self.assertWarns(RuntimeWarning):
            result = sqrt_inverse_matrix_semi_positive(
                self.matrix_s, threshold=lambda s: 1e12
            )
        self.assertGreater(result.abs().max().item(), 0.0)

    def test_threshold_rule_end_to_end(self):
        """A bound rule selects fewer neurons and is recorded as the applied value."""
        statistic = TensorStatistic(shape=None, update_function=lambda: (None, 0))  # type: ignore
        statistic.samples = 1
        rule = partial(KNOWN_THRESHOLD_RULES["mean_over_sqrt_n"], statistic)

        spectra = dict()
        _, _, with_rule = compute_optimal_added_parameters(
            matrix_s=self.matrix_s,
            matrix_n=self.matrix_n,
            statistical_threshold=rule,
            spectra=spectra,
        )
        _, _, without = compute_optimal_added_parameters(
            matrix_s=self.matrix_s,
            matrix_n=self.matrix_n,
            statistical_threshold=0.0,
        )
        self.assertLess(with_rule.shape[0], without.shape[0])
        self.assertAlmostEqual(
            spectra["extension"]["threshold"], without.mean().item(), places=5
        )

    def test_spectrum_summary(self):
        """The spectrum summary reports the correct counts and values."""
        spectrum = torch.tensor([4.0, 2.0, 1.0])
        summary = spectrum_summary(spectrum)
        self.assertIsInstance(summary, dict)


class TestNumericalFloor(TorchTestCase):
    """The numerical floor under every eigenvalue threshold."""

    d, rank, n = 64, 40, 5000

    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        z = torch.randn(self.n, self.rank, generator=generator, dtype=torch.float64)
        w = torch.randn(self.rank, self.d, generator=generator, dtype=torch.float64)
        self.x = z @ w  # rank-deficient samples, well conditioned on their span
        # S accumulated in float64, then stored in float32: its null space is float32
        # rounding noise.
        self.clean_s = self.x.T @ self.x / self.n
        self.s = self.clean_s.float()
        y = torch.randn(self.n, 3, generator=generator, dtype=torch.float64)
        self.clean_m = self.x.T @ y / self.n
        a = torch.randn(3, 2, generator=generator, dtype=torch.float64)
        self.clean_e = a @ a.T  # rank-deficient gradient covariance

        # A single null direction: with this seed, its float32 noise eigenvalue is
        # positive (checked in test_precision_term_uses_source_dtype), so the
        # negative-eigenvalue term has nothing to read and only the precision term
        # removes that noise.
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(200, 15, generator=generator, dtype=torch.float64)
        x = x @ torch.randn(15, 16, generator=generator, dtype=torch.float64)
        self.one_null_s = x.T @ x / 200
        y = torch.randn(200, 3, generator=generator, dtype=torch.float64)
        self.one_null_m = x.T @ y / 200

    @staticmethod
    def statistic(samples: int = 1, dtype: torch.dtype = torch.float32):
        """A statistic accumulated in dtype from a given number of samples."""
        statistic = TensorStatistic(
            shape=None, update_function=lambda: (torch.zeros(1, dtype=dtype), samples)
        )
        statistic.updated = False
        statistic.update()
        return statistic

    def kept(self, matrix, threshold, **kwargs) -> int:
        spectra = dict()
        sqrt_inverse_matrix_semi_positive(
            matrix, threshold=threshold, spectra=spectra, **kwargs
        )
        return spectra["kept"]

    def assert_close_to_clean(self, delta, expected):
        relative_error = (delta.double() - expected).norm() / expected.norm()
        self.assertLess(relative_error.item(), 1e-2)

    def test_pytorch_pinv_threshold(self):
        """It is PyTorch's default tolerance, in the dtype of the spectrum."""
        self.assertEqual(pytorch_pinv_threshold(torch.empty(0)), 0.0)
        self.assertEqual(pytorch_pinv_threshold(-torch.ones(2)), 0.0)
        spectrum = torch.tensor([1.0, 4.0], dtype=torch.float64)
        self.assertEqual(
            pytorch_pinv_threshold(spectrum), 2 * torch.finfo(torch.float64).eps * 4
        )

        eigenvalues = torch.linalg.eigvalsh(self.s)
        kept = int((eigenvalues > pytorch_pinv_threshold(eigenvalues)).sum())
        self.assertEqual(kept, int(torch.linalg.matrix_rank(self.s, hermitian=True)))
        self.assertEqual(kept, self.rank)

    def test_numerical_floor_terms(self):
        """Each term has its exact value, and none reaches lambda_1 in bfloat16."""
        eps32 = torch.finfo(torch.float32).eps
        eps_bf16 = torch.finfo(torch.bfloat16).eps
        spectrum = torch.tensor([-0.5, 1.0, 4.0], dtype=torch.float64)
        terms = numerical_floor_terms(spectrum, source_dtype=torch.float32)
        self.assertEqual(terms["worst_case"], pytorch_pinv_threshold(spectrum))
        self.assertEqual(terms["precision"], eps32 * 4)
        self.assertEqual(terms["negative_eigenvalue"], 1.0)
        terms = numerical_floor_terms(
            spectrum.abs(), source_dtype=torch.bfloat16, worst_case=False
        )
        self.assertEqual(terms["worst_case"], 0.0)
        self.assertEqual(terms["precision"], eps_bf16 * 4)
        self.assertEqual(terms["negative_eigenvalue"], 0.0)
        self.assertEqual(set(numerical_floor_terms(torch.empty(0)).values()), {0.0})

        # The first version, d * eps_bf16 * lambda_1, dropped everything for d >= 128
        d = 256
        eigenvalues = torch.linspace(1e-3, 1.0, d)
        spectra = dict()
        sqrt_inverse_matrix_semi_positive(
            torch.diag(eigenvalues), spectra=spectra, source_dtype=torch.bfloat16
        )
        self.assertLess(spectra["numerical_floor"], 1.0)
        self.assertGreater(spectra["kept"], 0)

    def test_operator_norm_noise_threshold_rule(self):
        """The rule is scale-equivariant and estimates the operator-norm error."""
        rule = resolve_threshold_rule("operator_norm_noise_threshold")
        # n = max(0, 1), negative eigenvalues count as 0: 2 * sqrt(4 * 4 / 1) + 4 / 1
        self.assertEqual(rule(self.statistic(samples=0), torch.tensor([-1.0, 4.0])), 12.0)
        d, n = 32, 4000
        x = torch.randn(
            n, d, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        )
        estimate = x.T @ x / n
        spectrum = torch.linalg.eigvalsh(estimate)
        statistic = self.statistic(samples=n)
        value = rule(statistic, spectrum)

        self.assertAlmostEqual(rule(statistic, 3 * spectrum), 3 * value)
        error = torch.linalg.matrix_norm(estimate - torch.eye(d), ord=2).item()
        self.assertGreater(value, 0.8 * error)
        self.assertLess(value, 1.5 * error)

    def test_every_threshold_is_floored(self):
        """None, a value and a rule all keep exactly the rank: none keeps rounding noise."""
        for threshold in (None, 0.0, lambda _: 0.0):
            spectra = dict()
            sqrt_inverse_matrix_semi_positive(
                self.s, threshold=threshold, spectra=spectra
            )
            self.assertEqual(spectra["kept"], self.rank)
            self.assertEqual(spectra["threshold"], spectra["numerical_floor"])
            self.assertEqual(
                spectra["numerical_floor"],
                max(numerical_floor_terms(spectra["eigenvalues"]).values()),
            )

    def test_negative_eigenvalue_term(self):
        """Twice the most negative eigenvalue removes the noise of a large null space."""
        # After a cast to float64 the worst-case and precision terms are negligible.
        spectra = dict()
        sqrt_inverse_matrix_semi_positive(
            self.s.double(), spectra=spectra, worst_case_numerical_floor=False
        )
        self.assertEqual(spectra["kept"], self.rank)
        terms = spectra["numerical_floor_terms"]
        self.assertEqual(spectra["numerical_floor"], terms["negative_eigenvalue"])

    def test_precision_term_uses_source_dtype(self):
        """With a single null direction, only the float32 epsilon removes its noise."""
        cast = self.one_null_s.float().double()
        # The noise eigenvalue is positive: there is nothing negative to read.
        self.assertGreater(torch.linalg.eigvalsh(cast)[0].item(), 0.0)
        self.assertEqual(self.kept(cast, None), 16)
        self.assertEqual(self.kept(cast, None, source_dtype=torch.float32), 15)

    def test_fallback_is_relative(self):
        """A threshold that drops the whole spectrum falls back to the floor, at any
        scale, whether it is a rule or a value."""
        for scale in (1e-6, 1.0, 1e6):
            for threshold in (lambda _: 1e30, lambda _: float("inf"), 1e30):
                with self.subTest(scale=scale), self.assertWarns(RuntimeWarning):
                    self.assertEqual(self.kept(scale * self.s, threshold), self.rank)

    def test_matrices_without_positive_spectrum(self):
        """A matrix far from PSD gives zero with a warning; a dead matrix silently."""
        with self.assertWarns(RuntimeWarning):
            result = sqrt_inverse_matrix_semi_positive(
                torch.diag(torch.tensor([1.0, -1.0]))
            )
        self.assertTrue(torch.equal(result, torch.zeros(2, 2)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = sqrt_inverse_matrix_semi_positive(torch.zeros(3, 3))
        self.assertTrue(torch.equal(result, torch.zeros(3, 3)))

    def test_pseudo_inverse_matrix_semi_positive(self):
        """It equals pinv on a PSD matrix, and drops a negative noise eigenvalue that
        pinv inverts with its wrong sign."""
        a = torch.randn(
            5, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        )
        matrix = a @ a.T + torch.eye(5)
        self.assertAllClose(
            pseudo_inverse_matrix_semi_positive(matrix), torch.linalg.pinv(matrix)
        )
        noisy = torch.diag(torch.tensor([2.0, -1e-3]))
        self.assertAllClose(
            torch.linalg.pinv(noisy), torch.diag(torch.tensor([0.5, -1e3]))
        )
        self.assertAllClose(
            pseudo_inverse_matrix_semi_positive(noisy),
            torch.diag(torch.tensor([0.5, 0.0])),
        )

    def test_optimal_delta_pinv_uses_source_epsilon(self):
        """The float64 pseudo-inverses of float32 statistics ignore their rounding noise."""
        delta, _ = optimal_delta(
            self.one_null_s.float(),
            self.one_null_m.float(),
            dtype=torch.float64,
            force_pseudo_inverse=True,
            tensor_covariance_loss_gradient=self.clean_e.float(),
        )
        self.assertEqual(delta.dtype, torch.float32)
        expected = (
            torch.linalg.pinv(self.clean_e)
            @ (torch.linalg.pinv(self.one_null_s) @ self.one_null_m).t()
        )
        self.assert_close_to_clean(delta, expected)

    @unittest_parametrize(
        (
            {"statistics_dtype": torch.float32, "dtype": torch.float64},
            {"statistics_dtype": torch.float64, "dtype": torch.float32},
        )
    )
    def test_optimal_delta_retry_uses_the_least_precise_epsilon(
        self, statistics_dtype: torch.dtype, dtype: torch.dtype
    ):
        """The float64 retry uses the float32 epsilon, whether float32 is the dtype of
        the statistics or that of the cast."""
        # A solution with a negative decrease triggers the retry.
        with (
            mock.patch("torch.linalg.solve", return_value=-self.one_null_m.to(dtype)),
            self.assertWarns(UserWarning),
        ):
            delta, _ = optimal_delta(
                self.one_null_s.to(statistics_dtype),
                self.one_null_m.to(statistics_dtype),
                dtype=dtype,
            )
        expected = (torch.linalg.pinv(self.one_null_s) @ self.one_null_m).t()
        self.assert_close_to_clean(delta, expected)

    def test_optimal_delta_retry_keeps_the_dtype(self):
        """The float64 retry returns tensors in the dtype of the statistics."""
        spectra = dict()
        with (
            self.assertWarns(UserWarning),
            unittest.mock.patch(
                "gromo.utils.tools.pseudo_inverse_matrix_semi_positive",
                wraps=pseudo_inverse_matrix_semi_positive,
            ) as wrapped,
        ):
            # S = -I: solve succeeds and the decrease is negative, which triggers the retry.
            delta, decrease = optimal_delta(
                -torch.eye(3),
                torch.ones(3, 2),
                dtype=torch.float64,
                spectra=spectra,
                worst_case_numerical_floor=False,
            )
        self.assertIs(wrapped.call_args.kwargs["worst_case_numerical_floor"], False)
        self.assertEqual(delta.dtype, torch.float32)
        self.assertEqual(decrease.dtype, torch.float32)
        self.assertEqual(spectra["singular_values"].dtype, torch.float32)


if __name__ == "__main__":
    main()
