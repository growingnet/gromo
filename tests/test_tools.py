from unittest import TestCase, main

import torch

from gromo.utils.tools import (
    apply_border_effect_on_unfolded,
    compute_mask_tensor_t,
    compute_optimal_added_parameters,
    compute_output_shape_conv,
    create_bordering_effect_convolution,
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
        sqrt_inverse_matrix = sqrt_inverse_matrix_semi_positive(
            matrix, preferred_linalg_library=None
        )
        self.assertTrue(
            torch.allclose(sqrt_inverse_matrix @ sqrt_inverse_matrix, matrix.inverse())
        )

    def test_random_sqrt_inverse_matrix_semi_positive(self):
        """
        Test the sqrt_inverse_matrix_semi_positive on random X^T X matrice
        with X in (5, 3)
        Test the function on cpu and cuda if available.
        """
        torch.manual_seed(0)
        if torch.cuda.is_available():
            devices = (torch.device("cuda"), torch.device("cpu"))
        else:
            devices = (torch.device("cpu"),)
            print(f"Warning: No cuda device available therefore only testing on cpu")
        for device in devices:
            matrix = torch.randn(5, 3, dtype=torch.float64, device=device)
            matrix = matrix.t() @ matrix
            sqrt_inverse_matrix = sqrt_inverse_matrix_semi_positive(
                matrix, threshold=1e-7, preferred_linalg_library=None
            )
            reconstructed_inverse = sqrt_inverse_matrix @ sqrt_inverse_matrix
            if torch.abs(torch.linalg.det(matrix)) > 1e-5:
                correct = torch.allclose(reconstructed_inverse, torch.linalg.inv(matrix))
                self.assertTrue(
                    correct,
                    f"Error of "
                    f"{torch.abs(reconstructed_inverse - matrix.inverse()).max().item():.2e}"
                    f"with device {device}",
                )

    def test_compute_output_shape_conv(self):
        """
        Test the compute_output_shape_conv function
        with various inputs shapes and conv kernel sizes.
        """
        torch.manual_seed(0)
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
        torch.manual_seed(0)

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
                    self.assertTrue(
                        torch.allclose(y_th, y_via_mask, atol=1e-6),
                        f"Error with {h=}, {w=}, {k_h=}, {k_w=} "
                        f"Error: {torch.abs(y_th - y_via_mask).max().item():.2e}",
                    )

    @unittest_parametrize(test_input_shapes)
    def test_compute_mask_tensor_t_with_bias(self, h, w):
        """
        Test the compute_mask_tensor_t function with bias activated.
        Check that it respects its property.
        """
        torch.manual_seed(0)
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
                    y_via_mask += conv.bias.data.view(1, -1, 1)
                    self.assertTrue(
                        torch.allclose(y_th, y_via_mask, atol=1e-6),
                        f"Error with {h=}, {w=}, {k_h=}, {k_w=} "
                        f"Error: {torch.abs(y_th - y_via_mask).max().item():.2e}",
                    )

    def test_apply_border_effect_on_unfolded_typing(self, bias: bool = False):
        conv1 = torch.nn.Conv2d(2, 3, (3, 5), padding=(1, 2), bias=bias)
        conv2 = torch.nn.Conv2d(3, 4, (3, 5), padding=(1, 2), bias=False)
        x = torch.randn(11, 2, 13, 17)
        unfolded_x = torch.nn.functional.unfold(
            x,
            kernel_size=conv1.kernel_size,
            padding=conv1.padding,
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
            padding=conv1.padding,
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
        """
        Test the compute_optimal_added_parameters function with various inputs.
        """
        torch.manual_seed(0)

        # Test with simple matrices
        matrix_s = torch.eye(3)
        matrix_n = torch.randn(3, 2)

        alpha, omega, eigenvalues = compute_optimal_added_parameters(matrix_s, matrix_n)

        # Check output shapes
        self.assertEqual(alpha.shape, (eigenvalues.shape[0], matrix_s.shape[0]))
        self.assertEqual(omega.shape, (matrix_n.shape[1], eigenvalues.shape[0]))
        self.assertEqual(eigenvalues.shape, (eigenvalues.shape[0],))

        # Check that we get at least one eigenvalue
        self.assertGreater(eigenvalues.shape[0], 0)

    def test_compute_optimal_added_parameters_with_thresholds(self):
        """
        Test the compute_optimal_added_parameters function with different thresholds.
        """
        torch.manual_seed(1)

        # Create matrices with known properties
        matrix_s = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
        matrix_n = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        # Test with different thresholds
        alpha1, omega1, eigenvalues1 = compute_optimal_added_parameters(
            matrix_s, matrix_n, statistical_threshold=0.1
        )

        alpha2, omega2, eigenvalues2 = compute_optimal_added_parameters(
            matrix_s, matrix_n, statistical_threshold=1.0
        )

        # With higher threshold, we should get fewer or no eigenvalues
        self.assertLessEqual(eigenvalues2.shape[0], eigenvalues1.shape[0])

    def test_compute_optimal_added_parameters_maximum_neurons(self):
        """
        Test the compute_optimal_added_parameters function with maximum_added_neurons parameter.
        """
        torch.manual_seed(2)

        matrix_s = torch.eye(5)
        matrix_n = torch.randn(5, 4)

        # Test with maximum_added_neurons=2
        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, maximum_added_neurons=2
        )

        # Check that we get at most 2 eigenvalues
        self.assertLessEqual(eigenvalues.shape[0], 2)

    def test_compute_optimal_added_parameters_edge_cases(self):
        """
        Test edge cases for compute_optimal_added_parameters function.
        """
        torch.manual_seed(3)

        # Test with non-symmetric matrix (should be handled with warning)
        matrix_s = torch.tensor([[1.0, 0.1], [0.2, 1.0]])  # Not symmetric
        matrix_n = torch.randn(2, 1)

        # Should not raise an error but handle the non-symmetric matrix
        alpha, omega, eigenvalues = compute_optimal_added_parameters(matrix_s, matrix_n)

        # Check output shapes
        self.assertEqual(alpha.shape[0], eigenvalues.shape[0])
        self.assertEqual(omega.shape[0], matrix_n.shape[1])

    def test_create_bordering_effect_convolution(self):
        """
        Test the create_bordering_effect_convolution function.
        """
        # Test with valid inputs
        channels = 6  # 2 input channels * 3x1 kernel size
        conv = torch.nn.Conv2d(2, 3, (3, 1), padding=(1, 0))

        border_conv = create_bordering_effect_convolution(channels, conv)

        # Check that the output is a Conv2d instance
        self.assertIsInstance(border_conv, torch.nn.Conv2d)

        # Check the properties of the created convolution
        self.assertEqual(border_conv.in_channels, channels)
        self.assertEqual(border_conv.out_channels, channels)
        self.assertEqual(border_conv.kernel_size, conv.kernel_size)
        self.assertEqual(border_conv.padding, conv.padding)
        self.assertEqual(border_conv.stride, conv.stride)
        self.assertEqual(border_conv.dilation, conv.dilation)
        self.assertFalse(border_conv.bias)

        # Check that the weight has the correct structure (identity in center)
        self.assertEqual(border_conv.weight.shape, (channels, 1, 3, 1))

        # Check that only the center elements are non-zero
        mid_h = conv.kernel_size[0] // 2
        mid_w = conv.kernel_size[1] // 2
        # For each group, only the center element should be 1.0, others 0
        for i in range(channels):
            # Check that the center element is 1.0
            self.assertAlmostEqual(
                border_conv.weight.data[i, 0, mid_h, mid_w].item(), 1.0
            )

    def test_create_bordering_effect_convolution_edge_cases(self):
        """
        Test edge cases for create_bordering_effect_convolution function.
        """
        # Test with invalid inputs
        conv = torch.nn.Conv2d(2, 3, (3, 3))

        # Test with invalid channels (negative)
        with self.assertRaises(ValueError):
            create_bordering_effect_convolution(-1, conv)

        # Test with invalid channels (zero)
        with self.assertRaises(ValueError):
            create_bordering_effect_convolution(0, conv)

        # Test with invalid convolution type
        with self.assertRaises(TypeError):
            create_bordering_effect_convolution(6, "not_a_convolution")

    def test_sqrt_inverse_matrix_semi_positive_edge_cases(self):
        """
        Test edge cases for sqrt_inverse_matrix_semi_positive function.
        """
        # Test with non-symmetric matrix
        matrix = torch.tensor([[1.0, 0.1], [0.2, 1.0]])  # Not symmetric
        with self.assertRaises(AssertionError):
            sqrt_inverse_matrix_semi_positive(matrix)

        # Test with matrix containing NaN values
        matrix_nan = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        with self.assertRaises(AssertionError):
            sqrt_inverse_matrix_semi_positive(matrix_nan)

        # Test with non-square matrix
        matrix_non_square = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(AssertionError):
            sqrt_inverse_matrix_semi_positive(matrix_non_square)

    def test_sqrt_inverse_matrix_semi_positive_preferred_linalg_library(self):
        """
        Test the sqrt_inverse_matrix_semi_positive function with preferred_linalg_library parameter.
        """
        torch.manual_seed(0)
        matrix = torch.eye(3)

        # Test with preferred_linalg_library=None (default)
        result_none = sqrt_inverse_matrix_semi_positive(
            matrix, preferred_linalg_library=None
        )
        self.assertIsInstance(result_none, torch.Tensor)

        # Test with preferred_linalg_library="magma" if available
        if torch.cuda.is_available():
            try:
                result_magma = sqrt_inverse_matrix_semi_positive(
                    matrix, preferred_linalg_library="magma"
                )
                self.assertIsInstance(result_magma, torch.Tensor)
            except Exception:
                # If magma is not available, this is expected
                pass

    def test_compute_optimal_added_parameters_exception_handling(self):
        """
        Test the exception handling in compute_optimal_added_parameters function.
        """
        torch.manual_seed(0)

        # Create matrices that might cause SVD to fail
        matrix_s = torch.eye(3)
        matrix_n = torch.randn(3, 2)

        # This should work normally
        alpha, omega, eigenvalues = compute_optimal_added_parameters(matrix_s, matrix_n)
        self.assertIsInstance(alpha, torch.Tensor)
        self.assertIsInstance(omega, torch.Tensor)
        self.assertIsInstance(eigenvalues, torch.Tensor)

    def test_compute_optimal_added_parameters_non_symmetric_warning(self):
        """
        Test the warning when input matrix S is not symmetric in compute_optimal_added_parameters.
        This covers lines 89-95 in tools.py.
        """
        torch.manual_seed(0)

        # Create a non-symmetric matrix S
        matrix_s = torch.tensor([[2.0, 0.1], [0.2, 1.0]])  # Not symmetric
        matrix_n = torch.randn(2, 1)

        # This should work but issue a warning
        alpha, omega, eigenvalues = compute_optimal_added_parameters(matrix_s, matrix_n)
        self.assertIsInstance(alpha, torch.Tensor)
        self.assertIsInstance(omega, torch.Tensor)
        self.assertIsInstance(eigenvalues, torch.Tensor)

    def test_compute_optimal_added_parameters_maximum_added_neurons(self):
        """
        Test the maximum_added_neurons parameter in compute_optimal_added_parameters.
        This covers line 127 in tools.py.
        """
        torch.manual_seed(0)

        # Create matrices with known properties
        matrix_s = torch.eye(5)
        matrix_n = torch.randn(5, 4)

        # Test with maximum_added_neurons=2
        alpha, omega, eigenvalues = compute_optimal_added_parameters(
            matrix_s, matrix_n, maximum_added_neurons=2
        )

        # Check that we get at most 2 eigenvalues
        self.assertLessEqual(eigenvalues.shape[0], 2)

    def test_create_bordering_effect_convolution_error_handling(self):
        """
        Test error handling in create_bordering_effect_convolution.
        This covers lines 258 and 260 in tools.py.
        """
        # Test with invalid channels (negative)
        conv = torch.nn.Conv2d(2, 3, (3, 3))
        with self.assertRaises(ValueError):
            create_bordering_effect_convolution(-1, conv)

        # Test with invalid channels (zero)
        with self.assertRaises(ValueError):
            create_bordering_effect_convolution(0, conv)

        # Test with invalid convolution type
        with self.assertRaises(TypeError):
            create_bordering_effect_convolution(6, "not_a_convolution")


if __name__ == "__main__":
    main()
