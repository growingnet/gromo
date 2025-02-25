from unittest import TestCase, main

import torch

from gromo.utils.tools import (
    compute_mask_tensor_t,
    compute_output_shape_conv,
    sqrt_inverse_matrix_semi_positive,
)

from .unittest_tools import unittest_parametrize


test_input_shapes = [
    {"h": 4, "w": 4},
    {"h": 4, "w": 5},
    {"h": 5, "w": 4},
    {"h": 5, "w": 5},
]


class TestTools(TestCase):
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


if __name__ == "__main__":
    main()
