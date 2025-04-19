from unittest import TestCase, main

import torch

from gromo.utils.tools import (
    apply_border_effect_on_unfolded,
    compute_mask_tensor_t,
    compute_output_shape_conv,
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


if __name__ == "__main__":
    main()
