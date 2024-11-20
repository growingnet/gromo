from unittest import TestCase, main

import torch

from gromo.tools import compute_output_shape_conv, sqrt_inverse_matrix_semi_positive


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
        with X in (50, 10)
        Test the function on cpu and cuda if available.
        """
        torch.manual_seed(0)
        if torch.cuda.is_available():
            devices = (torch.device("cuda"), torch.device("cpu"))
        else:
            devices = (torch.device("cpu"),)
            print(f"Waring: No cuda device available therefore only testing on cpu")
        for device in devices:
            for i in range(10):
                matrix = torch.randn(50, 10, dtype=torch.float64, device=device)
                matrix = matrix.t() @ matrix
                sqrt_inverse_matrix = sqrt_inverse_matrix_semi_positive(
                    matrix, threshold=1e-7, preferred_linalg_library=None
                )
                reconstructed_inverse = sqrt_inverse_matrix @ sqrt_inverse_matrix
                if torch.abs(torch.linalg.det(matrix)) > 1e-5:
                    correct = torch.allclose(
                        reconstructed_inverse, torch.linalg.inv(matrix)
                    )
                    self.assertTrue(
                        correct,
                        f"In example {i} error of "
                        f"{torch.abs(reconstructed_inverse - matrix.inverse()).max().item():.2e}"
                        f"with device {device}",
                    )

    def test_compute_output_shape_conv(self):
        """
        Test the compute_output_shape_conv function
        with various inputs shapes and conv kernel sizes.
        """
        for h in (2, 5, 11, 41):
            for w in (2, 5, 11, 41):
                for k_h in (1, 2, 3, 5, 7):
                    if k_h <= h:
                        for k_w in (1, 2, 3, 5, 7):
                            if k_w <= w:
                                conv = torch.nn.Conv2d(1, 1, (k_h, k_w))
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


if __name__ == "__main__":
    main()
