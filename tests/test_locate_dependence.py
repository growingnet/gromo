import time
import unittest

import torch

from gromo.utils.locate_dependence import gaussian_kernel, slow_gaussian_kernel


class TestLocateDependence(unittest.TestCase):
    def setUp(self) -> None:
        self.X = torch.rand((10, 500))

    def test_fast_gaussian_kernel(self) -> None:
        # start = time.time()
        K = gaussian_kernel(self.X)
        # print(f"Gaussian time {time.time() - start} s")
        # start = time.time()
        K_fast = slow_gaussian_kernel(self.X)
        # print(f"Slow Gaussian time {time.time() - start} s")

        self.assertEqual(K.shape, K_fast.shape)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                self.assertAlmostEqual(K[i, j].item(), K_fast[i, j].item(), places=6)


if __name__ == "__main__":
    unittest.main()
