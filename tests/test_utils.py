import random
import unittest

import torch

from gromo.utils.utils import *


class TestUtils(unittest.TestCase):
    def test_set_device(self) -> None:
        if torch.cuda.is_available():
            assert global_device() == torch.device("cuda")
        else:
            assert global_device() == torch.device("cpu")
        set_device("cuda")
        assert global_device() == torch.device("cuda")
        set_device("cpu")
        assert global_device() == torch.device("cpu")

    def test_torch_zeros(self) -> None:
        size = (random.randint(1, 10), random.randint(1, 10))
        tensor = torch_zeros(*size)
        tensor_device = torch.device("cuda" if tensor.is_cuda else "cpu")
        assert tensor_device == global_device()
        assert tensor.shape == size
        assert torch.all(tensor == 0)

    def test_torch_ones(self) -> None:
        size = (random.randint(1, 10), random.randint(1, 10))
        tensor = torch_ones(*size)
        tensor_device = torch.device("cuda" if tensor.is_cuda else "cpu")
        assert tensor_device == global_device()
        assert tensor.shape == size
        assert torch.all(tensor == 1)
