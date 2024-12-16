import random
import unittest

import torch

from gromo.utils.utils import *


class TestUtils(unittest.TestCase):
    def test_set_device(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(global_device(), torch.device("cuda"))
        else:
            self.assertEqual(global_device(), torch.device("cpu"))
        set_device("cuda")
        self.assertEqual(global_device(), torch.device("cuda"))
        set_device("cpu")
        self.assertEqual(global_device(), torch.device("cpu"))
        set_device(torch.device("cuda"))
        self.assertEqual(global_device(), torch.device("cuda"))
        set_device(torch.device("cpu"))
        self.assertEqual(global_device(), torch.device("cpu"))

    def test_torch_zeros(self) -> None:
        size = (random.randint(1, 10), random.randint(1, 10))
        tensor = torch_zeros(size)
        tensor_device = torch.device("cuda" if tensor.is_cuda else "cpu")
        self.assertEqual(tensor_device, global_device())
        self.assertEqual(tensor.shape, size)
        self.assertTrue(torch.all(tensor == 0))

    def test_torch_ones(self) -> None:
        size = (random.randint(1, 10), random.randint(1, 10))
        tensor = torch_ones(size)
        tensor_device = torch.device("cuda" if tensor.is_cuda else "cpu")
        self.assertEqual(tensor_device, global_device())
        self.assertEqual(tensor.shape, size)
        self.assertTrue(torch.all(tensor == 1))

    def test_activation_fn(self) -> None:
        self.assertIsInstance(activation_fn(None), nn.Identity)
        self.assertIsInstance(activation_fn("Id"), nn.Identity)
        self.assertIsInstance(activation_fn("Test"), nn.Identity)
        self.assertIsInstance(activation_fn("Softmax"), nn.Softmax)
        self.assertIsInstance(activation_fn("SELU"), nn.SELU)
        self.assertIsInstance(activation_fn("RELU"), nn.ReLU)

    def test_mini_batch_gradient_descent(self) -> None:
        callable_forward = lambda x: x**2 + 1
        cost_fn = lambda pred, y: torch.sum((pred - y) ** 2)
        x = torch.rand((5, 2), requires_grad=True, device=global_device())
        y = torch.rand((5, 1), device=global_device())
        lrate = 1e-3
        epochs = 50
        batch_size = 8
        with self.assertRaises(AttributeError):
            mini_batch_gradient_descent(
                callable_forward, cost_fn, x, y, lrate, epochs, batch_size, verbose=False
            )
            mini_batch_gradient_descent(
                callable_forward,
                cost_fn,
                x,
                y,
                lrate,
                epochs,
                batch_size,
                parameters=[],
                verbose=False,
            )

        parameters = [x]
        mini_batch_gradient_descent(
            callable_forward,
            cost_fn,
            x,
            y,
            lrate,
            epochs,
            batch_size,
            parameters,
            verbose=False,
        )

        model = nn.Linear(2, 1, device=global_device())
        eval_fn = lambda: None
        mini_batch_gradient_descent(
            model,
            cost_fn,
            x,
            y,
            lrate,
            epochs,
            batch_size,
            eval_fn=eval_fn,
            verbose=False,
        )
