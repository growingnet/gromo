from unittest import TestCase, main

import torch

from gromo.growing_module import GrowingModule
from gromo.utils.utils import global_device


class TestGrowingModule(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 3, device=global_device())
        self.x_ext = torch.randn(2, 7, device=global_device())
        self.layer = torch.nn.Linear(3, 5, bias=False, device=global_device())
        self.layer_in_extension = torch.nn.Linear(
            7, 5, bias=False, device=global_device()
        )
        self.layer_out_extension = torch.nn.Linear(
            3, 7, bias=False, device=global_device()
        )
        self.model = GrowingModule(
            self.layer, tensor_s_shape=(3, 3), tensor_m_shape=(3, 5), allow_growing=False
        )

    def test_weight(self):
        self.assertTrue(torch.equal(self.model.weight, self.layer.weight))

    def test_bias(self):
        self.assertTrue(self.model.bias is None)

    def test_forward(self):
        self.assertTrue(torch.equal(self.model(self.x), self.layer(self.x)))

    def test_extended_forward(self):
        y_th = self.layer(self.x)
        y, y_sup = self.model.extended_forward(self.x)
        self.assertIsNone(y_sup)
        self.assertTrue(torch.equal(y, y_th))

        # ========== Test with in extension ==========
        # extended input without extension crashes
        with self.assertRaises(ValueError):
            self.model.extended_forward(self.x, self.x_ext)

        # extended input with in extension
        self.model.extended_input_layer = self.layer_in_extension
        self.model.scaling_factor = 1
        y, y_sup = self.model.extended_forward(self.x, self.x_ext)
        self.assertIsNone(y_sup)
        self.assertTrue(torch.allclose(y, y_th + self.layer_in_extension(self.x_ext)))

        # no extension with an extended input crashes
        with self.assertRaises(ValueError):
            self.model.extended_forward(self.x)

        self.model.extended_input_layer = None

        # ========== Test with out extension ==========
        self.model.extended_output_layer = self.layer_out_extension
        y, y_sup = self.model.extended_forward(self.x)
        self.assertTrue(torch.equal(y, y_th))
        self.assertTrue(torch.equal(y_sup, self.layer_out_extension(self.x)))

    def test_str(self):
        self.assertIsInstance(str(self.model), str)

    def test_repr(self):
        self.assertIsInstance(repr(self.model), str)


if __name__ == "__main__":
    main()
