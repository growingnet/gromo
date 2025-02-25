from unittest import TestCase, main

import torch

from gromo.modules.growing_module import GrowingModule
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
        # extended input with in extension
        self.model.extended_input_layer = self.layer_in_extension
        self.model.scaling_factor = 1.0
        y, y_sup = self.model.extended_forward(self.x, self.x_ext)
        self.assertIsNone(y_sup)
        self.assertTrue(torch.allclose(y, y_th + self.layer_in_extension(self.x_ext)))

        # no extension with an extended input crashes
        with self.assertRaises(ValueError):
            self.model.extended_forward(self.x)

        self.model.extended_input_layer = None

        # ========== Test with out extension ==========
        # extended input without extension crashes
        with self.assertWarns(UserWarning):
            self.model.extended_forward(self.x, self.x_ext)

        self.model.extended_output_layer = self.layer_out_extension
        self.model._scaling_factor_next_module = 1.0
        y, y_sup = self.model.extended_forward(self.x)
        self.assertTrue(torch.equal(y, y_th))
        self.assertTrue(torch.equal(y_sup, self.layer_out_extension(self.x)))

    def test_str(self):
        self.assertIsInstance(str(self.model), str)

    def test_repr(self):
        self.assertIsInstance(repr(self.model), str)

    def test_init(self):
        with self.assertRaises(AssertionError):
            l1 = GrowingModule(
                torch.nn.Linear(3, 5, bias=False, device=global_device()),
                tensor_s_shape=(3, 3),
                tensor_m_shape=(3, 5),
                allow_growing=True,
            )

        l1 = GrowingModule(
            torch.nn.Linear(3, 5, bias=False, device=global_device()),
            tensor_s_shape=(3, 3),
            tensor_m_shape=(3, 5),
            allow_growing=False,
        )

        self.assertIsInstance(l1, GrowingModule)

        l2 = GrowingModule(
            torch.nn.Linear(5, 7, bias=False, device=global_device()),
            tensor_s_shape=(5, 5),
            tensor_m_shape=(5, 7),
            allow_growing=True,
            previous_module=l1,
        )

        self.assertIsInstance(l2, GrowingModule)
        self.assertTrue(l2.previous_module is l1)

    def test_delete_update(self):
        l1 = GrowingModule(
            torch.nn.Linear(3, 5, bias=False, device=global_device()),
            tensor_s_shape=(3, 3),
            tensor_m_shape=(3, 5),
            allow_growing=False,
        )
        l2 = GrowingModule(
            torch.nn.Linear(5, 7, bias=False, device=global_device()),
            tensor_s_shape=(5, 5),
            tensor_m_shape=(5, 7),
            allow_growing=True,
            previous_module=l1,
        )

        def reset(layer, first: bool) -> None:
            dummy_layer = torch.nn.Identity()
            layer.extended_output_layer = dummy_layer
            layer.optimal_delta_layer = dummy_layer
            if not first:
                layer.extended_input_layer = dummy_layer

        def reset_all():
            reset(l1, True)
            reset(l2, False)

        reset_all()
        l1.delete_update()
        self.assertIsInstance(l1.extended_output_layer, torch.nn.Identity)
        self.assertIsNone(l1.optimal_delta_layer)

        reset_all()
        with self.assertWarns(UserWarning):
            l2.delete_update(include_previous=False)
        self.assertIsNone(l2.extended_input_layer)
        self.assertIsInstance(l1.extended_output_layer, torch.nn.Identity)
        self.assertIsNone(l2.optimal_delta_layer)
        self.assertIsInstance(l2.extended_output_layer, torch.nn.Identity)

        reset_all()
        l2.delete_update()
        self.assertIsNone(l2.extended_input_layer)
        self.assertIsNone(l1.extended_output_layer)
        self.assertIsNone(l2.optimal_delta_layer)
        self.assertIsInstance(l2.extended_output_layer, torch.nn.Identity)

        reset_all()
        l2.delete_update(include_output=True)
        self.assertIsNone(l2.extended_input_layer)
        self.assertIsNone(l1.extended_output_layer)
        self.assertIsNone(l2.optimal_delta_layer)
        self.assertIsNone(l2.extended_output_layer)

        reset_all()
        l1.extended_output_layer = None
        l2.delete_update(include_previous=False)
        self.assertIsNone(l2.extended_input_layer)
        self.assertIsNone(l2.optimal_delta_layer)
        self.assertIsInstance(l2.extended_output_layer, torch.nn.Identity)

        # incorrect behavior
        reset(l1, False)
        with self.assertWarns(UserWarning):
            l1.delete_update()

        # incorrect behavior
        reset(l1, False)
        with self.assertRaises(TypeError):
            l1.previous_module = True  # type: ignore
            l1.delete_update()

        # incorrect behavior
        reset(l1, False)
        with self.assertRaises(TypeError):
            l1.previous_module = True  # type: ignore
            l1.delete_update(include_previous=False)


if __name__ == "__main__":
    main()
