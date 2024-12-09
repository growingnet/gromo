from unittest import TestCase, main
from copy import deepcopy

import torch

from gromo.conv2d_growing_module import Conv2dGrowingModule
from gromo.utils.utils import global_device



class MyTestCase(TestCase):
    def setUp(self):
        self.demo_layer = torch.nn.Conv2d(
            2,
            7,
            (3, 5),
            bias=False,
            device=global_device()
        )
        self.demo = Conv2dGrowingModule(
            in_channels=2,
            out_channels=7,
            kernel_size=(3, 5),
            use_bias=False
        )
        self.demo.layer = self.demo_layer

    def test_init(self):
        m = Conv2dGrowingModule(
            in_channels=2,
            out_channels=7,
            kernel_size=(3, 5),
            use_bias=False
        )
        self.assertIsInstance(m, Conv2dGrowingModule)

    def test_forward(self):
        torch.manual_seed(0)
        x = torch.randn(5, 2, 10, 10, device=global_device())
        y = self.demo(x)
        self.assertTrue(torch.equal(y, self.demo_layer(x)))

    def test_number_of_parameters(self):
        self.assertEqual(self.demo.number_of_parameters(), self.demo_layer.weight.numel())

    def test_str(self):
        self.assertIsInstance(str(self.demo), str)

    def test_layer_of_tensor(self):
        wl = self.demo.layer_of_tensor(self.demo_layer.weight.data)
        # way to test that wl == self.demo_layer
        torch.manual_seed(0)
        x = torch.randn(5, 2, 10, 10, device=global_device())
        y = self.demo(x)
        self.assertTrue(torch.equal(y, wl(x)))

    def test_layer_in_extension(self):
        in_extension = torch.nn.Conv2d(3,
                                            7, (3, 5), bias=False, device=global_device())
        local_demo = deepcopy(self.demo)
        local_demo.layer_in_extension(in_extension.weight)

        torch.manual_seed(0)
        x = torch.randn(23, 5, 10, 10, device=global_device())
        x_main = x[:, :2]
        x_ext = x[:, 2:]
        y_th = self.demo(x_main) + in_extension(x_ext)
        y = local_demo(x)
        self.assertTrue(torch.allclose(y, y_th, atol=1e-6),
                        f"Error: ({torch.abs(y - y_th).max().item():.2e})")

    def test_layer_out_extension(self):
        out_extension = torch.nn.Conv2d(2,
                                        1, (3, 5), bias=False, device=global_device())
        local_demo = deepcopy(self.demo)
        local_demo.layer_out_extension(out_extension.weight)

        torch.manual_seed(0)
        x = torch.randn(23, 2, 10, 10, device=global_device())
        y_main_th = self.demo(x)
        y_ext_th = out_extension(x)
        y = local_demo(x)
        y_main = y[:, :7]
        y_ext = y[:, 7:]
        self.assertTrue(torch.allclose(y_main, y_main_th, atol=1e-6),
                        f"Error: ({torch.abs(y_main - y_main_th).max().item():.2e})")
        self.assertTrue(torch.allclose(y_ext, y_ext_th, atol=1e-6),
                        f"Error: ({torch.abs(y_ext - y_ext_th).max().item():.2e})")

    def test_tensor_s_update(self):
        torch.manual_seed(0)
        x = torch.randn(23, 2, 10, 10, device=global_device())

        self.demo.init_computation()
        self.demo(x)

        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, 23)
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, 23)

        self.demo(x)
        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, 2 * 23)

        self.demo.reset_computation()

    def tensor_m_update(self):
        torch.manual_seed(0)
        x = torch.randn(23, 2, 10, 10, device=global_device())

        self.demo.init_computation()
        y = self.demo(x)
        loss = torch.norm(y)
        loss.backward()

        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, 23)
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, 23)

        y = self.demo(x)
        loss = torch.norm(y)
        loss.backward()
        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, 2 * 23)

        self.demo.reset_computation()

    def compute_optimal_delta(self):
        torch.manual_seed(0)
        x = torch.randn(23, 2, 10, 10, device=global_device())

        self.demo.init_computation()
        y = self.demo(x)
        loss = torch.norm(y)
        loss.backward()

        self.demo.tensor_s.update()
        self.demo.tensor_m.update()

        self.demo.compute_optimal_delta()
        self.assertTrue(self.demo.optimal_delta_layer is not None)
        self.assertIsInstance(self.demo.optimal_delta_layer, torch.nn.Conv2d)
        # TODO: improve the specificity of the test

        self.demo.reset_computation()
        self.demo.delete_update()

    # test input extended
    # test compute m prev update : how ?
    # test cross covariance update : how ?
    # test compute compute_prev_s_update update : how ?


if __name__ == '__main__':
    main()
