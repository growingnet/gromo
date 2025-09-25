from copy import deepcopy
from unittest import main

import torch

from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    FullConv2dGrowingModule,
    RestrictedConv2dGrowingModule,
)
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import compute_output_shape_conv
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase, indicator_batch
from tests.unittest_tools import unittest_parametrize


class TestConv2dGrowingModule(TorchTestCase):
    _tested_class = Conv2dGrowingModule

    def setUp(self):
        self.demo_layer = torch.nn.Conv2d(
            2, 7, (3, 5), bias=False, device=global_device()
        )
        self.demo = self._tested_class(
            in_channels=2, out_channels=7, kernel_size=(3, 5), use_bias=False
        )
        self.demo.layer = self.demo_layer

        self.demo_layer_b = torch.nn.Conv2d(
            2, 7, 3, padding=1, bias=True, device=global_device()
        )
        self.demo_b = self._tested_class(
            in_channels=2, out_channels=7, kernel_size=3, padding=1, use_bias=True
        )
        self.demo_b.layer = self.demo_layer_b

        torch.manual_seed(0)
        self.input_x = torch.randn(5, 2, 10, 10, device=global_device())

        self.bias_demos = {True: self.demo_b, False: self.demo}

        self.demo_couple = dict()
        for bias in (True, False):
            demo_in = self._tested_class(
                in_channels=2,
                out_channels=5,
                kernel_size=(3, 3),
                padding=1,
                use_bias=bias,
                device=global_device(),
            )
            demo_out = self._tested_class(
                in_channels=5,
                out_channels=7,
                kernel_size=(5, 5),
                use_bias=bias,
                previous_module=demo_in,
                device=global_device(),
            )
            self.demo_couple[bias] = (demo_in, demo_out)

    def test_init(self):
        # no bias
        m = Conv2dGrowingModule(
            in_channels=2, out_channels=7, kernel_size=(3, 5), use_bias=False
        )
        self.assertIsInstance(m, Conv2dGrowingModule)

        # with bias
        m = Conv2dGrowingModule(
            in_channels=2, out_channels=7, kernel_size=3, padding=1, use_bias=True
        )
        self.assertIsInstance(m, Conv2dGrowingModule)
        self.assertEqual(m.layer.padding, (1, 1))
        self.assertTrue(m.layer.bias is not None)
        self.assertEqual(m.layer.kernel_size, (3, 3))

    def test_forward(self):
        # no bias
        y = self.demo(self.input_x)
        self.assertTrue(torch.equal(y, self.demo_layer(self.input_x)))

        # with bias
        y = self.demo_b(self.input_x)
        self.assertTrue(torch.equal(y, self.demo_layer_b(self.input_x)))

    def test_padding(self):
        self.assertEqual(self.demo.padding, (0, 0))
        y = self.demo(self.input_x)
        self.assertShapeEqual(y, (-1, -1, 8, 6))
        self.demo.padding = (1, 2)
        self.assertEqual(self.demo.padding, (1, 2))
        y = self.demo(self.input_x)
        self.assertShapeEqual(y, (-1, -1, 10, 10))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_number_of_parameters(self, bias: bool):
        self.assertEqual(
            self.bias_demos[bias].number_of_parameters(),
            self.bias_demos[bias].layer.weight.numel()
            + (self.bias_demos[bias].layer.bias.numel() if bias else 0),
        )

    def test_str(self):
        self.assertIsInstance(str(self.demo), str)
        self.assertIsInstance(repr(self.demo), str)
        self.assertIsInstance(str(self.demo_b), str)
        for i in (0, 1, 2):
            self.assertIsInstance(self.demo.__str__(i), str)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_layer_of_tensor(self, bias: bool):
        wl = self.bias_demos[bias].layer_of_tensor(
            self.bias_demos[bias].layer.weight.data,
            self.bias_demos[bias].layer.bias.data if bias else None,
        )
        # way to test that wl == self.demo_layer
        y = self.bias_demos[bias](self.input_x)
        self.assertTrue(torch.equal(y, wl(self.input_x)))

        with self.assertRaises(AssertionError):
            _ = self.bias_demos[bias].layer_of_tensor(
                self.bias_demos[bias].layer.weight.data,
                self.demo_layer_b.bias.data if not bias else None,
            )

    def test_layer_in_extension(self):
        in_extension = torch.nn.Conv2d(3, 7, (3, 5), bias=False, device=global_device())
        local_demo = deepcopy(self.demo)
        local_demo.layer_in_extension(in_extension.weight)

        torch.manual_seed(0)
        x = torch.randn(23, 5, 10, 10, device=global_device())
        x_main = x[:, :2]
        x_ext = x[:, 2:]
        y_th = self.demo(x_main) + in_extension(x_ext)
        y = local_demo(x)
        self.assertAllClose(
            y,
            y_th,
            atol=1e-6,
            message=f"Error: ({torch.abs(y - y_th).max().item():.2e})",
        )

    def test_layer_out_extension_without_bias(self):
        out_extension = torch.nn.Conv2d(2, 5, (3, 5), bias=False, device=global_device())
        local_demo = deepcopy(self.demo)
        with self.assertWarns(UserWarning):
            local_demo.layer_out_extension(
                out_extension.weight, torch.empty(out_extension.out_channels)
            )

        y_main_th = self.demo(self.input_x)
        y_ext_th = out_extension(self.input_x)
        y = local_demo(self.input_x)
        y_main = y[:, : self.demo.out_channels]
        y_ext = y[:, self.demo.out_channels :]
        self.assertAllClose(
            y_main,
            y_main_th,
            atol=1e-6,
            message=f"Error: ({torch.abs(y_main - y_main_th).max().item():.2e})",
        )
        self.assertAllClose(
            y_ext,
            y_ext_th,
            atol=1e-6,
            message=f"Error: ({torch.abs(y_ext - y_ext_th).max().item():.2e})",
        )

    def test_layer_out_extension_with_bias(self):
        out_extension = torch.nn.Conv2d(
            2, 5, 3, bias=True, device=global_device(), padding=1
        )
        local_demo = deepcopy(self.demo_b)
        local_demo.layer_out_extension(out_extension.weight, out_extension.bias)

        y_main_th = self.demo_b(self.input_x)
        y_ext_th = out_extension(self.input_x)
        y = local_demo(self.input_x)
        y_main = y[:, : self.demo_b.out_channels]
        y_ext = y[:, self.demo_b.out_channels :]
        self.assertAllClose(
            y_main,
            y_main_th,
            atol=1e-6,
            message=f"Error: ({torch.abs(y_main - y_main_th).max().item():.2e})",
        )
        self.assertAllClose(
            y_ext,
            y_ext_th,
            atol=1e-6,
            message=f"Error: ({torch.abs(y_ext - y_ext_th).max().item():.2e})",
        )

    def test_tensor_s_update_without_bias(self):
        self.demo.store_input = True
        self.demo.tensor_s.init()
        self.demo(self.input_x)

        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, self.input_x.size(0))

        self.demo(self.input_x)
        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, 2 * self.input_x.size(0))

        f = self.demo.in_channels * self.demo.kernel_size[0] * self.demo.kernel_size[1]
        self.assertEqual(self.demo.tensor_s().shape, (f, f))
        self.assertAllClose(self.demo.tensor_s(), self.demo.tensor_s().transpose(0, 1))

    def test_tensor_s_update_with_bias(self):
        self.demo_b.store_input = True
        self.demo_b.tensor_s.init()
        self.demo_b(self.input_x)

        self.demo_b.tensor_s.update()
        self.assertEqual(self.demo_b.tensor_s.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        f = (
            self.demo_b.in_channels
            * self.demo_b.kernel_size[0]
            * self.demo_b.kernel_size[1]
            + 1
        )
        self.assertShapeEqual(self.demo_b.tensor_s(), (f, f))
        self.assertEqual(
            self.demo_b.tensor_s()[-1, -1], self.input_x.size(2) * self.input_x.size(3)
        )
        # we do the average on the number of samples n but
        # should we not do it on the number of blocks n * h * w ?
        self.assertAllClose(
            self.demo_b.tensor_s(), self.demo_b.tensor_s().transpose(0, 1)
        )

    def test_tensor_m_update_without_bias(self):
        self.demo.store_input = True
        self.demo.store_pre_activity = True
        self.demo.tensor_m.init()
        y = self.demo(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, self.input_x.size(0))

        y = self.demo(self.input_x)
        loss = torch.norm(y)
        loss.backward()
        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, 2 * self.input_x.size(0))

        f = self.demo.in_channels * self.demo.kernel_size[0] * self.demo.kernel_size[1]
        self.assertShapeEqual(self.demo.tensor_m(), (f, self.demo.out_channels))

    def test_tensor_m_update_with_bias(self):
        self.demo_b.store_input = True
        self.demo_b.store_pre_activity = True
        self.demo_b.tensor_m.init()
        y = self.demo_b(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        self.demo_b.tensor_m.update()
        self.assertEqual(self.demo_b.tensor_m.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        f = (
            self.demo_b.in_channels
            * self.demo_b.kernel_size[0]
            * self.demo_b.kernel_size[1]
            + 1
        )
        self.assertShapeEqual(self.demo_b.tensor_m(), (f, self.demo_b.out_channels))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_delta(self, bias: bool = False):
        if bias:
            demo = self.demo_b
        else:
            demo = self.demo

        demo.init_computation()
        y = demo(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        demo.tensor_s.update()
        demo.tensor_m.update()

        demo.compute_optimal_delta()
        self.assertShapeEqual(
            demo.delta_raw,
            (
                demo.out_channels,
                demo.in_channels * demo.kernel_size[0] * demo.kernel_size[1] + bias,
            ),
        )
        self.assertTrue(demo.optimal_delta_layer is not None)
        self.assertIsInstance(demo.optimal_delta_layer, torch.nn.Conv2d)
        if not bias:
            self.assertTrue(demo.optimal_delta_layer.bias is None)
        # TODO: improve the specificity of the test

        demo.compute_optimal_delta(dtype=torch.float64)
        self.assertIsInstance(demo.optimal_delta_layer, torch.nn.Conv2d)

        demo.reset_computation()
        demo.delete_update()

    def test_compute_optimal_delta_empirical(self):
        """
        Test the computation of delta with a simple example:
        We get a random theta as parameter of the layer
        We get each e_i = (0, ..., 0, 1, 0, ..., 0) as input and the loss is the norm of the output
        There fore the optimal delta is proportional to -theta.
        """
        self.demo.init_computation()
        input_x = indicator_batch((2, 3, 5), device=global_device())
        y = self.demo(input_x)
        assert y.shape == (2 * 3 * 5, 7, 1, 1)
        loss = torch.norm(y)
        loss.backward()

        self.demo.tensor_s.update()
        self.demo.tensor_m.update()
        self.demo.compute_optimal_delta()

        self.assertIsInstance(self.demo.optimal_delta_layer, torch.nn.Conv2d)

        self.demo.reset_computation()

        ratio_tensor = (
            self.demo.layer.weight.data / self.demo.optimal_delta_layer.weight.data
        )
        ratio_value: float = ratio_tensor.mean().item()
        self.assertGreaterEqual(
            ratio_value,
            0.0,
            f"Ratio value: {ratio_value} should be positive, as we do W - gamma * dW*",
        )
        self.assertAllClose(ratio_tensor, ratio_value * torch.ones_like(ratio_tensor))

        self.demo.scaling_factor = abs(ratio_value) ** 0.5
        self.demo.apply_change()

        y = self.demo(input_x)
        loss = torch.norm(y)
        self.assertLess(loss.item(), 1e-3)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_input_size(self, bias: bool):
        demo_layer = self.bias_demos[bias]

        # error
        with self.assertRaises(ValueError):
            demo_layer.input_size

        # no error but get None
        self.assertIsNone(demo_layer.update_input_size(force_update=False))

        # automatic setting
        demo_layer.store_input = True
        demo_layer(self.input_x)

        self.assertEqual(demo_layer.input_size, (10, 10))

        # manual
        with self.assertWarns(Warning):
            demo_layer.input_size = (7, 7)

        self.assertEqual(demo_layer.input_size, (7, 7))

        demo_layer.input_size = None
        self.assertIsNone(demo_layer._input_size)

    def test_input_size_with_recursive_calls(self):
        demo_couple: tuple[Conv2dGrowingModule, Conv2dGrowingModule] = self.demo_couple[
            False
        ]
        demo_in, demo_out = demo_couple

        demo_in.input_size = self.input_x.shape[2:]
        y = demo_in(self.input_x)
        self.assertEqual(demo_in.input_size, self.input_x.shape[2:])
        demo_out.update_input_size(compute_from_previous=True)
        self.assertEqual(demo_out.input_size, y.shape[2:])


class TestFullConv2dGrowingModule(TestConv2dGrowingModule):
    _tested_class = FullConv2dGrowingModule

    def test_zero_bottleneck(self):
        """Test behavior when bottleneck is fully resolved
        with parameter change for FullConv2d."""
        # Create FullConv2d equivalent of the demo layers
        demo_layer_1, demo_layer_2 = self.demo_couple[
            False
        ]  # Use without bias for simplicity

        net = torch.nn.Sequential(demo_layer_1, demo_layer_2)
        demo_layer_2.init_computation()

        # Use indicator batch for Conv2d - each sample has 1 in different spatial locations
        input_x = indicator_batch(
            (demo_layer_1.in_channels, 5, 5), device=global_device()
        )
        y = net(input_x)
        loss = torch.norm(y) ** 2 / 2
        loss.backward()
        demo_layer_2.update_computation()
        demo_layer_2.compute_optimal_updates()

        # For FullConv2d, tensor_n should be zero when bottleneck is fully resolved
        self.assertAllClose(
            demo_layer_2.tensor_n, torch.zeros_like(demo_layer_2.tensor_n), atol=1e-7
        )
        self.assertAllClose(
            demo_layer_2.eigenvalues_extension,
            torch.zeros_like(demo_layer_2.eigenvalues_extension),
            atol=1e-7,
        )

    def test_compute_m_prev_without_intermediate_input(self):
        """Check that the batch size is computed using stored variables for FullConv2d"""
        # Use predefined demo_couple objects
        demo_layer_1, demo_layer_2 = self.demo_couple[
            False
        ]  # Use without bias for simplicity

        net = torch.nn.Sequential(demo_layer_1, demo_layer_2)
        demo_layer_2.store_pre_activity = True
        demo_layer_1.store_input = True
        demo_layer_2.tensor_m_prev.init()

        # Create Conv2d input tensor
        input_x = torch.randn(
            11, demo_layer_1.in_channels, 10, 10, device=global_device()
        )

        y = demo_layer_1(input_x)
        loss = demo_layer_2(y).sum()
        loss.backward()

        demo_layer_1.update_input_size(input_x.shape[2:])
        demo_layer_2.update_input_size(y.shape[2:])

        demo_layer_2.tensor_m_prev.update()
        self.assertEqual(demo_layer_2.tensor_m_prev.samples, input_x.size(0))

    def test_masked_unfolded_prev_input_no_prev(self, bias: bool = True):
        demo = self.bias_demos[bias]
        demo.store_input = True
        demo(self.input_x)
        with self.assertRaises(ValueError):
            demo.masked_unfolded_prev_input()

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_masked_unfolded_prev_input(self, bias: bool = False):
        demos = self.demo_couple[bias]
        demos[0].store_input = True
        y = demos[0](self.input_x)
        demos[1].update_input_size(y.shape[2:])
        y = demos[1](y)

        masked_unfolded_tensor = demos[1].masked_unfolded_prev_input
        self.assertShapeEqual(
            masked_unfolded_tensor,
            (
                self.input_x.shape[0],
                y.shape[2] * y.shape[3],
                demos[1].kernel_size[0] * demos[1].kernel_size[1],
                demos[0].kernel_size[0] * demos[0].kernel_size[1] * demos[0].in_channels
                + bias,
            ),
        )

    def test_mask_tensor_t(self):
        with self.assertRaises(ValueError):
            _ = self.demo.mask_tensor_t

        hin, win = 11, 13
        x = torch.randn(1, 2, hin, win, device=global_device())
        hout, wout = self.demo(x).shape[2:]
        self.demo.input_size = (hin, win)

        tensor_t = self.demo.mask_tensor_t

        self.assertIsInstance(tensor_t, torch.Tensor)
        self.assertIsInstance(self.demo._mask_tensor_t, torch.Tensor)

        size_theoretic = (
            hout * wout,
            self.demo.kernel_size[0] * self.demo.kernel_size[1],
            hin * win,
        )
        for i, (t, t_th) in enumerate(zip(tensor_t.shape, size_theoretic)):
            self.assertEqual(t, t_th, f"Error for dim {i}: should be {t_th}, got {t}")

    def test_tensor_m_prev_update(self):
        with self.assertRaises(ValueError):
            # require a previous module
            self.demo.store_pre_activity = True
            self.demo.tensor_m_prev.init()

            y = self.demo(self.input_x)
            loss = torch.norm(y)
            loss.backward()

            self.demo.update_input_size(self.input_x.shape[2:])
            self.demo.tensor_m_prev.update()

        for bias in (True, False):
            with self.subTest(bias=bias):
                demo_couple = self.demo_couple[bias]
                demo_couple[0].store_input = True
                demo_couple[1].store_pre_activity = True
                demo_couple[1].tensor_m_prev.init()

                y = demo_couple[0](self.input_x)
                y = demo_couple[1](y)
                loss = torch.norm(y)
                loss.backward()

                demo_couple[0].update_input_size()
                demo_couple[1].update_input_size(compute_from_previous=True)
                demo_couple[1].tensor_m_prev.update()

                self.assertEqual(
                    demo_couple[1].tensor_m_prev.samples,
                    self.input_x.size(0),
                )

                s0 = demo_couple[0].in_channels * demo_couple[0].kernel_size[
                    0
                ] * demo_couple[0].kernel_size[1] + (1 if bias else 0)
                s1 = (
                    demo_couple[1].out_channels
                    * demo_couple[1].kernel_size[0]
                    * demo_couple[1].kernel_size[1]
                )

                self.assertShapeEqual(
                    demo_couple[1].tensor_m_prev(),
                    (s0, s1),
                )

    def test_cross_covariance_update(self):
        with self.assertRaises(ValueError):
            # require a previous module
            self.demo.store_input = True
            self.demo.cross_covariance.init()

            y = self.demo(self.input_x)
            loss = torch.norm(y)
            loss.backward()

            self.demo.update_input_size(self.input_x.shape[2:])
            self.demo.cross_covariance.update()

        for bias in (True, False):
            with self.subTest(bias=bias):
                demo_couple = self.demo_couple[bias]
                demo_couple[0].store_input = True
                demo_couple[1].store_input = True
                demo_couple[1].cross_covariance.init()

                y = demo_couple[0](self.input_x)
                y = demo_couple[1](y)
                loss = torch.norm(y)
                loss.backward()

                demo_couple[1].update_input_size()
                demo_couple[1].cross_covariance.update()

                self.assertEqual(
                    demo_couple[1].cross_covariance.samples,
                    self.input_x.size(0),
                )

                s0 = demo_couple[1].kernel_size[0] * demo_couple[1].kernel_size[1]
                s1 = demo_couple[0].in_channels * demo_couple[0].kernel_size[
                    0
                ] * demo_couple[0].kernel_size[1] + (1 if bias else 0)
                s2 = demo_couple[1].in_channels * demo_couple[1].kernel_size[
                    0
                ] * demo_couple[1].kernel_size[1] + (1 if bias else 0)

                self.assertShapeEqual(
                    demo_couple[1].cross_covariance(),
                    (s0, s1, s2),
                )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_s_growth_update(self, bias: bool):
        demo_couple = self.demo_couple[bias]
        demo_couple[0].store_input = True
        demo_couple[1].tensor_s_growth.init()

        y = demo_couple[0](self.input_x)
        y = demo_couple[1](y)
        loss = torch.norm(y)
        loss.backward()

        demo_couple[1].input_size = compute_output_shape_conv(
            demo_couple[0].input.shape[2:], demo_couple[0].layer
        )
        demo_couple[1].tensor_s_growth.update()

        self.assertEqual(demo_couple[1].tensor_s_growth.samples, self.input_x.size(0))

        s = demo_couple[0].in_channels * demo_couple[0].kernel_size[0] * demo_couple[
            0
        ].kernel_size[1] + (1 if bias else 0)

        self.assertShapeEqual(
            demo_couple[1].tensor_s_growth(),
            (s, s),
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_added_parameters(self, bias: bool):
        """
        Test sub_select_optimal_added_parameters in merge to compute_optimal_added_parameters
        """
        demo_couple = self.demo_couple[bias]
        demo_couple[0].store_input = True
        demo_couple[1].init_computation()
        demo_couple[1].tensor_s_growth.init()

        y = demo_couple[0](self.input_x)
        y = demo_couple[1](y)
        loss = torch.norm(y)
        loss.backward()

        demo_couple[1].update_computation()
        demo_couple[1].tensor_s_growth.update()

        s_shape_theory = demo_couple[0].in_channels * demo_couple[0].kernel_size[
            0
        ] * demo_couple[0].kernel_size[1] + (1 if bias else 0)
        self.assertShapeEqual(
            demo_couple[1].tensor_s_growth(), (s_shape_theory, s_shape_theory)
        )

        m_prev_shape_theory = (
            s_shape_theory,
            demo_couple[1].out_channels
            * demo_couple[1].kernel_size[0]
            * demo_couple[1].kernel_size[1],
        )
        self.assertShapeEqual(demo_couple[1].tensor_m_prev(), m_prev_shape_theory)

        demo_couple[1].compute_optimal_delta()
        alpha, alpha_b, omega, eigenvalues = demo_couple[
            1
        ].compute_optimal_added_parameters(
            numerical_threshold=0, statistical_threshold=0, maximum_added_neurons=10
        )

        self.assertShapeEqual(
            alpha,
            (
                -1,
                demo_couple[0].in_channels,
                demo_couple[0].kernel_size[0],
                demo_couple[0].kernel_size[1],
            ),
        )
        k = alpha.size(0)
        if bias:
            self.assertShapeEqual(alpha_b, (k,))
        else:
            self.assertIsNone(alpha_b)

        self.assertShapeEqual(
            omega,
            (
                demo_couple[1].out_channels,
                k,
                demo_couple[1].kernel_size[0],
                demo_couple[1].kernel_size[1],
            ),
        )

        self.assertShapeEqual(eigenvalues, (k,))

        self.assertIsInstance(demo_couple[0].extended_output_layer, torch.nn.Conv2d)
        self.assertIsInstance(demo_couple[1].extended_input_layer, torch.nn.Conv2d)

        demo_couple[1].sub_select_optimal_added_parameters(3)

        self.assertEqual(demo_couple[1].eigenvalues_extension.shape[0], 3)
        self.assertEqual(demo_couple[1].extended_input_layer.in_channels, 3)
        self.assertEqual(demo_couple[0].extended_output_layer.out_channels, 3)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_added_parameters_use_projected_gradient_false(
        self, bias: bool
    ):
        """
        Explicitly test the use_projected_gradient=False branch for coverage.
        """
        demo_couple = self.demo_couple[bias]
        demo_couple[1].init_computation()

        y = demo_couple[0](self.input_x)
        y = demo_couple[1](y)
        loss = torch.norm(y)
        loss.backward()

        demo_couple[1].update_computation()

        # Call with use_projected_gradient=False
        alpha, alpha_b, omega, eigenvalues = demo_couple[
            1
        ].compute_optimal_added_parameters(use_projected_gradient=False)

        self.assertShapeEqual(
            alpha,
            (
                -1,
                demo_couple[0].in_channels,
                demo_couple[0].kernel_size[0],
                demo_couple[0].kernel_size[1],
            ),
        )
        k = alpha.size(0)
        if bias:
            self.assertShapeEqual(alpha_b, (k,))
        else:
            self.assertIsNone(alpha_b)

        self.assertShapeEqual(
            omega,
            (
                demo_couple[1].out_channels,
                k,
                demo_couple[1].kernel_size[0],
                demo_couple[1].kernel_size[1],
            ),
        )

        self.assertShapeEqual(eigenvalues, (k,))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_added_parameters_empirical(self, bias: bool):
        demo_couple = self.demo_couple[bias]
        demo_couple_1 = FullConv2dGrowingModule(
            in_channels=5,
            out_channels=2,
            kernel_size=(3, 3),
            padding=1,
            use_bias=bias,
            device=global_device(),
            previous_module=demo_couple[0],
        )
        demo_couple = (demo_couple[0], demo_couple_1)
        demo_couple[0].weight.data.zero_()
        demo_couple[1].weight.data.zero_()
        if bias:
            demo_couple[0].bias.data.zero_()
            demo_couple[1].bias.data.zero_()

        demo_couple[0].store_input = True
        demo_couple[1].init_computation()
        demo_couple[1].tensor_s_growth.init()

        input_x = indicator_batch(
            (demo_couple[0].in_channels, 7, 11), device=global_device()
        )
        y = demo_couple[0](input_x)
        y = demo_couple[1](y)
        loss = ((y - input_x) ** 2).sum()
        loss.backward()

        demo_couple[1].update_computation()
        demo_couple[1].tensor_s_growth.update()

        demo_couple[1].compute_optimal_delta()
        demo_couple[1].delta_raw *= 0

        self.assertAllClose(
            -demo_couple[1].tensor_m_prev(),
            demo_couple[1].tensor_n,
            message="The tensor_m_prev should be equal to the tensor_n when the delta is zero",
        )

        demo_couple[1].compute_optimal_added_parameters()

        extension_network = torch.nn.Sequential(
            demo_couple[0].extended_output_layer,
            demo_couple[1].extended_input_layer,
        )

        amplitude_factor = 1e-2
        y = extension_network(input_x)
        new_loss = ((amplitude_factor * y - input_x) ** 2).sum().item()
        loss = loss.item()
        self.assertLess(
            new_loss,
            loss,
            msg=f"Despite the merge of new neurons the loss "
            f"has increased: {new_loss=} > {loss=}",
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_s_growth_custom_implementation(self, bias):
        """Test that FullConv2dGrowingModule has custom tensor_s_growth implementation."""
        demo = self.bias_demos[bias]

        # FullConv2dGrowingModule should have its own _tensor_s_growth attribute
        self.assertTrue(hasattr(demo, "_tensor_s_growth"))

        # Initialize computation to set up the tensor statistics
        demo.init_computation()
        x = torch.randn(2, demo.in_channels, 8, 8, device=global_device())
        output = demo(x)

        # Create a loss and backward pass to generate gradients
        loss = torch.norm(output)
        loss.backward()

        demo.update_computation()

        # tensor_s_growth should return the internal _tensor_s_growth, not previous module's tensor_s
        tensor_s_growth = demo.tensor_s_growth
        self.assertIs(tensor_s_growth, demo._tensor_s_growth)

        # Verify it's a TensorStatistic
        self.assertIsInstance(tensor_s_growth, TensorStatistic)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_s_growth_independence_from_previous_module(self, bias):
        """Test that FullConv2dGrowingModule tensor_s_growth is independent of previous module."""
        demo_couple = self.demo_couple[bias]
        demo_in, demo_out = demo_couple[0], demo_couple[1]

        # Set up a chain where demo_out has demo_in as previous_module
        demo_out.previous_module = demo_in

        # Initialize computations
        demo_in.init_computation()
        demo_out.init_computation()

        # Forward pass
        x = torch.randn(2, demo_in.in_channels, 8, 8, device=global_device())
        y = demo_in(x)
        demo_out.update_input_size(y.shape[2:])
        z = demo_out(y)

        # Create a loss and backward pass to generate gradients
        loss = torch.norm(z)
        loss.backward()

        # Update computations
        demo_in.update_computation()
        demo_out.update_computation()

        # tensor_s_growth for FullConv2dGrowingModule should NOT redirect to previous module
        # It should use its own _tensor_s_growth
        tensor_s_growth_out = demo_out.tensor_s_growth
        tensor_s_in = demo_in.tensor_s

        self.assertIsNot(tensor_s_growth_out, tensor_s_in)
        self.assertIs(tensor_s_growth_out, demo_out._tensor_s_growth)

    def test_tensor_s_growth_shape_correctness(self):
        """Test that tensor_s_growth returns tensors with correct shapes."""
        demo = self.bias_demos[True]  # Test with bias

        demo.init_computation()
        x = torch.randn(3, demo.in_channels, 6, 6, device=global_device())
        output = demo(x)

        # Create a loss and backward pass to generate gradients
        loss = torch.norm(output)
        loss.backward()

        demo.update_computation()

        # For FullConv2dGrowingModule, we need to ensure _tensor_s_growth has been computed
        # Check if tensor_s_growth has samples before calling it
        tensor_s_growth_stat = demo.tensor_s_growth
        self.assertIsInstance(tensor_s_growth_stat, TensorStatistic)

        # If it has samples, test the shape
        if tensor_s_growth_stat.samples > 0:
            tensor_s_growth = tensor_s_growth_stat()

            # For FullConv2dGrowingModule, the tensor should have specific dimensions
            # related to the unfolded input and the convolution parameters
            self.assertIsInstance(tensor_s_growth, torch.Tensor)
            self.assertEqual(len(tensor_s_growth.shape), 2)  # Should be a 2D tensor

            # Both dimensions should be equal (square matrix)
            self.assertEqual(tensor_s_growth.shape[0], tensor_s_growth.shape[1])
        else:
            # If no samples, just verify the tensor_s_growth property exists and is correct type
            self.assertIsInstance(tensor_s_growth_stat, TensorStatistic)


class TestRestrictedConv2dGrowingModule(TestConv2dGrowingModule):
    _tested_class = RestrictedConv2dGrowingModule

    def test_zero_bottleneck_restricted(self):
        """Test behavior when bottleneck is fully resolved
        with parameter change for RestrictedConv2d."""
        # Use predefined demo_couple objects
        demo_layer_1, demo_layer_2 = self.demo_couple[
            False
        ]  # Use without bias for simplicity

        net = torch.nn.Sequential(demo_layer_1, demo_layer_2)
        demo_layer_2.init_computation()

        # Use indicator batch for Conv2d - each sample has 1 in different spatial locations
        input_x = indicator_batch(
            (demo_layer_1.in_channels, 5, 5), device=global_device()
        )
        y = net(input_x)
        loss = torch.norm(y) ** 2 / 2
        loss.backward()
        demo_layer_2.update_computation()
        demo_layer_2.compute_optimal_updates()

        # For RestrictedConv2d, tensor_n should be zero when bottleneck is fully resolved
        self.assertAllClose(
            demo_layer_2.tensor_n, torch.zeros_like(demo_layer_2.tensor_n), atol=1e-7
        )
        self.assertAllClose(
            demo_layer_2.eigenvalues_extension,
            torch.zeros_like(demo_layer_2.eigenvalues_extension),
            atol=1e-7,
        )

    def test_compute_m_prev_without_intermediate_input_restricted(self):
        """Check that the batch size is computed using stored variables for RestrictedConv2d"""
        # Use predefined demo_couple objects
        demo_layer_1, demo_layer_2 = self.demo_couple[
            False
        ]  # Use without bias for simplicity

        net = torch.nn.Sequential(demo_layer_1, demo_layer_2)
        demo_layer_2.store_pre_activity = True
        demo_layer_1.store_input = True
        demo_layer_2.tensor_m_prev.init()

        # Create Conv2d input tensor
        input_x = torch.randn(11, demo_layer_1.in_channels, 5, 5, device=global_device())
        loss = net(input_x).sum()
        loss.backward()

        demo_layer_2.tensor_m_prev.update()
        self.assertEqual(demo_layer_2.tensor_m_prev.samples, input_x.size(0))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_s_growth_redirection(self, bias: bool):
        with self.assertRaises(ValueError):
            self.bias_demos[bias].tensor_s_growth.init()

        demo_in, demo_out = self.demo_couple[bias]
        demo_in.store_input = True
        demo_in.tensor_s.init()
        demo_in(self.input_x)
        demo_in.tensor_s.update()

        # tensor_s_growth is a property redirecting to previous_module.tensor_s
        self.assertTrue(torch.equal(demo_out.tensor_s_growth(), demo_in.tensor_s()))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_linear_layer_of_tensor(self, bias: bool):
        demo_layer = demo_out = RestrictedConv2dGrowingModule(
            in_channels=2,
            out_channels=3,
            kernel_size=(5, 5),
            padding=2,
            use_bias=bias,
            device=global_device(),
        )
        reference_layer = torch.nn.Linear(
            demo_layer.in_channels,
            demo_layer.out_channels,
            bias=bias,
            device=global_device(),
        )

        constructed_layer = demo_layer.linear_layer_of_tensor(
            reference_layer.weight.data, reference_layer.bias.data if bias else None
        )
        x = torch.randn(5, 7, 11, demo_layer.in_channels, device=global_device())
        y_ref = reference_layer(x)

        x = x.permute(0, 3, 1, 2)
        y_test = constructed_layer(x)
        y_test = y_test.permute(0, 2, 3, 1)

        self.assertAllClose(
            y_ref,
            y_test,
            atol=1e-6,
            message=f"The constructed convolution is not similar to a linear layer",
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_m_prev_update(self, bias: bool):
        demo_in, demo_out = self.demo_couple[bias]

        # TODO: remove this line to handle the general case
        # indeed this ensure that the output size (height, width) of
        # the first layer is the same as the second layer
        demo_out.padding = (2, 2)

        demo_in.store_input = True
        demo_out.store_pre_activity = True
        demo_out.tensor_m_prev.init()

        x = demo_in(self.input_x)
        y = demo_out(x)
        loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        loss.backward()

        demo_in.update_input_size()
        demo_out.tensor_m_prev.update()

        s0 = demo_in.in_channels * demo_in.kernel_size[0] * demo_in.kernel_size[1] + (
            1 if bias else 0
        )
        s1 = demo_out.out_channels

        self.assertShapeEqual(demo_out.tensor_m_prev(), (s0, s1))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_cross_covariance_update(self, bias: bool):
        demo_in, demo_out = self.demo_couple[bias]
        demo_out.__class__ = RestrictedConv2dGrowingModule

        demo_in.store_input = True
        demo_out.store_input = True
        demo_out.cross_covariance.init()

        x = demo_in(self.input_x)
        _ = demo_out(x)

        demo_in.update_input_size()
        demo_out.update_input_size()
        demo_out.cross_covariance.update()

        s1 = demo_in.in_channels * demo_in.kernel_size[0] * demo_in.kernel_size[1] + (
            1 if bias else 0
        )
        s2 = demo_out.in_channels * demo_out.kernel_size[0] * demo_out.kernel_size[1] + (
            1 if bias else 0
        )

        self.assertShapeEqual(demo_out.cross_covariance(), (s1, s2))

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_n_computation(self, bias: bool):
        demo_in, demo_out = self.demo_couple[bias]

        demo_in.store_input = True
        demo_out.store_input = True
        demo_out.store_pre_activity = True
        demo_out.tensor_m_prev.init()
        demo_out.cross_covariance.init()

        x = demo_in(self.input_x)
        y = demo_out(x)
        loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        loss.backward()

        demo_in.update_input_size()
        demo_out.update_input_size()
        demo_out.tensor_m_prev.update()
        demo_out.cross_covariance.update()

        demo_out.delta_raw = torch.zeros(
            demo_out.out_channels,
            demo_out.in_channels * demo_out.kernel_size[0] * demo_out.kernel_size[1]
            + bias,
            device=global_device(),
        )

        n = demo_out.tensor_n
        self.assertIsInstance(n, torch.Tensor)
        self.assertAllClose(
            n,
            -demo_out.tensor_m_prev(),
            message="The tensor_n should be equal to the tensor_m_prev when the delta is zero",
        )

        demo_out.delta_raw = torch.randn_like(demo_out.delta_raw)
        n = demo_out.tensor_n
        self.assertIsInstance(n, torch.Tensor)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_added_parameters(self, bias: bool):
        demo_in, demo_out = self.demo_couple[bias]

        demo_out.init_computation()

        x = demo_in(self.input_x)
        y = demo_out(x)
        loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        loss.backward()

        demo_out.update_computation()

        demo_out.delta_raw = torch.zeros(
            demo_out.out_channels,
            demo_out.in_channels * demo_out.kernel_size[0] * demo_out.kernel_size[1]
            + bias,
            device=global_device(),
        )

        alpha, alpha_b, omega, eigs = demo_out.compute_optimal_added_parameters()

        self.assertIsInstance(alpha, torch.Tensor)
        self.assertIsInstance(omega, torch.Tensor)
        self.assertIsInstance(eigs, torch.Tensor)
        if bias:
            self.assertIsInstance(alpha_b, torch.Tensor)
        else:
            self.assertIsNone(alpha_b)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_compute_optimal_added_parameters_use_projected_gradient_false(
        self, bias: bool
    ):
        """Test compute_optimal_added_parameters with use_projected_gradient=False for RestrictedConv2dGrowingModule."""
        demo_in, demo_out = self.demo_couple[bias]

        demo_out.init_computation()

        x = demo_in(self.input_x)
        y = demo_out(x)
        loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
        loss.backward()

        demo_out.update_computation()

        demo_out.delta_raw = torch.zeros(
            demo_out.out_channels,
            demo_out.in_channels * demo_out.kernel_size[0] * demo_out.kernel_size[1]
            + bias,
            device=global_device(),
        )

        # Test with use_projected_gradient=False
        alpha, alpha_b, omega, eigs = demo_out.compute_optimal_added_parameters(
            use_projected_gradient=False
        )

        self.assertIsInstance(alpha, torch.Tensor)
        self.assertIsInstance(omega, torch.Tensor)
        self.assertIsInstance(eigs, torch.Tensor)
        if bias:
            self.assertIsInstance(alpha_b, torch.Tensor)
        else:
            self.assertIsNone(alpha_b)


if __name__ == "__main__":
    main()
