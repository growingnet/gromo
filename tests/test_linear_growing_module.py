from copy import deepcopy
from unittest import TestCase, main

import torch

from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize


def theoretical_s_1(n, c):
    """
    Compute the theoretical value of the tensor S for the input and output of
    weight matrix W = (0 ... 0 \\ 0 1 0 ... 0 \\ 0 0 2 0 ... 0 \\ ... \\ 1 ... 1).

    Parameters
    ----------
    n:
        number of samples
    c:
        number of features

    Returns
    -------
    x1:
        input tensor 1
    x2:
        input tensor 2
    is1:
        theoretical value of the tensor nS for x1
    is2:
        theoretical value of the tensor 2nS for (x1, x2)
    os1:
        theoretical value of the tensor nS for the output of W(x1)
    os2:
        theoretical value of the tensor 2nS for the output of W((x1, x2))
    """

    va = torch.arange(c)
    v1 = torch.ones(c, dtype=torch.long)
    is0 = va.view(-1, 1) @ va.view(1, -1)
    isc = va.view(-1, 1) @ v1.view(1, -1)
    isc = isc + isc.T
    is1 = torch.ones(c, c)
    va_im = torch.arange(c + 1) ** 2
    va_im[-1] = c * (c - 1) // 2
    v1_im = torch.arange(c + 1)
    os0 = va_im.view(-1, 1) @ va_im.view(1, -1)
    osc = va_im.view(-1, 1) @ v1_im.view(1, -1)
    osc = osc + osc.T
    os1 = v1_im.view(-1, 1) @ v1_im.view(1, -1)

    x1 = torch.ones(n, c)
    x1 *= torch.arange(n).view(-1, 1)

    x2 = torch.tile(torch.arange(c), (n, 1))
    x2 += torch.arange(n).view(-1, 1)

    is_theory_1 = n * (n - 1) * (2 * n - 1) // 6 * is1

    os_theory_1 = n * (n - 1) * (2 * n - 1) // 6 * os1

    is_theory_2 = n * is0 + n * (n - 1) // 2 * isc + n * (n - 1) * (2 * n - 1) // 3 * is1

    os_theory_2 = n * os0 + n * (n - 1) // 2 * osc + n * (n - 1) * (2 * n - 1) // 3 * os1

    return x1, x2, is_theory_1, is_theory_2, os_theory_1, os_theory_2


class TestLinearGrowingModule(TorchTestCase):
    def setUp(self):
        self.n = 11
        # This assert is checking that the test is correct and not that the code is correct
        # that why it is not a self.assert*
        assert self.n % 2 == 1
        self.c = 5

        self.weight_matrix_1 = torch.ones(self.c + 1, self.c, device=global_device())
        self.weight_matrix_1[:-1] = torch.diag(torch.arange(self.c)).to(global_device())
        # W = (0 ... 0 \\ 0 1 0 ... 0 \\ 0 0 2 0 ... 0 \\ ... \\ 1 ... 1)

        torch.manual_seed(0)
        self.input_x = torch.randn((11, 5), device=global_device())
        self.demo_layers = dict()
        for bias in (True, False):
            self.demo_layer_1 = LinearGrowingModule(
                5,
                3,
                use_bias=bias,
                name=f"L1({'bias' if bias else 'no_bias'})",
                device=global_device(),
            )
            self.demo_layer_2 = LinearGrowingModule(
                3,
                7,
                use_bias=bias,
                name=f"L2({'bias' if bias else 'no_bias'})",
                previous_module=self.demo_layer_1,
                device=global_device(),
            )
            self.demo_layers[bias] = (self.demo_layer_1, self.demo_layer_2)

    def test_compute_s(self):
        x1, x2, is_th_1, is_th_2, os_th_1, os_th_2 = theoretical_s_1(self.n, self.c)

        output_module = LinearMergeGrowingModule(in_features=self.c + 1, name="output")
        layer = LinearGrowingModule(
            self.c, self.c + 1, use_bias=False, name="layer1", next_module=output_module
        )
        output_module.set_previous_modules([layer])

        net = torch.nn.Sequential(layer, output_module)

        layer.layer.weight.data = self.weight_matrix_1

        layer.tensor_s.init()
        layer.store_input = True
        output_module.tensor_s.init()
        output_module.store_activity = True

        # output_module.store_input = True
        output_module.previous_tensor_s.init()

        # forward pass 1
        _ = net(x1.float().to(global_device()))
        layer.tensor_s.update()
        output_module.tensor_s.update()
        output_module.previous_tensor_s.update()

        # check the values
        # input S
        self.assertAllClose(
            layer.tensor_s(), is_th_1.float().to(global_device()) / self.n
        )
        # output S
        self.assertAllClose(
            output_module.tensor_s()[: self.c + 1, : self.c + 1],
            os_th_1.float().to(global_device()) / self.n,
        )

        # input S computed from merge layer
        self.assertAllClose(
            output_module.previous_tensor_s(),
            is_th_1.float().to(global_device()) / self.n,
        )

        # forward pass 2
        _ = net(x2.float().to(global_device()))
        layer.tensor_s.update()
        output_module.tensor_s.update()
        output_module.previous_tensor_s.update()

        # check the values
        self.assertAllClose(
            layer.tensor_s(), is_th_2.float().to(global_device()) / (2 * self.n)
        )
        self.assertAllClose(
            output_module.tensor_s()[: self.c + 1, : self.c + 1],
            os_th_2.float().to(global_device()) / (2 * self.n),
        )

        self.assertAllClose(
            output_module.previous_tensor_s(),
            is_th_2.float().to(global_device()) / (2 * self.n),
        )

    def test_compute_delta(self):
        for reduction in {"mixed"}:  # { "mean", "sum"} do not work
            # mean: batch is divided by the number of samples in the batch
            # and the total is divided by the number of batches
            # mixed: batch is not divided
            # but the total is divided by the number of batches * batch_size
            # sum: batch is not divided
            # and the total is not divided
            batch_red = self.c if reduction == "mean" else 1
            loss_func = lambda x, y: torch.norm(x - y) ** 2 / batch_red

            for alpha in (0.1, 1.0, 10.0):
                layer = LinearGrowingModule(self.c, self.c, use_bias=False, name="layer1")
                layer.layer.weight.data = torch.zeros_like(
                    layer.layer.weight, device=global_device()
                )
                layer.tensor_s.init()
                layer.tensor_m.init()
                layer.store_input = True
                layer.store_pre_activity = True

                for _ in range(nb_batch := 3):
                    x = alpha * torch.eye(self.c, device=global_device())
                    y = layer(x)
                    loss = loss_func(x, y)
                    loss.backward()

                    layer.update_computation()

                # S
                self.assertAllClose(
                    layer.tensor_s(),
                    alpha**2 * torch.eye(self.c, device=global_device()) / self.c,
                    message=f"Error in S for {reduction=}, {alpha=}",
                )

                # dL / dA
                self.assertAllClose(
                    layer.pre_activity.grad,
                    -2 * alpha * torch.eye(self.c, device=global_device()) / batch_red,
                    message=f"Error in dL/dA for {reduction=}, {alpha=}",
                )

                # M
                self.assertAllClose(
                    layer.tensor_m(),
                    -2
                    * alpha**2
                    * torch.eye(self.c, device=global_device())
                    / self.c
                    / batch_red,
                    message=f"Error in M for {reduction=}, {alpha=}",
                )

                # dW*
                w, _, fo = layer.compute_optimal_delta()
                self.assertAllClose(
                    w,
                    -2 * torch.eye(self.c, device=global_device()) / batch_red,
                    message=f"Error in dW* for {reduction=}, {alpha=}",
                )

                factors = {
                    "mixed": 1,
                    "mean": self.c,  # batch size to compensate the batch normalization
                    "sum": self.c * nb_batch,  # number of samples
                }
                # <dW*, dL/dA>
                self.assertAlmostEqual(
                    fo.item(),
                    4 * alpha**2 / batch_red**2 * factors[reduction],
                    places=3,
                    msg=f"Error in <dW*, dL/dA> for {reduction=}, {alpha=}",
                )

    def test_str(self):
        self.assertIsInstance(str(LinearGrowingModule(5, 5)), str)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_extended_forward_out(self, bias):
        torch.manual_seed(0)
        # fixed layers
        l0 = torch.nn.Linear(5, 1, bias=bias, device=global_device())
        l_ext = torch.nn.Linear(5, 2, bias=bias, device=global_device())
        l_delta = torch.nn.Linear(5, 1, bias=bias, device=global_device())

        # changed layer
        layer = LinearGrowingModule(
            5, 1, use_bias=bias, name="layer1", device=global_device()
        )
        layer.weight.data.copy_(l0.weight.data)
        if bias:
            layer.bias.data.copy_(l0.bias.data)
        layer.optimal_delta_layer = l_delta
        layer.extended_output_layer = l_ext

        for gamma, gamma_next in ((0.0, 0.0), (1.0, 1.5), (5.0, 5.5)):
            layer.scaling_factor = gamma
            layer._scaling_factor_next_module[0] = gamma_next
            x = torch.randn((10, 5), device=global_device())
            self.assertAllClose(layer(x), l0(x))

            y_ext_1, y_ext_2 = layer.extended_forward(x)

            self.assertAllClose(y_ext_1, l0(x) - gamma**2 * l_delta(x))
            self.assertAllClose(y_ext_2, gamma_next * l_ext(x))

        layer.apply_change(apply_previous=False)
        y = layer(x)
        self.assertAllClose(y, l0(x) - gamma**2 * l_delta(x))

        layer._apply_output_changes()
        y_changed = layer(x)
        y_changed_1 = y_changed[:, :1]
        y_changed_2 = y_changed[:, 1:]
        self.assertAllClose(y_changed_1, l0(x) - gamma**2 * l_delta(x))
        self.assertAllClose(
            y_changed_2,
            gamma_next * l_ext(x),
            atol=1e-7,
            message=f"Error in applying change",
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_extended_forward_in(self, bias):
        torch.manual_seed(0)
        # fixed layers
        l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
        l_ext = torch.nn.Linear(5, 1, bias=bias, device=global_device())
        if bias:
            l_ext.bias.data.fill_(0)
        l_delta = torch.nn.Linear(3, 1, bias=bias, device=global_device())

        # changed layer
        layer = LinearGrowingModule(
            3, 1, use_bias=bias, name="layer1", device=global_device()
        )
        layer.weight.data.copy_(l0.weight.data)
        if bias:
            layer.bias.data.copy_(l0.bias.data)
        layer.optimal_delta_layer = l_delta
        layer.extended_input_layer = l_ext

        for gamma in (0.0, 1.0, 5.0):
            layer.zero_grad()
            layer.scaling_factor = gamma
            x = torch.randn((10, 3), device=global_device())
            x_ext = torch.randn((10, 5), device=global_device())
            self.assertAllClose(layer(x), l0(x))

            y, none = layer.extended_forward(x, x_ext)
            self.assertIsNone(none)

            self.assertAllClose(y, l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext))

            torch.norm(y).backward()

            self.assertIsNotNone(layer.scaling_factor.grad)

        layer.apply_change(apply_previous=False)
        x_cat = torch.concatenate((x, x_ext), dim=1)
        y = layer(x_cat)
        self.assertAllClose(
            y,
            l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext),
            message=(f"Error in applying change"),
        )

    def test_number_of_parameters(self):
        for in_layer in (1, 3):
            for out_layer in (1, 3):
                for bias in (True, False):
                    layer = LinearGrowingModule(
                        in_layer, out_layer, use_bias=bias, name="layer1"
                    )
                    self.assertEqual(
                        layer.number_of_parameters(),
                        in_layer * out_layer + bias * out_layer,
                    )

    def test_layer_in_extension(self):
        layer = LinearGrowingModule(3, 1, use_bias=False, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(1, 3))
        self.assertEqual(layer.number_of_parameters(), 3)
        self.assertEqual(layer.in_features, 3)

        x = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[6.0]]))

        layer.layer_in_extension(torch.tensor([[10]], dtype=torch.float32))
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.in_features, 4)
        self.assertEqual(layer.layer.in_features, 4)
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[46.0]]))

    def test_layer_out_extension(self):
        # without bias
        layer = LinearGrowingModule(1, 3, use_bias=False, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        self.assertEqual(layer.number_of_parameters(), 3)
        self.assertEqual(layer.out_features, 3)
        x = torch.tensor([[1]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[1.0, 1.0, 1.0]]))

        layer.layer_out_extension(torch.tensor([[10]], dtype=torch.float32))
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.out_features, 4)
        self.assertEqual(layer.layer.out_features, 4)

        y = layer(x)
        self.assertAllClose(y, torch.tensor([[1.0, 1.0, 1.0, 10.0]]))

        # with bias
        layer = LinearGrowingModule(1, 3, use_bias=True, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        layer.bias = torch.nn.Parameter(10 * torch.ones(3))
        self.assertEqual(layer.number_of_parameters(), 6)
        self.assertEqual(layer.out_features, 3)
        x = torch.tensor([[-1]], dtype=torch.float32)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[9.0, 9.0, 9.0]]))

        layer.layer_out_extension(
            torch.tensor([[10]], dtype=torch.float32),
            bias=torch.tensor([100], dtype=torch.float32),
        )
        self.assertEqual(layer.number_of_parameters(), 8)
        self.assertEqual(layer.out_features, 4)
        y = layer(x)
        self.assertAllClose(y, torch.tensor([[9.0, 9.0, 9.0, 90.0]]))

    def test_apply_change_delta_layer(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            l_delta = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            layer = LinearGrowingModule(
                3, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)
            layer.optimal_delta_layer = l_delta

            if bias:
                layer.bias.data.copy_(l0.bias.data)

            gamma = 5.0
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)

            x = torch.randn((10, 3), device=global_device())
            y = layer(x)
            self.assertAllClose(y, l0(x) - gamma**2 * l_delta(x))

    def test_apply_change_out_extension(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            l_ext = torch.nn.Linear(5, 2, bias=bias, device=global_device())
            layer = LinearGrowingModule(
                5, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)

            if bias:
                layer.bias.data.copy_(l0.bias.data)
            layer.extended_output_layer = l_ext

            gamma = 5.0
            gamma_next = 5.5
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)
            self.assertAllClose(layer.weight.data, l0.weight.data)

            layer._scaling_factor_next_module[0] = gamma_next
            layer._apply_output_changes()

            x = torch.randn((10, 5), device=global_device())
            y = layer(x)
            y1 = y[:, :1]
            y2 = y[:, 1:]
            self.assertAllClose(y1, l0(x))
            self.assertAllClose(y2, gamma_next * l_ext(x))

    def test_apply_change_in_extension(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            l_ext = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            if bias:
                l_ext.bias.data.fill_(0)
            layer = LinearGrowingModule(
                3, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)

            if bias:
                layer.bias.data.copy_(l0.bias.data)
            layer.extended_input_layer = l_ext

            gamma = 5.0
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)

            x_cat = torch.randn((10, 8), device=global_device())
            y = layer(x_cat)
            x = x_cat[:, :3]
            x_ext = x_cat[:, 3:]

            self.assertAllClose(
                y,
                l0(x) + gamma * l_ext(x_ext),
                atol=1e-7,
                message=(
                    f"Error in applying change: "
                    f"{(y - l0(x) - gamma * l_ext(x_ext)).abs().max():.2e}"
                ),
            )

    def test_sub_select_optimal_added_parameters_out(self):
        for bias in {True, False}:
            layer = LinearGrowingModule(3, 1, use_bias=bias, name="layer1")
            layer.extended_output_layer = torch.nn.Linear(3, 2, bias=bias)

            new_layer = torch.nn.Linear(3, 1, bias=bias)
            new_layer.weight.data = layer.extended_output_layer.weight.data[0].view(1, -1)
            if bias:
                new_layer.bias.data = layer.extended_output_layer.bias.data[0].view(1)

            layer._sub_select_added_output_dimension(1)

            self.assertAllClose(layer.extended_output_layer.weight, new_layer.weight)

            self.assertAllClose(layer.extended_output_layer.weight, new_layer.weight)

            if bias:
                self.assertAllClose(layer.extended_output_layer.bias, new_layer.bias)

    def test_sub_select_optimal_added_parameters_in(self):
        bias = False
        layer = LinearGrowingModule(1, 3, use_bias=bias, name="layer1")
        layer.extended_input_layer = torch.nn.Linear(2, 3, bias=bias)
        layer.eigenvalues_extension = torch.tensor([2.0, 1.0])

        new_layer = torch.nn.Linear(1, 3, bias=bias)
        new_layer.weight.data = layer.extended_input_layer.weight.data[:, 0].view(-1, 1)
        if bias:
            new_layer.bias.data = layer.extended_input_layer.bias.data

        layer.sub_select_optimal_added_parameters(1, sub_select_previous=False)

        self.assertAllClose(layer.extended_input_layer.weight, new_layer.weight)

        if bias:
            self.assertAllClose(layer.extended_input_layer.bias, new_layer.bias)

        self.assertAllClose(layer.eigenvalues_extension, torch.tensor([2.0]))

    def test_sample_number_invariant(self):
        invariants = [
            "tensor_s",
            "tensor_m",
            # "pre_activity",
            # "input",
            "delta_raw",
            "optimal_delta_layer",
            "parameter_update_decrease",
            "eigenvalues_extension",
            "tensor_m_prev",
            "cross_covariance",
        ]

        def linear_layer_equality(layer1, layer2, rtol=1e-5, atol=1e-8):
            return torch.allclose(
                layer1.weight, layer2.weight, atol=atol, rtol=rtol
            ) and (
                (layer1.bias is None and layer2.bias is None)
                or (torch.allclose(layer1.bias, layer2.bias, atol=atol, rtol=rtol))
            )

        def set_invariants(layer: LinearGrowingModule):
            _reference = dict()
            for inv in invariants:
                inv_value = getattr(layer, inv)
                if isinstance(inv_value, torch.Tensor):
                    _reference[inv] = inv_value.clone()
                elif isinstance(inv_value, torch.nn.Linear):
                    _reference[inv] = deepcopy(inv_value)
                elif isinstance(inv_value, TensorStatistic):
                    _reference[inv] = inv_value().clone()
                else:
                    raise ValueError(f"Invalid type for {inv} ({type(inv_value)})")
            return _reference

        def check_invariants(
            layer: LinearGrowingModule, reference: dict, rtol=1e-5, atol=1e-8
        ):
            for inv in invariants:
                new_inv_value = getattr(layer, inv)
                if isinstance(new_inv_value, torch.Tensor):
                    self.assertAllClose(
                        reference[inv],
                        new_inv_value,
                        rtol=rtol,
                        atol=atol,
                        message=f"Error on {inv=}",
                    )
                elif isinstance(new_inv_value, torch.nn.Linear):
                    self.assertTrue(
                        linear_layer_equality(
                            reference[inv], new_inv_value, rtol=rtol, atol=atol
                        ),
                        f"Error on {inv=}",
                    )
                elif isinstance(new_inv_value, TensorStatistic):
                    self.assertAllClose(
                        reference[inv],
                        new_inv_value(),
                        rtol=rtol,
                        atol=atol,
                        message=f"Error on {inv=}",
                    )
                else:
                    raise ValueError(f"Invalid type for {inv} ({type(new_inv_value)})")

        torch.manual_seed(0)
        layer_in = LinearGrowingModule(
            in_features=5,
            out_features=3,
            name="layer_in",
            post_layer_function=torch.nn.SELU(),
        )
        layer_out = LinearGrowingModule(
            in_features=3, out_features=7, name="layer_out", previous_module=layer_in
        )
        net = torch.nn.Sequential(layer_in, layer_out)

        def update_computation(double_batch=False):
            loss = torch.nn.MSELoss(reduction="sum")
            # loss = lambda x, y: torch.norm(x - y) ** 2
            torch.manual_seed(0)
            net.zero_grad()
            x = torch.randn((10, 5), device=global_device())
            if double_batch:
                x = torch.cat((x, x), dim=0)
            y = net(x)
            loss = loss(y, torch.zeros_like(y))
            loss.backward()
            layer_out.update_computation()
            layer_in.tensor_s.update()

        layer_out.init_computation()
        layer_in.tensor_s.init()

        update_computation()
        layer_out.compute_optimal_updates()

        reference = set_invariants(layer_out)

        for db in (False, True):
            update_computation(double_batch=db)
            layer_out.compute_optimal_updates()
            check_invariants(layer_out, reference)

        # simple test update without natural gradient
        layer_out.compute_optimal_updates(zero_delta=True)

    def test_compute_optimal_added_parameters(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                demo_layers = self.demo_layers[bias]
                demo_layers[0].store_input = True
                demo_layers[1].init_computation()
                demo_layers[1].tensor_s_growth.init()

                y = demo_layers[0](self.input_x)
                y = demo_layers[1](y)
                loss = torch.norm(y)
                loss.backward()

                demo_layers[1].update_computation()
                demo_layers[1].tensor_s_growth.update()

                demo_layers[1].compute_optimal_delta()
                alpha, alpha_b, omega, eigenvalues = demo_layers[
                    1
                ].compute_optimal_added_parameters()

                self.assertShapeEqual(
                    alpha,
                    (-1, demo_layers[0].in_features),
                )
                k = alpha.size(0)
                if bias:
                    self.assertShapeEqual(alpha_b, (k,))
                else:
                    self.assertIsNone(alpha_b)

                self.assertShapeEqual(
                    omega,
                    (
                        demo_layers[1].out_features,
                        k,
                    ),
                )

                self.assertShapeEqual(eigenvalues, (k,))

                self.assertIsInstance(
                    demo_layers[0].extended_output_layer, torch.nn.Linear
                )
                self.assertIsInstance(
                    demo_layers[1].extended_input_layer, torch.nn.Linear
                )

                # those tests are not working yet
                demo_layers[1].sub_select_optimal_added_parameters(2)
                self.assertEqual(demo_layers[1].eigenvalues_extension.shape[0], 2)
                self.assertEqual(demo_layers[1].extended_input_layer.in_features, 2)
                self.assertEqual(demo_layers[0].extended_output_layer.out_features, 2)

    def test_tensor_s_growth(self):
        for bias in (True, False):
            with self.subTest(bias=bias):
                demo_layers = self.demo_layers[bias]
                demo_layers[0].store_input = True
                demo_layers[1].tensor_s_growth.init()

                y = demo_layers[0](self.input_x)
                y = demo_layers[1](y)
                loss = torch.norm(y)
                loss.backward()

                demo_layers[1].tensor_s_growth.update()

                self.assertEqual(
                    demo_layers[1].tensor_s_growth.samples,
                    self.input_x.size(0),
                )
                s = demo_layers[0].in_features + demo_layers[0].use_bias
                self.assertShapeEqual(demo_layers[1].tensor_s_growth(), (s, s))

    def test_tensor_s_growth_errors(self):
        with self.assertRaises(AttributeError):
            self.demo_layers[True][1].tensor_s_growth = 1

        with self.assertRaises(ValueError):
            _ = self.demo_layers[True][0].tensor_s_growth


class TestLinearMergeGrowingModule(TorchTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.demo_modules = dict()
        for bias in (True, False):
            demo_merge = LinearMergeGrowingModule(
                in_features=3, name="merge", device=global_device()
            )
            demo_merge_prev = LinearGrowingModule(
                5,
                3,
                use_bias=bias,
                name="merge_prev",
                device=global_device(),
                next_module=demo_merge,
            )
            demo_merge_next = LinearGrowingModule(
                3,
                7,
                use_bias=bias,
                name="merge_next",
                device=global_device(),
                previous_module=demo_merge,
            )
            demo_merge.set_previous_modules([demo_merge_prev])
            demo_merge.set_next_modules([demo_merge_next])
            self.demo_modules[bias] = {
                "add": demo_merge,
                "prev": demo_merge_prev,
                "next": demo_merge_next,
                "seq": torch.nn.Sequential(demo_merge_prev, demo_merge, demo_merge_next),
            }
        self.input_x = torch.randn((11, 5), device=global_device())

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_init(self, bias):
        self.assertIsInstance(self.demo_modules[bias]["add"], LinearMergeGrowingModule)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_input_storage(self, bias):
        demo_layers = self.demo_modules[bias]
        demo_layers["next"].store_input = True
        self.assertEqual(demo_layers["add"].store_activity, 1)
        self.assertTrue(not demo_layers["next"]._internal_store_input)
        self.assertIsNone(demo_layers["next"].input)

        _ = demo_layers["seq"](self.input_x)

        self.assertShapeEqual(
            demo_layers["next"].input,
            (self.input_x.size(0), demo_layers["next"].in_features),
        )

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_activity_storage(self, bias):
        demo_layers = self.demo_modules[bias]
        demo_layers["prev"].store_pre_activity = True
        self.assertEqual(demo_layers["add"].store_input, 1)
        self.assertTrue(not demo_layers["prev"]._internal_store_pre_activity)
        self.assertIsNone(demo_layers["prev"].pre_activity)

        _ = demo_layers["seq"](self.input_x)

        self.assertShapeEqual(
            demo_layers["prev"].pre_activity,
            (self.input_x.size(0), demo_layers["prev"].out_features),
        )

    def test_update_scaling_factor(self):
        demo_layers = self.demo_modules[True]

        demo_layers["add"].update_scaling_factor(scaling_factor=0.5)
        self.assertEqual(demo_layers["prev"]._scaling_factor_next_module.item(), 0.5)
        self.assertEqual(demo_layers["prev"].scaling_factor.item(), 0.0)
        self.assertEqual(demo_layers["next"].scaling_factor.item(), 0.5)

    def test_update_scaling_factor_incorrect_input_module(self):
        demo_layers = self.demo_modules[True]
        demo_layers["add"].previous_modules = [demo_layers["prev"], torch.nn.Linear(7, 3)]
        with self.assertRaises(TypeError):
            demo_layers["add"].update_scaling_factor(scaling_factor=0.5)

    def test_update_scaling_factor_incorrect_output_module(self):
        demo_layers = self.demo_modules[True]
        demo_layers["add"].set_next_modules([demo_layers["next"], torch.nn.Linear(3, 7)])
        with self.assertRaises(TypeError):
            demo_layers["add"].update_scaling_factor(scaling_factor=0.5)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_set_previous_next_modules(self, bias):
        demo_layers = self.demo_modules[bias]
        new_input_layer = LinearGrowingModule(
            2,
            3,
            use_bias=bias,
            name="new_prev",
            device=global_device(),
            next_module=demo_layers["add"],
        )
        new_output_layer = LinearGrowingModule(
            3,
            2,
            use_bias=bias,
            name="new_next",
            device=global_device(),
            previous_module=demo_layers["add"],
        )

        self.assertEqual(
            demo_layers["add"].sum_in_features(), demo_layers["prev"].in_features
        )
        self.assertEqual(
            demo_layers["add"].sum_in_features(with_bias=True),
            demo_layers["prev"].in_features + bias,
        )
        self.assertEqual(
            demo_layers["add"].sum_out_features(), demo_layers["next"].out_features
        )

        demo_layers["add"].set_previous_modules([demo_layers["prev"], new_input_layer])
        demo_layers["add"].set_next_modules([demo_layers["next"], new_output_layer])

        self.assertEqual(
            demo_layers["add"].sum_in_features(),
            demo_layers["prev"].in_features + new_input_layer.in_features,
        )

        self.assertEqual(
            demo_layers["add"].sum_in_features(with_bias=True),
            demo_layers["prev"].in_features + bias + new_input_layer.in_features + bias,
        )

        self.assertEqual(
            demo_layers["add"].sum_out_features(),
            demo_layers["next"].out_features + new_output_layer.out_features,
        )


if __name__ == "__main__":
    main()
