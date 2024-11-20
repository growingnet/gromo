from copy import deepcopy
from unittest import TestCase, main

import torch

from gromo.linear_growing_module import LinearAdditionGrowingModule, LinearGrowingModule
from gromo.tensor_statistic import TensorStatistic
from gromo.utils.utils import global_device


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


class TestLinearGrowingModule(TestCase):
    def setUp(self):
        self.n = 11
        assert self.n % 2 == 1
        self.c = 5

        self.weight_matrix_1 = torch.ones(self.c + 1, self.c, device=global_device())
        self.weight_matrix_1[:-1] = torch.diag(torch.arange(self.c)).to(global_device())
        # W = (0 ... 0 \\ 0 1 0 ... 0 \\ 0 0 2 0 ... 0 \\ ... \\ 1 ... 1)

    def test_compute_s(self):
        x1, x2, is_th_1, is_th_2, os_th_1, os_th_2 = theoretical_s_1(self.n, self.c)

        output_module = LinearAdditionGrowingModule(in_features=self.c + 1, name="output")
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
        self.assertTrue(
            torch.allclose(layer.tensor_s(), is_th_1.float().to(global_device()) / self.n)
        )
        # output S
        self.assertTrue(
            torch.allclose(
                output_module.tensor_s()[: self.c + 1, : self.c + 1],
                os_th_1.float().to(global_device()) / self.n,
            )
        )
        # input S computed from addition layer
        self.assertTrue(
            torch.allclose(
                output_module.previous_tensor_s(),
                is_th_1.float().to(global_device()) / self.n,
            )
        )

        # forward pass 2
        _ = net(x2.float().to(global_device()))
        layer.tensor_s.update()
        output_module.tensor_s.update()
        output_module.previous_tensor_s.update()

        # check the values
        self.assertTrue(
            torch.allclose(
                layer.tensor_s(), is_th_2.float().to(global_device()) / (2 * self.n)
            )
        )
        self.assertTrue(
            torch.allclose(
                output_module.tensor_s()[: self.c + 1, : self.c + 1],
                os_th_2.float().to(global_device()) / (2 * self.n),
            )
        )
        self.assertTrue(
            torch.allclose(
                output_module.previous_tensor_s(),
                is_th_2.float().to(global_device()) / (2 * self.n),
            )
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
                self.assertTrue(
                    torch.allclose(
                        layer.tensor_s(),
                        alpha**2 * torch.eye(self.c, device=global_device()) / self.c,
                    ),
                    f"Error in S for {reduction=}, {alpha=}",
                )

                # dL / dA
                self.assertTrue(
                    torch.allclose(
                        layer.pre_activity.grad,
                        -2
                        * alpha
                        * torch.eye(self.c, device=global_device())
                        / batch_red,
                    ),
                    f"Error in dL/dA for {reduction=}, {alpha=}",
                )

                # M
                self.assertTrue(
                    torch.allclose(
                        layer.tensor_m(),
                        -2
                        * alpha**2
                        * torch.eye(self.c, device=global_device())
                        / self.c
                        / batch_red,
                    ),
                    f"Error in M for {reduction=}, {alpha=}",
                )

                # dW*
                w, _, fo = layer.compute_optimal_delta()
                self.assertTrue(
                    torch.allclose(
                        w, -2 * torch.eye(self.c, device=global_device()) / batch_red
                    ),
                    f"Error in dW* for {reduction=}, {alpha=}",
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

    def test_extended_forward_out(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            l_ext = torch.nn.Linear(5, 2, bias=bias, device=global_device())
            l_delta = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            layer = LinearGrowingModule(
                5, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)
            layer.optimal_delta_layer = l_delta

            if bias:
                layer.bias.data.copy_(l0.bias.data)
            layer.extended_output_layer = l_ext

            for gamma in (0.0, 1.0, 5.0):
                layer.scaling_factor = gamma
                x = torch.randn((10, 5), device=global_device())
                assert torch.allclose(layer(x), l0(x))

                y1, y2 = layer.extended_forward(x)

                assert torch.allclose(y1, l0(x) - gamma**2 * l_delta(x))
                assert torch.allclose(y2, gamma * l_ext(x))

            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)
            x = torch.randn((10, 5), device=global_device())
            y = layer(x)
            y1 = y[:, :1]
            y2 = y[:, 1:]
            self.assertTrue(torch.allclose(y1, l0(x) - gamma**2 * l_delta(x)))
            self.assertTrue(
                torch.allclose(y2, gamma * l_ext(x), atol=1e-7),
                f"Error in applying change: {(y2 - gamma * l_ext(x)).abs().max():.2e}",
            )

    def test_extended_forward_in(self):
        torch.manual_seed(0)
        for bias in {True, False}:
            l0 = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            l_ext = torch.nn.Linear(5, 1, bias=bias, device=global_device())
            if bias:
                l_ext.bias.data.fill_(0)
            l_delta = torch.nn.Linear(3, 1, bias=bias, device=global_device())
            layer = LinearGrowingModule(
                3, 1, use_bias=bias, name="layer1", device=global_device()
            )
            layer.weight.data.copy_(l0.weight.data)
            layer.optimal_delta_layer = l_delta

            if bias:
                layer.bias.data.copy_(l0.bias.data)
            layer.extended_input_layer = l_ext

            for gamma in (0.0, 1.0, 5.0):
                layer.zero_grad()
                layer.scaling_factor = gamma
                x = torch.randn((10, 3), device=global_device())
                x_ext = torch.randn((10, 5), device=global_device())
                assert torch.allclose(layer(x), l0(x))

                y, none = layer.extended_forward(x, x_ext)
                self.assertIsNone(none)

                assert torch.allclose(
                    y, l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext)
                )

                torch.norm(y).backward()

                self.assertIsNotNone(layer.scaling_factor.grad)

            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)
            x_cat = torch.randn((10, 8), device=global_device())

            y = layer(x_cat)
            x = x_cat[:, :3]
            x_ext = x_cat[:, 3:]
            self.assertTrue(
                torch.allclose(y, l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext)),
                f"Error in applying change: "
                f"{(y - l0(x) - gamma**2 * l_delta(x) + gamma * l_ext(x_ext)).abs().max():.2e}",
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
        self.assertTrue(torch.allclose(y, torch.tensor([[6.0]])))

        layer.layer_in_extension(torch.tensor([[10]], dtype=torch.float32))
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.in_features, 4)
        self.assertEqual(layer.layer.in_features, 4)
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        y = layer(x)
        self.assertTrue(torch.allclose(y, torch.tensor([[46.0]])))

    def test_layer_out_extension(self):
        # without bias
        layer = LinearGrowingModule(1, 3, use_bias=False, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        self.assertEqual(layer.number_of_parameters(), 3)
        self.assertEqual(layer.out_features, 3)
        x = torch.tensor([[1]], dtype=torch.float32)
        y = layer(x)
        self.assertTrue(torch.allclose(y, torch.tensor([[1.0, 1.0, 1.0]])))

        layer.layer_out_extension(torch.tensor([[10]], dtype=torch.float32))
        self.assertEqual(layer.number_of_parameters(), 4)
        self.assertEqual(layer.out_features, 4)
        self.assertEqual(layer.layer.out_features, 4)

        y = layer(x)
        self.assertTrue(torch.allclose(y, torch.tensor([[1.0, 1.0, 1.0, 10.0]])))

        # with bias
        layer = LinearGrowingModule(1, 3, use_bias=True, name="layer1")
        layer.weight = torch.nn.Parameter(torch.ones(3, 1))
        layer.bias = torch.nn.Parameter(10 * torch.ones(3))
        self.assertEqual(layer.number_of_parameters(), 6)
        self.assertEqual(layer.out_features, 3)
        x = torch.tensor([[-1]], dtype=torch.float32)
        y = layer(x)
        self.assertTrue(torch.allclose(y, torch.tensor([[9.0, 9.0, 9.0]])))

        layer.layer_out_extension(
            torch.tensor([[10]], dtype=torch.float32),
            bias=torch.tensor([100], dtype=torch.float32),
        )
        self.assertEqual(layer.number_of_parameters(), 8)
        self.assertEqual(layer.out_features, 4)
        y = layer(x)
        self.assertTrue(torch.allclose(y, torch.tensor([[9.0, 9.0, 9.0, 90.0]])))

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
            self.assertTrue(torch.allclose(y, l0(x) - gamma**2 * l_delta(x)))

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
            layer.scaling_factor = gamma
            layer.apply_change(apply_previous=False)

            x = torch.randn((10, 5), device=global_device())
            y = layer(x)
            y1 = y[:, :1]
            y2 = y[:, 1:]
            self.assertTrue(torch.allclose(y1, l0(x)))
            self.assertTrue(torch.allclose(y2, gamma * l_ext(x)))

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

            self.assertTrue(
                torch.allclose(y, l0(x) + gamma * l_ext(x_ext), atol=1e-7),
                f"Error in applying change: "
                f"{(y - l0(x) - gamma * l_ext(x_ext)).abs().max():.2e}",
            )

    def test_sub_select_optimal_added_parameters_out(self):
        for bias in {True, False}:
            layer = LinearGrowingModule(3, 1, use_bias=bias, name="layer1")
            layer.extended_output_layer = torch.nn.Linear(3, 2, bias=bias)
            layer.eigenvalues_extension = torch.tensor([2.0, 1.0])

            new_layer = torch.nn.Linear(3, 1, bias=bias)
            new_layer.weight.data = layer.extended_output_layer.weight.data[0].view(1, -1)
            if bias:
                new_layer.bias.data = layer.extended_output_layer.bias.data[0].view(1)

            layer.sub_select_optimal_added_parameters(1, sub_select_previous=False)

            self.assertTrue(
                torch.allclose(layer.extended_output_layer.weight, new_layer.weight)
            )
            if bias:
                self.assertTrue(
                    torch.allclose(layer.extended_output_layer.bias, new_layer.bias)
                )

            # self.assertEqual(layer.extended_output_layer, new_layer)
            self.assertTrue(
                torch.allclose(layer.eigenvalues_extension, torch.tensor([2.0, 1.0]))
            )

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

        self.assertTrue(
            torch.allclose(layer.extended_input_layer.weight, new_layer.weight)
        )
        if bias:
            self.assertTrue(
                torch.allclose(layer.extended_input_layer.bias, new_layer.bias)
            )
        self.assertTrue(torch.allclose(layer.eigenvalues_extension, torch.tensor([2.0])))

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
                inv_value = getattr(layer_out, inv)
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
                new_inv_value = getattr(layer_out, inv)
                if isinstance(new_inv_value, torch.Tensor):
                    self.assertTrue(
                        torch.allclose(
                            reference[inv], new_inv_value, rtol=rtol, atol=atol
                        ),
                        f"Error on {inv=}",
                    )
                elif isinstance(new_inv_value, torch.nn.Linear):
                    self.assertTrue(
                        linear_layer_equality(
                            reference[inv], new_inv_value, rtol=rtol, atol=atol
                        ),
                        f"Error on {inv=}",
                    )
                elif isinstance(new_inv_value, TensorStatistic):
                    self.assertTrue(
                        torch.allclose(
                            reference[inv], new_inv_value(), rtol=rtol, atol=atol
                        ),
                        f"Error on {inv=}",
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


if __name__ == "__main__":
    main()
