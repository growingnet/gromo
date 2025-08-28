from copy import deepcopy
from unittest import main

import pytest
import torch

from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    Conv2dMergeGrowingModule,
    FullConv2dGrowingModule,
    RestrictedConv2dGrowingModule,
)
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import compute_output_shape_conv
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase, indicator_batch
from tests.unittest_tools import unittest_parametrize


class TestConv2dMergeGrowingModule(TorchTestCase):
    def setUp(self):
        self.input_size = (8, 8)
        self.kernel_size = 3
        self.in_channels = 2
        self.hidden_channels = 3
        self.out_channels = 2
        self.batch_size = 5
        self.device = global_device()
        self.x = torch.randn(
            self.batch_size, self.in_channels, *self.input_size, device=self.device
        )
        self.y = torch.randn(self.batch_size, 1, device=self.device)
        self.loss_fn = torch.nn.MSELoss()

        # Previous dummy module
        self.prev = Conv2dGrowingModule(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            input_size=self.input_size,
            device=self.device,
            name="prev",
        )
        # Target module under test
        self.merge = Conv2dMergeGrowingModule(
            in_channels=self.hidden_channels,
            input_size=self.prev.out_width,
            next_kernel_size=self.kernel_size,
            device=self.device,
            name="merge",
        )
        # Next dummy module
        self.next = Conv2dGrowingModule(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            input_size=(self.prev.out_width, self.prev.out_height),
            post_layer_function=torch.nn.Flatten(),
            device=self.device,
            name="next",
        )
        self.mlp = torch.nn.Linear(self.next.out_features, 1, device=self.device)

        self.merge.set_previous_modules([self.prev])
        self.merge.set_next_modules([self.next])
        self.prev.next_module = self.merge
        self.next.previous_module = self.merge
        self.next.next_module = self.mlp

        self.net = torch.nn.Sequential(self.prev, self.merge, self.next, self.mlp)

        for module in (self.prev, self.merge, self.next):
            module.reset_computation()
            module.init_computation()

        y_pred = self.net(self.x)
        loss = self.loss_fn(y_pred, self.y)
        loss.backward()

        for module in (self.prev, self.merge, self.next):
            module.update_computation()

    def test_input_volume(self):
        self.assertEqual(self.merge.input_volume, self.prev.out_features)
        self.assertEqual(self.merge.input_volume, self.merge.out_features)

        self.merge.set_previous_modules([])
        with self.assertWarns(UserWarning):
            self.assertEqual(self.merge.input_volume, -1)

        self.merge._input_volume = 0
        self.assertEqual(self.merge.input_volume, 0)

    def test_padding(self):
        self.assertEqual(self.merge.padding, self.next.padding)

        self.merge.set_next_modules([])
        with self.assertWarns(UserWarning):
            self.assertEqual(self.merge.padding, 0)

        # the API now asserts that next modules must be Conv2dGrowingModule;
        # assign directly when testing the linear-branch behavior
        self.merge.next_modules = [LinearGrowingModule(self.merge.out_features, 1)]
        self.assertEqual(self.merge.padding, 0)

    def test_stride(self):
        self.assertEqual(self.merge.stride, self.next.stride)

        self.merge.set_next_modules([])
        with self.assertWarns(UserWarning):
            self.assertEqual(self.merge.stride, 1)

        # assign directly when testing the linear-branch behavior
        self.merge.next_modules = [LinearGrowingModule(self.merge.out_features, 1)]
        self.assertEqual(self.merge.stride, 1)

    def test_dilation(self):
        self.assertEqual(self.merge.dilation, self.next.layer.dilation)

        self.merge.set_next_modules([])
        with self.assertWarns(UserWarning):
            self.assertEqual(self.merge.dilation, 1)

        # assign directly when testing the linear-branch behavior
        self.merge.next_modules = [LinearGrowingModule(self.merge.out_features, 1)]
        self.assertEqual(self.merge.dilation, 1)

    def test_unfolded_extended_activity(self):
        unfolded = self.merge.unfolded_extended_activity
        self.assertEqual(unfolded.shape[0], self.batch_size)
        self.assertTrue(
            unfolded.shape[1] >= self.in_channels * self.kernel_size * self.kernel_size
        )

        # linear branch: flattened activity -> last column equals ones
        # assign directly when testing the linear-branch behavior
        self.merge.next_modules = [LinearGrowingModule(self.merge.out_features, 1)]
        # ensure activity is flattened as expected by the linear branch
        self.merge.activity = self.merge.activity.flatten(1)
        unfolded_lin = self.merge.unfolded_extended_activity
        self.assertEqual(unfolded_lin.shape[0], self.batch_size)
        self.assertEqual(
            unfolded_lin.shape[1], self.merge.out_features + self.merge.use_bias
        )
        self.assertTrue((unfolded_lin[:, -1] == 1).all())

    def test_construct_full_activity(self):
        full = self.merge.construct_full_activity()
        self.assertEqual(full.shape[0], self.batch_size)
        self.assertEqual(full.shape[1], self.prev.in_features + self.prev.use_bias)

    def test_compute_previous_s_update(self):
        S, n = self.merge.compute_previous_s_update()
        self.assertEqual(S.shape[0], self.prev.in_features + self.prev.use_bias)
        self.assertEqual(n, self.batch_size)

    def test_compute_previous_m_update(self):
        M, n = self.merge.compute_previous_m_update()
        self.assertEqual(M.shape[0], self.prev.in_features + self.prev.use_bias)
        self.assertEqual(M.shape[1], self.merge.in_channels)
        self.assertEqual(n, self.batch_size)

    def test_compute_s_update(self):
        S, n = self.merge.compute_s_update()
        D = self.hidden_channels * self.kernel_size * self.kernel_size + 1
        self.assertEqual(S.shape, (D, D))
        self.assertEqual(n, self.batch_size)

        # linear branch: flattened activity -> S computed accordingly
        # assign directly when testing the linear-branch behavior
        self.merge.next_modules = [LinearGrowingModule(self.merge.out_features, 1)]
        self.merge.activity = self.merge.activity.flatten(1)
        S_lin, n_lin = self.merge.compute_s_update()
        D_lin = self.merge.out_features + self.merge.use_bias
        self.assertEqual(S_lin.shape, (D_lin, D_lin))
        self.assertEqual(n_lin, self.batch_size)

    def test_set_previous_modules(self):
        # API now asserts types for previous modules
        with self.assertRaises(AssertionError):
            self.merge.set_previous_modules(["not a module"])
        with self.assertRaises(AssertionError):
            self.merge.set_previous_modules([torch.nn.Linear(1, 1)])

        self.merge.set_previous_modules([self.prev])
        self.assertEqual(
            self.merge.total_in_features, self.prev.in_features + self.prev.use_bias
        )
        self.assertEqual(self.merge.total_out_features, self.prev.out_features)

        self.merge.set_previous_modules([self.prev, self.prev])
        self.assertEqual(
            self.merge.total_in_features, (self.prev.in_features + self.prev.use_bias) * 2
        )
        self.assertEqual(self.merge.total_out_features, self.prev.out_features * 2)

        with self.assertRaises(ValueError):
            self.merge.set_previous_modules([self.prev, self.next])

    def test_set_next_modules(self):
        # setting next modules now performs early assertions on input types
        with self.assertRaises(AssertionError):
            self.merge.set_next_modules(["not a module"])
        with self.assertRaises(AssertionError):
            self.merge.set_next_modules([torch.nn.Linear(1, 1)])
        with self.assertRaises(AssertionError):
            self.merge.set_next_modules([LinearGrowingModule(1, 1)])

    def test_set_next_modules_kernel_and_channel_mismatch(self):
        # kernel size mismatch should raise AssertionError
        new_next_diff_ks = Conv2dGrowingModule(
            in_channels=self.merge.out_channels,
            out_channels=1,
            kernel_size=self.kernel_size + 1,
            input_size=(1, 1),
            device=self.device,
            name="new_next_diff_ks",
        )
        with self.assertRaises(AssertionError):
            self.merge.set_next_modules([new_next_diff_ks])

        # channel mismatch (in_channels != merge.out_channels) should raise AssertionError
        new_next_bad_ch = Conv2dGrowingModule(
            in_channels=self.merge.out_channels + 1,
            out_channels=1,
            kernel_size=self.kernel_size,
            input_size=(1, 1),
            device=self.device,
            name="new_next_bad_ch",
        )
        # ensure kernel sizes match so we hit the channel assertion
        with self.assertRaises(AssertionError):
            self.merge.set_next_modules([new_next_bad_ch])

    def test_out_channels_and_in_features_warning(self):
        # out_channels proxies in_channels
        self.assertEqual(self.merge.out_channels, self.merge.in_channels)
        # accessing in_features should warn and return in_channels
        with self.assertWarns(UserWarning):
            self.assertEqual(self.merge.in_features, self.merge.in_channels)

    def test_output_size_property(self):
        # output_size should equal input_size after init
        self.assertEqual(self.merge.output_size, self.merge.input_size)

    def test_padding_stride_dilation_notimplemented_for_invalid_next(self):
        # create fresh merge instances and directly assign invalid next_modules
        for prop in ("padding", "stride", "dilation"):
            merge_fresh = Conv2dMergeGrowingModule(
                in_channels=self.hidden_channels,
                input_size=self.prev.out_width,
                next_kernel_size=self.kernel_size,
                device=self.device,
                name=f"merge_{prop}",
            )
            # assign a next module that is neither Conv2dGrowingModule nor LinearGrowingModule
            merge_fresh.next_modules = [torch.nn.ReLU()]
            with self.assertRaises(NotImplementedError):
                _ = getattr(merge_fresh, prop)

    def test_unfolded_extended_activity_has_bias_column(self):
        # conv branch: last channel along dim=1 should be ones when use_bias
        unfolded = self.merge.unfolded_extended_activity
        self.assertEqual(unfolded.shape[0], self.batch_size)
        # bias column appended as last index along dim=1
        self.assertTrue((unfolded[:, -1, :] == 1).all())
        # linear branch: flattened activity -> last column equals ones
        # assign directly when testing the linear-branch behavior
        self.merge.next_modules = [LinearGrowingModule(self.merge.out_features, 1)]
        # ensure activity is flattened as expected by the linear branch
        self.merge.activity = self.merge.activity.flatten(1)
        unfolded_lin = self.merge.unfolded_extended_activity
        self.assertEqual(unfolded_lin.shape[0], self.batch_size)
        self.assertEqual(
            unfolded_lin.shape[1], self.merge.out_features + self.merge.use_bias
        )
        self.assertTrue((unfolded_lin[:, -1] == 1).all())

    def test_set_next_modules_updates_input_size_and_warns_when_already_sampled(self):
        # valid conv next should adopt merge.output_size
        new_next = Conv2dGrowingModule(
            in_channels=self.merge.out_channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            input_size=(1, 1),
            device=self.device,
            name="new_next",
        )
        self.merge.set_next_modules([new_next])
        self.assertEqual(new_next.input_size, self.merge.output_size)

        # simulate non-empty tensor_s to trigger a warning on re-setting next modules
        if self.merge.tensor_s is not None:
            self.merge.tensor_s.samples = 1
        with self.assertWarns(UserWarning):
            self.merge.set_next_modules([new_next])

    def test_set_previous_modules_warns_on_existing_prev_stats_and_handles_empty(self):
        # ensure previous statistics show non-zero samples to trigger warnings
        if self.merge.previous_tensor_s is not None:
            self.merge.previous_tensor_s.samples = 1
        if self.merge.previous_tensor_m is not None:
            self.merge.previous_tensor_m.samples = 1
        with self.assertWarns(UserWarning):
            self.merge.set_previous_modules([self.prev])

        # empty previous modules should clear previous tensors
        self.merge.set_previous_modules([])
        self.assertIsNone(self.merge.previous_tensor_s)
        self.assertIsNone(self.merge.previous_tensor_m)

    def test_set_previous_modules_kernel_mismatch_raises(self):
        # previous modules must have matching kernel_size
        prev_bad_ks = Conv2dGrowingModule(
            in_channels=self.prev.in_channels,
            out_channels=self.prev.out_channels,
            kernel_size=self.kernel_size + 1,
            input_size=(self.prev.out_width, self.prev.out_height),
            device=self.device,
            name="prev_bad_ks",
        )
        with self.assertRaises(AssertionError):
            self.merge.set_previous_modules([prev_bad_ks])

    def test_construct_full_activity_with_two_prev_modules(self):
        # create a second previous module identical to self.prev
        prev2 = Conv2dGrowingModule(
            in_channels=self.prev.in_channels,
            out_channels=self.prev.out_channels,
            kernel_size=self.kernel_size,
            input_size=(self.prev.out_width, self.prev.out_height),
            device=self.device,
            name="prev2",
        )
        # copy stored input from self.prev so unfolded_extended_input is available
        prev2.store_input = True
        prev2._internal_store_input = True
        prev2._input = self.prev._input
        self.merge.set_previous_modules([self.prev, prev2])
        full = self.merge.construct_full_activity()
        expected_dim = (self.prev.in_features + self.prev.use_bias) * 2
        L = int(self.prev.out_features / self.merge.in_channels)
        self.assertEqual(full.shape, (self.batch_size, expected_dim, L))
        # check concatenation order: first block equals first prev unfolded_extended_input
        first_block = full[:, : self.prev.in_features + self.prev.use_bias, :]
        self.assertTrue(torch.allclose(first_block, self.prev.unfolded_extended_input))

    def test_compute_previous_s_update_symmetry(self):
        S, n = self.merge.compute_previous_s_update()
        self.assertEqual(n, self.batch_size)
        # S should be symmetric
        self.assertTrue(torch.allclose(S, S.transpose(0, 1), atol=1e-6))

    def test_compute_s_update_invalid_next_raises(self):
        # fresh merge with activity stored but invalid next module assigned directly
        merge_bad = Conv2dMergeGrowingModule(
            in_channels=self.hidden_channels,
            input_size=self.prev.out_width,
            next_kernel_size=self.kernel_size,
            device=self.device,
            name="merge_bad",
        )
        merge_bad.store_activity = True
        # provide flattened activity (batch, D) so unfolded_extended_activity can concatenate bias
        D = self.hidden_channels * self.kernel_size * self.kernel_size
        merge_bad.activity = torch.randn(self.batch_size, D, device=self.device)
        # assign a next module that is not handled by the code paths
        merge_bad.next_modules = [torch.nn.ReLU()]
        with self.assertRaises(NotImplementedError):
            merge_bad.compute_s_update()

    def test_update_size_reallocates_previous_stats(self):
        # start with one previous
        self.merge.set_previous_modules([self.prev])
        s_shape_before = None
        m_shape_before = None
        if self.merge.previous_tensor_s is not None:
            s_shape_before = self.merge.previous_tensor_s._shape
        if self.merge.previous_tensor_m is not None:
            m_shape_before = self.merge.previous_tensor_m._shape

        # add a second previous and update size
        prev2 = Conv2dGrowingModule(
            in_channels=self.prev.in_channels,
            out_channels=self.prev.out_channels,
            kernel_size=self.kernel_size,
            input_size=(self.prev.out_width, self.prev.out_height),
            device=self.device,
            name="prev2",
        )
        # ensure prev2 reports stored input so unfolded_extended_input can be computed
        prev2.store_input = True
        prev2._internal_store_input = True
        prev2._input = self.x
        self.merge.set_previous_modules([self.prev, prev2])
        self.merge.update_size()

        expected_total_in = (self.prev.in_features + self.prev.use_bias) * 2
        self.assertEqual(self.merge.total_in_features, expected_total_in)
        self.assertEqual(self.merge.in_channels, self.prev.out_channels)
        self.assertEqual(
            self.merge.previous_tensor_s._shape, (expected_total_in, expected_total_in)
        )
        self.assertEqual(
            self.merge.previous_tensor_m._shape,
            (expected_total_in, self.merge.in_channels),
        )

        # removing previous modules should clear previous tensors
        self.merge.set_previous_modules([])
        self.merge.update_size()
        self.assertIsNone(self.merge.previous_tensor_s)
        self.assertIsNone(self.merge.previous_tensor_m)

    def test_layer_of_tensor_bias_mismatch_raises(self):
        # layer_of_tensor asserts bias presence matches use_bias
        demo = Conv2dGrowingModule(
            in_channels=2,
            out_channels=3,
            kernel_size=(3, 3),
            use_bias=True,
            device=global_device(),
        )
        # create a weight compatible tensor but pass bias=None to trigger assertion
        weight = torch.randn(3, 2, 3, 3, device=global_device())
        with self.assertRaises(AssertionError):
            _ = demo.layer_of_tensor(weight, bias=None)

    def test_input_volume_warns_without_previous(self):
        demo = Conv2dMergeGrowingModule(2, (4, 4), (3, 3), device=global_device())
        import pytest

        with pytest.warns(UserWarning):
            val = demo.input_volume
        assert val == -1

    def test_padding_stride_dilation_unsupported_next_raises(self):
        demo = Conv2dMergeGrowingModule(2, (4, 4), (3, 3), device=global_device())
        # directly assign an unsupported next module type to exercise NotImplementedError
        demo.next_modules = [object()]
        import pytest

        with pytest.raises(NotImplementedError):
            _ = demo.padding
        with pytest.raises(NotImplementedError):
            _ = demo.stride
        with pytest.raises(NotImplementedError):
            _ = demo.dilation

    def test_bordered_unfolded_prev_input_previous_none_or_merge_raises(self):
        demo = RestrictedConv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=3, device=global_device()
        )
        # no previous -> ValueError
        import pytest

        with pytest.raises(ValueError):
            _ = demo.bordered_unfolded_extended_prev_input

        # previous is Conv2dMergeGrowingModule -> NotImplementedError branch
        prev = Conv2dMergeGrowingModule(1, (4, 4), 3, device=global_device())
        demo.previous_module = prev
        with pytest.raises(NotImplementedError):
            _ = demo.bordered_unfolded_extended_prev_input

    def test_compute_m_prev_and_cross_covariance_raise_when_no_previous(self):
        demo = RestrictedConv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=3, device=global_device()
        )
        import pytest

        with pytest.raises(ValueError):
            demo.compute_m_prev_update()
        with pytest.raises(ValueError):
            demo.compute_cross_covariance_update()


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

    def test_compute_m_prev_update_notimplemented_cases(self):
        """Exercise NotImplementedError branches of compute_m_prev_update by assigning
        different previous_module types and passing an explicit desired_activation.
        """
        demo = self.demo
        device = global_device()
        batch = 2
        # shape compatible with flatten(start_dim=-2)
        desired_activation = torch.randn(batch, demo.out_channels, 1, device=device)

        # LinearGrowingModule branch should raise NotImplementedError
        demo.previous_module = LinearGrowingModule(in_features=1, out_features=1)
        with self.assertRaises(NotImplementedError):
            demo.compute_m_prev_update(desired_activation=desired_activation)

        # LinearMergeGrowingModule branch should raise NotImplementedError
        lm = LinearMergeGrowingModule(in_features=1, device=device)
        demo.previous_module = lm
        with self.assertRaises(NotImplementedError):
            demo.compute_m_prev_update(desired_activation=desired_activation)

        # Conv2dMergeGrowingModule branch should raise NotImplementedError
        cm = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(1, 1), next_kernel_size=1, device=device
        )
        demo.previous_module = cm
        with self.assertRaises(NotImplementedError):
            demo.compute_m_prev_update(desired_activation=desired_activation)

    def test_sub_select_and_bordered_prev_notimplemented_cases(self):
        """Cover NotImplementedError branches in sub-selection and bordered unfolded prev input."""
        device = global_device()

        # sub_select_optimal_added_parameters should raise NotImplementedError for some previous_module types
        demo = Conv2dGrowingModule(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            use_bias=True,
            device=device,
        )
        # set extended_input_layer to satisfy XOR assertion
        demo.extended_input_layer = demo.layer

        # provide eigenvalues_extension so sub-selection assertion passes and
        # the function can reach the NotImplementedError branches for previous types
        demo.eigenvalues_extension = torch.tensor([1.0], device=device)

        demo.previous_module = LinearMergeGrowingModule(in_features=1, device=device)
        with self.assertRaises(NotImplementedError):
            demo.sub_select_optimal_added_parameters(keep_neurons=1)

        demo.previous_module = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(1, 1), next_kernel_size=1, device=device
        )
        with self.assertRaises(NotImplementedError):
            demo.sub_select_optimal_added_parameters(keep_neurons=1)

        # bordered_unfolded_extended_prev_input should raise NotImplementedError for Conv2dMergeGrowingModule
        r = RestrictedConv2dGrowingModule(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            device=device,
        )
        r.previous_module = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(1, 1), next_kernel_size=1, device=device
        )
        with self.assertRaises(NotImplementedError):
            _ = r.bordered_unfolded_extended_prev_input

    def test_restricted_and_full_compute_m_prev_and_cross_notimplemented(self):
        """Exercise NotImplementedError branches in compute_m_prev_update and cross covariance for Restricted/Full classes."""
        device = global_device()

        # Restricted compute_m_prev_update: LinearGrowingModule and others -> NotImplementedError
        r = RestrictedConv2dGrowingModule(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            device=device,
        )
        r.previous_module = LinearGrowingModule(in_features=1, out_features=1)
        with self.assertRaises(NotImplementedError):
            r.compute_m_prev_update(
                desired_activation=torch.randn(1, r.out_channels, device=device)
            )

        r.previous_module = LinearMergeGrowingModule(in_features=1, device=device)
        with self.assertRaises(NotImplementedError):
            r.compute_m_prev_update(
                desired_activation=torch.randn(1, r.out_channels, device=device)
            )

        r.previous_module = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(1, 1), next_kernel_size=1, device=device
        )
        with self.assertRaises(NotImplementedError):
            r.compute_m_prev_update(
                desired_activation=torch.randn(1, r.out_channels, device=device)
            )

        # Full compute_cross_covariance_update: unsupported previous types -> NotImplementedError
        f = FullConv2dGrowingModule(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            device=device,
        )
        f.previous_module = LinearGrowingModule(in_features=1, out_features=1)
        with self.assertRaises(NotImplementedError):
            f.compute_cross_covariance_update()

        f.previous_module = LinearMergeGrowingModule(in_features=1, device=device)
        with self.assertRaises(NotImplementedError):
            f.compute_cross_covariance_update()

        f.previous_module = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(1, 1), next_kernel_size=1, device=device
        )
        with self.assertRaises(NotImplementedError):
            f.compute_cross_covariance_update()

    def test_more_notimplemented_branches(self):
        """Additional small tests to hit NotImplementedError/ValueError branches."""
        device = global_device()

        # compute_m_prev_update is not implemented in the base GrowingModule -> NotImplementedError
        demo = Conv2dGrowingModule(
            in_channels=2, out_channels=2, kernel_size=3, device=device
        )
        with self.assertRaises(NotImplementedError):
            demo.compute_m_prev_update()

        # compute_m_prev_update should raise NotImplementedError for LinearGrowingModule previous
        demo.previous_module = LinearGrowingModule(in_features=1, out_features=1)
        with self.assertRaises(NotImplementedError):
            demo.compute_m_prev_update(
                desired_activation=torch.randn(1, demo.out_channels, device=device)
            )

        # compute_cross_covariance_update should raise NotImplementedError for LinearMergeGrowingModule previous
        demo.previous_module = LinearMergeGrowingModule(in_features=1, device=device)
        with self.assertRaises(NotImplementedError):
            demo.compute_cross_covariance_update()

        # compute_optimal_added_parameters should raise NotImplementedError for LinearMergeGrowingModule previous
        # prepare minimal preconditions
        prev = Conv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=1, input_size=(1, 1), device=device
        )
        layer = Conv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=1, input_size=(1, 1), device=device
        )
        layer.previous_module = LinearMergeGrowingModule(in_features=1, device=device)
        # set minimal tensors to avoid earlier assertions
        layer.delta_raw = torch.zeros((layer.out_channels, 1), device=device)
        layer._tensor_m_prev = lambda: torch.zeros((1, 1), device=device)
        layer._cross_covariance = lambda: torch.zeros((1, 1), device=device)
        with self.assertRaises(NotImplementedError):
            layer.compute_optimal_added_parameters()

    def test_sub_select_requires_eigenvalues_when_input_branch(self):
        device = global_device()
        demo = Conv2dGrowingModule(
            in_channels=2, out_channels=3, kernel_size=3, use_bias=True, device=device
        )
        # set only extended_input_layer to satisfy XOR but not eigenvalues_extension
        demo.extended_input_layer = demo.layer
        demo.previous_module = Conv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=1, input_size=(1, 1), device=device
        )
        import pytest

        with pytest.raises(AssertionError):
            demo.sub_select_optimal_added_parameters(keep_neurons=1)

    def test_compute_m_prev_and_cross_covariance_unsupported_previous_raises(self):
        device = global_device()
        r = RestrictedConv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=3, device=device
        )
        # assign an unsupported previous type
        r.previous_module = object()
        import pytest

        with pytest.raises(NotImplementedError):
            r.compute_m_prev_update()
        with pytest.raises(NotImplementedError):
            r.compute_cross_covariance_update()

    # bordered_unfolded_extended_prev_input is tested elsewhere; skip here

    def test_compute_optimal_added_parameters_notimplemented_early(self):
        """Monkeypatch auxiliary computation to reach NotImplementedError for Linear previous types."""
        device = global_device()

        layer = Conv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=1, input_size=(1, 1), device=device
        )

        # monkeypatch the internal auxiliary method to return minimal consistent tensors
        def fake_auxiliary(**kwargs):
            # alpha shape: (k, ?), omega shape: (out_channels, k), eigenvalues length k
            k = 1
            alpha = torch.zeros((k, 1), device=device)
            omega = torch.zeros((layer.out_channels, k), device=device)
            eig = torch.tensor([1.0], device=device)
            return alpha, omega, eig

        layer._auxiliary_compute_alpha_omega = fake_auxiliary

        # LinearGrowingModule previous should raise NotImplementedError early
        layer.previous_module = LinearGrowingModule(
            in_features=1, out_features=1, device=device
        )
        with self.assertRaises(NotImplementedError):
            layer.compute_optimal_added_parameters()

    def test_init_with_int_input_and_kernel_and_subselect_output_branch(self):
        """Cover int->tuple conversion in init and the extended_output branch of sub-selection."""
        device = global_device()

        # int conversions: input_size and next_kernel_size passed as ints
        m = Conv2dMergeGrowingModule(
            in_channels=2, input_size=8, next_kernel_size=3, device=device
        )
        # ensure converted to tuples
        self.assertIsInstance(m.input_size, tuple)
        self.assertIsInstance(m.kernel_size, tuple)

        # extended_output_layer branch: set extended_output_layer and previous conv module
        demo = Conv2dGrowingModule(
            in_channels=1, out_channels=2, kernel_size=1, input_size=(2, 2), device=device
        )
        # set an extended output layer (simulate previous extension)
        demo.extended_output_layer = demo.layer
        # set eigenvalues so assertion passes when sub-selecting (but extended_output branch doesn't require eigenvalues)
        demo.eigenvalues_extension = torch.tensor([1.0], device=device)

        # previous module that supports _sub_select_added_output_dimension (Conv2dGrowingModule)
        prev = Conv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=1, input_size=(2, 2), device=device
        )
        # previous module must have an extended_output_layer for its _sub_select to work
        prev.extended_output_layer = prev.layer
        demo.previous_module = prev

        # should not raise (sub-selects both output and previous appropriately)
        demo.sub_select_optimal_added_parameters(keep_neurons=1)

        # also verify compute_optimal_added_parameters raises for known unsupported previous types
        layer = Conv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=1, input_size=(1, 1), device=device
        )
        layer.previous_module = LinearMergeGrowingModule(in_features=1, device=device)
        with self.assertRaises(NotImplementedError):
            layer.compute_optimal_added_parameters()

        conv_merge_prev = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(1, 1), next_kernel_size=1, device=device
        )
        layer.previous_module = conv_merge_prev
        with self.assertRaises(NotImplementedError):
            layer.compute_optimal_added_parameters()

    def test_compute_m_prev_update_with_conv_merge_previous_raises(self):
        """compute_m_prev_update should raise NotImplementedError when previous is Conv2dMergeGrowingModule."""
        device = global_device()
        demo = Conv2dGrowingModule(
            in_channels=2, out_channels=2, kernel_size=3, input_size=(4, 4), device=device
        )
        demo.previous_module = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(2, 2), next_kernel_size=1, device=device
        )
        with self.assertRaises(NotImplementedError):
            demo.compute_m_prev_update()

    def test_compute_optimal_added_parameters_with_conv_merge_previous_raises(self):
        """compute_optimal_added_parameters should raise NotImplementedError for Conv2dMerge previous even if auxiliary returns minimal tensors."""
        device = global_device()
        layer = Conv2dGrowingModule(
            in_channels=1, out_channels=1, kernel_size=1, input_size=(1, 1), device=device
        )

        # monkeypatch the internal auxiliary method to return minimal consistent tensors
        def fake_auxiliary(**kwargs):
            k = 1
            alpha = torch.zeros((k, 1), device=device)
            omega = torch.zeros((layer.out_channels, k), device=device)
            eig = torch.tensor([1.0], device=device)
            return alpha, omega, eig

        layer._auxiliary_compute_alpha_omega = fake_auxiliary

        layer.previous_module = Conv2dMergeGrowingModule(
            in_channels=1, input_size=(1, 1), next_kernel_size=1, device=device
        )
        with self.assertRaises(NotImplementedError):
            layer.compute_optimal_added_parameters()

    # Unknown/other previous module types are not tested here (covered elsewhere)

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
    def test_update_input_size(self, bias: bool):
        demo_layer = self.bias_demos[bias]

        # error
        with self.assertRaises(AssertionError):
            demo_layer.update_input_size()

        # automatic setting
        demo_layer.store_input = True
        demo_layer(self.input_x)

        demo_layer.update_input_size()
        self.assertEqual(demo_layer.input_size, (10, 10))

        # manual
        with self.assertWarns(Warning):
            demo_layer.update_input_size((7, 7))

        self.assertEqual(demo_layer.input_size, (7, 7))


class TestFullConv2dGrowingModule(TestConv2dGrowingModule):
    _tested_class = FullConv2dGrowingModule

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
        with self.assertRaises(AssertionError):
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
                demo_couple[1].update_input_size()
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
        ].compute_optimal_added_parameters()

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


def test_conv2d_merge_properties_with_linear_next():
    device = global_device()
    m = Conv2dMergeGrowingModule(
        in_channels=2, input_size=(4, 4), next_kernel_size=1, device=device
    )
    # set a Linear next module -> padding/stride/dilation should return defaults for Linear
    m.next_modules = [LinearGrowingModule(in_features=1, out_features=1, device=device)]
    assert m.padding == 0
    assert m.stride == 1
    assert m.dilation == 1


def test_unfolded_extended_activity_with_linear_next_returns_tensor():
    device = global_device()
    m = Conv2dMergeGrowingModule(
        in_channels=2, input_size=(4, 4), next_kernel_size=1, device=device
    )
    m.activity = torch.randn(2, m.in_channels, 4, 4, device=device).flatten(1)
    m.next_modules = [LinearGrowingModule(in_features=1, out_features=1, device=device)]
    out = m.unfolded_extended_activity
    assert isinstance(out, torch.Tensor)


def test_set_next_and_previous_modules_assertions():
    device = global_device()
    m = Conv2dMergeGrowingModule(
        in_channels=2, input_size=(4, 4), next_kernel_size=1, device=device
    )
    # next modules must be Conv2dGrowingModule -> setting LinearGrowingModule should assert
    with pytest.raises(AssertionError):
        m.set_next_modules(
            [LinearGrowingModule(in_features=1, out_features=1, device=device)]
        )
    with pytest.raises(AssertionError):
        m.set_previous_modules(
            [LinearGrowingModule(in_features=1, out_features=1, device=device)]
        )


def test_construct_full_activity_without_previous_raises():
    device = global_device()
    m = Conv2dMergeGrowingModule(
        in_channels=2, input_size=(4, 4), next_kernel_size=1, device=device
    )
    with pytest.raises(AssertionError):
        m.construct_full_activity()


def test_update_input_size_and_bordered_unfolded_errors():
    device = global_device()
    layer = RestrictedConv2dGrowingModule(
        in_channels=1, out_channels=1, kernel_size=3, input_size=(-1, -1), device=device
    )
    # update_input_size without any info should raise AssertionError
    with pytest.raises(AssertionError):
        layer.update_input_size()

    # bordered_unfolded_extended_prev_input without previous should raise ValueError
    with pytest.raises(ValueError):
        _ = layer.bordered_unfolded_extended_prev_input


def test_unfolded_extended_activity_raises_runtime_when_activity_not_flattened():
    device = global_device()
    m = Conv2dMergeGrowingModule(
        in_channels=2, input_size=(4, 4), next_kernel_size=1, device=device
    )
    # activity is 4D but next is linear branch -> concatenation will fail
    m.activity = torch.randn(2, m.in_channels, 4, 4, device=device)
    m.next_modules = [LinearGrowingModule(in_features=1, out_features=1, device=device)]
    with pytest.raises(RuntimeError):
        _ = m.unfolded_extended_activity


def test_bordered_unfolded_raises_notimplemented_for_conv_merge_previous():
    device = global_device()
    layer = RestrictedConv2dGrowingModule(
        in_channels=1, out_channels=1, kernel_size=3, input_size=(4, 4), device=device
    )
    layer.previous_module = Conv2dMergeGrowingModule(
        in_channels=1, input_size=(2, 2), next_kernel_size=1, device=device
    )
    with pytest.raises(NotImplementedError):
        _ = layer.bordered_unfolded_extended_prev_input
