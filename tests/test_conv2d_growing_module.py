import random
from copy import deepcopy
from unittest import main

import torch

from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    Conv2dMergeGrowingModule,
    FullConv2dGrowingModule,
    RestrictedConv2dGrowingModule,
)
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import compute_output_shape_conv
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase, indicator_batch
from tests.unittest_tools import unittest_parametrize


class TestConv2dMergeGrowingModule(TorchTestCase):
    _tested_class = Conv2dMergeGrowingModule

    def setUp(self):
        """Set up a compact conv merge topology used across tests.

        We build a small chain: prev (Conv2d) -> merge (Conv2dMerge) -> next (Conv2d)
        with consistent kernel sizes so we can run forward/backward and compute stats.
        """
        torch.manual_seed(0)
        random.seed(0)

        # Common shapes
        self.batch = 4
        self.in_channels_prev = 2
        self.merge_in_channels = 3
        self.kernel_size = (3, 3)
        self.input_hw = (8, 8)

        # Previous conv module: 2 -> 3 channels, 3x3, no padding
        self.prev = Conv2dGrowingModule(
            in_channels=self.in_channels_prev,
            out_channels=self.merge_in_channels,
            kernel_size=self.kernel_size,
            input_size=self.input_hw,
            use_bias=False,
            device=global_device(),
        )

        # Merge module expects in_channels that match prev.out_channels
        self.merge = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=(
                self.input_hw[0] - self.kernel_size[0] + 1,
                self.input_hw[1] - self.kernel_size[1] + 1,
            ),
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        # Next conv module used for properties (padding/stride/dilation)
        # and activity storage
        self.next = Conv2dGrowingModule(
            in_channels=self.merge_in_channels,
            out_channels=5,
            kernel_size=self.kernel_size,
            padding=1,
            input_size=self.merge.output_size,
            use_bias=True,
            device=global_device(),
        )

        # Wire modules for the common single-previous scenario
        self.merge.set_previous_modules([self.prev])
        self.merge.set_next_modules([self.next])

        # Typical input image batch
        self.input_x = torch.randn(
            self.batch, self.in_channels_prev, *self.input_hw, device=global_device()
        )

    def test_init_and_basic_properties(self):
        """Test constructor and simple property access with and without neighbors."""
        m = self.merge

        # out_channels mirrors in_channels
        self.assertEqual(m.out_channels, self.merge_in_channels)

        # output_size echoes provided input_size
        self.assertEqual(m.output_size, m.input_size)

        # input_volume with previous modules present delegates to previous.output_volume
        self.assertEqual(m.input_volume, self.prev.output_volume)

        # If no previous modules -> warning and -1
        m.previous_modules = []
        m.input_size = None
        with self.assertWarns(UserWarning):
            self.assertEqual(m.input_volume, -1)

    def test_input_volume_with_explicit_value(self):
        """Test input_volume when _input_volume is explicitly set."""
        m = self.merge
        # Explicitly set _input_volume
        m._input_volume = 42
        self.assertEqual(m.input_volume, 42)

        # Reset to None to test fallback to previous modules
        m._input_volume = None
        self.assertEqual(m.input_volume, self.prev.output_volume)

    def test_output_volume_with_reshaping(self):
        """Test output_volume when the ouput is reshaped."""
        m = self.merge
        # Set post_merge_function
        m.post_merge_function = torch.nn.AvgPool2d(2, 2)
        output_size = compute_output_shape_conv(self.prev.input_size, self.prev)
        output_size = (output_size[0] - 2) / 2 + 1
        output_volume = m.in_channels * output_size * output_size
        self.assertEqual(m.output_volume, output_volume)

        # Set reshaping function
        m.reshape_function = torch.nn.Flatten()
        self.assertEqual(m.output_volume, output_volume)
        m.reshape_function = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.assertEqual(m.output_volume, m.in_channels)

        # Reset
        m.input_size = None
        self.assertEqual(m.output_volume, self.prev.output_volume)

    def test_constructor_int_conversions(self):
        """Test that int input_size and next_kernel_size are converted to tuples."""
        # Test with int input_size
        merge_with_int_input = Conv2dMergeGrowingModule(
            in_channels=2,
            input_size=10,  # int instead of tuple
            next_kernel_size=(3, 3),
            device=global_device(),
        )
        self.assertEqual(merge_with_int_input.input_size, (10, 10))

        # Test with int next_kernel_size
        merge_with_int_kernel = Conv2dMergeGrowingModule(
            in_channels=2,
            input_size=(8, 8),
            next_kernel_size=5,  # int instead of tuple
            device=global_device(),
        )
        self.assertEqual(merge_with_int_kernel.kernel_size, (5, 5))

    def test_padding_stride_dilation_properties(self):
        """Test padding/stride/dilation derivation for conv next and
        warning path when missing."""
        m = self.merge
        # With conv next
        self.assertEqual(m.padding, self.next.layer.padding)
        self.assertEqual(m.stride, self.next.layer.stride)
        self.assertEqual(m.dilation, self.next.layer.dilation)

        # Without next modules -> warnings and defaults
        m.set_next_modules([])
        with self.assertWarns(UserWarning):
            self.assertEqual(m.padding, (0, 0))
        with self.assertWarns(UserWarning):
            self.assertEqual(m.stride, (1, 1))
        with self.assertWarns(UserWarning):
            self.assertEqual(m.dilation, (1, 1))

        # With LinearGrowingModule next
        linear_next = LinearGrowingModule(m.output_volume, 10, device=global_device())
        m.set_next_modules([linear_next])
        default_padding = ((self.kernel_size[0] - 1) / 2, (self.kernel_size[0] - 1) / 2)
        self.assertEqual(m.padding, default_padding)
        self.assertEqual(m.stride, (1, 1))
        self.assertEqual(m.dilation, (1, 1))

    def test_set_previous_modules_and_shapes(self):
        """Test set_previous_modules happy path and shape bookkeeping with
        multiple previous nodes."""
        # Create a second previous conv with same kernel size and out_channels
        prev2 = Conv2dGrowingModule(
            in_channels=1,
            out_channels=self.merge_in_channels,
            kernel_size=self.kernel_size,
            input_size=self.input_hw,
            use_bias=True,
            device=global_device(),
        )

        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        # Set previous modules and verify shapes for previous S/M
        m.set_previous_modules([self.prev, prev2])

        # total_in_features = sum(in_features + use_bias) across previous modules
        expected_tif = self.prev.in_features + int(self.prev.use_bias)
        expected_tif += prev2.in_features + int(prev2.use_bias)
        self.assertEqual(m.total_in_features, expected_tif)
        self.assertEqual(m.previous_tensor_s._shape, (expected_tif, expected_tif))
        self.assertEqual(m.previous_tensor_m._shape, (expected_tif, m.in_channels))

        # Wrong type -> TypeError
        with self.assertRaises(TypeError):
            m.set_previous_modules([torch.nn.Conv2d(1, 1, 1)])  # type: ignore[arg-type]

        # Channel mismatch -> ValueError
        bad_prev = Conv2dGrowingModule(
            in_channels=1,
            out_channels=self.merge_in_channels + 1,
            kernel_size=self.kernel_size,
            input_size=self.input_hw,
            device=global_device(),
        )
        with self.assertRaises(ValueError):
            m.set_previous_modules([bad_prev])

        # Kernel size mismatch -> assertion
        ks_bad = (5, 5)
        bad_prev2 = Conv2dGrowingModule(
            in_channels=1,
            out_channels=self.merge_in_channels,
            kernel_size=ks_bad,
            input_size=self.input_hw,
            device=global_device(),
        )
        with self.assertRaises(ValueError):
            m.set_previous_modules([bad_prev2])

        # Output volume mismatch -> ValueError
        bad_prev3 = Conv2dGrowingModule(
            in_channels=1,
            out_channels=self.merge_in_channels,
            kernel_size=self.kernel_size,
            padding=(1, 1),
            input_size=self.input_hw,
            device=global_device(),
        )
        with self.assertRaises(ValueError):
            m.set_previous_modules([bad_prev3])

    def test_set_next_modules_assertions_and_side_effects(self):
        """Test set_next_modules assertions and side-effects on connected conv modules."""
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        # Prepare two next conv modules with matching kernel sizes
        n1 = Conv2dGrowingModule(
            in_channels=self.merge_in_channels,
            out_channels=4,
            kernel_size=self.kernel_size,
            input_size=m.output_size,
            device=global_device(),
        )
        n2 = Conv2dGrowingModule(
            in_channels=self.merge_in_channels,
            out_channels=6,
            kernel_size=self.kernel_size,
            input_size=m.output_size,
            device=global_device(),
        )

        # Non-empty tensor_s triggers a warning
        dummy = TensorStatistic((2, 2), lambda: (torch.zeros(2, 2), 1))
        dummy.samples = 1
        object.__setattr__(m, "tensor_s", dummy)
        with self.assertWarns(UserWarning):
            m.set_next_modules([n1, n2])

        # Kernel size mismatch among next modules -> assertion
        n3 = Conv2dGrowingModule(
            in_channels=self.merge_in_channels,
            out_channels=5,
            kernel_size=(5, 5),
            input_size=m.output_size,
            device=global_device(),
        )
        with self.assertRaises(AssertionError):
            m.set_next_modules([n1, n3])

        # Kernel size mismatch between merge and next -> assertion
        m2 = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=(5, 5),
            device=global_device(),
        )
        with self.assertRaises(AssertionError):
            m2.set_next_modules([n1])

    def test_construct_full_activity_and_previous_updates(self):
        """Test construct_full_activity content and previous S/M updates
        with a single previous module."""
        # Ensure prev stores input for unfolded access
        self.prev.store_input = True

        # Forward through prev to populate prev.input
        _ = self.prev(self.input_x)

        # Full activity should reduce to prev.unfolded_extended_input when single previous
        full_act = self.merge.construct_full_activity()
        self.assertAllClose(full_act, self.prev.unfolded_extended_input)

        # Compute previous S update and verify shape and basic symmetry
        s_prev, n_s = self.merge.compute_previous_s_update()
        self.assertEqual(n_s, self.input_x.size(0))
        self.assertEqual(s_prev.shape, (full_act.size(1), full_act.size(1)))
        self.assertAllClose(s_prev, s_prev.T)

        # Prepare gradient on merge.pre_activity via a small chain to test M update
        # Make next store_input so merge stores its activity,
        # and merge store_input for gradients
        self.next.store_input = True
        self.merge.store_input = True  # Enable input storage for gradient computation
        seq = torch.nn.Sequential(self.prev, self.merge, self.next)
        y = seq(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        m_prev, n_m = self.merge.compute_previous_m_update()
        self.assertEqual(n_m, self.input_x.size(0))
        self.assertEqual(m_prev.shape, (full_act.size(1), self.merge.in_channels))

    def test_unfolded_extended_activity_and_s_update_conv_next(self):
        """Test unfolded_extended_activity (conv next branch) and
        compute_s_update output shape."""
        # Create a synthetic activity map at the merge (before post merge)
        self.merge.store_activity = True
        h, w = self.merge.output_size
        # Use a non-padded next conv (stride/dilation already set in setUp)
        self.merge.activity = torch.randn(
            self.batch, self.merge_in_channels, h + 2, w + 2, device=global_device()
        )

        unfolded_ext = self.merge.unfolded_extended_activity
        # D = C * kx * ky + 1 (bias)
        d = self.merge_in_channels * self.kernel_size[0] * self.kernel_size[1] + 1
        self.assertEqual(unfolded_ext.shape[1], d)

        s_update, n = self.merge.compute_s_update()
        self.assertEqual(n, self.merge.activity.shape[0])
        self.assertEqual(s_update.shape, (d, d))
        self.assertAllClose(s_update, s_update.T)

    def test_compute_s_update_assertions(self):
        """Test compute_s_update assertions for activity storage."""
        m = self.merge

        # Test when store_activity is False
        m.store_activity = False
        with self.assertRaises(AssertionError):
            m.compute_s_update()

        # Test when activity is None
        m.store_activity = True
        m.activity = None
        with self.assertRaises(AssertionError):
            m.compute_s_update()

    def test_compute_s_update_not_implemented_next_module(self):
        """Test NotImplementedError for unsupported next module types
        in compute_s_update."""
        m = self.merge
        # Set up valid activity - use 2D to avoid tensor concatenation issues in
        # unfolded_extended_activity
        m.store_activity = True
        m.activity = torch.randn(self.batch, m.out_features, device=global_device())

        # Create unsupported module type by bypassing type checks
        class UnsupportedNextModule:
            pass

        # Save original next modules and temporarily replace to test NotImplementedError
        original_next = m.next_modules.copy()

        # We need to bypass the validation in set_next_modules for this test
        # by directly modifying the internal list after validation
        m.set_next_modules([])  # Clear first
        # Then directly inject the unsupported module to test the NotImplementedError path
        m.next_modules = [UnsupportedNextModule()]  # type: ignore[list-item]

        with self.assertRaises(NotImplementedError):
            m.compute_s_update()

        # Restore original next modules using proper method
        m.set_next_modules(original_next)

    def test_set_previous_modules_warnings(self):
        """Test warning paths in set_previous_modules."""
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        # Set up tensor S with samples to trigger warning - line 224
        m.previous_tensor_s = TensorStatistic((2, 2), lambda: (torch.zeros(2, 2), 1))
        m.previous_tensor_s.samples = 1

        # Should warn about non-empty tensor S
        with self.assertWarns(UserWarning):
            m.set_previous_modules([self.prev])

        # Reset and test tensor M warning - line 228
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        m.previous_tensor_m = TensorStatistic((2, 2), lambda: (torch.zeros(2, 2), 1))
        m.previous_tensor_m.samples = 1

        # Should warn about non-empty tensor M
        with self.assertWarns(UserWarning):
            m.set_previous_modules([self.prev])

    def test_update_size_reallocates_tensors(self):
        """Test update_size both when resizing to new totals and when clearing tensors."""
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        # First with no previous modules -> tensors become None
        m.previous_modules = []
        m.previous_tensor_s = TensorStatistic((1, 1), lambda: (torch.zeros(1, 1), 1))
        m.previous_tensor_m = TensorStatistic((1, 1), lambda: (torch.zeros(1, 1), 1))
        m.update_size()
        self.assertIsNone(m.previous_tensor_s)
        self.assertIsNone(m.previous_tensor_m)

        # Now add previous modules to test reallocation
        m.set_previous_modules([self.prev])
        expected_tif = self.prev.in_features + int(self.prev.use_bias)

        # Trigger update_size by modifying module parameters
        m.update_size()
        self.assertEqual(m.total_in_features, expected_tif)
        self.assertIsNotNone(m.previous_tensor_s)
        self.assertIsNotNone(m.previous_tensor_m)

    def test_set_previous_modules_input_size_auto_setting(self):
        """Test that input_size is auto-set when None during set_previous_modules."""
        # Create merge with None input_size
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=(6, 6),  # Start with a size
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        # Use setattr to bypass type checking for test purposes
        setattr(m, "input_size", None)

        # Set previous modules - should auto-set input_size
        m.set_previous_modules([self.prev])

        # Input size should be set from the previous module
        expected_size = (self.prev.out_width, self.prev.out_height)
        self.assertEqual(m.input_size, expected_size)

        # Now with a previous module, then change previous to force reallocation
        m.set_previous_modules([self.prev])
        self.assertIsNotNone(m.previous_tensor_s)
        tif_initial = m.total_in_features

        # Add an extra previous with bias to increase total_in_features
        prev_extra = Conv2dGrowingModule(
            in_channels=1,
            out_channels=self.merge_in_channels,
            kernel_size=self.kernel_size,
            input_size=self.input_hw,
            use_bias=True,
            device=global_device(),
        )
        m.set_previous_modules([self.prev, prev_extra])
        self.assertGreater(m.total_in_features, tif_initial)
        if m.previous_tensor_s is not None:
            self.assertEqual(
                m.previous_tensor_s._shape,
                (m.total_in_features, m.total_in_features),
            )
        if m.previous_tensor_m is not None:
            self.assertEqual(
                m.previous_tensor_m._shape, (m.total_in_features, m.in_channels)
            )

    def test_input_size_auto_set_when_none(self):
        """Ensure input_size is auto-derived from previous module when not set."""
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=(1, 1),  # placeholder, will be set to None below
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )
        # Force None to exercise the branch inside set_previous_modules
        object.__setattr__(m, "input_size", None)
        m.set_previous_modules([self.prev])
        self.assertEqual(m.input_size, (self.prev.out_width, self.prev.out_height))

    def test_compute_s_update_requires_activity(self):
        """Assert compute_s_update raises when activity storage is disabled/missing."""
        m = self.merge
        m.store_activity = False
        m.activity = None
        with self.assertRaises(AssertionError):
            _ = m.compute_s_update()

    def test_unfolded_extended_activity_conv_without_bias(self):
        """Test unfolded_extended_activity for Conv2d path without bias."""
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )
        m.use_bias = False
        m.set_previous_modules([self.prev])
        m.set_next_modules([self.next])  # Conv2d next module

        # Set activity and test unfolded path without bias
        m.store_activity = True
        m.activity = torch.randn(
            self.batch, self.merge_in_channels, 6, 6, device=global_device()
        )

        unfolded = m.unfolded_extended_activity
        expected_d = (
            self.merge_in_channels * self.kernel_size[0] * self.kernel_size[1]
        )  # No +1 for bias
        self.assertEqual(unfolded.shape[1], expected_d)

    def test_update_size_tensor_shape_mismatch(self):
        """Test update_size when tensor shapes don't match."""
        m = Conv2dMergeGrowingModule(
            in_channels=self.merge_in_channels,
            input_size=self.merge.input_size,
            next_kernel_size=self.kernel_size,
            device=global_device(),
        )

        # Set up with previous modules to ensure total_in_features > 0
        m.set_previous_modules([self.prev])

        # Force initial update to create tensors with correct shapes
        m.update_size()

        # Verify tensors were created and get their shapes
        self.assertIsNotNone(m.previous_tensor_s)
        self.assertIsNotNone(m.previous_tensor_m)

        # Manually create tensors with wrong shapes to force reallocation
        wrong_shape_s = (5, 5)  # Different from expected shape
        wrong_shape_m = (5, 2)  # Different from expected shape

        m.previous_tensor_s = TensorStatistic(
            wrong_shape_s,
            device=m.device,
            name=f"S[-1]({m.name})",
            update_function=m.compute_previous_s_update,
        )
        m.previous_tensor_m = TensorStatistic(
            wrong_shape_m,
            device=m.device,
            name=f"M[-1]({m.name})",
            update_function=m.compute_previous_m_update,
        )

        # Call update_size - should trigger tensor reallocation due to shape mismatch
        m.update_size()

        # Verify new tensors were created with correct shapes
        expected_tif = m.total_in_features
        self.assertEqual(m.previous_tensor_s._shape, (expected_tif, expected_tif))
        self.assertEqual(m.previous_tensor_m._shape, (expected_tif, m.in_channels))


class TestConv2dGrowingModuleBase(TorchTestCase):
    _tested_class = Conv2dGrowingModule

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.input_x = torch.randn(5, 2, 10, 10, device=global_device())

    def create_demo_layers(
        self, bias: bool, hidden_channels: int = 5
    ) -> tuple[_tested_class, _tested_class]:
        demo_in = self._tested_class(
            in_channels=2,
            out_channels=hidden_channels,
            kernel_size=(3, 3),
            padding=1,
            use_bias=bias,
            device=global_device(),
            name="first_layer",
        )
        demo_out = self._tested_class(
            in_channels=hidden_channels,
            out_channels=7,
            kernel_size=(5, 5),
            use_bias=bias,
            previous_module=demo_in,
            device=global_device(),
            name="second_layer",
        )
        return demo_in, demo_out

    def create_demo_layers_with_extension(
        self, bias: bool, hidden_channels: int = 5, include_eigenvalues: bool = False
    ):
        extension_size = 3
        demo_in, demo_out = self.create_demo_layers(bias, hidden_channels)
        demo_in.extended_output_layer = torch.nn.Conv2d(
            in_channels=demo_in.in_channels,
            out_channels=extension_size,
            kernel_size=(3, 3),
            bias=bias,
            device=global_device(),
        )
        demo_out.extended_input_layer = torch.nn.Conv2d(
            in_channels=extension_size,
            out_channels=demo_out.out_channels,
            kernel_size=(5, 5),
            bias=bias,
            device=global_device(),
        )
        if include_eigenvalues:
            demo_out.eigenvalues_extension = torch.rand(
                extension_size, device=global_device()
            )
            demo_out.eigenvalues_extension[0] += 1.0  # ensure decreasing order
        return demo_in, demo_out


class TestConv2dGrowingModule(TestConv2dGrowingModuleBase):
    _tested_class = Conv2dGrowingModule

    def setUp(self):
        super().setUp()
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

        self.bias_demos = {True: self.demo_b, False: self.demo}

        self.demo_couple = {b: self.create_demo_layers(b) for b in (True, False)}

    def test_get_fan_in_from_layer(self):
        """Test get_fan_in_from_layer method."""
        layer = self.demo_couple[True][0]
        self.assertEqual(
            layer.get_fan_in_from_layer(torch.nn.Conv2d(3, 7, kernel_size=(2, 5))),
            3 * 2 * 5,
        )
        self.assertEqual(
            layer.get_fan_in_from_layer(torch.nn.Conv2d(2, 3, kernel_size=(7, 5))),
            2 * 7 * 5,
        )

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
        We get each e_i = (0, ..., 0, 1, 0, ..., 0) as input and the loss is
        the norm of the output
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

    @unittest_parametrize(
        (
            {"zero_fan_in": True, "zero_fan_out": True, "bias": True},
            {"zero_fan_in": False, "zero_fan_out": True, "bias": True},
            {"zero_fan_in": True, "zero_fan_out": False, "bias": False},
        )
    )
    def test_sub_select_optimal_added_parameters_zeroing(
        self,
        bias: bool = True,
        zero_fan_in: bool = True,
        zero_fan_out: bool = False,
        select: int = 1,
    ) -> None:
        """Test sub_select with zeros_if_not_enough for Conv2d layers."""
        layer_in, layer_out = self.create_demo_layers_with_extension(
            bias=bias,
            include_eigenvalues=True,
        )

        # Set eigenvalues with clear ordering
        extension_size = layer_out.eigenvalues_extension.shape[0]
        layer_out.eigenvalues_extension = torch.tensor(
            [1.0, 0.5, 0.1][:extension_size], device=global_device()
        )

        layer_out.sub_select_optimal_added_parameters(
            keep_neurons=select,
            zeros_if_not_enough=True,
            zeros_fan_in=zero_fan_in,
            zeros_fan_out=zero_fan_out,
        )

        assert isinstance(layer_in.extended_output_layer, torch.nn.Conv2d)
        assert isinstance(layer_out.extended_input_layer, torch.nn.Conv2d)

        # Check eigenvalues are zeroed for non-selected neurons
        self.assertAllClose(
            layer_out.eigenvalues_extension[select:],
            torch.zeros_like(layer_out.eigenvalues_extension[select:]),
        )

        if zero_fan_in:
            # Check that fan-in weights are zeroed for non-selected neurons
            self.assertTrue(
                torch.all(layer_in.extended_output_layer.weight[select:] == 0)
            )
            if bias and layer_in.extended_output_layer.bias is not None:
                self.assertTrue(
                    torch.all(layer_in.extended_output_layer.bias[select:] == 0)
                )

        if zero_fan_out:
            # Check that fan-out weights are zeroed for non-selected neurons
            self.assertTrue(
                torch.all(layer_out.extended_input_layer.weight[:, select:] == 0)
            )


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

        # Use indicator batch for Conv2d - each sample has 1 in different
        # spatial locations
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
            demo_layer_2.tensor_n, torch.zeros_like(demo_layer_2.tensor_n), atol=1e-6
        )
        assert isinstance(demo_layer_2.eigenvalues_extension, torch.Tensor)
        self.assertAllClose(
            demo_layer_2.eigenvalues_extension,
            torch.zeros_like(demo_layer_2.eigenvalues_extension),
            atol=2e-6,
        )

    def test_compute_m_prev_without_intermediate_input(self):
        """Check that the batch size is computed using stored variables for FullConv2d"""
        # Use predefined demo_couple objects
        demo_layer_1, demo_layer_2 = self.demo_couple[
            False
        ]  # Use without bias for simplicity

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
        Test sub_select_optimal_added_parameters in merge to
        compute_optimal_added_parameters
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
            message=(
                "The tensor_m_prev should be equal to the tensor_n when the delta is zero"
            ),
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

        # tensor_s_growth should return the internal _tensor_s_growth,
        # not previous module's tensor_s
        tensor_s_growth = demo.tensor_s_growth
        self.assertIs(tensor_s_growth, demo._tensor_s_growth)

        # Verify it's a TensorStatistic
        self.assertIsInstance(tensor_s_growth, TensorStatistic)

    @unittest_parametrize(({"bias": True}, {"bias": False}))
    def test_tensor_s_growth_independence_from_previous_module(self, bias):
        """Test that FullConv2dGrowingModule tensor_s_growth is independent of
        previous module."""
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

        # tensor_s_growth for FullConv2dGrowingModule should NOT redirect
        # to previous module
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

        # For FullConv2dGrowingModule, we need to ensure _tensor_s_growth has
        # been computed
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
            # If no samples, just verify the tensor_s_growth property
            # exists and is correct type
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

        # Use indicator batch for Conv2d - each sample has 1 in different
        # spatial locations
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
            demo_layer_2.tensor_n, torch.zeros_like(demo_layer_2.tensor_n), atol=1e-6
        )
        self.assertAllClose(
            demo_layer_2.eigenvalues_extension,
            torch.zeros_like(demo_layer_2.eigenvalues_extension),
            atol=1e-6,
        )

    def test_compute_m_prev_without_intermediate_input_restricted(self):
        """Check that the batch size is computed using stored variables
        for RestrictedConv2d"""
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
        demo_layer = RestrictedConv2dGrowingModule(
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
            message="The constructed convolution is not similar to a linear layer",
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
            message=(
                "The tensor_n should be equal to the tensor_m_prev when the delta is zero"
            ),
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
        """Test compute_optimal_added_parameters with use_projected_gradient=False
        for RestrictedConv2dGrowingModule."""
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

    def test_bordered_unfolded_extended_prev_input_shape(self):
        """
        Test that bordered_unfolded_extended_prev_input runs correctly and
        returns proper shape.

        This test mimics the setup from minimal_crashing_code_2 to ensure that the
        bordered_unfolded_extended_prev_input property works correctly and returns
        a tensor with the expected shape, particularly testing the fix for the
        border effect bug.
        """
        # Create previous module (first conv layer) - mimics minimal_crashing_code_2
        prev_module = RestrictedConv2dGrowingModule(
            in_channels=2,
            out_channels=7,
            kernel_size=(3, 5),
            stride=2,
            padding=1,
            use_bias=False,
            input_size=(7, 11),
        )

        # Create current module (second conv layer)
        current_module = RestrictedConv2dGrowingModule(
            in_channels=7,
            out_channels=11,
            kernel_size=(5, 3),
            stride=1,
            padding=1,
            use_bias=False,
            previous_module=prev_module,
        )

        # Set up input and forward pass
        x = torch.randn(1, 2, 13, 17, device=global_device())
        prev_module.store_input = True

        # Forward pass through both modules
        intermediate = prev_module(x)
        output = current_module(intermediate)

        # Update input sizes
        # Access bordered_unfolded_extended_prev_input - this should not crash
        bordered_tensor = current_module.bordered_unfolded_extended_prev_input

        # Verify tensor structure
        self.assertShapeEqual(
            bordered_tensor,
            (
                output.shape[0],  # Batch size
                prev_module.in_channels
                * prev_module.kernel_size[0]
                * prev_module.kernel_size[1],
                output.shape[2] * output.shape[3],
            ),
        )


class TestCreateLayerExtensionsConv2d(TestConv2dGrowingModuleBase):
    """Test create_layer_extensions method for Conv2dGrowingModule."""

    def test_create_layer_extensions_with_copy_uniform(self) -> None:
        """Test create_layer_extensions with copy_uniform initialization."""

        # Subtest 1: With features
        with self.subTest(case="with_features"):
            # Create two connected growing modules without extensions
            layer_in, layer_out = self.create_demo_layers(bias=True, hidden_channels=5)

            # Store existing weight stds for comparison
            layer_in_weight_std = layer_in.layer.weight.std().item()
            layer_out_weight_std = layer_out.layer.weight.std().item()

            # Call create_layer_extensions with copy_uniform initialization
            extension_size = 3
            layer_out.create_layer_extensions(
                extension_size=extension_size,
                output_extension_init="copy_uniform",
                input_extension_init="copy_uniform",
            )

            # Verify extensions were created
            self.assertIsInstance(
                layer_in.extended_output_layer,
                torch.nn.Conv2d,
                msg="extended_output_layer should be created",
            )
            assert isinstance(layer_in.extended_output_layer, torch.nn.Conv2d)

            self.assertIsInstance(
                layer_out.extended_input_layer,
                torch.nn.Conv2d,
                msg="extended_input_layer should be created",
            )
            assert isinstance(layer_out.extended_input_layer, torch.nn.Conv2d)

            # Verify newly added weights std match existing weights
            # For copy_uniform: bound = sqrt(3) * std(W_next)
            # For extended_output_layer, it uses layer_out weights
            # For extended_input_layer, it uses layer_in weights

            # The std should approximately match the layer weights
            # Allow some tolerance due to random initialization
            self.assertAlmostEqual(
                layer_in.extended_output_layer.weight.std().item(),
                layer_in_weight_std,
                delta=layer_out_weight_std * 0.5,
                msg="extended_output_layer std should match layer_out weights std",
            )
            self.assertAlmostEqual(
                layer_out.extended_input_layer.weight.std().item(),
                layer_out_weight_std,
                delta=layer_out_weight_std * 0.5,
                msg="extended_input_layer std should match layer_out weights std",
            )

            # Perform extended forward pass with random input
            # Extended forward through layer_in to get both standard and extended
            # outputs
            y, y_ext = layer_in.extended_forward(x=self.input_x)
            assert isinstance(y_ext, torch.Tensor)

            # Verify intermediate extended results have correct shapes
            self.assertShapeEqual(
                y,
                (self.input_x.shape[0], layer_in.out_channels, -1, -1),
                msg="layer_in standard output has correct batch size",
            )
            self.assertShapeEqual(
                y_ext,
                (self.input_x.shape[0], extension_size, -1, -1),
                msg="layer_in extended output has correct shape",
            )

            # Extended forward through layer_out
            z, z_ext = layer_out.extended_forward(
                x=y, x_ext=y_ext, use_optimal_delta=False
            )
            self.assertShapeEqual(
                z,
                (self.input_x.shape[0], layer_out.out_channels, -1, -1),
                msg="layer_out standard output has correct shape",
            )
            self.assertIsNone(
                z_ext,
                msg="layer_out has no extended output when only input extension added",
            )

        # Subtest 2: Without features (hidden_channels=0)
        with self.subTest(case="without_features"):
            # Create two connected growing modules with 0 hidden channels
            layer_in, layer_out = self.create_demo_layers(bias=False, hidden_channels=0)

            # When out_channels=0, the layer has no weights
            # So copy_uniform should fallback to 1/sqrt(fan_in)
            extension_size = 3

            with self.assertWarns(UserWarning):
                # UserWarning: std(): degrees of freedom is <= 0.
                # This happens because the layer has no weights to compute std from.
                layer_out.create_layer_extensions(
                    extension_size=extension_size,
                    output_extension_init="copy_uniform",
                    input_extension_init="copy_uniform",
                )

            # Verify extensions were created
            self.assertIsInstance(
                layer_in.extended_output_layer,
                torch.nn.Conv2d,
                msg="extended_output_layer should be created",
            )
            self.assertIsInstance(
                layer_out.extended_input_layer,
                torch.nn.Conv2d,
                msg="extended_input_layer should be created",
            )

            # Type assertions for linter
            assert isinstance(layer_in.extended_output_layer, torch.nn.Conv2d)
            assert isinstance(layer_out.extended_input_layer, torch.nn.Conv2d)

            # When there are no hidden channels, the std should be 1/sqrt(fan_in)
            # For Conv2d: fan_in = in_channels * kernel_h * kernel_w
            # For extended_output_layer:
            # fan_in = layer_in.in_channels * kernel_h * kernel_w
            expected_output_ext_std = (
                1.0
                / (
                    layer_in.in_channels
                    * layer_in.kernel_size[0]
                    * layer_in.kernel_size[1]
                )
                ** 0.5
            )
            # For extended_input_layer:
            # fan_in = extension_size * kernel_h * kernel_w
            expected_input_ext_std = (
                1.0
                / (extension_size * layer_out.kernel_size[0] * layer_out.kernel_size[1])
                ** 0.5
            )

            # Verify std matches expected values
            # Allow tolerance for small sample statistics
            self.assertAlmostEqual(
                layer_in.extended_output_layer.weight.std().item(),
                expected_output_ext_std,
                delta=expected_output_ext_std * 0.5,
                msg=f"extended_output_layer std should be ~{expected_output_ext_std}",
            )
            self.assertAlmostEqual(
                layer_out.extended_input_layer.weight.std().item(),
                expected_input_ext_std,
                delta=expected_input_ext_std * 0.5,
                msg=f"extended_input_layer std should be ~{expected_input_ext_std}",
            )


class TestNeuronCountingConv2d(TestConv2dGrowingModuleBase):
    """Test in_neurons property and growth-related methods for Conv2dGrowingModule."""

    def test_in_neurons_returns_in_channels(self) -> None:
        """Test that in_neurons returns in_channels for Conv2d modules."""
        layer = Conv2dGrowingModule(
            in_channels=5,
            out_channels=3,
            kernel_size=(3, 3),
            device=global_device(),
        )
        self.assertEqual(layer.in_neurons, 5)
        self.assertEqual(layer.in_neurons, layer.in_channels)

    def test_target_in_channels_initialization(self) -> None:
        """Test that target_in_neurons is correctly initialized via target_in_channels."""
        # Without target
        layer = Conv2dGrowingModule(
            in_channels=5,
            out_channels=3,
            kernel_size=(3, 3),
            device=global_device(),
        )
        self.assertIsNone(layer.target_in_neurons)
        self.assertEqual(layer._initial_in_neurons, 5)

        # With target
        layer_with_target = Conv2dGrowingModule(
            in_channels=5,
            out_channels=3,
            kernel_size=(3, 3),
            target_in_channels=10,
            device=global_device(),
        )
        self.assertEqual(layer_with_target.target_in_neurons, 10)
        self.assertEqual(layer_with_target._initial_in_neurons, 5)

    def test_missing_neurons_for_conv2d(self) -> None:
        """Test missing_neurons for Conv2dGrowingModule."""
        layer = Conv2dGrowingModule(
            in_channels=5,
            out_channels=3,
            kernel_size=(3, 3),
            target_in_channels=10,
            device=global_device(),
        )
        self.assertEqual(layer.missing_neurons(), 5)

    def test_number_of_neurons_to_add_for_conv2d(self) -> None:
        """Test number_of_neurons_to_add for Conv2dGrowingModule."""
        layer = Conv2dGrowingModule(
            in_channels=5,
            out_channels=3,
            kernel_size=(3, 3),
            target_in_channels=15,
            device=global_device(),
        )
        # Total to add: 15 - 5 = 10
        self.assertEqual(layer.number_of_neurons_to_add(number_of_growth_steps=2), 5)


if __name__ == "__main__":
    main()
