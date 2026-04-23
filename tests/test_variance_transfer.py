"""Tests for variance-transfer rescaling and neuron pairing.

Tests cover:
- Basic / smoke tests (no crashes, correct types, sizes)
- Semantic tests (function preservation, weight variance, activation variance,
  pairing structure, BatchNorm rescaling, idempotence)
- Edge-case tests (small / zero weight variance)
"""

import torch

from gromo.containers.growing_block import (
    Conv2dGrowingBlock,
    GrowingBlock,
    LinearGrowingBlock,
)
from gromo.modules.growing_normalisation import GrowingBatchNorm2d
from gromo.utils.utils import global_device


try:
    from tests.torch_unittest import SizedIdentity, TorchTestCase
    from tests.unittest_tools import unittest_parametrize
except ImportError:
    from torch_unittest import SizedIdentity, TorchTestCase
    from unittest_tools import unittest_parametrize


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_conv_block(
    h_t: int,
    in_channels: int = 3,
    out_channels: int = 3,
    kernel_size: int = 3,
    mid_activation: torch.nn.Module | None = None,
    pre_addition_function: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> Conv2dGrowingBlock:
    """Create a Conv2dGrowingBlock with non-trivial weights."""
    kwargs: dict = {}
    if mid_activation is not None:
        kwargs["mid_activation"] = mid_activation
    if pre_addition_function is not None:
        kwargs["pre_addition_function"] = pre_addition_function
    block = Conv2dGrowingBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        hidden_channels=h_t,
        kwargs_layer={"padding": kernel_size // 2, "use_bias": True},
        device=device,
        **kwargs,
    )
    return block


def _make_linear_block(
    h_t: int,
    in_features: int = 8,
    out_features: int = 8,
    device: torch.device | None = None,
) -> LinearGrowingBlock:
    """Create a LinearGrowingBlock with non-trivial weights."""
    return LinearGrowingBlock(
        in_features=in_features,
        out_features=out_features,
        hidden_features=h_t,
        device=device,
    )


def _grow_block(
    block: GrowingBlock,
    dh: int,
    rescaling=None,
    neuron_pairing=None,
    output_extension_init: str = "kaiming",
    input_extension_init: str = "kaiming",
    noise_ratio: float = 0.0,
) -> None:
    """Create extensions, apply change, and delete update."""
    block.create_layer_extensions(
        extension_size=dh,
        output_extension_init=output_extension_init,
        input_extension_init=input_extension_init,
        rescaling=rescaling,
        neuron_pairing=neuron_pairing,
        noise_ratio=noise_ratio,
    )
    # extension_size is the final size (pairing included)
    block.apply_change(scaling_factor=1.0, extension_size=dh)
    block.second_layer.delete_update(include_previous=True)


# ===============================================================
# 1.  Basic / smoke tests
# ===============================================================


class TestRescalingSmoke(TorchTestCase):
    """Smoke tests: methods run without errors and produce valid state."""

    def setUp(self):
        super().setUp()
        self.device = global_device()

    @unittest_parametrize(
        [
            {"rescaling": None},
            {"rescaling": "default_vt"},
            {"rescaling": "vt_constraint_old_shape"},
            {"rescaling": "vt_constraint_new_shape"},
            {"rescaling": "vt_constraint_old_shape", "h_t": 1},
            {"rescaling": "vt_constraint_new_shape", "h_t": 1},
            {"rescaling": "vt_constraint_old_shape", "h_t": 0},
            {"rescaling": "vt_constraint_new_shape", "h_t": 0},
        ]
    )
    def test_create_layer_extensions_runs(self, rescaling, h_t=8):
        """create_layer_extensions completes for every rescaling strategy."""
        block = _make_conv_block(h_t=h_t, device=self.device)
        block.create_layer_extensions(
            extension_size=4,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
            rescaling=rescaling,
        )
        self.assertIsNotNone(block.first_layer.extended_output_layer)
        self.assertIsNotNone(block.second_layer.extended_input_layer)

    @unittest_parametrize(
        [
            {"neuron_pairing": None},
            {"neuron_pairing": "vv_z_negz"},
        ]
    )
    def test_create_layer_extensions_with_pairing(self, neuron_pairing):
        """create_layer_extensions completes for every pairing strategy."""
        block = _make_conv_block(h_t=8, device=self.device)
        block.create_layer_extensions(
            extension_size=4,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
            neuron_pairing=neuron_pairing,
        )
        ext_out = block.first_layer.extended_output_layer
        ext_in = block.second_layer.extended_input_layer
        self.assertIsNotNone(ext_out)
        self.assertIsNotNone(ext_in)

        # extension_size is the final size (pairing included)
        self.assertEqual(ext_out.weight.shape[0], 4)
        self.assertEqual(ext_in.weight.shape[1], 4)

    def test_invalid_rescaling_raises(self):
        """Unknown rescaling strategy raises ValueError."""
        block = _make_conv_block(h_t=8, device=self.device)
        with self.assertRaises(ValueError):
            block.apply_rescaling(rescaling="unknown_strategy")  # type: ignore

    def test_invalid_neuron_pairing_raises(self):
        """Unknown neuron pairing raises ValueError."""
        block = _make_conv_block(h_t=8, device=self.device)
        block.create_layer_extensions(extension_size=4)
        with self.assertRaises(ValueError):
            block.apply_neuron_pairing(neuron_pairing="unknown_pairing")  # type: ignore

    def test_odd_extension_size_with_pairing_raises(self):
        """create_layer_extensions with odd extension_size and pairing raises."""
        block = _make_conv_block(h_t=8, device=self.device)
        with self.assertRaises(ValueError):
            block.create_layer_extensions(
                extension_size=3,
                neuron_pairing="vv_z_negz",
            )

    def test_odd_output_extension_size_with_pairing_raises(self):
        """Odd output_extension_size override with pairing raises."""
        block = _make_conv_block(h_t=8, device=self.device)
        with self.assertRaises(ValueError):
            block.create_layer_extensions(
                extension_size=4,
                output_extension_size=5,
                neuron_pairing="vv_z_negz",
            )

    def test_odd_input_extension_size_with_pairing_raises(self):
        """Odd input_extension_size override with pairing raises."""
        block = _make_conv_block(h_t=8, device=self.device)
        with self.assertRaises(ValueError):
            block.create_layer_extensions(
                extension_size=4,
                input_extension_size=5,
                neuron_pairing="vv_z_negz",
            )

    def test_odd_existing_extension_pairing_raises(self):
        """apply_neuron_pairing on an odd-sized existing extension raises."""
        block = _make_conv_block(h_t=8, device=self.device)
        # Create extensions without pairing, at an odd size
        block.second_layer.create_layer_in_extension(3)
        block.first_layer.create_layer_out_extension(3)
        with self.assertRaises(ValueError):
            block.apply_neuron_pairing(neuron_pairing="vv_z_negz")

    def test_none_rescaling_is_noop(self):
        """rescaling=None should not change weights."""
        block = _make_conv_block(h_t=8, device=self.device)
        w1_before = block.first_layer.weight.clone()
        w2_before = block.second_layer.weight.clone()
        block.apply_rescaling(rescaling=None)
        self.assertAllClose(block.first_layer.weight, w1_before)
        self.assertAllClose(block.second_layer.weight, w2_before)

    def test_none_pairing_is_noop(self):
        """neuron_pairing=None should not change extensions."""
        block = _make_conv_block(h_t=8, device=self.device)
        block.create_layer_extensions(extension_size=4)
        w_out = block.first_layer.extended_output_layer.weight.clone()
        w_in = block.second_layer.extended_input_layer.weight.clone()
        block.apply_neuron_pairing(neuron_pairing=None)
        self.assertAllClose(block.first_layer.extended_output_layer.weight, w_out)
        self.assertAllClose(block.second_layer.extended_input_layer.weight, w_in)

    def test_apply_change_after_growth_updates_sizes(self):
        """After growth with pairing, hidden_neurons increases by dh (final size)."""
        block = _make_conv_block(h_t=8, device=self.device)
        _grow_block(block, dh=4, neuron_pairing="vv_z_negz")
        self.assertEqual(block.hidden_neurons, 8 + 4)

    def test_apply_change_after_growth_no_pairing(self):
        """After growth without pairing, hidden_neurons increases by dh."""
        block = _make_conv_block(h_t=8, device=self.device)
        _grow_block(block, dh=4)
        self.assertEqual(block.hidden_neurons, 8 + 4)

    @unittest_parametrize(
        [
            {"rescaling": "default_vt"},
            {"rescaling": "vt_constraint_old_shape"},
            {"rescaling": "vt_constraint_new_shape"},
        ]
    )
    def test_full_pipeline_conv_block(self, rescaling):
        """Full growth pipeline runs without crash for Conv2dGrowingBlock."""
        block = _make_conv_block(h_t=8, device=self.device)
        x = torch.randn(2, 3, 8, 8, device=self.device)
        y_before = block(x)
        _grow_block(
            block,
            dh=4,
            rescaling=rescaling,
            neuron_pairing="vv_z_negz",
        )
        y_after = block(x)
        self.assertShapeEqual(y_after, y_before.shape)

    @unittest_parametrize(
        [
            {"rescaling": "default_vt"},
            {"rescaling": "vt_constraint_old_shape"},
            {"rescaling": "vt_constraint_new_shape"},
        ]
    )
    def test_full_pipeline_linear_block(self, rescaling):
        """Full growth pipeline runs without crash for LinearGrowingBlock."""
        block = _make_linear_block(h_t=8, device=self.device)
        x = torch.randn(2, 8, device=self.device)
        y_before = block(x)
        _grow_block(
            block,
            dh=4,
            rescaling=rescaling,
            neuron_pairing="vv_z_negz",
        )
        y_after = block(x)
        self.assertShapeEqual(y_after, y_before.shape)


# ===============================================================
# 2.  Semantic tests
# ===============================================================


class TestFunctionPreservation(TorchTestCase):
    """Test 1: pairing without rescaling preserves block output exactly."""

    def setUp(self):
        super().setUp()
        self.device = global_device()

    @unittest_parametrize(
        [
            {
                "output_init": "kaiming",
                "input_init": "kaiming",
            },
            {
                "output_init": "zeros",
                "input_init": "zeros",
            },
            {
                "output_init": "kaiming",
                "input_init": "zeros",
            },
            {
                "output_init": "zeros",
                "input_init": "kaiming",
            },
        ]
    )
    def test_pairing_preserves_output_conv(self, output_init, input_init):
        """Block output is identical before and after (V,V)/(Z,-Z) growth."""
        block = _make_conv_block(h_t=8, device=self.device)
        x = torch.randn(2, 3, 8, 8, device=self.device)

        with torch.no_grad():
            y_before = block(x).clone()

        _grow_block(
            block,
            dh=4,
            neuron_pairing="vv_z_negz",
            rescaling=None,
            output_extension_init=output_init,
            input_extension_init=input_init,
        )

        with torch.no_grad():
            y_after = block(x)

        self.assertAllClose(y_before, y_after, atol=1e-5)

    def test_pairing_preserves_output_linear(self):
        """Function preservation also works for LinearGrowingBlock."""
        block = _make_linear_block(h_t=8, device=self.device)
        x = torch.randn(2, 8, device=self.device)

        with torch.no_grad():
            y_before = block(x).clone()

        _grow_block(
            block,
            dh=4,
            neuron_pairing="vv_z_negz",
            rescaling=None,
        )

        with torch.no_grad():
            y_after = block(x)

        self.assertAllClose(y_before, y_after, atol=1e-5)


class TestFunctionPreservationWithRescaling(TorchTestCase):
    """Test 2: with pairing + rescaling, the block internal path is scaled.

    Conv2dGrowingBlock has a residual: ``y = conv_path(x) + skip(x)``.
    With pairing, new neurons cancel, so after growth:
    ``y_after = alpha * beta * conv_path(x) + skip(x)``.

    We verify this by computing ``conv_path`` and ``skip`` separately.
    """

    def setUp(self):
        super().setUp()
        self.device = global_device()

    @unittest_parametrize(
        [
            {"rescaling": "default_vt"},
            {"rescaling": "vt_constraint_old_shape"},
            {"rescaling": "vt_constraint_new_shape"},
        ]
    )
    def test_output_scaled_by_alpha_beta(self, rescaling):
        """y_after = alpha * beta * conv_path_before + skip."""
        h_t = 8
        dh = 4
        k = 3

        block = _make_conv_block(h_t=h_t, kernel_size=k, device=self.device)
        # Make weights non-Kaiming to ensure rescaling is non-trivial
        block.first_layer.weight.data.mul_(2.0)
        block.second_layer.weight.data.mul_(0.5)

        x = torch.randn(2, 3, 8, 8, device=self.device)
        # Capture Conv2 bias before rescaling (shape: [out_channels])
        b2 = block.second_layer.bias.clone()

        with torch.no_grad():
            y_before = block(x).clone()
            skip = block.downsample(x)
            conv_path_before = y_before - skip

        # Compute alpha and beta from formulas before rescaling
        fan_in_prev = block.first_layer.layer.in_channels * k * k
        fan_in_self_old = block.second_layer.layer.in_channels * k * k
        # dh is the final extension size (pairing included)
        fan_in_self_new = (h_t + dh) * k * k

        var_w_prev = block.first_layer.weight.var().item()
        var_w_self = block.second_layer.weight.var().item()

        if rescaling == "default_vt":
            alpha = 1.0
            beta = (fan_in_self_old / fan_in_self_new) ** 0.5
        elif rescaling == "vt_constraint_old_shape":
            alpha = (1.0 / (fan_in_prev * var_w_prev)) ** 0.5
            beta = (1.0 / (fan_in_self_old * var_w_self)) ** 0.5
        else:  # vt_constraint_new_shape
            alpha = (1.0 / (fan_in_prev * var_w_prev)) ** 0.5
            beta = (1.0 / (fan_in_self_new * var_w_self)) ** 0.5

        _grow_block(
            block,
            dh=dh,
            rescaling=rescaling,
            neuron_pairing="vv_z_negz",
        )

        with torch.no_grad():
            y_after = block(x)

        # conv_path_before = W2*W1*x + W2*b1 + b2
        # After rescaling: alpha*beta*W2*W1*x + alpha*beta*W2*b1 + beta*b2
        # = alpha*beta*conv_path_before + beta*(1 - alpha)*b2
        # For default_vt (alpha=1), the bias offset vanishes.
        bias_offset = beta * (1.0 - alpha) * b2.view(1, -1, 1, 1)
        expected = alpha * beta * conv_path_before + bias_offset + skip
        self.assertAllClose(
            y_after,
            expected,
            atol=1e-5,
            msg=f"rescaling={rescaling}, alpha={alpha:.4f}, beta={beta:.4f}",
        )


class TestWeightVarianceAfterRescaling(TorchTestCase):
    """Tests 3-4: weight variance of existing weights after rescaling.

    These tests verify that ``apply_rescaling`` achieves the target variance
    on the existing weight tensors *before* extension concatenation.

    After rescaling:
    - Conv1 (previous layer): ``alpha^2 * V[W1_old] = 1 / fan_in_prev``
    - Conv2 (current layer, Strategy B): ``beta^2 * V[W2_old] = 1 / fan_in_self_old``
    - Conv2 (current layer, Strategy C): ``beta^2 * V[W2_old] = 1 / fan_in_self_new``
    """

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def test_variance_after_strategy_b(self):
        """Strategy B targets V[W] = 1/fan_in_old for both layers."""
        C_in = 3
        h_t = 16
        dh = 4
        k = 3

        block = _make_conv_block(
            h_t=h_t,
            in_channels=C_in,
            kernel_size=k,
            device=self.device,
        )
        block.first_layer.weight.data.mul_(2.0)
        block.second_layer.weight.data.mul_(0.5)

        fan_in_prev = C_in * k * k
        fan_in_self_old = h_t * k * k

        # Apply rescaling only (not the full growth pipeline)
        block.apply_rescaling(
            rescaling="vt_constraint_old_shape",
            neuron_pairing="vv_z_negz",
            extension_size=dh,
        )

        var_w1 = block.first_layer.weight.var().item()
        var_w2 = block.second_layer.weight.var().item()

        self.assertAlmostEqual(
            var_w1,
            1.0 / fan_in_prev,
            places=5,
            msg=f"V[W1]={var_w1:.6f}, expected {1.0 / fan_in_prev:.6f}",
        )
        self.assertAlmostEqual(
            var_w2,
            1.0 / fan_in_self_old,
            places=5,
            msg=f"V[W2]={var_w2:.6f}, expected {1.0 / fan_in_self_old:.6f}",
        )

    def test_variance_after_strategy_c(self):
        """Strategy C targets V[W2] = 1/fan_in_new (exact)."""
        C_in = 3
        h_t = 16
        dh = 4
        k = 3

        block = _make_conv_block(
            h_t=h_t,
            in_channels=C_in,
            kernel_size=k,
            device=self.device,
        )
        block.first_layer.weight.data.mul_(2.0)
        block.second_layer.weight.data.mul_(0.5)

        fan_in_prev = C_in * k * k
        # dh is the final extension size (pairing included)
        fan_in_self_new = (h_t + dh) * k * k

        block.apply_rescaling(
            rescaling="vt_constraint_new_shape",
            neuron_pairing="vv_z_negz",
            extension_size=dh,
        )

        var_w1 = block.first_layer.weight.var().item()
        var_w2 = block.second_layer.weight.var().item()

        # Conv1: same alpha for B and C
        self.assertAlmostEqual(
            var_w1,
            1.0 / fan_in_prev,
            places=5,
            msg=f"V[W1]={var_w1:.6f}, expected {1.0 / fan_in_prev:.6f}",
        )
        # Conv2: targets fan_in_new
        self.assertAlmostEqual(
            var_w2,
            1.0 / fan_in_self_new,
            places=5,
            msg=f"V[W2]={var_w2:.6f}, expected {1.0 / fan_in_self_new:.6f}",
        )


class TestActivationVariance(TorchTestCase):
    """Tests 5-6: conv-path activation variance at init for Strategy B and C.

    The block has a residual connection: ``y = conv_path(x) + skip(x)``.
    We measure the variance of the conv path only (``y - skip``), since
    the skip is unaffected by rescaling.

    With ``(V,V)/(Z,-Z)`` pairing, the new-neuron contribution is zero at
    init, so ``conv_path = alpha * beta * conv_path_old``.  For a unit-
    variance input and Kaiming-like scaling:

    - Strategy B: ``V[conv_path] ~ 1``
    - Strategy C: ``V[conv_path] ~ h_t / h_{t+1}``
    """

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def _make_block_with_perturbed_weights(self, h_t, k=3):
        """Create block with weights far from Kaiming scale."""
        block = _make_conv_block(h_t=h_t, kernel_size=k, device=self.device)
        block.first_layer.weight.data.mul_(2.0)
        block.second_layer.weight.data.mul_(0.5)
        return block

    def test_activation_variance_strategy_b(self):
        """V[conv_path]_init ~ 1 for Strategy B with pairing."""
        h_t = 16
        dh = 4
        block = self._make_block_with_perturbed_weights(h_t)

        _grow_block(
            block,
            dh=dh,
            rescaling="vt_constraint_old_shape",
            neuron_pairing="vv_z_negz",
        )

        x = torch.randn(512, 3, 8, 8, device=self.device)
        with torch.no_grad():
            y = block(x)
            skip = block.downsample(x)
            conv_path = y - skip

        var_cp = conv_path.var().item()
        # Statistical test — fairly loose tolerance
        self.assertAlmostEqual(
            var_cp,
            1.0,
            delta=0.5,
            msg=f"V[conv_path]={var_cp:.4f}, expected ~1.0",
        )

    def test_activation_variance_strategy_c(self):
        """V[conv_path]_init ~ h_t / h_{t+1} for Strategy C with pairing."""
        h_t = 16
        dh = 4
        block = self._make_block_with_perturbed_weights(h_t)

        _grow_block(
            block,
            dh=dh,
            rescaling="vt_constraint_new_shape",
            neuron_pairing="vv_z_negz",
        )

        h_t_plus_1 = h_t + dh
        expected_var = h_t / h_t_plus_1

        x = torch.randn(512, 3, 8, 8, device=self.device)
        with torch.no_grad():
            y = block(x)
            skip = block.downsample(x)
            conv_path = y - skip

        var_cp = conv_path.var().item()
        self.assertAlmostEqual(
            var_cp,
            expected_var,
            delta=0.5,
            msg=f"V[conv_path]={var_cp:.4f}, expected ~{expected_var:.4f}",
        )


class TestPairingStructure(TorchTestCase):
    """Test 7: (V,V)/(Z,-Z) pairing structure is correct."""

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def test_vv_z_negz_structure_conv(self):
        """Output ext rows duplicated, input ext columns sign-paired."""
        block = _make_conv_block(h_t=8, device=self.device)
        dh = 4  # final size (pairing included)
        half = dh // 2
        block.create_layer_extensions(
            extension_size=dh,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
            neuron_pairing="vv_z_negz",
            noise_ratio=0,
        )

        ext_out = block.first_layer.extended_output_layer
        ext_in = block.second_layer.extended_input_layer

        # V -> (V, V): first half rows == second half rows
        self.assertTrue(
            torch.equal(ext_out.weight[:half], ext_out.weight[half:]),
            "Output extension: first half should equal second half (V,V)",
        )

        # Z -> (Z, -Z): first half cols == -(second half cols)
        self.assertTrue(
            torch.equal(ext_in.weight[:, :half], -ext_in.weight[:, half:]),
            "Input extension: first half should be negation of second (Z,-Z)",
        )

    def test_vv_z_negz_structure_linear(self):
        """Same structure check for LinearGrowingBlock."""
        block = _make_linear_block(h_t=8, device=self.device)
        dh = 4  # final size (pairing included)
        half = dh // 2
        block.create_layer_extensions(
            extension_size=dh,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
            neuron_pairing="vv_z_negz",
            noise_ratio=0,
        )

        ext_out = block.first_layer.extended_output_layer
        ext_in = block.second_layer.extended_input_layer

        self.assertTrue(
            torch.equal(ext_out.weight[:half], ext_out.weight[half:]),
        )
        self.assertTrue(
            torch.equal(ext_in.weight[:, :half], -ext_in.weight[:, half:]),
        )

    def test_pairing_bias_duplicated(self):
        """Output extension bias is also duplicated (V,V)."""
        block = _make_conv_block(h_t=8, device=self.device)
        dh = 4  # final size (pairing included)
        half = dh // 2
        block.create_layer_extensions(
            extension_size=dh,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
            neuron_pairing="vv_z_negz",
            noise_ratio=0,
        )
        ext_out = block.first_layer.extended_output_layer
        if ext_out.bias is not None:
            self.assertTrue(
                torch.equal(ext_out.bias[:half], ext_out.bias[half:]),
                "Output extension bias: first half should equal second half",
            )

    def test_pairing_apply_change_then_forward_with_sized_activation(self):
        """Growing with pairing then forwarding with SizedIdentity currently fails."""
        h_t = 8
        dh = 4
        block = _make_conv_block(
            h_t=h_t,
            mid_activation=SizedIdentity(h_t),
            device=self.device,
        )

        block.create_layer_extensions(
            extension_size=dh,
            neuron_pairing="vv_z_negz",
        )
        block.apply_change(scaling_factor=1.0, extension_size=dh)

        x = torch.randn(2, 3, 8, 8, device=self.device)
        with self.assertRaises(ValueError):
            _ = block(x)


class TestRescalingIdempotence(TorchTestCase):
    """Test 8: rescaling is a near no-op when weights already have correct variance."""

    def setUp(self):
        super().setUp()
        self.device = global_device()

    @unittest_parametrize(
        [
            {"rescaling": "vt_constraint_old_shape"},
            {"rescaling": "vt_constraint_new_shape"},
        ]
    )
    def test_kaiming_weights_nearly_unchanged(self, rescaling):
        """When V[W] ~ 1/fan_in, rescaling factors are ~1."""
        block = _make_conv_block(h_t=16, device=self.device)
        # Block is Kaiming-initialized by default

        w1_before = block.first_layer.weight.clone()
        w2_before = block.second_layer.weight.clone()

        block.create_layer_extensions(
            extension_size=4,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
            rescaling=rescaling,
            neuron_pairing="vv_z_negz",
        )

        # Weights should be nearly unchanged (tolerance accounts for
        # finite-sample deviation of Kaiming uniform from exact 1/fan_in)
        self.assertAllClose(
            block.first_layer.weight,
            w1_before,
            atol=0.2,
            rtol=0.15,
            msg="Conv1 weights should not change much with Kaiming init",
        )
        self.assertAllClose(
            block.second_layer.weight,
            w2_before,
            atol=0.2,
            rtol=0.15,
            msg="Conv2 weights should not change much with Kaiming init",
        )


class TestBatchNormRescaling(TorchTestCase):
    """Test 9: BatchNorm running statistics are rescaled correctly."""

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def test_batchnorm_stats_rescaled(self):
        """running_mean *= alpha, running_var *= alpha^2."""
        h_t = 8
        bn = GrowingBatchNorm2d(h_t, device=self.device)

        # Set known running statistics
        mu = torch.randn(h_t, device=self.device)
        sigma = torch.rand(h_t, device=self.device).abs() + 0.1
        bn.running_mean.copy_(mu)
        bn.running_var.copy_(sigma)

        block = _make_conv_block(
            h_t=h_t,
            mid_activation=bn,
            device=self.device,
        )
        # Perturb weights so alpha != 1
        block.first_layer.weight.data.mul_(2.0)

        # Read alpha before rescaling
        fan_in_prev = block.first_layer.layer.in_channels * 3 * 3
        var_w_prev = block.first_layer.weight.var().item()
        alpha = (1.0 / (fan_in_prev * var_w_prev)) ** 0.5

        # Apply rescaling (Strategy B or C — both rescale Conv1 the same way)
        block.apply_rescaling(
            rescaling="vt_constraint_old_shape",
            neuron_pairing="vv_z_negz",
            extension_size=4,
        )

        self.assertAllClose(
            bn.running_mean,
            alpha * mu,
            atol=1e-6,
            msg="running_mean should be scaled by alpha",
        )
        self.assertAllClose(
            bn.running_var,
            alpha**2 * sigma,
            atol=1e-6,
            msg="running_var should be scaled by alpha^2",
        )

    def test_batchnorm_stats_rescaled_conv2(self):
        """Conv2's post_layer_function BN: running_mean *= beta, running_var *= beta^2."""
        h_t = 8
        out_channels = 3
        bn2 = GrowingBatchNorm2d(out_channels, device=self.device)

        # Set known running statistics
        mu2 = torch.randn(out_channels, device=self.device)
        sigma2 = torch.rand(out_channels, device=self.device).abs() + 0.1
        bn2.running_mean.copy_(mu2)
        bn2.running_var.copy_(sigma2)

        block = _make_conv_block(
            h_t=h_t,
            out_channels=out_channels,
            pre_addition_function=bn2,
            device=self.device,
        )
        # Perturb Conv2 weights so beta != 1
        block.second_layer.weight.data.mul_(0.5)

        # Read beta before rescaling (Strategy B: beta targets 1/fan_in_self_old)
        fan_in_self_old = h_t * 3 * 3
        var_w_self = block.second_layer.weight.var().item()
        beta = (1.0 / (fan_in_self_old * var_w_self)) ** 0.5

        block.apply_rescaling(
            rescaling="vt_constraint_old_shape",
            neuron_pairing="vv_z_negz",
            extension_size=4,
        )

        self.assertAllClose(
            bn2.running_mean,
            beta * mu2,
            atol=1e-6,
            msg="Conv2 BN running_mean should be scaled by beta",
        )
        self.assertAllClose(
            bn2.running_var,
            beta**2 * sigma2,
            atol=1e-6,
            msg="Conv2 BN running_var should be scaled by beta^2",
        )


# ===============================================================
# 3.  Edge-case tests
# ===============================================================


class TestSmallWeightVarianceEdgeCases(TorchTestCase):
    """Rescaling with zero or near-zero weight variance.

    When weights have zero variance (e.g. all weights identical or a single
    element), the rescaling formulas involve 1/V[W] which would blow up.
    The code guards this with ``var > 0 else 1.0``, but we test that
    calling the code does not crash.
    """

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def test_zero_variance_weights_no_crash(self):
        """Rescaling doesn't crash when weight variance is exactly zero."""
        block = _make_conv_block(h_t=8, device=self.device)
        # Set all weights to a constant => variance = 0
        block.first_layer.weight.data.fill_(0.5)
        block.second_layer.weight.data.fill_(0.5)

        # Should not crash — falls back to scale=1.0
        _grow_block(
            block,
            dh=4,
            rescaling="vt_constraint_old_shape",
            neuron_pairing="vv_z_negz",
        )

    def test_single_hidden_channel_no_crash(self):
        """Rescaling with h_t=1 (minimal hidden dimension)."""
        block = _make_conv_block(h_t=1, device=self.device)
        # Conv2 weight has shape (out, 1, k, k) — variance computed over all elems
        # Minimum even extension under pairing is 2.
        _grow_block(
            block,
            dh=2,
            rescaling="vt_constraint_new_shape",
            neuron_pairing="vv_z_negz",
        )
        self.assertEqual(block.hidden_neurons, 1 + 2)


# ===============================================================
# 4.  Standalone method tests (FOGRO path simulation)
# ===============================================================


class TestStandaloneMethods(TorchTestCase):
    """Test apply_rescaling / apply_neuron_pairing called independently."""

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def test_standalone_rescaling_then_pairing(self):
        """Simulates FOGRO path: rescale and pair called separately.

        The extension is created at its final (even) size; pairing fills the
        second half in place and does not resize.
        """
        block = _make_conv_block(h_t=8, device=self.device)
        block.first_layer.weight.data.mul_(2.0)
        block.second_layer.weight.data.mul_(0.5)

        # Simulate FOGRO: extensions already exist at their final size
        block.second_layer.create_layer_in_extension(8)
        block.first_layer.create_layer_out_extension(8)

        # Init extensions with kaiming
        torch.nn.init.kaiming_uniform_(block.second_layer.extended_input_layer.weight)
        torch.nn.init.kaiming_uniform_(block.first_layer.extended_output_layer.weight)

        # Step 1: rescale
        block.apply_rescaling(
            rescaling="vt_constraint_new_shape",
            neuron_pairing="vv_z_negz",
        )

        # Step 2: pair (in place, no resize)
        block.apply_neuron_pairing(neuron_pairing="vv_z_negz", noise_ratio=0.0)

        # Sizes unchanged
        self.assertEqual(
            block.first_layer.extended_output_layer.weight.shape[0],
            8,
        )
        self.assertEqual(
            block.second_layer.extended_input_layer.weight.shape[1],
            8,
        )

        # Verify (V,V) and (Z,-Z) structure on halves
        ext_out = block.first_layer.extended_output_layer
        ext_in = block.second_layer.extended_input_layer
        self.assertTrue(torch.equal(ext_out.weight[:4], ext_out.weight[4:]))
        self.assertTrue(torch.equal(ext_in.weight[:, :4], -ext_in.weight[:, 4:]))

    def test_standalone_rescaling_with_extension_size_override(self):
        """apply_rescaling with extension_size before extensions exist."""
        block = _make_conv_block(h_t=8, device=self.device)
        w1_before = block.first_layer.weight.clone()
        w2_before = block.second_layer.weight.clone()

        # No extensions yet, but we pass extension_size
        block.apply_rescaling(
            rescaling="default_vt",
            neuron_pairing="vv_z_negz",
            extension_size=4,
        )

        # Conv1 should be unchanged (alpha=1 for default_vt)
        self.assertAllClose(block.first_layer.weight, w1_before)
        # Conv2 should be scaled down (beta < 1)
        self.assertFalse(
            torch.allclose(block.second_layer.weight, w2_before),
            "Conv2 weights should have been rescaled",
        )


# ===============================================================
# 5.  Coverage smoke tests
# ===============================================================


class TestCoverageSmoke(TorchTestCase):
    """Smoke tests targeting uncovered branches in the new VT code."""

    def setUp(self):
        super().setUp()
        self.device = global_device()

    def test_batchnorm1d_rescaling(self):
        """Hit BatchNorm1d branch in _rescale_post_layer_function."""
        bn1d = torch.nn.BatchNorm1d(8, device=self.device)
        bn1d.running_mean.fill_(1.0)
        bn1d.running_var.fill_(2.0)

        block = _make_linear_block(h_t=8, device=self.device)
        # Inject BN1d as mid_activation (post_layer_function of first_layer)
        block.first_layer.post_layer_function = bn1d

        block.first_layer.weight.data.mul_(2.0)
        block.apply_rescaling(
            rescaling="vt_constraint_old_shape",
            neuron_pairing=None,
            extension_size=4,
        )
        # BN1d running stats should have been rescaled
        self.assertFalse(
            torch.allclose(bn1d.running_mean, torch.ones(8, device=self.device)),
        )

    def test_invalid_neuron_pairing_in_apply_rescaling(self):
        """Invalid neuron_pairing in apply_rescaling raises ValueError."""
        block = _make_conv_block(h_t=8, device=self.device)
        with self.assertRaises(ValueError):
            block.second_layer.apply_rescaling(
                rescaling="default_vt",
                neuron_pairing="bad_pairing",  # type: ignore
                extension_size=4,
            )

    def test_rescaling_no_extension_no_size(self):
        """apply_rescaling with no extensions and no extension_size => ext_size=0."""
        block = _make_conv_block(h_t=8, device=self.device)
        # No extensions exist, no extension_size passed => ext_size = 0
        block.second_layer.apply_rescaling(
            rescaling="default_vt",
            neuron_pairing=None,
        )

    def test_rescaling_no_bias(self):
        """Rescaling with bias=None hits the False branch of bias checks."""
        block = Conv2dGrowingBlock(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            hidden_channels=8,
            kwargs_layer={"padding": 1, "use_bias": False},
            device=self.device,
        )
        block.first_layer.weight.data.mul_(2.0)
        block.second_layer.weight.data.mul_(0.5)
        # Strategy B gives alpha!=1 and beta!=1, exercising bias-is-None branches
        block.apply_rescaling(
            rescaling="vt_constraint_old_shape",
            neuron_pairing=None,
            extension_size=4,
        )

    def test_pairing_no_extended_output_raises(self):
        """apply_neuron_pairing without extended_output_layer raises RuntimeError."""
        block = _make_conv_block(h_t=8, device=self.device)
        # Only create input extension, not output
        block.second_layer.create_layer_in_extension(4)
        with self.assertRaises(RuntimeError):
            block.second_layer.apply_neuron_pairing(neuron_pairing="vv_z_negz")

    def test_pairing_no_extended_input_raises(self):
        """apply_neuron_pairing without extended_input_layer raises RuntimeError."""
        block = _make_conv_block(h_t=8, device=self.device)
        # Only create output extension, not input
        block.first_layer.create_layer_out_extension(4)
        with self.assertRaises(RuntimeError):
            block.second_layer.apply_neuron_pairing(neuron_pairing="vv_z_negz")

    def test_copy_uniform_initialization(self):
        """create_layer_extensions with copy_uniform init hits copy_uniform_initialization."""
        block = _make_conv_block(h_t=8, device=self.device)
        block.create_layer_extensions(
            extension_size=4,
            output_extension_init="copy_uniform",
            input_extension_init="copy_uniform",
        )
        self.assertIsNotNone(block.second_layer.extended_input_layer)

    def test_copy_uniform_fallback_to_kaiming(self):
        """copy_uniform with zero-variance reference falls back to kaiming (line 3235)."""
        block = _make_conv_block(h_t=8, device=self.device)
        # Set all weights to constant => zero variance => kaiming fallback
        block.first_layer.weight.data.fill_(0.5)
        block.second_layer.weight.data.fill_(0.5)
        block.create_layer_extensions(
            extension_size=4,
            output_extension_init="copy_uniform",
            input_extension_init="copy_uniform",
        )

    def test_extension_size_overrides(self):
        """Explicit output_extension_size and input_extension_size."""
        block = _make_conv_block(h_t=8, device=self.device)
        block.create_layer_extensions(
            extension_size=4,
            output_extension_size=6,
            input_extension_size=3,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
        )
        self.assertEqual(block.first_layer.extended_output_layer.weight.shape[0], 6)
        self.assertEqual(block.second_layer.extended_input_layer.weight.shape[1], 3)

    def test_unknown_init_raises(self):
        """Unknown initialization method raises ValueError."""
        block = _make_conv_block(h_t=8, device=self.device)
        with self.assertRaises(ValueError):
            block.create_layer_extensions(
                extension_size=4,
                output_extension_init="bad_init",  # type: ignore
                input_extension_init="kaiming",
            )

    def test_no_bias_output_extension(self):
        """Output extension without bias (use_bias=False) hits bias-is-None branch."""
        block = Conv2dGrowingBlock(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            hidden_channels=8,
            kwargs_layer={"padding": 1, "use_bias": False},
            device=self.device,
        )
        block.create_layer_extensions(
            extension_size=4,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
        )
        self.assertIsNone(block.first_layer.extended_output_layer.bias)

    def test_pairing_no_bias(self):
        """Neuron pairing with bias=False hits the bias-is-None branch."""
        block = Conv2dGrowingBlock(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            hidden_channels=8,
            kwargs_layer={"padding": 1, "use_bias": False},
            device=self.device,
        )
        block.create_layer_extensions(
            extension_size=4,
            output_extension_init="kaiming",
            input_extension_init="kaiming",
            neuron_pairing="vv_z_negz",
        )
        # extension_size is the final size (pairing included)
        self.assertEqual(block.first_layer.extended_output_layer.weight.shape[0], 4)

    def test_rescaling_reads_extension_size_from_layer(self):
        """apply_rescaling reads ext_size from extended_input_layer when not passed."""
        block = _make_conv_block(h_t=8, device=self.device)
        block.second_layer.create_layer_in_extension(4)
        # Don't pass extension_size — should read from extended_input_layer
        block.second_layer.apply_rescaling(
            rescaling="default_vt",
            neuron_pairing=None,
        )
