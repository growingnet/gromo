"""
Tests for GrowingLoRALinear and GrowingLoRAConv2d (module-level).

Tests cover:
- GrowingLoRALinear: init, forward, merge, utilities, FOGRO pipeline
- GrowingLoRAConv2d: init, forward, merge, FOGRO pipeline
- LinearGrowingModule interoperability
- lora_dropout: forward behaviour and extra_repr
- extended_forward with rank > 0
"""

from unittest import TestCase

import torch
import torch.nn as nn

from gromo.containers.growing_block import LinearGrowingBlock
from gromo.growra.container import (
    get_growing_lora_model,
    get_lora_modules,
)
from gromo.growra.module import GrowingLoRAConv2d, GrowingLoRALinear
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device, set_device


def _linear(*args, **kwargs):
    return nn.Linear(*args, device=global_device(), **kwargs)


def _conv2d(*args, **kwargs):
    return nn.Conv2d(*args, device=global_device(), **kwargs)


def _randn(*args, **kwargs):
    return torch.randn(*args, device=global_device(), **kwargs)


def _ones(*args, **kwargs):
    return torch.ones(*args, device=global_device(), **kwargs)


class TestGrowingLoRALinearInit(TestCase):
    """Tests for GrowingLoRALinear initialization."""

    def test_basic_init(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4, alpha=2.0)
        self.assertIsInstance(lora, LinearGrowingBlock)
        self.assertEqual(lora.in_features, 10)
        self.assertEqual(lora.out_features, 20)
        self.assertEqual(lora.rank, 4)
        self.assertEqual(lora.alpha, 2.0)

    def test_init_rank_zero(self):
        linear = _linear(5, 3)
        lora = GrowingLoRALinear(linear, rank=0)
        self.assertEqual(lora.rank, 0)
        self.assertAlmostEqual(lora.scaling, 0.0)

    def test_original_frozen(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        for p in lora.linear.parameters():
            self.assertFalse(p.requires_grad)

    def test_lora_params_trainable(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        params = lora.lora_parameters()
        self.assertTrue(len(params) > 0)
        for p in params:
            self.assertTrue(p.requires_grad)

    def test_scaling_property(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4, alpha=2.0)
        self.assertAlmostEqual(lora.scaling, 0.5)

    def test_scaling_property_larger(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4, alpha=8.0)
        self.assertAlmostEqual(lora.scaling, 2.0)

    def test_first_second_layer_properties(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        self.assertEqual(lora.first_layer.in_features, 10)
        self.assertEqual(lora.first_layer.out_features, 4)
        self.assertEqual(lora.second_layer.in_features, 4)
        self.assertEqual(lora.second_layer.out_features, 20)

    def test_weight_bias_properties(self):
        linear = _linear(10, 20, bias=True)
        lora = GrowingLoRALinear(linear, rank=2)
        self.assertIs(lora.weight, linear.weight)
        self.assertIs(lora.bias, linear.bias)

    def test_no_bias(self):
        linear = _linear(10, 20, bias=False)
        lora = GrowingLoRALinear(linear, rank=2)
        self.assertIsNone(lora.bias)

    def test_extra_repr(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4, alpha=2.0)
        r = lora.extra_repr()
        self.assertIn("in_features=10", r)
        self.assertIn("out_features=20", r)
        self.assertIn("rank=4", r)
        self.assertIn("alpha=2.0", r)

    def test_explicit_device_overrides_global_device_for_internal_layers(self):
        original_device = global_device()
        try:
            set_device(torch.device("meta"))
            linear = nn.Linear(10, 20, device=torch.device("cpu"))
            lora = GrowingLoRALinear(linear, rank=0, device=torch.device("cpu"))
        finally:
            set_device(original_device)

        self.assertEqual(lora.device, torch.device("cpu"))
        self.assertEqual(lora.first_layer.weight.device, torch.device("cpu"))
        self.assertEqual(lora.second_layer.weight.device, torch.device("cpu"))


class TestGrowingLoRALinearForward(TestCase):
    """Tests for GrowingLoRALinear forward pass."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        x = _randn(5, 10)
        out = lora(x)
        self.assertEqual(out.shape, (5, 20))

    def test_forward_rank_zero_equals_original(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=0)
        x = _randn(5, 10)
        out_lora = lora(x)
        out_orig = linear(x)
        self.assertTrue(torch.allclose(out_lora, out_orig))

    def test_forward_with_nonzero_weights(self):
        """With non-zero B weights, output differs from original."""
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(5, 10)
        out_lora = lora(x)
        out_orig = linear(x)
        self.assertFalse(torch.allclose(out_lora, out_orig))

    def test_forward_3d_input(self):
        """Test with sequence-like input (batch, seq, features)."""
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        x = _randn(3, 7, 10)
        out = lora(x)
        self.assertEqual(out.shape, (3, 7, 20))

    def test_forward_batched(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        x = _randn(5, 8, 10)
        y = lora(x)
        self.assertEqual(y.shape, (5, 8, 20))

    def test_gradient_flows_to_lora_only(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(5, 10)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        # LoRA params should have gradients
        for p in lora.lora_parameters():
            self.assertIsNotNone(p.grad)
        # Frozen params should not
        self.assertIsNone(lora.linear.weight.grad)


class TestGrowingLoRALinearMerge(TestCase):
    """Tests for GrowingLoRALinear merge_lora."""

    def setUp(self):
        torch.manual_seed(42)

    def test_merge_produces_linear(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        merged = lora.merge_lora()
        self.assertIsInstance(merged, nn.Linear)
        self.assertEqual(merged.in_features, 10)
        self.assertEqual(merged.out_features, 20)

    def test_merge_output_matches_forward(self):
        """Merged layer should produce same output as LoRA forward."""
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4, alpha=4.0)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(8, 10)
        with torch.no_grad():
            out_lora = lora(x)
            merged = lora.merge_lora()
            out_merged = merged(x)
        self.assertTrue(
            torch.allclose(out_lora, out_merged, atol=1e-5),
            f"Max diff: {(out_lora - out_merged).abs().max().item()}",
        )

    def test_merge_rank_zero(self):
        linear = _linear(10, 20)
        w_orig = linear.weight.data.clone()
        lora = GrowingLoRALinear(linear, rank=0)
        merged = lora.merge_lora()
        self.assertTrue(torch.allclose(merged.weight, w_orig))

    def test_merge_with_bias(self):
        linear = _linear(10, 20, bias=True)
        b_orig = linear.bias.data.clone()
        lora = GrowingLoRALinear(linear, rank=4)
        merged = lora.merge_lora()
        self.assertIsNotNone(merged.bias)
        self.assertTrue(torch.allclose(merged.bias, b_orig))

    def test_merge_without_bias(self):
        linear = _linear(10, 20, bias=False)
        lora = GrowingLoRALinear(linear, rank=4)
        merged = lora.merge_lora()
        self.assertIsNone(merged.bias)

    def test_merge_does_not_modify_original(self):
        linear = _linear(10, 20)
        original_weight = linear.weight.data.clone()
        lora = GrowingLoRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        _ = lora.merge_lora()
        self.assertTrue(torch.allclose(linear.weight.data, original_weight))


class TestGrowingLoRALinearUtilities(TestCase):
    """Tests for GrowingLoRALinear utility methods."""

    def test_lora_parameters(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        params = lora.lora_parameters()
        self.assertTrue(len(params) > 0)
        for p in params:
            self.assertTrue(p.requires_grad)

    def test_reset_lora(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        lora.reset_lora()
        self.assertTrue(
            torch.allclose(
                lora.second_layer.weight.data,
                torch.zeros_like(lora.second_layer.weight.data),
            )
        )


# ===================== _matches_target Tests =====================


class TestFOGROGrowthPipeline(TestCase):
    """Test the FOGRO growth pipeline on GrowingLoRALinear."""

    def setUp(self):
        self.in_features = 8
        self.out_features = 6
        self.batch_size = 16

    def _make_lora(self, rank=0):
        linear = _linear(self.in_features, self.out_features)
        return GrowingLoRALinear(linear, rank=rank, alpha=1.0)

    def test_init_computation(self):
        lora = self._make_lora(rank=0)
        lora.init_computation()
        self.assertTrue(lora.first_layer.store_input)
        self.assertTrue(lora.second_layer.store_pre_activity)

    def test_reset_computation(self):
        lora = self._make_lora(rank=0)
        lora.init_computation()
        lora.reset_computation()
        self.assertFalse(lora.first_layer.store_input)
        self.assertFalse(lora.second_layer.store_pre_activity)

    def test_full_fogro_loop_rank_zero(self):
        """Test full FOGRO loop starting from rank 0."""
        lora = self._make_lora(rank=0)

        lora.init_computation()

        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()

        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=4,
            compute_delta=False,
            use_covariance=False,
            use_projection=False,
            alpha_zero=True,
            omega_zero=False,
            ignore_singular_values=True,
        )

        self.assertIsNotNone(lora.first_layer.extended_output_layer)
        self.assertIsNotNone(lora.second_layer.extended_input_layer)
        self.assertIsNotNone(lora.eigenvalues_extension)

        old_rank = lora.rank
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        self.assertEqual(lora.rank, old_rank + 2)

        # Verify forward still works
        x2 = _randn(3, self.in_features)
        y = lora(x2)
        self.assertEqual(y.shape, (3, self.out_features))

        lora.reset_computation()

    def test_full_fogro_loop_nonzero_rank(self):
        """Test full FOGRO loop from nonzero rank with full fogro method."""
        lora = self._make_lora(rank=2)

        lora.init_computation()

        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()
        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=3,
            compute_delta=True,
            use_covariance=True,
            use_projection=True,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
        )

        self.assertIsNotNone(lora.eigenvalues_extension)

        old_rank = lora.rank
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        self.assertEqual(lora.rank, old_rank + 2)

        lora.reset_computation()

    def test_fogro_tiny_method(self):
        """Test FOGRO growth with TINY method."""
        lora = self._make_lora(rank=2)

        lora.init_computation()
        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()
        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=3,
            compute_delta=False,
            use_covariance=True,
            use_projection=True,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=False,
        )
        self.assertIsNotNone(lora.eigenvalues_extension)
        lora.reset_computation()

    def test_first_order_improvement(self):
        """Test that first_order_improvement is accessible after compute."""
        lora = self._make_lora(rank=0)
        lora.init_computation()
        x = _randn(self.batch_size, self.in_features)
        lora.zero_grad()
        output = lora(x)
        loss = (output**2).sum() / 2
        loss.backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=4,
            compute_delta=False,
            use_covariance=False,
            use_projection=False,
            alpha_zero=True,
            omega_zero=False,
            ignore_singular_values=True,
        )
        improvement = lora.first_order_improvement
        self.assertIsInstance(improvement, torch.Tensor)


class TestGrowingLoRALinearWithLinearGrowingModule(TestCase):
    """Tests that GrowingLoRALinear works with LinearGrowingModule as input."""

    def setUp(self):
        torch.manual_seed(42)

    def test_init_from_linear_growing_module(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowingLoRALinear(lgm, rank=4, alpha=2.0)
        self.assertEqual(lora.in_features, 10)
        self.assertEqual(lora.out_features, 20)
        self.assertEqual(lora.rank, 4)

    def test_forward_shape(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowingLoRALinear(lgm, rank=4)
        x = _randn(5, 10)
        out = lora(x)
        self.assertEqual(out.shape, (5, 20))

    def test_forward_rank_zero(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowingLoRALinear(lgm, rank=0)
        x = _randn(5, 10)
        out_lora = lora(x)
        out_orig = lgm(x)
        self.assertTrue(torch.allclose(out_lora, out_orig))

    def test_frozen(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        GrowingLoRALinear(lgm, rank=4)
        for p in lgm.parameters():
            self.assertFalse(p.requires_grad)

    def test_merge_lora(self):
        lgm = LinearGrowingModule(
            in_features=10, out_features=20, name="test", device=global_device()
        )
        lora = GrowingLoRALinear(lgm, rank=4)
        merged = lora.merge_lora()
        self.assertIsInstance(merged, nn.Linear)
        self.assertEqual(merged.weight.shape, (20, 10))

    def test_apply_growing_lora_on_lgm_model(self):
        """get_growing_lora_model should detect LinearGrowingModule layers."""
        lgm1 = LinearGrowingModule(
            in_features=10, out_features=20, name="l1", device=global_device()
        )
        lgm2 = LinearGrowingModule(
            in_features=20, out_features=5, name="l2", device=global_device()
        )
        model = nn.Sequential(lgm1, nn.ReLU(), lgm2)
        lora_model = get_growing_lora_model(model)
        lora_mods = get_lora_modules(lora_model)
        self.assertEqual(len(lora_mods), 2)
        for m in lora_mods:
            self.assertIsInstance(m, GrowingLoRALinear)

    def test_forward_after_apply_on_lgm_model(self):
        lgm1 = LinearGrowingModule(
            in_features=10, out_features=20, name="l1", device=global_device()
        )
        lgm2 = LinearGrowingModule(
            in_features=20, out_features=5, name="l2", device=global_device()
        )
        model = nn.Sequential(lgm1, nn.ReLU(), lgm2)
        lora_model = get_growing_lora_model(model)
        x = _randn(3, 10)
        out = lora_model(x)
        self.assertEqual(out.shape, (3, 5))


# --------- Tests for GrowingLoRAConv2d ---------


class TestGrowingLoRAConv2dInit(TestCase):
    """Tests for GrowingLoRAConv2d initialization."""

    def test_basic_init(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4, alpha=2.0)
        self.assertEqual(lora.in_channels, 3)
        self.assertEqual(lora.out_channels, 16)
        self.assertEqual(lora.rank, 4)
        self.assertEqual(lora.alpha, 2.0)

    def test_init_rank_zero(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=0)
        self.assertEqual(lora.rank, 0)
        self.assertAlmostEqual(lora.scaling, 0.0)

    def test_original_frozen(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4)
        for p in lora.conv.parameters():
            self.assertFalse(p.requires_grad)

    def test_lora_params_trainable(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4)
        params = lora.lora_parameters()
        self.assertTrue(len(params) > 0)
        for p in params:
            self.assertTrue(p.requires_grad)

    def test_scaling(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4, alpha=8.0)
        self.assertAlmostEqual(lora.scaling, 2.0)

    def test_weight_bias_properties(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        lora = GrowingLoRAConv2d(conv, rank=2)
        self.assertIs(lora.weight, conv.weight)
        self.assertIs(lora.bias, conv.bias)

    def test_extra_repr(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4, alpha=2.0)
        r = lora.extra_repr()
        self.assertIn("in_channels=3", r)
        self.assertIn("out_channels=16", r)
        self.assertIn("rank=4", r)


class TestGrowingLoRAConv2dForward(TestCase):
    """Tests for GrowingLoRAConv2d forward pass."""

    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4)
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_forward_rank_zero_equals_original(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=0)
        x = _randn(2, 3, 8, 8)
        out_lora = lora(x)
        out_orig = conv(x)
        self.assertTrue(torch.allclose(out_lora, out_orig))

    def test_forward_with_stride(self):
        conv = _conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4)
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        self.assertEqual(out.shape, (2, 16, 4, 4))

    def test_gradient_flows_to_lora_only(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4)
        nn.init.normal_(lora.second_layer.weight)
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        for p in lora.lora_parameters():
            self.assertIsNotNone(p.grad)
        self.assertIsNone(lora.conv.weight.grad)


class TestGrowingLoRAConv2dMerge(TestCase):
    """Tests for GrowingLoRAConv2d merge."""

    def setUp(self):
        torch.manual_seed(42)

    def test_merge_rank_zero(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=0)
        merged = lora.merge_lora()
        self.assertIsInstance(merged, nn.Conv2d)
        self.assertTrue(torch.allclose(merged.weight, conv.weight))

    def test_merge_preserves_shape(self):
        conv = _conv2d(3, 16, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=4)
        merged = lora.merge_lora()
        self.assertEqual(merged.weight.shape, conv.weight.shape)


class TestGrowingLoRAConv2dFOGRO(TestCase):
    """Basic FOGRO pipeline test for GrowingLoRAConv2d."""

    def setUp(self):
        torch.manual_seed(42)

    def test_fogro_pipeline_rank_zero(self):
        conv = _conv2d(3, 8, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=0, alpha=1.0)

        lora.init_computation()

        x = _randn(4, 3, 8, 8)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        lora.update_computation()

        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=False,
            use_projection=False,
            alpha_zero=True,
            omega_zero=False,
            ignore_singular_values=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        lora.reset_computation()

        self.assertGreater(lora.rank, 0)

    def test_forward_after_growth(self):
        conv = _conv2d(3, 8, kernel_size=3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=0, alpha=1.0)

        lora.init_computation()
        x = _randn(4, 3, 8, 8)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        lora.update_computation()
        lora.compute_optimal_updates(
            maximum_added_neurons=2,
            compute_delta=False,
            use_covariance=False,
            use_projection=False,
            alpha_zero=True,
            omega_zero=False,
            ignore_singular_values=True,
        )
        lora.sub_select_optimal_added_parameters(keep_neurons=2)
        lora.apply_change(scaling_factor=1.0, extension_size=2)
        lora.reset_computation()

        x2 = _randn(2, 3, 8, 8)
        out = lora(x2)
        self.assertEqual(out.shape, (2, 8, 8, 8))


# ===================== Dropout Tests =====================


class TestLoRADropoutLinear(TestCase):
    """Tests for lora_dropout in GrowingLoRALinear."""

    def setUp(self):
        torch.manual_seed(0)

    def test_dropout_stored(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=4, lora_dropout=0.3)
        self.assertAlmostEqual(lora.lora_dropout.p, 0.3)

    def test_forward_train_mode_is_stochastic(self):
        """In train mode with dropout, two forwards should differ."""
        lora = GrowingLoRALinear(_linear(10, 20), rank=4, lora_dropout=0.9)
        lora.train()
        x = _ones(16, 10)
        out1 = lora(x)
        out2 = lora(x)
        self.assertFalse(torch.allclose(out1, out2))

    def test_forward_eval_mode_is_deterministic(self):
        """In eval mode, dropout is disabled — two forwards must be identical."""
        lora = GrowingLoRALinear(_linear(10, 20), rank=4, lora_dropout=0.9)
        lora.eval()
        x = _ones(16, 10)
        self.assertTrue(torch.allclose(lora(x), lora(x)))

    def test_extra_repr_shows_dropout(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=2, lora_dropout=0.25)
        self.assertIn("lora_dropout=0.25", lora.extra_repr())

    def test_extra_repr_no_dropout_by_default(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=2)
        self.assertNotIn("lora_dropout", lora.extra_repr())

    def test_extended_forward_with_nonzero_rank(self):
        """extended_forward with rank > 0 returns base + scaling * lora."""
        lora = GrowingLoRALinear(_linear(10, 20), rank=4)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))

    def test_forward_rank0_store_input(self):
        """rank=0 with store_input=True (after init_computation) uses lora path."""
        lora = GrowingLoRALinear(_linear(10, 20), rank=0, lora_dropout=0.5)
        lora.init_computation()
        x = _randn(4, 10)
        out = lora(x)
        self.assertEqual(out.shape, (4, 20))
        lora.reset_computation()

    def test_explicit_activation_skips_default(self):
        """Passing activation explicitly skips the `if activation is None` branch."""
        lora = GrowingLoRALinear(_linear(10, 20), rank=2, activation=nn.ReLU())
        x = _randn(3, 10)
        self.assertEqual(lora(x).shape, (3, 20))

    def test_extended_forward_rank_zero(self):
        """extended_forward with rank=0 and no growth returns early (line 155)."""
        lora = GrowingLoRALinear(_linear(10, 20), rank=0)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))


class TestLoRADropoutConv2d(TestCase):
    """Tests for lora_dropout in GrowingLoRAConv2d."""

    def setUp(self):
        torch.manual_seed(0)

    def test_dropout_stored(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, lora_dropout=0.3)
        self.assertAlmostEqual(lora.lora_dropout.p, 0.3)

    def test_forward_train_mode_is_stochastic(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, lora_dropout=0.9)
        lora.train()
        x = _ones(4, 3, 8, 8)
        self.assertFalse(torch.allclose(lora(x), lora(x)))

    def test_forward_eval_mode_is_deterministic(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, lora_dropout=0.9)
        lora.eval()
        x = _ones(4, 3, 8, 8)
        self.assertTrue(torch.allclose(lora(x), lora(x)))

    def test_extra_repr_shows_dropout(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3), rank=2, lora_dropout=0.5)
        self.assertIn("lora_dropout=0.5", lora.extra_repr())

    def test_extra_repr_no_dropout_by_default(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3), rank=2)
        self.assertNotIn("lora_dropout", lora.extra_repr())

    def test_extended_forward_with_nonzero_rank(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_forward_rank0_store_input(self):
        """rank=0 with store_input=True (after init_computation) uses lora path."""
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=0, lora_dropout=0.5)
        lora.init_computation()
        x = _randn(2, 3, 8, 8)
        out = lora(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))
        lora.reset_computation()

    def test_explicit_activation_skips_default(self):
        """Passing activation explicitly skips the `if activation is None` branch."""
        lora = GrowingLoRAConv2d(
            _conv2d(3, 8, 3, padding=1), rank=2, activation=nn.ReLU()
        )
        x = _randn(2, 3, 8, 8)
        self.assertEqual(lora(x).shape, (2, 8, 8, 8))

    def test_extended_forward_rank_zero(self):
        """extended_forward with rank=0 and no growth returns early (line 348)."""
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=0)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_wrap_conv2d_growing_module(self):
        """GrowingLoRAConv2d accepts a Conv2dGrowingModule (line 245)."""
        cgm = Conv2dGrowingModule(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            name="test_conv",
            device=global_device(),
        )
        lora = GrowingLoRAConv2d(cgm, rank=2)
        self.assertEqual(lora.in_channels, 3)
        self.assertEqual(lora.out_channels, 8)

    def test_merge_lora_conv2d_growing_module(self):
        """merge_lora on a Conv2dGrowingModule-backed LoRA (line 363)."""
        cgm = Conv2dGrowingModule(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            name="test_merge",
            device=global_device(),
        )
        lora = GrowingLoRAConv2d(cgm, rank=2, alpha=1.0)
        merged = lora.merge_lora()
        self.assertIsInstance(merged, nn.Conv2d)
        self.assertEqual(merged.weight.shape[0], 8)

    def test_merge_lora_no_bias(self):
        """merge_lora on conv without bias (line 389->391 False branch)."""
        conv = _conv2d(3, 8, 3, padding=1, bias=False)
        lora = GrowingLoRAConv2d(conv, rank=2, alpha=1.0)
        merged = lora.merge_lora()
        self.assertIsNone(merged.bias)


class TestDoRALinear(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_init_matches_base_layer(self):
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=0, use_dora=True)
        x = _randn(4, 10)
        self.assertTrue(torch.allclose(lora(x), linear(x), atol=1e-6))

    def test_explicit_device_skips_default_device_inference(self):
        device = global_device()
        linear = _linear(10, 20)
        lora = GrowingLoRALinear(linear, rank=0, device=device, activation=nn.ReLU())
        self.assertEqual(
            lora.first_layer.weight.device, torch.device(linear.weight.device)
        )

    def test_extended_forward_dora_rank_zero(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=0, use_dora=True)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))

    def test_extended_forward_dora_nonzero_rank(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=4, use_dora=True)
        x = _randn(3, 10)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (3, 20))

    def test_magnitude_is_trainable(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=4, use_dora=True)
        self.assertIsNotNone(lora.magnitude)
        assert lora.magnitude is not None
        self.assertTrue(lora.magnitude.requires_grad)
        self.assertTrue(any(p is lora.magnitude for p in lora.lora_parameters()))

    def test_extra_repr_mentions_dora(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=4, use_dora=True)
        self.assertIn("use_dora=True", lora.extra_repr())

    def test_merge_matches_forward(self):
        lora = GrowingLoRALinear(_linear(10, 20), rank=4, alpha=4.0, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        with torch.no_grad():
            assert lora.magnitude is not None
            lora.magnitude.mul_(1.1)
        x = _randn(6, 10)
        with torch.no_grad():
            out_lora = lora(x)
            out_merged = lora.merge_lora()(x)
        self.assertTrue(torch.allclose(out_lora, out_merged, atol=1e-5))


class TestDoRAConv2d(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_init_matches_base_layer(self):
        conv = _conv2d(3, 8, 3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=0, use_dora=True)
        x = _randn(2, 3, 8, 8)
        self.assertTrue(torch.allclose(lora(x), conv(x), atol=1e-6))

    def test_explicit_device_skips_default_device_inference(self):
        device = global_device()
        conv = _conv2d(3, 8, 3, padding=1)
        lora = GrowingLoRAConv2d(conv, rank=0, device=device, activation=nn.ReLU())
        self.assertEqual(lora.first_layer.weight.device, torch.device(conv.weight.device))

    def test_extended_forward_dora_rank_zero(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=0, use_dora=True)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_extended_forward_dora_nonzero_rank(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        x = _randn(2, 3, 8, 8)
        out = lora.extended_forward(x)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    def test_magnitude_is_trainable(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        self.assertIsNotNone(lora.magnitude)
        assert lora.magnitude is not None
        self.assertTrue(any(p is lora.magnitude for p in lora.lora_parameters()))

    def test_extra_repr_mentions_dora(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        self.assertIn("use_dora=True", lora.extra_repr())

    def test_merge_matches_forward(self):
        lora = GrowingLoRAConv2d(_conv2d(3, 8, 3, padding=1), rank=4, use_dora=True)
        nn.init.normal_(lora.first_layer.weight)
        nn.init.normal_(lora.second_layer.weight)
        with torch.no_grad():
            assert lora.magnitude is not None
            lora.magnitude.mul_(0.9)
        x = _randn(2, 3, 8, 8)
        with torch.no_grad():
            out_lora = lora(x)
            out_merged = lora.merge_lora()(x)
        self.assertTrue(torch.allclose(out_lora, out_merged, atol=1e-5))
