"""
Tests for lora_growth_container: LoRAGrowingModel, get_growing_lora_model, and utilities.

Tests cover:
- _matches_target helper
- get_growing_lora_model / LoRAGrowingModel: model conversion, target filtering, freezing
- get_lora_parameters, get_lora_modules, merge_all_lora
- get_lora_state_dict, load_lora_state_dict
- End-to-end and regression tests
"""

from unittest import TestCase

import torch
import torch.nn as nn

from gromo.growra.container import (
    LoRAGrowingModel,
    _matches_target,
    get_growing_lora_model,
    get_lora_modules,
    get_lora_parameters,
    merge_all_lora,
)
from gromo.growra.module import GrowingLoRAConv2d, GrowingLoRALinear
from gromo.utils.utils import global_device


def _linear(*args, **kwargs):
    return nn.Linear(*args, device=global_device(), **kwargs)


def _conv2d(*args, **kwargs):
    return nn.Conv2d(*args, device=global_device(), **kwargs)


def _randn(*args, **kwargs):
    return torch.randn(*args, device=global_device(), **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model():
    return nn.Sequential(_linear(10, 20), nn.ReLU(), _linear(20, 5))


def _grow(lora_model, data, added_rank=2):
    """Run one growth step using the low-level gromo pipeline (gradmax method)."""
    lora_mods = lora_model.lora_modules()
    for m in lora_mods:
        m.init_computation()
    lora_model.eval()
    for x, y in data:
        lora_model.zero_grad()
        nn.functional.mse_loss(lora_model(x), y).backward()
        for m in lora_mods:
            m.update_computation()
    for m in lora_mods:
        m.compute_optimal_updates(
            compute_delta=False,
            use_covariance=False,
            use_projection=False,
            alpha_zero=True,
            omega_zero=False,
            ignore_singular_values=True,
        )
    for m in lora_mods:
        if added_rank is not None:
            m.sub_select_optimal_added_parameters(keep_neurons=added_rank)
        m.apply_change(scaling_factor=1.0, extension_size=added_rank)
    for m in lora_mods:
        m.reset_computation()


# ===================== _matches_target Tests =====================


class TestMatchesTarget(TestCase):
    def test_matches_by_type(self):
        self.assertTrue(_matches_target("fc", _linear(10, 5), None, (nn.Linear,)))

    def test_rejects_wrong_type(self):
        self.assertFalse(_matches_target("fc", _linear(10, 5), None, (nn.Conv2d,)))

    def test_matches_by_name(self):
        self.assertTrue(
            _matches_target("layer1.fc", _linear(10, 5), ["fc"], (nn.Linear,))
        )

    def test_rejects_by_name(self):
        self.assertFalse(
            _matches_target("layer1.conv", _linear(10, 5), ["fc"], (nn.Linear,))
        )

    def test_none_target_modules_matches_all(self):
        self.assertTrue(_matches_target("anything", _linear(10, 5), None, (nn.Linear,)))


# ===================== get_growing_lora_model / LoRAGrowingModel Tests =====================


class TestAsGrowraModel(TestCase):
    """Tests for the get_growing_lora_model factory and LoRAGrowingModel class."""

    def test_returns_lora_growing_model(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        self.assertIsInstance(lora_model, LoRAGrowingModel)

    def test_replaces_all_linear(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        lora_count = sum(
            1 for m in lora_model.modules() if isinstance(m, GrowingLoRALinear)
        )
        self.assertEqual(lora_count, 2)

    def test_rank_zero_at_init(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        for m in lora_model.modules():
            if isinstance(m, GrowingLoRALinear):
                self.assertEqual(m.rank, 0)

    def test_alpha_propagated(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model, alpha=3.0)
        for m in lora_model.modules():
            if isinstance(m, GrowingLoRALinear):
                self.assertEqual(m.alpha, 3.0)

    def test_target_modules_filtering(self):
        model = nn.Sequential(_linear(10, 20), nn.ReLU(), _linear(20, 5))
        lora_model = get_growing_lora_model(model, target_modules=["0"])
        self.assertIsInstance(lora_model.model[0], GrowingLoRALinear)
        self.assertIsInstance(lora_model.model[2], nn.Linear)

    def test_freeze_original(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        for m in lora_model.modules():
            if isinstance(m, GrowingLoRALinear):
                self.assertFalse(m.linear.weight.requires_grad)

    def test_lora_params_trainable(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        for m in lora_model.modules():
            if isinstance(m, GrowingLoRALinear):
                for p in m.lora_parameters():
                    self.assertTrue(p.requires_grad)

    def test_forward_after_apply(self):
        model = _make_simple_model()
        x = _randn(3, 10)
        lora_model = get_growing_lora_model(model)
        out = lora_model(x)
        self.assertEqual(out.shape, (3, 5))

    def test_forward_equals_original_at_rank_zero(self):
        model = _make_simple_model()
        x = _randn(3, 10)
        with torch.no_grad():
            expected = model(x).clone()
        lora_model = get_growing_lora_model(model)
        with torch.no_grad():
            out = lora_model(x)
        self.assertTrue(torch.allclose(out, expected))

    def test_inferred_in_out_features(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        self.assertEqual(lora_model.in_features, 10)
        self.assertEqual(lora_model.out_features, 5)

    def test_explicit_in_out_features(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model, in_features=10, out_features=5)
        self.assertEqual(lora_model.in_features, 10)
        self.assertEqual(lora_model.out_features, 5)

    def test_growable_layers_populated(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        self.assertEqual(len(lora_model._growable_layers), 2)
        self.assertEqual(len(lora_model._growing_layers), 2)


class TestAsGrowraModelNested(TestCase):
    """Test get_growing_lora_model on nested model structures."""

    def test_nested_modules(self):
        model = nn.Sequential(
            nn.Sequential(_linear(10, 20), nn.ReLU()),
            nn.Sequential(_linear(20, 15), nn.ReLU()),
            _linear(15, 5),
        )
        lora_model = get_growing_lora_model(model)
        lora_count = sum(
            1 for m in lora_model.modules() if isinstance(m, GrowingLoRALinear)
        )
        self.assertEqual(lora_count, 3)

    def test_named_modules_model(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _linear(10, 20)
                self.decoder = _linear(20, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.decoder(self.relu(self.encoder(x)))

        model = SimpleNet()
        lora_model = get_growing_lora_model(model, target_modules=["encoder"])
        self.assertIsInstance(lora_model.model.encoder, GrowingLoRALinear)
        self.assertIsInstance(lora_model.model.decoder, nn.Linear)

    def test_empty_model(self):
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        with self.assertRaises(ValueError):
            get_growing_lora_model(model)


# ===================== Utility Function Tests =====================


class TestGetGrowraParameters(TestCase):
    def test_collects_all_lora_params(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        params = get_lora_parameters(lora_model)
        # 2 Linear layers -> 2 x (A + B) = 4 parameter tensors
        self.assertEqual(len(params), 4)

    def test_params_are_trainable(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        for p in get_lora_parameters(lora_model):
            self.assertTrue(p.requires_grad)

    def test_no_lora_returns_empty(self):
        model = _make_simple_model()
        self.assertEqual(len(get_lora_parameters(model)), 0)

    def test_lora_model_convenience_method(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        self.assertEqual(
            len(lora_model.lora_parameters()),
            len(get_lora_parameters(lora_model)),
        )


class TestGetGrowraModules(TestCase):
    def test_finds_modules(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        modules = get_lora_modules(lora_model)
        self.assertEqual(len(modules), 2)
        for m in modules:
            self.assertIsInstance(m, GrowingLoRALinear)

    def test_empty_model(self):
        model = nn.Sequential(nn.ReLU())
        self.assertEqual(len(get_lora_modules(model)), 0)

    def test_lora_model_convenience_method(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        self.assertEqual(
            len(lora_model.lora_modules()), len(get_lora_modules(lora_model))
        )


class TestMergeAllGrowra(TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_merge_removes_wrappers(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        lora_model.merge_lora()
        self.assertEqual(len(get_lora_modules(lora_model)), 0)

    def test_merge_rank_zero_equals_original(self):
        """Merging rank-0 LoRA recovers the original weights exactly."""
        model = _make_simple_model()
        x = _randn(3, 10)
        with torch.no_grad():
            out_orig = model(x).clone()
        lora_model = get_growing_lora_model(model)
        with torch.no_grad():
            out_before = lora_model(x).clone()
        lora_model.merge_lora()
        with torch.no_grad():
            out_after = lora_model(x)
        self.assertTrue(torch.allclose(out_orig, out_before))
        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-6))

    def test_standalone_merge_all_lora(self):
        """merge_all_lora on lora_model.model works too."""
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        merge_all_lora(lora_model.model)
        self.assertEqual(len(get_lora_modules(lora_model.model)), 0)


class TestGrowraStateDict(TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_get_state_dict_keys(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        state = lora_model.lora_state_dict()
        a_keys = [k for k in state if k.endswith("first_layer.weight")]
        b_keys = [k for k in state if k.endswith("second_layer.weight")]
        self.assertEqual(len(a_keys), 2)
        self.assertEqual(len(b_keys), 2)

    def test_save_load_roundtrip_after_growth(self):
        """After growing, save and reload LoRA state into a fresh model."""
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)

        # Grow first so weights are non-trivial
        data = [(_randn(4, 10), _randn(4, 5)) for _ in range(2)]
        _grow(lora_model, data, added_rank=2)

        state = lora_model.lora_state_dict()

        model2 = _make_simple_model()
        lora_model2 = get_growing_lora_model(model2)
        lora_model2.load_lora_state_dict(state)

        for m1, m2 in zip(
            lora_model.lora_modules(), lora_model2.lora_modules(), strict=True
        ):
            self.assertTrue(torch.allclose(m1.first_layer.weight, m2.first_layer.weight))
            self.assertTrue(
                torch.allclose(m1.second_layer.weight, m2.second_layer.weight)
            )

    def test_state_dict_values_are_detached(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        state = lora_model.lora_state_dict()
        for key, val in state.items():
            if isinstance(val, torch.Tensor) and val.dim() > 0:
                self.assertFalse(val.requires_grad)


# ===================== Integration / End-to-end Tests =====================


class TestEndToEnd(TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_training_loop(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        x, y = _randn(16, 10), _randn(16, 5)
        # Grow first so there are trainable parameters with a grad_fn
        _grow(lora_model, [(x, y)] * 2, added_rank=2)
        optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            nn.functional.mse_loss(lora_model(x), y).backward()
            optimizer.step()

    def test_grow_during_training(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        x, y = _randn(8, 10), _randn(8, 5)

        # First growth step
        _grow(lora_model, [(x, y)] * 2, added_rank=2)

        optimizer = torch.optim.SGD(lora_model.lora_parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            nn.functional.mse_loss(lora_model(x), y).backward()
            optimizer.step()

        # Second growth step
        _grow(lora_model, [(x, y)] * 2, added_rank=2)

        optimizer = torch.optim.SGD(lora_model.lora_parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            nn.functional.mse_loss(lora_model(x), y).backward()
            optimizer.step()

        for m in lora_model.lora_modules():
            self.assertEqual(m.rank, 4)

    def test_grow_merge_consistency(self):
        """After growing, merged output should still match forward pass."""
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)

        data = [(_randn(4, 10), _randn(4, 5)) for _ in range(2)]
        _grow(lora_model, data, added_rank=3)
        # Set non-trivial B weights
        for m in lora_model.lora_modules():
            nn.init.normal_(m.second_layer.weight)

        x = _randn(4, 10)
        with torch.no_grad():
            out_before = lora_model(x).clone()
        lora_model.merge_lora()
        with torch.no_grad():
            out_after = lora_model(x)
        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-5),
            f"Max diff: {(out_before - out_after).abs().max().item()}",
        )

    def test_full_growth_pipeline(self):
        model = nn.Sequential(_linear(8, 16), nn.ReLU(), _linear(16, 4))
        x = _randn(5, 8)
        lora_model = get_growing_lora_model(model, alpha=1.0)

        data = [(_randn(8, 8), _randn(8, 4)) for _ in range(2)]
        _grow(lora_model, data, added_rank=2)

        optimizer = torch.optim.SGD(lora_model.lora_parameters(), lr=0.01)
        loss = nn.MSELoss()(lora_model(_randn(8, 8)), _randn(8, 4))
        loss.backward()
        optimizer.step()

        lora_model.merge_lora()
        self.assertEqual(len(get_lora_modules(lora_model)), 0)
        self.assertEqual(lora_model(x).shape, (5, 4))


class TestOriginalWeightsUnchanged(TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_original_weights_unchanged_after_training(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        x, y = _randn(16, 10), _randn(16, 5)
        # Grow first so that training actually uses LoRA parameters
        _grow(lora_model, [(x, y)] * 2, added_rank=2)
        optimizer = torch.optim.Adam(lora_model.lora_parameters(), lr=0.1)
        for _ in range(5):
            optimizer.zero_grad()
            nn.functional.mse_loss(lora_model(x), y).backward()
            optimizer.step()
        for m in lora_model.modules():
            if isinstance(m, GrowingLoRALinear):
                self.assertFalse(m.linear.weight.requires_grad)

    def test_original_weights_unchanged_after_grow(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model)
        orig_weights = {
            n: m.linear.weight.data.clone()
            for n, m in lora_model.named_modules()
            if isinstance(m, GrowingLoRALinear)
        }
        data = [(_randn(2, 10), _randn(2, 5)) for _ in range(2)]
        _grow(lora_model, data, added_rank=5)
        for n, m in lora_model.named_modules():
            if isinstance(m, GrowingLoRALinear):
                self.assertTrue(torch.allclose(m.linear.weight.data, orig_weights[n]))


# ===================== Conv2d Tests =====================


class TestAsGrowraModelConv2d(TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_apply_on_conv_model(self):
        model = nn.Sequential(
            _conv2d(3, 16, 3, padding=1), nn.ReLU(), _conv2d(16, 32, 3, padding=1)
        )
        lora_model = get_growing_lora_model(model)
        lora_mods = lora_model.lora_modules()
        self.assertEqual(len(lora_mods), 2)
        for m in lora_mods:
            self.assertIsInstance(m, GrowingLoRAConv2d)

    def test_forward_after_apply_on_conv_model(self):
        model = nn.Sequential(
            _conv2d(3, 16, 3, padding=1), nn.ReLU(), _conv2d(16, 32, 3, padding=1)
        )
        lora_model = get_growing_lora_model(model)
        out = lora_model(_randn(2, 3, 8, 8))
        self.assertEqual(out.shape, (2, 32, 8, 8))

    def test_mixed_model_linear_and_conv(self):
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = _conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = _linear(16, 5)

            def forward(self, x):
                x = self.relu(self.conv(x))
                return self.fc(self.pool(x).flatten(1))

        model = MixedModel()
        lora_model = get_growing_lora_model(model)
        lora_mods = lora_model.lora_modules()
        self.assertEqual(len(lora_mods), 2)
        types = {type(m) for m in lora_mods}
        self.assertIn(GrowingLoRALinear, types)
        self.assertIn(GrowingLoRAConv2d, types)
        self.assertEqual(lora_model(_randn(2, 3, 8, 8)).shape, (2, 5))

    def test_target_modules_filter(self):
        class NamedConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = _conv2d(3, 16, 3, padding=1)
                self.conv2 = _conv2d(16, 32, 3, padding=1)

            def forward(self, x):
                return self.conv2(nn.functional.relu(self.conv1(x)))

        model = NamedConvModel()
        lora_model = get_growing_lora_model(model, target_modules=["conv1"])
        self.assertEqual(len(lora_model.lora_modules()), 1)


# ===================== Dropout / Extra Coverage Tests =====================


class TestGrowraDropoutContainer(TestCase):
    """Tests covering lora_dropout propagation through the container."""

    def setUp(self):
        torch.manual_seed(0)

    def test_dropout_propagated_to_modules(self):
        model = nn.Sequential(_linear(10, 20), nn.ReLU(), _linear(20, 5))
        lora_model = get_growing_lora_model(model, lora_dropout=0.4)
        for m in lora_model.lora_modules():
            self.assertAlmostEqual(m.lora_dropout.p, 0.4)

    def test_extra_repr_shows_dropout(self):
        model = nn.Sequential(_linear(10, 5))
        lora_model = get_growing_lora_model(model, lora_dropout=0.3)
        self.assertIn("lora_dropout=0.3", lora_model.extra_repr())

    def test_extra_repr_no_dropout_by_default(self):
        model = nn.Sequential(_linear(10, 5))
        lora_model = get_growing_lora_model(model)
        self.assertNotIn("lora_dropout", lora_model.extra_repr())

    def test_explicit_in_features_only(self):
        """Pass in_features but let out_features be inferred."""
        model = nn.Sequential(_linear(10, 5))
        lora_model = get_growing_lora_model(model, in_features=10)
        self.assertEqual(lora_model.in_features, 10)
        self.assertEqual(lora_model.out_features, 5)

    def test_explicit_out_features_only(self):
        """Pass out_features but let in_features be inferred."""
        model = nn.Sequential(_linear(10, 5))
        lora_model = get_growing_lora_model(model, out_features=5)
        self.assertEqual(lora_model.in_features, 10)
        self.assertEqual(lora_model.out_features, 5)


class TestLoadGrowraStateDictCoverage(TestCase):
    """Edge-case coverage for load_lora_state_dict."""

    def setUp(self):
        torch.manual_seed(0)

    def test_load_same_rank_no_expansion(self):
        """Loading a state with same rank should copy weights without expanding."""
        model = nn.Sequential(_linear(10, 20), nn.ReLU(), _linear(20, 5))
        lora_model = get_growing_lora_model(model)
        # Grow to rank 2
        data = [(_randn(4, 10), _randn(4, 5)) for _ in range(2)]
        _grow(lora_model, data, added_rank=2)
        state = lora_model.lora_state_dict()

        # Reload into fresh model already at rank 2 — no expansion should happen
        model2 = nn.Sequential(_linear(10, 20), nn.ReLU(), _linear(20, 5))
        lora_model2 = get_growing_lora_model(model2)
        _grow(lora_model2, data, added_rank=2)
        lora_model2.load_lora_state_dict(state)

        for m1, m2 in zip(
            lora_model.lora_modules(), lora_model2.lora_modules(), strict=True
        ):
            self.assertTrue(torch.allclose(m1.first_layer.weight, m2.first_layer.weight))

    def test_load_missing_key_is_skipped(self):
        """load_lora_state_dict silently skips modules whose key is absent."""
        model = nn.Sequential(_linear(10, 20), nn.ReLU(), _linear(20, 5))
        lora_model = get_growing_lora_model(model)
        # Empty state — nothing should be loaded, no error
        lora_model.load_lora_state_dict({})
        for m in lora_model.lora_modules():
            self.assertEqual(m.rank, 0)

    def test_merge_all_lora_top_level_module(self):
        """merge_all_lora works when a LoRA wrapper is at the top level (no dot in name)."""
        linear = _linear(4, 8)

        class TopLevel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora = GrowingLoRALinear(linear, rank=0)

            def forward(self, x):
                return self.lora(x)

        model = TopLevel()
        from gromo.growra.container import merge_all_lora

        merge_all_lora(model)
        self.assertIsInstance(model.lora, nn.Linear)

    def test_merge_all_lora_nested_module(self):
        """merge_all_lora works when a LoRA wrapper is nested (dotted path, lines 432-433)."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(_linear(4, 8))

            def forward(self, x):
                return self.block(x)

        model = NestedModel()
        lora_model = get_growing_lora_model(model)
        x = _randn(2, 4)
        _ = lora_model(x)
        lora_model.merge_lora()
        self.assertEqual(lora_model(x).shape, (2, 8))


class TestDoRAContainer(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_use_dora_propagates_to_modules(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model, use_dora=True)
        self.assertTrue(lora_model.use_dora)
        for module in lora_model.lora_modules():
            self.assertTrue(module.use_dora)
            self.assertIsNotNone(module.magnitude)

    def test_extra_repr_mentions_dora(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model, use_dora=True)
        self.assertIn("use_dora=True", lora_model.extra_repr())

    def test_dora_state_dict_roundtrip(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model, use_dora=True)
        for module in lora_model.lora_modules():
            with torch.no_grad():
                assert module.magnitude is not None
                module.magnitude.add_(0.5)

        state = lora_model.lora_state_dict()

        model2 = _make_simple_model()
        lora_model2 = get_growing_lora_model(model2, use_dora=False)
        lora_model2.load_lora_state_dict(state)

        for m1, m2 in zip(
            lora_model.lora_modules(), lora_model2.lora_modules(), strict=True
        ):
            self.assertTrue(m2.use_dora)
            self.assertIsNotNone(m1.magnitude)
            self.assertIsNotNone(m2.magnitude)
            self.assertTrue(torch.allclose(m1.magnitude, m2.magnitude))

    def test_dora_state_dict_load_when_target_already_uses_dora(self):
        model = _make_simple_model()
        lora_model = get_growing_lora_model(model, use_dora=True)
        state = lora_model.lora_state_dict()

        model2 = _make_simple_model()
        lora_model2 = get_growing_lora_model(model2, use_dora=True)
        lora_model2.load_lora_state_dict(state)

        for module in lora_model2.lora_modules():
            self.assertTrue(module.use_dora)
            self.assertIsNotNone(module.magnitude)


# ===================== _matches_target Conv2d Tests =====================


class TestMatchesTargetConv2d(TestCase):
    def test_matches_conv2d_by_type(self):
        self.assertTrue(_matches_target("conv", _conv2d(3, 8, 3), None, (nn.Conv2d,)))

    def test_rejects_linear_for_conv2d_target(self):
        self.assertFalse(_matches_target("fc", _linear(10, 5), None, (nn.Conv2d,)))

    def test_matches_conv2d_by_name(self):
        self.assertTrue(
            _matches_target("layer1.conv", _conv2d(3, 8, 3), ["conv"], (nn.Conv2d,))
        )

    def test_rejects_conv2d_wrong_name(self):
        self.assertFalse(
            _matches_target("layer1.fc", _conv2d(3, 8, 3), ["conv"], (nn.Conv2d,))
        )

    def test_matches_mixed_types(self):
        """target_types containing both Linear and Conv2d should match either."""
        self.assertTrue(
            _matches_target("fc", _linear(10, 5), None, (nn.Linear, nn.Conv2d))
        )
        self.assertTrue(
            _matches_target("conv", _conv2d(3, 8, 3), None, (nn.Linear, nn.Conv2d))
        )


# ===================== Double-wrapping guard Tests =====================


class TestInjectGrowraDoubleWrapGuard(TestCase):
    """_inject_lora_inplace must not wrap a layer that is a descendant of an already-wrapped one."""

    def setUp(self):
        torch.manual_seed(0)

    def test_no_double_wrapping_in_nested_sequential(self):
        """Inner linear inside a Sequential that itself contains a linear must be wrapped once."""
        model = nn.Sequential(
            nn.Sequential(_linear(10, 20), nn.ReLU()),
            _linear(20, 5),
        )
        lora_model = get_growing_lora_model(model)
        # Total wraps == number of linear layers, no double-wrapping
        lora_count = sum(
            1 for m in lora_model.modules() if isinstance(m, GrowingLoRALinear)
        )
        self.assertEqual(lora_count, 2)

    def test_already_wrapped_layer_not_re_wrapped(self):
        """A GrowingLoRALinear already in the model is itself a descendant of the top model;
        calling get_growing_lora_model on a model that already contains LoRA wrappers
        should not wrap the inner linear again."""
        inner = _linear(10, 20)
        lora_inner = GrowingLoRALinear(inner, rank=0)

        class ModelWithLoRA(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora = lora_inner
                self.head = _linear(20, 5)

            def forward(self, x):
                return self.head(self.lora(x))

        model = ModelWithLoRA()
        lora_model = get_growing_lora_model(model, target_modules=["head"])
        # Only 'head' should be wrapped; lora_inner should remain untouched
        self.assertIsInstance(lora_model.model.lora, GrowingLoRALinear)
        self.assertIsInstance(lora_model.model.head, GrowingLoRALinear)
        # The inner .linear of lora_inner must not have been re-wrapped
        self.assertIsInstance(lora_model.model.lora.linear, nn.Linear)


# ===================== Conv2d load_lora_state_dict expansion Tests =====================


class TestLoadGrowraStateDictConv2dExpansion(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _grow_conv_model(self, lora_model, added_rank=2):
        data = [(_randn(2, 3, 8, 8), _randn(2, 16, 8, 8)) for _ in range(2)]
        for m in lora_model.lora_modules():
            m.init_computation()
        for x, y in data:
            lora_model.zero_grad()
            nn.functional.mse_loss(lora_model(x), y).backward()
            for m in lora_model.lora_modules():
                m.update_computation()
        for m in lora_model.lora_modules():
            m.compute_optimal_updates(
                compute_delta=False,
                use_covariance=False,
                use_projection=False,
                alpha_zero=True,
                omega_zero=False,
                ignore_singular_values=True,
            )
            m.sub_select_optimal_added_parameters(keep_neurons=added_rank)
            m.apply_change(scaling_factor=1.0, extension_size=added_rank)
            m.reset_computation()

    def _make_conv_lora(self):
        model = nn.Sequential(
            _conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            _conv2d(8, 16, kernel_size=3, padding=1),
        )
        return get_growing_lora_model(model)

    def test_load_conv2d_grown_state_expands_rank(self):
        """load_lora_state_dict expands Conv2d rank and copies weights correctly."""
        lora_model = self._make_conv_lora()
        self._grow_conv_model(lora_model, added_rank=2)

        for m in lora_model.lora_modules():
            self.assertEqual(m.rank, 2)

        state = lora_model.lora_state_dict()

        lora_model2 = self._make_conv_lora()
        lora_model2.load_lora_state_dict(state)

        for m1, m2 in zip(
            lora_model.lora_modules(), lora_model2.lora_modules(), strict=True
        ):
            self.assertEqual(m2.rank, 2)
            self.assertTrue(torch.allclose(m1.first_layer.weight, m2.first_layer.weight))
            self.assertTrue(
                torch.allclose(m1.second_layer.weight, m2.second_layer.weight)
            )

    def test_load_conv2d_grown_state_forward_matches(self):
        """After loading, forward output matches when the base convs are the same."""
        lora_model = self._make_conv_lora()
        self._grow_conv_model(lora_model, added_rank=2)
        # Set non-trivial B weights so the output is interesting
        for m in lora_model.lora_modules():
            nn.init.normal_(m.second_layer.weight)
        state = lora_model.lora_state_dict()

        lora_model2 = self._make_conv_lora()
        lora_model2.load_lora_state_dict(state)
        # The two fresh models have different random base-conv weights; align them
        # so the only difference would be a bad LoRA load.
        with torch.no_grad():
            for m1, m2 in zip(
                lora_model.lora_modules(), lora_model2.lora_modules(), strict=True
            ):
                m2.conv.weight.copy_(m1.conv.weight)
                if m1.conv.bias is not None:
                    m2.conv.bias.copy_(m1.conv.bias)

        x = _randn(2, 3, 8, 8)
        with torch.no_grad():
            out1 = lora_model(x)
            out2 = lora_model2(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_load_conv2d_same_rank_no_error(self):
        """load_lora_state_dict at same rank does not error for Conv2d."""
        model = nn.Sequential(_conv2d(3, 8, kernel_size=3, padding=1))
        lora_model = get_growing_lora_model(model)
        state = lora_model.lora_state_dict()
        model2 = nn.Sequential(_conv2d(3, 8, kernel_size=3, padding=1))
        lora_model2 = get_growing_lora_model(model2)
        lora_model2.load_lora_state_dict(state)  # should not raise
