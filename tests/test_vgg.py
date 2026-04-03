import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import gromo.containers.vgg as vgg_module
from gromo.containers.vgg import (
    VGG,
    NormKwargs,
    _reduce_growing_conv_widths,
    base_batch_norm_kwargs,
    init_full_vgg_structure,
)
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.growing_normalisation import (
    GrowingBatchNorm2d,
    GrowingGroupNorm,
    GrowingLayerNorm,
)
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


try:
    from torchvision.models import vgg11, vgg11_bn

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class TestVGG(TorchTestCase):
    """Test the configurable VGG initializer and growth behavior."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    @staticmethod
    def _stage_conv_widths(model: VGG) -> list[tuple[int, ...]]:
        return [
            tuple(conv.out_features for conv in stage_block.growing_modules)
            for stage_block in model.stage_blocks
        ]

    @staticmethod
    def _conv_allow_growing_flags(model: VGG) -> list[bool]:
        return [bool(module._allow_growing) for module in model._feature_growing_modules]

    @staticmethod
    def _growable_layer_targets(model: VGG) -> list[tuple[str, int, int | None]]:
        return [
            (type(layer).__name__, int(layer.in_neurons), layer.target_in_neurons)
            for layer in model._growable_layers
        ]

    @staticmethod
    def _first_conv(model: VGG) -> Conv2dGrowingModule:
        return model._feature_growing_modules[0]

    @staticmethod
    def _classifier_linears(model: VGG) -> list[LinearGrowingModule]:
        return [
            module
            for module in model.classifier
            if isinstance(module, LinearGrowingModule)
        ]

    @staticmethod
    def _copy_torchvision_vgg_weights(reference: nn.Module, model: VGG) -> None:
        reference_convs = [
            module for module in reference.features if isinstance(module, nn.Conv2d)
        ]
        model_convs = model._feature_growing_modules
        assert len(reference_convs) == len(model_convs)
        for ref_conv, model_conv in zip(reference_convs, model_convs, strict=True):
            with torch.no_grad():
                model_conv.weight.copy_(ref_conv.weight)
                assert model_conv.bias is not None
                assert ref_conv.bias is not None
                model_conv.bias.copy_(ref_conv.bias)

        reference_bns = [
            module for module in reference.features if isinstance(module, nn.BatchNorm2d)
        ]
        model_bns = [
            module
            for module in model.features.modules()
            if isinstance(module, GrowingBatchNorm2d)
        ]
        assert len(reference_bns) == len(model_bns)
        for ref_bn, model_bn in zip(reference_bns, model_bns, strict=True):
            with torch.no_grad():
                if model_bn.weight is not None:
                    assert ref_bn.weight is not None
                    model_bn.weight.copy_(ref_bn.weight)
                if model_bn.bias is not None:
                    assert ref_bn.bias is not None
                    model_bn.bias.copy_(ref_bn.bias)
                if model_bn.running_mean is not None:
                    assert ref_bn.running_mean is not None
                    model_bn.running_mean.copy_(ref_bn.running_mean)
                if model_bn.running_var is not None:
                    assert ref_bn.running_var is not None
                    model_bn.running_var.copy_(ref_bn.running_var)
                if model_bn.num_batches_tracked is not None:
                    assert ref_bn.num_batches_tracked is not None
                    model_bn.num_batches_tracked.copy_(ref_bn.num_batches_tracked)

        reference_linears = [
            module for module in reference.classifier if isinstance(module, nn.Linear)
        ]
        model_linears = TestVGG._classifier_linears(model)
        assert len(reference_linears) == len(model_linears)
        for ref_linear, model_linear in zip(
            reference_linears, model_linears, strict=True
        ):
            with torch.no_grad():
                model_linear.weight.copy_(ref_linear.weight)
                assert model_linear.bias is not None
                assert ref_linear.bias is not None
                model_linear.bias.copy_(ref_linear.bias)

    def test_init_full_vgg_structure_variations(self):
        """Test init_full_vgg_structure with valid and invalid configurations."""
        model = (
            init_full_vgg_structure()
        )  # By default grown VGG11, with resolution_factor 1/32
        self.assertIsInstance(model, VGG)
        self.assertEqual(
            self._stage_conv_widths(model),
            [(64,), (128,), (8, 256), (16, 512), (16, 512)],
        )
        self.assertEqual(
            self._conv_allow_growing_flags(model),
            [False, False, False, True, False, True, False, True],
        )

        model_single_block = init_full_vgg_structure(number_of_conv_per_stage=1)
        self.assertEqual(
            self._stage_conv_widths(model_single_block),
            [(64,), (128,), (256,), (512,), (512,)],
        )
        self.assertEqual(
            self._conv_allow_growing_flags(model_single_block),
            [False, False, False, False, False],
        )

        model_varied_blocks = init_full_vgg_structure(
            number_of_conv_per_stage=(1, 2, 3, 1, 2),
        )
        self.assertEqual(
            self._stage_conv_widths(model_varied_blocks),
            [(64,), (4, 128), (8, 8, 256), (512,), (16, 512)],
        )
        self.assertEqual(
            self._conv_allow_growing_flags(model_varied_blocks),
            [False, False, True, False, True, True, False, False, True],
        )

        model_with_higher_reduction_factor = init_full_vgg_structure(
            number_of_conv_per_stage=(1, 2, 3, 1, 2),
            reduction_factor=0.25,
        )
        self.assertEqual(
            self._stage_conv_widths(model_with_higher_reduction_factor),
            [(64,), (32, 128), (64, 64, 256), (512,), (128, 512)],
        )

        model_without_reduction = init_full_vgg_structure(
            number_of_conv_per_stage=(1, 2, 3, 1, 2),
            reduction_factor=None,
        )
        self.assertEqual(
            self._stage_conv_widths(model_without_reduction),
            [(64,), (128, 128), (256, 256, 256), (512,), (512, 512)],
        )

        model_hidden_int = init_full_vgg_structure(
            hidden_channels=(16, 32, 64, 128, 256),
            number_of_conv_per_stage=2,
        )
        self.assertEqual(
            self._stage_conv_widths(model_hidden_int),
            [(1, 16), (1, 32), (2, 64), (4, 128), (8, 256)],
        )

        model_hidden_mixed = init_full_vgg_structure(
            hidden_channels=(16, (32, 48), 64, 128, (192, 256)),
            number_of_conv_per_stage=2,
        )
        self.assertEqual(
            self._stage_conv_widths(model_hidden_mixed),
            [(1, 16), (1, 48), (2, 64), (4, 128), (6, 256)],
        )

        model_torch_size = init_full_vgg_structure(
            input_shape=torch.Size((1, 32, 32)),  # type: ignore[arg-type]
            nb_stages=3,
            number_of_conv_per_stage=1,
            hidden_channels=(8, 16, 32),
        )
        self.assertEqual(self._first_conv(model_torch_size).in_channels, 1)

        model_explicit_in = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            in_features=2,
            nb_stages=3,
            number_of_conv_per_stage=1,
            hidden_channels=(8, 16, 32),
        )
        self.assertEqual(self._first_conv(model_explicit_in).in_channels, 2)

        with self.assertRaises(AssertionError):
            init_full_vgg_structure(input_shape=torch.Size((3, 32)))  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            init_full_vgg_structure(number_of_conv_per_stage=(1, 2, 3))  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            init_full_vgg_structure(
                number_of_conv_per_stage=(1, "2", 3, 4, 5)  # type: ignore[arg-type]
            )

        with self.assertRaises(ValueError):
            init_full_vgg_structure(number_of_conv_per_stage=(1, 0, 3, 4, 5))

        with self.assertRaises(ValueError):
            init_full_vgg_structure(hidden_channels=(8, 16, 32, 64))

        with self.assertRaises(ValueError):
            init_full_vgg_structure(
                hidden_channels=(8, (16, 20, 24), 32, 64, 128),
                number_of_conv_per_stage=2,
            )

        with self.assertRaises(TypeError):
            init_full_vgg_structure(
                hidden_channels=(8, "invalid", 32, 64, 128),  # type: ignore[arg-type]
                number_of_conv_per_stage=2,
            )

        with self.assertRaises(TypeError):
            init_full_vgg_structure(
                hidden_channels=(8, (16, "20"), 32, 64, 128),  # type: ignore[arg-type]
                number_of_conv_per_stage=2,
            )

        with self.assertRaises(ValueError):
            init_full_vgg_structure(
                hidden_channels=(8, (16, 0), 32, 64, 128),
                number_of_conv_per_stage=2,
            )

        with self.assertRaises(TypeError):
            init_full_vgg_structure(number_of_fc_layers="3")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            init_full_vgg_structure(number_of_fc_layers=0)

        with self.assertRaises(TypeError):
            init_full_vgg_structure(fc_layer_width="64")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            init_full_vgg_structure(fc_layer_width=0)

        with self.assertRaises(TypeError):
            init_full_vgg_structure(reduction_factor="0.5")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            init_full_vgg_structure(reduction_factor=0)

        with self.assertRaises(ValueError):
            init_full_vgg_structure(reduction_factor=1.5)

        with patch.object(vgg_module, "min", return_value=0):
            with self.assertRaises(ValueError):
                init_full_vgg_structure(
                    nb_stages=1,
                    number_of_conv_per_stage=1,
                    hidden_channels=None,
                )

    @unittest.skipUnless(HAS_TORCHVISION, "torchvision is not installed")
    def test_torchvision_vgg11_parity(self):
        """Temporary parity check against torchvision VGG11."""
        model = init_full_vgg_structure(
            input_shape=(3, 224, 224),
            out_features=1000,
            normalization=None,
            number_of_conv_per_stage=(1, 1, 2, 2, 2),
            hidden_channels=(64, 128, 256, 512, 512),
            number_of_fc_layers=3,
            fc_layer_width=4096,
            reduction_factor=None,
        )
        reference = vgg11(weights=None)
        self._copy_torchvision_vgg_weights(reference, model)
        model.eval()
        reference.eval()

        self.assertEqual(model.final_spatial_shape, (7, 7))
        self.assertEqual(model.avgpool.output_size, (7, 7))
        self.assertEqual(
            self._stage_conv_widths(model),
            [(64,), (128,), (256, 256), (512, 512), (512, 512)],
        )
        self.assertEqual(
            [layer.in_features for layer in self._classifier_linears(model)],
            [512 * 7 * 7, 4096, 4096],
        )
        self.assertEqual(
            [layer.out_features for layer in self._classifier_linears(model)],
            [4096, 4096, 1000],
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in reference.parameters()),
        )

        x = torch.randn(2, 3, 224, 224)
        model_output = model(x)
        reference_output = reference(x)
        self.assertShapeEqual(model_output, reference_output.shape)
        self.assertAllClose(model_output, reference_output, atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(HAS_TORCHVISION, "torchvision is not installed")
    def test_torchvision_vgg11_bn_parity(self):
        """Temporary parity check against torchvision VGG11 with batch norm."""
        model = init_full_vgg_structure(
            input_shape=(3, 224, 224),
            out_features=1000,
            normalization="batch",
            number_of_conv_per_stage=(1, 1, 2, 2, 2),
            hidden_channels=(64, 128, 256, 512, 512),
            number_of_fc_layers=3,
            fc_layer_width=4096,
            reduction_factor=None,
        )
        reference = vgg11_bn(weights=None)
        self._copy_torchvision_vgg_weights(reference, model)
        model.eval()
        reference.eval()

        batch_norm_layers = [
            module for module in model.modules() if isinstance(module, GrowingBatchNorm2d)
        ]
        self.assertEqual(len(batch_norm_layers), 8)
        self.assertEqual(model.final_spatial_shape, (7, 7))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in reference.parameters()),
        )

        x = torch.randn(2, 3, 224, 224)
        model_output = model(x)
        reference_output = reference(x)
        self.assertShapeEqual(model_output, reference_output.shape)
        self.assertAllClose(model_output, reference_output, atol=1e-6, rtol=1e-6)

    def test_direct_constructor_and_helper_methods(self):
        """Test direct VGG constructor validation and private helper methods."""
        with self.assertRaises(ValueError):
            vgg_module._VGGStageBlock(
                stage_channels=(8,),
                target_stage_channels=(8, 8),
                in_channels=3,
                spatial_shape=(32, 32),
                build_post_conv_layers=lambda *_: nn.Identity(),
                growing_conv_type=Conv2dGrowingModule,
                device=torch.device("cpu"),
                name="bad_stage",
            )

        with self.assertRaises(ValueError):
            vgg_module._VGGStageBlock(
                stage_channels=(),
                target_stage_channels=(),
                in_channels=3,
                spatial_shape=(32, 32),
                build_post_conv_layers=lambda *_: nn.Identity(),
                growing_conv_type=Conv2dGrowingModule,
                device=torch.device("cpu"),
                name="empty_stage",
            )

        default_target_model = VGG(
            cfg=[8, "M", 16, "M"],
            target_cfg=None,
            in_features=3,
            normalization="batch",
            num_classes=5,
            number_of_fc_layers=1,
            fc_layer_width=16,
            input_spatial_shape=(32, 32),
        )
        self.assertEqual(len(default_target_model.features), 4)
        self.assertEqual(default_target_model.final_spatial_shape, (8, 8))

        no_trailing_pool_model = VGG(
            cfg=["M", 8, "M", 16],
            target_cfg=["M", 8, "M", 16],
            in_features=3,
            normalization="batch",
            num_classes=5,
            number_of_fc_layers=1,
            fc_layer_width=16,
            input_spatial_shape=(32, 32),
        )
        self.assertEqual(len(no_trailing_pool_model.stage_blocks), 2)
        self.assertEqual(no_trailing_pool_model.final_spatial_shape, (16, 16))

        no_spatial_shape_model = VGG(
            cfg=[8, "M", 16, "M"],
            target_cfg=None,
            in_features=3,
            normalization="batch",
            num_classes=5,
            number_of_fc_layers=1,
            fc_layer_width=16,
        )
        self.assertEqual(len(no_spatial_shape_model.features), 4)
        self.assertEqual(no_spatial_shape_model.final_spatial_shape, (7, 7))

        cfg = [8, "M", 16, 4, "M"]
        model = VGG(
            cfg=cfg,
            target_cfg=[8, "M", 16, 16, "M"],
            in_features=3,
            normalization="batch",
            normalization_kwargs={"eps": 1e-3},
            num_classes=5,
            number_of_fc_layers=1,
            fc_layer_width=16,
            input_spatial_shape=(32, 32),
        )

        self.assertEqual(VGG._validate_normalization(None), None)
        self.assertEqual(VGG._validate_normalization("batch"), "batch")
        self.assertEqual(VGG._validate_normalization("group"), "group")
        self.assertEqual(VGG._validate_normalization("layer"), "layer")
        with self.assertRaises(ValueError):
            VGG._validate_normalization("instance")  # type: ignore[arg-type]

        activation_copy = model._make_activation()
        self.assertIsInstance(activation_copy, nn.ReLU)
        self.assertIsNot(activation_copy, model.activation)

        normalization = model._build_growing_normalization(4, (32, 32))
        self.assertIsInstance(normalization, GrowingBatchNorm2d)
        self.assertEqual(normalization.eps, 1e-3)

        post_conv_layers = model._build_post_conv_layers(4, (32, 32))
        self.assertIsInstance(post_conv_layers, nn.Sequential)

        model._update_batch_norm_kwargs({"momentum": 0.25})
        self.assertEqual(model.batch_norm_kwargs["momentum"], 0.25)
        with self.assertRaises(ValueError):
            model._update_batch_norm_kwargs({"unknown": 1.0})  # type: ignore[arg-type]

        model.normalization = "group"
        model._update_group_norm_kwargs({"num_groups": 4})
        self.assertEqual(model.group_norm_kwargs["num_groups"], 4)
        with self.assertRaises(ValueError):
            model._update_group_norm_kwargs({"momentum": 0.1})  # type: ignore[arg-type]

        model.normalization = "layer"
        model._update_layer_norm_kwargs({"bias": False})
        self.assertFalse(model.layer_norm_kwargs["bias"])
        with self.assertRaises(ValueError):
            model._update_layer_norm_kwargs({"affine": False})  # type: ignore[arg-type]

        layer_norm = model._build_growing_normalization(4, (32, 32))
        self.assertIsInstance(layer_norm, GrowingLayerNorm)
        with self.assertRaises(ValueError):
            model._build_growing_normalization(4)

        model.normalization = None
        self.assertIsNone(model._build_growing_normalization(4, (32, 32)))
        post_conv_without_norm = model._build_post_conv_layers(4, (32, 32))
        self.assertIsInstance(post_conv_without_norm, nn.ReLU)

        model.normalization = "invalid"  # type: ignore[assignment]
        with self.assertRaises(AssertionError):
            model._build_growing_normalization(4, (32, 32))

        with self.assertRaises(ValueError):
            VGG(
                cfg=[8, "M"],
                target_cfg=[8, 16],
                normalization="batch",
                num_classes=3,
                number_of_fc_layers=1,
                fc_layer_width=8,
                input_spatial_shape=(32, 32),
            )

        with self.assertRaises(ValueError):
            VGG(
                cfg=[8, "M"],
                target_cfg=[8, "M"],
                normalization=None,
                normalization_kwargs={"eps": 1e-4},
                num_classes=3,
                number_of_fc_layers=1,
                fc_layer_width=8,
                input_spatial_shape=(32, 32),
            )

        with self.assertRaises(ValueError):
            VGG(
                cfg=[8, "M"],
                target_cfg=[8, "M"],
                normalization="batch",
                num_classes=3,
                number_of_fc_layers=0,
                input_spatial_shape=(32, 32),
            )

        with self.assertRaises(ValueError):
            VGG(
                cfg=[8, "M"],
                target_cfg=[8, "M"],
                normalization="batch",
                num_classes=3,
                number_of_fc_layers=1,
                fc_layer_width=0,
                input_spatial_shape=(32, 32),
            )

        with self.assertRaises(ValueError):
            VGG(
                cfg=[8, "M"],
                target_cfg=[8, "M"],
                normalization="batch",
                num_classes=3,
                number_of_fc_layers=1,
                fc_layer_width=8,
                initial_fc_layer_width=0,
                input_spatial_shape=(32, 32),
            )

        with self.assertRaises(ValueError):
            VGG(
                cfg=[8, "M"],
                target_cfg=[8, "M", 16],
                normalization="batch",
                num_classes=3,
                number_of_fc_layers=1,
                fc_layer_width=8,
                input_spatial_shape=(32, 32),
            )

        with self.assertRaises(ValueError):
            VGG(
                cfg=[8, "M"],
                target_cfg=[8, "M"],
                normalization="layer",
                num_classes=3,
                number_of_fc_layers=1,
                fc_layer_width=8,
            )

        self.assertEqual(
            _reduce_growing_conv_widths((16, 16, 16), None),
            (16, 16, 16),
        )
        self.assertEqual(
            _reduce_growing_conv_widths((16, 16, 16), 0.25),
            (4, 4, 16),
        )

        classifier_growth_model = VGG(
            cfg=[8, "M"],
            target_cfg=[8, "M"],
            in_features=3,
            normalization="batch",
            num_classes=5,
            number_of_fc_layers=2,
            fc_layer_width=16,
            initial_fc_layer_width=4,
            input_spatial_shape=(32, 32),
        )
        classifier_growable = [
            layer
            for layer in classifier_growth_model._growable_layers
            if isinstance(layer, LinearGrowingModule)
        ]
        self.assertEqual(len(classifier_growable), 1)
        self.assertEqual(classifier_growable[0].in_features, 4)
        self.assertEqual(classifier_growable[0].target_in_neurons, 16)

    def test_growable_targets_follow_vgg_target_widths(self):
        """Test that growable layers keep stage target widths."""
        model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            nb_stages=3,
            number_of_conv_per_stage=(1, 3, 3),
            hidden_channels=(32, 64, 128),
            number_of_fc_layers=1,
            reduction_factor=0.125,
        )

        self.assertEqual(
            self._stage_conv_widths(model),
            [(32,), (8, 8, 64), (16, 16, 128)],
        )
        self.assertEqual(model.final_spatial_shape, (4, 4))
        self.assertEqual(
            self._growable_layer_targets(model),
            [
                ("RestrictedConv2dGrowingModule", 8, 64),
                ("RestrictedConv2dGrowingModule", 8, 64),
                ("RestrictedConv2dGrowingModule", 16, 128),
                ("RestrictedConv2dGrowingModule", 16, 128),
            ],
        )

        mixed_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            nb_stages=2,
            number_of_conv_per_stage=(3, 3),
            hidden_channels=((24, 40, 56), (48, 80, 96)),
            number_of_fc_layers=1,
            reduction_factor=0.125,
        )
        self.assertEqual(
            self._growable_layer_targets(mixed_model),
            [
                ("RestrictedConv2dGrowingModule", 3, 24),
                ("RestrictedConv2dGrowingModule", 5, 40),
                ("RestrictedConv2dGrowingModule", 6, 48),
                ("RestrictedConv2dGrowingModule", 10, 80),
            ],
        )

        classifier_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            nb_stages=2,
            number_of_conv_per_stage=1,
            hidden_channels=(32, 64),
            number_of_fc_layers=3,
            fc_layer_width=32,
            reduction_factor=0.25,
        )
        classifier_linears = [
            module
            for module in classifier_model.classifier
            if isinstance(module, LinearGrowingModule)
        ]
        self.assertEqual(classifier_linears[0].out_features, 8)
        self.assertEqual(classifier_linears[1].in_features, 8)
        self.assertEqual(classifier_linears[1].target_in_neurons, 32)
        self.assertEqual(classifier_linears[2].in_features, 8)
        self.assertEqual(classifier_linears[2].target_in_neurons, 32)
        self.assertEqual(
            self._growable_layer_targets(classifier_model),
            [
                ("LinearGrowingModule", 8, 32),
                ("LinearGrowingModule", 8, 32),
            ],
        )

    def test_group_and_layer_normalization_forward(self):
        """Test group and layer normalization integration in VGG."""
        device = global_device()
        x = torch.randn(2, 3, 32, 32, device=device)

        group_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=7,
            nb_stages=3,
            number_of_conv_per_stage=(1, 2, 1),
            hidden_channels=(8, 16, 32),
            normalization="group",
            normalization_kwargs={"num_groups": 4, "eps": 1e-4, "affine": False},
            reduction_factor=None,
            device=device,
            init_weights=False,
        )
        group_norm_layers = [
            module
            for module in group_model.modules()
            if isinstance(module, GrowingGroupNorm)
        ]
        self.assertGreater(len(group_norm_layers), 0)
        self.assertEqual(group_norm_layers[0].num_groups, 4)
        self.assertEqual(group_norm_layers[0].eps, 1e-4)
        self.assertFalse(group_norm_layers[0].affine)
        self.assertShapeEqual(group_model(x), (2, 7))

        layer_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=7,
            nb_stages=3,
            number_of_conv_per_stage=(1, 2, 1),
            hidden_channels=(8, 16, 32),
            normalization="layer",
            normalization_kwargs={
                "eps": 1e-4,
                "elementwise_affine": False,
                "bias": False,
            },
            reduction_factor=None,
            device=device,
            init_weights=False,
        )
        layer_norm_layers = [
            module
            for module in layer_model.modules()
            if isinstance(module, GrowingLayerNorm)
        ]
        self.assertGreater(len(layer_norm_layers), 0)
        self.assertEqual(layer_norm_layers[0].normalized_shape, (8, 32, 32))
        self.assertEqual(layer_norm_layers[1].normalized_shape, (16, 16, 16))
        self.assertEqual(layer_norm_layers[2].normalized_shape, (16, 16, 16))
        self.assertFalse(layer_norm_layers[0].elementwise_affine)
        self.assertIsNone(layer_norm_layers[0].bias)
        self.assertShapeEqual(layer_model(x), (2, 7))

    def test_forward_backward(self):
        """Test forward and backward passes of the initialized VGG model."""
        model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=3,
            number_of_conv_per_stage=(1, 2, 1),
            hidden_channels=(16, 32, 64),
            normalization="batch",
            device=global_device(),
        )

        x = torch.randn(4, 3, 32, 32, device=global_device())
        output = model(x)
        self.assertShapeEqual(output, (4, 10), "Output shape should be (4, 10)")
        output.sum().backward()

    def test_normalization_configuration_forward(self):
        """Test forward passes with no normalization and custom BatchNorm kwargs."""
        device = global_device()
        x = torch.randn(2, 3, 32, 32, device=device)

        with self.subTest(normalization="invalid"):
            with self.assertRaises(ValueError):
                init_full_vgg_structure(
                    input_shape=(3, 32, 32),
                    out_features=7,
                    nb_stages=3,
                    number_of_conv_per_stage=(1, 2, 1),
                    hidden_channels=(8, 16, 32),
                    normalization="invalid",  # type: ignore[arg-type]
                    device=device,
                )

        with self.subTest(normalization="none"):
            model_no_norm = init_full_vgg_structure(
                input_shape=(3, 32, 32),
                out_features=7,
                nb_stages=3,
                number_of_conv_per_stage=(1, 2, 1),
                hidden_channels=(8, 16, 32),
                normalization=None,
                device=device,
            )
            norm_layers = [
                module
                for module in model_no_norm.modules()
                if isinstance(module, torch.nn.BatchNorm2d)
            ]
            self.assertEqual(norm_layers, [])
            self.assertShapeEqual(model_no_norm(x), (2, 7))

            with self.assertRaises(ValueError):
                init_full_vgg_structure(
                    input_shape=(3, 32, 32),
                    out_features=7,
                    nb_stages=3,
                    number_of_conv_per_stage=(1, 2, 1),
                    hidden_channels=(8, 16, 32),
                    normalization=None,
                    normalization_kwargs={"eps": 1e-3},
                    device=device,
                )

        with self.subTest(normalization="batch"):
            normalization_kwargs: NormKwargs = {
                "eps": 1e-3,
                "momentum": 0.25,
                "affine": False,
                "track_running_stats": False,
            }
            model_batch_norm = init_full_vgg_structure(
                input_shape=(3, 32, 32),
                out_features=7,
                nb_stages=3,
                number_of_conv_per_stage=(1, 2, 1),
                hidden_channels=(8, 16, 32),
                normalization="batch",
                normalization_kwargs=normalization_kwargs,
                init_weights=False,
                device=device,
            )
            norm_layers = [
                module
                for module in model_batch_norm.modules()
                if isinstance(module, torch.nn.BatchNorm2d)
            ]
            self.assertGreater(len(norm_layers), 0)
            for norm_layer in norm_layers:
                self.assertEqual(norm_layer.eps, normalization_kwargs["eps"])
                self.assertEqual(norm_layer.momentum, normalization_kwargs["momentum"])
                self.assertEqual(norm_layer.affine, normalization_kwargs["affine"])
                self.assertEqual(
                    norm_layer.track_running_stats,
                    normalization_kwargs["track_running_stats"],
                )
            self.assertShapeEqual(model_batch_norm(x), (2, 7))

        with self.subTest(normalization="defaults"):
            model_default_norm = init_full_vgg_structure(
                input_shape=(3, 32, 32),
                out_features=7,
                nb_stages=3,
                number_of_conv_per_stage=(1, 2, 1),
                hidden_channels=(8, 16, 32),
                normalization="batch",
                device=device,
            )
            first_norm = next(
                module
                for module in model_default_norm.modules()
                if isinstance(module, torch.nn.BatchNorm2d)
            )
            self.assertEqual(first_norm.eps, base_batch_norm_kwargs["eps"])
            self.assertEqual(first_norm.momentum, base_batch_norm_kwargs["momentum"])

    def test_extended_forward_with_layer_extensions(self):
        """Test extended_forward and verify active extensions change the output."""
        model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=3,
            number_of_conv_per_stage=(1, 3, 3),
            hidden_channels=(32, 64, 128),
            number_of_fc_layers=3,
            fc_layer_width=64,
            reduction_factor=0.125,
            normalization="batch",
            device=torch.device("cpu"),
        )
        model.eval()
        x = torch.randn(2, 3, 32, 32)

        output1 = model.extended_forward(x)
        self.assertShapeEqual(output1, (2, 10))

        first_growable_layer = model._growable_layers[0]
        assert isinstance(first_growable_layer, Conv2dGrowingModule)
        first_growable_layer.create_layer_extensions(
            extension_size=8,
            output_extension_init="copy_uniform",
            input_extension_init="copy_uniform",
        )

        self.assertIsNotNone(first_growable_layer.extended_input_layer)
        assert isinstance(first_growable_layer.previous_module, Conv2dGrowingModule)
        self.assertIsNotNone(first_growable_layer.previous_module.extended_output_layer)

        classifier_linears = [
            module
            for module in model.classifier
            if isinstance(module, LinearGrowingModule)
        ]
        classifier_linears[0].create_layer_extensions(
            extension_size=4,
            output_extension_size=4,
            input_extension_size=4 * 4 * 4,
            output_extension_init="copy_uniform",
            input_extension_init="copy_uniform",
        )
        classifier_linears[1].create_layer_extensions(
            extension_size=4,
            output_extension_init="copy_uniform",
            input_extension_init="copy_uniform",
        )

        output2 = model.extended_forward(x)
        self.assertShapeEqual(output2, (2, 10))
        self.assertAllClose(
            output1,
            output2,
            message="With scaling_factor=0, output should not change",
        )

        model.set_scaling_factor(1.0)
        output3 = model.extended_forward(x)
        self.assertShapeEqual(output3, (2, 10))
        max_diff = torch.abs(output1 - output3).max().item()
        self.assertGreater(
            max_diff,
            1e-4,
            f"Outputs should differ after enabling extensions, but max diff is {max_diff}",
        )

    def test_scheduling_and_growth_bookkeeping(self):
        """Test scheduling and growth bookkeeping without metadata refresh."""
        model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=3,
            number_of_conv_per_stage=(1, 3, 3),
            hidden_channels=(32, 64, 128),
            number_of_fc_layers=1,
            reduction_factor=0.125,
            device=torch.device("cpu"),
        )

        feature_layer = model._growable_layers[0]
        assert isinstance(feature_layer, Conv2dGrowingModule)
        classifier_layer = model._classifier_growing_modules[0]

        model.set_growing_layers(index=0)
        self.assertEqual(len(model._growing_layers), 1)
        self.assertIs(model._growing_layers[0], model._growable_layers[0])
        self.assertTrue(
            all(layer.target_in_neurons is not None for layer in model._growable_layers)
        )

        model.init_computation()
        self.assertTrue(feature_layer.store_input)
        self.assertTrue(feature_layer.store_pre_activity)
        self.assertFalse(classifier_layer.store_pre_activity)

        neurons_to_add = model.number_of_neurons_to_add(number_of_growth_steps=5)
        self.assertIsInstance(neurons_to_add, int)
        self.assertGreater(neurons_to_add, 0)

        model.set_growing_layers(scheduling_method="all")
        with self.assertRaises(RuntimeError):
            model.number_of_neurons_to_add(number_of_growth_steps=2)

        model.set_growing_layers(scheduling_method="sequential", index=0)
        model.currently_updated_layer_index = 0
        missing = model.missing_neurons()
        self.assertIsInstance(missing, int)
        self.assertGreater(missing, 0)

        plain_conv_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=2,
            number_of_conv_per_stage=(1, 3),
            hidden_channels=(16, 32),
            number_of_fc_layers=1,
            reduction_factor=0.5,
            growing_conv_type=Conv2dGrowingModule,
        )
        plain_conv_model.set_growing_layers(scheduling_method="all")
        self.assertGreater(len(plain_conv_model._growing_layers), 0)

    def test_first_order_improvement_and_complete_growth(self):
        """Test VGG growth metrics and completing growth to target widths."""
        model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=3,
            number_of_conv_per_stage=(1, 3, 3),
            hidden_channels=(32, 64, 128),
            number_of_fc_layers=1,
            reduction_factor=0.125,
            device=torch.device("cpu"),
        )

        layer = model._growable_layers[0]
        assert isinstance(layer, Conv2dGrowingModule)
        update_decrease = 1.0
        eigenvalues = [10.0, 5.0]
        layer.parameter_update_decrease = torch.tensor(update_decrease)
        layer.eigenvalues_extension = torch.tensor(eigenvalues)

        improvement = layer.first_order_improvement
        self.assertIsInstance(improvement.item(), float)
        self.assertAlmostEqual(
            improvement.item(),
            update_decrease + sum(x**2 for x in eigenvalues),
        )

        initial_missing = layer.missing_neurons()
        self.assertGreater(initial_missing, 0)
        initial_in_neurons = layer.in_neurons

        model.complete_growth(
            extension_kwargs={
                "output_extension_init": "zeros",
                "input_extension_init": "zeros",
            }
        )

        self.assertGreater(layer.in_neurons, initial_in_neurons)
        self.assertEqual(layer.missing_neurons(), 0)
        full_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=3,
            number_of_conv_per_stage=(1, 3, 3),
            hidden_channels=(32, 64, 128),
            number_of_fc_layers=1,
            reduction_factor=1.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(model.number_of_parameters(), full_model.number_of_parameters())

        model.set_growing_layers(scheduling_method="all")
        self.assertTrue(
            all(
                growable_layer.missing_neurons() == 0
                for growable_layer in model._growable_layers
            )
        )

    def test_initialize_weights_handles_conv_without_bias(self):
        """Test the conv initialization branch when a convolution has no bias."""
        model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=2,
            number_of_conv_per_stage=(1, 2),
            hidden_channels=(8, 16),
            number_of_fc_layers=1,
            init_weights=False,
        )
        first_conv = self._first_conv(model)
        first_conv.bias = None
        model._initialize_weights()
        self.assertIsNone(first_conv.bias)

    def test_initialize_weights_handles_norms_without_weight_or_bias(self):
        """Test normalization init branches when affine weights or biases are absent."""
        group_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=2,
            number_of_conv_per_stage=(1, 1),
            hidden_channels=(8, 16),
            normalization="group",
            normalization_kwargs={"num_groups": 1, "affine": False},
            init_weights=False,
        )
        group_norm = next(
            module
            for module in group_model.modules()
            if isinstance(module, GrowingGroupNorm)
        )
        self.assertIsNone(group_norm.weight)
        self.assertIsNone(group_norm.bias)
        group_model._initialize_weights()

        layer_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            nb_stages=2,
            number_of_conv_per_stage=(1, 1),
            hidden_channels=(8, 16),
            normalization="layer",
            normalization_kwargs={"elementwise_affine": True, "bias": False},
            init_weights=False,
        )
        layer_norm = next(
            module
            for module in layer_model.modules()
            if isinstance(module, GrowingLayerNorm)
        )
        self.assertIsNotNone(layer_norm.weight)
        self.assertIsNone(layer_norm.bias)
        layer_model._initialize_weights()

    def test_custom_activation_and_classifier_configuration(self):
        """Test custom activation wiring and configurable classifier layout."""
        model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=11,
            nb_stages=3,
            number_of_conv_per_stage=1,
            hidden_channels=(8, 16, 32),
            normalization="batch",
            activation=nn.LeakyReLU(negative_slope=0.2),
            number_of_fc_layers=4,
            fc_layer_width=128,
        )

        activations = [
            module for module in model.modules() if isinstance(module, nn.LeakyReLU)
        ]
        self.assertGreater(len(activations), 0)

        classifier_linears = [
            module
            for module in model.classifier
            if isinstance(module, LinearGrowingModule)
        ]
        classifier_dropouts = [
            module for module in model.classifier if isinstance(module, nn.Dropout)
        ]
        self.assertEqual(len(classifier_linears), 4)
        self.assertEqual(len(classifier_dropouts), 3)

        self.assertEqual(model.final_spatial_shape, (4, 4))
        self.assertEqual(classifier_linears[0].in_features, 32 * 4 * 4)
        self.assertEqual(classifier_linears[0].out_features, 4)
        self.assertEqual(classifier_linears[1].in_features, 4)
        self.assertEqual(classifier_linears[1].out_features, 4)
        self.assertEqual(classifier_linears[1].target_in_neurons, 128)
        self.assertEqual(classifier_linears[2].in_features, 4)
        self.assertEqual(classifier_linears[2].out_features, 4)
        self.assertEqual(classifier_linears[2].target_in_neurons, 128)
        self.assertEqual(classifier_linears[3].in_features, 4)
        self.assertEqual(classifier_linears[3].out_features, 11)
        self.assertEqual(classifier_linears[3].target_in_neurons, 128)

        single_fc_model = init_full_vgg_structure(
            input_shape=(3, 32, 32),
            out_features=11,
            nb_stages=3,
            number_of_conv_per_stage=1,
            hidden_channels=(8, 16, 32),
            number_of_fc_layers=1,
            fc_layer_width=128,
        )
        single_fc_linears = [
            module
            for module in single_fc_model.classifier
            if isinstance(module, LinearGrowingModule)
        ]
        single_fc_dropouts = [
            module
            for module in single_fc_model.classifier
            if isinstance(module, nn.Dropout)
        ]
        self.assertEqual(len(single_fc_linears), 1)
        self.assertEqual(len(single_fc_dropouts), 0)
        self.assertEqual(single_fc_model.final_spatial_shape, (4, 4))
        self.assertEqual(single_fc_linears[0].in_features, 32 * 4 * 4)
        self.assertEqual(single_fc_linears[0].out_features, 11)

        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        self.assertShapeEqual(output, (2, 11))
