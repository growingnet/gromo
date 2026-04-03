from __future__ import annotations

from copy import deepcopy
from math import ceil, prod
from typing import Callable, Literal, TypeAlias, TypedDict, Union, cast

import torch
import torch.nn as nn

from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    RestrictedConv2dGrowingModule,
)
from gromo.modules.growing_normalisation import (
    GrowingBatchNorm2d,
    GrowingGroupNorm,
    GrowingLayerNorm,
)
from gromo.modules.linear_growing_module import LinearGrowingModule


class NormKwargs(TypedDict, total=False):
    """Optional normalization configuration."""

    # BatchNorm keys
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # GroupNorm key
    num_groups: int
    # LayerNorm keys
    elementwise_affine: bool
    bias: bool


class CompleteBatchNormKwargs(TypedDict):
    """Complete batch-normalization configuration."""

    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool


class CompleteGroupNormKwargs(TypedDict):
    """Complete group-normalization configuration."""

    num_groups: int
    eps: float
    affine: bool


class CompleteLayerNormKwargs(TypedDict):
    """Complete layer-normalization configuration."""

    eps: float
    elementwise_affine: bool
    bias: bool


base_batch_norm_kwargs: CompleteBatchNormKwargs = {
    "eps": 1e-5,
    "momentum": 0.1,
    "affine": True,
    "track_running_stats": True,
}

base_group_norm_kwargs: CompleteGroupNormKwargs = {
    "num_groups": 32,
    "eps": 1e-5,
    "affine": True,
}

base_layer_norm_kwargs: CompleteLayerNormKwargs = {
    "eps": 1e-5,
    "elementwise_affine": True,
    "bias": True,
}

NormalizationType: TypeAlias = Literal["batch", "group", "layer"]


class _VGGStageBlock(nn.Module):
    """VGG stage made of consecutive convolutions sharing one target stage width."""

    def __init__(
        self,
        stage_channels: tuple[int, ...],
        target_stage_channels: tuple[int, ...],
        *,
        in_channels: int,
        spatial_shape: tuple[int, int] | None,
        build_post_conv_layers: Callable[[int, tuple[int, int] | None], nn.Module],
        growing_conv_type: type[Conv2dGrowingModule],
        device: torch.device | str | None,
        name: str,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(target_stage_channels):
            raise ValueError(
                "stage_channels and target_stage_channels must have the same length."
            )
        if len(stage_channels) == 0:
            raise ValueError("A VGG stage must contain at least one convolution.")

        self.convs = nn.Sequential()
        self.growing_modules: list[Conv2dGrowingModule] = []
        self.growable_layers: list[Conv2dGrowingModule] = []

        previous_growing_layer: Conv2dGrowingModule | None = None
        current_in_channels = in_channels
        for layer_index, (out_channels, _) in enumerate(
            zip(stage_channels, target_stage_channels, strict=True)
        ):
            is_first_conv = layer_index == 0
            post_layer_function = build_post_conv_layers(
                out_channels,
                spatial_shape,
            )
            target_in_channels = (
                None if is_first_conv else target_stage_channels[layer_index - 1]
            )
            conv = growing_conv_type(
                in_channels=current_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                use_bias=True,
                previous_module=None if is_first_conv else previous_growing_layer,
                post_layer_function=post_layer_function,
                allow_growing=not is_first_conv,
                target_in_channels=target_in_channels,
                name=f"{name}.convs.{layer_index}.conv",
                device=device,
            )
            self.convs.append(conv)
            self.growing_modules.append(conv)
            if (
                not is_first_conv
                and target_in_channels is not None
                and target_in_channels > conv.in_channels
            ):
                self.growable_layers.append(conv)
            previous_growing_layer = conv
            current_in_channels = out_channels

        self.out_channels = current_in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.convs:
            x = module(x)
        return x

    def extended_forward(
        self,
        x: torch.Tensor,
        x_ext: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for module in self.convs:
            x, x_ext = module.extended_forward(x, x_ext)
        return x, x_ext


class VGG(SequentialGrowingModel):
    """Growable VGG backbone with torchvision-aligned classifier."""

    def __init__(
        self,
        cfg: list[Union[str, int]],
        target_cfg: list[Union[str, int]] | None = None,
        in_features: int = 3,
        activation: nn.Module = nn.ReLU(inplace=True),
        normalization: NormalizationType | None = "batch",
        normalization_kwargs: NormKwargs | None = None,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
        number_of_fc_layers: int = 3,
        fc_layer_width: int = 4096,
        initial_fc_layer_width: int | None = None,
        input_spatial_shape: tuple[int, int] | None = None,
        device: torch.device | str | None = None,
        growing_conv_type: type[Conv2dGrowingModule] = RestrictedConv2dGrowingModule,
    ) -> None:
        super().__init__(in_features=in_features, out_features=num_classes, device=device)
        if number_of_fc_layers <= 0:
            raise ValueError(
                f"number_of_fc_layers must be > 0, got {number_of_fc_layers}."
            )
        if fc_layer_width <= 0:
            raise ValueError(f"fc_layer_width must be > 0, got {fc_layer_width}.")
        if initial_fc_layer_width is not None and initial_fc_layer_width <= 0:
            raise ValueError(
                f"initial_fc_layer_width must be > 0, got {initial_fc_layer_width}."
            )

        self.activation = activation.to(device)
        self.normalization: NormalizationType | None = self._validate_normalization(
            normalization
        )
        self.batch_norm_kwargs: CompleteBatchNormKwargs = base_batch_norm_kwargs.copy()
        self.group_norm_kwargs: CompleteGroupNormKwargs = base_group_norm_kwargs.copy()
        self.layer_norm_kwargs: CompleteLayerNormKwargs = base_layer_norm_kwargs.copy()
        if normalization_kwargs is not None:
            if self.normalization == "batch":
                self._update_batch_norm_kwargs(normalization_kwargs)
            elif self.normalization == "group":
                self._update_group_norm_kwargs(normalization_kwargs)
            elif self.normalization == "layer":
                self._update_layer_norm_kwargs(normalization_kwargs)
            else:
                raise ValueError(
                    "normalization_kwargs cannot be provided when normalization is None."
                )

        self.growing_conv_type = growing_conv_type
        self.initial_fc_layer_width = (
            fc_layer_width if initial_fc_layer_width is None else initial_fc_layer_width
        )
        self.input_spatial_shape = input_spatial_shape
        if self.normalization == "layer" and self.input_spatial_shape is None:
            raise ValueError(
                "input_spatial_shape must be provided when normalization='layer'."
            )

        current_spatial_shape = self.input_spatial_shape

        self.features = nn.Sequential()
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential()
        if target_cfg is None:
            target_cfg = cfg

        feature_stages: list[tuple[list[int], list[int], bool]] = []
        current_stage: list[int] = []
        current_target_stage: list[int] = []
        for value, target_value in zip(cfg, target_cfg, strict=True):
            if value == "M":
                if target_value != "M":
                    raise ValueError(
                        "target_cfg must match cfg pooling structure exactly."
                    )
                if current_stage:
                    feature_stages.append((current_stage, current_target_stage, True))
                    current_stage = []
                    current_target_stage = []
                continue
            current_stage.append(cast("int", value))
            current_target_stage.append(cast("int", target_value))
        if current_stage:
            feature_stages.append((current_stage, current_target_stage, False))

        self.stage_blocks = nn.ModuleList()
        self._feature_growing_modules: list[Conv2dGrowingModule] = []
        self._classifier_growing_modules: list[LinearGrowingModule] = []
        growable_layers: list[Conv2dGrowingModule | LinearGrowingModule] = []

        in_channels = in_features
        current_stage_target_out_channels = in_features
        feature_module_index = 0
        for stage_index, (stage_channels, target_stage_channels, has_pool) in enumerate(
            feature_stages
        ):
            stage_block = _VGGStageBlock(
                tuple(stage_channels),
                tuple(target_stage_channels),
                in_channels=in_channels,
                spatial_shape=current_spatial_shape,
                build_post_conv_layers=self._build_post_conv_layers,
                growing_conv_type=growing_conv_type,
                device=self.device,
                name=f"features.{feature_module_index}",
            )
            self.stage_blocks.append(stage_block)
            self.features.append(stage_block)
            feature_module_index += 1
            self._feature_growing_modules.extend(stage_block.growing_modules)
            growable_layers.extend(stage_block.growable_layers)
            in_channels = stage_block.out_channels
            current_stage_target_out_channels = target_stage_channels[-1]
            if has_pool:
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
                feature_module_index += 1
                if current_spatial_shape is not None:
                    current_spatial_shape = (
                        current_spatial_shape[0] // 2,
                        current_spatial_shape[1] // 2,
                    )

        self.final_spatial_shape = (
            (7, 7) if current_spatial_shape is None else current_spatial_shape
        )
        self.avgpool = nn.AdaptiveAvgPool2d(self.final_spatial_shape)

        classifier_growing_layers: list[LinearGrowingModule] = []
        previous_classifier_layer = (
            self._feature_growing_modules[-1] if self._feature_growing_modules else None
        )
        classifier_target_in_features = current_stage_target_out_channels * prod(
            self.final_spatial_shape
        )
        classifier_current_in_features = in_channels * prod(self.final_spatial_shape)
        for classifier_layer_idx in range(number_of_fc_layers):
            is_last_layer = classifier_layer_idx == number_of_fc_layers - 1
            layer_in_features = classifier_current_in_features
            target_in_features = (
                classifier_target_in_features
                if classifier_layer_idx == 0
                else fc_layer_width
            )
            layer_out_features = (
                num_classes if is_last_layer else self.initial_fc_layer_width
            )
            post_layer_function: nn.Module = (
                nn.Identity() if is_last_layer else self._make_activation()
            )

            linear = LinearGrowingModule(
                in_features=layer_in_features,
                out_features=layer_out_features,
                use_bias=True,
                previous_module=previous_classifier_layer,
                post_layer_function=post_layer_function,
                allow_growing=(
                    classifier_layer_idx > 0 and target_in_features > layer_in_features
                ),
                name=f"classifier.{3 * classifier_layer_idx}",
                device=self.device,
                target_in_features=target_in_features,
            )
            self.classifier.append(linear)
            self._classifier_growing_modules.append(linear)
            if classifier_layer_idx > 0 and target_in_features > layer_in_features:
                classifier_growing_layers.append(linear)
            previous_classifier_layer = linear

            if not is_last_layer:
                self.classifier.append(nn.Dropout(p=dropout))
                classifier_current_in_features = self.initial_fc_layer_width

        growable_layers.extend(classifier_growing_layers)
        self._growable_layers = list(growable_layers)
        self.set_growing_layers(scheduling_method="all")

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the standard VGG forward pass."""
        for module in self.features:
            x = module(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        for module in self.classifier:
            x = module(x)
        return x

    def extended_forward(
        self,
        x: torch.Tensor,
        mask: dict | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run the forward pass while tracking the growth extension branch."""
        x_ext = None
        for module in self.features:
            if isinstance(module, (_VGGStageBlock, Conv2dGrowingModule)):
                x, x_ext = module.extended_forward(x, x_ext)
            else:
                x = module(x)
                if x_ext is not None:
                    x_ext = module(x_ext)
        x = self.avgpool(x)
        if x_ext is not None:
            x_ext = self.avgpool(x_ext)
        x = self.flatten(x)
        if x_ext is not None:
            x_ext = self.flatten(x_ext)
        for module in self.classifier:
            if isinstance(module, LinearGrowingModule):
                x, x_ext = module.extended_forward(x, x_ext)
            else:
                x = module(x)
                if x_ext is not None:
                    x_ext = module(x_ext)
        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, Conv2dGrowingModule):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(
                module, (GrowingBatchNorm2d, GrowingGroupNorm, GrowingLayerNorm)
            ):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, LinearGrowingModule):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _validate_normalization(
        normalization: NormalizationType | None,
    ) -> NormalizationType | None:
        if normalization is None:
            return None
        if normalization in {"batch", "group", "layer"}:
            return normalization
        raise ValueError(
            "normalization must be 'batch', 'group', 'layer' or None, "
            f"got {normalization!r}."
        )

    def _update_batch_norm_kwargs(self, normalization_kwargs: NormKwargs) -> None:
        for key, value in normalization_kwargs.items():
            if key not in self.batch_norm_kwargs:
                raise ValueError(
                    f"Unsupported BatchNorm kwarg {key!r}. "
                    "Allowed keys: eps, momentum, affine, track_running_stats."
                )
            self.batch_norm_kwargs[key] = value  # type: ignore[index]

    def _update_group_norm_kwargs(self, normalization_kwargs: NormKwargs) -> None:
        for key, value in normalization_kwargs.items():
            if key not in self.group_norm_kwargs:
                raise ValueError(
                    f"Unsupported GroupNorm kwarg {key!r}. "
                    "Allowed keys: num_groups, eps, affine."
                )
            self.group_norm_kwargs[key] = value  # type: ignore[index]

    def _update_layer_norm_kwargs(self, normalization_kwargs: NormKwargs) -> None:
        for key, value in normalization_kwargs.items():
            if key not in self.layer_norm_kwargs:
                raise ValueError(
                    f"Unsupported LayerNorm kwarg {key!r}. "
                    "Allowed keys: eps, elementwise_affine, bias."
                )
            self.layer_norm_kwargs[key] = value  # type: ignore[index]

    def _make_activation(self) -> nn.Module:
        return deepcopy(self.activation)

    def _build_growing_normalization(
        self,
        num_channels: int,
        spatial_shape: tuple[int, int] | None = None,
    ) -> nn.Module | None:
        if self.normalization is None:
            return None
        if self.normalization == "batch":
            return GrowingBatchNorm2d(
                num_channels,
                eps=self.batch_norm_kwargs["eps"],
                momentum=self.batch_norm_kwargs["momentum"],
                affine=self.batch_norm_kwargs["affine"],
                track_running_stats=self.batch_norm_kwargs["track_running_stats"],
                device=self.device,
            )
        if self.normalization == "group":
            return GrowingGroupNorm(
                num_groups=self.group_norm_kwargs["num_groups"],
                num_channels=num_channels,
                eps=self.group_norm_kwargs["eps"],
                affine=self.group_norm_kwargs["affine"],
                device=self.device,
            )
        if self.normalization == "layer":
            if spatial_shape is None:
                raise ValueError(
                    "spatial_shape must be provided when normalization='layer'."
                )
            return GrowingLayerNorm(
                [num_channels, *spatial_shape],
                eps=self.layer_norm_kwargs["eps"],
                elementwise_affine=self.layer_norm_kwargs["elementwise_affine"],
                bias=self.layer_norm_kwargs["bias"],
                device=self.device,
            )
        raise AssertionError("Normalization should have been validated before use.")

    def _build_post_conv_layers(
        self,
        num_channels: int,
        spatial_shape: tuple[int, int] | None = None,
    ) -> nn.Module:
        normalization = self._build_growing_normalization(num_channels, spatial_shape)
        layers: list[nn.Module] = []
        if normalization is not None:
            layers.append(normalization)
        layers.append(self._make_activation())
        if len(layers) == 1:
            return layers[0]
        return nn.Sequential(*layers)


def _reduce_growing_conv_widths(
    stage_hidden_per_block: tuple[int, ...],
    reduction_factor: float | None,
) -> tuple[int, ...]:
    if reduction_factor is None:
        return stage_hidden_per_block
    return tuple(
        stage_width
        if block_idx == len(stage_hidden_per_block) - 1
        else ceil(stage_width * reduction_factor)
        for block_idx, stage_width in enumerate(stage_hidden_per_block)
    )


def init_full_vgg_structure(
    input_shape: tuple[int, int, int] = (3, 224, 224),
    in_features: int | None = None,
    out_features: int = 1000,
    device: torch.device | str | None = None,
    activation: nn.Module = nn.ReLU(inplace=True),
    normalization: NormalizationType | None = "batch",
    normalization_kwargs: NormKwargs | None = None,
    hidden_channels: tuple[int | tuple[int, ...], ...] | None = None,
    number_of_conv_per_stage: int | tuple[int, ...] = (1, 1, 2, 2, 2),
    nb_stages: int = 5,
    init_weights: bool = True,
    dropout: float = 0.5,
    number_of_fc_layers: int = 3,
    fc_layer_width: int = 4096,
    reduction_factor: float | None = 1 / 32,
    growing_conv_type: type[Conv2dGrowingModule] = RestrictedConv2dGrowingModule,
) -> VGG:
    """Initialize a customizable VGG-style model with full stage/block control.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of the input tensor (C, H, W). Used to infer in_features when
        ``in_features`` is not provided.
    in_features : int | None
        Number of input channels. If None, inferred from ``input_shape``.
    out_features : int
        Number of output classes.
    device : torch.device | str | None
        Device to run the model on.
    activation : nn.Module
        Activation function to use after each normalized convolution/linear
        block in the feature extractor and classifier hidden layers.
    normalization : NormalizationType | None
        Normalization layer to use. Supported values are ``"batch"``,
        ``"group"``, ``"layer"``, and ``None``.
    normalization_kwargs : NormKwargs | None
        Additional keyword arguments passed to the selected normalization layer.
        Supported keys are:
        ``eps``, ``momentum``, ``affine``, ``track_running_stats`` for batch norm;
        ``num_groups``, ``eps``, ``affine`` for group norm;
        ``eps``, ``elementwise_affine``, ``bias`` for layer norm.
    hidden_channels : tuple[int | tuple[int, ...], ...] | None
        Explicit channels per stage/block. If provided, each stage entry can be:
        - int: same width for all blocks in the stage
        - tuple[int, ...]: explicit width per block in the stage
    number_of_conv_per_stage : int | tuple[int, ...]
        Number of convolutions in each stage. If an int is provided, the
        same value is used across all stages.
    nb_stages : int
        Number of convolution stages.
    init_weights : bool
        If True, initialize model weights with VGG-style initialization.
    dropout : float
        Dropout probability used in the classifier.
    number_of_fc_layers : int
        Number of fully connected layers in the classifier.
    fc_layer_width : int
        Target hidden width of fully connected classifier layers.
    reduction_factor : float | None
        Factor used to initialize layers with ``allow_growing=True`` from
        their target width. For convolutions, this applies to layers that are
        not the first convolution of a stage. For the classifier, this applies
        to growable hidden fully connected layers. If ``None``, keep the target
        width unchanged for those layers.
    growing_conv_type : type[Conv2dGrowingModule]
        Convolution growing module type to instantiate.

    Returns
    -------
    VGG
        The initialized VGG model.

    Raises
    ------
    TypeError
        If classifier, reduction, stage-convolution, or hidden-channel
        arguments have invalid types.
    ValueError
        If classifier sizes are not strictly positive, ``reduction_factor`` is
        out of bounds, stage/block shapes are inconsistent, normalization
        arguments are invalid, or normalization-specific requirements are not
        met.
    """
    if isinstance(input_shape, torch.Size):
        input_shape = tuple(input_shape)  # type: ignore[assignment]
    if len(input_shape) != 3:
        raise ValueError(
            f"input_shape must be a sequence of length 3 (C, H, W), got {input_shape}."
        )
    if in_features is None:
        in_features = input_shape[0]

    if not isinstance(number_of_fc_layers, int):
        raise TypeError(
            f"number_of_fc_layers must be an int, got {type(number_of_fc_layers).__name__}."
        )
    if number_of_fc_layers <= 0:
        raise ValueError(f"number_of_fc_layers must be > 0, got {number_of_fc_layers}.")

    if not isinstance(fc_layer_width, int):
        raise TypeError(
            f"fc_layer_width must be an int, got {type(fc_layer_width).__name__}."
        )
    if fc_layer_width <= 0:
        raise ValueError(f"fc_layer_width must be > 0, got {fc_layer_width}.")
    if reduction_factor is not None and not isinstance(reduction_factor, (int, float)):
        raise TypeError(
            "reduction_factor must be a float, int, or None, "
            f"got {type(reduction_factor).__name__}."
        )
    if reduction_factor is not None and not (0 < float(reduction_factor) <= 1):
        raise ValueError(
            "reduction_factor must satisfy 0 < reduction_factor <= 1, "
            f"got {reduction_factor}."
        )

    if isinstance(number_of_conv_per_stage, int):
        conv_per_stage: tuple[int, ...] = (number_of_conv_per_stage,) * nb_stages
    elif (
        isinstance(number_of_conv_per_stage, (list, tuple))
        and len(number_of_conv_per_stage) == nb_stages
    ):
        conv_per_stage = tuple(number_of_conv_per_stage)
    else:
        raise TypeError(
            f"number_of_conv_per_stage must be an int or a tuple of {nb_stages} ints."
        )

    for stage_idx, num_conv in enumerate(conv_per_stage):
        if not isinstance(num_conv, int):
            raise TypeError(
                f"Stage {stage_idx}: number_of_conv_per_stage must contain ints, "
                f"got {type(num_conv).__name__}."
            )
        if num_conv <= 0:
            raise ValueError(
                f"Stage {stage_idx}: number_of_conv_per_stage must be > 0, "
                f"got {num_conv}."
            )

    hidden_channels_per_block: list[tuple[int, ...]] = []
    target_hidden_channels_per_block: list[tuple[int, ...]] = []
    if hidden_channels is not None:
        if len(hidden_channels) != nb_stages:
            raise ValueError(
                f"hidden_channels must have {nb_stages} elements (one per stage), "
                f"but got {len(hidden_channels)}."
            )
        for stage_idx, stage_hidden in enumerate(hidden_channels):
            num_conv = conv_per_stage[stage_idx]
            if isinstance(stage_hidden, int):
                stage_hidden_per_block = (stage_hidden,) * num_conv
            elif isinstance(stage_hidden, (list, tuple)):
                if len(stage_hidden) != num_conv:
                    raise ValueError(
                        f"Stage {stage_idx}: hidden_channels has {len(stage_hidden)} "
                        f"elements but number_of_conv_per_stage is {num_conv}."
                    )
                stage_hidden_per_block = tuple(stage_hidden)
            else:
                raise TypeError(
                    f"Stage {stage_idx}: hidden_channels element must be int or tuple, "
                    f"got {type(stage_hidden).__name__}."
                )

            for block_idx, stage_width in enumerate(stage_hidden_per_block):
                if not isinstance(stage_width, int):
                    raise TypeError(
                        f"Stage {stage_idx} Block {block_idx}: hidden_channels must "
                        f"contain ints, got {type(stage_width).__name__}."
                    )
                if stage_width <= 0:
                    raise ValueError(
                        f"Stage {stage_idx} Block {block_idx}: hidden_channels must "
                        f"be > 0, got {stage_width}."
                    )

            target_hidden_channels_per_block.append(stage_hidden_per_block)
            hidden_channels_per_block.append(
                _reduce_growing_conv_widths(
                    stage_hidden_per_block, float(reduction_factor)
                )
                if reduction_factor is not None
                else stage_hidden_per_block
            )
    else:
        for stage_idx in range(nb_stages):
            num_conv = conv_per_stage[stage_idx]
            stage_hidden = 64 * min(2**stage_idx, 8)
            if stage_hidden <= 0:
                raise ValueError(
                    f"Stage {stage_idx}: computed hidden width must be > 0, "
                    f"got {stage_hidden}."
                )
            target_stage_hidden = (stage_hidden,) * num_conv
            target_hidden_channels_per_block.append(target_stage_hidden)
            hidden_channels_per_block.append(
                _reduce_growing_conv_widths(target_stage_hidden, float(reduction_factor))
                if reduction_factor is not None
                else target_stage_hidden
            )

    cfg: list[Union[str, int]] = []
    target_cfg: list[Union[str, int]] = []
    for stage_hidden in hidden_channels_per_block:
        cfg.extend(stage_hidden)
        cfg.append("M")
    for target_stage_hidden in target_hidden_channels_per_block:
        target_cfg.extend(target_stage_hidden)
        target_cfg.append("M")

    return VGG(
        cfg=cfg,
        target_cfg=target_cfg,
        in_features=in_features,
        activation=activation,
        normalization=normalization,
        normalization_kwargs=normalization_kwargs,
        num_classes=out_features,
        init_weights=init_weights,
        dropout=dropout,
        number_of_fc_layers=number_of_fc_layers,
        fc_layer_width=fc_layer_width,
        initial_fc_layer_width=(
            ceil(fc_layer_width * float(reduction_factor))
            if reduction_factor is not None
            else fc_layer_width
        ),
        input_spatial_shape=input_shape[1:],
        device=device,
        growing_conv_type=growing_conv_type,
    )


if __name__ == "__main__":
    try:
        from torchvision.models import vgg11
    except ImportError:
        print("torchvision is not installed; skipping VGG parity smoke test.")
    else:
        torch.manual_seed(0)
        model = init_full_vgg_structure(
            input_shape=(3, 224, 224),
            out_features=1000,
            normalization=None,
            number_of_conv_per_stage=(1, 1, 2, 2, 2),
            hidden_channels=(64, 128, 256, 512, 512),
            number_of_fc_layers=3,
            fc_layer_width=4096,
            reduction_factor=None,
            device=torch.device("cpu"),
        )
        reference = vgg11(weights=None)

        reference_convs = [
            module for module in reference.features if isinstance(module, nn.Conv2d)
        ]
        assert len(reference_convs) == len(model._feature_growing_modules)
        for ref_conv, model_conv in zip(
            reference_convs, model._feature_growing_modules, strict=True
        ):
            with torch.no_grad():
                model_conv.weight.copy_(ref_conv.weight)
                assert model_conv.bias is not None
                assert ref_conv.bias is not None
                model_conv.bias.copy_(ref_conv.bias)

        reference_linears = [
            module for module in reference.classifier if isinstance(module, nn.Linear)
        ]
        model_linears = [
            module
            for module in model.classifier
            if isinstance(module, LinearGrowingModule)
        ]
        assert len(reference_linears) == len(model_linears)
        for ref_linear, model_linear in zip(
            reference_linears, model_linears, strict=True
        ):
            with torch.no_grad():
                model_linear.weight.copy_(ref_linear.weight)
                assert model_linear.bias is not None
                assert ref_linear.bias is not None
                model_linear.bias.copy_(ref_linear.bias)

        model.eval()
        reference.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            model_output = model(x)
            reference_output = reference(x)

        max_diff = torch.max(torch.abs(model_output - reference_output)).item()
        print("torchvision VGG11 parity smoke test")
        print(
            f"parameter_count_match={model.number_of_parameters() == sum(p.numel() for p in reference.parameters())}"
        )
        print(f"output_shape={tuple(model_output.shape)}")
        print(f"max_abs_diff={max_diff:.6e}")
