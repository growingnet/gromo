"""
Module to create a ResNet with basic blocks (like ResNet-18 or ResNet-34).
Will allow to extend the basic blocks with more intermediate channels
and to add basic blocks add the end of the stages.
"""

from math import ceil

import torch
from torch import nn

from gromo.containers.growing_block import Conv2dGrowingBlock
from gromo.containers.sequential_growing_container import SequentialGrowingContainer
from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    RestrictedConv2dGrowingModule,
)
from gromo.modules.growing_normalisation import GrowingBatchNorm2d


class ResNetBasicBlock(SequentialGrowingContainer):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1000,
        device: torch.device | str | None = None,
        activation: nn.Module = nn.ReLU(),
        input_block_kernel_size: int = 3,
        output_block_kernel_size: int = 3,
        hidden_channels: tuple[int, ...] = (0, 0, 0, 0),
        small_inputs: bool = False,
        inplanes: int = 64,
        use_preactivation: bool = True,
        growing_conv_type: type[Conv2dGrowingModule] = RestrictedConv2dGrowingModule,
    ) -> None:
        """
        Initialize the ResNet with basic blocks.
        Parameters
        ----------
        in_features : int
            Number of input features (channels).
        out_features : int
            Number of output features (channels).
        device : torch.device | str | None
            Device to run the model on.
        activation : nn.Module
            Activation function to use.
        input_block_kernel_size : int
            Kernel size for the input block.
        output_block_kernel_size : int
            Kernel size for the output block.
        hidden_channels : tuple[int, ...]
            Number of hidden channels for the first block of each stage.
            Length determines nb_stages.
        small_inputs : bool
            If True, adapt the network for small input images (e.g., CIFAR-10/100).
            This uses smaller kernels, no stride, and
            no max pooling in the initial layers.
        inplanes : int
            Number of initial planes (channels) after the first convolution.
            (Default is 64 as in standard ResNet architectures.)
        use_preactivation : bool
            If True, use full pre-activation ResNet (BN-ReLU before conv).
            If False, use classical ResNet (conv-BN-ReLU).
        growing_conv_type : type[Conv2dGrowingModule]
            Type of convolutional growing module to use
            (e.g. RestrictedConv2dGrowingModule, FullConv2dGrowingModule, ...).
        """
        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )
        self.activation = activation.to(device)
        self.small_inputs = small_inputs
        self.use_preactivation = use_preactivation
        self.inplanes = inplanes
        self.input_block_kernel_size = input_block_kernel_size
        self.output_block_kernel_size = output_block_kernel_size
        self.growing_conv_type = growing_conv_type

        nb_stages = len(hidden_channels)
        self.pre_net = self._build_pre_net(in_features, inplanes)

        self.stages: nn.ModuleList = nn.ModuleList()
        for i in range(nb_stages):
            input_channels = inplanes * (2 ** max(0, i - 1))
            output_channels = inplanes * (2**i)
            stage_hidden_channels = hidden_channels[i]

            # For small inputs, adjust stride behavior
            stage_stride = 2 if (i > 0 and not (small_inputs and i == 1)) else 1
            if small_inputs and i == 1:
                stage_stride = 1

            stage = nn.Sequential()
            block = self._create_block(
                in_channels=input_channels,
                out_channels=output_channels,
                hidden_channels=stage_hidden_channels,
                stride=stage_stride,
                name=f"Stage {i} Block 0",
                use_downsample=(i > 0),
            )
            stage.append(block)
            if not use_preactivation:
                stage.append(self.activation)
            self.stages.append(stage)

        self.post_net = self._build_post_net(inplanes * (2 ** (nb_stages - 1)))

        # Initialize the growing layers list with all
        # RestrictedConv2dGrowingBlock instances
        self._growing_layers = []
        for stage in self.stages:  # type: ignore
            stage: nn.Sequential
            for block in stage:  # type: ignore
                block: Conv2dGrowingBlock | nn.Module
                if isinstance(block, Conv2dGrowingBlock):
                    self._growable_layers.append(block)

    def _build_pre_net(self, in_features: int, inplanes: int) -> nn.Sequential:
        """Build the pre-network (stem) based on input size and architecture type."""
        if self.small_inputs:
            # For small inputs like CIFAR-10/100 (32x32)
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_features,
                    inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=self.device,
                ),
            ]
        else:
            # For large inputs like ImageNet (224x224)
            layers = [
                nn.Conv2d(
                    in_features,
                    inplanes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                    device=self.device,
                ),
            ]

        if not self.use_preactivation:
            layers.append(nn.BatchNorm2d(inplanes, device=self.device))
            layers.append(self.activation)

        if not self.small_inputs:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return nn.Sequential(*layers)

    def _build_post_net(self, final_channels: int) -> nn.Sequential:
        """Build the post-network (head) based on architecture type."""
        layers: list[nn.Module] = []
        if self.use_preactivation:
            layers.append(nn.BatchNorm2d(final_channels, device=self.device))
            layers.append(self.activation)
        layers.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(final_channels, self.out_features, device=self.device),
            ]
        )
        return nn.Sequential(*layers)

    def _create_block(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        stride: int,
        name: str,
        use_downsample: bool = False,
        input_block_kernel_size: int | None = None,
        output_block_kernel_size: int | None = None,
    ) -> Conv2dGrowingBlock:
        """Create a ResNet block with the appropriate configuration."""
        if input_block_kernel_size is None:
            input_block_kernel_size = self.input_block_kernel_size
        if output_block_kernel_size is None:
            output_block_kernel_size = self.output_block_kernel_size

        kwargs_first_layer = {
            "kernel_size": input_block_kernel_size,
            "padding": 1,
            "use_bias": False,
            "stride": stride,
        }
        kwargs_layer = {
            "kernel_size": output_block_kernel_size,
            "padding": 1,
            "use_bias": False,
        }
        mid_activation = nn.Sequential(
            GrowingBatchNorm2d(hidden_channels, device=self.device),
            self.activation,
        )

        if self.use_preactivation:
            pre_activation: nn.Module | None = nn.Sequential(
                nn.BatchNorm2d(in_channels, device=self.device),
                self.activation,
            )
            pre_addition_function: nn.Module = nn.Identity()
            downsample: nn.Module = (
                nn.Sequential(
                    nn.BatchNorm2d(in_channels, device=self.device),
                    self.activation,
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        device=self.device,
                    ),
                )
                if use_downsample
                else nn.Identity()
            )
        else:
            pre_activation = None
            pre_addition_function = nn.BatchNorm2d(out_channels, device=self.device)
            downsample = (
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        device=self.device,
                    ),
                    nn.BatchNorm2d(out_channels, device=self.device),
                )
                if use_downsample
                else nn.Identity()
            )

        return Conv2dGrowingBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            kwargs_first_layer=kwargs_first_layer,
            kwargs_layer=kwargs_layer,
            pre_activation=pre_activation,
            mid_activation=mid_activation,
            extended_mid_activation=self.activation,
            pre_addition_function=pre_addition_function,
            name=name,
            target_hidden_channels=out_channels,
            downsample=downsample,
            growing_conv_type=self.growing_conv_type,
            device=self.device,
        )

    def append_block(
        self,
        stage_index: int = 0,
        input_block_kernel_size: int | None = None,
        output_block_kernel_size: int | None = None,
        hidden_channels: int = 0,
    ) -> None:
        """
        Append a new block to the specified stage of the ResNet.
        """
        if not self.use_preactivation:
            assert hidden_channels > 0, (
                "As you are using the classical ResNet, "
                "hidden_channels must be greater than 0."
            )
        if stage_index < 0 or stage_index >= len(self.stages):
            raise IndexError(
                f"Stage {stage_index} is out of range. "
                f"There are {len(self.stages)} stages."
            )

        stage: nn.Sequential = self.stages[stage_index]  # type: ignore
        # For classical mode, the last element is activation, so use -2
        ref_block_idx = -1 if self.use_preactivation else -2
        input_channels = stage[ref_block_idx].out_features
        output_channels = input_channels

        new_block = self._create_block(
            in_channels=input_channels,
            out_channels=output_channels,
            hidden_channels=hidden_channels,
            stride=1,
            name=f"Stage {stage_index} Block {len(stage)}",
            use_downsample=False,
            input_block_kernel_size=input_block_kernel_size,
            output_block_kernel_size=output_block_kernel_size,
        )

        stage.append(new_block)
        if not self.use_preactivation:
            stage.append(self.activation)
        # Add the new block to the growing layers list
        self._growable_layers.append(new_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        for stage in self.stages:
            x = stage(x)
        x = self.post_net(x)
        return x

    def extended_forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        mask: dict | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        x = self.pre_net(x)
        for stage in self.stages:  # type: ignore
            stage: nn.Sequential
            for block in stage:  # type: ignore
                block: Conv2dGrowingBlock | nn.Module
                if isinstance(block, Conv2dGrowingBlock):
                    x = block.extended_forward(x)
                else:
                    x = block(x)
        x = self.post_net(x)
        return x


def init_full_resnet_structure(
    input_shape: tuple[int, int, int] = (3, 224, 224),
    in_features: int | None = None,
    out_features: int = 1000,
    device: torch.device | str | None = None,
    activation: nn.Module = nn.ReLU(),
    input_block_kernel_size: int = 3,
    output_block_kernel_size: int = 3,
    reduction_factor: float = 1 / 64,
    hidden_channels: tuple[int | tuple[int, ...], ...] | None = None,
    small_inputs: bool | None = None,
    number_of_blocks_per_stage: int | tuple[int, ...] = 2,
    inplanes: int = 64,
    nb_stages: int = 4,
    use_preactivation: bool = True,
    growing_conv_type: type[Conv2dGrowingModule] = RestrictedConv2dGrowingModule,
) -> ResNetBasicBlock:
    """
    Initialize a customizable ResNet-style model with basic blocks.
    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of the input tensor (C, H, W).
        Used to infer initial in_features and
        to adjust the architecture for small inputs if needed.
    in_features : int | None
        Number of input features (channels). If None,
        it will be inferred from input_shape.
    out_features : int
        Number of output features (channels).
    device : torch.device | str | None
        Device to run the model on.
    activation : nn.Module
        Activation function to use.
    input_block_kernel_size : int
        Kernel size for the input block.
    output_block_kernel_size : int
        Kernel size for the output block.
    reduction_factor : float
        Factor to reduce the number of channels in the bottleneck.
        If 0, starts with no channels. If 1, starts with all channels.
        Ignored if hidden_channels is provided.
    hidden_channels : tuple[int | tuple[int, ...], ...] | None
        Explicit hidden channels per stage/block. If provided, overrides
        reduction_factor. Can be:
        - tuple of int: same hidden_channels for all blocks in each stage
        - tuple of tuples: per-block hidden_channels for each stage
        - mixed: some stages with int (uniform), some with tuple (per-block)
        Length must match nb_stages, and inner tuple lengths must match
        number_of_blocks_per_stage.
    small_inputs : bool | None
        If True, adapt the network for small input images (e.g., CIFAR-10/100).
        This uses smaller kernels, no stride, and no max pooling in the initial layers.
    number_of_blocks_per_stage : int | tuple[int, ...]
        Number of basic blocks per stage. If an integer is provided, the same number
        of blocks will be used for all stages. If a tuple is provided, it should
        contain `nb_stages` integers specifying the number of blocks for each stage.
    inplanes : int
        Number of initial planes (channels) after the first convolution.
        (Default is 64 as in standard ResNet architectures.)
    nb_stages : int
        Number of stages in the ResNet.
    use_preactivation : bool
        If True, use full pre-activation ResNet (BN-ReLU before conv).
        If False, use classical ResNet (conv-BN-ReLU).
    growing_conv_type : type[Conv2dGrowingModule]
        Type of convolutional growing module to use
        (e.g. RestrictedConv2dGrowingModule, FullConv2dGrowingModule, ...).

    Returns
    -------
    ResNetBasicBlock
        The initialized ResNet-18 model.
    """
    if isinstance(input_shape, torch.Size):
        input_shape = tuple(input_shape)  # type: ignore
        assert len(input_shape) == 3, "input_shape must be a tuple of (C, H, W)"
    if in_features is None:
        in_features = input_shape[0]
    if small_inputs is None:
        small_inputs = input_shape[1] <= 32 and input_shape[2] <= 32

    # Normalize number_of_blocks_per_stage to a tuple
    if isinstance(number_of_blocks_per_stage, int):
        blocks_per_stage: tuple[int, ...] = (number_of_blocks_per_stage,) * nb_stages
    elif (
        isinstance(number_of_blocks_per_stage, (list, tuple))
        and len(number_of_blocks_per_stage) == nb_stages
    ):
        blocks_per_stage = tuple(number_of_blocks_per_stage)
    else:
        raise TypeError(
            f"number_of_blocks_per_stage must be an int or a tuple of {nb_stages} ints."
        )

    # Normalize hidden_channels to a tuple of tuples
    # (one tuple per stage, one int per block)
    if hidden_channels is not None:
        if len(hidden_channels) != nb_stages:
            raise ValueError(
                f"hidden_channels must have {nb_stages} elements (one per stage), "
                f"but got {len(hidden_channels)}."
            )
        hidden_channels_per_block: list[tuple[int, ...]] = []
        for stage_idx, stage_hidden in enumerate(hidden_channels):
            num_blocks = blocks_per_stage[stage_idx]
            if isinstance(stage_hidden, int):
                # Same hidden_channels for all blocks in this stage
                hidden_channels_per_block.append((stage_hidden,) * num_blocks)
            elif isinstance(stage_hidden, (list, tuple)):
                if len(stage_hidden) != num_blocks:
                    raise ValueError(
                        f"Stage {stage_idx}: hidden_channels has {len(stage_hidden)} "
                        f"elements but number_of_blocks_per_stage is {num_blocks}."
                    )
                hidden_channels_per_block.append(tuple(stage_hidden))
            else:
                raise TypeError(
                    f"Stage {stage_idx}: hidden_channels element must be int or tuple, "
                    f"got {type(stage_hidden).__name__}."
                )
    else:
        # Compute hidden_channels from reduction_factor
        hidden_channels_per_block = []
        for stage_idx in range(nb_stages):
            num_blocks = blocks_per_stage[stage_idx]
            stage_hidden = ceil(inplanes * (2**stage_idx) * reduction_factor)
            hidden_channels_per_block.append((stage_hidden,) * num_blocks)

    # Extract first block's hidden_channels for each stage (for ResNetBasicBlock)
    initial_hidden_channels = tuple(
        hidden_channels_per_block[i][0] for i in range(nb_stages)
    )

    model = ResNetBasicBlock(
        in_features=in_features,
        out_features=out_features,
        device=device,
        activation=activation,
        input_block_kernel_size=input_block_kernel_size,
        output_block_kernel_size=output_block_kernel_size,
        hidden_channels=initial_hidden_channels,
        small_inputs=small_inputs,
        inplanes=inplanes,
        use_preactivation=use_preactivation,
        growing_conv_type=growing_conv_type,
    )

    # Append additional blocks to complete each stage
    for stage_index in range(nb_stages):
        for block_idx in range(1, blocks_per_stage[stage_index]):
            block_hidden = hidden_channels_per_block[stage_index][block_idx]
            model.append_block(
                stage_index=stage_index,
                input_block_kernel_size=input_block_kernel_size,
                output_block_kernel_size=output_block_kernel_size,
                hidden_channels=block_hidden,
            )
    return model


if __name__ == "__main__":
    # Example usage and simple test
    print("=" * 60)
    print("Full Pre-activation ResNet")
    print("=" * 60)
    model_preact = init_full_resnet_structure(
        input_shape=(3, 224, 224),
        out_features=1_000,
        reduction_factor=1,
        number_of_blocks_per_stage=2,
        use_preactivation=True,
    )  # number of parameters: 11,688,616
    print(model_preact)

    from torchinfo import summary

    summary(model_preact, input_size=(1, 3, 224, 224))

    preact_params = sum(p.numel() for p in model_preact.parameters())
    print(f"Number of parameters (pre-activation): {preact_params}")
    assert (
        preact_params == 11_688_616
    ), f"Expected 11,688,616 parameters but got {preact_params}"

    print("\n" + "=" * 60)
    print("Classical ResNet")
    print("=" * 60)
    model_classical = init_full_resnet_structure(
        input_shape=(3, 224, 224),
        out_features=1_000,
        reduction_factor=1,
        number_of_blocks_per_stage=2,
        use_preactivation=False,
    )
    print(model_classical)

    summary(model_classical, input_size=(1, 3, 224, 224))

    classical_params = sum(p.numel() for p in model_classical.parameters())
    print(f"Number of parameters (classical): {classical_params}")

    # Compare with torchvision ResNet-18
    import torchvision.models as models

    torchvision_resnet18 = models.resnet18(weights=None)
    torchvision_params = sum(p.numel() for p in torchvision_resnet18.parameters())
    print(f"Number of parameters (torchvision ResNet-18): {torchvision_params}")
    assert (
        classical_params == torchvision_params
    ), f"Expected {torchvision_params} parameters but got {classical_params}"

    print("\n" + "=" * 60)
    print("Custom hidden_channels example")
    print("=" * 60)
    # Example with mixed hidden_channels: int for uniform stages, tuple for per-block
    model_custom = init_full_resnet_structure(
        input_shape=(3, 224, 224),
        out_features=1_000,
        hidden_channels=(32, 64, (100, 150), 256),  # mixed: int and tuple
        number_of_blocks_per_stage=2,
        use_preactivation=True,
    )
    print("Hidden channels per stage/block:")
    for i, stage in enumerate(model_custom.stages):
        block_hidden = [
            b.hidden_neurons
            for b in stage  # type: ignore
            if isinstance(b, Conv2dGrowingBlock)
        ]
        print(f"  Stage {i}: {block_hidden}")
    summary(model_custom, input_size=(1, 3, 224, 224))
