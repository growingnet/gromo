"""
Module to create a ResNet with basic blocks (like ResNet-18 or ResNet-34).
Will allow to extend the basic blocks with more intermediate channels
and to add basic blocks add the end of the stages.
"""

import torch
from torch import nn

from gromo.containers.growing_block import RestrictedConv2dGrowingBlock
from gromo.containers.sequential_growing_container import SequentialGrowingContainer
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
        reduction_factor: float = 0.0,
        small_inputs: bool = False,
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
        reduction_factor : float
            Factor to reduce the number of channels in the bottleneck.
            If 0, starts with no channels. If 1, starts with all channels.
        small_inputs : bool
            If True, adapt the network for small input images (e.g., CIFAR-10/100).
            This uses smaller kernels, no stride, and
            no max pooling in the initial layers.
        """
        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )
        self.activation = activation.to(device)
        self.small_inputs = small_inputs
        self.reduction_factor = reduction_factor
        inplanes = 64

        if small_inputs:
            # For small inputs like CIFAR-10/100 (32x32)
            # Use 3x3 conv with stride=1, no max pooling
            self.pre_net = nn.Sequential(
                nn.Conv2d(
                    in_features,
                    inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=self.device,
                ),
                # nn.BatchNorm2d(inplanes, device=self.device),
                # self.activation,
            )
        else:
            # For large inputs like ImageNet (224x224)
            # Use 7x7 conv with stride=2, followed by max pooling
            self.pre_net = nn.Sequential(
                nn.Conv2d(
                    in_features,
                    inplanes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                    device=self.device,
                ),
                # nn.BatchNorm2d(inplanes, device=self.device),
                # self.activation,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.stages: nn.ModuleList = nn.ModuleList()
        nb_stages = 4
        for i in range(nb_stages):
            # for the future we could remove the basic block of the first stage
            # as there is no dowsampling
            stage = nn.Sequential()
            input_channels = inplanes * (2 ** max(0, i - 1))
            output_channels = inplanes * (2**i)
            hidden_channels = int(inplanes * (2**i) * self.reduction_factor)

            # For small inputs, adjust stride behavior
            # Skip stride=2 for the first stage to preserve spatial resolution
            stage_stride = 2 if (i > 0 and not (small_inputs and i == 1)) else 1
            if small_inputs and i == 1:
                # For small inputs, we might want stride=1 for the first downsampling
                # stage to avoid losing too much spatial resolution too quickly
                stage_stride = 1

            stage.append(
                RestrictedConv2dGrowingBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    hidden_channels=hidden_channels,
                    kwargs_first_layer={
                        "kernel_size": input_block_kernel_size,
                        "padding": 1,
                        "use_bias": False,
                        "stride": stage_stride,
                    },
                    kwargs_layer={
                        "kernel_size": output_block_kernel_size,
                        "padding": 1,
                        "use_bias": False,
                    },
                    pre_activation=nn.Sequential(
                        nn.BatchNorm2d(input_channels, device=self.device),
                        self.activation,
                    ),
                    mid_activation=nn.Sequential(
                        GrowingBatchNorm2d(hidden_channels, device=self.device),
                        self.activation,
                    ),
                    extended_mid_activation=self.activation,
                    name=f"Stage {i} Block 0",
                    downsample=(
                        nn.Sequential(
                            nn.BatchNorm2d(input_channels, device=self.device),
                            self.activation,
                            nn.Conv2d(
                                in_channels=input_channels,
                                out_channels=output_channels,
                                kernel_size=1,
                                stride=stage_stride,
                                bias=False,
                                device=self.device,
                            ),
                        )
                        if i > 0
                        else torch.nn.Identity()
                    ),
                    device=self.device,
                )
            )
            self.stages.append(stage)

        self.post_net = nn.Sequential(
            nn.BatchNorm2d(inplanes * (2 ** (nb_stages - 1)), device=self.device),
            self.activation,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(
                inplanes * (2 ** (nb_stages - 1)), out_features, device=self.device
            ),
        )

        # Initialize the growing layers list with all
        # RestrictedConv2dGrowingBlock instances
        self._growing_layers = []
        for stage in self.stages:  # type: ignore
            stage: nn.Sequential
            for block in stage:  # type: ignore
                block: RestrictedConv2dGrowingBlock
                self._growable_layers.append(block)

    def append_block(
        self,
        stage_index: int = 0,
        input_block_kernel_size: int = 3,
        output_block_kernel_size: int = 3,
        hidden_channels: int = 0,
    ) -> None:
        """
        Append a new block to the specified stage of the ResNet.
        """
        if stage_index < 0 or stage_index >= len(self.stages):
            raise IndexError(
                f"Stage {stage_index} is out of range. "
                f"There are {len(self.stages)} stages."
            )
        stage: nn.Sequential = self.stages[stage_index]  # type: ignore
        input_channels = stage[-1].out_features
        output_channels = input_channels
        new_block = RestrictedConv2dGrowingBlock(
            in_channels=input_channels,
            out_channels=output_channels,
            hidden_channels=hidden_channels,
            kwargs_first_layer={
                "kernel_size": input_block_kernel_size,
                "padding": 1,
                "use_bias": False,
                "stride": 1,
            },
            kwargs_layer={
                "kernel_size": output_block_kernel_size,
                "padding": 1,
                "use_bias": False,
            },
            pre_activation=nn.Sequential(
                nn.BatchNorm2d(input_channels, device=self.device),
                self.activation,
            ),
            mid_activation=nn.Sequential(
                GrowingBatchNorm2d(hidden_channels, device=self.device),
                self.activation,
            ),
            extended_mid_activation=self.activation,
            name=f"Stage {stage_index} Block {len(stage)}",
            device=self.device,
        )
        stage.append(new_block)
        # Add the new block to the growing layers list
        self._growable_layers.append(new_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        for stage in self.stages:
            x = stage(x)
        x = self.post_net(x)
        return x

    def extended_forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: torch.Tensor
    ) -> torch.Tensor:
        x = self.pre_net(x)
        for stage in self.stages:  # type: ignore
            stage: nn.Sequential
            for block in stage:  # type: ignore
                block: RestrictedConv2dGrowingBlock
                x = block.extended_forward(x)
        x = self.post_net(x)
        return x

    def number_of_neurons_to_add(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, growth_step=1
    ) -> int:
        """Get the number of neurons to add in the next growth step."""
        layer = self._growable_layers[self.layer_to_grow_index]
        return (
            layer.out_features - int(layer.out_features * self.reduction_factor)
        ) // growth_step


def init_full_resnet_structure(
    input_shape: tuple[int, int, int] = (3, 224, 224),
    in_features: int | None = None,
    out_features: int = 1000,
    device: torch.device | str | None = None,
    activation: nn.Module = nn.ReLU(),
    input_block_kernel_size: int = 3,
    output_block_kernel_size: int = 3,
    reduction_factor: float = 1 / 64,
    small_inputs: bool | None = None,
    number_of_blocks_per_stage: int | tuple[int, int, int, int] = 2,
) -> ResNetBasicBlock:
    """
    Initialize a ResNet-18 model with basic blocks.
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
    small_inputs : bool | None
        If True, adapt the network for small input images (e.g., CIFAR-10/100).
        This uses smaller kernels, no stride, and no max pooling in the initial layers.
    number_of_blocks_per_stage : int | tuple[int, int, int, int]
        Number of basic blocks per stage. If an integer is provided, the same number
        of blocks will be used for all four stages. If a tuple is provided, it should
        contain four integers specifying the number of blocks for each stage.
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

    model = ResNetBasicBlock(
        in_features=in_features,
        out_features=out_features,
        device=device,
        activation=activation,
        input_block_kernel_size=input_block_kernel_size,
        output_block_kernel_size=output_block_kernel_size,
        reduction_factor=reduction_factor,
        small_inputs=small_inputs,
    )
    if (
        isinstance(number_of_blocks_per_stage, (list, tuple))
        and len(number_of_blocks_per_stage) == 4
    ):
        blocks_per_stage = number_of_blocks_per_stage
    elif isinstance(number_of_blocks_per_stage, int):
        blocks_per_stage = (number_of_blocks_per_stage,) * 4
    else:
        raise TypeError(
            "number_of_blocks_per_stage must be an int or a tuple of four ints."
        )
    # Append additional blocks to match ResNet-18 architecture
    for stage_index in range(4):
        for _ in range(1, blocks_per_stage[stage_index]):
            model.append_block(
                stage_index=stage_index,
                input_block_kernel_size=input_block_kernel_size,
                output_block_kernel_size=output_block_kernel_size,
                hidden_channels=int(model.stages[stage_index][0].hidden_features),  # type: ignore
            )
    return model
