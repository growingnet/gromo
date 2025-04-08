from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


class GrowingMLPBlock(GrowingContainer):
    """
    Represents a block of a growing network.

    Sequence of layers:
    - Activation pre
    - Layer first
    - Activation mid
    - Layer second
    """

    def __init__(
        self,
        num_features: int,
        hidden_features: int = 0,
        dropout: float = 0.0,
        name: Optional[str] = None,
        kwargs_layer: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the block.

        Parameters
        ----------
        num_features : int
            Number of input and output features, in case of convolutional layer, the number of channels.
        hidden_features : int
            Number of hidden features, if zero the block is the zero function.
        dropout : float
            Dropout rate.
        name : Optional[str]
            Name of the block.
        kwargs_layer : Optional[Dict[str, Any]]
            Dictionary of arguments for the layers (e.g., bias, ...).
        """
        if kwargs_layer is None:
            kwargs_layer = {}

        super().__init__(in_features=num_features, out_features=num_features)
        self.name = name

        self.first_layer = LinearGrowingModule(
            num_features,
            hidden_features,
            post_layer_function=nn.GELU(),
            **kwargs_layer,
        )
        self.dropout = nn.Dropout(dropout)
        self.second_layer = LinearGrowingModule(
            hidden_features,
            num_features,
            post_layer_function=nn.Identity(),
            previous_module=self.first_layer,
            **kwargs_layer,
        )
        self.enable_extended_forward = False

        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        self._growing_layers = [self.second_layer]

    def extended_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        y, y_ext = self.first_layer.extended_forward(x)
        y = self.dropout(y)
        if y_ext is not None:
            y_ext = self.dropout(y_ext)
        y, _ = self.second_layer.extended_forward(y, y_ext)

        assert (
            _ is None
        ), f"The output of layer 2 {self.second_layer.name} should not be extended."
        del y_ext

        y = self.dropout(y)
        return y

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        y = self.first_layer(x)
        y = self.dropout(y)
        y = self.second_layer(y)
        y = self.dropout(y)
        return y

    @staticmethod
    def tensor_statistics(tensor: Tensor) -> Dict[str, float]:
        min_value = tensor.min().item()
        max_value = tensor.max().item()
        mean_value = tensor.mean().item()
        std_value = tensor.std().item() if tensor.numel() > 1 else -1
        return {
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
        }

    def weights_statistics(self) -> Dict[int, Dict[str, Any]]:
        statistics = {}
        for i, layer in enumerate([self.first_layer, self.second_layer]):
            if layer.weight.numel() == 0:
                continue
            statistics[i] = {
                "weight": self.tensor_statistics(layer.weight),
            }
            if layer.bias is not None:
                statistics[i]["bias"] = self.tensor_statistics(layer.bias)
            statistics[i]["input_shape"] = layer.in_features
            statistics[i]["output_shape"] = layer.out_features
        return statistics

    def update_information(self) -> Dict[str, Any]:
        layer_information = {
            "update_value": self.second_layer.first_order_improvement,
            "parameter_improvement": self.second_layer.parameter_update_decrease,
            "eigenvalues_extension": self.second_layer.eigenvalues_extension,
            "scaling_factor": self.second_layer.scaling_factor,
            "added_neurons": (
                0
                if self.second_layer.eigenvalues_extension is None
                else self.second_layer.eigenvalues_extension.size(0)
            ),
        }
        return layer_information


class GrowingTokenMixer(GrowingContainer):
    """
    Represents a token mixer in a growing network.
    """

    def __init__(
        self,
        num_patches: int,
        num_features: int,
        hidden_features: int,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the token mixer.

        Parameters
        ----------
        num_patches : int
            Number of patches.
        num_features : int
            Number of features.
        hidden_features : int
            Number of hidden features.
        dropout : float
            Dropout rate.
        name : Optional[str]
            Name of the token mixer.
        """
        super().__init__(in_features=num_features, out_features=num_features)
        self.norm = nn.LayerNorm(num_features, device=self.device)
        self.mlp = GrowingMLPBlock(num_patches, hidden_features, dropout)
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        self._growing_layers = self.mlp._growing_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the token mixer.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        out = x + residual
        return out

    def extended_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the token mixer with the current modifications.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        residual = x
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.mlp.extended_forward(y)
        y = y.transpose(1, 2)
        out = y + residual
        return out

    def weights_statistics(self) -> Dict[int, Dict[str, Any]]:
        return self.mlp.weights_statistics()

    def update_information(self) -> Dict[str, Any]:
        return self.mlp.update_information()


class GrowingChannelMixer(GrowingContainer):
    """
    Represents a channel mixer in a growing network.
    """

    def __init__(
        self,
        num_features: int,
        hidden_features: int,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the channel mixer.

        Parameters
        ----------
        num_features : int
            Number of features.
        hidden_features : int
            Number of hidden features.
        dropout : float
            Dropout rate.
        name : Optional[str]
            Name of the channel mixer.
        """
        super().__init__(in_features=num_features, out_features=num_features)
        self.norm = nn.LayerNorm(num_features, device=self.device)
        self.mlp = GrowingMLPBlock(num_features, hidden_features, dropout)
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        self._growing_layers = self.mlp._growing_layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the channel mixer.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        out = x + residual
        return out

    def extended_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the channel mixer with the current modifications.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        residual = x
        x = self.norm(x)
        x = self.mlp.extended_forward(x)
        out = x + residual
        return out

    def weights_statistics(self) -> Dict[int, Dict[str, Any]]:
        return self.mlp.weights_statistics()

    def update_information(self) -> Dict[str, Any]:
        return self.mlp.update_information()


class GrowingMixerLayer(GrowingContainer):
    """
    Represents a mixer layer in a growing network.
    """

    def __init__(
        self,
        num_patches: int,
        num_features: int,
        hidden_dim_token: int,
        hidden_dim_channel: int,
        dropout: float,
        name: Optional[str] = "Mixer Layer",
    ) -> None:
        """
        Initialize the mixer layer.

        Parameters
        ----------
        num_patches : int
            Number of patches.
        num_features : int
            Number of features.
        hidden_dim_token : int
            Number of hidden token features.
        hidden_dim_channel : int
            Number of hidden channel features.
        dropout : float
            Dropout rate.
        name : Optional[str]
            Name of the mixer layer.
        """
        super().__init__(in_features=num_features, out_features=num_features)
        self.token_mixer = GrowingTokenMixer(
            num_patches, num_features, hidden_dim_token, dropout
        )
        self.channel_mixer = GrowingChannelMixer(
            num_features, hidden_dim_channel, dropout
        )
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        self._growing_layers = list()
        self._growing_layers.extend(self.token_mixer._growing_layers)
        self._growing_layers.extend(self.channel_mixer._growing_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the mixer layer.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

    def extended_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the mixer layer with the current modifications.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        x = self.token_mixer.extended_forward(x)
        x = self.channel_mixer.extended_forward(x)
        return x

    def weights_statistics(self) -> Dict[int, Dict[str, Any]]:
        statistics = {}
        statistics[0] = self.token_mixer.weights_statistics()
        statistics[1] = self.channel_mixer.weights_statistics()
        return statistics

    def update_information(self) -> Dict[str, Any]:
        layer_information = {
            "token_mixer": self.token_mixer.update_information(),
            "channel_mixer": self.channel_mixer.update_information(),
        }
        return layer_information


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches**2
    return num_patches


class GrowingMLPMixer(GrowingContainer):
    """
    Represents a growing MLP mixer network.
    """

    def __init__(
        self,
        in_features: tuple[int, int, int] = (3, 32, 32),
        out_features: int = 10,
        patch_size: int = 4,
        num_features: int = 128,
        hidden_dim_token: int = 64,
        hidden_dim_channel: int = 512,
        num_blocks: int = 8,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the growing MLP mixer.

        Parameters
        ----------
        in_features : tuple[int, int, int]
            Input features (channels, height, width).
        out_features : int
            Number of output features.
        patch_size : int
            Size of each patch.
        num_features : int
            Number of features.
        hidden_dim_token : int
            Number of hidden token features.
        hidden_dim_channel : int
            Number of hidden channel features.
        num_blocks : int
            Number of mixer blocks.
        dropout : float
            Dropout rate.
        device : Optional[torch.device]
            Device to use for computation.
        """
        in_channels, image_size, _ = in_features
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(
            in_features=torch.tensor(in_features).prod().int().item(),
            out_features=out_features,
        )
        self.device = device
        self.patcher = nn.Conv2d(
            in_channels,
            num_features,
            kernel_size=patch_size,
            stride=patch_size,
            device=self.device,
        )
        self.mixers: nn.ModuleList[GrowingMixerLayer] = nn.ModuleList(
            [
                GrowingMixerLayer(
                    num_patches,
                    num_features,
                    hidden_dim_token,
                    hidden_dim_channel,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(num_features, self.out_features, device=self.device)
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        self._growing_layers = list()
        for mixer in self.mixers:
            self._growing_layers.append(mixer.token_mixer.mlp.second_layer)
            self._growing_layers.append(mixer.channel_mixer.mlp.second_layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the growing MLP mixer.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1).view(batch_size, -1, num_features)
        embedding = patches
        for mixer in self.mixers:
            embedding = mixer(embedding)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

    def extended_forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the growing MLP mixer with the current modifications.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1).view(batch_size, -1, num_features)
        embedding = patches
        for mixer in self.mixers:
            embedding = mixer.extended_forward(embedding)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

    def weights_statistics(self) -> Dict[int, Dict[str, Any]]:
        statistics = {}
        for i, mixer in enumerate(self.mixers):
            statistics[i] = mixer.weights_statistics()
        return statistics

    def update_information(self) -> Dict[str, Any]:
        model_information = {}
        for i, mixer in enumerate(self.mixers):
            model_information[i] = mixer.update_information()
        return model_information


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # define the input shape
    img_size = 32
    input_shape = (3, img_size, img_size)
    # define the patch size
    patch_size = 4
    # define the number of features
    num_features = 32
    # define the number of hidden token features
    hidden_token_features = 8
    # define the number of hidden channel features
    hidden_channel_features = 16
    # define the number of output features
    num_classes = 10
    # define the number of blocks
    num_blocks = 1

    # create the growing residual MLP
    model = GrowingMLPMixer(
        in_features=input_shape,
        out_features=num_classes,
        patch_size=patch_size,
        num_features=num_features,
        hidden_dim_token=hidden_token_features,
        hidden_dim_channel=hidden_channel_features,
        num_blocks=num_blocks,
    )

    # print the model
    print(model)

    # define the input tensor
    x = torch.rand((2, *input_shape))
    print(f"Input shape: {x.shape}")
    # forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")
