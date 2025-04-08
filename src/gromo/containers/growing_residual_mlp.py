"""
Module to define a two layer block similar to a BasicBlock in ResNet.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


class GrowingResidualBlock(GrowingContainer):
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
        activation: Optional[nn.Module] = None,
        name: str = "block",
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
        activation : Optional[nn.Module]
            Activation function to use, if None use the identity function.
        name : str
            Name of the block.
        kwargs_layer : Optional[Dict[str, Any]]
            Dictionary of arguments for the layers (e.g., bias, ...).
        """
        if kwargs_layer is None:
            kwargs_layer = {}

        super().__init__(in_features=num_features, out_features=num_features)
        self.name = name
        self.num_features = num_features
        self.hidden_features = hidden_features

        self.norm = nn.LayerNorm(
            num_features, elementwise_affine=False, device=self.device
        )
        self.activation = activation if activation is not None else nn.Identity()
        self.first_layer = LinearGrowingModule(
            num_features,
            hidden_features,
            post_layer_function=self.activation,
            name="first_layer",
            **kwargs_layer,
        )
        self.second_layer = LinearGrowingModule(
            hidden_features,
            num_features,
            post_layer_function=nn.Identity(),
            previous_module=self.first_layer,
            name="second_layer",
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
        if self.hidden_features > 0:
            x = self.norm(x)
            y = self.activation(x)
            y, y_ext = self.first_layer.extended_forward(y)
            y, _ = self.second_layer.extended_forward(y, y_ext)
            assert (
                _ is None
            ), f"The output of layer 2 {self.second_layer.name} should not be extended."
            del y_ext
            x = y + x
        return x

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
        if self.hidden_features > 0:
            x = self.norm(x)
            y = self.activation(x)
            y = self.first_layer(y)
            y = self.second_layer(y)
            x = y + x
        return x

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
        statistics[0] = {
            "weight": self.tensor_statistics(self.first_layer.weight),
        }
        if self.first_layer.bias is not None:
            statistics[0]["bias"] = self.tensor_statistics(self.first_layer.bias)
        statistics[1] = {
            "weight": self.tensor_statistics(self.second_layer.weight),
        }
        if self.second_layer.bias is not None:
            statistics[1]["bias"] = self.tensor_statistics(self.second_layer.bias)
        statistics["hidden_shape"] = self.hidden_features
        return statistics

    def update_information(self) -> Dict[str, Any]:
        layer_information = {
            "update_value": self.second_layer.first_order_improvement,
            "parameter_improvement": self.second_layer.parameter_update_decrease,
            "eigenvalues_extension": self.second_layer.eigenvalues_extension,
        }
        return layer_information


class GrowingResidualMLP(GrowingContainer):
    def __init__(
        self,
        in_features: torch.Size | tuple[int, ...],
        out_features: int,
        num_features: int,
        hidden_features: int,
        num_blocks: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        device: torch.device = None,
    ) -> None:

        in_features = torch.tensor(in_features).prod().int().item()
        super(GrowingResidualMLP, self).__init__(
            in_features=in_features,
            out_features=out_features,
            device=device,
        )
        self.num_features = num_features
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks

        # embedding
        self.embedding = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, num_features, device=self.device),
        )

        # blocks
        self.blocks = torch.nn.ModuleList(
            [
                GrowingResidualBlock(
                    num_features,
                    hidden_features,
                    activation=activation,
                    name=f"block {i}",
                )
                for i in range(num_blocks)
            ]
        )

        # final projection
        self.projection = nn.Linear(num_features, out_features, device=self.device)
        self.set_growing_layers()

    def set_growing_layers(self):
        self._growing_layers = list(block.second_layer for block in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block.extended_forward(x)
        x = self.projection(x)
        return x

    def select_update(self, layer_index: int, verbose: bool = False) -> int:
        for i, layer in enumerate(self._growing_layers):
            if verbose:
                print(f"Block {i} improvement: {layer.first_order_improvement}")
                print(
                    f"Block {i} parameter improvement: {layer.parameter_update_decrease}"
                )
                print(f"Block {i} eigenvalues extension: {layer.eigenvalues}")
            if i != layer_index:
                if verbose:
                    print(f"Deleting block {i}")
                layer.delete_update()
            else:
                self.currently_updated_layer_index = i
        return self.currently_updated_layer_index

    @staticmethod
    def tensor_statistics(tensor) -> dict[str, float]:
        min_value = tensor.min().item()
        max_value = tensor.max().item()
        mean_value = tensor.mean().item()
        if tensor.numel() > 1:
            std_value = tensor.std().item()
        else:
            std_value = -1
        return {
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
        }

    def weights_statistics(self) -> dict[int, dict[str, dict[str, float]]]:
        statistics = {}
        for i, block in enumerate(self.blocks):
            statistics[i] = {"weight_0": self.tensor_statistics(block.first_layer.weight)}
            if block.first_layer.bias is not None:
                statistics[i]["bias_0"] = self.tensor_statistics(block.first_layer.bias)

            statistics[i]["weight_1"] = self.tensor_statistics(block.second_layer.weight)
            if block.second_layer.bias is not None:
                statistics[i]["bias_1"] = self.tensor_statistics(block.second_layer.bias)

            statistics[i]["hidden_shape"] = block.hidden_features
        return statistics

    def update_information(self):
        information = dict()
        for i, layer in enumerate(self._growing_layers):
            layer_information = dict()
            layer_information["update_value"] = layer.first_order_improvement
            layer_information["parameter_improvement"] = layer.parameter_update_decrease
            layer_information["eigenvalues_extension"] = layer.eigenvalues_extension
            information[i] = layer_information
        return information


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # define the input shape
    input_shape = (3, 32, 32)  # Example for 32x32 RGB image
    # define the number of features
    num_features = 64
    # define the number of hidden features
    hidden_features = 32
    # define the number of output features
    num_classes = 10
    # define the number of blocks
    num_blocks = 4

    # create the growing residual MLP
    model = GrowingResidualMLP(
        in_features=input_shape,
        out_features=num_classes,
        num_features=num_features,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        activation=nn.ReLU(),
    )

    # print the model
    print(model)

    # define the input tensor
    x = torch.rand((2, *input_shape))
    print(f"Input shape: {x.shape}")
    # forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")
