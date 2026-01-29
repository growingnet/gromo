from typing import Any

import torch
from torch import Tensor, nn

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.linear_growing_module import LinearGrowingModule


class GrowingMLP(GrowingContainer):
    """
    Represents a growing MLP network.

    Parameters
    ----------
    in_features : int | list[int] | tuple[int, ...]
        Number of input features.
    out_features : int
        Number of output features.
    hidden_size : int
        Size of hidden layers.
    number_hidden_layers : int
        Number of hidden layers.
    activation : nn.Module
        Activation function.
    use_bias : bool
        Whether to use bias in layers.
    flatten : bool
        Whether to flatten the input before passing it through the network.
    device : torch.device | None, optional
        Device to use for computation, by default None.

    Raises
    ------
    TypeError
        if input features are not of type int, list or tuple
    """

    def __init__(
        self,
        in_features: int | list[int] | tuple[int, ...],
        out_features: int,
        hidden_size: int,
        number_hidden_layers: int,
        activation: nn.Module = nn.SELU(),
        use_bias: bool = True,
        flatten: bool = True,
        device: torch.device | None = None,
    ) -> None:
        if isinstance(in_features, int):
            pass
        elif isinstance(in_features, (list, tuple)):
            if flatten:
                in_features = int(torch.tensor(in_features).prod().item())
            else:
                in_features = in_features[-1]
        else:
            raise TypeError(
                f"Expected in_features to be int, list, or tuple, got {type(in_features)}"
            )
        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )

        # Flatten input
        self.flatten = nn.Flatten(start_dim=1) if flatten else nn.Identity()
        self.layers = nn.ModuleList()
        self.layers.append(
            LinearGrowingModule(
                self.in_features,
                hidden_size,
                post_layer_function=activation,
                use_bias=use_bias,
                name="Layer 0",
                device=self.device,
            )
        )
        for i in range(number_hidden_layers - 1):
            self.layers.append(
                LinearGrowingModule(
                    hidden_size,
                    hidden_size,
                    post_layer_function=activation,
                    previous_module=self.layers[-1],  # type: ignore
                    use_bias=use_bias,
                    name=f"Layer {i + 1}",
                    device=self.device,
                )
            )
        self.layers.append(
            LinearGrowingModule(
                hidden_size,
                self.out_features,
                previous_module=self.layers[-1],  # type: ignore
                use_bias=use_bias,
                name=f"Layer {number_hidden_layers}",
                device=self.device,
            )
        )

        self.set_growing_layers()

    def set_growing_layers(self, index: int | None = None) -> None:
        """Reference all growable layers of the model in the _growing_layers private attribute"""
        if index is not None:
            self._growing_layers = [self.layers[index]]  # type: ignore
        else:
            self._growing_layers = list(self.layers[1:])  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the growing MLP.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def extended_forward(self, x: Tensor) -> Tensor:  # type: ignore
        """
        Forward pass of the growing MLP with the current modifications.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        x = self.flatten(x)
        x_ext = None
        for layer in self.layers:
            x, x_ext = layer.extended_forward(x, x_ext)
        return x

    def update_information(self) -> dict[str, Any]:
        """Update information for all growing layers including first order improvement

        Returns
        -------
        dict[str, Any]
            information dictionary
        """
        information = {}
        for i, layer in enumerate(self._growing_layers):
            layer_information = {
                "update_value": layer.first_order_improvement,
                "parameter_improvement": layer.parameter_update_decrease,
                "eigenvalues_extension": layer.eigenvalues_extension,
            }
            information[i] = layer_information
        return information

    def normalise(self, verbose: bool = False) -> None:
        """Normalize the weight of the model

        Parameters
        ----------
        verbose : bool, optional
            print info, by default False
        """
        max_values = torch.zeros(len(self.layers), device=self.device)
        for i, layer in enumerate(self.layers):
            max_values[i] = layer.weight.abs().max()
        normalisation = self.normalisation_factor(max_values)
        if verbose:
            print(f"Normalisation: {list(enumerate(normalisation))}")
        current_normalisation = torch.ones(1, device=self.device)
        for i, layer in enumerate(self.layers):
            layer.weight.data = layer.weight.data * normalisation[i]
            current_normalisation *= normalisation[i]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data * current_normalisation

    @staticmethod
    def normalisation_factor(values: Tensor) -> Tensor:
        """
        Compute normalisation factor for the values in the tensor.

        Parameters
        ----------
        values : Tensor
            Values to be normalised.

        Returns
        -------
        Tensor
            Normalisation factors.
        """
        normalisation = values.prod().pow(1 / values.numel())
        return normalisation.repeat(values.shape) / values

    def __str__(self) -> str:
        return "\n".join(str(layer) for layer in self.layers)

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, item: int) -> LinearGrowingModule:
        assert (
            0 <= item < len(self.layers)
        ), f"{item=} should be in [0, {len(self.layers)})"
        return self.layers[item]  # type: ignore


class Perceptron(GrowingMLP):
    """Represents a Perceptron MLP

    Parameters
    ----------
    in_features : int
        input features
    hidden_feature : int
        hidden features
    out_features : int
        output features
    activation : nn.Module, optional
        activation function, by default nn.Sigmoid()
    use_bias : bool, optional
        use bias, by default True
    flatten : bool, optional
        flatten the input, by default True
    device : torch.device | None, optional
        default device, by default None
    """

    def __init__(
        self,
        in_features: int,
        hidden_feature: int,
        out_features: int,
        activation: nn.Module = nn.Sigmoid(),
        use_bias: bool = True,
        flatten: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_feature,
            number_hidden_layers=1,
            activation=activation,
            use_bias=use_bias,
            flatten=flatten,
            device=device,
        )


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # define the input shape
    input_features = 784  # Example for flattened 28x28 image
    # define the number of hidden features
    hidden_size = 128
    # define the number of output features
    num_classes = 10
    # define the number of hidden layers
    number_hidden_layers = 3

    # create the growing MLP
    model = GrowingMLP(
        in_features=input_features,
        out_features=num_classes,
        hidden_size=hidden_size,
        number_hidden_layers=number_hidden_layers,
        activation=nn.ReLU(),
    )

    # print the model
    print(model)

    # define the input tensor
    x = torch.rand((2, input_features))
    print(f"Input shape: {x.shape}")
    # forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")
