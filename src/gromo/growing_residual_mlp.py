"""
Module to define a two layer block similar to a BasicBlock in ResNet.
"""

import torch
import torch.nn as nn

from gromo.linear_growing_module import LinearAdditionGrowingModule, LinearGrowingModule
from gromo.utils.utils import global_device


all_layer_types = {
    "linear": {"layer": LinearGrowingModule, "addition": LinearAdditionGrowingModule},
}


class GrowingResidualMLP(torch.nn.Module):
    def __init__(
        self,
        input_shape: torch.Size | tuple[int, ...],
        num_features: int,
        num_blocks: int,
        hidden_features: int,
        num_classes: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ) -> None:

        super(GrowingResidualMLP, self).__init__()
        self.input_shape = input_shape
        self.in_features = torch.tensor(input_shape).prod().int().item()
        self.num_features = num_features
        self.hidden_features = hidden_features
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.activation = activation

        # embedding
        # print(f"Embedding: {self.in_features} -> {self.num_features}")
        self.embedding = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.in_features, num_features, device=global_device()),
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
        self.projection = nn.Linear(num_features, num_classes, device=global_device())

        # current updated block
        self.currently_updated_block: GrowingResidualBlock | None = None
        self.currently_updated_block_index: int | None = None

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

    def init_computation(self):
        for block in self.blocks:
            block.init_computation()

    def update_computation(self):
        for block in self.blocks:
            block.update_computation()

    def reset_computation(self):
        for block in self.blocks:
            block.reset_computation()
        self.currently_updated_block = None
        self.currently_updated_block_index = None

    def delete_update(self):
        for block in self.blocks:
            block.delete_update()
        self.currently_updated_block = None
        self.currently_updated_block_index = None

    def compute_optimal_update(
        self,
        part: str = "all",
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        for block in self.blocks:
            block.compute_optimal_update(
                part=part,
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                dtype=dtype,
            )

    def select_best_update(self, verbose: bool = False) -> int:
        first_order_improvement_values = [
            block.first_order_improvement for block in self.blocks
        ]
        max_improvement = max(first_order_improvement_values)
        for i, block in enumerate(self.blocks):
            if verbose:
                print(f"Block {i} improvement: {block.first_order_improvement}")
                print(
                    f"Block {i} parameter improvement: {block.parameter_update_decrease}"
                )
                print(f"Block {i} eigenvalues extension: {block.eigenvalues}")
            if block.first_order_improvement < max_improvement:
                if verbose:
                    print(f"Deleting block {i}")
                block.delete_update()
            else:
                self.currently_updated_block = block
                self.currently_updated_block_index = i
        return self.currently_updated_block_index

    def select_update(self, block_index: int, verbose: bool = False) -> int:
        for i, block in enumerate(self.blocks):
            if verbose:
                print(f"Block {i} improvement: {block.first_order_improvement}")
                print(
                    f"Block {i} parameter improvement: {block.parameter_update_decrease}"
                )
                print(f"Block {i} eigenvalues extension: {block.eigenvalues}")
            if i != block_index:
                if verbose:
                    print(f"Deleting block {i}")
                block.delete_update()
            else:
                self.currently_updated_block = block
                self.currently_updated_block_index = i
        return self.currently_updated_block_index

    @property
    def first_order_improvement(self) -> torch.Tensor:
        return self.currently_updated_block.first_order_improvement

    def apply_change(self) -> None:
        for block in self.blocks:
            block.apply_change()

    def number_of_parameters(self):
        num_param = sum(p.numel() for p in self.embedding.parameters())
        for block in self.blocks:
            num_param += block.number_of_parameters()
        num_param += sum(p.numel() for p in self.projection.parameters())
        return num_param
        # return sum(p.numel() for p in self.parameters())

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
        for i, block in enumerate(self.blocks):
            information[i] = block.update_information()
        return information


class GrowingResidualBlock(torch.nn.Module):
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
        activation: torch.nn.Module | None = None,
        name: str = "block",
        kwargs_layer: dict | None = None,
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        num_features: int
            number of input and output features, in cas of convolutional layer, the number of channels
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        layer_type: str
            type of layer to use either "linear" or "conv"
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (e.g. bias, ...)
        """
        if kwargs_layer is None:
            kwargs_layer = {}

        super(GrowingResidualBlock, self).__init__()
        self.name = name

        self.norm = nn.LayerNorm(
            num_features, elementwise_affine=False, device=global_device()
        )
        self.activation: torch.nn.Module = activation
        self.first_layer = LinearGrowingModule(
            num_features,
            hidden_features,
            post_layer_function=activation,
            name=f"first_layer",
            **kwargs_layer,
        )
        self.second_layer = LinearGrowingModule(
            hidden_features,
            num_features,
            post_layer_function=torch.nn.Identity(),
            previous_module=self.first_layer,
            name=f"second_layer",
            **kwargs_layer,
        )

        self.enable_extended_forward = False

        # self.activation_derivative = torch.func.grad(mid_activation)(torch.tensor(1e-5))
        # TODO: FIX this
        self.activation_derivative = 1

    def __setattr__(self, key, value):
        if key in [
            "scaling_factor",
            "eigenvalues_extension",
            "parameter_update_decrease",
            "first_order_improvement",
        ]:
            self.second_layer.__setattr__(key, value)
        else:
            nn.Module.__setattr__(self, key, value)

    @property
    def hidden_features(self):
        return self.second_layer.in_features

    @property
    def scaling_factor(self):
        return self.second_layer.scaling_factor

    @property
    def eigenvalues_extension(self):
        return self.second_layer.eigenvalues_extension

    @property
    def parameter_update_decrease(self):
        return self.second_layer.parameter_update_decrease

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        return self.second_layer.first_order_improvement

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        if self.hidden_features > 0:
            x = self.norm(x)
            y = self.activation(x)
            y = self.first_layer(y)
            y = self.second_layer(y)
            x = y + x
        return x

    # @property
    # def in_activity(self) -> torch.Tensor:
    #     """
    #     Get the input activity of the block.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         input activity
    #     """
    #     return self.first_layer.input
    #
    # def set_store_in_activity(self, value: bool):
    #     """
    #     Set the store_in_activity parameter of the block.
    #     If True, the block will store the activity after the first activation
    #     function.
    #
    #     Parameters
    #     ----------
    #     value: bool
    #         value to set
    #     """
    #     self.first_layer.store_input = True

    def init_computation(self):
        """
        Initialise the computation of the block.
        """
        self.first_layer.init_computation()
        self.second_layer.init_computation()

    def update_computation(self, desired_activation: torch.Tensor | None = None):
        """
        Update the computation of the block.

        Parameters
        ----------
        desired_activation: torch.Tensor
            desired direction of the output variation of the block
        """
        self.first_layer.update_computation()
        self.second_layer.update_computation()

    def reset_computation(self):
        """
        Reset the computation of the block.
        """
        self.first_layer.reset_computation()
        self.second_layer.reset_computation()

    def delete_update(self):
        """
        Delete the update of the block.
        """
        self.first_layer.delete_update()
        self.second_layer.delete_update(include_previous=True)

    def compute_optimal_update(
        self,
        part: str = "all",
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute the optimal update for second layer and additional neurons.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        """
        assert part in [
            "all",
            "parameter",
            "neuron",
        ], f"{part=} should be in ['all', 'parameter', 'neuron']"

        if part == "parameter":
            _, _, _ = self.second_layer.compute_optimal_delta(dtype=dtype)
            # _, _, _ = self.second_layer.compute_optimal_delta(dtype=dtype)
        elif part == "neuron":
            _, _ = self.second_layer.compute_optimal_updates(
                zero_delta=True,
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                dtype=dtype,
            )
            self.second_layer.optimal_delta_layer = None
            self.second.layer.parameter_update_decrease = 0
        elif part == "all":
            _, _ = self.second_layer.compute_optimal_updates(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                update_previous=True,
                dtype=dtype,
            )

    def apply_change(self) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        self.second_layer.apply_change(apply_previous=True)
        # self.hidden_features += self.eigenvalues_extension.shape[0]

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int,
    ) -> None:
        """
        Select the first keep_neurons neurons of the optimal added parameters.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        """
        self.eigenvalues = self.eigenvalues[:keep_neurons]
        self.second_layer.sub_select_optimal_added_parameters(
            keep_neurons, sub_select_previous=True
        )

    def number_of_parameters(self):
        num_param = self.first_layer.number_of_parameters()
        num_param += self.second_layer.number_of_parameters()
        return num_param

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

    def weights_statistics(self) -> dict[int, dict[str, dict[str, float] | int]]:
        statistics = {}
        for i, layer in enumerate([self.first_layer, self.second_layer]):
            statistics[i] = {
                "weight": self.tensor_statistics(layer.weight),
            }
            if layer.bias is not None:
                statistics[i]["bias"] = self.tensor_statistics(layer.bias)
            statistics[i]["input_shape"] = layer.num_features
            statistics[i]["output_shape"] = layer.out_features
        return statistics

    def update_information(self):
        layer_information = dict()
        layer_information["update_value"] = self.first_order_improvement
        layer_information["parameter_improvement"] = self.parameter_update_decrease
        layer_information["eigenvalues_extension"] = self.eigenvalues_extension
        return layer_information
