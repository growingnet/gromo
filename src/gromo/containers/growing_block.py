"""
Module to define a two layer block similar to a BasicBlock in ResNet.
"""

import torch

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.growing_module import GrowingModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)


all_layer_types = {
    "linear": {"layer": LinearGrowingModule, "merge": LinearMergeGrowingModule},
    "conv": {"layer": RestrictedConv2dGrowingModule, "merge": None},
}


class GrowingBlock(GrowingContainer):
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
        in_features: int,
        out_features: int,
        hidden_features: int = 0,
        layer_type: str = "linear",
        pre_activation: torch.nn.Module | None = None,
        mid_activation: torch.nn.Module | None = None,
        name: str = "block",
        kwargs_layer: dict | None = None,
        kwargs_first_layer: dict | None = None,
        kwargs_second_layer: dict | None = None,
        downsample: torch.nn.Module = torch.nn.Identity(),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        in_features: int
            number of input features, in cas of convolutional layer, the number of channels
        out_features: int
            number of output features
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        layer_type: str
            type of layer to use either "linear" or "conv"
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        pre_activation: torch.nn.Module | None
            activation function to use before the first layer, if None use the activation function
        mid_activation: torch.nn.Module | None
            activation function to use between the two layers, if None use the activation function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (e.g. bias, ...)
        kwargs_first_layer: dict | None
            dictionary of arguments for the first layer, if None use kwargs_layer
        kwargs_second_layer: dict | None
            dictionary of arguments for the second layer, if None use kwargs_layer
        downsample: torch.nn.Module
            operation to apply on the residual stream
        """
        assert layer_type in all_layer_types, f"Layer type {layer_type} not supported."
        assert (
            in_features == out_features or downsample is not None
        ), "Incompatible dimensions: in_features must match out_features or downsample must be provided."
        super(GrowingBlock, self).__init__(
            in_features=in_features,
            out_features=out_features,
        )
        self.name = name
        self.device = device
        self.hidden_features = hidden_features

        self.pre_activation: torch.nn.Module = pre_activation
        self.first_layer: GrowingModule = None
        self.second_layer: GrowingModule = None
        self.downsample = downsample

        self.enable_extended_forward = False
        self.eigenvalues = None
        self.parameter_update_decrease = None

        # self.activation_derivative = torch.func.grad(mid_activation)(torch.tensor(1e-5))
        # TODO: FIX this
        self.activation_derivative = 1

    @property
    def scaling_factor(self):
        return self.second_layer.scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, value: float):
        """
        Set the scaling factor for the second layer.
        """
        self.second_layer.scaling_factor = value

    @staticmethod
    def set_default_values(
        activation: torch.nn.Module | None = torch.nn.Identity(),
        pre_activation: torch.nn.Module | None = None,
        mid_activation: torch.nn.Module | None = None,
        kwargs_layer: dict | None = None,
        kwargs_first_layer: dict | None = None,
        kwargs_second_layer: dict | None = None,
    ) -> tuple[torch.nn.Module, torch.nn.Module, dict, dict]:
        """
        Set default values for the block.
        """
        if activation is None:
            activation = torch.nn.Identity()
        if pre_activation is None:
            pre_activation = activation
        if mid_activation is None:
            mid_activation = activation
        if kwargs_layer is None:
            kwargs_layer = dict()
        if kwargs_first_layer is None:
            kwargs_first_layer = kwargs_layer
        if kwargs_second_layer is None:
            kwargs_second_layer = kwargs_layer
        return pre_activation, mid_activation, kwargs_first_layer, kwargs_second_layer

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
        identity: torch.Tensor = self.downsample(x)
        x = self.pre_activation(x)
        if self.hidden_features > 0:
            x, x_ext = self.first_layer.extended_forward(x)
            x, _ = self.second_layer.extended_forward(x, x_ext)
            assert (
                _ is None
            ), f"The output of layer 2 {self.second_layer.name} should not be extended."

            return x + identity
        elif self.first_layer.extended_output_layer is not None:
            x = self.scaling_factor * self.first_layer.extended_output_layer(x)
            x = self.first_layer.extended_post_layer_function(x)
            assert (
                self.second_layer.extended_input_layer is not None
            ), f"Second layer {self.second_layer.name} should have an extended output layer."
            x = self.scaling_factor * self.second_layer.extended_input_layer(x)
            x = self.second_layer.extended_post_layer_function(x)

            return x + identity
        else:
            return identity

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
        identity: torch.Tensor = self.downsample(x)
        if self.hidden_features == 0:
            if self.first_layer.store_input:
                self.first_layer._input = self.pre_activation(x).detach()
            if self.second_layer.store_pre_activity:
                self.second_layer._pre_activity = identity
                self.second_layer._pre_activity.retain_grad()
            self.second_layer.tensor_s_growth.updated = False
            self.second_layer.tensor_m_prev.updated = False
            self.second_layer.cross_covariance.updated = False
            return identity
        else:

            out = self.pre_activation(x)
            out = self.first_layer(out)
            out = self.second_layer(out)
            return out + identity

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
        # growth part
        self.first_layer.store_input = True
        self.second_layer.store_pre_activity = True
        self.second_layer.tensor_m_prev.init()
        self.second_layer.tensor_s_growth.init()

        if self.hidden_features > 0:
            self.second_layer.cross_covariance.init()

            # natural gradient part
            self.second_layer.store_input = True
            self.second_layer.tensor_s.init()
            self.second_layer.tensor_m.init()

    def update_computation(self):
        """
        Update the computation of the block.
        """
        # growth part
        self.second_layer.tensor_m_prev.update()
        self.second_layer.tensor_s_growth.update()

        if self.hidden_features > 0:
            self.second_layer.cross_covariance.update()

            # natural gradient part
            self.second_layer.tensor_m.update()
            self.second_layer.tensor_s.update()

    def reset_computation(self):
        """
        Reset the computation of the block.
        """
        self.first_layer.store_input = False
        self.second_layer.store_input = False
        self.second_layer.store_pre_activity = False
        self.second_layer.tensor_s.reset()
        self.second_layer.tensor_m.reset()
        self.second_layer.tensor_m_prev.reset()
        self.second_layer.cross_covariance.reset()
        self.second_layer.tensor_s_growth.reset()

    def delete_update(self):
        """
        Delete the update of the block.
        """
        self.second_layer.optimal_delta_layer = None
        self.second_layer.extended_input_layer = None
        self.first_layer.extended_input_layer = None

    def compute_optimal_updates(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
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
        if self.hidden_features > 0:
            _, _, self.parameter_update_decrease = (
                self.second_layer.compute_optimal_delta()
            )
        alpha, alpha_bias, _, self.eigenvalues = (
            self.second_layer.compute_optimal_added_parameters(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                use_projected_gradient=self.hidden_features > 0,
            )
        )
        self.first_layer.extended_output_layer = self.first_layer.layer_of_tensor(
            alpha, alpha_bias
        )

    def apply_change(self) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        assert self.eigenvalues is not None, "No optimal added parameters computed."
        self.first_layer.apply_change()
        self.second_layer.apply_change()
        self.hidden_features += self.eigenvalues.shape[0]

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
        assert self.eigenvalues is not None, "No optimal added parameters computed."
        self.eigenvalues = self.eigenvalues[:keep_neurons]
        self.second_layer.sub_select_optimal_added_parameters(keep_neurons)

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        assert self.eigenvalues is not None, "No optimal added parameters computed."
        return (
            self.parameter_update_decrease
            + self.activation_derivative * (self.eigenvalues**2).sum()
        )


class LinearGrowingBlock(GrowingBlock):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 0,
        layer_type: str = "linear",
        activation: torch.nn.Module | None = torch.nn.Identity(),
        pre_activation: torch.nn.Module | None = None,
        mid_activation: torch.nn.Module | None = None,
        name: str = "block",
        kwargs_layer: dict | None = None,
        kwargs_first_layer: dict | None = None,
        kwargs_second_layer: dict | None = None,
        downsample: torch.nn.Module = torch.nn.Identity(),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        in_out_features: int
            number of input and output features, in cas of convolutional layer, the number of channels
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        layer_type: str
            type of layer to use either "linear" or "conv"
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        pre_activation: torch.nn.Module | None
            activation function to use before the first layer, if None use the activation function
        mid_activation: torch.nn.Module | None
            activation function to use between the two layers, if None use the activation function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (eg bias, ...)
        kwargs_first_layer: dict | None
            dictionary of arguments for the first layer, if None use kwargs_layer
        kwargs_second_layer: dict | None
            dictionary of arguments for the second layer, if None use kwargs_layer
        """
        super(LinearGrowingBlock, self).__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            layer_type=layer_type,
            pre_activation=pre_activation,
            mid_activation=mid_activation,
            name=name,
            kwargs_layer=kwargs_layer,
            kwargs_first_layer=kwargs_first_layer,
            kwargs_second_layer=kwargs_second_layer,
            downsample=downsample,
            device=device,
        )
        self.pre_activation, mid_activation, kwargs_first_layer, kwargs_second_layer = (
            self.set_default_values(
                activation=activation,
                pre_activation=pre_activation,
                mid_activation=mid_activation,
                kwargs_layer=kwargs_layer,
                kwargs_first_layer=kwargs_first_layer,
                kwargs_second_layer=kwargs_second_layer,
            )
        )
        self.first_layer = LinearGrowingModule(
            in_features=in_features,
            out_features=hidden_features,
            name=f"{name}(first_layer)",
            post_layer_function=mid_activation,
            **kwargs_first_layer,
        )
        self.second_layer = LinearGrowingModule(
            in_features=hidden_features,
            out_features=out_features,
            name=f"{name}(second_layer)",
            previous_module=self.first_layer,
            **kwargs_second_layer,
        )


class RestrictedConv2dGrowingBlock(GrowingBlock):
    """
    RestrictedConv2dGrowingBlock is a GrowingBlock for RestrictedConv2d layers.

    This creates a two-layer block similar to LinearGrowingBlock but using
    RestrictedConv2dGrowingModule layers instead of LinearGrowingModule layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 0,
        activation: torch.nn.Module | None = None,
        pre_activation: torch.nn.Module | None = None,
        mid_activation: torch.nn.Module | None = None,
        extended_mid_activation: torch.nn.Module | None = None,
        name: str = "conv_block",
        kwargs_layer: dict | None = None,
        kwargs_first_layer: dict | None = None,
        kwargs_second_layer: dict | None = None,
        downsample: torch.nn.Module = torch.nn.Identity(),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialise the convolutional block.

        Parameters
        ----------
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        hidden_channels: int
            number of hidden channels, if zero the block is the zero function
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        pre_activation: torch.nn.Module | None
            activation function to use before the first layer, if None use the activation function
        mid_activation: torch.nn.Module | None
            activation function to use between the two layers, if None use the activation function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (e.g. use_bias, ...)
        kwargs_first_layer: dict | None
            dictionary of arguments for the first layer, if None use kwargs_layer
        kwargs_second_layer: dict | None
            dictionary of arguments for the second layer, if None use kwargs_layer
        downsample: torch.nn.Module
            operation to apply on the residual stream
        """
        super(RestrictedConv2dGrowingBlock, self).__init__(
            in_features=in_channels,
            out_features=out_channels,
            hidden_features=hidden_channels,
            layer_type="conv",  # This would need to be added to all_layer_types
            pre_activation=pre_activation,
            mid_activation=mid_activation,
            name=name,
            kwargs_layer=kwargs_layer,
            kwargs_first_layer=kwargs_first_layer,
            kwargs_second_layer=kwargs_second_layer,
            downsample=downsample,
            device=device,
        )

        self.pre_activation, mid_activation, kwargs_first_layer, kwargs_second_layer = (
            self.set_default_values(
                activation=activation,
                pre_activation=pre_activation,
                mid_activation=mid_activation,
                kwargs_layer=kwargs_layer,
                kwargs_first_layer=kwargs_first_layer,
                kwargs_second_layer=kwargs_second_layer,
            )
        )
        if extended_mid_activation is None:
            extended_mid_activation = mid_activation

        # Set default values for conv layer kwargs
        if kwargs_first_layer is None:
            kwargs_first_layer = {}
        if kwargs_second_layer is None:
            kwargs_second_layer = {}

        self.first_layer = RestrictedConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            name=f"{name}(first_layer)",
            post_layer_function=mid_activation,
            extended_post_layer_function=extended_mid_activation,
            device=device,
            **kwargs_first_layer,
        )
        self.second_layer = RestrictedConv2dGrowingModule(
            in_channels=hidden_channels,
            out_channels=out_channels,
            name=f"{name}(second_layer)",
            previous_module=self.first_layer,
            device=device,
            **kwargs_second_layer,
        )
