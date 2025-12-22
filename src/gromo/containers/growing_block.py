"""
Module to define a two layer block similar to a BasicBlock in ResNet.
"""

from typing import Any
from warnings import warn

import torch

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.growing_module import GrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule


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
        first_layer: GrowingModule,
        second_layer: GrowingModule,
        in_features: int,
        out_features: int,
        pre_activation: torch.nn.Module = torch.nn.Identity(),
        name: str = "block",
        downsample: torch.nn.Module = torch.nn.Identity(),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        first_layer: GrowingModule
            first layer of the block
        second_layer: GrowingModule
            second layer of the block
        in_features: int
            number of input features, in case of convolutional layer,
            the number of channels
        out_features: int
            number of output features
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        pre_activation: torch.nn.Module
            activation function to use before the first layer
        name: str
            name of the block
        downsample: torch.nn.Module
            operation to apply on the residual stream
        device: torch.device | None
            device to use for the block
        """
        assert in_features == out_features or not isinstance(
            downsample, torch.nn.Identity
        ), (
            f"Incompatible dimensions: in_features ({in_features}) must match "
            f"out_features ({out_features}) or downsample ({downsample}) "
            f"must be a non-identity module."
        )
        super(GrowingBlock, self).__init__(
            in_features=in_features,
            out_features=out_features,
        )
        self.name = name
        self.device = device

        self.pre_activation: torch.nn.Module = pre_activation
        self.first_layer: GrowingModule = first_layer
        self.second_layer: GrowingModule = second_layer
        self.downsample = downsample

        # self.activation_derivative = torch.func.grad(mid_activation)(torch.tensor(1e-5))
        # TODO: FIX this
        self.activation_derivative = 1

    def __str__(self, verbose: int = 0) -> str:
        if verbose == 0:
            return (
                f"{self.name} ({self.first_layer.__str__()} -> "
                f"{self.second_layer.__str__()})"
            )
        elif verbose == 1:
            return (
                f"{self.name}:\n"
                f"{self.first_layer.__str__(verbose=1)}"
                f"\n->\n"
                f"{self.second_layer.__str__(verbose=1)}"
            )
        elif verbose >= 2:
            return (
                f"{self.name}:\n"
                f"Pre-activation: {self.pre_activation}\n"
                f"Downsample: {self.downsample}\n"
                f"{self.first_layer.__str__(verbose=2)}"
                f"\n->\n"
                f"{self.second_layer.__str__(verbose=2)}"
            )
        else:
            raise ValueError("verbose must be a non-negative integer.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "optimal_delta_layer":
            # We can't use directly @optimal_delta_layer.setter because of
            # inheritance issues. But we want to be able to set this attribute
            # so we use a @.setter to indicate it to the linter and redirect here.
            GrowingBlock.optimal_delta_layer.fset(self, value)  # type: ignore
        else:
            return super().__setattr__(name, value)

    @property
    def optimal_delta_layer(self) -> torch.nn.Module | None:
        """
        Get the optimal delta layer of the block.
        """
        return self.second_layer.optimal_delta_layer

    @optimal_delta_layer.setter
    def optimal_delta_layer(self, value: torch.nn.Module | None):
        """
        Set the optimal delta layer of the block.
        """
        self.second_layer.optimal_delta_layer = value

    @property
    def hidden_features(self) -> int:
        warn(
            "hidden_features is deprecated, use hidden_neurons instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.second_layer.in_neurons

    @property
    def hidden_neurons(self) -> int:
        return self.second_layer.in_neurons

    @property
    def eigenvalues_extension(self):
        return self.second_layer.eigenvalues_extension

    @property
    def parameter_update_decrease(self):
        return self.second_layer.parameter_update_decrease

    @parameter_update_decrease.setter
    def parameter_update_decrease(self, value: torch.Tensor | float):
        """
        Set the parameter update decrease for the block.
        """
        if isinstance(value, float):
            value = torch.tensor(value, device=self.device)
        elif not isinstance(value, torch.Tensor):
            raise TypeError(
                "parameter_update_decrease must be a float or a torch.Tensor."
            )

        self.second_layer.parameter_update_decrease = value

    @property
    def scaling_factor(self):
        return self.second_layer.scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, value: float):
        """
        Set the scaling factor for the second layer.
        """
        self.second_layer.scaling_factor = value  # type: ignore

    @staticmethod
    def set_default_values(
        activation: torch.nn.Module | None = None,
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

    def extended_forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        mask: dict = {},  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor
        mask: None
            mask tensor (not used)

        Returns
        -------
        torch.Tensor
            output tensor
        """
        identity: torch.Tensor = self.downsample(x)
        x = self.pre_activation(x)
        if self.hidden_neurons > 0:
            x, x_ext = self.first_layer.extended_forward(x)
            x, _ = self.second_layer.extended_forward(x, x_ext)
            assert (
                _ is None
            ), f"The output of layer 2 {self.second_layer.name} should not be extended."

            return x + identity
        elif self.first_layer.extended_output_layer is not None:
            x = self.scaling_factor * self.first_layer.extended_output_layer(x)
            x = self.first_layer.extended_post_layer_function(x)
            assert self.second_layer.extended_input_layer is not None, (
                f"Second layer {self.second_layer.name} should have an "
                f"extended output layer."
            )
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
        if self.hidden_neurons == 0:
            if self.first_layer.store_input:
                self.first_layer._input = self.pre_activation(x).detach()

            out = torch.zeros_like(identity)
            if self.second_layer.store_pre_activity:
                self.second_layer._pre_activity = out
                self.second_layer._pre_activity.requires_grad_(True)
                self.second_layer._pre_activity.retain_grad()
            self.second_layer.tensor_s_growth.updated = False
            self.second_layer.tensor_m_prev.updated = False
            self.second_layer.cross_covariance.updated = False
        else:
            out = self.pre_activation(x)
            out = self.first_layer(out)
            out = self.second_layer(out)
        return out + identity

    def init_computation(self):
        """
        Initialise the computation of the block.
        """
        # growth part
        self.first_layer.store_input = True
        self.second_layer.store_pre_activity = True
        self.second_layer.tensor_m_prev.init()
        self.second_layer.tensor_s_growth.init()

        if self.hidden_neurons > 0:
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

        if self.hidden_neurons > 0:
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

    def delete_update(self, **kwargs):
        """
        Delete the update of the block.
        """
        self.second_layer.delete_update(**kwargs)

    def set_scaling_factor(self, factor: float) -> None:
        """Assign scaling factor to all growing layers

        Parameters
        ----------
        factor : float
            scaling factor
        """
        self.second_layer.set_scaling_factor(factor)

    def compute_optimal_updates(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
        use_projected_gradient: bool = True,
    ) -> None:
        """
        Compute the optimal update for second layer and additional neurons.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root
            of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        dtype: torch.dtype
            dtype for the computation of the optimal delta and added parameters
        use_projected_gradient: bool
            whereas to use the projected gradient ie `tensor_n` or the raw `tensor_m`
        """
        if self.hidden_neurons > 0:
            _, _, _ = self.second_layer.compute_optimal_delta()
        else:
            self.second_layer.parameter_update_decrease = torch.tensor(
                0.0, device=self.device
            )
        self.second_layer.compute_optimal_added_parameters(
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
            use_projected_gradient=self.hidden_neurons > 0 and use_projected_gradient,
            dtype=dtype,
            update_previous=True,
        )

    def apply_change(
        self,
        extension_size: int | None = None,
        scaling_factor: float | torch.Tensor | None = None,
        apply_delta: bool = True,
        apply_extension: bool = True,
    ) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        self.second_layer.apply_change(
            scaling_factor=scaling_factor,
            extension_size=extension_size,
            apply_delta=apply_delta,
            apply_extension=apply_extension,
        )

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int | None = None,
        threshold: float | None = None,
        sub_select_previous: bool = True,
        zeros_if_not_enough: bool = False,
        zeros_fan_in: bool = True,
        zeros_fan_out: bool = False,
    ) -> None:
        """
        Select the first keep_neurons neurons of the optimal added parameters
        linked to this layer.

        Parameters
        ----------
        keep_neurons: int | None
            number of neurons to keep, if None, the number of neurons
            is determined by the threshold
        threshold: float | None
            threshold to determine the number of neurons to keep, if None,
            keep_neurons must be provided
        sub_select_previous: bool
            if True, sub-select the previous layer added parameters as well
        zeros_if_not_enough: bool
            if True, will keep the all neurons and set the non selected ones to zero
            (either first or last depending on zeros_fan_in and zeros_fan_out)
        zeros_fan_in: bool
            if True and zeros_if_not_enough is True, will set the non selected
            fan-in parameters to zero
        zeros_fan_out: bool
            if True and zeros_if_not_enough is True, will set the non selected
            fan-out parameters to zero
        """
        assert (
            self.eigenvalues_extension is not None
        ), "No optimal added parameters computed."
        self.second_layer.sub_select_optimal_added_parameters(
            keep_neurons=keep_neurons,
            threshold=threshold,
            sub_select_previous=sub_select_previous,
            zeros_if_not_enough=zeros_if_not_enough,
            zeros_fan_in=zeros_fan_in,
            zeros_fan_out=zeros_fan_out,
        )

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

    def create_layer_extensions(
        self,
        extension_size: int,
        output_extension_size: int | None = None,
        input_extension_size: int | None = None,
        output_extension_init: str = "copy_uniform",
        input_extension_init: str = "copy_uniform",
    ) -> None:
        """
        Create the layer input and output extensions of given sizes.

        Allow to have different sizes for input and output extensions,
        this is useful for example if you connect a convolutional layer
        to a linear layer.

        Parameters
        ----------
        extension_size: int
            size of the extension to create
        output_extension_size: int | None
            size of the output extension to create, if None use extension_size
        input_extension_size: int | None
            size of the input extension to create, if None use extension_size
        output_extension_init: str
            Initialization method for the output extension. Possible values include
            "copy_uniform", "zeros", or other supported initialization strategies.
        input_extension_init: str
            Initialization method for the input extension. Possible values include
            "copy_uniform", "zeros", or other supported initialization strategies.
        """
        self.second_layer.create_layer_extensions(
            extension_size=extension_size,
            output_extension_size=output_extension_size,
            input_extension_size=input_extension_size,
            output_extension_init=output_extension_init,
            input_extension_init=input_extension_init,
        )

    def normalize_optimal_updates(self, **kwargs) -> None:
        """
        Normalize the optimal updates.
        """
        self.second_layer.normalize_optimal_updates(**kwargs)

    def missing_neurons(self) -> int:
        """
        Get the number of missing neurons to reach the target hidden features.

        Returns
        -------
        int
            number of missing neurons
        """
        return self.second_layer.missing_neurons()

    def number_of_neurons_to_add(
        self,
        **kwargs,
    ) -> int:
        """Get the number of neurons to add in the next growth step.

        Parameters
        ----------
        method : str
            Method to use for determining the number of neurons to add.
            Options are "fixed_proportional".
        number_of_growth_steps : int
            Number of growth steps planned, used only if method is "proportional".

        Returns
        -------
        int
            Number of neurons to add.
        """
        return self.second_layer.number_of_neurons_to_add(**kwargs)

    def complete_growth(self, **kwargs) -> None:
        """Complete the growth procedure for the block.

        Parameters
        ----------
        extension_kwargs : dict
            Keyword arguments for the extension procedure.
        """
        self.second_layer.complete_growth(**kwargs)


class LinearGrowingBlock(GrowingBlock):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 0,
        target_hidden_features: int | None = None,
        activation: torch.nn.Module | None = torch.nn.Identity(),
        pre_activation: torch.nn.Module | None = None,
        mid_activation: torch.nn.Module | None = None,
        extended_mid_activation: torch.nn.Module | None = None,
        name: str = "block",
        kwargs_layer: dict | None = None,
        kwargs_first_layer: dict | None = None,
        kwargs_second_layer: dict | None = None,
        downsample: torch.nn.Module = torch.nn.Identity(),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        in_features: int
            number of input channels
        out_features: int
            number of output channels
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        pre_activation: torch.nn.Module | None
            activation function to use before the first layer,
            if None use the activation function
        mid_activation: torch.nn.Module | None
            activation function to use between the two layers,
            if None use the activation function
        extended_mid_activation: torch.nn.Module | None
            activation function to use between the two layers in the extended forward,
            if None use the mid_activation
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
        device: torch.device | None
            device to use for the block
        """
        pre_activation, mid_activation, kwargs_first_layer, kwargs_second_layer = (
            self.set_default_values(
                activation=activation,
                pre_activation=pre_activation,
                mid_activation=mid_activation,
                kwargs_layer=kwargs_layer,
                kwargs_first_layer=kwargs_first_layer,
                kwargs_second_layer=kwargs_second_layer,
            )
        )
        first_layer = LinearGrowingModule(
            in_features=in_features,
            out_features=hidden_features,
            name=f"{name}(first_layer)",
            post_layer_function=mid_activation,
            extended_post_layer_function=extended_mid_activation,
            **kwargs_first_layer,
        )
        second_layer = LinearGrowingModule(
            in_features=hidden_features,
            out_features=out_features,
            name=f"{name}(second_layer)",
            target_in_features=target_hidden_features,
            previous_module=first_layer,
            **kwargs_second_layer,
        )
        super(LinearGrowingBlock, self).__init__(
            in_features=in_features,
            out_features=out_features,
            pre_activation=pre_activation,
            name=name,
            first_layer=first_layer,
            second_layer=second_layer,
            downsample=downsample,
            device=device,
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
        kernel_size: int | tuple[int, int] | None = None,
        hidden_channels: int = 0,
        target_hidden_channels: int | None = None,
        activation: torch.nn.Module | None = None,
        pre_activation: torch.nn.Module | None = None,
        mid_activation: torch.nn.Module | None = None,
        extended_mid_activation: torch.nn.Module | None = None,
        name: str = "conv_block",
        kwargs_layer: dict | None = None,
        kwargs_first_layer: dict | None = None,
        kwargs_second_layer: dict | None = None,
        downsample: torch.nn.Module = torch.nn.Identity(),
        device: torch.device | None = None,
    ) -> None:
        """
        Initialise the convolutional block.

        Parameters
        ----------
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        kernel_size: int | tuple[int, int] | None
            size of the convolutional kernel
        hidden_channels: int
            number of hidden channels, if zero the block is the zero function
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        pre_activation: torch.nn.Module | None
            activation function to use before the first layer,
            if None use the activation function
        mid_activation: torch.nn.Module | None
            activation function to use between the two layers,
            if None use the activation function
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
        device: torch.device | None
            device to use for the block
        """
        pre_activation, mid_activation, kwargs_first_layer, kwargs_second_layer = (
            self.set_default_values(
                activation=activation,
                pre_activation=pre_activation,
                mid_activation=mid_activation,
                kwargs_layer=kwargs_layer,
                kwargs_first_layer=kwargs_first_layer,
                kwargs_second_layer=kwargs_second_layer,
            )
        )

        for kwargs in (kwargs_first_layer, kwargs_second_layer):
            if "kernel_size" not in kwargs:
                if kernel_size is None:
                    raise ValueError(f"kernel_size must be specified for {name}.")
                kwargs["kernel_size"] = kernel_size
            elif kernel_size is not None:
                warn(
                    f"kernel_size specified in both arguments and kwargs for {name}, "
                    f"using value from kwargs."
                )

        first_layer = RestrictedConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            name=f"{name}(first_layer)",
            post_layer_function=mid_activation,
            extended_post_layer_function=extended_mid_activation,
            device=device,
            **kwargs_first_layer,
        )
        second_layer = RestrictedConv2dGrowingModule(
            in_channels=hidden_channels,
            out_channels=out_channels,
            name=f"{name}(second_layer)",
            target_in_channels=target_hidden_channels,
            previous_module=first_layer,
            device=device,
            **kwargs_second_layer,
        )

        super(RestrictedConv2dGrowingBlock, self).__init__(
            in_features=in_channels,
            out_features=out_channels,
            pre_activation=pre_activation,
            name=name,
            first_layer=first_layer,
            second_layer=second_layer,
            downsample=downsample,
            device=device,
        )
