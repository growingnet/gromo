from warnings import warn

import torch

from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import (
    apply_border_effect_on_unfolded,
    compute_mask_tensor_t,
    compute_output_shape_conv,
    create_bordering_effect_convolution,
)
from gromo.utils.utils import global_device


class Conv2dMergeGrowingModule(MergeGrowingModule):
    pass


class Conv2dGrowingModule(GrowingModule):
    """
    Conv2dGrowingModule is a GrowingModule for a Conv2d layer.

    Parameters
    ----------
    For the parameters in_channels, out_channels, kernel_size, stride, padding, dilation,
     use_bias they are the same as in torch.nn.Conv2d.

    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int]
    padding: int | tuple[int, int]
    dilation: int | tuple[int, int]
    post_layer_function: torch.nn.Module
        function applied after the layer (e.g. activation function)
    previous_module: GrowingModule | MergeGrowingModule | None
        previous module in the network (None if the first module),
        needed to extend the layer
    next_module: GrowingModule | MergeGrowingModule | None
        next module in the network (None if the last module)
    allow_growing: bool
        whether the layer can grow in input size
    device: torch.device | None
        device for the layer
    name: str | None
        name of the layer used for debugging purpose
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        input_size: tuple[int, int] = (-1, -1),
        # groups: int = 1,
        use_bias: bool = True,
        post_layer_function: torch.nn.Module = torch.nn.Identity(),
        extended_post_layer_function: torch.nn.Module | None = None,
        previous_module: GrowingModule | MergeGrowingModule | None = None,
        next_module: GrowingModule | MergeGrowingModule | None = None,
        allow_growing: bool = False,
        device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        device = device if device is not None else global_device()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        super(Conv2dGrowingModule, self).__init__(
            layer=torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=use_bias,
                device=device,
            ),
            post_layer_function=post_layer_function,
            extended_post_layer_function=extended_post_layer_function,
            previous_module=previous_module,
            next_module=next_module,
            allow_growing=allow_growing,
            tensor_s_shape=(
                use_bias + in_channels * kernel_size[0] * kernel_size[1],
                use_bias + in_channels * kernel_size[0] * kernel_size[1],
            ),
            tensor_m_shape=(
                use_bias + in_channels * kernel_size[0] * kernel_size[1],
                out_channels,
            ),
            device=device,
            name=name,
        )
        self.kernel_size = self.layer.kernel_size

        self.input_size: tuple[int, int] = input_size
        self.use_bias = use_bias

    # Information functions
    # TODO: implement activation_gradient ?
    # this function is used to estimate the F.O. improvement of the loss after the extension of the network
    # however this won't work if we do not have only the activation function as the post_layer_function

    @property
    def padding(self):
        return self.layer.padding

    @padding.setter
    def padding(self, value):
        self.layer.padding = value

    @property
    def unfolded_extended_input(self) -> torch.Tensor:
        """
        Return the unfolded input extended with a channel of ones if the bias is used.

        Returns
        -------
        torch.Tensor
            unfolded input extended
        """
        # TODO: maybe we could compute it only once if we use it multiple times (e.g. in S, S_prev, M, M_prev, P...)
        unfolded_input = torch.nn.functional.unfold(
            self.input,
            self.layer.kernel_size,
            padding=self.layer.padding,
            stride=self.layer.stride,
            dilation=self.layer.dilation,
        )
        if self.use_bias:
            return torch.cat(
                (
                    unfolded_input,
                    torch.ones(
                        unfolded_input.shape[0],
                        1,
                        unfolded_input.shape[2],
                        device=self.device,
                    ),
                ),
                dim=1,
            )
        else:
            return unfolded_input

    def number_of_parameters(self) -> int:
        """
        Return the number of parameters of the layer.

        Returns
        -------
        int
            number of parameters
        """
        return (
            self.layer.in_channels
            * self.layer.out_channels
            * self.layer.kernel_size[0]
            * self.layer.kernel_size[1]
            + self.layer.out_channels * self.use_bias
        )

    def __str__(self, verbose: int = 0) -> str:
        if verbose == 0:
            return (
                f"Conv2dGrowingModule({self.name if self.name else ' '})"
                f"(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"use_bias={self.use_bias})"
            )
        else:
            return super(Conv2dGrowingModule, self).__str__(verbose=verbose)

    # Statistics computation
    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S.
        With the input tensor B, the update is
        S := (B^c_F)^T B^c_F \in (C d[+1]d[+1], C d[+1]d[+1]).

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update
        """
        assert (
            self.store_input
        ), f"The input must be stored to compute the update of S. (error in {self.name})"
        assert (
            self.input is not None
        ), f"The input must be stored to compute the update of S. (error in {self.name})"
        unfolded_extended_input = self.unfolded_extended_input
        return (
            torch.einsum(
                "iam, ibm -> ab",
                unfolded_extended_input,
                unfolded_extended_input,
            ),
            self.input.shape[0],
        )

    def compute_m_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M.
        With the input tensor X and dLoss/dA the gradient of the loss
        with respect to the pre-activity:
        M = B[-1]^T dA

        Parameters
        ----------
        desired_activation: torch.Tensor | None
            desired variation direction of the output  of the layer

        Returns
        -------
        torch.Tensor
            update of the tensor M
        int
            number of samples used to compute the update
        """
        if desired_activation is None:
            desired_activation = self.pre_activity.grad
        desired_activation = desired_activation.flatten(start_dim=-2)

        return (
            torch.einsum(
                "iam, icm -> ac", self.unfolded_extended_input, desired_activation
            ),
            self.input.shape[0],
        )

    # Layer edition
    def layer_of_tensor(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.nn.Conv2d:
        """
        Create a layer with the same characteristics (excepted the shape)
         with weight as parameter and bias as bias.

        Parameters
        ----------
        weight: torch.Tensor
            weight of the layer
        bias: torch.Tensor | None
            bias of the layer

        Returns
        -------
        torch.nn.Linear
            layer with the same characteristics
        """
        assert self.use_bias is (bias is not None), (
            f"The new layer should have a bias ({bias is not None=}) if and only if "
            f"the main layer bias ({self.use_bias =}) is not None."
        )
        for i in (0, 1):
            assert (
                weight.shape[2 + i] == self.layer.kernel_size[i]
            ), f"{weight.shape[2 + i]=} should be equal to {self.layer.kernel_size[i]=}"
        self.layer.kernel_size: tuple[int, int]
        self.layer.stride: tuple[int, int]
        self.layer.dilation: tuple[int, int]

        new_layer = torch.nn.Conv2d(
            weight.shape[1],
            weight.shape[0],
            bias=self.use_bias,
            device=self.device,
            kernel_size=self.layer.kernel_size,
            stride=self.layer.stride,
            padding=self.layer.padding,
            dilation=self.layer.dilation,
        )
        new_layer.weight = torch.nn.Parameter(weight)
        if self.use_bias:
            new_layer.bias = torch.nn.Parameter(bias)
        return new_layer

    # FIXME: should we implement .add_parameters

    def layer_in_extension(self, weight: torch.Tensor) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the input of the layer is extended but not the output.

        Parameters
        ----------
        weight: torch.Tensor (out_channels, K, kernel_size[0], kernel_size[1])
            weight of the extension
        """
        assert (
            weight.shape[0] == self.out_channels
        ), f"{weight.shape[0]=} should be equal to {self.out_features=}"
        for i in (0, 1):
            assert (
                weight.shape[2 + i] == self.layer.kernel_size[i]
            ), f"{weight.shape[2 + i]=} should be equal to {self.layer.kernel_size[i]=}"

        # TODO: check this is working
        self.layer = self.layer_of_tensor(
            weight=torch.cat((self.weight, weight), dim=1), bias=self.bias
        )

        self.in_channels += weight.shape[1]
        self._tensor_s = TensorStatistic(
            (
                (self.in_channels + self.use_bias)
                * self.layer.kernel_size[0]
                * self.layer.kernel_size[1],
                (self.in_channels + self.use_bias)
                * self.layer.kernel_size[0]
                * self.layer.kernel_size[1],
            ),
            update_function=self.compute_s_update,
            name=self.tensor_s.name,
        )
        self.tensor_m = TensorStatistic(
            (
                (self.in_channels + self.use_bias)
                * self.layer.kernel_size[0]
                * self.layer.kernel_size[1],
                self.out_channels,
            ),
            update_function=self.compute_m_update,
            name=self.tensor_m.name,
        )

    def layer_out_extension(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the output of the layer is extended but not the input.

        Parameters
        ----------
        weight: torch.Tensor (K, in_features)
            weight of the extension
        bias: torch.Tensor (K) | None
            bias of the extension if needed
        """
        assert (
            weight.shape[1] == self.in_channels
        ), f"{weight.shape[1]=} should be equal to {self.in_channels=}"
        assert (
            bias is None or bias.shape[0] == weight.shape[0]
        ), f"{bias.shape[0]=} should be equal to {weight.shape[0]=}"

        if self.use_bias:
            assert (
                bias is not None
            ), f"The bias of the extension should be provided because the layer has a bias"
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0),
                bias=torch.cat((self.layer.bias, bias), dim=0),
            )
        else:
            if bias is not None:
                warn(
                    f"The bias of the extension should not be provided because the layer has no bias.",
                    UserWarning,
                )
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0), bias=None
            )

        self.out_channels += weight.shape[0]
        self.tensor_m = TensorStatistic(
            (self.in_channels + self.use_bias, self.out_channels),
            update_function=self.compute_m_update,
            name=self.tensor_m.name,
        )

    def _sub_select_added_output_dimension(self, keep_neurons: int) -> None:
        """
        Select the first `keep_neurons` neurons of the optimal added output dimension.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        """
        assert (
            self.extended_output_layer is not None
        ), f"The layer should have an extended output layer to sub-select the output dimension."
        self.extended_output_layer = self.layer_of_tensor(
            self.extended_output_layer.weight[:keep_neurons],
            bias=(
                self.extended_output_layer.bias[:keep_neurons]
                if self.extended_output_layer.bias is not None
                else None
            ),
        )

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int,
        sub_select_previous: bool = True,
    ) -> None:
        """
        Select the first `keep_neurons` neurons of the optimal added parameters.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        sub_select_previous: bool
            if True, sub-select the previous layer added parameters as well
        """
        assert (self.extended_input_layer is None) ^ (
            self.extended_output_layer is None
        ), "The layer should have an extended input xor output layer."
        if self.extended_input_layer is not None:
            self.extended_input_layer = self.layer_of_tensor(
                self.extended_input_layer.weight[:, :keep_neurons],
                bias=self.extended_input_layer.bias,
            )
            assert self.eigenvalues_extension is not None, (
                f"The eigenvalues of the extension should be computed before "
                f"sub-selecting the optimal added parameters."
            )
            self.eigenvalues_extension = self.eigenvalues_extension[:keep_neurons]

        if sub_select_previous:
            if self.previous_module is None:
                raise ValueError(
                    f"No previous module for {self.name}. "
                    "Therefore new neurons cannot be sub-selected."
                )
            elif isinstance(self.previous_module, LinearGrowingModule):
                self.previous_module._sub_select_added_output_dimension(keep_neurons)
            elif isinstance(self.previous_module, LinearMergeGrowingModule):
                raise NotImplementedError(f"TODO")
            elif isinstance(self.previous_module, Conv2dGrowingModule):
                self.previous_module._sub_select_added_output_dimension(keep_neurons)
            elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
                raise NotImplementedError(f"TODO")
            else:
                raise NotImplementedError(
                    f"The sub-selection of the optimal added parameters is not implemented "
                    f"yet for {type(self.previous_module)} as previous module."
                )

    def update_input_size(self, input_size: tuple[int, int] | None = None) -> None:
        """
        Update the input size of the layer. Either according to the parameter or the input currently stored.

        Parameters
        ----------
        input_size: tuple[int, int] | None
            new input size
        """
        if input_size is not None:
            new_size = input_size
        elif self.store_input and self.input is not None:
            new_size = self.input.shape[-2:]
        elif self.previous_module and self.previous_module.input_size != (-1, -1):
            new_size = compute_output_shape_conv(
                self.previous_module.input_size, self.previous_module.layer
            )
        else:
            raise AssertionError(f"Unable to compute the input size for {self.name}.")

        if self.input_size != (-1, -1) and new_size != self.input_size:
            warn(
                f"The input size of the layer {self.name} has changed from {self.input_size} to {new_size}."
                f"This may lead to errors if the size of the tensor statistics "
                f"and of the mask tensor T are not updated."
            )
        self.input_size = new_size

    def update_computation(self) -> None:
        """
        Update the computation of the layer.
        """
        self.update_input_size()
        super(Conv2dGrowingModule, self).update_computation()


class RestrictedConv2dGrowingModule(Conv2dGrowingModule):
    """
    Conv2dGrowingModule for a Conv2d layer with a growth scheme Conv -> Conv 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        input_size: tuple[int, int] = (-1, -1),
        # groups: int = 1,
        use_bias: bool = True,
        post_layer_function: torch.nn.Module = torch.nn.Identity(),
        extended_post_layer_function: torch.nn.Module | None = None,
        previous_module: GrowingModule | MergeGrowingModule | None = None,
        next_module: GrowingModule | MergeGrowingModule | None = None,
        allow_growing: bool = False,
        device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        super(RestrictedConv2dGrowingModule, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            input_size=input_size,
            use_bias=use_bias,
            post_layer_function=post_layer_function,
            extended_post_layer_function=extended_post_layer_function,
            previous_module=previous_module,
            next_module=next_module,
            allow_growing=allow_growing,
            device=device,
            name=name,
        )
        self.bordering_convolution = None

    def linear_layer_of_tensor(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.nn.Conv2d:
        """
        Create a layer with the same characteristics (excepted the shape)
        with `weight` as parameter for the central weights and bias as bias
        i.e. `weight` are the parameters of 1x1 convolution.

        Parameters
        ----------
        weight: torch.Tensor
            weight of the layer of size (out_channels, in_channels) or
            (out_channels, in_channels, 1, 1)
        bias: torch.Tensor | None
            bias of the layer

        Returns
        -------
        torch.nn.Conv2d
            layer with the same characteristics
        """
        assert self.use_bias is (bias is not None), (
            f"The new layer should have a bias ({bias is not None=}) if and only if "
            f"the main layer bias ({self.use_bias =}) is not None."
        )

        assert weight.dim() in (
            2,
            4,
        ), f"weight should have 2 or 4 dimensions, but got {weight.dim()=}."
        assert weight.dim() == 2 or weight.shape[2:] == (1, 1), (
            f"weight should have 2 dimensions or the last two dimensions should be (1, 1), "
            f"but got {weight.shape=}."
        )

        if weight.dim() == 2:
            weight = weight.unsqueeze(-1).unsqueeze(-1)

        self.layer.kernel_size: tuple[int, int]
        self.layer.stride: tuple[int, int]
        self.layer.dilation: tuple[int, int]

        full_weight = torch.zeros(
            weight.shape[0],
            weight.shape[1],
            self.kernel_size[0],
            self.kernel_size[1],
            device=self.device,
        )
        mid = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        full_weight[:, :, mid[0] : mid[0] + 1, mid[1] : mid[1] + 1] = weight

        new_layer = torch.nn.Conv2d(
            weight.shape[1],
            weight.shape[0],
            bias=self.use_bias,
            device=self.device,
            kernel_size=self.layer.kernel_size,
            stride=self.layer.stride,
            padding=self.layer.padding,
            dilation=self.layer.dilation,
        )
        new_layer.weight = torch.nn.Parameter(full_weight)
        if self.use_bias:
            new_layer.bias = torch.nn.Parameter(bias)
        return new_layer

    @property
    def bordered_unfolded_extended_prev_input(self) -> torch.Tensor:
        """
        Return the unfolded input extended with a channel of ones if the bias is used of
        the previous layer and with border effect of the current layer already applied.
        """
        if self.previous_module is None:
            raise ValueError(
                f"Cannot compute the bordered unfolded input without a previous module for {self.name}."
            )
        if self.bordering_convolution is None:
            if isinstance(self.previous_module, Conv2dGrowingModule):
                self.bordering_convolution = create_bordering_effect_convolution(
                    self.previous_module.in_channels
                    * self.previous_module.kernel_size[0]
                    * self.previous_module.kernel_size[1]
                    + self.previous_module.use_bias,
                    convolution=self.layer,
                )
            elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
                raise NotImplementedError
            else:
                raise NotImplementedError
        self.previous_module.update_input_size()
        return apply_border_effect_on_unfolded(
            unfolded_tensor=self.previous_module.unfolded_extended_input,
            original_size=self.previous_module.input_size,
            border_effect_conv=self.layer,
            identity_conv=self.bordering_convolution,
        )

    def compute_m_prev_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M_{-2} := B[-2] <x> dA.
        Precisely: M_{-2}(bca) = Bt[-2](ixab) dA(icx)
        where Bt[-2] is the masked unfolded input of the previous layer.

        Parameters
        ----------
        desired_activation: torch.Tensor | None
            desired variation direction of the output  of the layer

        Returns
        -------
        torch.Tensor
            update of the tensor M_{-2}
        int
            number of samples used to compute the update
        """
        if desired_activation is None:
            desired_activation = self.pre_activity.grad
        desired_activation = desired_activation.flatten(start_dim=-2)

        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus M_{-2} is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            unfolded_extended_input = self.bordered_unfolded_extended_prev_input
            assert unfolded_extended_input.shape[0] == desired_activation.shape[0], (
                f"The number of samples is incoherent: {unfolded_extended_input.shape[0]=} "
                f"and {desired_activation.shape[0]=} should be equal."
            )
            assert unfolded_extended_input.shape[2] == desired_activation.shape[2], (
                f"The number of features is incoherent: {unfolded_extended_input.shape[2]=} "
                f"and {desired_activation.shape[2]=} should be equal."
            )
            return (
                torch.einsum(
                    "iax, icx -> ac",
                    unfolded_extended_input,
                    desired_activation,
                ),
                desired_activation.shape[0],
            )
        elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        else:
            raise NotImplementedError(
                f"The computation of M_{-2} is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    def compute_cross_covariance_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor P := B[-2] <x> B[-1].
        Precisely: P(ab) = Bc[-2](iax) Bc[-1](ibx)
        where Bc[-2] is the unfolded input of the previous layer
        and Bc[-1] is the unfolded input of the current layer.

        Returns
        -------
        torch.Tensor
            update of the tensor P
        int
            number of samples used to compute the update
        """
        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus the cross covariance is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            return (
                torch.einsum(
                    "iax, ibx -> ab",
                    self.bordered_unfolded_extended_prev_input,
                    self.unfolded_extended_input,
                ),
                self.input.shape[0],
            )
        elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        else:
            raise NotImplementedError(
                f"The computation of P is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    @property
    def tensor_n(self) -> torch.Tensor:
        """
        Compute the tensor N for the layer with the current M_-2, C and optimal delta.

        Returns
        -------
        torch.Tensor
            N
        """
        assert (
            self.tensor_m_prev() is not None
        ), f"The tensor M_{-2} should be computed before the tensor N for {self.name}."
        assert (
            self.cross_covariance() is not None
        ), f"The cross covariance should be computed before the tensor N for {self.name}."
        assert isinstance(
            self.cross_covariance(), torch.Tensor
        ), f"The cross covariance should be a tensor for {self.name}, is {type(self.cross_covariance())}."
        assert (
            self.cross_covariance().shape[1]
            == self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            + self.use_bias
        ), (
            f"The cross covariance should have shape "
            f"(..., {self.in_channels * self.kernel_size[0] * self.kernel_size[1]  + self.use_bias})"
            f" but got {self.cross_covariance().shape}."
        )
        assert (
            self.delta_raw is not None
        ), f"The optimal delta should be computed before the tensor N for {self.name}."
        assert isinstance(
            self.delta_raw, torch.Tensor
        ), f"The optimal delta should be a tensor for {self.name}, is {type(self.delta_raw)}."
        assert (
            self.delta_raw.shape[1]
            == self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            + self.use_bias
        ), (
            f"Expected delta_raw.shape[1] == {self.in_channels * self.kernel_size[0] * self.kernel_size[1] + self.use_bias}, "
            f"but got {self.delta_raw.shape[1]} (full shape: {self.delta_raw.shape})."
        )
        assert (
            self.delta_raw.shape[0] == self.out_channels
        ), f"Expected delta_raw.shape[0] == {self.out_channels}, but got {self.delta_raw.shape[0]}."
        return -self.tensor_m_prev() - torch.einsum(
            "ab, cb -> ac", self.cross_covariance(), self.delta_raw
        )

    def compute_optimal_added_parameters(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        dtype: torch.dtype = torch.float32,
        use_projected_gradient: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Compute the optimal added parameters to extend the input layer.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        update_previous: bool
            whether to change the previous layer extended_output_layer
        dtype: torch.dtype
            dtype for S and N during the computation
        use_projected_gradient: bool
            whereas to use the projected gradient ie `tensor_n` or the raw `tensor_m`

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
            optimal added weights (alpha weights, alpha bias, omega) and eigenvalues lambda
        """
        alpha, omega, self.eigenvalues_extension = self._auxiliary_compute_alpha_omega(
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
            dtype=dtype,
            use_projected_gradient=use_projected_gradient,
        )

        k = self.eigenvalues_extension.shape[0]
        assert alpha.shape[0] == omega.shape[1] == k, (
            f"alpha and omega should have the same number of added neurons {k}."
            f"but got {alpha.shape} and {omega.shape}."
        )
        assert (
            omega.shape[0] == self.out_channels
        ), f"omega should have the same number of output features as the layer."

        if self.previous_module.use_bias:
            alpha_weight = alpha[:, :-1]
            alpha_bias = alpha[:, -1]
        else:
            alpha_weight = alpha
            alpha_bias = None

        if isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: should we implement Lin -> Conv")
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            alpha_weight = alpha_weight.reshape(
                k,
                self.previous_module.in_channels,
                self.previous_module.kernel_size[0],
                self.previous_module.kernel_size[1],
            )
        elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
            raise NotImplementedError("TODO: implement this: Conv Add -> Conv")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError("TODO: should we implement Lin Add -> Conv")
        else:
            raise NotImplementedError

        assert omega.shape == (
            self.out_channels,
            k,
        ), (
            f"omega should have shape ({k}, {self.out_channels}, {self.kernel_size[0]}, {self.kernel_size[1]})"
            f"but got {omega.shape}."
        )
        assert alpha.shape[0] == k, (
            f"alpha should have shape ({k}, ...)" f"but got {alpha.shape}."
        )

        self.extended_input_layer = self.linear_layer_of_tensor(
            omega,
            bias=(
                torch.zeros(self.out_channels, device=self.device)
                if self.use_bias
                else None
            ),
        )

        if update_previous:
            if isinstance(
                self.previous_module, LinearGrowingModule | Conv2dGrowingModule
            ):
                self.previous_module.extended_output_layer = (
                    self.previous_module.layer_of_tensor(alpha_weight, alpha_bias)
                )
            elif isinstance(
                self.previous_module,
                LinearMergeGrowingModule | Conv2dMergeGrowingModule,
            ):
                raise NotImplementedError("TODO: implement this")
            else:
                raise NotImplementedError(
                    f"The computation of the optimal added parameters is not implemented "
                    f"yet for {type(self.previous_module)} as previous module."
                )

        return alpha_weight, alpha_bias, omega, self.eigenvalues_extension


class FullConv2dGrowingModule(Conv2dGrowingModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        input_size: tuple[int, int] = (-1, -1),
        # groups: int = 1,
        use_bias: bool = True,
        post_layer_function: torch.nn.Module = torch.nn.Identity(),
        extended_post_layer_function: torch.nn.Module | None = None,
        previous_module: GrowingModule | MergeGrowingModule | None = None,
        next_module: GrowingModule | MergeGrowingModule | None = None,
        allow_growing: bool = False,
        device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        super(FullConv2dGrowingModule, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            input_size=input_size,
            use_bias=use_bias,
            post_layer_function=post_layer_function,
            extended_post_layer_function=extended_post_layer_function,
            previous_module=previous_module,
            next_module=next_module,
            allow_growing=allow_growing,
            device=device,
            name=name,
        )
        self._mask_tensor_t: torch.Tensor | None = None
        self._tensor_s_growth = TensorStatistic(
            None,
            update_function=self.compute_s_growth_update,
            device=self.device,
            name=f"S_growth({name})",
        )

    @property
    def mask_tensor_t(self) -> torch.Tensor:
        """
        Compute the tensor T for the layer.

        Returns
        -------
        torch.Tensor
            mask tensor T
        """
        assert self.input_size != (-1, -1), (
            f"The input size should be set before computing the mask tensor T "
            f"for {self.name}."
        )
        self.layer: torch.nn.Conv2d  # CHECK: why do we need to specify the type here?
        if self._mask_tensor_t is None:
            self._mask_tensor_t = compute_mask_tensor_t(self.input_size, self.layer).to(
                self.device
            )
        return self._mask_tensor_t

    @property
    def masked_unfolded_prev_input(self) -> torch.Tensor:
        """
        Return the previous masked unfolded activation.

        Returns
        -------
        torch.Tensor
            previous masked unfolded activation
        """
        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}."
                "Therefore the previous masked unfolded activation is not defined."
            )
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            return torch.einsum(
                "ial, jel -> ijea",
                self.previous_module.unfolded_extended_input,
                self.mask_tensor_t,
            )
        elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        else:
            raise NotImplementedError(
                f"The computation of the previous masked unfolded activation is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    def compute_m_prev_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M_{-2} := B[-2] <x> dA.
        Precisely: M_{-2}(bca) = Bt[-2](ixab) dA(icx)
        where Bt[-2] is the masked unfolded input of the previous layer.

        Parameters
        ----------
        desired_activation: torch.Tensor | None
            desired variation direction of the output  of the layer

        Returns
        -------
        torch.Tensor
            update of the tensor M_{-2}
        int
            number of samples used to compute the update
        """
        if desired_activation is None:
            desired_activation = self.pre_activity.grad
        desired_activation = desired_activation.flatten(start_dim=-2)

        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus M_{-2} is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            return (
                torch.einsum(
                    "ixab, icx -> bca",
                    self.masked_unfolded_prev_input,
                    desired_activation,
                ).flatten(start_dim=-2),
                desired_activation.shape[0],
            )
        elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        else:
            raise NotImplementedError(
                f"The computation of M_{-2} is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    def compute_s_growth_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S_growth.
        With the input tensor B, the update is
        S_growth := (Bt)^T Bt \in (C dd, C dd).

        Returns
        -------
        torch.Tensor
            update of the tensor S_growth
        int
            number of samples used to compute the update
        """
        return (
            torch.einsum(
                "ijea, ijeb -> ab",
                self.masked_unfolded_prev_input,
                self.masked_unfolded_prev_input,
            ),
            self.masked_unfolded_prev_input.shape[0],
        )

    def compute_cross_covariance_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor P := B[-2] <x> B[-1].
        Precisely: P(abe) = Bt[-2](ixab) Bc[-1](iex)
        where Bt[-2] is the masked unfolded input of the previous layer
        and Bc[-1] is the unfolded input of the current layer.

        Returns
        -------
        torch.Tensor
            update of the tensor P
        int
            number of samples used to compute the update
        """
        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus the cross covariance is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            return (
                torch.einsum(
                    "ixab, iex -> abe",
                    self.masked_unfolded_prev_input,
                    self.unfolded_extended_input,
                ),
                self.input.shape[0],
            )
        elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
            raise NotImplementedError("TODO: implement this")
        else:
            raise NotImplementedError(
                f"The computation of P is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    @property
    def tensor_s_growth(self) -> TensorStatistic:
        """
        Override `tensor_s_growth` to redirect to `self._tensor_s_growth` instead of `self.previous_module.tensor_s`.
        """
        return self._tensor_s_growth

    @property
    def tensor_n(self) -> torch.Tensor:
        """
        Compute the tensor N for the layer with the current M_-2, C and optimal delta.

        Returns
        -------
        torch.Tensor
            N
        """
        assert (
            self.tensor_m_prev() is not None
        ), f"The tensor M_{-2} should be computed before the tensor N for {self.name}."
        assert (
            self.cross_covariance() is not None
        ), f"The cross covariance should be computed before the tensor N for {self.name}."
        assert (
            self.delta_raw is not None
        ), f"The optimal delta should be computed before the tensor N for {self.name}."
        return -self.tensor_m_prev() - torch.einsum(
            "abe, ce -> bca", self.cross_covariance(), self.delta_raw
        ).flatten(start_dim=-2)

    def compute_optimal_added_parameters(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        dtype: torch.dtype = torch.float32,
        use_projected_gradient: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Compute the optimal added parameters to extend the input layer.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        update_previous: bool
            whether to change the previous layer extended_output_layer
        dtype: torch.dtype
            dtype for S and N during the computation
        use_projected_gradient: bool
            whereas to use the projected gradient ie `tensor_n` or the raw `tensor_m`

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
            optimal added weights (alpha weights, alpha bias, omega) and eigenvalues lambda
        """
        alpha, omega, self.eigenvalues_extension = self._auxiliary_compute_alpha_omega(
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
            dtype=dtype,
            use_projected_gradient=use_projected_gradient,
        )

        k = self.eigenvalues_extension.shape[0]
        assert alpha.shape[0] == omega.shape[1] == k, (
            f"alpha and omega should have the same number of added neurons {k}."
            f"but got {alpha.shape} and {omega.shape}."
        )
        assert (
            omega.shape[0]
            == self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        ), f"omega should have the same number of output features as the layer."

        if self.previous_module.use_bias:
            alpha_weight = alpha[:, :-1]
            alpha_bias = alpha[:, -1]
        else:
            alpha_weight = alpha
            alpha_bias = None

        if isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: should we implement Lin -> Conv")
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            alpha_weight = alpha_weight.reshape(
                k,
                self.previous_module.in_channels,
                self.previous_module.kernel_size[0],
                self.previous_module.kernel_size[1],
            )
        elif isinstance(self.previous_module, Conv2dMergeGrowingModule):
            raise NotImplementedError("TODO: implement this: Conv Add -> Conv")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError("TODO: should we implement Lin Add -> Conv")
        else:
            raise NotImplementedError

        omega = omega.reshape(
            self.out_channels, self.kernel_size[0], self.kernel_size[1], k
        ).permute(0, 3, 1, 2)

        assert omega.shape == (
            self.out_channels,
            k,
            self.kernel_size[0],
            self.kernel_size[1],
        ), (
            f"omega should have shape ({k}, {self.out_channels}, {self.kernel_size[0]}, {self.kernel_size[1]})"
            f"but got {omega.shape}."
        )
        assert alpha.shape[0] == k, (
            f"alpha should have shape ({k}, ...)" f"but got {alpha.shape}."
        )

        self.extended_input_layer = self.layer_of_tensor(
            omega,
            bias=(
                torch.zeros(self.out_channels, device=self.device)
                if self.use_bias
                else None
            ),
        )

        if update_previous:
            if isinstance(
                self.previous_module, LinearGrowingModule | Conv2dGrowingModule
            ):
                self.previous_module.extended_output_layer = (
                    self.previous_module.layer_of_tensor(alpha_weight, alpha_bias)
                )
            elif isinstance(
                self.previous_module,
                LinearMergeGrowingModule | Conv2dMergeGrowingModule,
            ):
                raise NotImplementedError("TODO: implement this")
            else:
                raise NotImplementedError(
                    f"The computation of the optimal added parameters is not implemented "
                    f"yet for {type(self.previous_module)} as previous module."
                )

        return alpha_weight, alpha_bias, omega, self.eigenvalues_extension
