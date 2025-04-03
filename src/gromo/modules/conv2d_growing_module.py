from warnings import warn

import torch

from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import (
    compute_mask_tensor_t,
    compute_optimal_added_parameters,
    compute_output_shape_conv,
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

        # TODO: update S_growth shape in layer_in_extension
        self.input_size: tuple[int, int] = input_size
        self._mask_tensor_t: torch.Tensor | None = None
        self.use_bias = use_bias

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

    # Information functions
    # TODO: implement activation_gradient ?
    # this function is used to estimate the F.O. improvement of the loss after the extension of the network
    # however this won't work if we do not have only the activation function as the post_layer_function

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
        return (
            torch.einsum(
                "iam, ibm -> ab",
                self.unfolded_extended_input,
                self.unfolded_extended_input,
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
        return (
            -self.tensor_m_prev()
            - torch.einsum("abe, ce -> bca", self.cross_covariance(), self.delta_raw)
        ).flatten(start_dim=-2)

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

    # Optimal update computation
    def compute_optimal_delta(
        self,
        update: bool = True,
        dtype: torch.dtype = torch.float32,
        force_pseudo_inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]:
        """
        Compute the optimal delta for the layer using current S and M tensors.

        dW* = M S[-1]^-1 (if needed we use the pseudo-inverse)

        Compute dW* (and dBias* if needed) and update the optimal_delta_layer attribute.
        L(A + gamma * B * dW) = L(A) - gamma * d + o(gamma)
        where d is the first order decrease and gamma the scaling factor.

        Parameters
        ----------
        update: bool
            if True update the optimal delta layer attribute
        dtype: torch.dtype
            dtype for S and M during the computation
        force_pseudo_inverse: bool
            whether to use the pseudo-inverse in any case

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]
            optimal delta for the weights, the biases if needed and the first order decrease
        """
        tensor_s = self.tensor_s()
        tensor_m = self.tensor_m()

        saved_dtype = tensor_s.dtype
        if tensor_s.dtype != dtype:
            tensor_s = tensor_s.to(dtype=dtype)
        if tensor_m.dtype != dtype:
            tensor_m = tensor_m.to(dtype=dtype)

        if not force_pseudo_inverse:
            try:
                self.delta_raw = torch.linalg.solve(tensor_s, tensor_m).t()
            except torch.linalg.LinAlgError:
                force_pseudo_inverse = True
                # self.delta_raw = torch.linalg.lstsq(tensor_s, tensor_m).solution.t()
                # do not use lstsq because it does not work with the GPU
                warn(
                    f"Using the pseudo-inverse for the computation of the optimal delta "
                    f"for {self.name}."
                )
        if force_pseudo_inverse:
            self.delta_raw = (torch.linalg.pinv(tensor_s) @ tensor_m).t()

        assert self.delta_raw is not None, "self.delta_raw should be computed by now."
        assert (
            self.delta_raw.isnan().sum() == 0
        ), f"The optimal delta should not contain NaN values for {self.name}."
        self.parameter_update_decrease = torch.trace(tensor_m @ self.delta_raw)
        if self.parameter_update_decrease < 0:
            warn(
                f"The parameter update decrease should be positive, "
                f"but got {self.parameter_update_decrease=} for layer {self.name}."
            )
            if not force_pseudo_inverse:
                warn(
                    f"Trying to use the pseudo-inverse for {self.name} with torch.float64."
                )
                return self.compute_optimal_delta(
                    update=update, dtype=torch.float64, force_pseudo_inverse=True
                )
            else:
                warn(
                    f"Failed to compute the optimal delta for {self.name}, set"
                    f"delta to zero."
                )
                self.delta_raw = torch.zeros_like(self.delta_raw)
        self.delta_raw = self.delta_raw.to(dtype=saved_dtype)

        assert self.delta_raw.shape[0] == self.out_channels, (
            f"delta_raw should have shape ({self.out_features=},...)"
            f"but got {self.delta_raw.shape=}"
        )
        if self.use_bias:
            assert self.delta_raw.shape[1] == (
                self.in_channels * self.kernel_size[0] * self.kernel_size[1] + 1
            ), (
                f"delta_raw should have shape (..., {self.in_channels * self.kernel_size[0] * self.kernel_size[1] + 1=}) "
                f"but got (..., {self.delta_raw.shape[1]})"
            )
            delta_weight = self.delta_raw[:, :-1]
            delta_bias = self.delta_raw[:, -1]
        else:
            assert self.delta_raw.shape[1] == (
                self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            ), (
                f"delta_raw should have shape (..., {self.in_channels * self.kernel_size[0] * self.kernel_size[1]=})"
                f"but got {self.delta_raw.shape=}"
            )
            delta_weight = self.delta_raw
            delta_bias = None

        delta_weight = delta_weight.reshape(
            self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )

        if update:
            self.optimal_delta_layer = self.layer_of_tensor(delta_weight, delta_bias)
        return delta_weight, delta_bias, self.parameter_update_decrease

    def compute_optimal_added_parameters(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        dtype: torch.dtype = torch.float32,
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

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
            optimal added weights (alpha weights, alpha bias, omega) and eigenvalues lambda
        """
        if self.delta_raw is None:
            self.compute_optimal_delta()
        try:
            matrix_n = self.tensor_n
        except AttributeError as e:
            raise AttributeError(
                "It seems that the tensor N is not accessible. I have no idea why this occurs sometimes."
            ) from e

        assert self.previous_module, (
            f"No previous module for {self.name}."
            "Therefore neuron addition is not possible."
        )
        matrix_s = self.tensor_s_growth()

        saved_dtype = matrix_s.dtype
        if matrix_n.dtype != dtype:
            matrix_n = matrix_n.to(dtype=dtype)
        if matrix_s.dtype != dtype:
            matrix_s = matrix_s.to(dtype=dtype)
        alpha, omega, self.eigenvalues_extension = compute_optimal_added_parameters(
            matrix_s=matrix_s,
            matrix_n=matrix_n,
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
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

        alpha = alpha.to(dtype=saved_dtype)
        omega = omega.to(dtype=saved_dtype)
        self.eigenvalues_extension = self.eigenvalues_extension.to(dtype=saved_dtype)

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

        if self.input_size != (-1, -1) and self.input.shape[-2:] != self.input_size:
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
