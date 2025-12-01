import types
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


class Conv2dMergeGrowingModule(MergeGrowingModule):
    def __init__(
        self,
        in_channels: int,
        input_size: int | tuple[int, int],
        next_kernel_size: int | tuple[int, int],
        post_merge_function: torch.nn.Module = torch.nn.Identity(),
        previous_modules: list[GrowingModule | MergeGrowingModule] | None = None,
        next_modules: list[GrowingModule | MergeGrowingModule] | None = None,
        allow_growing: bool = False,
        input_volume: int | None = None,
        device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        self.use_bias = True
        self.in_channels: int = in_channels
        self._input_volume = input_volume
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size: tuple[int, int] = input_size
        if isinstance(next_kernel_size, int):
            next_kernel_size = (next_kernel_size, next_kernel_size)
        self.kernel_size: tuple[int, int] = next_kernel_size
        super(Conv2dMergeGrowingModule, self).__init__(
            post_merge_function=post_merge_function,
            previous_modules=previous_modules,
            next_modules=next_modules,
            allow_growing=allow_growing,
            tensor_s_shape=(
                in_channels * next_kernel_size[0] * next_kernel_size[1] + self.use_bias,
                in_channels * next_kernel_size[0] * next_kernel_size[1] + self.use_bias,
            ),
            device=device,
            name=name,
        )

    @property
    def input_volume(self) -> int:
        if self._input_volume is not None:
            return self._input_volume
        if self.input_size is not None:
            return self.input_size[0] * self.input_size[1] * self.in_channels
        if len(self.previous_modules) <= 0:
            warn(
                "Cannot derive the number of features of Conv2dMergeGrowingModule "
                "without setting at least one previous module"
            )
            return -1
        return self.previous_modules[0].output_volume

    @property
    def output_volume(self) -> int:
        return self.input_volume

    @property
    def out_channels(self) -> int:
        return self.in_channels

    @property
    def in_features(self) -> int:
        return self.in_channels

    @property
    def out_features(self) -> int:
        return self.in_channels

    @property
    def output_size(self) -> tuple[int, int]:
        return self.input_size  # TODO: check for exceptions!

    @property
    def padding(self) -> tuple[int, int]:
        if len(self.next_modules) <= 0:
            warn(
                "Cannot derive the padding of Conv2dMergeGrowingModule without setting "
                "at least one next module"
            )
            return (0, 0)
        elif isinstance(self.next_modules[0], Conv2dGrowingModule):
            return self.next_modules[0].layer.padding
        elif isinstance(self.next_modules[0], Conv2dMergeGrowingModule):
            return self.next_modules[0].padding
        elif isinstance(
            self.next_modules[0], (LinearGrowingModule, LinearMergeGrowingModule)
        ):
            return ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        else:
            raise NotImplementedError

    @property
    def stride(self) -> tuple[int, int]:
        if len(self.next_modules) <= 0:
            warn(
                "Cannot derive the stride of Conv2dMergeGrowingModule without setting "
                "at least one next module"
            )
            return (1, 1)
        elif isinstance(self.next_modules[0], Conv2dGrowingModule):
            return self.next_modules[0].layer.stride
        elif isinstance(self.next_modules[0], Conv2dMergeGrowingModule):
            return self.next_modules[0].stride
        elif isinstance(
            self.next_modules[0], (LinearGrowingModule, LinearMergeGrowingModule)
        ):
            return (1, 1)
        else:
            raise NotImplementedError

    @property
    def dilation(self) -> tuple[int, int]:
        if len(self.next_modules) <= 0:
            warn(
                "Cannot derive the dilation of Conv2dMergeGrowingModule without setting "
                "at least one next module"
            )
            return (1, 1)
        elif isinstance(self.next_modules[0], Conv2dGrowingModule):
            return self.next_modules[0].layer.dilation
        elif isinstance(self.next_modules[0], Conv2dMergeGrowingModule):
            return self.next_modules[0].dilation
        elif isinstance(
            self.next_modules[0], (LinearGrowingModule, LinearMergeGrowingModule)
        ):
            return (1, 1)
        else:
            raise NotImplementedError

    @property
    def unfolded_extended_activity(self) -> torch.Tensor:
        """
        Return the unfolded activity extended with a channel of ones if the bias is used.
        Shape := (batch_size, D, L) where
        D = in_channels * kernel_size[0] * kernel_size[1] [+1 if bias]
        L = number of output spatial locations (e.g., H x W)

        Returns
        -------
        torch.Tensor
            unfolded activity extended
        """
        unfolded_activity = torch.nn.functional.unfold(
            self.activity,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        if self.use_bias:
            return torch.cat(
                (
                    unfolded_activity,
                    torch.ones(
                        unfolded_activity.shape[0],
                        1,
                        unfolded_activity.shape[2],
                        device=self.device,
                    ),
                ),
                dim=1,
            )
        else:
            return unfolded_activity

    def set_next_modules(
        self, next_modules: list[GrowingModule | MergeGrowingModule]
    ) -> None:
        if self.tensor_s is not None and self.tensor_s.samples > 0:
            warn(
                f"You are setting the next modules of {self.name} with a "
                f"non-empty tensor S."
            )

        self.next_modules = next_modules if next_modules else []

        # For Conv2d modules, check kernel size compatibility
        for module in self.next_modules:
            if isinstance(module, Conv2dGrowingModule):
                assert tuple(module.kernel_size) == tuple(self.kernel_size), (
                    f"Kernel size of next Conv2d modules {module.kernel_size} must match "
                    f"this module's kernel_size {self.kernel_size} (error in {self.name})"
                )

            if isinstance(module, (Conv2dGrowingModule, Conv2dMergeGrowingModule)):
                assert module.in_channels == self.out_channels, (
                    f"Next module input channels {module.in_channels} should match "
                    f"{self.out_channels=}"
                )
                # assert module.input_volume == self.output_volume, f"Next module input
                # volume {module.input_volume} should match {self.output_volume=}"
            elif isinstance(module, (LinearGrowingModule, LinearMergeGrowingModule)):
                assert module.in_features == self.output_volume, (
                    f"Next module input features {module.in_features} should match "
                    f"{self.output_volume=}"
                )
            else:
                raise NotImplementedError(
                    f"All next modules must be instances of Conv2dGrowingModule, "
                    f"Conv2dMergeGrowingModule, LinearGrowingModule or "
                    f"LinearMergeGrowingModule (error in {self.name})."
                )

    def set_previous_modules(
        self, previous_modules: list[MergeGrowingModule | GrowingModule]
    ) -> None:
        if self.previous_tensor_s is not None and self.previous_tensor_s.samples > 0:
            warn(
                f"You are setting the previous modules of {self.name} with a "
                f"non-empty previous tensor S."
            )
        if self.previous_tensor_m is not None and self.previous_tensor_m.samples > 0:
            warn(
                f"You are setting the previous modules of {self.name} with a "
                f"non-empty previous tensor M."
            )

        self.previous_modules = previous_modules if previous_modules else []

        # Then check kernel size constraints for all Conv2d modules
        # if len(prev_list) > 0:
        # first_ks = tuple(prev_list[0].kernel_size)
        # assert all(
        #     tuple(m.kernel_size) == first_ks for m in prev_list
        # ), (
        #     f"All previous modules must have the same kernel_size "
        #     f"(error in {self.name}). Got {[m.kernel_size for m in prev_list]}"
        # )
        # assert (
        #     tuple(self.kernel_size) == first_ks
        # ), (
        #     f"Kernel size of previous modules {first_ks} must match "
        #     f"this module's kernel_size {self.kernel_size} (error in {self.name})."
        # )

        self.total_in_features = 0
        for module in self.previous_modules:
            if not isinstance(module, (Conv2dGrowingModule, Conv2dMergeGrowingModule)):
                raise TypeError(
                    "The previous modules must be Conv2dGrowingModule or "
                    "Conv2dMergeGrowingModule."
                )

            if module.out_channels != self.in_channels:
                raise ValueError(
                    "The input channels must match the output channels of "
                    "the previous modules."
                )
            if isinstance(module, Conv2dGrowingModule):
                if module.output_volume != self.input_volume:
                    raise ValueError(
                        f"The output volume of the previous modules "
                        f"{module.output_volume} should match the "
                        f"input volume {self.input_volume=}."
                    )
                self.total_in_features += module.in_features + module.use_bias

        if self.total_in_features > 0:
            if self.input_size is None:
                self.input_size = (
                    self.previous_modules[0].out_width,
                    self.previous_modules[0].out_height,
                )
            self.previous_tensor_s = TensorStatistic(
                (
                    self.total_in_features,
                    self.total_in_features,
                ),
                device=self.device,
                name=f"S[-1]({self.name})",
                update_function=self.compute_previous_s_update,
            )
            self.previous_tensor_m = TensorStatistic(
                (self.total_in_features, self.in_channels),
                device=self.device,
                name=f"M[-1]({self.name})",
                update_function=self.compute_previous_m_update,
            )
        else:
            self.previous_tensor_s = None
            self.previous_tensor_m = None

    def construct_full_activity(self) -> torch.Tensor:
        """Construct the full activity tensor B from the input of all previous modules.
        B = (B_1, B_2, ..., B_k) in (n, )

        Returns
        -------
        torch.Tensor
            _description_
        """
        assert self.previous_modules, f"No previous modules for {self.name}."
        n = self.previous_modules[0].input.shape[0]
        nb_patch = int(self.previous_modules[0].output_volume / self.in_channels)
        full_activity = torch.ones(
            (
                n,
                self.total_in_features,
                nb_patch,
            ),
            device=self.device,
        )
        current_index = 0
        for module in self.previous_modules:
            full_activity[
                :, current_index : current_index + module.in_features + module.use_bias, :
            ] = module.unfolded_extended_input
            # (n, in_channels*ks0*ks1+bias, w_out*h_out)
            current_index += module.in_features + module.use_bias
        return full_activity

    def compute_previous_s_update(self) -> tuple[torch.Tensor, int]:
        full_activity = self.construct_full_activity()
        return (
            torch.einsum(
                "iam, ibm -> ab",
                full_activity,
                full_activity,
            ),
            full_activity.shape[0],
        )

    def compute_previous_m_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M for the input of all previous modules.
        B: full activity tensor
        M = dLoss/dA^T B

        Returns
        -------
        torch.Tensor
            update of the tensor M
        int
            number of samples used to compute the update
        """
        full_activity = (
            self.construct_full_activity()
        )  # (n, total_in_parameters, W_out*H_out)
        desired_activation = self.pre_activity.grad.flatten(start_dim=-2)
        return (
            torch.einsum("iam, icm -> ac", full_activity, desired_activation),
            self.input.shape[0],
        )

    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of tensor S based on the unfolded activity tensor.

        This method captures the second-order statistics from the output
        activity of the previous convolutional layers, formatted as an unfolded tensor.

        Depending on the type of the following (next) module, the tensor S is computed
        as follows:

        - If the next module is a `Conv2dGrowingModule`:
            The unfolded activity tensor is 3D: (batch_size, D, L), where
                - D = output_channels * kernel_size^2 [+1 if bias]
                - L = number of output spatial locations (e.g., H x W)
            Then S is computed via:
                S = ∑_{i,m} A[i,:,m] A[i,:,m]^T = einsum("iam, ibm -> ab", A, A)

        - If the next module is a `LinearGrowingModule`:
            The unfolded activity is treated as a flattened matrix of
            shape (batch_size, D), and S is computed via:
                S = ∑_{i} A[i]^T A[i] = einsum("ij, ik -> jk", A, A)

        Returns
        -------
        torch.Tensor
            second-order update matrix S of shape (D, D), where D depends on whether
            the next module is convolutional or linear.
        int
            batch size used in the computation.

        Raises
        ------
        NotImplementedError
            ff the type of the next module is unsupported for S computation.
        """
        assert self.store_activity, (
            f"The activity must be stored to compute the update of S. "
            f"(error in {self.name})"
        )
        assert self.activity is not None, (
            f"The activity must be stored to compute the update of S. "
            f"(error in {self.name})"
        )

        batch_size = self.activity.shape[0]
        unfolded_activity = self.unfolded_extended_activity

        update = torch.einsum("iam, ibm -> ab", unfolded_activity, unfolded_activity)

        return update, batch_size

    def update_size(self) -> None:
        if len(self.previous_modules) > 0:
            new_channels = self.previous_modules[0].out_channels
            self.in_channels = new_channels
        self.total_in_features = self.sum_in_features(with_bias=True)

        if self.total_in_features > 0:
            if self._input_volume is not None:
                self._input_volume = None  # reset calculation of input volume
            if self.tensor_s is None or self.tensor_s._shape != (
                self.in_channels * self.kernel_size[0] * self.kernel_size[1]
                + self.use_bias,
                self.in_channels * self.kernel_size[0] * self.kernel_size[1]
                + self.use_bias,
            ):
                self.tensor_s = TensorStatistic(
                    (
                        self.in_channels * self.kernel_size[0] * self.kernel_size[1]
                        + self.use_bias,
                        self.in_channels * self.kernel_size[0] * self.kernel_size[1]
                        + self.use_bias,
                    ),
                    update_function=self.compute_s_update,
                    name=f"S({self.name})",
                )
            if self.previous_tensor_s is None or self.previous_tensor_s._shape != (
                self.total_in_features,
                self.total_in_features,
            ):
                self.previous_tensor_s = TensorStatistic(
                    (
                        self.total_in_features,
                        self.total_in_features,
                    ),
                    device=self.device,
                    name=f"S[-1]({self.name})",
                    update_function=self.compute_previous_s_update,
                )
            if self.previous_tensor_m is None or self.previous_tensor_m._shape != (
                self.total_in_features,
                self.in_channels,
            ):
                self.previous_tensor_m = TensorStatistic(
                    (self.total_in_features, self.in_channels),
                    device=self.device,
                    name=f"M[-1]({self.name})",
                    update_function=self.compute_previous_m_update,
                )
        else:
            self.previous_tensor_s = None
            self.previous_tensor_m = None


class Conv2dGrowingModule(GrowingModule):
    _layer_type = torch.nn.Conv2d
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
        input_size: tuple[int, int] | None = None,
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
        self.layer: torch.nn.Conv2d
        self.kernel_size = self.layer.kernel_size

        self._input_size: tuple[int, int] | None = input_size
        self.use_bias = use_bias

        self.layer.forward = types.MethodType(self.__make_safe_forward(), self.layer)

    # Information functions
    # TODO: implement activation_gradient ?
    # this function is used to estimate the F.O. improvement of the loss after the
    # extension of the network however this won't work if we do not have only the
    # activation function as the post_layer_function

    @property
    def in_channels(self):
        return self.layer.in_channels

    @property
    def out_channels(self):
        return self.layer.out_channels

    @property
    def padding(self):
        return self.layer.padding

    @padding.setter
    def padding(self, value):
        self.layer.padding = value

    @property
    def dilation(self):
        return self.layer.dilation

    @property
    def stride(self):
        return self.layer.stride

    def __out_dimension(self, dim: int) -> int:
        return (
            int(
                (self.input_size[dim] - self.kernel_size[dim] + 2 * self.padding[dim])
                / self.stride[dim]
            )
            + 1
        )

    @property
    def out_width(self) -> int:
        """Compute the width of the output feature map.

        Returns
        -------
        int
            output feature map width
        """
        return self.__out_dimension(0)

    @property
    def out_height(self) -> int:
        """Compute the height of the output feature map.

        Returns
        -------
        int
            output feature map height
        """
        return self.__out_dimension(1)

    @property
    def input_volume(self) -> int:
        """Calculate total number of elements in the input tensor.

        Returns
        -------
        int
            total number of input elements
        """
        return self.in_channels * self.input_size[0] * self.input_size[1]

    @property
    def output_volume(self) -> int:
        """Compute total number of elements in the output tensor

        Returns
        -------
        int
            total number of output elements
        """
        return self.out_channels * self.out_width * self.out_height

    @property
    def in_features(self) -> int:
        """
        Return the number of input features (fan-in) of this convolutional layer when
        cast as a linear layer.

        Concretely, this integer equals:

            in_channels * kernel_height * kernel_width

        It represents the size of the input vector obtained by unfolding the convolutional
        input (i.e. the 'fan_in' for a flattened convolution). We use the name
        ``in_features`` (instead of ``fan_in``) to remain consistent with the naming
        conventions used in :class:`LinearGrowingModule`.

        Returns
        -------
        int
            number of input features (fan-in)
        """
        return self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    @property
    def out_features(self) -> int:
        """
        Return the number of output features (fan-out) of this convolutional layer when
        cast as a linear layer.

        Concretely, this integer equals:

            out_channels

        Returns
        -------
        int
            number of output features (fan-out)
        """
        return self.out_channels

    @property
    def unfolded_extended_input(self) -> torch.Tensor:
        """
        Return the unfolded input extended with a channel of ones if the bias is used.

        Returns
        -------
        torch.Tensor
            unfolded input extended
        """
        # TODO: maybe we could compute it only once if we use it multiple times
        # (e.g. in S, S_prev, M, M_prev, P...)
        unfolded_input = torch.nn.functional.unfold(
            self.input,
            self.kernel_size,
            padding=self.padding,  # pyright: ignore[reportArgumentType]
            stride=self.stride,
            dilation=self.dilation,
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

    def __make_safe_forward(self):
        def _forward(conv_self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
            if self.out_channels == 0:
                n = input.shape[0]
                return torch.zeros(
                    n,
                    0,
                    self.out_height,
                    self.out_width,
                    device=self.device,
                    requires_grad=True,
                )
            if self.in_channels == 0:
                n = input.shape[0]
                return torch.zeros(
                    n,
                    self.out_channels,
                    self.out_height,
                    self.out_width,
                    device=self.device,
                    requires_grad=True,
                )
            return torch.nn.Conv2d.forward(conv_self, input)

        return _forward

    # Statistics computation
    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        r"""
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
            assert isinstance(desired_activation, torch.Tensor), (
                f"The gradient of the pre-activity must be a torch.Tensor "
                f"(error in {self.name})."
            )
        desired_activation = desired_activation.flatten(start_dim=-2)

        return (
            torch.einsum(
                "iam, icm -> ac", self.unfolded_extended_input, desired_activation
            ),
            self.input.shape[0],
        )

    # Layer edition
    def layer_of_tensor(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        force_bias: bool = True,
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
        force_bias: bool
            if True, the created layer require a bias
            if `self.use_bias` is True

        Returns
        -------
        torch.nn.Linear
            layer with the same characteristics
        """
        if force_bias:
            assert self.use_bias is (bias is not None), (
                f"The new layer should have a bias ({bias is not None=}) if and only if "
                f"the main layer bias ({self.use_bias =}) is not None."
            )
        for i in (0, 1):
            assert (
                weight.shape[2 + i] == self.layer.kernel_size[i]
            ), f"{weight.shape[2 + i]=} should be equal to {self.layer.kernel_size[i]=}"

        new_layer = torch.nn.Conv2d(
            weight.shape[1],
            weight.shape[0],
            bias=self.use_bias,
            device=self.device,
            kernel_size=self.kernel_size,  # pyright: ignore[reportArgumentType]
            stride=self.stride,  # pyright: ignore[reportArgumentType]
            padding=self.padding,  # pyright: ignore[reportArgumentType]
            dilation=self.dilation,  # pyright: ignore[reportArgumentType]
        )
        new_layer.weight = torch.nn.Parameter(weight)
        if bias is not None:
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
        ), f"{weight.shape[0]=} should be equal to {self.out_channels=}"
        for i in (0, 1):
            assert (
                weight.shape[2 + i] == self.layer.kernel_size[i]
            ), f"{weight.shape[2 + i]=} should be equal to {self.layer.kernel_size[i]=}"

        # TODO: check this is working
        self.layer = self.layer_of_tensor(
            weight=torch.cat((self.weight, weight), dim=1), bias=self.bias
        )

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
            ), "The bias of the extension should be provided because the layer has a bias"
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0),
                bias=torch.cat((self.layer.bias, bias), dim=0),
            )
        else:
            if bias is not None:
                warn(
                    "The bias of the extension should not be provided "
                    "because the layer has no bias.",
                    UserWarning,
                )
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0), bias=None
            )

        self.tensor_m = TensorStatistic(
            (self.in_channels + self.use_bias, self.out_channels),
            update_function=self.compute_m_update,
            name=self.tensor_m.name,
        )

    def update_input_size(
        self,
        input_size: tuple[int, int] | torch.Size | None = None,
        compute_from_previous: bool = False,
        force_update: bool = True,
    ) -> tuple[int, int] | None:
        """
        Update the input size of the layer.

        Either according to the parameter or the input currently stored.

        Parameters
        ----------
        input_size: tuple[int, int] | None
            new input size
        compute_from_previous: bool
            whether to compute the input size from the previous module
            assuming its output size won't be affected by the post-layer function
        force_update: bool
            whether to force the update even if the input size is already set
            (_input_size is not None)

        Returns
        -------
        tuple[int, int] | None
            updated input size if it could be computed, None otherwise
        """
        if input_size is not None:
            new_size = tuple(input_size)
        elif self.store_input and self.input is not None:
            new_size: tuple[int, ...] = tuple(self.input.shape[2:])
        elif not force_update and self._input_size is not None:
            return self._input_size
        elif (
            compute_from_previous
            and self.previous_module
            and (
                prev_input_size := self.previous_module.update_input_size(
                    force_update=False
                )
            )
        ):
            # we get it this way instead of self.previous_module.input_size
            # to avoid errors if the previous module input size can't be computed
            new_size = compute_output_shape_conv(
                prev_input_size, self.previous_module.layer
            )
        else:
            # if we cannot compute it, just return the current value
            return self._input_size

        if self._input_size is not None and new_size != self._input_size:
            warn(
                f"The input size of the layer {self.name} has changed "
                f"from {self._input_size} to {new_size}."
                f"This may lead to errors if the size of the tensor statistics "
                f"and of the mask tensor T are not updated."
            )

        assert (
            len(new_size) == 2
        ), f"The input size should be a tuple of two integers, but got {new_size=}."
        self._input_size = new_size
        return self._input_size

    def update_computation(self) -> None:
        """
        Update the computation of the layer.
        """
        self.update_input_size()
        super(Conv2dGrowingModule, self).update_computation()

    @staticmethod
    def get_fan_in_from_layer(layer: torch.nn.Conv2d) -> int:  # type: ignore
        """
        Get the fan_in (number of input features) from a given layer.

        Parameters
        ----------
        layer: torch.nn.Conv2d
            layer to get the fan_in from

        Returns
        -------
        int
            fan_in of the layer
        """
        assert isinstance(
            layer, torch.nn.Conv2d
        ), f"The layer should be a torch.nn.Conv2d but got {type(layer)}."
        return layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]

    def create_layer_in_extension(self, extension_size: int) -> None:
        """
        Create the layer input extension of given size.

        Parameters
        ----------
        extension_size: int
            size of the extension to create
        """
        # Create a conv2d layer for input extension
        self.extended_input_layer = torch.nn.Conv2d(
            extension_size,
            self.out_channels,
            kernel_size=self.kernel_size,  # pyright: ignore[reportArgumentType]
            stride=self.stride,  # pyright: ignore[reportArgumentType]
            padding=self.padding,  # pyright: ignore[reportArgumentType]
            dilation=self.dilation,  # pyright: ignore[reportArgumentType]
            bias=self.use_bias,
            device=self.device,
        )

    def create_layer_out_extension(self, extension_size: int) -> None:
        """
        Create the layer output extension of given size.

        Parameters
        ----------
        extension_size: int
            size of the extension to create
        """
        # Create a conv2d layer for output extension
        self.extended_output_layer = torch.nn.Conv2d(
            self.in_channels,
            extension_size,
            kernel_size=self.kernel_size,  # pyright: ignore[reportArgumentType]
            stride=self.stride,  # pyright: ignore[reportArgumentType]
            padding=self.padding,  # pyright: ignore[reportArgumentType]
            dilation=self.dilation,  # pyright: ignore[reportArgumentType]
            bias=self.use_bias,
            device=self.device,
        )


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
        input_size: tuple[int, int] | None = None,
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
            f"weight should have 2 dimensions or the last two dimensions "
            f"should be (1, 1), but got {weight.shape=}."
        )

        if weight.dim() == 2:
            weight = weight.unsqueeze(-1).unsqueeze(-1)

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
            kernel_size=self.kernel_size,  # pyright: ignore[reportArgumentType]
            stride=self.stride,  # pyright: ignore[reportArgumentType]
            padding=self.padding,  # pyright: ignore[reportArgumentType]
            dilation=self.dilation,  # pyright: ignore[reportArgumentType]
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
                f"Cannot compute the bordered unfolded input without a previous "
                f"module for {self.name}."
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
        input_size = self.update_input_size(compute_from_previous=True)
        assert input_size is not None, (
            f"The input size should be known to compute the bordered unfolded "
            f"input for {self.name}."
        )
        return apply_border_effect_on_unfolded(
            unfolded_tensor=self.previous_module.unfolded_extended_input,
            original_size=input_size,
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
            assert isinstance(desired_activation, torch.Tensor), (
                f"The gradient of the pre-activity must be a torch.Tensor "
                f"(error in {self.name})."
            )
        desired_activation = desired_activation.flatten(start_dim=-2)

        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus M_{-2} is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement M_prev for LinearGrowingModule")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError(
                "TODO: implement M_prev for LinearMergeGrowingModule"
            )
        elif isinstance(self.previous_module, Conv2dGrowingModule):
            unfolded_extended_input = self.bordered_unfolded_extended_prev_input
            assert unfolded_extended_input.shape[0] == desired_activation.shape[0], (
                f"The number of samples is incoherent: "
                f"{unfolded_extended_input.shape[0]=} "
                f"and {desired_activation.shape[0]=} should be equal."
            )
            assert unfolded_extended_input.shape[2] == desired_activation.shape[2], (
                f"The number of features is incoherent: "
                f"{unfolded_extended_input.shape[2]=} "
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
                f"No previous module for {self.name}. Thus the cross covariance is "
                f"not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement cross cov for LinearGrowingModule")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError(
                "TODO: implement cross cov for LinearMergeGrowingModule"
            )
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
        assert self.cross_covariance() is not None, (
            f"The cross covariance should be computed before the "
            f"tensor N for {self.name}."
        )
        assert isinstance(self.cross_covariance(), torch.Tensor), (
            f"The cross covariance should be a tensor for {self.name}, is "
            f"{type(self.cross_covariance())}."
        )
        assert (
            self.cross_covariance().shape[1]
            == self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            + self.use_bias
        ), (
            f"The cross covariance should have shape "
            f"(..., {self.in_channels * self.kernel_size[0] * self.kernel_size[1] + self.use_bias})"  # noqa: E501
            f" but got {self.cross_covariance().shape}."
        )
        assert (
            self.delta_raw is not None
        ), f"The optimal delta should be computed before the tensor N for {self.name}."
        assert isinstance(self.delta_raw, torch.Tensor), (
            f"The optimal delta should be a tensor for {self.name}, "
            f"is {type(self.delta_raw)}."
        )
        _expected_delta_shape_1 = (
            self.in_channels * self.kernel_size[0] * self.kernel_size[1] + self.use_bias
        )
        assert self.delta_raw.shape[1] == _expected_delta_shape_1, (
            f"Expected delta_raw.shape[1] == "
            f"{_expected_delta_shape_1}, but got {self.delta_raw.shape[1]} "
            f"(full shape: {self.delta_raw.shape})."
        )
        assert self.delta_raw.shape[0] == self.out_channels, (
            f"Expected delta_raw.shape[0] == {self.out_channels}, "
            f"but got {self.delta_raw.shape[0]}."
        )
        return -self.tensor_m_prev() + torch.einsum(
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
            threshold to consider an eigenvalue as zero in the square root of the
            inverse of S
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
            optimal added weights (alpha weights, alpha bias, omega) and
            eigenvalues lambda
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
        ), "omega should have the same number of output features as the layer."

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
            f"omega should have shape ({k}, {self.out_channels}, {self.kernel_size[0]}, "
            f"{self.kernel_size[1]}) but got {omega.shape}."
        )
        assert (
            alpha.shape[0] == k
        ), f"alpha should have shape ({k}, ...) but got {alpha.shape}."

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
        input_size: tuple[int, int] | None = None,
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
                f"The computation of the previous masked unfolded activation is not "
                f"implemented yet for {type(self.previous_module)} as previous module."
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
            assert isinstance(desired_activation, torch.Tensor), (
                f"The gradient of the pre-activity must be a torch.Tensor "
                f"(error in {self.name})."
            )
        desired_activation = desired_activation.flatten(start_dim=-2)

        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus M_{-2} is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement M_prev for LinearGrowingModule")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError(
                "TODO: implement M_prev for LinearMergeGrowingModule"
            )
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
        r"""
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
                f"No previous module for {self.name}. Thus the cross covariance"
                f" is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            raise NotImplementedError("TODO: implement cross cov for LinearGrowingModule")
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            raise NotImplementedError(
                "TODO: implement cross cov for LinearMergeGrowingModule"
            )
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
        Override `tensor_s_growth` to redirect to `self._tensor_s_growth` instead
        of `self.previous_module.tensor_s`.
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
        assert self.cross_covariance() is not None, (
            f"The cross covariance should be computed before "
            f"the tensor N for {self.name}."
        )
        assert (
            self.delta_raw is not None
        ), f"The optimal delta should be computed before the tensor N for {self.name}."
        return -self.tensor_m_prev() + torch.einsum(
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
            threshold to consider an eigenvalue as zero in the square root of
            the inverse of S
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
            optimal added weights (alpha weights, alpha bias, omega) and
            eigenvalues lambda
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
        ), "omega should have the same number of output features as the layer."

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
            f"omega should have shape ({k}, {self.out_channels}, {self.kernel_size[0]}, "
            f"{self.kernel_size[1]}) but got {omega.shape}."
        )
        assert (
            alpha.shape[0] == k
        ), f"alpha should have shape ({k}, ...) but got {alpha.shape}."

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
