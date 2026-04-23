import warnings
from typing import Any, Iterator, Literal, Protocol, get_args, runtime_checkable

import numpy as np
import torch

from gromo.config.loader import load_config
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import (
    compute_optimal_added_parameters,
    optimal_delta,
)
from gromo.utils.utils import (
    compute_tensor_stats,
    get_correct_device,
    known_activations_zero_plus_gradient,
)


# Constants for gradient computation
GRADIENT_COMPUTATION_EPSILON = 1e-5  # Small perturbation for gradient computation


class MergeGrowingModule(torch.nn.Module):
    """
    Module to connect multiple modules with an merge operation.
    This module does not perform the merge operation, it is done by the user.
    """

    def __init__(
        self,
        post_merge_function: torch.nn.Module = torch.nn.Identity(),
        previous_modules: list["MergeGrowingModule | GrowingModule"] | None = None,
        next_modules: list["MergeGrowingModule | GrowingModule"] | None = None,
        allow_growing: bool = False,
        tensor_s_shape: tuple[int, int] | None = None,
        device: torch.device | None = None,
        name: str | None = None,
    ) -> None:
        super(MergeGrowingModule, self).__init__()
        self._name = name
        self.name = (
            self.__class__.__name__
            if name is None
            else f"{self.__class__.__name__}({name})"
        )
        self._config_data, _ = load_config()
        self.device = get_correct_device(self, device)

        self.post_merge_function: torch.nn.Module = post_merge_function
        if self.post_merge_function:
            self.post_merge_function = self.post_merge_function.to(self.device)
        self._allow_growing = allow_growing

        self.store_input = 0
        self.input = None

        self.store_activity = 0
        self.activity = None

        self.tensor_s = TensorStatistic(
            tensor_s_shape,
            update_function=self.compute_s_update,
            device=self.device,
            name=f"S({self.name})",
        )

        self.previous_tensor_s: TensorStatistic | None = None
        self.previous_tensor_m: TensorStatistic | None = None

        self.previous_modules: list[MergeGrowingModule | GrowingModule] = []
        self.set_previous_modules(previous_modules)
        self.next_modules: list[MergeGrowingModule | GrowingModule] = []
        self.set_next_modules(next_modules)

    @property
    def input_volume(self) -> int:
        """Expected input volume

        Returns
        -------
        int
            input volume

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def output_volume(self) -> int:
        """Expected output volume

        Returns
        -------
        int
            output volume

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def number_of_successors(self) -> int:
        """Get the number of succeeding modules

        Returns
        -------
        int
            number of next modules
        """
        return len(self.next_modules)

    @property
    def number_of_predecessors(self) -> int:
        """Get the number of preceding modules

        Returns
        -------
        int
            number of previous modules
        """
        return len(self.previous_modules)

    def grow(self):
        """
        Function to call after growing previous or next modules.
        """
        # mainly used to reset the shape of the tensor S, M, prev S and prev M
        self.set_next_modules(self.next_modules)
        self.set_previous_modules(self.previous_modules)

    def add_next_module(self, module: "MergeGrowingModule | GrowingModule") -> None:
        """
        Add a module to the next modules of the current module.

        Parameters
        ----------
        module: MergeGrowingModule | GrowingModule
            next module to add
        """
        self.next_modules.append(module)
        self.set_next_modules(
            self.next_modules
        )  # TODO: maybe it is possible to avoid this

    def add_previous_module(self, module: "MergeGrowingModule | GrowingModule") -> None:
        """
        Add a module to the previous modules of the current module.

        Parameters
        ----------
        module: MergeGrowingModule | GrowingModule
            previous module to add
        """
        self.previous_modules.append(module)
        self.set_previous_modules(self.previous_modules)

    def set_next_modules(
        self, next_modules: list["MergeGrowingModule | GrowingModule"]
    ) -> None:
        """
        Set the next modules of the current module.

        Parameters
        ----------
        next_modules: list[MergeGrowingModule | GrowingModule]
            list of next modules

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def set_previous_modules(
        self, previous_modules: list["MergeGrowingModule | GrowingModule"]
    ) -> None:
        """
        Set the previous modules of the current module.

        Parameters
        ----------
        previous_modules: list[MergeGrowingModule | GrowingModule]
            list of previous modules

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def __str__(self, verbose=1):
        if verbose == 0:
            return f"{self.__class__.__name__} module."
        elif verbose == 1:
            previous_modules = (
                len(self.previous_modules) if self.previous_modules else "no"
            )
            next_modules = len(self.next_modules) if self.next_modules else "no"
            return (
                f"{self.__class__.__name__} module with {previous_modules} "
                f"previous modules and {next_modules} next modules."
            )
        elif verbose >= 2:
            txt = [
                f"{self.__class__.__name__} module.",
                f"\tPrevious modules: {self.previous_modules}",
                f"\tNext modules: {self.next_modules}",
                f"\tPost merge function: {self.post_merge_function}",
                f"\tAllow growing: {self._allow_growing}",
                f"\tStore input: {self.store_input}",
                f"\tStore activity: {self.store_activity}",
                f"\tTensor S: {self.tensor_s}",
                f"\tPrevious tensor S: {self.previous_tensor_s}",
                f"\tPrevious tensor M: {self.previous_tensor_m}",
            ]
            return "\n".join(txt)
        else:
            raise ValueError(f"verbose={verbose} is not a valid value.")

    def __repr__(self, *args: Any, **kwargs: Any):
        return self.__str__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.
        If needed, store the activity and pre-activity tensors.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        for t in (self.tensor_s, self.previous_tensor_s, self.previous_tensor_m):
            if t:
                t.updated = False

        if self.store_input > 0:
            self.input = x
            self.input.retain_grad()

        if self.post_merge_function and (x is not None):
            y = self.post_merge_function(x)
        else:
            y = x

        if self.store_activity > 0:
            self.activity = y.detach()
            self.tensor_s.updated = False  # reset the update flag

        return y

    @property
    def pre_activity(self) -> torch.Tensor:
        """Get the pre activity of the layer

        Returns
        -------
        torch.Tensor
            pre activity tensor
        """
        return self.input

    def projected_v_goal(self) -> torch.Tensor:
        """
        Compute the projected gradient of the goal with respect to the activity of the layer.

        dLoss/dA_proj := dLoss/dA - dW B[-1] where A is the pre-activation vector of the
        layer, and dW the optimal delta for all the previous layers

        Returns
        -------
        torch.Tensor
            projected gradient of the goal with respect to the activity of the next layer
            dLoss/dA - dW B[-1]
        """
        v_proj = self.pre_activity.grad.clone().detach()
        for module in self.previous_modules:
            if isinstance(module, GrowingModule):
                v_proj -= module.optimal_delta_layer(module.input)
            elif isinstance(module, MergeGrowingModule):
                for prev_module in module.previous_modules:
                    v_proj -= prev_module.optimal_delta_layer(prev_module.input)

        return v_proj

    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S. Should be added to the type of layer.

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def compute_previous_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S for the input of all previous modules.

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def compute_previous_m_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M for the input of all previous modules.

        Returns
        -------
        torch.Tensor
            update of the tensor M
        int
            number of samples used to compute the update

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def init_computation(self) -> None:
        """
        Initialize the computation of the optimal added parameters.
        """
        self.store_input = True
        self.store_activity = True
        self.tensor_s.init()
        for module in self.previous_modules:
            module.store_input = True
            module.store_pre_activity = True
        if self.previous_tensor_s is not None:
            self.previous_tensor_s.init()
        if self.previous_tensor_m is not None:
            self.previous_tensor_m.init()

    def update_computation(self) -> None:
        """
        Update the computation of the optimal added parameters.
        """
        self.tensor_s.update()
        if self.previous_tensor_s is not None:
            self.previous_tensor_s.update()
        if self.previous_tensor_m is not None:
            self.previous_tensor_m.update()

    def reset_computation(self) -> None:
        """
        Reset the computation of the optimal added parameters.
        """
        self.store_input = False
        self.store_activity = False
        self.tensor_s.reset()
        for module in self.previous_modules:
            module.store_input = False
            module.store_pre_activity = False
        if self.previous_tensor_s is not None:
            self.previous_tensor_s.reset()
        if self.previous_tensor_m is not None:
            self.previous_tensor_m.reset()

    def delete_update(self, include_previous: bool = False) -> None:
        """
        Delete the update of the optimal added parameters.
        """
        self.activity = None
        self.input = None

        if include_previous:
            for previous_module in self.previous_modules:
                if isinstance(previous_module, GrowingModule):
                    previous_module.delete_update(
                        include_previous=False, delete_output=True
                    )

    def compute_optimal_delta(
        self,
        update: bool = True,
        return_deltas: bool = False,
        force_pseudo_inverse: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """
        Compute the optimal delta for each previous layer using current S and M tensors.
        dW* = M S[-1]^-1 (if needed we use the pseudo-inverse)
        Compute dW* (and dBias* if needed) and update the optimal_delta_layer attribute.

        Parameters
        ----------
        update: bool, optional
            if True update the optimal delta layer attribute, by default True
        return_deltas: bool, optional
            if True return the deltas, by default False
        force_pseudo_inverse: bool, optional
            if True, use the pseudo-inverse to compute the optimal delta even if the, by default False
            matrix is invertible
        dtype: torch.dtype
            dtype for S and M during the computation

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]] | None
            optimal delta for the weights and the biases if needed
        """
        assert self.previous_tensor_s is not None, (
            f"No previous tensor S for {self.name}."
        )
        assert self.previous_tensor_m is not None, (
            f"No previous tensor M for {self.name}."
        )
        previous_tensor_s = self.previous_tensor_s()
        previous_tensor_m = self.previous_tensor_m()
        assert previous_tensor_s.shape[0] == self.total_in_features, (
            f"The inverse of S should have the same number of features as the input "
            f"of all previous modules. Expected {self.total_in_features}. Got {previous_tensor_s.shape[0]}."
        )
        assert previous_tensor_m.shape == (self.total_in_features, self.in_features), (
            f"The tensor M should have shape ({self.total_in_features}, {self.in_features}). "
            f"Got {previous_tensor_m.shape}."
        )
        delta, self.parameter_update_decrease = optimal_delta(
            previous_tensor_s,
            previous_tensor_m,
            dtype=dtype,
            force_pseudo_inverse=force_pseudo_inverse,
        )

        deltas = []
        current_index = 0
        for module in self.previous_modules:
            if isinstance(module, MergeGrowingModule):
                continue
            delta_w = delta[:, current_index : current_index + module.in_features]
            if module.use_bias:
                delta_b = delta[:, current_index + module.in_features]
            else:
                delta_b = None

            # change the shape of the delta_w and delta_b to match the layer
            delta_w = delta_w.reshape(*module.weight.shape)
            if update:
                module.optimal_delta_layer = module.layer_of_tensor(delta_w, delta_b)
            # elif isinstance(module, MergeGrowingModule):
            #     if update:
            #         if module.post_merge_function.is_non_linear():
            #             warnings.warn(
            #                 f"The previous module {module.name} is a MergeGrowingModule with a non-linear post merge function. "
            #                 f"The optimal delta may not be accurate.",
            #                 UserWarning,
            #             )
            #         else:
            #             module.set_optimal_delta_layers(delta_w, delta_b)

            if return_deltas:
                deltas.append((delta_w, delta_b))

            current_index += module.in_features + module.use_bias

        if return_deltas:
            return deltas
        else:
            return None

    def _grow_post_merge_function(self, extension_size: int) -> None:
        """Apply growth to sized activation functions

        Parameters
        ----------
        extension_size: int
            size of extension
        """
        if isinstance(self.post_merge_function, torch.nn.Sequential):
            for module in self.post_merge_function:
                if hasattr(module, "grow"):
                    module.grow(extension_size)  # type: ignore
        elif hasattr(self.post_merge_function, "grow"):
            self.post_merge_function.grow(extension_size)  # type: ignore

    def update_size(self) -> None:
        """
        Update the size of the module
        Check number of previous modules and update input channels and tensor sizes
        """
        if len(self.previous_modules) > 0:
            new_size: int = self.previous_modules[0].out_features
            self.in_features = new_size
        self.total_in_features = self.sum_in_features(with_bias=True)

        tensor_s_shape = (
            self.in_features + int(self.use_bias),
            self.in_features + int(self.use_bias),
        )
        if self.tensor_s._shape != tensor_s_shape:
            self.tensor_s = TensorStatistic(
                tensor_s_shape,
                update_function=self.compute_s_update,
                device=self.device,
                name=f"S({self.name})",
            )

        if self.total_in_features > 0:
            if self.previous_tensor_s._shape != (
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
            if self.previous_tensor_m._shape != (
                self.total_in_features,
                self.in_features,
            ):
                self.previous_tensor_m = TensorStatistic(
                    (self.total_in_features, self.in_features),
                    device=self.device,
                    name=f"M[-1]({self.name})",
                    update_function=self.compute_previous_m_update,
                )
        else:
            self.previous_tensor_s = None
            self.previous_tensor_m = None

    @property
    def number_of_parameters(self) -> int:
        """Get the number of parameters of the layer

        Returns
        -------
        int
            number of parameters
        """
        return 0

    def parameters(
        self,
        recurse: bool = True,  # noqa: ARG002
    ) -> Iterator[torch.nn.Parameter]:
        """Parameter iterator

        Parameters
        ----------
        recurse: bool, optional
            use recursion, by default True

        Returns
        -------
        Iterator[torch.nn.Parameter]
            parameters iterator
        """
        return iter([])

    def sum_in_features(self, with_bias: bool = False) -> int:
        """Count total in_features of previous modules

        Parameters
        ----------
        with_bias: bool, optional
            add bias to the sum, by default False

        Returns
        -------
        int
            sum of previous in_features
        """
        if with_bias:
            return sum(
                module.in_features + int(module.use_bias)
                for module in self.previous_modules
                if isinstance(module, GrowingModule)
            )
        return sum(
            module.in_features
            for module in self.previous_modules
            if isinstance(module, GrowingModule)
        )

    def sum_out_features(self) -> int:
        """Count total out_features of next modules

        Returns
        -------
        int
            sum of next out_features
        """
        return np.sum([module.out_features for module in self.next_modules])

    def update_scaling_factor(self, scaling_factor: torch.Tensor | float) -> None:
        """
        Update the scaling factor of all next modules and
        the output_extension_scaling of the previous modules.
        Does only work if previous and next modules are GrowingModule.

        Parameters
        ----------
        scaling_factor: torch.Tensor | float
            scaling factor to apply to the optimal delta

        Raises
        ------
        TypeError
            if the previous and next modules are not of type GrowingModule
        """
        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.item()
        for module in self.previous_modules:
            if isinstance(module, GrowingModule):
                module.output_extension_scaling = scaling_factor  # type: ignore
            else:
                raise TypeError(
                    f"Previous module must be a GrowingModule, got {type(module)}"
                )
        for module in self.next_modules:
            if isinstance(module, GrowingModule):
                # Use the legacy aggregated setter to fan out to all sub-factors
                # (matching the historical behaviour of update_scaling_factor).
                module.scaling_factor = scaling_factor  # type: ignore
            else:
                raise TypeError(
                    f"Next module must be a GrowingModule, got {type(module)}"
                )

    def __del__(self) -> None:
        # Delete previous GrowingModules
        for prev_module in list(self.previous_modules):
            if isinstance(prev_module, GrowingModule):
                prev_module.__del__()
            elif isinstance(prev_module, MergeGrowingModule):
                if self in prev_module.next_modules:
                    prev_module.next_modules.remove(self)
                    prev_module.update_size()
        self.previous_modules = []
        # Delete next GrowingModules
        for next_module in list(self.next_modules):
            if isinstance(next_module, GrowingModule):
                next_module.__del__()
            elif isinstance(next_module, MergeGrowingModule):
                if self in next_module.previous_modules:
                    next_module.previous_modules.remove(self)
                    next_module.update_size()
        self.next_modules = []


@runtime_checkable
class SupportsExtendedForward(Protocol):
    """Protocol for modules that provide an extended_forward method.

    Modules implementing this protocol can be used inside ``post_layer_function``
    of a :class:`GrowingModule` without requiring a separate
    ``extended_post_layer_function``.

    :meth:`extended_forward` receives both the *main* pre-activation ``x`` (N
    channels / features) and the *extension* pre-activation ``x_ext`` (M channels
    / features), and must return both processed tensors.  This mirrors the
    ``(activity, supplementary_activity)`` convention of
    :meth:`GrowingModule.extended_forward`.
    """

    def extended_forward(
        self, x: torch.Tensor | None, x_ext: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply the module to both the main and extension pre-activations.

        Parameters
        ----------
        x: torch.Tensor | None
            Main pre-activation tensor (N channels / features), or ``None``
            when the main path is irrelevant.
        x_ext: torch.Tensor | None
            Extension pre-activation tensor (M channels / features), or ``None``
            when there is no extension.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            ``(processed_x, processed_x_ext)`` — both tensors after applying
            the module, retaining their respective shapes.  ``None`` inputs
            propagate as ``None`` outputs.
        """
        ...


class GrowingModule(torch.nn.Module):
    """
    Abstract class for a Module of dynamic size

    Parameters
    ----------
    layer: torch.nn.Module
        layer of the module
    tensor_s_shape: tuple[int, int] | None
        shape of the tensor S
    tensor_m_shape: tuple[int, int] | None
        shape of the tensor M
    post_layer_function: torch.nn.Module, optional
        function to apply after the layer, by default torch.nn.Identity()
    extended_post_layer_function: torch.nn.Module | None, optional
        extended function to apply after the layer, by default None
    allow_growing: bool
        if True, the module can grow (require a previous GrowingModule)
    previous_module: torch.nn.Module | None
        previous module
    next_module: torch.nn.Module | None
        next module
    device: torch.device | None
        device to use
    name: str | None
        name of the module
    target_in_neurons: int | None, optional
        target fan-in size, by default None
    initial_in_neurons: int | None, optional
        initial fan-in size, by default None
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        tensor_s_shape: tuple[int, int] | None = None,
        tensor_m_shape: tuple[int, int] | None = None,
        post_layer_function: torch.nn.Module = torch.nn.Identity(),
        extended_post_layer_function: torch.nn.Module | None = None,
        allow_growing: bool = True,
        previous_module: torch.nn.Module | None = None,
        next_module: torch.nn.Module | None = None,
        device: torch.device | None = None,
        name: str | None = None,
        target_in_neurons: int | None = None,
        initial_in_neurons: int | None = None,
    ) -> None:
        if tensor_s_shape is None:
            warnings.warn(
                "The tensor S shape is not provided."
                "It will automatically be determined but we encourage to provide it.",
                UserWarning,
            )
        else:
            assert len(tensor_s_shape) == 2, "The shape of the tensor S must be 2D."
            assert tensor_s_shape[0] == tensor_s_shape[1], "The tensor S must be square."
            if tensor_m_shape is not None:
                assert tensor_s_shape[0] == tensor_m_shape[0], (
                    f"The input matrices S and M must have compatible shapes."
                    f"(got {tensor_s_shape=} and {tensor_m_shape=})"
                )

        super(GrowingModule, self).__init__()
        self._name = name
        self.name = (
            self.__class__.__name__
            if name is None
            else f"{self.__class__.__name__}({name})"
        )
        self.target_in_neurons = target_in_neurons
        self._initial_in_neurons = initial_in_neurons
        self._config_data, _ = load_config()
        self.device = get_correct_device(self, device)

        self.layer: torch.nn.Module = layer.to(self.device)
        # TODO: don't allow non-linearity if prev module is merge
        self.post_layer_function: torch.nn.Module = post_layer_function.to(self.device)
        if extended_post_layer_function is not None:
            self._has_explicit_extended_post_layer_function: bool = True
            warnings.warn(
                "The `extended_post_layer_function` parameter is deprecated and will be "
                "removed in a future version. Implement `extended_forward` on the modules "
                "used in `post_layer_function` instead (see `SupportsExtendedForward`).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.extended_post_layer_function: torch.nn.Module = (
                extended_post_layer_function.to(self.device)
            )
        else:
            self._has_explicit_extended_post_layer_function = False
            self.extended_post_layer_function = self.post_layer_function

        self._allow_growing = allow_growing
        assert not self._allow_growing or isinstance(
            previous_module, (GrowingModule, MergeGrowingModule)
        ), (
            f"to grow previous_module must be an instance of GrowingModule"
            f"or MergeGrowingModule, but got {type(previous_module)}"
        )

        self.next_module: torch.nn.Module | None = next_module
        self.previous_module: torch.nn.Module | None = previous_module

        self.__dict__["store_input"] = False
        self.__dict__["store_pre_activity"] = False
        # self.store_activity = False

        self._internal_store_input = False
        self._internal_store_pre_activity = False
        # self._internal_store_activity = False

        self._input: torch.Tensor | None = None
        self._pre_activity: torch.Tensor | None = None
        self._input_size: tuple[int, ...] | None = None

        self._tensor_s = TensorStatistic(
            tensor_s_shape,
            update_function=self.compute_s_update,
            device=self.device,
            name=f"S({self.name})",
        )
        self.tensor_m = TensorStatistic(
            tensor_m_shape,
            update_function=self.compute_m_update,
            device=self.device,
            name=f"M({self.name})",
        )

        # the optimal update used to compute v_projected
        self.optimal_delta_layer: torch.nn.Module | None = None
        # Three independent scaling tensors:
        #  * optimal_delta_scaling scales optimal_delta_layer
        #  * input_extension_scaling scales extended_input_layer (this layer)
        #  * output_extension_scaling scales extended_output_layer (this layer's
        #    output extension, consumed by the next module's extended_forward)
        self.optimal_delta_scaling: torch.Tensor = torch.zeros(1, device=self.device)
        self.optimal_delta_scaling.requires_grad = True
        self.input_extension_scaling: torch.Tensor = torch.zeros(1, device=self.device)
        self.input_extension_scaling.requires_grad = True
        self.output_extension_scaling: torch.Tensor = torch.zeros(1, device=self.device)
        self.output_extension_scaling.requires_grad = True
        # Legacy aggregated scalar; kept in sync by the scaling_factor setter so old
        # callers reading `module.scaling_factor` keep working.
        self.scaling_factor: torch.Tensor = torch.zeros(1, device=self.device)
        self.scaling_factor.requires_grad = True

        self.extended_input_layer: torch.nn.Module | None = None
        self.extended_output_layer: torch.nn.Module | None = None

        # when updating a layer with t * optimal_delta_layer having a change of activity
        # of dA we have L(A + dA) = L(A) - t * parameter_update_decrease + o(t)
        self.parameter_update_decrease: torch.Tensor | None = None

        # when increasing this layer with sqrt(t) * extended_input_layer and
        # the previous with sqrt(t) * extended_output_layer having a change of activity
        # of dA we have (with sigma the activation function in post_layer_function):
        # L(A + dA) = L(A) - t * sigma'(0) * (eigenvalues_extension ** 2).sum() + o(t)
        self.eigenvalues_extension: torch.Tensor | None = None
        self._activation_gradient_previous_module: torch.Tensor | None = None

        self.delta_raw: torch.Tensor | None = None

        # if self._allow_growing: # FIXME: should we add this condition?
        self.tensor_m_prev = TensorStatistic(
            None,
            update_function=self.compute_m_prev_update,
            device=self.device,
            name=f"M_prev({self.name})",
        )
        self.cross_covariance = TensorStatistic(
            None,
            update_function=self.compute_cross_covariance_update,
            device=self.device,
            name=f"C({self.name})",
        )
        # Empirical Fisher / gradient covariance of the pre-activity on the
        # output-channel axis: E[g g^T] for per-sample gradient vectors g,
        # equivalently E[dA^T dA] for batched dA. Shape (cp, cp).
        self.covariance_loss_gradient = TensorStatistic(
            None,
            update_function=self.compute_covariance_loss_gradient_update,
            device=self.device,
            name=f"E({self.name})",
        )

    @property
    def in_neurons(self) -> int:
        """Number of input neurons

        Returns
        -------
        int
            number of input neurons

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def in_features(self) -> int:
        """Fan-in size

        Returns
        -------
        int
            fan-in size

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def out_features(self) -> int:
        """Fan-out size

        Returns
        -------
        int
            fan-out size

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    # Parameters
    @property
    def input_volume(self) -> int:
        """Expected input volume

        Returns
        -------
        int
            input volume

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def output_volume(self) -> int:
        """Expected output volume

        Returns
        -------
        int
            output volume

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    # Information functions
    @property
    def weight(self) -> torch.Tensor:
        """Get the weight of the layer

        Returns
        -------
        torch.Tensor
            weight tensor
        """
        return self.layer.weight

    @property
    def bias(self) -> torch.Tensor:
        """Get the bias of the layer

        Returns
        -------
        torch.Tensor
            bias tensor
        """
        return self.layer.bias

    @property
    def activation_gradient(self) -> torch.Tensor:
        """
        Return the derivative of the activation function before this layer at 0+.

        /!/ A caching mechanism is used to avoid recomputing the value multiple times.
        Therefore, if the previous module changes its post layer function,
        the cache must be cleared manually by setting
        _activation_gradient_previous_module to None.

        Returns
        -------
        torch.Tensor
            derivative of the activation function before this layer at 0+

        Raises
        ------
        NotImplementedError
            abstract method
        """
        if self._activation_gradient_previous_module is None:
            if isinstance(self.previous_module, GrowingModule):
                inspected_function = self.previous_module.post_layer_function
            elif isinstance(self.previous_module, MergeGrowingModule):
                inspected_function = self.previous_module.post_merge_function
            else:
                raise NotImplementedError(
                    f"The computation of the activation gradient is not implemented yet "
                    f"for {type(self.previous_module)} as previous module."
                )

            if type(inspected_function) in known_activations_zero_plus_gradient:
                self._activation_gradient_previous_module = torch.tensor(
                    known_activations_zero_plus_gradient[type(inspected_function)],
                    device=self.device,
                )
            elif isinstance(inspected_function, torch.nn.Sequential):
                value = torch.tensor(1.0, device=self.device)
                for module in inspected_function:
                    if type(module) in known_activations_zero_plus_gradient:
                        value *= known_activations_zero_plus_gradient[type(module)]
                    elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        pass
                    else:
                        warnings.warn(
                            f"The computation of the activation gradient does not work "
                            f"necessarily with {type(module)} in a Sequential. "
                            f"We will try to compute it numerically.",
                            UserWarning,
                        )

                        value *= torch.func.grad(  # pyright: ignore[reportPrivateImportUsage]
                            module
                        )(torch.tensor(GRADIENT_COMPUTATION_EPSILON, device=self.device))
                self._activation_gradient_previous_module = value

            else:
                warnings.warn(
                    f"The computation of the activation gradient does not work "
                    f"necessarily with {type(inspected_function)}. "
                    f"We will try to compute it numerically.",
                    UserWarning,
                )
                self._activation_gradient_previous_module = torch.func.grad(  # pyright: ignore[reportPrivateImportUsage]
                    inspected_function
                )(torch.tensor(GRADIENT_COMPUTATION_EPSILON, device=self.device))
        assert self._activation_gradient_previous_module is not None
        return self._activation_gradient_previous_module

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """
        Return the parameters of the layer.

        Parameters
        ----------
        recurse: bool
            if True, return the parameters of the submodules

        Returns
        -------
        Iterator[torch.nn.Parameter]
            iterator over the parameters of the layer
        """
        return self.layer.parameters(recurse=recurse)

    def number_of_parameters(self) -> int:
        """
        Return the number of parameters of the layer.

        Returns
        -------
        int
            number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def set_scaling_factor(self, factor: float) -> None:
        """Assign scaling factor to all growing layers

        Parameters
        ----------
        factor: float
            scaling factor
        """
        self.scaling_factor = factor  # type: ignore

    def __str__(self, verbose=0):
        if verbose == 0:
            return f"{self.name} module with {self.number_of_parameters()} parameters."
        elif verbose == 1:
            return (
                f"{self.name} module with {self.number_of_parameters()} parameters "
                f"({self._allow_growing=}, {self.store_input=}, "
                f"{self.store_pre_activity=})."
            )
        elif verbose >= 2:
            txt = [
                f"{self.name} module with {self.number_of_parameters()} parameters.",
                f"\tLayer: {self.layer}",
                f"\tPost layer function: {self.post_layer_function}",
                f"\tAllow growing: {self._allow_growing}",
                f"\tStore input: {self.store_input}",
                f"\t{self._internal_store_input=}",
                f"\tStore pre-activity: {self.store_pre_activity}",
                f"\t{self._internal_store_pre_activity=}",
                f"\tTensor S (internal): {self._tensor_s}",
                f"\tTensor S: {self.tensor_s}",
                f"\tTensor M: {self.tensor_m}",
                f"\tOptimal delta layer: {self.optimal_delta_layer}",
                f"\tExtended input layer: {self.extended_input_layer}",
                f"\tExtended output layer: {self.extended_output_layer}",
            ]
            return "\n".join(txt)
        else:
            raise ValueError(f"verbose={verbose} is not a valid value.")

    def __repr__(self, *args: Any, **kwargs: Any):
        return self.__str__(*args, **kwargs)

    def __setattr__(self, key, value):
        if key == "store_input" and value is not self.store_input:
            self.__dict__["store_input"] = value
            if isinstance(self.previous_module, MergeGrowingModule):
                # As a MergeGrowingModule may have multiple next modules we need to
                # keep track of the number of modules that require the activity to be
                # stored. Hence we store it as long as one of the module requires it.
                self.previous_module.store_activity += 1 if value else -1
            else:
                self._internal_store_input = value
        elif key == "store_pre_activity" and value is not self.store_pre_activity:
            self.__dict__["store_pre_activity"] = value
            if isinstance(self.next_module, MergeGrowingModule):
                self.next_module.store_input += 1 if value else -1
            else:
                self._internal_store_pre_activity = value
        elif key == "previous_module" or key == "next_module":
            self.__dict__[key] = value
        elif key in (
            "scaling_factor",
            "optimal_delta_scaling",
            "input_extension_scaling",
            "output_extension_scaling",
        ):
            if isinstance(value, torch.Tensor):
                assert value.shape == (1,), f"{key} must be a scalar tensor."
                torch.nn.Module.__setattr__(self, key, value)
            else:
                assert isinstance(value, (int, float)), f"{key} must be a scalar."
                if key in self.__dict__:
                    self.__dict__[key].data[0] = value
                else:
                    torch.nn.Module.__setattr__(
                        self, key, torch.tensor([float(value)], device=self.device)
                    )
            if key == "scaling_factor":
                # Legacy API: fan out to the three sub-factors with the historical
                # coupling (delta scaled by value**2, extensions by value).
                value_f = (
                    value.item() if isinstance(value, torch.Tensor) else float(value)
                )
                self.__dict__["optimal_delta_scaling"].data[0] = value_f**2
                self.__dict__["input_extension_scaling"].data[0] = value_f
                if self.previous_module is None:
                    pass
                elif isinstance(self.previous_module, GrowingModule):
                    self.previous_module.__dict__["output_extension_scaling"].data[0] = (
                        value_f
                    )
                elif isinstance(self.previous_module, MergeGrowingModule):
                    pass
                else:
                    raise TypeError(
                        f"Previous module must be a GrowingModule or "
                        f"MergeGrowingModule, got {type(self.previous_module)}"
                    )
        elif key == "weight":
            self.layer.weight = value
        elif key == "bias":
            self.layer.bias = value
        else:
            # Warning: if you use __dict__ to set an attribute instead of
            # Module.__setattr__, the attribute will not be registered as a
            # parameter of the module ie .parameters() will not return it.
            torch.nn.Module.__setattr__(self, key, value)

    # Forward and storing
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.
        If needed, store the activity and pre-activity tensors.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        self._tensor_s.updated = False
        self.tensor_m.updated = False
        self.tensor_m_prev.updated = False
        self.cross_covariance.updated = False
        self.covariance_loss_gradient.updated = False
        if isinstance(self.previous_module, GrowingModule):
            # TODO: change this condition by using self._allow_growing
            self.tensor_s_growth.updated = False

        if self._internal_store_input:
            self._input = x.detach()

        pre_activity: torch.Tensor = self.layer(x)

        if self._internal_store_pre_activity:
            self._pre_activity = pre_activity
            self._pre_activity.retain_grad()

        return self.post_layer_function(pre_activity)

    def _apply_extended_post_layer_function(
        self,
        pre_activity: torch.Tensor | None,
        supplementary_pre_activity: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply the post-layer function to both main and extension pre-activations.

        When ``extended_post_layer_function`` was provided explicitly (deprecated),
        the two functions are applied separately for backward compatibility.

        Otherwise ``post_layer_function`` is queried:

        * If it implements :class:`SupportsExtendedForward`, its
          ``extended_forward(x, x_ext)`` method is called unconditionally;
          ``None`` handling is delegated to the implementation.
        * If it is a ``nn.Sequential``, each sub-module is applied in turn,
          threading ``(x, x_ext)`` through.  Sub-modules that implement
          :class:`SupportsExtendedForward` use ``extended_forward``; the rest
          are applied independently to each non-``None`` tensor (valid for
          stateless modules such as ``nn.ReLU``).
        * Fallback: the module is applied independently to each non-``None``
          tensor.

        ``None`` inputs propagate as ``None`` outputs for the corresponding
        element of the returned tuple.

        Parameters
        ----------
        pre_activity: torch.Tensor | None
            Main pre-activation tensor (N channels / features), or ``None``
            when the main path is irrelevant (e.g. zero hidden neurons).
        supplementary_pre_activity: torch.Tensor | None
            Extension pre-activation tensor (M channels / features) produced by
            ``extended_output_layer``, or ``None`` when there is no extension.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            ``(activity, supplementary_activity)``.  Each element is ``None``
            when the corresponding input was ``None``.
        """
        fn = self.post_layer_function
        if self._has_explicit_extended_post_layer_function:
            return (
                fn(pre_activity) if pre_activity is not None else None,
                (
                    self.extended_post_layer_function(supplementary_pre_activity)
                    if supplementary_pre_activity is not None
                    else None
                ),
            )
        elif isinstance(fn, SupportsExtendedForward):
            return fn.extended_forward(pre_activity, supplementary_pre_activity)
        elif isinstance(fn, torch.nn.Sequential):
            x: torch.Tensor | None = pre_activity
            x_ext: torch.Tensor | None = supplementary_pre_activity
            for module in fn:
                if isinstance(module, SupportsExtendedForward):
                    x, x_ext = module.extended_forward(x, x_ext)
                else:
                    if x is not None:
                        x = module(x)
                    if x_ext is not None:
                        x_ext = module(x_ext)
            return x, x_ext
        else:
            return (
                fn(pre_activity) if pre_activity is not None else None,
                (
                    fn(supplementary_pre_activity)
                    if supplementary_pre_activity is not None
                    else None
                ),
            )

    def extended_forward(
        self,
        x: torch.Tensor,
        x_ext: torch.Tensor | None = None,
        use_optimal_delta: bool = True,
        use_extended_input: bool = True,
        use_extended_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the module with layer extension and layer update scaled
        according to the scaling factors:
        - `optimal_delta_layer` is scaled by `optimal_delta_scaling`
        - `extended_input_layer` is scaled by `input_extension_scaling`
        - `extended_output_layer` is scaled by `output_extension_scaling`
        WARNING: does not store the input and pre-activity tensors.
        WARNING: the scaling factor is squared for the optimal delta and
        linear for the extension. (Instead of linear for the optimal delta and
        root squared for the extension as in the theory).

        Parameters
        ----------
        x: torch.Tensor
            input tensor
        x_ext: torch.Tensor | None
            extension tensor
        use_optimal_delta: bool, optional
            if True, use the optimal delta layer, default True
        use_extended_input: bool, optional
            if True, use the extended input layer, default True
        use_extended_output: bool, optional
            if True, use the extended output layer, default True

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            output tensor and extension tensor

        Raises
        ------
        ValueError
            if the input is extended and x_ext is not provided
        """
        pre_activity = self.layer(x)

        if self.optimal_delta_layer is not None and use_optimal_delta:
            pre_activity -= self.optimal_delta_scaling * self.optimal_delta_layer(x)

        if use_extended_input:
            if self.extended_input_layer:
                if x_ext is None:
                    raise ValueError(
                        f"x_ext must be provided got None for {self.name}."
                        f"As the input is extended, an extension is needed."
                    )
                pre_activity += self.input_extension_scaling * self.extended_input_layer(
                    x_ext
                )
            else:
                if x_ext is not None:  # TODO: and is not empty
                    warnings.warn(
                        f"x_ext must be None got {x_ext} for {self.name}. As the input "
                        f"is not extended, no extension is needed.",
                        UserWarning,
                    )

        if self.extended_output_layer and use_extended_output:
            supplementary_pre_activity = (
                self.output_extension_scaling * self.extended_output_layer(x)
            )
            activity, supplementary_activity = self._apply_extended_post_layer_function(
                pre_activity, supplementary_pre_activity
            )
            assert activity is not None  # pre_activity is always a Tensor here
        else:
            activity = self.post_layer_function(pre_activity)
            supplementary_activity = None

        return activity, supplementary_activity

    def update_input_size(
        self,
        input_size: tuple[int, ...] | None = None,
        compute_from_previous: bool = False,
        force_update: bool = True,
    ) -> tuple[int, ...] | None:
        """
        Update the input size of the layer. Either according to the parameter or
        the input currently stored.

        Parameters
        ----------
        input_size: tuple[int, ...] | None
            new input size
        compute_from_previous: bool
            whether to compute the input size from the previous module
            assuming its output size won't be affected by the post-layer function
        force_update: bool
            whether to force the update even if the input size is already set
            (_input_size is not None)

        Returns
        -------
        tuple[int, ...] | None
            updated input size if it could be computed, None otherwise

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def input_size(self) -> tuple[int, ...]:
        """Get the expected shape of the input excluding batch size and channels

        Returns
        -------
        tuple[int, ...]
            input shape

        Raises
        ------
        ValueError
            if the input size is not given and cannot be calculated
        """
        if self._input_size is None:
            self.update_input_size()
            if self._input_size is None:
                raise ValueError(
                    f"The input size of the layer {self.name} is not defined."
                )
        return self._input_size

    @input_size.setter
    def input_size(self, value: tuple[int, ...] | None) -> None:
        if value is not None:
            self.update_input_size(value)
        else:
            self._input_size = None

    @property
    def input(self) -> torch.Tensor:
        """Get the input of the layer

        Returns
        -------
        torch.Tensor
            input tensor

        Raises
        ------
        ValueError
            if the input is not stored
        """
        if self.store_input:
            if self._internal_store_input:
                assert self._input is not None, (
                    "The input is not stored. Apparently it was not computed yet."
                )
                return self._input
            else:
                assert self.previous_module, (
                    "A previous module is needed to store the input."
                    "Otherwise self._internal_store_input must be set to True."
                )
                return self.previous_module.activity
        else:
            raise ValueError("The input is not stored.")

    @property
    def input_extended(self) -> torch.Tensor:
        """
        Return the input extended ones if the bias is used.

        Returns
        -------
        torch.Tensor
            input extended

        Raises
        ------
        NotImplementedError
            abstract method if bias is used
        """
        if self.use_bias:
            raise NotImplementedError
        else:
            return self.input

    @property
    def pre_activity(self) -> torch.Tensor:
        """Get the pre activity of the layer

        Returns
        -------
        torch.Tensor
            pre activity tensor

        Raises
        ------
        ValueError
            if the pre activity is not stored
        """
        if self.store_pre_activity:
            if self._internal_store_pre_activity:
                assert self._pre_activity is not None, (
                    "The pre-activity is not stored. Apparently it was not computed yet."
                )
                return self._pre_activity
            else:
                assert self.next_module, (
                    "A next module is needed to store the input."
                    "Otherwise self._internal_store_pre_activity must be set to True."
                )
                return self.next_module.input
        else:
            raise ValueError(f"The pre-activity is not stored for {self.name}.")

    # Statistics computation
    def projected_v_goal(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute the projected gradient of the goal with respect to the activity
        of the layer.

        dLoss/dA_proj := dLoss/dA - dW B[-1] where A is the pre-activation vector of the
        layer, and dW the optimal delta for the layer

        Parameters
        ----------
        input_vector: torch.Tensor
            input vector B[-1] of shape (n_samples, in_features)

        Returns
        -------
        torch.Tensor
            projected gradient of the goal with respect to the activity of the next layer
            dLoss/dA - dW B[-1]
        """
        assert self.optimal_delta_layer, (
            "The optimal delta layer is not computed."
            "Therefore the projected gradient cannot be computed."
        )
        return self.pre_activity.grad - self.optimal_delta_layer(input_vector)

    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S. Should be added to the type of layer.

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def tensor_s(self) -> TensorStatistic:
        """
        Return the tensor S of the layer.
        Either the tensor S computed locally or the tensor S of the previous merge layer.

        Returns
        -------
        TensorStatistic
            tensor S
        """
        if isinstance(self.previous_module, MergeGrowingModule):
            return self.previous_module.tensor_s
        else:
            return self._tensor_s

    @property
    def tensor_s_growth(self):
        """
        Redirect to the tensor S of the previous module.
        """
        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus S growth is not defined."
            )
        elif isinstance(self.previous_module, GrowingModule):
            return self.previous_module.tensor_s
        elif isinstance(self.previous_module, MergeGrowingModule):
            raise NotImplementedError(
                f"S growth is not implemented for module preceded by an "
                f"MergeGrowingModule. (error in {self.name})"
            )
        else:
            raise NotImplementedError(
                f"S growth is not implemented yet for {type(self.previous_module)} "
                f"as previous module."
            )

    @tensor_s_growth.setter
    def tensor_s_growth(self, value) -> None:  # noqa: ARG002
        """
        Allow to set the tensor_s_growth but has no effect.
        """
        raise AttributeError(
            f"You tried to set tensor_s_growth of a GrowingModule (name={self.name})."
            f"This is not allowed because tensor_s_growth refers to the previous module's"
            f" tensor_s, not the current module's tensor_s."
        )

    def compute_m_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M. Should be added to the type of layer.

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

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def compute_m_prev_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M_{-2} := dA B[-2]^T.

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

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def compute_cross_covariance_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor C := B[-1] B[-2]^T.

        Returns
        -------
        torch.Tensor
            update of the tensor C
        int
            number of samples used to compute the update

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def compute_n_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor N. Should be added to the type of layer.

        Returns
        -------
        torch.Tensor
            update of the tensor N
        int
            number of samples used to compute the update

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def compute_covariance_loss_gradient_update(
        self,
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the empirical Fisher / gradient covariance
        :math:`E_s = dA^T dA` summed over the batch (and over spatial positions
        for convolutional layers), i.e. the sum of per-sample outer products.
        Should be implemented by each layer type.

        Returns
        -------
        torch.Tensor
            update of the gradient covariance, shape (cp, cp)
        int
            number of samples used to compute the update

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def tensor_n(self) -> torch.Tensor:
        """
        Compute the tensor N for the layer with the current M_{-2}, C and optimal delta.

        Returns
        -------
        torch.Tensor
            N

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    # Layer addition
    def layer_of_tensor(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        force_bias: bool = True,
    ) -> torch.nn.Module:
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
        torch.nn.Module
            layer with the same characteristics

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def add_parameters(self, **kwargs: Any) -> None:
        """
        Grow the module by adding new parameters to the layer.

        Parameters
        ----------
        **kwargs: Any
            typically include the values of the new parameters to add to the layer

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def layer_in_extension(self, weight: torch.Tensor) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the input of the layer is extended but not the output.

        Parameters
        ----------
        weight: torch.Tensor
            weight of the extension

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def layer_out_extension(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the output of the layer is extended but not the input.

        Parameters
        ----------
        weight: torch.Tensor
            weight of the extension
        bias: torch.Tensor | None
            bias of the extension if needed

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def parameter_step(
        self, delta_weights: torch.Tensor, delta_biases: torch.Tensor | None = None
    ) -> None:
        """
        Update the parameters of the layer with the given deltas.

        Parameters
        ----------
        delta_weights: torch.Tensor
            delta values for the weights
        delta_biases: torch.Tensor | None
            delta values for the biases, if None, the biases are not updated
        """
        self.layer.weight.data += delta_weights
        if delta_biases is not None:
            self.layer.bias.data += delta_biases

    def _sub_select_added_output_dimension(
        self, keep_neurons: int, zeros_if_not_enough: bool = False
    ) -> None:
        """
        Select the first `keep_neurons` neurons of the optimal added output dimension.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        zeros_if_not_enough: bool
            if True, will keep the all neurons and set the non selected ones to zero
        """
        assert self.extended_output_layer is not None, (
            f"The layer {self.name} should have an extended output layer to "
            f"sub-select the output dimension."
        )
        if not zeros_if_not_enough:
            if keep_neurons == 0:
                self.extended_output_layer = None
            else:
                self.extended_output_layer = self.layer_of_tensor(
                    self.extended_output_layer.weight[:keep_neurons],
                    bias=(
                        self.extended_output_layer.bias[:keep_neurons]
                        if self.extended_output_layer.bias is not None
                        else None
                    ),
                )
        else:
            self.extended_output_layer.weight.data[keep_neurons:] = 0.0
            if self.extended_output_layer.bias is not None:
                self.extended_output_layer.bias.data[keep_neurons:] = 0.0

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

        Raises
        ------
        ValueError
            if there is no previous module
        NotImplementedError
            if the previous module is not the same class
        """
        assert self.eigenvalues_extension is not None, (
            f"The eigenvalues of the extension should be computed before "
            f"sub-selecting the optimal added parameters for {self.name}."
        )
        if keep_neurons is None:
            keep_neurons = int(torch.sum(self.eigenvalues_extension >= threshold).item())
        zeros_fan_in = zeros_fan_in and zeros_if_not_enough

        if self.extended_input_layer is not None:
            if not zeros_if_not_enough:
                if keep_neurons == 0:
                    self.extended_input_layer = None
                    self.eigenvalues_extension = None
                else:
                    self.eigenvalues_extension = self.eigenvalues_extension[:keep_neurons]
                    self.extended_input_layer = self.layer_of_tensor(
                        self.extended_input_layer.weight[:, :keep_neurons],
                        bias=self.extended_input_layer.bias,
                        force_bias=False,
                    )
            else:
                self.eigenvalues_extension[keep_neurons:] = 0.0
                assert zeros_fan_in or zeros_fan_out, (
                    "At least one of zeros_fan_in or zeros_fan_out must be True "
                    "if zeros_if_not_enough is True."
                )
                if zeros_fan_out:
                    self.extended_input_layer.weight.data[:, keep_neurons:] = 0.0

        if sub_select_previous:
            if self.previous_module is None:
                raise ValueError(
                    f"No previous module for {self.name}. "
                    "Therefore new neurons cannot be sub-selected."
                )
            elif isinstance(self.previous_module, GrowingModule):
                if isinstance(self.previous_module, self.__class__):
                    self.previous_module._sub_select_added_output_dimension(
                        keep_neurons, zeros_if_not_enough=zeros_fan_in
                    )
                else:
                    raise NotImplementedError(
                        f"The sub-selection of the optimal added parameters "
                        f"is not implemented yet for a connection from "
                        f"{type(self.previous_module)} to {type(self)}."
                    )
            elif isinstance(self.previous_module, MergeGrowingModule):
                raise NotImplementedError("TODO")
            else:
                raise NotImplementedError(
                    f"The sub-selection of the optimal added parameters "
                    f"is not implemented yet for {type(self.previous_module)} "
                    f"as previous module."
                )

    def _apply_output_changes(
        self, scaling_factor: float | torch.Tensor | None = None, extension_size: int = 0
    ) -> None:
        """
        Extend the layer output with the current layer output extension,
        with `self.output_extension_scaling` if no scaling factor is provided.

        Parameters
        ----------
        scaling_factor: float | torch.Tensor | None, optional
            override for the output extension scale; defaults to
            `self.output_extension_scaling`.
        extension_size: int, optional
            size of extension, by default 0
        """
        if scaling_factor is None:
            scaling_factor = self.output_extension_scaling
        else:
            if isinstance(scaling_factor, (int, float, np.number)):
                scaling_factor = torch.tensor(scaling_factor, device=self.device)
            if not (
                abs(scaling_factor.item() - self.output_extension_scaling.item()) < 1e-4
            ):
                warnings.warn(
                    f"Scaling factor {scaling_factor} is different from the one used"
                    f" during the extended_forward {self.output_extension_scaling}."
                )
        if extension_size > 0 or self.extended_output_layer is not None:
            assert isinstance(self.extended_output_layer, torch.nn.Module), (
                f"The layer {self.name} has no output extension but an"
                f" extension of size {extension_size} was requested."
            )
            self.layer_out_extension(
                weight=scaling_factor * self.extended_output_layer.weight,
                bias=(
                    scaling_factor * self.extended_output_layer.bias
                    if self.extended_output_layer.bias is not None
                    else None
                ),
            )

            # Grow potential BatchNorm parameters
            self._grow_post_layer_function(extension_size=extension_size)

            # Update the size of the next module
            if isinstance(self.next_module, MergeGrowingModule):
                self.next_module.update_size()
                self.next_module._grow_post_merge_function(extension_size=extension_size)

    def _grow_post_layer_function(self, extension_size: int) -> None:
        """Apply growth to sized activation functions

        Parameters
        ----------
        extension_size: int
            size of extension
        """
        if isinstance(self.post_layer_function, torch.nn.Sequential):
            for module in self.post_layer_function:
                if hasattr(module, "grow"):
                    module.grow(extension_size)  # type: ignore
        elif hasattr(self.post_layer_function, "grow"):
            self.post_layer_function.grow(extension_size)  # type: ignore

    def apply_change(
        self,
        scaling_factor: float | torch.Tensor | None = None,
        apply_previous: bool = True,
        apply_delta: bool = True,
        apply_extension: bool = True,
        extension_size: int | None = None,
        optimal_delta_scaling: float | torch.Tensor | None = None,
        input_extension_scaling: float | torch.Tensor | None = None,
        output_extension_scaling: float | torch.Tensor | None = None,
    ) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factors.
        This means that the layer input is extended with the current layer output
        extension and the previous layer output is extended with the previous layer
        output extension both scaled by the relevant extension scaling factors.
        This also means that the layer output is not extended.

        Parameters
        ----------
        scaling_factor: float | torch.Tensor | None
            legacy aggregated scaling factor; sets `optimal_delta_scaling = value**2`
            and both extension scalings to `value` (mirrors the historical coupling).
        apply_previous: bool
            if True apply the change to the previous layer, by default True
        apply_delta: bool
            if True apply the optimal delta to the layer, by default True
        apply_extension: bool
            if True apply the extension to the layer, by default True
        extension_size: int | None
            size of the extension to apply, by default None and get automatically
            determined using `self.eigenvalues_extension.shape[0]`
        optimal_delta_scaling: float | torch.Tensor | None
            override for `self.optimal_delta_scaling` for this call (and
            persisted on the module). When `None`, the current attribute is used.
        input_extension_scaling: float | torch.Tensor | None
            override for `self.input_extension_scaling`.
        output_extension_scaling: float | torch.Tensor | None
            override for `self.previous_module.output_extension_scaling`.

        Raises
        ------
        ValueError
            if the layer has no extension but an extension_size above zero was requested
        NotImplementedError
            if the previous module is not of type GrowingModule
        """
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor  # type: ignore
        if optimal_delta_scaling is not None:
            self.optimal_delta_scaling = optimal_delta_scaling  # type: ignore
        if input_extension_scaling is not None:
            self.input_extension_scaling = input_extension_scaling  # type: ignore
        if output_extension_scaling is not None and isinstance(
            self.previous_module, GrowingModule
        ):
            self.previous_module.output_extension_scaling = output_extension_scaling  # type: ignore

        delta_scale = self.optimal_delta_scaling
        in_ext_scale = self.input_extension_scaling

        if apply_delta and self.optimal_delta_layer is not None:
            self.parameter_step(
                delta_weights=-delta_scale * self.optimal_delta_layer.weight.data,
                delta_biases=(
                    -delta_scale * self.optimal_delta_layer.bias.data
                    if self.optimal_delta_layer.bias is not None
                    else None
                ),
            )
        if apply_extension:
            if self.extended_input_layer:
                assert self.extended_input_layer.bias is None or torch.allclose(
                    self.extended_input_layer.bias,
                    torch.zeros_like(self.extended_input_layer.bias),
                ), "The bias of the input extension must be null."
                if in_ext_scale.item() == 0:
                    warnings.warn(
                        "input_extension_scaling is null. "
                        "The input extension will have no effect."
                    )
                self.layer_in_extension(
                    weight=in_ext_scale * self.extended_input_layer.weight
                )

            if apply_previous and self.previous_module is not None:
                if isinstance(self.previous_module, GrowingModule):
                    if self.previous_module.extended_output_layer is not None:
                        if extension_size is None:
                            assert self.eigenvalues_extension is not None, (
                                "We need to determine the size of the extension but "
                                "it was not given as parameter nor could be automatically"
                                " determined as self.eigenvalues_extension is None"
                                f"(Error occurred in {self.name})"
                            )
                            extension_size = self.eigenvalues_extension.shape[0]
                    else:
                        if extension_size is None:
                            extension_size = 0
                        elif extension_size > 0:
                            raise ValueError(
                                f"The layer {self.name} has no input extension but an"
                                f" extension of size {extension_size} was requested."
                            )
                    self.previous_module._apply_output_changes(
                        extension_size=extension_size,
                    )
                elif isinstance(self.previous_module, MergeGrowingModule):
                    raise NotImplementedError  # TODO
                else:
                    raise NotImplementedError

            # Update the size of the previous and next modules
            if isinstance(self.previous_module, MergeGrowingModule):
                self.previous_module.update_size()
            if isinstance(self.next_module, MergeGrowingModule):
                self.next_module.update_size()

    # Optimal update computation
    def compute_optimal_delta(
        self,
        update: bool = True,
        dtype: torch.dtype = torch.float32,
        force_pseudo_inverse: bool = False,
        use_fisher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]:
        r"""
        Compute the optimal delta for the layer using current S and M tensors.

        With ``tensor_m`` shaped ``(in_features(+bias), out_features)``, the raw
        optimal update returned by ``optimal_delta`` corresponds to
        :math:`(S^-1 M)^T`, using the pseudo-inverse of ``S`` when needed.
        When ``use_fisher`` is True, the empirical Fisher / gradient covariance
        :math:`E_s = \mathbb{E}[dA dA^T]` is used as an output-feature left preconditioner, so
        the update is correspondingly preconditioned on the output side.

        Compute dW* (and dBias* if needed) and update the optimal_delta_layer attribute.
        L(A + gamma * B * dW) = L(A) - gamma * d + o(gamma)
        where d is the first order decrease and gamma the scaling factor.

        Parameters
        ----------
        update: bool
            if True update the optimal delta layer attribute and the first order decrease
        dtype: torch.dtype
            dtype for S and M during the computation
        force_pseudo_inverse: bool
            if True, use the pseudo-inverse to compute the optimal delta even if the
            matrix is invertible
        use_fisher: bool
            if True, use the empirical Fisher / gradient covariance as a left
            preconditioner. Relies on the independence hypothesis from the math
            notes (`@hyp:independence`).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]
            optimal delta for the weights, the biases if needed and
            the first order decrease
        """
        tensor_s = self.tensor_s()
        tensor_m = self.tensor_m()
        tensor_covariance_loss_gradient = (
            self.covariance_loss_gradient() if use_fisher else None
        )

        self.delta_raw, parameter_update_decrease = optimal_delta(
            tensor_s,
            tensor_m,
            dtype=dtype,
            force_pseudo_inverse=force_pseudo_inverse,
            tensor_covariance_loss_gradient=tensor_covariance_loss_gradient,
        )

        if self.use_bias:
            delta_weight = self.delta_raw[:, :-1]
            delta_bias = self.delta_raw[:, -1]
        else:
            delta_weight = self.delta_raw
            delta_bias = None

        delta_weight = delta_weight.reshape(*self.weight.shape)

        if update:
            self.optimal_delta_layer = self.layer_of_tensor(delta_weight, delta_bias)
            self.parameter_update_decrease = parameter_update_decrease
        return delta_weight, delta_bias, parameter_update_decrease

    def _auxiliary_compute_alpha_omega(
        self,
        numerical_threshold: float = 1e-6,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
        use_covariance: bool = True,
        alpha_zero: bool = False,
        omega_zero: bool = False,
        use_projection: bool = True,
        ignore_singular_values: bool = False,
        use_fisher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Auxiliary function to compute the optimal added parameters (alpha, omega, k)

        This function operates on primitive options, not method names.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of
            the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        dtype: torch.dtype
            dtype for S and N during the computation
        use_covariance: bool
            if True, use S matrix (covariance preconditioning), else use Identity
        alpha_zero: bool
            if True, set alpha (incoming weights) to zero, else compute from SVD
        omega_zero: bool
            if True, set omega (outgoing weights) to zero, else compute from SVD
        use_projection: bool
            if True, use projected gradient (tensor_n), else use raw gradient (-tensor_m_prev)
        ignore_singular_values: bool
            if True, ignore singular values and treat them as 1, only using singular
            vectors for the update direction
        use_fisher: bool
            if True, use the covariance of the loss gradient as an additional
            preconditioner when computing the neuron extension

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            optimal added weights alpha, omega and eigenvalues lambda
        """
        assert self.previous_module, (
            f"No previous module for {self.name}."
            "Therefore neuron addition is not possible."
        )

        # Determine matrix_n based on use_projection
        if use_projection:
            matrix_n = self.tensor_n
        else:
            matrix_n = -self.tensor_m_prev()

        # Determine matrix_s based on use_covariance
        matrix_s = self.tensor_s_growth() if use_covariance else None

        # Determine matrix_e based on use_fisher (current layer's gradient covariance)
        matrix_e = self.covariance_loss_gradient() if use_fisher else None

        saved_dtype = matrix_n.dtype
        if matrix_n.dtype != dtype:
            matrix_n = matrix_n.to(dtype=dtype)
        if matrix_s is not None and matrix_s.dtype != dtype:
            matrix_s = matrix_s.to(dtype=dtype)
        if matrix_e is not None and matrix_e.dtype != dtype:
            matrix_e = matrix_e.to(dtype=dtype)

        # Call tools function with primitive options
        alpha, omega, eigenvalues_extension = compute_optimal_added_parameters(
            matrix_s=matrix_s,
            matrix_n=matrix_n,
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
            alpha_zero=alpha_zero,
            omega_zero=omega_zero,
            ignore_singular_values=ignore_singular_values,
            matrix_covariance_loss_gradient=matrix_e,
        )

        alpha = alpha.to(dtype=saved_dtype)
        omega = omega.to(dtype=saved_dtype)
        eigenvalues_extension = eigenvalues_extension.to(dtype=saved_dtype)

        return alpha, omega, eigenvalues_extension

    def _compute_optimal_added_parameters(
        self,
        numerical_threshold: float = 1e-6,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        dtype: torch.dtype = torch.float32,
        use_covariance: bool = True,
        alpha_zero: bool = False,
        omega_zero: bool = False,
        use_projection: bool = True,
        ignore_singular_values: bool = False,
        use_fisher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Compute the optimal added parameters to extend the input layer.
        Update the extended_input_layer and the eigenvalues_extension.

        This is a private method that operates on primitive options, not method names.

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
        use_covariance: bool
            if True, use S matrix (covariance preconditioning), else use Identity
        alpha_zero: bool
            if True, set alpha (incoming weights) to zero, else compute from SVD
        omega_zero: bool
            if True, set omega (outgoing weights) to zero, else compute from SVD
        use_projection: bool
            if True, use projected gradient (tensor_n), else use raw gradient (-tensor_m_prev)
        ignore_singular_values: bool
            if True, ignore singular values and treat them as 1, only using singular
            vectors for the update direction
        use_fisher: bool
            if True, use the covariance of the loss gradient as an additional
            preconditioner when computing the neuron extension

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]
            optimal added weights alpha weights, alpha bias, omega and eigenvalues lambda

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        assert self.parameter_update_decrease is not None, (
            "The first order improvement is not computed. "
            "Use compute_optimal_delta before."
        )
        if self.eigenvalues_extension is not None:
            return (
                self.parameter_update_decrease
                + self.activation_gradient * (self.eigenvalues_extension**2).sum()
            )
        else:
            return self.parameter_update_decrease

    def compute_optimal_updates(
        self,
        numerical_threshold: float = 1e-6,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        dtype: torch.dtype = torch.float32,
        compute_delta: bool = True,
        use_covariance: bool = True,
        alpha_zero: bool = False,
        omega_zero: bool = False,
        use_projection: bool = True,
        ignore_singular_values: bool = False,
        use_fisher: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Compute the optimal update and additional neurons.

        This method computes optimal weight updates for growing neural networks
        by analyzing gradient statistics and covariance information.

        Hyper-parameters to reproduce papers:
        -------------------------------------

        - TINY (Growing Tiny Networks: Spotting Expressivity Bottlenecks and Fixing Them Optimally)
        compute_delta: bool = False,
        use_covariance: bool = True,
        alpha_zero: bool = False,
        omega_zero: bool = False,
        use_projection: bool = True,
        ignore_singular_values: bool = False,

        - GradMax (GradMax: Growing Neural Networks using Gradient Information)
        compute_delta: bool = False,
        use_covariance: bool = False,
        alpha_zero: bool = True,
        omega_zero: bool = False,
        use_projection: bool = False,
        ignore_singular_values: bool = True,

        Parameters
        ----------
        numerical_threshold: float
            Threshold to consider an eigenvalue as zero in the square root of
            the inverse of S (covariance matrix).
        statistical_threshold: float
            Threshold to consider an eigenvalue as zero in the SVD of S^{-1/2} N.
        maximum_added_neurons: int | None
            Maximum number of added neurons. If None, all significant neurons are kept.
        update_previous: bool
            Whether to change the previous layer's extended_output_layer.
        dtype: torch.dtype
            Data type for the computation of the optimal delta and added parameters.
        compute_delta: bool
            Whether to compute optimal delta for existing weights. When True, updates
            existing parameters before computing new neuron additions.
        use_covariance: bool
            Whether to use the S matrix (covariance preconditioning). When False,
            S is treated as the identity matrix.
        alpha_zero: bool
            Whether to set alpha (incoming weights to new neurons) to zero. When True,
            new neurons start with zero incoming weights.
        omega_zero: bool
            Whether to set omega (outgoing weights from new neurons) to zero. When True,
            new neurons start with zero outgoing weights.
        use_projection: bool
            Whether to use projected gradient (tensor_n) versus raw gradient
            (-tensor_m_prev) for computing new neuron parameters.
        ignore_singular_values: bool
            Whether to ignore singular values and treat them as 1. When True, only the
            singular vectors are used for the update direction.
        use_fisher: bool
            Whether to use the covariance of the loss gradient as an additional
            preconditioner for delta and neuron-extension computations.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Optimal extension for the previous layer (weights and biases).
            Returns (None, None) when previous_module is None.

        Raises
        ------
        NotImplementedError
            If the previous module is not of type GrowingModule.
        """
        # Keep side effects coherent across configurations:
        # - compute_delta=True: compute and store full delta update.
        # - compute_delta=False/use_projection=True: compute delta statistics only
        #   (needed for tensor_n), but do not create optimal_delta_layer.
        # - compute_delta=False/use_projection=False: no natural-gradient step,
        #   so set the corresponding first-order term to zero.
        if compute_delta:
            self.compute_optimal_delta(update=True, dtype=dtype, use_fisher=use_fisher)
        else:
            self.optimal_delta_layer = None
            self.parameter_update_decrease = torch.tensor(
                0.0,
                device=self.device,
                dtype=self.weight.dtype,
            )
            if use_projection and self.previous_module is not None:
                self.compute_optimal_delta(
                    update=False, dtype=dtype, use_fisher=use_fisher
                )
            else:
                self.delta_raw = None
                self.parameter_update_decrease = torch.tensor(
                    0.0,
                    device=self.device,
                    dtype=self.weight.dtype,
                )

        if self.previous_module is None:
            return None, None
        elif isinstance(self.previous_module, GrowingModule):
            alpha_weight, alpha_bias, _, _ = self._compute_optimal_added_parameters(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                update_previous=update_previous,
                dtype=dtype,
                use_covariance=use_covariance,
                alpha_zero=alpha_zero,
                omega_zero=omega_zero,
                use_projection=use_projection,
                ignore_singular_values=ignore_singular_values,
                use_fisher=use_fisher,
            )
            return alpha_weight, alpha_bias
        elif isinstance(self.previous_module, MergeGrowingModule):
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError

    def init_computation(self) -> None:
        """
        Initialize the computation of the optimal added parameters.
        """
        self.store_input = True
        self.store_pre_activity = True
        self.tensor_s.init()
        self.tensor_m.init()
        self.covariance_loss_gradient.init()
        if self.previous_module is None:
            return
        elif isinstance(self.previous_module, GrowingModule):
            self.previous_module.store_input = True
            self.tensor_m_prev.init()
            self.cross_covariance.init()
            self.tensor_s_growth.init()
        elif isinstance(self.previous_module, MergeGrowingModule):
            self.previous_module.init_computation()
        else:
            raise NotImplementedError

    def update_computation(self) -> None:
        """
        Update the computation of the optimal added parameters.
        """
        self.tensor_s.update()
        self.tensor_m.update()
        self.covariance_loss_gradient.update()
        if self.previous_module is None:
            return
        elif isinstance(self.previous_module, GrowingModule):
            self.tensor_m_prev.update()
            self.cross_covariance.update()
            self.tensor_s_growth.update()
        elif isinstance(self.previous_module, MergeGrowingModule):
            self.previous_module.update_computation()
        else:
            raise NotImplementedError

    def reset_computation(self) -> None:
        """
        Reset the computation of the optimal added parameters.
        """
        self.store_input = False
        self.store_pre_activity = False
        self.tensor_s.reset()
        self.tensor_m.reset()
        self.covariance_loss_gradient.reset()
        if self.previous_module is None:
            return
        elif isinstance(self.previous_module, GrowingModule):
            self.tensor_m_prev.reset()
            self.cross_covariance.reset()
            self.tensor_s_growth.reset()
        elif isinstance(self.previous_module, MergeGrowingModule):
            self.previous_module.reset_computation()
        else:
            raise NotImplementedError

    def delete_update(
        self,
        include_previous: bool = True,
        delete_delta: bool = True,
        delete_input: bool = True,
        delete_output: bool = False,
    ) -> None:
        """
        Delete the updates of the layer:
        - optimal_delta_layer
        - extended_input_layer and associated extensions

        By default, we do not delete the extended_output_layer of this layer because it
        could be required by the next layer.

        Parameters
        ----------
        include_previous: bool, optional
            delete the extended_output_layer of the previous layer, by default True
        delete_delta: bool, optional
            delete the optimal_delta_layer of the module, by default True
        delete_input: bool, optional
            delete the extended_input_layer of this module, by default True
        delete_output: bool, optional
            delete the extended_output_layer of this layer, by default False
            warning: this does not delete the extended_input_layer of the next layer

        Raises
        ------
        NotImplementedError
            if include_previous is True and the previous module is of type MergeGrowingModule
        TypeError
            if previous module is not of type GrowingModule or MergeGrowingModule
        """
        if delete_delta:
            self.optimal_delta_layer = None
        self.scaling_factor = 0.0  # type: ignore
        # this type problem is due to the use of the setter to change the scaling factor
        self.parameter_update_decrease = None
        self.eigenvalues_extension = None
        self._pre_activity = None
        self._input = None

        # delete extended_output_layer
        if delete_output:
            self.extended_output_layer = None

        # delete previous module extended_output_layer
        if self.extended_input_layer is not None and delete_input:
            # delete extended_input_layer
            self.extended_input_layer = None
            if self.previous_module is not None:
                # normal behavior
                if include_previous:
                    if isinstance(self.previous_module, GrowingModule):
                        self.previous_module.extended_output_layer = None
                    elif isinstance(self.previous_module, MergeGrowingModule):
                        raise NotImplementedError  # TODO
                        # two options for future implementation:
                        # 1. Do nothing(ie replace raise NotImplementedError by return or
                        # a warning) and let the user fully in charge of deleting the
                        # associated extensions.
                        # 2. Delete associated extension ie all previous extended output,
                        # all parallel extended input and maybe more as we could have
                        # skip connections...

                    else:
                        raise TypeError(
                            f"Unexpected type for previous_module of {self.name}"
                            f"got {type(self.previous_module)} instead of GrowingModule "
                            f"or MergeGrowingModule."
                        )
                # risky behavior
                else:  # include_previous is False
                    if isinstance(self.previous_module, GrowingModule):
                        if self.previous_module.extended_output_layer is not None:
                            warnings.warn(
                                f"The extended_input_layer of {self.name} has been"
                                f" deleted. However, the extended_output_layer associated "
                                f"stored in the previous module named "
                                f"{self.previous_module.name} has not been deleted."
                                "This may lead to errors when using extended_forward.",
                                UserWarning,
                            )
                        # otherwise it is ok as user already deleted
                        # the extended_output_layer
                    elif isinstance(self.previous_module, MergeGrowingModule):
                        return
                        # the user intentionally decided to take care of deletion of the
                        # other extensions we do not raise a warning (in contrast with the
                        # GrowingModule case) as  this is way more likely to happen
                        # with MergeGrowingModule
                    else:
                        raise TypeError(
                            f"Unexpected type for previous_module of {self.name}"
                            f"got {type(self.previous_module)} instead of GrowingModule "
                            f"or MergeGrowingModule."
                        )
            # incorrect behavior
            else:  # self.previous_module is None
                warnings.warn(
                    f"The extended_input_layer of {self.name} has been deleted."
                    "However, no previous module is associated with this layer."
                    "Therefore, no extended_output_layer has been deleted."
                    "This is not supposed to happen as to grow a layer a previous "
                    "module is needed.",
                    UserWarning,
                )

    def __del__(self) -> None:
        # Unset next module of self.previous_module
        if hasattr(self, "previous_module") and self.previous_module is not None:
            if isinstance(self.previous_module, GrowingModule):
                self.previous_module.next_module = None
            elif isinstance(self.previous_module, MergeGrowingModule):
                if self in self.previous_module.next_modules:
                    self.previous_module.next_modules.remove(self)
                    self.previous_module.update_size()
            self.previous_module = None
        # Unset previous module of self.next_module
        if hasattr(self, "next_module") and self.next_module is not None:
            if isinstance(self.next_module, GrowingModule):
                self.next_module.previous_module = None
            elif isinstance(self.next_module, MergeGrowingModule):
                if self in self.next_module.previous_modules:
                    self.next_module.previous_modules.remove(self)
                    self.next_module.update_size()
            self.next_module = None

    def weights_statistics(self) -> dict[str, dict[str, float]]:
        """
        Get the statistics of the weights in the growing layer.

        Returns
        -------
        dict[str, dict[str, float]]
            A dictionary where keys are weights names and
            values are dictionaries of weight statistics.
        """
        layer_stats = {
            "weight": compute_tensor_stats(self.layer.weight),
        }
        if self.layer.bias is not None:
            layer_stats["bias"] = compute_tensor_stats(self.layer.bias)

        return layer_stats

    def scale_parameter_update(self, scale: float) -> None:
        """
        Scale the parameter update by a given factor.
        This means scaling the optimal delta and the parameter_update_decrease.

        Parameters
        ----------
        scale: float
            The factor by which to scale the parameter update.
        """
        if self.optimal_delta_layer is not None:
            self.scale_layer(self.optimal_delta_layer, scale)
            if self.parameter_update_decrease is not None:
                self.parameter_update_decrease *= scale

    @staticmethod
    def scale_layer(layer: torch.nn.Module, scale: float) -> torch.nn.Module:
        """
        Scale the weights and biases of a given layer by a specified factor.

        Parameters
        ----------
        layer: torch.nn.Module
            The layer whose parameters are to be scaled.
        scale: float
            The factor by which to scale the layer's parameters.

        Returns
        -------
        torch.nn.Module
            The layer with scaled parameters.
        """
        if hasattr(layer, "weight") and layer.weight is not None:
            layer.weight.data *= scale
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data *= scale
        return layer

    def scale_layer_extension(
        self,
        scale: float | None,
        scale_output: float | None,
        scale_input: float | None,
    ) -> None:
        """
        Scale the layer extension by a given factor.
        This means scaling the extended_input_layer, the extended_output_layer and
        the eigenvalues_extension.
        However as the eigenvalues_extension will be squared they will be
        scaled by sqrt(scale_input * scale_output).

        Parameters
        ----------
        scale: float | None
            The factor by which to scale the layer extension.
            If not None, replace both scale_input and scale_output
            if they are not None.
        scale_output: float | None
            The factor by which to scale the layer output extension.
        scale_input: float | None
            The factor by which to scale the layer input extension.
            If not None, scale must be None.

        Raises
        ------
        ValueError
            Cannot scale layer extension if one of the extensions is None
        """
        scales: list[float | None] = [scale_output, scale_input]  # type: ignore
        for i, specific_scale in enumerate(scales):
            if specific_scale is None:
                assert scale is not None, (
                    "scale can't be None if scale_input or scale_output is None."
                )
                scales[i] = scale
        assert all(isinstance(s, float) for s in scales)
        scales: list[float]

        if (
            self.extended_input_layer is None
            or self.previous_module is None
            or self.previous_module.extended_output_layer is None
        ):
            raise ValueError(
                "Cannot scale layer extension as one of the extensions is None."
            )
        self.scale_layer(self.extended_input_layer, scales[1])
        self.scale_layer(self.previous_module.extended_output_layer, scales[0])
        if self.eigenvalues_extension is not None:
            self.eigenvalues_extension *= (scales[0] * scales[1]) ** 0.5

    def get_fan_in_from_layer(
        self, layer: torch.nn.Module | None = None, num_neurons: int | None = None
    ) -> int:
        """
        Get the fan_in (number of input features) from a given layer
        or from a given number of neurons.

        Parameters
        ----------
        layer: torch.nn.Module | None
            layer to get the fan_in from
        num_neurons: int | None
            number of neurons in the layer

        Returns
        -------
        int
            fan_in of the layer

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def normalize_optimal_updates(
        self,
        std_target: float | None = None,
        normalization_type: str = "legacy_normalization",
        gradmax_scale: float = 1.0,
    ) -> None:
        r"""
        Normalize optimal update to target standard deviation

        Normalize the optimal updates so that the standard deviation of the
        weights of the updates is equal to std_target.
        If std_target is None, we automatically determine it.
        We use the standard deviation of the weights of the layer if it has weights.
        If the layer has no weights, we aim to have a std of 1 / sqrt(in_features).

        If normalization_type is "equalize_second_layer":
        Let s be the target standard deviation then:
        - optimal_delta_layer is scaled to have a std of s (so
        by s / std(optimal_delta_layer))
        - extended_input_layer is scaled to have a std of s (so
        by s / std(extended_input_layer))
        - extended_output_layer is scaled to match the scaling of the extended_input_layer
        and the optimal_delta_layer
        (so by std(extended_input_layer) / std(optimal_delta_layer))

        If normalization_type is "equalize_extensions":
        Let s be the target standard deviation then:
        - extended_input_layer is scaled to have a std of s (so
        by s / std(extended_input_layer))
        - extended_output_layer is scaled to have a std of s (so
        by s / std(extended_output_layer))
        - optimal_delta_layer is scaled to match the scaling of the extended_input_layer
        and the extended_output_layer
        (so by s ** 2 / (std(extended_input_layer) * std(extended_output_layer)))
        Note that the goal here is to give both extensions the same scale; for the
        output extension this is not the std of the layer it extends (that layer
        belongs to ``self.previous_module``), so the result is *not* scale-matched
        to the target layer. Use "match_extending_layer" for that behaviour.

        If normalization_type is "match_extending_layer":
        Each update component is scaled to match the std of the layer it modifies:
        - extended_input_layer is scaled to std(self.layer.weight)
        - previous_module.extended_output_layer is scaled to
        std(previous_module.layer.weight)
        - optimal_delta_layer is scaled to std(self.layer.weight)
        In this mode ``std_target`` is ignored; each component uses the std of its
        target layer. A previous GrowingModule with a ``.layer`` exposing weights
        is required.

        If normalization_type is "gradmax_normalization":
        Let ``gradmax_scale`` be :math:`s` (default 1). Let :math:`c = s \cdot \text{mean}_i \|W_i\|`
        where :math:`\|W_i\|` are the L2 (or Frobenius per channel) norms of the existing
        slices of ``self.layer.weight`` along the input / fan-in axis (axis 1 for ``nn.Linear``).
        Each new column (added neuron) w of ``extended_input_layer.weight`` is rescaled as
        ``w <- w / ||w|| * c`` when ``||w|| > 0``.

        For each such column j, let r_j be the factor applied (target norm divided by
        the column norm before scaling). If ``eigenvalues_extension`` is present and
        has one entry per added neuron, it is updated as
        ``eigenvalues_extension[j] *= sqrt(r_j)``, matching the one-sided analogue of
        ``scale_layer_extension`` (which uses ``*= sqrt(scale_output * scale_input)``)
        when only the input extension is rescaled.

        Parameters
        ----------
        std_target: float | None
            target standard deviation for the weights of the updates. Ignored
            when ``normalization_type == "match_extending_layer"``.
        normalization_type: str
            type of normalization to use, one of 'equalize_second_layer',
            'equalize_extensions', 'match_extending_layer', 'weird_normalization',
            'legacy_normalization', 'gradmax_normalization'
        gradmax_scale: float
            For ``gradmax_normalization`` only: scalar :math:`s` in :math:`c = s \\cdot \\text{mean}(\\|W_i\\|)`.
            Must be positive. Default ``1.0``.

        Raises
        ------
        ValueError
            if there is no previous module or the normalization_type is invalid
        """
        existing_normalizations = [
            "equalize_second_layer",
            "equalize_extensions",
            "match_extending_layer",
            "weird_normalization",
            "legacy_normalization",
            "gradmax_normalization",
        ]

        if normalization_type == "gradmax_normalization":
            if gradmax_scale <= 0:
                raise ValueError(f"gradmax_scale must be positive, got {gradmax_scale}.")
            if (
                self.extended_input_layer is None
                or not hasattr(self.extended_input_layer, "weight")
                or not hasattr(self.layer, "weight")
            ):
                raise ValueError(
                    "gradmax_normalization requires extended_input_layer and layer "
                    "with weights."
                )

            with torch.no_grad():
                layer_weight = self.layer.weight
                extension_weight = self.extended_input_layer.weight

                # Added neurons correspond to axis 1 in extension weight.
                # Guard: if tensor rank leaves no dims to reduce, norm is ill-defined.
                reduced_dims_extension = tuple(
                    dim for dim in range(extension_weight.dim()) if dim != 1
                )
                if len(reduced_dims_extension) == 0:
                    warnings.warn(
                        "gradmax_normalization: extension weight has no reducible "
                        "dimensions, skipping.",
                        UserWarning,
                    )
                    return

                # Existing neurons also correspond to axis 1 in base layer weight.
                # Guard: same check for the base layer weight.
                reduced_dims_existing = tuple(
                    dim for dim in range(layer_weight.dim()) if dim != 1
                )
                if len(reduced_dims_existing) == 0:
                    warnings.warn(
                        "gradmax_normalization: layer weight has no reducible "
                        "dimensions, skipping.",
                        UserWarning,
                    )
                    return

                # Guard: empty axis (e.g. shape [out, 0]), mean() would return NaN.
                existing_norms = (layer_weight**2).sum(dim=reduced_dims_existing).sqrt()
                if existing_norms.numel() == 0:
                    warnings.warn(
                        "gradmax_normalization: no existing neurons to compute "
                        "reference norm, skipping.",
                        UserWarning,
                    )
                    return

                # Guard: if all existing weights are zero, normalizing would
                # collapse extension weights to zero, which is undesirable.
                target_norm = existing_norms.mean() * float(gradmax_scale)
                if target_norm <= 0:
                    warnings.warn(
                        "gradmax_normalization: target norm is zero (all existing "
                        "weights are zero), skipping.",
                        UserWarning,
                    )
                    return

                extension_norms = (
                    (extension_weight**2).sum(dim=reduced_dims_extension).sqrt()
                )
                non_zero_mask = extension_norms > 0
                if non_zero_mask.any():
                    reshape_shape = [1] * extension_weight.dim()
                    reshape_shape[1] = extension_weight.shape[1]
                    scales = torch.ones_like(extension_norms)
                    scales[non_zero_mask] = target_norm / extension_norms[non_zero_mask]
                    extension_weight.mul_(scales.view(*reshape_shape))
                    if self.eigenvalues_extension is not None:
                        self.eigenvalues_extension.mul_(
                            torch.sqrt(
                                scales.to(
                                    device=self.eigenvalues_extension.device,
                                    dtype=self.eigenvalues_extension.dtype,
                                )
                            )
                        )
            return

        # Determine target standard deviation (not used by match_extending_layer,
        # which computes per-target stds below).
        if std_target is None:
            if (
                hasattr(self.layer, "weight")
                and self.layer.weight is not None
                and self.layer.weight.numel() > 0
                and (std_target := self.layer.weight.std().item()) > 0
            ):
                std_target = std_target
            else:
                # Use 1 / sqrt(in_features) as default
                assert self.extended_input_layer is not None, (
                    "Cannot determine std_target automatically as the layer has no "
                    "weights and there is no extended_input_layer to get the "
                    "number of input features from."
                )
                std_target = 1.0 / (
                    self.get_fan_in_from_layer(self.extended_input_layer) ** 0.5
                )
        assert isinstance(std_target, float) and std_target > 0, (
            "std_target must be a positive float."
        )

        def _std_of_layer(layer: torch.nn.Module | None) -> float | None:
            """
            Return the std of ``layer.weight``, or a kaiming-like fallback.

            Returns ``None`` when ``layer`` is missing or has no weights;
            returns ``layer.weight.std().item()`` when positive; otherwise
            returns ``get_fan_in_from_layer(layer) ** -0.5``.

            Parameters
            ----------
            layer : torch.nn.Module | None
                Layer whose ``.weight`` std we want.

            Returns
            -------
            float | None
                The std, the fallback, or ``None`` when no weights exist.
            """
            if (
                layer is None
                or not hasattr(layer, "weight")
                or layer.weight is None
                or layer.weight.numel() == 0
            ):
                return None
            else:
                std = layer.weight.std().item()
                if std > 0:
                    return std
                else:
                    return self.get_fan_in_from_layer(layer) ** (-0.5)

        def _scale_to(layer: torch.nn.Module | None, target_std: float) -> float:
            """
            Scaling factor that brings ``layer``'s std to ``target_std``.

            Returns 1.0 when ``layer`` has no weights to scale.

            Parameters
            ----------
            layer : torch.nn.Module | None
                Layer being scaled.
            target_std : float
                Desired std after scaling.

            Returns
            -------
            float
                Multiplicative factor to apply to ``layer.weight``.
            """
            current_std = _std_of_layer(layer)
            return 1.0 if current_std is None else target_std / current_std

        if normalization_type == "equalize_second_layer":
            delta_scale = _scale_to(self.optimal_delta_layer, std_target)
            input_extension_scale = _scale_to(self.extended_input_layer, std_target)
            output_extension_scale = delta_scale / input_extension_scale
        elif normalization_type == "match_extending_layer":
            if not isinstance(self.previous_module, GrowingModule):
                raise ValueError(
                    "match_extending_layer requires a previous GrowingModule "
                    "with a .layer attribute."
                )
            # Target std for each component = std of the layer it modifies.
            # When the target layer has no weights (e.g. an identity/empty
            # layer), fall back to a kaiming-like std computed from the
            # extension's fan-in.
            prev_ext = self.previous_module.extended_output_layer
            self_std = _std_of_layer(self.layer)
            if self_std is None and self.extended_input_layer is not None:
                self_std = self.get_fan_in_from_layer(self.extended_input_layer) ** -0.5
            prev_std = _std_of_layer(self.previous_module.layer)
            assert isinstance(self_std, float), f"{type(self_std)} is not float"
            if prev_std is None and prev_ext is not None:
                prev_std = self.get_fan_in_from_layer(prev_ext) ** -0.5
            assert isinstance(prev_std, float), f"{type(prev_std)} is not float"
            input_extension_scale = _scale_to(self.extended_input_layer, self_std)
            output_extension_scale = _scale_to(prev_ext, prev_std)
            delta_scale = _scale_to(self.optimal_delta_layer, self_std)
        elif normalization_type == "equalize_extensions":
            if self.previous_module is None:
                raise ValueError(
                    "Cannot use equalize_extensions normalization "
                    "as there is no previous module."
                )
            assert isinstance(self.previous_module, GrowingModule)
            input_extension_scale = _scale_to(self.extended_input_layer, std_target)
            output_extension_scale = _scale_to(
                self.previous_module.extended_output_layer, std_target
            )
            delta_scale = input_extension_scale * output_extension_scale
        elif normalization_type in ("legacy_normalization", "weird_normalization"):
            delta_scale = _scale_to(self.optimal_delta_layer, std_target)
            output_extension_scale = _scale_to(self.extended_input_layer, std_target)
            input_extension_scale = 1.0
        else:
            raise ValueError(
                f"normalization_type must be one of {existing_normalizations}, "
                f"got {normalization_type} instead."
            )

        # Apply scaling using existing methods
        if self.optimal_delta_layer is not None and delta_scale != 1.0:
            self.scale_parameter_update(delta_scale)

        if (
            self.extended_input_layer is not None
            and self.previous_module is not None
            and hasattr(self.previous_module, "extended_output_layer")
            and self.previous_module.extended_output_layer is not None
        ):
            self.scale_layer_extension(
                scale=None,
                scale_output=output_extension_scale,
                scale_input=input_extension_scale,
            )

    def create_layer_in_extension(self, extension_size: int) -> None:
        """
        Create the layer input extension of given size.

        Parameters
        ----------
        extension_size: int
            size of the extension to create

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    def create_layer_out_extension(self, extension_size: int) -> None:
        """
        Create the layer output extension of given size.

        Parameters
        ----------
        extension_size: int
            size of the extension to create

        Raises
        ------
        NotImplementedError
            abstract method
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Variance-transfer rescaling and neuron pairing
    # ------------------------------------------------------------------

    _KNOWN_RESCALING_STRATEGIES_TYPE = Literal[
        "default_vt", "vt_constraint_old_shape", "vt_constraint_new_shape"
    ]
    _KNOWN_NEURON_PAIRINGS_TYPE = Literal["vv_z_negz"]

    @staticmethod
    @torch.no_grad()
    def _rescale_post_layer_function(
        post_layer_fn: torch.nn.Module,
        scale: float,
    ) -> None:
        """Rescale BatchNorm running statistics to match a weight scaling.

        When existing weights are multiplied by *scale*, the BatchNorm
        running mean must be multiplied by *scale* and the running variance
        by *scale*^2 so that the normalised output remains consistent.

        Parameters
        ----------
        post_layer_fn: torch.nn.Module
            The post-layer function (may contain BatchNorm sub-modules).
        scale: float
            Multiplicative factor that was applied to the preceding layer's
            weights.
        """
        for m in post_layer_fn.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if m.running_mean is not None:
                    m.running_mean.mul_(scale)
                if m.running_var is not None:
                    m.running_var.mul_(scale**2)

    @torch.no_grad()
    def apply_rescaling(
        self,
        rescaling: _KNOWN_RESCALING_STRATEGIES_TYPE | None = None,
        neuron_pairing: _KNOWN_NEURON_PAIRINGS_TYPE | None = None,
        extension_size: int | None = None,
    ) -> None:
        """Rescale existing weights in-place before extension concatenation.

        Implements three variance-transfer strategies from [1]_:

        * ``"default_vt"`` (Strategy A): beta = sqrt(fan_in_old / fan_in_new),
          alpha = 1 (the previous layer input is not extended).
        * ``"vt_constraint_old_shape"`` (Strategy B): alpha and beta chosen so
          that V[W] = 1 / fan_in_old after rescaling.
        * ``"vt_constraint_new_shape"`` (Strategy C): alpha and beta chosen so
          that V[W] = 1 / fan_in_new after rescaling.

        ``"none"`` is a no-op.

        The current layer (*self*) is the one whose fan_in grows (Conv2 in a
        block context).  The previous layer has its fan_out grow (Conv1).

        Parameters
        ----------
        rescaling: _KNOWN_RESCALING_STRATEGIES_TYPE | None
            Rescaling strategy.  One of ``"default_vt"``,
            ``"vt_constraint_old_shape"``, ``"vt_constraint_new_shape"``.
        neuron_pairing: _KNOWN_NEURON_PAIRINGS_TYPE | None
            Neuron-pairing strategy that will be applied *after* rescaling.
            Validated for unknown values but no longer influences the
            fan-in computation — ``extension_size`` is the final size.
            One of ``"none"``, ``"vv_z_negz"``.
        extension_size: int | None
            Final number of neurons in the extension (pairing included).
            If ``None``, the size is read from the existing
            ``extended_input_layer``.

        Raises
        ------
        ValueError
            If *rescaling* or *neuron_pairing* is not a recognised strategy.

        References
        ----------
        .. [1] Yuan et al., "Accelerated Training via Incrementally
           Growing Neural Networks using Variance Transfer and Learning Rate
           Adaptation", 2024.
        """
        if rescaling is None:
            return
        if rescaling not in get_args(GrowingModule._KNOWN_RESCALING_STRATEGIES_TYPE):
            raise ValueError(
                f"Unknown rescaling strategy '{rescaling}'. "
                f"Available: {get_args(GrowingModule._KNOWN_RESCALING_STRATEGIES_TYPE)}."
            )
        if neuron_pairing is not None and neuron_pairing not in get_args(
            GrowingModule._KNOWN_NEURON_PAIRINGS_TYPE
        ):
            raise ValueError(
                f"Unknown neuron pairing '{neuron_pairing}'. "
                f"Available: {get_args(GrowingModule._KNOWN_NEURON_PAIRINGS_TYPE)}."
            )

        # When in_neurons == 0 (e.g. hidden_channels=0), there are no
        # existing weights to rescale — nothing to do.
        if self.in_neurons == 0:
            return

        # --- Determine extension size ---
        if extension_size is not None:
            ext_size = extension_size
        elif self.extended_input_layer is not None:
            # Input extension shape: (out, ext_channels, *kernel)
            ext_size = self.extended_input_layer.weight.shape[1]
        else:
            ext_size = 0

        # --- Fan-in values ---
        fan_in_self_old = self.get_fan_in_from_layer(self.layer)

        # --- Extended Fan-in values --
        # ext_size is the final extension size, pairing included
        fan_in_extension = self.get_fan_in_from_layer(num_neurons=ext_size)

        fan_in_self_new = fan_in_self_old + fan_in_extension

        assert isinstance(self.previous_module, GrowingModule), (
            f"apply_rescaling requires a GrowingModule as previous_module, "
            f"got {type(self.previous_module)}."
        )
        fan_in_prev = self.previous_module.get_fan_in_from_layer(
            self.previous_module.layer
        )

        # --- Compute alpha (previous layer) and beta (current layer) ---
        if rescaling == "default_vt":
            alpha = 1.0
            beta = (fan_in_self_old / fan_in_self_new) ** 0.5

        elif rescaling == "vt_constraint_old_shape":
            var_w_prev = self.previous_module.weight.var().item()
            var_w_self = self.weight.var().item()
            alpha = (1.0 / (fan_in_prev * var_w_prev) if var_w_prev > 0 else 1.0) ** 0.5
            beta = (
                1.0 / (fan_in_self_old * var_w_self) if var_w_self > 0 else 1.0
            ) ** 0.5

        elif rescaling == "vt_constraint_new_shape":
            var_w_prev = self.previous_module.weight.var().item()
            var_w_self = self.weight.var().item()
            alpha = (1.0 / (fan_in_prev * var_w_prev) if var_w_prev > 0 else 1.0) ** 0.5
            beta = (
                1.0 / (fan_in_self_new * var_w_self) if var_w_self > 0 else 1.0
            ) ** 0.5
        else:
            # this should not be reached
            raise ValueError(
                f"Unknown rescaling strategy '{rescaling}'. "
                f"Available: {get_args(GrowingModule._KNOWN_RESCALING_STRATEGIES_TYPE)}."
            )

        # --- Mutate weights in-place ---
        if alpha != 1.0:
            self.previous_module.weight.data.mul_(alpha)
            if self.previous_module.bias is not None:
                self.previous_module.bias.data.mul_(alpha)
            self._rescale_post_layer_function(
                self.previous_module.post_layer_function, alpha
            )

        if beta != 1.0:
            self.weight.data.mul_(beta)
            if self.bias is not None:
                self.bias.data.mul_(beta)
            self._rescale_post_layer_function(self.post_layer_function, beta)

    @torch.no_grad()
    def apply_neuron_pairing(
        self,
        neuron_pairing: _KNOWN_NEURON_PAIRINGS_TYPE | None = None,
        noise_ratio: float = 0.001,
    ) -> None:
        """Fill the second half of the extensions according to the pairing rule.

        The extension layers are expected to already have their **final**
        (even) size.  Only the first half (``[: dh // 2]`` rows / columns)
        must be initialised; the second half is overwritten in place:

        * Output extension (previous layer): V -> (V, V).
          The first ``dh // 2`` rows are left untouched; the second
          ``dh // 2`` rows are copies of the first half.
        * Input extension (current layer): Z -> (Z, -Z).
          The first ``dh_in // 2`` columns are left untouched; the second
          ``dh_in // 2`` columns are negated copies of the first half.

        At initialisation this ensures the net contribution of new neurons is
        zero, preserving the function represented by the network.  A small
        amount of noise is then added to the full input extension weight to
        break the symmetry between paired neurons, allowing them to learn
        different features during training.

        Must be called **after** extensions are created and initialised.

        Parameters
        ----------
        neuron_pairing: _KNOWN_NEURON_PAIRINGS_TYPE | None
            Pairing strategy.  One of ``"none"``, ``"vv_z_negz"``.
        noise_ratio: float
            Fraction of the standard deviation of the input extension weights
            used as the noise level for symmetry breaking.  Set to ``0`` to
            disable noise (exact function preservation).  Default ``0.001``.

        Raises
        ------
        ValueError
            If *neuron_pairing* is not a recognised strategy, or if one of
            the extensions has an odd leading dimension.
        RuntimeError
            If the required extension layers do not exist.
        """
        if neuron_pairing is None:
            return

        if neuron_pairing not in get_args(GrowingModule._KNOWN_NEURON_PAIRINGS_TYPE):
            raise ValueError(
                f"Unknown neuron pairing '{neuron_pairing}'. "
                f"Available: {get_args(GrowingModule._KNOWN_NEURON_PAIRINGS_TYPE)}."
            )
        assert isinstance(self.previous_module, GrowingModule), (
            f"apply_neuron_pairing requires a GrowingModule as "
            f"previous_module, got {type(self.previous_module)}."
        )

        # --- Output extension: V -> (V, V) ---
        ext_out = self.previous_module.extended_output_layer
        if ext_out is None:
            raise RuntimeError(
                "Cannot apply neuron pairing: previous module has "
                "no extended_output_layer."
            )
        dh = ext_out.weight.shape[0]
        if dh % 2 != 0:
            raise ValueError(
                f"apply_neuron_pairing requires an even-sized output "
                f"extension, got dh={dh}."
            )
        half_out = dh // 2
        ext_out.weight.data[half_out:].copy_(ext_out.weight.data[:half_out])
        if ext_out.bias is not None:
            ext_out.bias.data[half_out:].copy_(ext_out.bias.data[:half_out])

        # --- Input extension: Z -> (Z, -Z) ---
        ext_in = self.extended_input_layer
        if ext_in is None:
            raise RuntimeError(
                "Cannot apply neuron pairing: current module has no extended_input_layer."
            )
        dh_in = ext_in.weight.shape[1]
        if dh_in % 2 != 0:
            raise ValueError(
                f"apply_neuron_pairing requires an even-sized input "
                f"extension, got dh_in={dh_in}."
            )
        half_in = dh_in // 2
        ext_in.weight.data[:, half_in:].copy_(-ext_in.weight.data[:, :half_in])
        # Input extension bias is always False (no bias on fan-in side)

        # --- Symmetry-breaking noise on input extension ---
        if noise_ratio > 0:
            noise_std = noise_ratio * ext_in.weight.data.std()
            ext_in.weight.data.add_(torch.randn_like(ext_in.weight.data) * noise_std)

    @torch.no_grad()
    def copy_uniform_initialization(
        self,
        tensor: torch.Tensor,
        reference_tensor: torch.Tensor | None,
        fan_in: int,
    ) -> None:
        """
        Initialize tensor with uniform law aligned on reference

        Initialize the tensor with a uniform law with bounds
        -sqrt(std(W)), sqrt(std(W))
        where std(W) is the empirical standard deviation of the reference_tensor
        if the reference_tensor has a non-zero variance.
        Otherwise, use bounds
        -sqrt(6 / fan_in), sqrt(6 / fan_in)
        where fan_in is the number of input features of the reference tensor + extension.

        Parameters
        ----------
        tensor: torch.Tensor
            tensor to initialize
        reference_tensor: torch.Tensor | None
            tensor to get the standard deviation from or None to use Kaiming init
        fan_in: int
            number of input features of the base tensor + extension
        """
        # Fallback to Kaiming uniform initialization bounds
        if (
            reference_tensor is None
            or reference_tensor.numel() < 2
            or (std_dev := reference_tensor.std().item()) == 0
        ):
            self.kaiming_initialization(tensor, reference_tensor, fan_in)
        else:
            # Initialize with uniform distribution
            bound = 3.0**0.5 * std_dev
            torch.nn.init.uniform_(tensor, -bound, bound)

    @torch.no_grad()
    def kaiming_initialization(
        self,
        tensor: torch.Tensor,
        reference_tensor: torch.Tensor | None,
        fan_in: int,
    ) -> None:
        """
        Initialize tensor with Kaiming.

        Parameters
        ----------
        tensor: torch.Tensor
            tensor to initialize
        reference_tensor: torch.Tensor | None
            Unused
        fan_in: int
            number of input features of the base tensor + extension
        """
        del reference_tensor
        bound = (2.0 * 3.0 / fan_in) ** 0.5
        torch.nn.init.uniform_(tensor, -bound, bound)

    @torch.no_grad()
    def create_layer_extensions(
        self,
        extension_size: int,
        output_extension_size: int | None = None,
        input_extension_size: int | None = None,
        output_extension_init: str = "copy_uniform",
        input_extension_init: str = "copy_uniform",
        neuron_pairing: _KNOWN_NEURON_PAIRINGS_TYPE | None = None,
        rescaling: _KNOWN_RESCALING_STRATEGIES_TYPE | None = None,
        noise_ratio: float = 0.001,
    ) -> None:
        """
        Create extension for layer input and output.

        Create the layer input and output extensions of given sizes,
        optionally rescaling existing weights and applying neuron pairing.

        Allow to have different sizes for input and output extensions,
        this is useful for example if you connect a convolutional layer
        to a linear layer.

        The execution order is:

        1. **Rescaling** — existing weights are rescaled in-place (before
           extensions are created, so that ``copy_uniform`` init reads the
           rescaled weights as reference).
        2. **Extension creation** — physical extension layers are allocated
           at their final (post-pairing) size.
        3. **Initialisation** — the first half of each extension is
           initialised when ``neuron_pairing`` is active, the full extension
           otherwise.
        4. **Neuron pairing** — the already-allocated second half of each
           extension is filled in place via (V,V)/(Z,-Z).

        Parameters
        ----------
        extension_size: int
            Size of the extension to create.
        output_extension_size: int | None
            Size of the output extension to create, if ``None`` use
            *extension_size*.
        input_extension_size: int | None
            Size of the input extension to create, if ``None`` use
            *extension_size*.
        output_extension_init: str
            Initialisation method for the output extension.  Must be one of
            the keys in ``known_inits`` (``"copy_uniform"``, ``"kaiming"``,
            ``"zeros"``), default ``"copy_uniform"``.
        input_extension_init: str
            Initialisation method for the input extension.  Must be one of
            the keys in ``known_inits`` (``"copy_uniform"``, ``"kaiming"``,
            ``"zeros"``), default ``"copy_uniform"``.
        neuron_pairing: _KNOWN_NEURON_PAIRINGS_TYPE | None
            Neuron-pairing strategy applied after initialisation.
            ``"none"`` (default) or ``"vv_z_negz"``.
            /!/ When ``neuron_pairing`` is active, ``extension_size`` (and
            ``output_extension_size`` / ``input_extension_size``) is the
            **final** size, pairing included, and must be even. A
            ``ValueError`` is raised otherwise.
        rescaling: _KNOWN_RESCALING_STRATEGIES_TYPE | None
            Variance-transfer rescaling strategy applied before extension
            creation.  ``"none"`` (default), ``"default_vt"``,
            ``"vt_constraint_old_shape"``, or ``"vt_constraint_new_shape"``.
        noise_ratio: float
            Fraction of the standard deviation of the input extension weights
            used as the noise level for symmetry breaking after neuron
            pairing.  Set to ``0`` for exact function preservation.
            Default ``0.001``.

        Notes
        -----
        Additional initialization methods can be added by registering them in
        the local ``known_inits`` dictionary of this method.  Each
        initialization callable is applied to the extension weight tensor and
        to the extension bias tensor, if the layer has a bias.

        The callable must accept the following arguments:

        tensor: torch.Tensor
            Tensor of the weight/bias extension, to initialize.
        reference_tensor: torch.Tensor | None
            Weight/bias tensor from the layer before extension.
        fan_in: int
            The fan_in of the layer, after including the extension.

        An initialization callable may also modify the existing weights/biases,
        by mutating ``reference_tensor``.

        Raises
        ------
        ValueError
            If unknown initialization method, rescaling strategy, or neuron
            pairing.
        """
        if output_extension_size is None:
            output_extension_size = extension_size
        if input_extension_size is None:
            input_extension_size = extension_size
        assert isinstance(self.previous_module, GrowingModule), (
            f"The layer {self.name} has no previous module."
            "Therefore, neuron addition is not possible."
        )

        if neuron_pairing is not None:
            for param_name, value in (
                ("extension_size", extension_size),
                ("output_extension_size", output_extension_size),
                ("input_extension_size", input_extension_size),
            ):
                if value % 2 != 0:
                    raise ValueError(
                        f"neuron_pairing={neuron_pairing!r} requires an even "
                        f"{param_name}, got {value}."
                    )

        # Step 1: Rescaling (before extensions, so copy_uniform reads
        # rescaled weights as reference)
        self.apply_rescaling(
            rescaling=rescaling,
            neuron_pairing=neuron_pairing,
            extension_size=input_extension_size,
        )

        # Step 2: Create extension layers at their final size
        self.previous_module.create_layer_out_extension(output_extension_size)
        self.create_layer_in_extension(input_extension_size)

        known_inits = {
            "copy_uniform": self.copy_uniform_initialization,
            "kaiming": self.kaiming_initialization,
            "zeros": lambda tensor, _, __: torch.nn.init.zeros_(tensor),
            # Future initializations can be added here
        }

        for init in (output_extension_init, input_extension_init):
            if init not in known_inits:
                raise ValueError(
                    f"Unknown initialization method '{init}'. "
                    f"Available methods are: {list(known_inits.keys())}."
                )

        # Step 3: Initialize extensions. When pairing is active only the
        # first half is initialised; the second half is filled by
        # apply_neuron_pairing.
        half_in = input_extension_size // 2 if neuron_pairing else None
        half_out = output_extension_size // 2 if neuron_pairing else None

        # Initialize input extension
        layer_to_init = self.extended_input_layer
        assert isinstance(layer_to_init, torch.nn.Module), (
            f"The layer {self.name} has no input extension."
            "Therefore, it can't be initialized."
        )
        init_fn = known_inits[input_extension_init]
        base_fan_in = self.get_fan_in_from_layer(layer_to_init)
        ext_fan_in = self.get_fan_in_from_layer(self.layer)

        if half_in is not None:
            weight_view = layer_to_init.weight[:, :half_in]
        else:
            weight_view = layer_to_init.weight
        init_fn(weight_view, self.weight, base_fan_in + ext_fan_in)
        if layer_to_init.bias is not None:
            bias_view = (
                layer_to_init.bias[:half_in]
                if half_in is not None
                else layer_to_init.bias
            )
            init_fn(bias_view, self.bias, base_fan_in + ext_fan_in)

        # Initialize output extension
        layer_to_init = self.previous_module.extended_output_layer
        assert isinstance(layer_to_init, torch.nn.Module), (
            f"The previous layer {self.previous_module.name} has no output extension."
            "Therefore, it can't be initialized."
        )

        init_fn = known_inits[output_extension_init]
        prev_fan_in = self.previous_module.get_fan_in_from_layer(
            self.previous_module.layer
        )

        if half_out is not None:
            weight_view = layer_to_init.weight[:half_out]
        else:
            weight_view = layer_to_init.weight
        init_fn(weight_view, self.previous_module.weight, prev_fan_in)
        if layer_to_init.bias is not None:
            bias_view = (
                layer_to_init.bias[:half_out]
                if half_out is not None
                else layer_to_init.bias
            )
            init_fn(bias_view, self.previous_module.bias, prev_fan_in)

        # Step 4: Neuron pairing — fill the second halves in place
        self.apply_neuron_pairing(
            neuron_pairing=neuron_pairing,
            noise_ratio=noise_ratio,
        )

    def missing_neurons(self) -> int:
        """
        Get the number of missing neurons to reach the target hidden features.

        Returns
        -------
        int
            number of missing neurons

        Raises
        ------
        ValueError
            if target_in_neurons are not set
        """
        if self.target_in_neurons is None:
            raise ValueError(
                "Target in neurons is not set, cannot compute missing neurons."
            )
        return self.target_in_neurons - self.in_neurons

    def number_of_neurons_to_add(
        self,
        method: str = "fixed_proportional",
        number_of_growth_steps: int = 1,
    ) -> int:
        """Get the number of neurons to add in the next growth step.

        Methods
        -------
        - fixed_proportional: add a fixed proportion of the total number of neurons
          to add at each growth step. The amount to add is computed as
          an integer division as a consequence a few neurons may remain to be added
          after all growth steps have been performed.


        Parameters
        ----------
        method: str
            Method to use for determining the number of neurons to add.
            Options are "fixed_proportional".
        number_of_growth_steps: int
            Number of growth steps planned, used only if method is "fixed_proportional".

        Returns
        -------
        int
            Number of neurons to add.

        Raises
        ------
        ValueError
            if target_in_neurons or initial_in_neurons are not set or the method is unknown
        """
        if method == "fixed_proportional":
            if self.target_in_neurons is None:
                raise ValueError(
                    "Target in neurons is not set, cannot compute neurons to add."
                )
            if self._initial_in_neurons is None:
                raise ValueError(
                    "Initial in neurons is not set, cannot compute neurons to add."
                )
            total_to_add = self.target_in_neurons - self._initial_in_neurons
            return total_to_add // number_of_growth_steps
        else:
            raise ValueError(f"Unknown method: {method}.")

    def complete_growth(self, extension_kwargs: Any) -> None:
        """
        Complete the growth to the target size.

        Parameters
        ----------
        extension_kwargs: Any
            Additional arguments for creating layer extensions.
        """
        neurons_to_add = self.missing_neurons()
        if neurons_to_add > 0:
            self.create_layer_extensions(
                extension_size=neurons_to_add,
                **extension_kwargs,
            )
            self.apply_change(extension_size=neurons_to_add, scaling_factor=1.0)
            self.delete_update(include_previous=True)


if __name__ == "__main__":
    help(GrowingModule)
