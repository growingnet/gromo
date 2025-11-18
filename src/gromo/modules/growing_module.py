import warnings
from typing import Iterator

import numpy as np
import torch

from gromo.config.loader import load_config
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.tools import compute_optimal_added_parameters, optimal_delta
from gromo.utils.utils import compute_tensor_stats, get_correct_device


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
        raise NotImplementedError

    @property
    def output_volume(self) -> int:
        raise NotImplementedError

    @property
    def number_of_successors(self):
        return len(self.next_modules)

    @property
    def number_of_predecessors(self):
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
        module
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
        module
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
        next_modules
            list of next modules
        """
        raise NotImplementedError

    def set_previous_modules(
        self, previous_modules: list["MergeGrowingModule | GrowingModule"]
    ) -> None:
        """
        Set the previous modules of the current module.

        Parameters
        ----------
        previous_modules
            list of previous modules
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
                f"\tPrevious modules : {self.previous_modules}",
                f"\tNext modules : {self.next_modules}",
                f"\tPost merge function : {self.post_merge_function}",
                f"\tAllow growing : {self._allow_growing}",
                f"\tStore input : {self.store_input}",
                f"\tStore activity : {self.store_activity}",
                f"\tTensor S : {self.tensor_s}",
                f"\tPrevious tensor S : {self.previous_tensor_s}",
                f"\tPrevious tensor M : {self.previous_tensor_m}",
            ]
            return "\n".join(txt)
        else:
            raise ValueError(f"verbose={verbose} is not a valid value.")

    def __repr__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def pre_activity(self):
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
        update: bool, default True
            if True update the optimal delta layer attribute
        return_deltas: bool, default False
            if True return the deltas
        force_pseudo_inverse: bool, default False
            if True, use the pseudo-inverse to compute the optimal delta even if the
            matrix is invertible
        dtype: torch.dtype
            dtype for S and M during the computation
        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]] | None
            optimal delta for the weights and the biases if needed
        """
        assert (
            self.previous_tensor_s is not None
        ), f"No previous tensor S for {self.name}."
        assert (
            self.previous_tensor_m is not None
        ), f"No previous tensor M for {self.name}."
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
        delta, _ = optimal_delta(
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

    def update_size(self) -> None:
        """
        Update the input and output size of the module
        """
        if len(self.previous_modules) > 0:
            new_size = self.previous_modules[0].out_features
            self.in_features = new_size
        self.total_in_features = self.sum_in_features(with_bias=True)

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
    def number_of_parameters(self):
        return 0

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return iter([])

    def sum_in_features(self, with_bias: bool = False) -> int:
        """Count total in_features of previous modules

        Returns
        -------
        int
            sum of previous in_features
        """
        if with_bias:
            return sum(
                module.in_features + module.use_bias
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
        the _next_module_scaling_factor of the previous modules.
        Does only work if previous and next modules are GrowingModule.

        Parameters
        ----------
        scaling_factor: torch.Tensor | float
            scaling factor to apply to the optimal delta
        """
        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.item()
        for module in self.previous_modules:
            if isinstance(module, GrowingModule):
                module._scaling_factor_next_module.data[0] = scaling_factor
            else:
                raise TypeError(
                    f"Previous module must be a GrowingModule, got {type(module)}"
                )
        for module in self.next_modules:
            if isinstance(module, GrowingModule):
                module.__dict__["scaling_factor"].data[0] = scaling_factor
            else:
                raise TypeError(
                    f"Next module must be a GrowingModule, got {type(module)}"
                )

    def __del__(self) -> None:
        # Delete previous GrowingModules
        for prev_module in self.previous_modules:
            if isinstance(prev_module, GrowingModule):
                prev_module.__del__()
            elif isinstance(prev_module, MergeGrowingModule):
                if self in prev_module.next_modules:
                    prev_module.next_modules.remove(self)
                    prev_module.update_size()
        self.previous_modules = []
        # Delete next GrowingModules
        for next_module in self.next_modules:
            if isinstance(next_module, GrowingModule):
                next_module.__del__()
            elif isinstance(next_module, MergeGrowingModule):
                if self in next_module.previous_modules:
                    next_module.previous_modules.remove(self)
                    next_module.update_size()
        self.next_modules = []


class GrowingModule(torch.nn.Module):
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
    ) -> None:
        """
        Initialize a GrowingModule.

        Parameters
        ----------
        layer: torch.nn.Module
            layer of the module
        tensor_s_shape: tuple[int, int] | None
            shape of the tensor S
        tensor_m_shape: tuple[int, int] | None
            shape of the tensor M
        post_layer_function: torch.nn.Module
            function to apply after the layer
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
        """
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
        self._config_data, _ = load_config()
        self.device = get_correct_device(self, device)

        self.layer: torch.nn.Module = layer.to(self.device)
        # TODO: don't allow non-linearity if prev module is merge
        self.post_layer_function: torch.nn.Module = post_layer_function.to(self.device)
        if extended_post_layer_function is None:
            self.extended_post_layer_function = self.post_layer_function
        else:
            self.extended_post_layer_function = extended_post_layer_function.to(
                self.device
            )
        if isinstance(self.extended_post_layer_function, torch.nn.Sequential):
            for module in self.extended_post_layer_function:
                if hasattr(module, "num_features"):
                    warnings.warn(
                        f"Warning in {self.name}: The extended post layer "
                        f"function may get a variable input size."
                    )
        elif hasattr(self.extended_post_layer_function, "num_features"):
            warnings.warn(
                f"Warning in {self.name}: The extended post layer "
                f"function may get a variable input size."
            )

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
        # self.tensor_n = TensorStatistic(output_shape, update_function=self.compute_n_update)

        # the optimal update used to compute v_projected
        self.optimal_delta_layer: torch.nn.Module | None = None
        self.scaling_factor: torch.Tensor = torch.zeros(1, device=self.device)
        self.scaling_factor.requires_grad = True
        # to avoid having to link to the next module we get a copy of the scaling factor
        # of the next module to use it in the extended_forward
        self._scaling_factor_next_module = torch.zeros(1, device=self.device)

        self.extended_input_layer: torch.nn.Module | None = None
        self.extended_output_layer: torch.nn.Module | None = None

        # when updating a layer with t * optimal_delta_layer having a change of activity of dA
        # we have L(A + dA) = L(A) - t * parameter_update_decrease + o(t)
        self.parameter_update_decrease: torch.Tensor | None = None

        # when increasing this layer with sqrt(t) * extended_input_layer and
        # the previous with sqrt(t) * extended_output_layer having a change of activity of dA
        # we have L(A + dA) = L(A) - t * sigma'(0) * (eigenvalues_extension ** 2).sum() + o(t)
        self.eigenvalues_extension: torch.Tensor | None = None

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

    @property
    def in_features(self) -> int:
        raise NotImplementedError

    @property
    def out_features(self) -> int:
        raise NotImplementedError

    # Parameters
    @property
    def input_volume(self) -> int:
        return self.layer.in_features

    @property
    def output_volume(self) -> int:
        return self.layer.out_features

    # Information functions
    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    @property
    def activation_gradient(self) -> torch.Tensor:
        """
        Return the derivative of the activation function before this layer at 0+.

        Returns
        -------
        torch.Tensor
            derivative of the activation function before this layer at 0+
        """
        raise NotImplementedError

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """
        Return the parameters of the layer.

        Parameters
        ----------
        recurse: bool
            if True, return the parameters of the submodules

        Returns
        -------
        Iterator[Parameter]
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
        factor : float
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
                f"\tLayer : {self.layer}",
                f"\tPost layer function : {self.post_layer_function}",
                f"\tAllow growing : {self._allow_growing}",
                f"\tStore input : {self.store_input}",
                f"\t{self._internal_store_input=}",
                f"\tStore pre-activity : {self.store_pre_activity}",
                f"\t{self._internal_store_pre_activity=}",
                f"\tTensor S (internal) : {self._tensor_s}",
                f"\tTensor S : {self.tensor_s}",
                f"\tTensor M : {self.tensor_m}",
                f"\tOptimal delta layer : {self.optimal_delta_layer}",
                f"\tExtended input layer : {self.extended_input_layer}",
                f"\tExtended output layer : {self.extended_output_layer}",
            ]
            return "\n".join(txt)
        else:
            raise ValueError(f"verbose={verbose} is not a valid value.")

    def __repr__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)

    def __setattr__(self, key, value):
        if key == "store_input" and value is not self.store_input:
            self.__dict__["store_input"] = value
            if isinstance(self.previous_module, MergeGrowingModule):
                # As a MergeGrowingModule may have multiple next modules
                # we need to keep track of the number of modules that require the activity
                # to be stored. Hence we store it as long as one of the module requires it.
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
        elif key == "scaling_factor":
            if isinstance(value, torch.Tensor):
                assert value.shape == (1,), "The scaling factor must be a scalar."
                torch.nn.Module.__setattr__(self, key, value)
            else:
                assert isinstance(
                    value, (int, float)
                ), "The scaling factor must be a scalar."
                self.__dict__[key].data[0] = value
                # FIXME: should we not recreate the tensor? (problem with the gradient)
            if self.previous_module is None:
                pass
            elif isinstance(self.previous_module, GrowingModule):
                self.previous_module._scaling_factor_next_module.data[0] = (
                    self.scaling_factor.item()
                )
            elif isinstance(self.previous_module, MergeGrowingModule):
                # self.previous_module.update_scaling_factor(self.scaling_factor)
                pass
            else:
                raise TypeError(
                    f"Previous module must be a GrowingModule or MergeGrowingModule, got {type(self.previous_module)}"
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
    def forward(self, x):
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
        according to the scaling factor.
        WARNING: does not store the input and pre-activity tensors.
        WARNING: the scaling factor is squared for the optimal delta and
        linear for the extension. (Instead of linear for the optimal delta and
        squared for the extension as in the theory).

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
        tuple[torch.Tensor, torch.Tensor]
            output tensor and extension tensor
        """
        pre_activity = self.layer(x)

        # FIXME: should the scaling factor be squared with torch.sign?
        linear_factor = self.scaling_factor**2 * torch.sign(self.scaling_factor)
        sqrt_factor = self.scaling_factor

        if self.optimal_delta_layer is not None and use_optimal_delta:
            pre_activity -= linear_factor * self.optimal_delta_layer(x)

        if use_extended_input:
            if self.extended_input_layer:
                if x_ext is None:
                    raise ValueError(
                        f"x_ext must be provided got None for {self.name}."
                        f"As the input is extended, an extension is needed."
                    )
                pre_activity += sqrt_factor * self.extended_input_layer(x_ext)
            else:
                if x_ext is not None:  # TODO: and is not empty
                    warnings.warn(
                        f"x_ext must be None got {x_ext} for {self.name}. As the input is not extended, no extension is needed.",
                        UserWarning,
                    )

        if self.extended_output_layer and use_extended_output:
            supplementary_pre_activity = (
                self._scaling_factor_next_module * self.extended_output_layer(x)
            )
            supplementary_activity = self.extended_post_layer_function(
                supplementary_pre_activity
            )
        else:
            supplementary_activity = None

        activity = self.post_layer_function(pre_activity)

        return activity, supplementary_activity

    def update_input_size(
        self,
        input_size: tuple[int, ...] | None = None,
        compute_from_previous: bool = False,
        force_update: bool = True,
    ) -> tuple[int, ...] | None:
        """
        Update the input size of the layer. Either according to the parameter or the input currently stored.

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
        """
        raise NotImplementedError

    @property
    def input_size(self) -> tuple[int, ...]:
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
        if self.store_input:
            if self._internal_store_input:
                assert (
                    self._input is not None
                ), "The input is not stored. Apparently it was not computed yet."
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
        """
        if self.use_bias:
            raise NotImplementedError
        else:
            return self.input

    @property
    def pre_activity(self) -> torch.Tensor:
        if self.store_pre_activity:
            if self._internal_store_pre_activity:
                assert (
                    self._pre_activity is not None
                ), "The pre-activity is not stored. Apparently it was not computed yet."
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
        Compute the projected gradient of the goal with respect to the activity of the layer.

        dLoss/dA_proj := dLoss/dA - dW B[-1] where A is the pre-activation vector of the
        layer, and dW the optimal delta for the layer

        Parameters
        ----------
        input_vector: torch.Tensor of shape (n_samples, in_features)
            input vector B[-1]

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
                f"S growth is not implemented for module preceded by an MergeGrowingModule."
                f" (error in {self.name})"
            )
        else:
            raise NotImplementedError(
                f"S growth is not implemented yet for {type(self.previous_module)} as previous module."
            )

    @tensor_s_growth.setter
    def tensor_s_growth(self, value) -> None:
        """
        Allow to set the tensor_s_growth but has no effect.
        """
        raise AttributeError(
            f"You tried to set tensor_s_growth of a GrowingModule (name={self.name})."
            "This is not allowed because tensor_s_growth refers to the previous module's tensor_s, not the current module's tensor_s."
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
        """
        raise NotImplementedError

    def compute_n_update(self):
        """
        Compute the update of the tensor N. Should be added to the type of layer.

        Returns
        -------
        torch.Tensor
            update of the tensor N
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
        """
        raise NotImplementedError

    def add_parameters(self, **kwargs) -> None:
        """
        Grow the module by adding new parameters to the layer.

        Parameters
        ----------
        kwargs: dict
            typically include the values of the new parameters to add to the layer
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
        with the scaling factor of the next module if no scaling factor is provided.

        Parameters
        ----------
        scaling_factor: float | torch.Tensor
            scaling factor to apply to the optimal delta
        """
        if scaling_factor is None:
            scaling_factor = self._scaling_factor_next_module
        else:
            if isinstance(scaling_factor, (int, float, np.number)):
                scaling_factor = torch.tensor(scaling_factor, device=self.device)
            if not (
                abs(scaling_factor.item() - self._scaling_factor_next_module.item())
                < 1e-4
            ):
                warnings.warn(
                    f"Scaling factor {scaling_factor} is different from the one "
                    f"used during the extended_forward {self._scaling_factor_next_module}."
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

            if isinstance(self.post_layer_function, torch.nn.Sequential):
                for module in self.post_layer_function:
                    if hasattr(module, "grow"):
                        module.grow(extension_size)
            elif hasattr(self.post_layer_function, "grow"):
                self.post_layer_function.grow(extension_size)

            # Update the size of the next module
            if isinstance(self.next_module, MergeGrowingModule):
                self.next_module.update_size()

    def apply_change(
        self,
        scaling_factor: float | torch.Tensor | None = None,
        apply_previous: bool = True,
        apply_delta: bool = True,
        apply_extension: bool = True,
        extension_size: int | None = None,
    ) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        This means that the layer input is extended with the current layer output
        extension and the previous layer output is extended with the previous layer
        output extension both scaled by the current scaling factor.
        This also means that the layer output is not extended.

        Parameters
        ----------
        scaling_factor: float | torch.Tensor | None
            scaling factor to apply to the optimal delta,
             if None use the current scaling factor
        apply_previous: bool
            if True apply the change to the previous layer, by default True
        apply_delta: bool
            if True apply the optimal delta to the layer, by default True
        apply_extension: bool
            if True apply the extension to the layer, by default True
        extension_size: int | None
            size of the extension to apply, by default None and get automatically
            determined using `self.eigenvalues_extension.shape[0]`
        """
        # print(f"==================== Applying change to {self.name} ====================")
        if scaling_factor is not None:
            self.scaling_factor = scaling_factor  # type: ignore
            # this type problem is due to the use of the setter to change the scaling factor
        linear_factor = self.scaling_factor**2 * torch.sign(self.scaling_factor)
        sqrt_factor = self.scaling_factor
        if apply_delta and self.optimal_delta_layer is not None:
            self.parameter_step(
                delta_weights=-linear_factor * self.optimal_delta_layer.weight.data,
                delta_biases=(
                    -linear_factor * self.optimal_delta_layer.bias.data
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
                if self.scaling_factor == 0:
                    warnings.warn(
                        "The scaling factor is null. The input extension will have no effect."
                    )
                self.layer_in_extension(
                    weight=sqrt_factor * self.extended_input_layer.weight
                )

            if apply_previous and self.previous_module is not None:
                if isinstance(self.previous_module, GrowingModule):
                    if self.previous_module.extended_output_layer is not None:
                        if extension_size is None:
                            assert self.eigenvalues_extension is not None, (
                                "We need to determine the size of the extension but "
                                "it was not given as parameter nor could be automatically "
                                "determined as self.eigenvalues_extension is None"
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
                        scaling_factor=self.scaling_factor,
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
            if True update the optimal delta layer attribute and the first order decrease
        dtype: torch.dtype
            dtype for S and M during the computation
        force_pseudo_inverse: bool
            if True, use the pseudo-inverse to compute the optimal delta even if the
            matrix is invertible

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | float]
            optimal delta for the weights, the biases if needed and the first order decrease
        """
        tensor_s = self.tensor_s()
        tensor_m = self.tensor_m()

        self.delta_raw, self.parameter_update_decrease = optimal_delta(
            tensor_s, tensor_m, dtype=dtype, force_pseudo_inverse=force_pseudo_inverse
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
        return delta_weight, delta_bias, self.parameter_update_decrease

    def _auxiliary_compute_alpha_omega(
        self,
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
        use_projected_gradient: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Auxiliary function to compute the optimal added parameters (alpha, omega, k)

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        dtype: torch.dtype
            dtype for S and N during the computation
        use_projected_gradient: bool
            whereas to use the projected gradient ie `tensor_n` or the raw `tensor_m`

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            optimal added weights alpha, omega and eigenvalues lambda
        """
        if use_projected_gradient:
            matrix_n = self.tensor_n
        else:
            matrix_n = -self.tensor_m_prev()
        # It seems that sometimes the tensor N is not accessible.
        # I have no idea why this occurs sometimes.

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
        alpha, omega, eigenvalues_extension = compute_optimal_added_parameters(
            matrix_s=matrix_s,
            matrix_n=matrix_n,
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
        )

        alpha = alpha.to(dtype=saved_dtype)
        omega = omega.to(dtype=saved_dtype)
        eigenvalues_extension = eigenvalues_extension.to(dtype=saved_dtype)

        return alpha, omega, eigenvalues_extension

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
        Update the extended_input_layer and the eigenvalues_extension.

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
            optimal added weights alpha weights, alpha bias, omega and eigenvalues lambda
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
        numerical_threshold: float = 1e-10,
        statistical_threshold: float = 1e-5,
        maximum_added_neurons: int | None = None,
        update_previous: bool = True,
        dtype: torch.dtype = torch.float32,
        use_projected_gradient: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute the optimal update  and additional neurons.

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
            dtype for the computation of the optimal delta and added parameters
        use_projected_gradient: bool
            whereas to use the projected gradient ie `tensor_n` or the raw `tensor_m`

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            optimal extension for the previous layer (weights and biases)
        """
        self.compute_optimal_delta(dtype=dtype)

        if self.previous_module is None:
            return  # FIXME: change the definition of the function
        elif isinstance(self.previous_module, GrowingModule):
            alpha_weight, alpha_bias, _, _ = self.compute_optimal_added_parameters(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                update_previous=update_previous,
                dtype=dtype,
                use_projected_gradient=use_projected_gradient,
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
        include_previous : bool, optional
            delete the extended_output_layer of the previous layer, by default True
        delete_delta : bool, optional
            delete the optimal_delta_layer of the module, by default True
        delete_input : bool, optional
            delete the extended_input_layer of this module, by default True
        delete_output : bool, optional
            delete the extended_output_layer of this layer, by default False
            warning: this does not delete the extended_input_layer of the next layer

        Raises
        ------
        NotImplementedError
            raised when include_previous is True and the previous module is of type MergeGrowingModule
        TypeError
            raised when the previous module is not of type GrowingModule or MergeGrowingModule
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
                        # otherwise it is ok as user already deleted the extended_output_layer
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
        scale : float
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
        layer : torch.nn.Module
            The layer whose parameters are to be scaled.
        scale : float
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
        scale : float | None
            The factor by which to scale the layer extension.
            If not None, replace both scale_input and scale_output
            if they are not None.
        scale_output : float | None
            The factor by which to scale the layer output extension.
        scale_input : float | None
            The factor by which to scale the layer input extension.
            If not None, scale must be None.
        """
        scales: list[float | None] = [scale_output, scale_input]  # type: ignore
        for i, specific_scale in enumerate(scales):
            if specific_scale is None:
                assert (
                    scale is not None
                ), "scale can't be None if scale_input or scale_output is None."
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

    @staticmethod
    def get_fan_in_from_layer(layer: torch.nn.Module) -> int:
        """
        Get the fan_in (number of input features) from a given layer.

        Parameters
        ----------
        layer: torch.nn.Module
            layer to get the fan_in from

        Returns
        -------
        int
            fan_in of the layer
        """
        raise NotImplementedError

    def normalize_optimal_updates(self, std_target: float | None = None) -> None:
        """
        Normalize optimal update to target standard deviation

        Normalize the optimal updates so that the standard deviation of the
        weights of the updates is equal to std_target.
        If std_target is None, we automatically determine it.
        We use the standard deviation of the weights of the layer if it has weights.
        If the layer has no weights, we aim to have a std of 1 / sqrt(in_features).

        Let s the target standard deviation then:
        - optimal_delta_layer is scaled to have a std of s (so
        by s / std(optimal_delta_layer))
        - extended_input_layer is scaled to have a std of s (so
        by s / std(extended_input_layer))
        - extended_output_layer is scaled to match the scaling of the extended_input_layer
        and the optimal_delta_layer
        (so by std(extended_input_layer) / std(optimal_delta_layer))

        Parameters
        ----------
        std_target : float | None
            target standard deviation for the weights of the updates
        """
        # Determine target standard deviation
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

        delta_scale = 1.0
        # Get current standard deviations and calculate scaling factors
        if self.optimal_delta_layer is not None and hasattr(
            self.optimal_delta_layer, "weight"
        ):
            current_std = self.optimal_delta_layer.weight.std().item()
            if current_std > 0:
                delta_scale = std_target / current_std

        if self.extended_input_layer is not None and hasattr(
            self.extended_input_layer, "weight"
        ):
            current_std = self.extended_input_layer.weight.std().item()
            if current_std > 0:
                input_extension_scale = std_target / current_std
            else:
                input_extension_scale = 1.0 / (
                    self.get_fan_in_from_layer(self.extended_input_layer) ** 0.5
                )
        else:
            input_extension_scale = 1.0

        # Calculate output extension scale to maintain relationship
        output_extension_scale = input_extension_scale / delta_scale

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
        """
        raise NotImplementedError

    def create_layer_out_extension(self, extension_size: int) -> None:
        """
        Create the layer output extension of given size.

        Parameters
        ----------
        extension_size: int
            size of the extension to create
        """
        raise NotImplementedError

    @torch.no_grad()
    def copy_uniform_initialization(
        self, tensor: torch.Tensor, reference_tensor: torch.Tensor, fan_in: int
    ) -> None:
        """
        Initialize tensor with uniform law aligned on reference

        Initialize the tensor with a uniform law with bounds
        -sqrt(std(W)), sqrt(std(W))
        where std(W) is the empirical standard deviation of the reference_tensor
        if the reference_tensor has a non-zero variance.
        Otherwise, use bounds
        -1 / sqrt(fan_in), 1 / sqrt(fan_in)
        where fan_in is the number of input features of the
        extension.

        Parameters
        ----------
        tensor: torch.Tensor
            tensor to initialize
        reference_tensor: torch.Tensor
            tensor to get the standard deviation from
        fan_in: int
            number of input features of the extension
        """
        # Get the standard deviation from the reference_tensor
        if (
            reference_tensor is not None
            and (std_dev := reference_tensor.std().item()) > 0
        ):
            std_dev = std_dev
        else:
            # Fallback to Kaiming uniform initialization bounds
            std_dev = 1.0 / (fan_in**0.5)

        # Initialize with uniform distribution
        # bound = std_dev**0.5
        bound = 3.0**0.5 * std_dev
        torch.nn.init.uniform_(tensor, -bound, bound)

    @torch.no_grad()
    def create_layer_extensions(
        self,
        extension_size: int,
        output_extension_size: int | None = None,
        input_extension_size: int | None = None,
        output_extension_init: str = "copy_uniform",
        input_extension_init: str = "copy_uniform",
    ) -> None:
        """
        Create extension for layer input and output.

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
            Initialization method for the output extension. Must be one of the keys
            in `known_inits` (e.g., "copy_uniform", "zeros"). Default is "copy_uniform".
        input_extension_init: str
            Initialization method for the input extension. Must be one of the keys in
            `known_inits` (e.g., "copy_uniform", "zeros"). Default is "copy_uniform".
        """
        if output_extension_size is None:
            output_extension_size = extension_size
        if input_extension_size is None:
            input_extension_size = extension_size
        assert isinstance(self.previous_module, GrowingModule), (
            f"The layer {self.name} has no previous module."
            "Therefore, neuron addition is not possible."
        )
        self.previous_module.create_layer_out_extension(output_extension_size)
        self.create_layer_in_extension(input_extension_size)

        known_inits = {
            "copy_uniform": self.copy_uniform_initialization,
            "zeros": lambda tensor, _, __: torch.nn.init.zeros_(tensor),
            # Future initializations can be added here
        }

        for init in (output_extension_init, input_extension_init):
            if init not in known_inits:
                raise ValueError(
                    f"Unknown initialization method '{init}'. "
                    f"Available methods are: {list(known_inits.keys())}."
                )

        # Initialize input extension
        layer_to_init = self.extended_input_layer
        assert isinstance(layer_to_init, torch.nn.Module), (
            f"The layer {self.name} has no input extension."
            "Therefore, it can't be initialized."
        )
        init = input_extension_init

        known_inits[init](
            layer_to_init.weight, self.weight, self.get_fan_in_from_layer(layer_to_init)
        )
        if layer_to_init.bias is not None:
            known_inits[init](
                layer_to_init.bias, self.bias, self.get_fan_in_from_layer(layer_to_init)
            )

        # Initialize output extension
        layer_to_init = self.previous_module.extended_output_layer
        assert isinstance(layer_to_init, torch.nn.Module), (
            f"The previous layer {self.previous_module.name} has no output extension."
            "Therefore, it can't be initialized."
        )
        init = output_extension_init
        known_inits[init](
            layer_to_init.weight,
            self.previous_module.weight,
            self.previous_module.get_fan_in_from_layer(layer_to_init),
        )
        if layer_to_init.bias is not None:
            known_inits[init](
                layer_to_init.bias,
                self.previous_module.bias,
                self.previous_module.get_fan_in_from_layer(layer_to_init),
            )


if __name__ == "__main__":
    help(GrowingModule)
