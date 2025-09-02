from warnings import warn

import torch

from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.utils.tensor_statistic import TensorStatistic
from gromo.utils.utils import global_device


# Constants for gradient computation
GRADIENT_COMPUTATION_EPSILON = 1e-5  # Small perturbation for gradient computation


class LinearMergeGrowingModule(MergeGrowingModule):
    def __init__(
        self,
        post_merge_function: torch.nn.Module = torch.nn.Identity(),
        previous_modules=None,
        next_modules=None,
        allow_growing: bool = False,
        in_features: int = None,
        device: torch.device | None = None,
        name: str = None,
    ) -> None:
        device = device if device is not None else global_device()
        self.use_bias = True
        self.total_in_features: int = -1
        self.in_features = in_features
        self.out_features = in_features
        # TODO: check if we can automatically get the input shape
        super(LinearMergeGrowingModule, self).__init__(
            post_merge_function=post_merge_function,
            previous_modules=previous_modules,
            next_modules=next_modules,
            allow_growing=allow_growing,
            tensor_s_shape=(
                in_features + self.use_bias,
                in_features + self.use_bias,
            ),  # FIXME: +1 for the bias
            device=device,
            name=name,
        )

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
        if self.tensor_s is not None and self.tensor_s.samples > 0:
            warn(
                f"You are setting the next modules of {self.name} with a non-empty tensor S."
            )
        self.next_modules = next_modules if next_modules else []
        # self.use_bias = any(module.use_bias for module in self.next_modules)
        assert all(
            modules.in_features == self.out_features for modules in self.next_modules
        ), f"The output features must match the input features of the next modules."

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
        if self.previous_tensor_s is not None and self.previous_tensor_s.samples > 0:
            warn(
                f"You are setting the previous modules of {self.name} with a non-empty previous tensor S."
            )
        if self.previous_tensor_m is not None and self.previous_tensor_m.samples > 0:
            warn(
                f"You are setting the previous modules of {self.name} with a non-empty previous tensor M."
            )

        self.previous_modules = previous_modules if previous_modules else []
        self.total_in_features = 0
        for module in self.previous_modules:
            if not isinstance(module, (LinearGrowingModule, LinearMergeGrowingModule)):
                raise TypeError("The previous modules must be LinearGrowingModule.")
            if module.out_features != self.in_features:
                raise ValueError(
                    "The input features must match the output features of the previous modules."
                )
            self.total_in_features += module.in_features
            self.total_in_features += module.use_bias
        if self.total_in_features > 0:
            self.previous_tensor_s = TensorStatistic(
                (
                    self.total_in_features,
                    self.total_in_features,
                ),
                device=self.device,
                name=f"S[-1]({self.name})",
                update_function=self.compute_previous_s_update,
            )
        else:
            self.previous_tensor_s = None

        if self.total_in_features > 0:
            self.previous_tensor_m = TensorStatistic(
                (self.total_in_features, self.in_features),
                device=self.device,
                name=f"M[-1]({self.name})",
                update_function=self.compute_previous_m_update,
            )
        else:
            self.previous_tensor_m = None

    def construct_full_activity(self):
        """
        Construct the full activity tensor B from the input of all previous modules.
        B = (B_1, B_2, ..., B_k) in (n, C1 + C2 + ... + Ck) with Ck the number
        of features of the k-th module.
        With B_i = (X_i, 1) in (n, C_i' + 1) if the bias is used.

        Returns
        -------
        torch.Tensor
            full activity tensor
        """
        # TODO: optimize the construction of the full activity tensor
        # the merge module should directly store the full activity tensor
        # and not access it in the previous modules
        assert self.previous_modules, f"No previous modules for {self.name}."
        full_activity = torch.ones(
            (self.previous_modules[0].input.shape[0], self.total_in_features),
            device=self.device,
        )
        current_index = 0
        for (
            module
        ) in self.previous_modules:  # FIXME: what if a previous module is a merge
            if module.use_bias:
                full_activity[:, current_index : current_index + module.in_features] = (
                    module.input
                )
                # full_activity[:, current_index + module.in_features] = 1
                current_index += module.in_features + 1
            else:
                full_activity[:, current_index : current_index + module.in_features] = (
                    module.input
                )
                current_index += module.in_features
        return full_activity

    def compute_previous_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S for the input of all previous modules.
        B: full activity tensor
        S = B^T B

        Returns
        -------
        torch.Tensor
            update of the tensor S
        int
            number of samples used to compute the update
        """
        full_activity = self.construct_full_activity()
        return (
            torch.einsum("ij,ik->jk", full_activity, full_activity),
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
        # assert self.input.grad is not None, f"No gradient for input for {self.name}."
        # assert self.pre_activity.grad is not None, f"No gradient for pre_activity for {self.name}."
        full_activity = self.construct_full_activity()
        return (
            torch.einsum("ij,ik->jk", full_activity, self.pre_activity.grad),
            self.input.shape[0],
        )

    def compute_s_update(self):
        """
        Compute the update of the tensor S.
        With the input tensor X, the update is U^{j k} = X^{i j} X^{i k}.

        Returns
        -------
        torch.Tensor
            update of the tensor S
        """
        assert (
            self.store_activity
        ), f"The input must be stored to compute the update of S. (error in {self.name})"
        assert (
            self.activity is not None
        ), f"The input must be stored to compute the update of S. (error in {self.name})"
        if self.use_bias:
            # TODO: optimize this : either store directly the extended input or
            #  do manually the computation B^T B = (X^T X & mean(X)^T\\ mean(X) n)
            input_extended = torch.cat(
                (
                    self.activity,
                    torch.ones(self.activity.shape[0], 1, device=self.device),
                ),
                dim=1,
            )
            return (
                torch.einsum("ij,ik->jk", input_extended, input_extended),
                self.activity.shape[0],
            )
        else:
            return (
                torch.einsum("ij,ik->jk", self.activity, self.activity),
                self.activity.shape[0],
            )


class LinearGrowingModule(GrowingModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
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
        super(LinearGrowingModule, self).__init__(
            layer=torch.nn.Linear(
                in_features, out_features, bias=use_bias, device=device
            ),
            post_layer_function=post_layer_function,
            extended_post_layer_function=extended_post_layer_function,
            previous_module=previous_module,
            next_module=next_module,
            allow_growing=allow_growing,
            tensor_s_shape=(in_features + use_bias, in_features + use_bias),
            tensor_m_shape=(in_features + use_bias, out_features),
            device=device,
            name=name,
        )
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features

    # Information functions
    @property
    def activation_gradient(self) -> torch.Tensor:
        """
        Return the derivative of the activation function before this layer at 0+.

        Returns
        -------
        torch.Tensor
            derivative of the activation function before this layer at 0+
        """
        if isinstance(self.previous_module, GrowingModule):
            return torch.func.grad(self.previous_module.post_layer_function)(
                torch.tensor(GRADIENT_COMPUTATION_EPSILON, device=self.device)
            )
        elif isinstance(self.previous_module, MergeGrowingModule):
            return torch.func.grad(self.previous_module.post_merge_function)(
                torch.tensor(GRADIENT_COMPUTATION_EPSILON, device=self.device)
            )
        else:
            raise NotImplementedError(
                f"The computation of the activation gradient is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    @property
    def input_extended(self) -> torch.Tensor:
        """
        Return the input extended with a column of ones if the bias is used.

        Returns
        -------
        torch.Tensor
            input extended
        """
        if self.use_bias:
            # TODO (optimize this): we could directly store the extended input
            return torch.cat(
                (self.input, torch.ones(*self.input.shape[:-1], 1, device=self.device)),
                dim=-1,
            )
        else:
            return self.input

    def number_of_parameters(self) -> int:
        """
        Return the number of parameters of the layer.

        Returns
        -------
        int
            number of parameters
        """
        return (
            self.layer.in_features * self.layer.out_features
            + self.layer.out_features * self.use_bias
        )

    def __str__(self, verbose: int = 0) -> str:
        if verbose == 0:
            return (
                f"LinearGrowingModule({self.name if self.name else ' '})"
                f"(in_features={self.in_features}, "
                f"out_features={self.out_features}, use_bias={self.use_bias})"
            )
        else:
            return super(LinearGrowingModule, self).__str__(verbose=verbose)

    # Statistics computation
    def compute_s_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor S.
        With the input tensor B, the update is U^{j k} = B^{i j} B^{i k}.

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
        input_extended = self.input_extended
        return (
            torch.einsum(
                "ij,ik->jk",
                torch.flatten(input_extended, 0, -2),
                torch.flatten(input_extended, 0, -2),
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
        return (
            torch.einsum(
                "ij,ik->jk",
                torch.flatten(self.input_extended, 0, -2),
                torch.flatten(desired_activation, 0, -2),
            ),
            self.input.shape[0],
        )

    def compute_m_prev_update(
        self, desired_activation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor M_{-2} := B[-2]^T dA .

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
        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus M_{-2} is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            return (
                torch.einsum(
                    "ij,ik->jk",
                    torch.flatten(self.previous_module.input_extended, 0, -2),
                    torch.flatten(desired_activation, 0, -2),
                ),
                self.input.shape[0],
            )
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            if self.previous_module.number_of_successors > 1:
                warn("The previous module has multiple successors.")
            return (
                torch.einsum(
                    "ij,ik->jk",
                    self.previous_module.construct_full_activity(),
                    desired_activation,
                ),
                self.input.shape[0],
            )
        else:
            raise NotImplementedError(
                f"The computation of M_{-2} is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    def compute_cross_covariance_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor P := B[-2]^T B[-1] .

        Returns
        -------
        torch.Tensor
            update of the tensor P
        int
            number of samples used to compute the update
        """
        if self.previous_module is None:
            raise ValueError(
                f"No previous module for {self.name}. Thus P is not defined."
            )
        elif isinstance(self.previous_module, LinearGrowingModule):
            return (
                torch.einsum(
                    "ij,ik->jk",
                    torch.flatten(self.previous_module.input_extended, 0, -2),
                    torch.flatten(self.input_extended, 0, -2),
                ),
                self.input.shape[0],
            )
        elif isinstance(self.previous_module, LinearMergeGrowingModule):
            return (
                torch.einsum(
                    "ij,ik->jk",
                    self.previous_module.construct_full_activity(),
                    self.input,
                ),
                self.input.shape[0],
            )
        else:
            raise NotImplementedError(
                f"The computation of P is not implemented yet "
                f"for {type(self.previous_module)} as previous module."
            )

    def compute_n_update(self) -> tuple[torch.Tensor, int]:
        """
        Compute the update of the tensor N.
        With the input tensor X and V[+1] the projected desired update at the next layer
        (V[+1] = dL/dA[+1] - dW[+1]* B), the update is U^{j k} = X^{i j} V[+1]^{i k}.

        Returns
        -------
        torch.Tensor
            update of the tensor N
        int
            number of samples used to compute the update
        """
        if isinstance(self.next_module, LinearGrowingModule):
            return (
                torch.einsum(
                    "ij,ik->jk",
                    torch.flatten(self.input, 0, -2),
                    torch.flatten(
                        self.next_module.projected_v_goal(self.next_module.input), 0, -2
                    ),
                ),
                torch.tensor(self.input.shape[:-1]).prod().int().item(),
            )
        else:
            raise TypeError("The next module must be a LinearGrowingModule.")

    @property
    def tensor_n(self) -> torch.Tensor:
        """
        Compute the tensor N for the layer with the current M_{-2}, P and optimal delta.

        Returns
        -------
        torch.Tensor
            N
        """
        assert len(self.tensor_m_prev().shape) == 2, (
            f"The shape of M_-2 should be (out_features, in_features) but "
            f"got {self.tensor_m_prev().shape}."
        )
        assert len(self.cross_covariance().shape) == 2, (
            f"The shape of C should be (in_features, in_features) but "
            f"got {self.cross_covariance().shape}."
        )
        assert (
            self.delta_raw is not None
        ), f"The optimal delta should be computed before computing N for {self.name}."
        assert len(self.delta_raw.shape) == 2, (
            f"The shape of the optimal delta should be (out_features, in_features) but "
            f"got {self.optimal_delta().shape}."
        )
        assert self.tensor_m_prev().shape[1] == self.out_features, (
            f"The number of output features of M_-2 should be equal to the number of"
            f"output features of the layer but "
            f"got {self.tensor_m_prev().shape[1]} and {self.out_features}."
        )
        assert self.cross_covariance().shape[1] == self.in_features + self.use_bias, (
            f"The number of input features of P should be equal to the number of input "
            f"features of the layer but got {self.cross_covariance().shape[1]} "
            f"and {self.in_features + self.use_bias}."
            f"{self.name=}, {self.cross_covariance().shape=}"
        )
        return -self.tensor_m_prev() - self.cross_covariance() @ self.delta_raw.T

    # Layer edition
    def layer_of_tensor(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.nn.Linear:
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
        new_layer = torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=self.use_bias, device=self.device
        )
        new_layer.weight = torch.nn.Parameter(weight)
        if self.use_bias:
            new_layer.bias = torch.nn.Parameter(bias)
        return new_layer

    def add_parameters(
        self,
        matrix_extension: torch.Tensor | None,
        bias_extension: torch.Tensor | None,
        added_in_features: int = 0,
        added_out_features: int = 0,
    ) -> None:
        """
        Add new parameters to the layer.

        Parameters
        ----------
        matrix_extension: torch.Tensor
            extension of the weight matrix of the layer if None,
            the layer is extended with zeros
            should be of shape:
            - (out_features, added_in_features) if added_in_features > 0
            - (added_out_features, in_features) if added_out_features > 0
        bias_extension: torch.Tensor of shape (added_out_features,)
            extension of the bias vector of the layer if None,
            the layer is extended with zeros
        added_in_features: int >= 0
            number of input features added if None, the number of input
            features is not changed
        added_out_features: int >= 0
            number of output features added if None, the number of output
            features is not changed

        Raises
        ------
        AssertionError
            if we try to add input and output features at the same time
        """
        assert (added_in_features > 0) ^ (
            added_out_features > 0
        ), "cannot add input and output features at the same time"
        if added_in_features > 0:
            if matrix_extension is None:
                matrix_extension = torch.zeros(
                    self.out_features, added_in_features, device=self.device
                )
            else:
                assert matrix_extension.shape == (self.out_features, added_in_features), (
                    f"matrix_extension should have shape "
                    f"{(self.out_features, added_in_features)}, "
                    f"but got {matrix_extension.shape}"
                )
            self.layer_in_extension(
                weight=matrix_extension,
            )

        if added_out_features > 0:
            if matrix_extension is None:
                matrix_extension = torch.zeros(
                    added_out_features, self.in_features, device=self.device
                )
            else:
                assert matrix_extension.shape == (added_out_features, self.in_features), (
                    f"matrix_extension should have shape "
                    f"{(added_out_features, self.in_features)}, "
                    f"but got {matrix_extension.shape}"
                )
            if bias_extension is None:
                bias_extension = torch.zeros(added_out_features, device=self.device)
            else:
                assert bias_extension.shape == (added_out_features,), (
                    f"bias_extension should have shape {(added_out_features,)}, "
                    f"but got {bias_extension.shape}"
                )

            self.layer_out_extension(
                matrix_extension,
                bias=bias_extension,
            )

        warn(
            f"The size of {self.name} has been changed to "
            f"({self.in_features}, {self.out_features}) but it is up to the user"
            f" to change the connected layers."
        )

    def layer_in_extension(self, weight: torch.Tensor) -> None:
        """
        Extend the layer with the parameters of layer assuming
        that the input of the layer is extended but not the output.

        Parameters
        ----------
        weight: torch.Tensor (out_features, K)
            weight of the extension
        """
        assert (
            weight.shape[0] == self.out_features
        ), f"{weight.shape[0]=} should be equal to {self.out_features=}"
        self.layer = self.layer_of_tensor(
            weight=torch.cat((self.weight, weight), dim=1), bias=self.bias
        )

        self.in_features += weight.shape[1]
        self._tensor_s = TensorStatistic(
            (self.in_features + self.use_bias, self.in_features + self.use_bias),
            update_function=self.compute_s_update,
            name=self.tensor_s.name,
        )
        self.tensor_m = TensorStatistic(
            (self.in_features + self.use_bias, self.out_features),
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
            weight.shape[1] == self.in_features
        ), f"{weight.shape[1]=} should be equal to {self.in_features=}"
        assert (
            bias is None or bias.shape[0] == weight.shape[0]
        ), f"{bias.shape[0]=} should be equal to {weight.shape[0]=}"
        assert (
            not self.use_bias or bias is not None
        ), f"The bias of the extension should be provided because the layer has a bias"

        if self.use_bias:
            assert (
                bias is not None
            ), f"The bias of the extension should be provided because the layer has a bias"
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0),
                bias=torch.cat((self.layer.bias, bias), dim=0),
            )
        else:
            self.layer = self.layer_of_tensor(
                weight=torch.cat((self.weight, weight), dim=0), bias=None
            )

        self.out_features += weight.shape[0]
        self.tensor_m = TensorStatistic(
            (self.in_features + self.use_bias, self.out_features),
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
        Select the first keep_neurons neurons of the optimal added parameters
        linked to this layer.

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
            if isinstance(self.previous_module, LinearGrowingModule):
                self.previous_module._sub_select_added_output_dimension(keep_neurons)
            elif isinstance(self.previous_module, LinearMergeGrowingModule):
                raise NotImplementedError
            else:
                raise NotImplementedError(
                    f"The computation of the optimal added parameters is not implemented "
                    f"yet for {type(self.previous_module)} as previous module."
                )

    # Optimal update computation
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
            optimal added weights alpha weights, alpha bias, omega and eigenvalues lambda
        """
        alpha, omega, self.eigenvalues_extension = self._auxiliary_compute_alpha_omega(
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
            dtype=dtype,
            use_projected_gradient=use_projected_gradient,
        )
        k = self.eigenvalues_extension.shape[0]
        assert alpha.shape[0] == omega.shape[1], (
            f"alpha and omega should have the same number of added neurons."
            f"but got {alpha.shape} and {omega.shape}."
        )
        assert (
            omega.shape[0] == self.out_features
        ), f"omega should have the same number of output features as the layer."
        assert omega.shape == (self.out_features, k), (
            f"omega should have shape {(self.out_features, k)}, " f"but got {omega.shape}"
        )

        if self.previous_module.use_bias:
            alpha_weight = alpha[:, :-1]
            alpha_bias = alpha[:, -1]
        else:
            alpha_weight = alpha
            alpha_bias = None

        self.extended_input_layer = self.layer_of_tensor(
            omega,
            bias=(
                torch.zeros(self.out_features, device=self.device)
                if self.use_bias
                else None
            ),
        )

        if update_previous:
            if isinstance(self.previous_module, LinearGrowingModule):
                self.previous_module.extended_output_layer = self.layer_of_tensor(
                    alpha_weight, alpha_bias
                )
            elif isinstance(self.previous_module, LinearMergeGrowingModule):
                raise NotImplementedError
            else:
                raise NotImplementedError(
                    f"The computation of the optimal added parameters is not implemented "
                    f"yet for {type(self.previous_module)} as previous module."
                )

        return alpha_weight, alpha_bias, omega, self.eigenvalues_extension
