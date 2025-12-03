import copy
import operator
import warnings
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_dag import Expansion, GrowingDAG, InterMergeExpansion
from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    Conv2dMergeGrowingModule,
)
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import (
    line_search,
    mini_batch_gradient_descent,
)


class GrowingGraphNetwork(GrowingContainer):
    """Growing DAG Network

    Parameters
    ----------
    in_features : int
        size of input features
    out_features : int
        size of output dimension
    loss_fn : torch.nn.Module
        loss function
    use_bias : bool, optional
        automatically use bias in the layers, by default True
    use_batch_norm : bool, optional
        use batch normalization on the last layer, by default False
    neurons : int, optional
        default number of neurons to add at each step, by default 20
    device : str | None, optional
        default device, by default None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        loss_fn: torch.nn.Module,
        neurons: int = 20,
        neuron_epochs: int = 100,
        neuron_lrate: float = 1e-3,
        neuron_batch_size: int = 256,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        layer_type: str = "linear",
        name: str = "",
        input_shape: tuple[int, int] = None,
        device: str | None = None,
    ) -> None:
        super(GrowingGraphNetwork, self).__init__(
            in_features=in_features,
            out_features=out_features,
            device=device,
        )
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.layer_type = layer_type
        self._name = name
        self.input_shape = input_shape

        # Neuron addition
        self.neurons = neurons
        self.neuron_epochs = neuron_epochs
        self.neuron_lrate = neuron_lrate
        self.neuron_batch_size = neuron_batch_size

        self.global_step = 0
        self.global_epoch = 0
        self.loss_fn = loss_fn

        self.reset_network()
        self.set_growing_layers()

    def set_growing_layers(self):
        self._growing_layers.append(self.dag)

    def init_computation(self):
        self.dag.init_computation()

    def update_computation(self):
        self.dag.update_computation()

    def reset_computation(self):
        self.dag.reset_computation()

    def compute_optimal_delta(
        self,
        update: bool = True,
        return_deltas: bool = False,
        force_pseudo_inverse: bool = False,
    ):
        self.dag.compute_optimal_delta(
            update=update,
            return_deltas=return_deltas,
            force_pseudo_inverse=force_pseudo_inverse,
        )

    def delete_update(self) -> None:
        self.dag.delete_update()

    def update_size(self) -> None:
        super().update_size()
        self.dag.update_size()
        self.in_features = self.dag.nodes[self.dag.root]["size"]
        self.out_features = self.dag.nodes[self.dag.end]["size"]

    def init_empty_graph(self) -> None:
        """Create empty DAG with start and end nodes"""
        self.dag = GrowingDAG(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.neurons,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            default_layer_type=self.layer_type,
            name=self._name,
            input_shape=self.input_shape,
            device=self.device,
        )

        if (
            self.dag.root,
            self.dag.end,
        ) in self.dag.edges and self.layer_type == "linear":
            self.dag.remove_edge(self.dag.root, self.dag.end)

    def reset_network(self) -> None:
        """Reset graph to empty"""
        self.init_empty_graph()
        self.global_step = 0
        self.global_epoch = 0
        self.growth_history = {}

    def block_forward(
        self,
        layer_fn: Callable,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        bias: torch.Tensor,
        x: torch.Tensor,
        sigma: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        """
        Output of block connection with specific weights
        Calculates A = omega*sigma(alpha*x + b)

        Parameters
        ----------
        layer_fn : Callable
            functional operation either `F.linear` or `F.conv2d`
        alpha : torch.Tensor
            alpha input weights (new_neurons, in_features) or (new_channels, in_channels, *kernel_size)
        omega : torch.Tensor
            omega output weights (out_features, new_neurons) or (out_channels, new_channels, *kernel_size)
        bias : torch.Tensor
            bias of input layer (new_neurons,) or (new_channels,)
        x : torch.Tensor
            input vector (*in_features, batch_size)
        sigma : nn.Module
            activation function

        Returns
        -------
        torch.Tensor
            pre-activity of new connection block (*out_features, batch_size)
        """
        bias = bias.sum(dim=1).view(-1)
        hidden = sigma(layer_fn(x, alpha, bias=bias, **kwargs))
        out = layer_fn(hidden, omega, bias=None, **kwargs)
        return out

    def bottleneck_loss(
        self, activity: torch.Tensor, bottleneck: torch.Tensor
    ) -> torch.Tensor:
        """Loss of new weights with respect to the expressivity bottleneck

        Parameters
        ----------
        activity : torch.Tensor
            updated pre-activity of connection
        bottleneck : torch.Tensor
            expressivity bottleneck

        Returns
        -------
        torch.Tensor
            norm of loss
        """
        loss = activity - bottleneck
        return (loss**2).sum() / loss.numel()

    def bi_level_bottleneck_optimization(
        self,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        bias: torch.Tensor,
        B: torch.Tensor | str,
        sigma: nn.Module,
        bottleneck: torch.Tensor | str,
        input_keys: list[str],
        target_keys: list[str],
        linear: bool = True,
        operation_args: dict = {},
        verbose: bool = True,
    ) -> list[float]:
        """Bi-level optimization of new weights block with respect to the expressivity bottleneck
        # Calculates f = ||A - bottleneck||^2

        Parameters
        ----------
        alpha : torch.Tensor
            alpha input weights (neurons, in_features)
        omega : torch.Tensor
            omega output weights (out_features, neurons)
        bias : torch.Tensor
            bias of input layer (neurons,)
        B : torch.Tensor
            input vector (batch_size, in_features)
        sigma : nn.Module
            activation function
        bottleneck : torch.Tensor
            expressivity bottleneck on the output of the block
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        list[float]
            evolution of bottleneck loss over training of the block
        """

        def forward_fn(x):
            return self.block_forward(
                F.linear if linear else F.conv2d,
                alpha,
                omega,
                bias,
                x,
                sigma,
                **operation_args if not linear else {},
            )

        loss_history, _ = mini_batch_gradient_descent(
            model=forward_fn,
            parameters=[alpha, omega, bias],
            cost_fn=self.bottleneck_loss,
            X=B,
            Y=bottleneck,
            x_keys=input_keys,
            y_keys=target_keys,
            batch_size=self.neuron_batch_size,
            lrate=self.neuron_lrate,
            max_epochs=self.neuron_epochs,
            fast=True,
            verbose=verbose,
        )

        return loss_history

    def joint_bottleneck_optimization(
        self,
        activity: torch.Tensor,
        existing_activity: torch.Tensor,
        desired_update: torch.Tensor,
    ) -> float:
        # Joint optimization of new and existing weights with respect to the expressivity bottleneck
        # Calculates f = ||A + dW*B - dLoss/dA||^2
        # TODO
        raise NotImplementedError("Joint optimization of weights is not implemented yet!")

    def expand_node(
        self,
        expansion,
        bottlenecks: dict[str, torch.Tensor] | str,
        activities: dict[str, torch.Tensor] | str,
        verbose: bool = True,
    ) -> list:
        """Increase block dimension by expanding node with more neurons
        Increase output size of incoming layers and input size of outgoing layers
        Train new neurons to minimize the expressivity bottleneck

        Parameters
        ----------
        expansion : Expansion
            object with expansion information
        bottlenecks : dict
            dictionary with node names as keys and their calculated bottleneck tensors as values
        activities : dict
            dictionary with node names as keys and their pre-activity tensors as values
        parallel : bool, optional
            take into account parallel connections, by default True
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        list
            bottleneck loss history
        """

        node_module = self.dag.get_node_module(expansion.expanding_node)
        if isinstance(expansion, InterMergeExpansion):
            prev_node_modules = expansion.previous_nodes
            next_node_modules = expansion.next_nodes
        elif isinstance(expansion, Expansion):
            prev_node_modules = self.dag.get_node_modules(expansion.previous_nodes)
            next_node_modules = self.dag.get_node_modules(expansion.next_nodes)

        if type(bottlenecks) != type(activities):
            raise TypeError(
                f"Bottleneck and activities variables should have the same type. Got {type(bottlenecks)=} and {type(activities)=}"
            )

        bottleneck_keys, input_x_keys = [], []
        if isinstance(bottlenecks, str):
            assert isinstance(activities, str)
            bottleneck = bottlenecks
            input_x = activities
            for next_node_module in next_node_modules:
                bottleneck_keys.append(next_node_module._name)
            for prev_node_module in prev_node_modules:
                input_x_keys.append(prev_node_module._name)
        elif isinstance(bottlenecks, dict):
            assert isinstance(activities, dict)
            bottleneck, input_x = [], []
            for next_node_module in next_node_modules:
                assert next_node_module._name is not None
                bottleneck.append(bottlenecks[next_node_module._name])
            bottleneck = torch.cat(bottleneck, dim=1)  # (batch_size, total_out_features)
            for prev_node_module in prev_node_modules:
                assert prev_node_module._name is not None
                input_x.append(activities[prev_node_module._name])
            input_x = torch.cat(input_x, dim=1)  # (batch_size, total_in_features)
        else:
            raise TypeError(
                f"Inappropriate type for `bottlenecks` variable. Expected dict[str, torch.Tensor] or str. Got {type(bottleneck_keys)}"
            )

        total_in_features = sum([edge.in_features if isinstance(edge, LinearGrowingModule) else edge.in_channels for edge in expansion.in_edges])  # type: ignore
        total_out_features = sum([edge.out_features if isinstance(edge, LinearGrowingModule) else edge.out_channels for edge in expansion.out_edges])  # type: ignore
        in_edges = len(expansion.in_edges)

        # Initialize alpha and omega weights
        if isinstance(node_module, Conv2dMergeGrowingModule):
            alpha = torch.rand(
                (self.neurons, total_in_features, *node_module.kernel_size),
                device=self.device,
            )
            omega = torch.rand(
                (total_out_features, self.neurons, *node_module.kernel_size),
                device=self.device,
            )
        else:
            alpha = torch.rand((self.neurons, total_in_features), device=self.device)
            omega = torch.rand((total_out_features, self.neurons), device=self.device)
        bias = torch.rand((self.neurons, in_edges), device=self.device)
        alpha = alpha / np.sqrt(alpha.numel())
        omega = omega / np.sqrt(omega.numel())
        bias = bias / np.sqrt(bias.numel())
        alpha = alpha.detach().clone().requires_grad_()
        omega = omega.detach().clone().requires_grad_()
        bias = bias.detach().clone().requires_grad_()

        # Gradient descent on bottleneck
        # [bi-level]  loss = edge_weight - bottleneck
        # [joint opt] loss = edge_weight + possible updates - desired_update
        loss_history = self.bi_level_bottleneck_optimization(
            alpha=alpha,
            omega=omega,
            bias=bias,
            B=input_x,
            sigma=node_module.post_merge_function,
            bottleneck=bottleneck,
            input_keys=input_x_keys,
            target_keys=bottleneck_keys,
            linear=isinstance(node_module, LinearMergeGrowingModule),
            operation_args={
                "padding": "same",
            },
            verbose=verbose,
        )

        # Record layer extensions of new block
        i = 0
        alpha = alpha.view(self.neurons, -1)  # (neurons, total_in_features)
        for i_edge, prev_edge_module in enumerate(expansion.in_edges):
            # Output extension for alpha weights
            in_features = int(prev_edge_module.in_features)  # type: ignore
            prev_edge_module._scaling_factor_next_module[0] = 1  # type: ignore

            _weight = alpha[:, i : i + in_features]
            _weight = _weight.view((self.neurons, *prev_edge_module.weight.shape[1:]))
            _bias = bias[:, i_edge]

            prev_edge_module.extended_output_layer = prev_edge_module.layer_of_tensor(
                weight=_weight,
                bias=_bias,
            )  # bias is mandatory
            i += in_features
        i = 0
        omega = omega.view(-1, self.neurons)  # (total_out_features, neurons)
        for next_edge_module in expansion.out_edges:
            # Input extension for omega weights
            if isinstance(next_edge_module, LinearGrowingModule):
                out_features = int(next_edge_module.out_features)  # type: ignore
                next_edge_module.extended_input_layer = nn.Linear(
                    self.neurons, out_features, bias=False
                )
            elif isinstance(next_edge_module, Conv2dGrowingModule):
                out_features = int(
                    next_edge_module.out_channels
                    * next_edge_module.kernel_size[0]
                    * next_edge_module.kernel_size[1]
                )
                next_edge_module.extended_input_layer = nn.Conv2d(
                    in_channels=self.neurons,
                    out_channels=next_edge_module.out_channels,
                    bias=False,
                    kernel_size=next_edge_module.layer.kernel_size,
                    stride=next_edge_module.layer.stride,
                    padding=next_edge_module.layer.padding,
                    dilation=next_edge_module.layer.dilation,
                )
            next_edge_module.scaling_factor = 1  # type: ignore

            _weight = omega[i : i + out_features, :]
            _weight = _weight.view(
                (
                    next_edge_module.weight.shape[0],
                    self.neurons,
                    *next_edge_module.weight.shape[2:],
                )
            )
            # next_edge_module.extended_input_layer = next_edge_module.layer_of_tensor(
            #     weight=_weight,
            # ) # throws error because of bias
            next_edge_module.extended_input_layer.weight = nn.Parameter(
                _weight,
            )
            i += out_features

        return loss_history

    def update_edge_weights(
        self,
        expansion: Expansion,
        bottlenecks: dict[str, torch.Tensor] | str,
        activities: dict[str, torch.Tensor] | str,
        verbose: bool = True,
    ) -> list:
        """Update weights of a single layer edge
        Train layer to minimize the expressivity bottleneck

        Parameters
        ----------
        expansion : Expansion
            object with expansion information
        bottlenecks : dict
            dictionary with node names as keys and their calculated bottleneck tensors as values
        activities : dict
            dictionary with node names as keys and their pre-activity tensors as values
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        list
            bottleneck loss history
        """

        new_edge_module = self.dag.get_edge_module(
            expansion.previous_node, expansion.next_node
        )
        prev_node_module = self.dag.get_node_module(expansion.previous_node)
        next_node_module = self.dag.get_node_module(expansion.next_node)
        assert prev_node_module._name is not None
        assert next_node_module._name is not None

        if type(bottlenecks) != type(activities):
            raise TypeError(
                f"Bottleneck and activities variables should have the same type. Got {type(bottlenecks)=} and {type(activities)=}"
            )

        if isinstance(bottlenecks, str):
            assert isinstance(activities, str)
            bottleneck_keys = [next_node_module._name]
            activity_keys = [prev_node_module._name]
            bottleneck = bottlenecks
            activity = activities
        elif isinstance(bottlenecks, dict):
            assert isinstance(activities, dict)
            bottleneck = bottlenecks[next_node_module._name]
            activity = activities[prev_node_module._name]
            bottleneck_keys, activity_keys = [], []
        else:
            raise TypeError(
                f"Inappropriate type for `bottlenecks` variable. Expected dict[str, torch.Tensor] or str. Got {type(bottleneck_keys)}"
            )

        linear = isinstance(new_edge_module, LinearGrowingModule)

        if linear:
            weight = torch.rand(
                (new_edge_module.out_features, new_edge_module.in_features),
                device=self.device,
            )
            bias = torch.rand((new_edge_module.out_features), device=self.device)
        else:
            weight = torch.rand(
                (
                    new_edge_module.out_channels,
                    new_edge_module.in_channels,
                    *new_edge_module.kernel_size,
                ),
                device=self.device,
            )
            bias = torch.rand((new_edge_module.out_channels), device=self.device)
        weight = weight / np.sqrt(weight.numel())
        bias = bias / np.sqrt(bias.numel())
        weight = weight.detach().clone().requires_grad_()
        bias = bias.detach().clone().requires_grad_()

        if linear:
            forward_fn = lambda activity: F.linear(activity, weight, bias)
        else:
            forward_fn = lambda activity: F.conv2d(activity, weight, bias, padding="same")

        loss_history, _ = mini_batch_gradient_descent(
            model=forward_fn,
            parameters=[weight, bias],
            cost_fn=self.bottleneck_loss,
            X=activity,
            Y=bottleneck,
            x_keys=activity_keys,
            y_keys=bottleneck_keys,
            batch_size=self.neuron_batch_size,
            lrate=self.neuron_lrate,
            max_epochs=self.neuron_epochs,
            fast=True,
            verbose=verbose,
        )

        # Record layer extensions
        new_edge_module.optimal_delta_layer = new_edge_module.layer_of_tensor(
            weight, bias
        )

        return loss_history

    def find_amplitude_factor(
        self,
        dataloader: DataLoader,
        mask: dict = {},
    ) -> float:
        """Find amplitude factor with line search

        Parameters
        ----------
        dataloader : DataLoader
            dataloader with input features and target
        mask : dict, optional
            extension mask for specific nodes and edges, by default {}
            example: mask["edges"] for edges and mask["nodes"] for nodes

        Returns
        -------
        float
            amplitude factor that minimizes overall loss
        """

        def simulate_loss(factor):
            self.set_scaling_factor(factor)

            loss = []
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.extended_forward(x, mask=mask)
                    loss.append(self.loss_fn(pred, y).item())

            return np.mean(loss).item()

        factor, _ = line_search(simulate_loss)
        return factor

    def execute_expansions(
        self,
        actions: Sequence[Expansion],
        bottleneck: dict[str, torch.Tensor] | str,
        input_B: dict[str, torch.Tensor] | str,
        amplitude_factor: bool,
        evaluate: bool,
        train_dataloader: DataLoader = None,
        dev_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        verbose: bool = False,
    ) -> None:
        """Execute all DAG expansions and save statistics

        Parameters
        ----------
        actions : Sequence[Expansion]
            list with growth actions information
        bottleneck : dict
            dictionary of calculated expressivity bottleneck at each pre-activity
        input_B : dict
            dictionary of post-activity input of each node
        amplitude_factor : bool
            use amplitude factor on new neurons
        evaluate : bool
            evaluate expansion on the data
        train_dataloader : DataLoader, optional
            train dataloader, used if evaluate=True
        dev_dataloader : DataLoader, optional
            development dataloader, used if evaluate=True or amplitude_factor=True
        val_dataloader : DataLoader, optional
            validation dataloader, used if evaluate=True
        verbose : bool, optional
            print info, by default False
        """
        if amplitude_factor:
            assert (
                dev_dataloader is not None
            ), "Development DataLoader should be given if amplitude_factor is True"
        if evaluate:
            assert (
                train_dataloader is not None
            ), "Train DataLoader should be given if evaluate is True"
            assert (
                dev_dataloader is not None
            ), "Development DataLoader should be given if evaluate is True"
            assert (
                val_dataloader is not None
            ), "Validation DataLoader should be given if evaluate is True"
        # Execute all graph growth options
        for expansion in actions:
            # Create a new edge
            if expansion.type == "new edge":
                if verbose:
                    print(
                        f"Adding direct edge from {expansion.previous_node} to {expansion.next_node}"
                    )

                expansion.growth_history = copy.copy(self.growth_history)
                expansion.expand()
                expansion.update_growth_history(
                    self.global_step,
                )

                # Update weight of next_node's incoming edge
                bott_loss_history = self.update_edge_weights(
                    expansion=expansion,
                    bottlenecks=bottleneck,
                    activities=input_B,
                    verbose=verbose,
                )

            # Create/Expand node
            elif (expansion.type == "new node") or (expansion.type == "expanded node"):
                expansion.growth_history = copy.copy(self.growth_history)
                expansion.expand()
                expansion.update_growth_history(
                    self.global_step,
                )

                # Update weights of new edges
                bott_loss_history = self.expand_node(
                    expansion=expansion,
                    bottlenecks=bottleneck,
                    activities=input_B,
                    verbose=verbose,
                )

            # Find amplitude factor that minimizes the overall loss
            if amplitude_factor:
                mask = {
                    "nodes": [expansion.expanding_node],
                    "edges": expansion.new_edges,
                }
                self.find_amplitude_factor(dev_dataloader, mask)
            else:
                factor = 1.0
            self.set_scaling_factor(factor)

            # Evaluate
            expansion.metrics["scaling_factor"] = factor
            expansion.metrics["loss_bott"] = bott_loss_history[-1]
            if evaluate:
                expansion.evaluate(
                    self.dag,
                    train_dataloader=train_dataloader,
                    dev_dataloader=dev_dataloader,
                    val_dataloader=val_dataloader,
                    loss_fn=self.loss_fn,
                )

    def restrict_action_space(
        self,
        actions: list[Expansion],
        chosen_outputs: list[str] | None = None,
        chosen_inputs: list[str] | None = None,
    ) -> list[Expansion]:
        """Reduce action space to connect only to specific node positions
        Can only restrict input or output one at a time

        Parameters
        ----------
        actions : list[Expansion]
            list with growth actions information
        chosen_outputs : list[str], optional
            output node position to restrict to
        chosen_inputs : list[str], optional
            input node position to restrict to

        Returns
        -------
        list[Expansion]
            reduced list with growth actions information
        """
        if chosen_inputs is None and chosen_outputs is None:
            warnings.warn(
                "No input or output was given to restrict the actions. No restriction will happen.",
                UserWarning,
            )
            return actions
        if chosen_inputs is not None and chosen_outputs is not None:
            raise NotImplementedError(
                "You can only restrict inputs or outputs one at a time."
            )
        new_actions = []
        for expansion in actions:
            new_node = expansion.expanding_node
            next_node = expansion.next_nodes
            prev_node = expansion.previous_nodes
            if not isinstance(next_node, list):
                next_node = [next_node]
            if not isinstance(prev_node, list):
                prev_node = [prev_node]
            if chosen_outputs is not None:
                if new_node in chosen_outputs:
                    # Case: expand current node
                    new_actions.append(expansion)
                    continue
                elif len(set(chosen_outputs).intersection(next_node)) != 0:
                    # Case: expand or add immediate previous node
                    new_actions.append(expansion)
                    continue
            elif chosen_inputs is not None:
                # Case: connect previous node
                if len(set(chosen_inputs).intersection(prev_node)) != 0:
                    new_actions.append(expansion)
                    continue
        return new_actions

    def choose_growth_best_action(
        self, options: Sequence[Expansion], use_bic: bool = False, verbose: bool = False
    ) -> None:
        """Choose the growth action with the minimum validation loss greedily
        Log average metrics of the current growth step
        Reconstruct chosen graph and discard the rest

        Parameters
        ----------
        options : Sequence[Expansion]
            list with all possible graphs and their statistics
        use_bic : bool, optional
            use BIC to select the network expansion, by default False
        verbose : bool, optional
            print info, by default False
        """
        # Greedy choice based on validation loss
        selection = {}
        if use_bic:
            for index, option in enumerate(options):
                selection[index] = option.metrics["BIC"]
        else:
            for index, option in enumerate(options):
                selection[index] = option.metrics["loss_val"]

        best_ind = min(selection.items(), key=operator.itemgetter(1))[0]

        if verbose:
            print("Chose option", best_ind)

        # Reconstruct graph
        self.chosen_action = options[best_ind]

        # Make selected nodes and edges non candidate
        self.dag.toggle_node_candidate(self.chosen_action.expanding_node, candidate=False)
        self.dag.toggle_edge_candidate(
            self.chosen_action.previous_node,
            self.chosen_action.next_node,
            candidate=False,
        )

        # Discard unused edges or nodes
        for index, option in enumerate(options):
            if index != best_ind:
                if option.type == "new edge":
                    self.dag.remove_edge(option.previous_node, option.next_node)
                elif option.type == "new node":
                    self.dag.remove_node(option.expanding_node)
        del options

        # Delete updates based on mask
        for prev_node, next_node in self.dag.edges:
            if prev_node == self.chosen_action.expanding_node:
                delete_input = False
                delete_output = True
            elif next_node == self.chosen_action.expanding_node:
                delete_input = True
                delete_output = False
            else:
                delete_input = True
                delete_output = True

            edge_module = self.dag.get_edge_module(prev_node, next_node)
            edge_module.delete_update(
                include_previous=False,
                delete_delta=False,
                delete_input=delete_input,
                delete_output=delete_output,
            )

    def apply_change(self) -> None:
        # Apply changes
        for prev_node, next_node in self.dag.edges:
            factor = self.chosen_action.metrics["scaling_factor"]
            edge_module = self.dag.get_edge_module(prev_node, next_node)

            edge_module.scaling_factor = factor
            edge_module._scaling_factor_next_module.data[0] = factor
            edge_module.apply_change(scaling_factor=factor, apply_previous=False)
            if edge_module.extended_output_layer is not None:
                edge_module._apply_output_changes(
                    scaling_factor=factor, extension_size=self.neurons
                )

        if self.chosen_action.type != "new edge":
            if self.chosen_action.expanding_node in self.dag.nodes:
                expanding_node = self.chosen_action.expanding_node
            else:
                expanding_node = self.chosen_action.adjacent_expanding_node
            # Update size of expanded node
            self.update_size()
            # Rename new node to standard name
            self.dag.rename_nodes({expanding_node: expanding_node.split("_")[0]})

        # Transfer metrics
        self.growth_history = copy.copy(self.chosen_action.growth_history)
        self.growth_loss_train = self.chosen_action.metrics.get("loss_train")
        self.growth_loss_dev = self.chosen_action.metrics.get("loss_dev")
        self.growth_loss_val = self.chosen_action.metrics.get("loss_val")
        self.growth_acc_train = self.chosen_action.metrics.get("acc_train")
        self.growth_acc_dev = self.chosen_action.metrics.get("acc_dev")
        self.growth_acc_val = self.chosen_action.metrics.get("acc_val")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of DAG network

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output of model
        """
        return self.dag(x)

    def extended_forward(self, x: torch.Tensor, mask: dict = {}) -> torch.Tensor:
        """Forward function of DAG network including extensions of the modules

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        mask : dict, optional
            extension mask for specific nodes and edges, by default {}
            example: mask["edges"] for edges and mask["nodes"] for nodes

        Returns
        -------
        torch.Tensor
            output of the extended model
        """
        return self.dag.extended_forward(x, mask=mask)

    def parameters(self) -> Iterator:
        """Iterator of network parameters

        Yields
        ------
        Iterator
            parameters iterator
        """
        return self.dag.parameters()
