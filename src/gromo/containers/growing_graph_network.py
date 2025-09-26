import copy
import operator
import warnings
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_dag import Expansion, GrowingDAG
from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    Conv2dMergeGrowingModule,
)
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import (
    evaluate_extended_dataset,
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
        use_bias: bool = True,
        use_batch_norm: bool = False,
        layer_type: str = "linear",
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
        self.input_shape = input_shape
        self.neurons = neurons

        self.global_step = 0
        self.global_epoch = 0
        self.loss_fn = loss_fn

        self.reset_network()

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

    def init_empty_graph(self) -> None:
        """Create empty DAG with start and end nodes"""
        self.dag = GrowingDAG(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.neurons,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            default_layer_type=self.layer_type,
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
        self.growth_history_step()

    def growth_history_step(
        self, neurons_added: list = [], neurons_updated: list = [], nodes_added: list = []
    ) -> None:
        """Record recent modifications on history dictionary

        Parameters
        ----------
        neurons_added : list, optional
            list of edges that were added or increased in dimension, by default []
        neurons_updated : list, optional
            list of edges whose weights were updated, by default []
        nodes_added : list, optional
            list of nodes that were added, by default []
        """
        # TODO: keep track of updated edges/neurons_updated
        if self.global_step not in self.growth_history:
            self.growth_history[self.global_step] = {}

        keep_max = lambda new_value, key: max(
            self.growth_history[self.global_step].get(key, 0), new_value
        )

        step = {}
        for edge in self.dag.edges:
            new_value = (
                2 if edge in neurons_added else 1 if edge in neurons_updated else 0
            )
            step[str(edge)] = keep_max(new_value, str(edge))

        for node in self.dag.nodes:
            new_value = 2 if node in nodes_added else 0
            step[str(node)] = keep_max(new_value, str(node))

        self.growth_history[self.global_step].update(step)

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
        B: torch.Tensor,
        sigma: nn.Module,
        bottleneck: torch.Tensor,
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

        def forward_fn(B):
            return self.block_forward(
                F.linear if linear else F.conv2d,
                alpha,
                omega,
                bias,
                B,
                sigma,
                **operation_args if not linear else {},
            )

        # # TODO FUTURE : try with extended forward, you have to set extended layers on all modules, avoid copying the model
        # new_activity = self.block_forward(alpha, omega, B.T, sigma).T # (batch_size, total_out_features)
        loss_history, _ = mini_batch_gradient_descent(
            model=forward_fn,
            parameters=[alpha, omega, bias],
            cost_fn=self.bottleneck_loss,
            X=B,
            Y=bottleneck,
            batch_size=256,
            lrate=1e-3,
            max_epochs=100,
            fast=True,
            verbose=verbose,
            # loss_name="expected bottleneck",
            # title=f"[Step {self.global_step}] Adding new block",
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
        bottlenecks: dict,
        activities: dict,
        dataloader: DataLoader,
        amplitude_factor: bool = True,
        parallel: bool = True,
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
        dataloader: DataLoader
            development dataloader with input features and targets
        amplitude_factor : bool, optional
            find and apply amplitude factor on the block and its parallel connections, by default True
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
        prev_node_modules = self.dag.get_node_modules(expansion.previous_nodes)
        next_node_modules = self.dag.get_node_modules(expansion.next_nodes)

        bottleneck, input_x = [], []
        for next_node_module in next_node_modules:
            bottleneck.append(bottlenecks[next_node_module._name])
        bottleneck = torch.cat(bottleneck, dim=1)  # (batch_size, total_out_features)
        for prev_node_module in prev_node_modules:  # TODO: check correct order
            input_x.append(activities[prev_node_module._name])
        input_x = torch.cat(input_x, dim=1)  # (batch_size, total_in_features)

        total_in_features = input_x.shape[1]
        total_out_features = bottleneck.shape[1]
        in_edges = len(node_module.previous_modules)

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
        bias = torch.rand(
            (self.neurons, in_edges), device=self.device
        )  # TODO: fix bias for multiple input layers
        alpha = alpha / np.sqrt(alpha.numel())
        omega = omega / np.sqrt(omega.numel())
        bias = bias / np.sqrt(
            bias.numel()
        )  # TODO: fix bias, now using one for all input layers
        alpha = alpha.detach().clone().requires_grad_()
        omega = omega.detach().clone().requires_grad_()
        bias = bias.detach().clone().requires_grad_()

        # Gradient descent on bottleneck
        # [bi-level]  loss = edge_weight - bottleneck
        # [joint opt] loss = edge_weight + possible updates - desired_update
        loss_history = self.bi_level_bottleneck_optimization(
            alpha,
            omega,
            bias,
            input_x,
            node_module.post_merge_function,
            bottleneck,
            linear=isinstance(node_module, LinearMergeGrowingModule),
            operation_args={
                "padding": "same",
            },
            verbose=verbose,
        )

        # TODO: find applitude factor, create function that applies changes, extended_forward
        # same as I did to apply changes

        # Record layer extensions of new block
        i = 0
        alpha = alpha.view(self.neurons, -1)  # (neurons, total_in_features)
        for i_edge, prev_edge_module in enumerate(node_module.previous_modules):
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
        for next_edge_module in node_module.next_modules:
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

        if amplitude_factor:
            # Find amplitude factor that minimizes the overall loss
            mask = {"nodes": [expansion.expanding_node]}
            factor = self.find_amplitude_factor(
                net=self.dag,
                dataloader=dataloader,
                node_module=node_module,
                next_node_modules=next_node_modules,
                mask=mask,
            )
        else:
            factor = 1

        # Apply final changes
        for prev_edge_module in node_module.previous_modules:
            # we do not need to change the _scaling_factor_next_module as it is
            # given as a parameter of _apply_output_changes
            prev_edge_module._scaling_factor_next_module[0] = factor  # Warning

        for next_node_module in next_node_modules:
            for parallel_module in next_node_module.previous_modules:
                parallel_module.scaling_factor = factor

        # TODO FUTURE : Save updates to return

        return loss_history, factor

    def update_edge_weights(
        self,
        expansion: Expansion,
        bottlenecks: dict,
        activities: dict,
        dataloader: DataLoader,
        amplitude_factor: bool = True,
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
        dataloader : DataLoader
            development dataloader with input features and targets
        amplitude_factor : bool, optional
            find and apply amplitude factor on the block and its parallel connections, by default True
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

        bottleneck = bottlenecks[next_node_module._name]
        activity = activities[prev_node_module._name]

        # TODO: gradient to find edge weights
        # [bi-level]  loss = edge_weight - bottleneck
        # [joint opt] loss = edge_weight + possible updates - desired_update

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
            batch_size=256,
            lrate=1e-3,
            max_epochs=100,
            fast=True,
            verbose=verbose,
        )

        # Record layer extensions
        new_edge_module.optimal_delta_layer = new_edge_module.layer_of_tensor(
            weight, bias
        )

        # Find amplitude factor with line search
        # TODO: fix squared value, or check why
        if amplitude_factor:
            factor = self.find_amplitude_factor(
                net=self.dag,
                dataloader=dataloader,
                node_module=next_node_module,
                mask={"edges": [(expansion.previous_node, expansion.next_node)]},
            )
        else:
            factor = 1.0

        # TODO: Apply existing weight updates to the rest of the edges, or all at once
        for edge in next_node_module.previous_modules:
            edge.scaling_factor = factor

        return loss_history, factor

    def find_amplitude_factor(
        self,
        net: GrowingContainer,
        dataloader: DataLoader,
        node_module: LinearMergeGrowingModule,
        next_node_modules: list[LinearMergeGrowingModule] = None,
        mask: dict = {},
    ) -> float:
        """Find amplitude factor with line search

        Parameters
        ----------
        net : GrowingDAG
            network of interest
        dataloader : DataLoader
            dataloader with input features and target
        node_module : LinearMergeGrowingModule
            node module to be extended or node module at the end of the edge in case of single edge
        next_node_modules : list[LinearMergeGrowingModule], optional
            next node modules of module to be extended, leave empty in case of single edge, by default None
        mask : dict, optional
            extension mask for specific nodes and edges, by default {}
            example: mask["edges"] for edges and mask["nodes"] for nodes

        Returns
        -------
        float
            amplitude factor that minimizes overall loss
        """

        def simulate_loss(factor):
            for prev_edge_module in node_module.previous_modules:
                prev_edge_module.scaling_factor = factor
                prev_edge_module._scaling_factor_next_module[0] = factor
            if next_node_modules is not None:
                for next_node_module in next_node_modules:
                    for parallel_edge_module in next_node_module.previous_modules:
                        parallel_edge_module.scaling_factor = factor

            loss = []
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = net.extended_forward(x, mask=mask)
                    loss.append(self.loss_fn(pred, y).item())

            return np.mean(loss).item()

        factor, _ = line_search(simulate_loss)
        return factor

    def execute_expansions(
        self,
        actions: list[Expansion],
        bottleneck: dict,
        input_B: dict,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        val_dataloader: DataLoader,
        amplitude_factor: bool,
        discard_underperforming: bool = False,
        discard_metric: str = "loss_val",
        verbose: bool = False,
    ) -> None:
        """Execute all DAG expansions and save statistics

        Parameters
        ----------
        actions : list[Expansion]
            list with growth actions information
        bottleneck : dict
            dictionary of calculated expressivity bottleneck at each pre-activity
        input_B : dict
            dictionary of post-activity input of each node
        train_dataloader : DataLoader
            train dataloader
        dev_dataloader : DataLoader
            development dataloader
        val_dataloader : DataLoader
            validation dataloader
        amplitude_factor : bool
            use amplitude factor on new neurons
        verbose : bool, optional
            print info, by default False
        """
        # Execute all graph growth options
        best_metric_value = np.inf
        best_metric_index = -1
        for i, expansion in enumerate(actions):
            # Create a new edge
            if expansion.type == "new edge":
                if verbose:
                    print(
                        f"Adding direct edge from {expansion.previous_node} to {expansion.next_node}"
                    )

                expansion.growth_history = copy.copy(self.growth_history)
                expansion.expand()
                expansion.update_growth_history(
                    self.global_step, neurons_added=expansion.new_edges
                )

                # Update weight of next_node's incoming edge
                bott_loss_history, factor = self.update_edge_weights(
                    expansion=expansion,
                    bottlenecks=bottleneck,
                    activities=input_B,
                    dataloader=dev_dataloader,
                    amplitude_factor=amplitude_factor,
                    verbose=verbose,
                )

            # Create/Expand node
            elif (expansion.type == "new node") or (expansion.type == "expanded node"):
                expansion.growth_history = copy.copy(self.growth_history)
                expansion.expand()
                expansion.update_growth_history(
                    self.global_step,
                    nodes_added=[expansion.expanding_node],
                    neurons_added=expansion.new_edges,
                )

                # Update weights of new edges
                bott_loss_history, factor = self.expand_node(
                    expansion=expansion,
                    bottlenecks=bottleneck,
                    activities=input_B,
                    dataloader=dev_dataloader,
                    amplitude_factor=amplitude_factor,
                    verbose=verbose,
                )

            # Evaluate
            mask = {
                "nodes": [expansion.expanding_node],
                "edges": (
                    [expansion.new_edges]
                    if expansion.type == "new edge"
                    else expansion.new_edges
                ),
            }
            acc_train, loss_train = evaluate_extended_dataset(
                self.dag, train_dataloader, loss_fn=self.loss_fn, mask=mask
            )
            acc_dev, loss_dev = evaluate_extended_dataset(
                self.dag, dev_dataloader, loss_fn=self.loss_fn, mask=mask
            )
            acc_val, loss_val = evaluate_extended_dataset(
                self.dag, val_dataloader, loss_fn=self.loss_fn, mask=mask
            )

            # TODO: return all info instead of saving
            expansion.metrics["scaling_factor"] = factor
            expansion.metrics["loss_bott"] = bott_loss_history[-1]
            expansion.metrics["loss_train"] = loss_train
            expansion.metrics["loss_dev"] = loss_dev
            expansion.metrics["loss_val"] = loss_val
            expansion.metrics["acc_train"] = acc_train
            expansion.metrics["acc_dev"] = acc_dev
            expansion.metrics["acc_val"] = acc_val
            # expansion.metrics["nb_params"] = expansion.dag.count_parameters_all()
            # expansion.metrics["BIC"] = self.BIC(
            #     expansion.dag, loss_val, n=len(val_dataloader.dataset)
            # )

            if discard_underperforming and len(actions) > 1:
                if expansion.metrics[discard_metric] <= best_metric_value:
                    if verbose:
                        if best_metric_index >= 0:
                            print(
                                f"Discarding expansion with {discard_metric} = {best_metric_value:.4f} > {expansion.metrics[discard_metric]:.4f}"
                            )
                        print(
                            f"New best expansion with {discard_metric} = {expansion.metrics[discard_metric]:.4f}"
                        )
                    if i != 0:
                        actions[best_metric_index].discard()
                    best_metric_value = expansion.metrics[discard_metric]
                    best_metric_index = i
                else:
                    if verbose:
                        print(
                            f"Discarding expansion with {discard_metric} = {expansion.metrics[discard_metric]:.4f} > {best_metric_value:.4f}"
                        )
                    del expansion

    def restrict_action_space(
        self, actions: list[Expansion], chosen_position: str
    ) -> list[Expansion]:
        """Reduce action space to contribute only to specific node position

        Parameters
        ----------
        actions : list[Expansion]
            list with growth actions information
        chosen_position : str
            node position to restrict to

        Returns
        -------
        list[Expansion]
            reduced list with growth actions information
        """
        new_actions = []
        for expansion in actions:
            new_node = expansion.expanding_node
            next_node = expansion.next_nodes
            if new_node == chosen_position:
                # Case: expand current node
                new_actions.append(expansion)
            elif isinstance(next_node, list) and chosen_position in next_node:
                # Case: expand immediate previous node
                new_actions.append(expansion)
            elif next_node == chosen_position:
                # Case: add new previous node
                new_actions.append(expansion)
            else:
                del expansion
        return new_actions

    def choose_growth_best_action(
        self, options: list[Expansion], use_bic: bool = False, verbose: bool = False
    ) -> None:
        """Choose the growth action with the minimum validation loss greedily
        Log average metrics of the current growth step
        Reconstruct chosen graph and discard the rest

        Parameters
        ----------
        options : list[Expansion]
            list with all possible graphs and their statistics
        use_bic : bool, optional
            use BIC to select the network expansion, by default False
        verbose : bool, optional
            print info, by default False
        """
        # Greedy choice based on validation loss
        selection = {}
        if use_bic:
            for index, expansion in enumerate(options):
                selection[index] = expansion.metrics["BIC"]
        else:
            for index, expansion in enumerate(options):
                selection[index] = expansion.metrics["loss_val"]

        best_ind = min(selection.items(), key=operator.itemgetter(1))[0]

        if verbose:
            print("Chose option", best_ind)

        # Reconstruct graph
        best_option = options[best_ind]
        del options  # TODO: memory freed?

        self.dag = copy.copy(best_option.dag)
        self.growth_history = copy.copy(best_option.growth_history)
        self.growth_loss_train = best_option.metrics["loss_train"]
        self.growth_loss_dev = best_option.metrics["loss_dev"]
        self.growth_loss_val = best_option.metrics["loss_val"]
        self.growth_acc_train = best_option.metrics["acc_train"]
        self.growth_acc_dev = best_option.metrics["acc_dev"]
        self.growth_acc_val = best_option.metrics["acc_val"]
        del best_option

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

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of DAG network including extensions of the modules

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output of the extended model
        """
        return self.dag.extended_forward(x)

    def parameters(self) -> Iterator:
        """Iterator of network parameters

        Yields
        ------
        Iterator
            parameters iterator
        """
        return self.dag.parameters()

    def BIC(self, net: GrowingDAG, loss: float, n: int) -> float:
        """Bayesian Information Criterion
        BIC = k*log(n) - 2log(L), where k is the number of parameters

        Parameters
        ----------
        net : GrowingDAG
            network of interest
        loss : float
            loss of the model
        n : int
            number of samples used for training

        Returns
        -------
        float
            BIC score
        """
        k = net.count_parameters_all()
        return k * np.log2(n) - 2 * np.log2(loss)
