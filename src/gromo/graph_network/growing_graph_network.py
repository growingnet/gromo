import copy
import operator
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


try:
    from config.loader import load_config  # type: ignore
    from graph_network.GrowableDAG import GrowableDAG  # type: ignore
    from linear_growing_module import LinearAdditionGrowingModule  # type: ignore
    from utils.utils import (  # type: ignore
        f1_micro,
        global_device,
        line_search,
        mini_batch_gradient_descent,
        set_from_conf,
    )
except ModuleNotFoundError:
    from gromo.config.loader import load_config
    from gromo.graph_network.growable_dag import GrowableDAG
    from gromo.linear_growing_module import LinearAdditionGrowingModule
    from gromo.utils.utils import (
        f1_micro,
        global_device,
        line_search,
        mini_batch_gradient_descent,
        set_from_conf,
    )


class GrowingGraphNetwork(torch.nn.Module):
    """Growable DAG Network

    Parameters
    ----------
    in_features : int, optional
        size of input features, by default 5
    out_features : int, optional
        size of output dimension, by default 1
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
        in_features: int = 5,
        out_features: int = 1,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        neurons: int = 20,
        device: str | None = None,
    ) -> None:
        super(GrowingGraphNetwork, self).__init__()
        self._config_data, _ = load_config()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.neurons = neurons
        self.device = (
            device
            if device is not None
            else set_from_conf(self, "device", global_device(), setter=False)
        )

        self.global_step = 0
        self.global_epoch = 0
        self.loss_fn = nn.CrossEntropyLoss()

        self.reset_network()

    def init_empty_graph(self) -> None:
        """Create empty DAG with start and end nodes"""

        start = "start"
        end = "end"
        edges = [(start, end)]
        node_attributes = {
            start: {
                "type": "L",  # shows what follows
                "size": self.in_features,
                # "activation": "id",
            },
            end: {
                "type": "L",
                "size": self.out_features,
                "use_batch_norm": self.use_batch_norm,
            },
        }
        edge_attributes = {"type": "L", "use_bias": self.use_bias}

        DAG_parameters = {}
        DAG_parameters["edges"] = edges
        DAG_parameters["node_attributes"] = node_attributes
        DAG_parameters["edge_attributes"] = edge_attributes
        DAG_parameters["device"] = self.device

        self.dag = GrowableDAG(DAG_parameters)
        if (start, end) in self.dag.edges:
            self.dag.remove_edge(start, end)

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
        alpha: torch.Tensor,
        omega: torch.Tensor,
        bias: torch.Tensor,
        x: torch.Tensor,
        sigma: nn.Module,
    ) -> torch.Tensor:
        """Output of block connection with specific weights
        Calculates A = omega*sigma(alpha*x + b)

        Parameters
        ----------
        alpha : torch.Tensor
            alpha input weights (neurons, in_features)
        omega : torch.Tensor
            omega output weights (out_features, neurons)
        bias : torch.Tensor
            bias of input layer (neurons,)
        x : torch.Tensor
            input vector (in_features, batch_size)
        sigma : nn.Module
            activation function

        Returns
        -------
        torch.Tensor
            pre-activity of new connection block (out_features, batch_size)
        """
        return torch.matmul(
            omega, sigma(torch.matmul(alpha, x) + bias.sum(dim=1).unsqueeze(1))
        )

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
            return self.block_forward(alpha, omega, bias, B.T, sigma).T

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
        node: str,
        prev_nodes: list[str],
        next_nodes: list[str],
        bottlenecks: dict,
        activities: dict,
        x: torch.Tensor,
        y: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        amplitude_factor: bool = True,
        parallel: bool = True,
        verbose: bool = True,
    ) -> tuple[float, float, float, float, list]:
        """Increase block dimension by expanding node with more neurons
        Increase output size of incoming layers and input size of outgoing layers
        Train new neurons to minimize the expressivity bottleneck

        Parameters
        ----------
        node : str
            name of node where we add neurons
        prev_nodes : list[str]
            list of predecessor connected nodes
        next_nodes : list[str]
            list of successor connected nodes
        bottlenecks : dict
            dictionary with node names as keys and their calculated bottleneck tensors as values
        activities : dict
            dictionary with node names as keys and their pre-activity tensors as values
        x : torch.Tensor
            train input features batch
        y : torch.Tensor
            train true labels of batch
        x1 : torch.Tensor
            development input features batch
        y1 : torch.Tensor
            development true labels batch
        amplitude_factor : bool, optional
            find and apply amplitude factor on the block and its parallel connections, by default True
        parallel : bool, optional
            take into account parallel connections, by default True
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        tuple[float, float, float, float, list]
            train loss, development loss, train accuracy, development accuracy, bottleneck loss history
        """

        node_module = self.dag.get_node_module(node)
        prev_node_modules = self.dag.get_node_modules(prev_nodes)
        next_node_modules = self.dag.get_node_modules(next_nodes)

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
            node_module.post_addition_function,
            bottleneck,
            verbose=verbose,
        )

        # TODO: find applitude factor, create function that applies changes, extended_forward
        # same as I did to apply changes

        # Record layer extensions of new block
        i = 0
        for i_edge, prev_edge_module in enumerate(node_module.previous_modules):
            # Output extension for alpha weights
            in_features = prev_edge_module.in_features
            prev_edge_module._scaling_factor_next_module[0] = 1
            prev_edge_module.extended_output_layer = prev_edge_module.layer_of_tensor(
                weight=alpha[:, i : i + in_features],
                bias=bias[:, i_edge],  # TODO: fix for multiple input layers
            )  # bias is mandatory
            i += in_features
        i = 0
        for next_edge_module in node_module.next_modules:
            # Input extension for omega weights
            out_features = next_edge_module.out_features
            next_edge_module.scaling_factor = 1
            # next_edge_module.extended_input_layer = next_edge_module.layer_of_tensor(
            #     weight=omega[i : i + out_features, :]
            # ) # throws error because of bias
            next_edge_module.extended_input_layer = nn.Linear(
                self.neurons, out_features, bias=False
            )
            next_edge_module.extended_input_layer.weight = nn.Parameter(
                omega[i : i + out_features, :]
            )
            i += out_features

        if amplitude_factor:

            def simulate_factors(factor):
                for prev_edge_module in node_module.previous_modules:
                    prev_edge_module._scaling_factor_next_module[0] = factor
                for next_node_module in next_node_modules:
                    for parallel_edge_module in next_node_module.previous_modules:
                        parallel_edge_module.scaling_factor = factor

                with torch.no_grad():
                    pred = self.extended_forward(x1)
                    loss = self.loss_fn(pred, y1).item()

                return loss

            # Find amplitude factor that minimizes the overall loss
            factor, min_loss = line_search(simulate_factors, verbose=verbose)
        else:
            factor = 1

        # Apply final changes
        for prev_edge_module in node_module.previous_modules:
            # we do not need to change the _scaling_factor_next_module as it is
            # given as a parameter of _apply_output_changes
            # prev_edge_module._scaling_factor_next_module = factor
            prev_edge_module._apply_output_changes(factor)
            # Delete activities
            prev_edge_module.delete_update(include_previous=False)

        for next_node_module in next_node_modules:
            for parallel_module in next_node_module.previous_modules:
                parallel_module.scaling_factor = factor
                parallel_module.apply_change(apply_previous=False)
                # Delete activities
                parallel_module.delete_update(include_previous=False)
            # Delete activities
            next_node_module.delete_update()

        node_module.delete_update()

        # Update size
        self.dag.nodes[node]["size"] += self.neurons
        # Evaluation
        acc_train, loss_train = self.evaluate(x, y, verbose=False)
        acc_dev, loss_dev = self.evaluate(x1, y1, verbose=False)

        # TODO FUTURE : Save updates to return

        return loss_train, loss_dev, acc_train, acc_dev, loss_history

    def update_edge_weights(
        self,
        prev_node: str,
        next_node: str,
        bottlenecks: dict,
        activities: dict,
        x: torch.Tensor,
        y: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        amplitude_factor: bool = True,
        verbose: bool = True,
    ) -> tuple[float, float, float, float, list]:
        """Update weights of a single layer edge
        Train layer to minimize the expressivity bottleneck

        Parameters
        ----------
        prev_node : str
            node at the start of the edge
        next_node : str
            node at the end of the edge
        bottlenecks : dict
            dictionary with node names as keys and their calculated bottleneck tensors as values
        activities : dict
            dictionary with node names as keys and their pre-activity tensors as values
        x : torch.Tensor
            train input features batch
        y : torch.Tensor
            train true labels batch
        x1 : torch.Tensor
            development input features batch
        y1 : torch.Tensor
            development true labels batch
        amplitude_factor : bool, optional
            find and apply amplitude factor on the block and its parallel connections, by default True
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        tuple[float, float, float, float, list]
            train loss, development loss, train accuracy, development accuracy, bottleneck loss history
        """

        new_edge_module = self.dag.get_edge_module(prev_node, next_node)
        prev_node_module = self.dag.get_node_module(prev_node)
        next_node_module = self.dag.get_node_module(next_node)

        bottleneck = bottlenecks[next_node_module._name]
        activity = activities[prev_node_module._name]

        # TODO: gradient to find edge weights
        # [bi-level]  loss = edge_weight - bottleneck
        # [joint opt] loss = edge_weight + possible updates - desired_update

        weight = torch.rand(
            (new_edge_module.out_features, new_edge_module.in_features),
            device=self.device,
        )
        bias = torch.rand((new_edge_module.out_features), device=self.device)
        weight = weight / np.sqrt(weight.numel())
        bias = bias / np.sqrt(bias.numel())
        weight = weight.detach().clone().requires_grad_()
        bias = bias.detach().clone().requires_grad_()

        # # Testing
        # weight = torch.nn.init.orthogonal_(weight)
        forward_fn = lambda activity: nn.functional.linear(activity, weight, bias)

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
            gamma = self.find_input_amplitude_factor(
                x1, y1, next_node_module, verbose
            )  # MEMORY ISSUE
        else:
            gamma = 1.0

        # Apply new edge weights
        # new_edge = self.dag.get_edge_module(prev_node, next_node)
        # print(delta_W_star[new_edge.name][0].shape)
        # print(new_edge.layer.weight[:5, 0])
        # # ATTENTION: Only applies the optimal change
        # new_edge.scaling_factor = gamma # is multiplied squared
        # new_edge.apply_change()
        # print(new_edge.layer.weight[:5, 0])

        # TODO: Apply existing weight updates to the rest of the edges, or all at once
        for edge in next_node_module.previous_modules:
            edge.scaling_factor = gamma
            edge.apply_change(apply_previous=False)
            edge.reset_computation()
            edge.delete_update(include_previous=False)

        # next_node_module.reset_computation()
        next_node_module.delete_update()

        # Important to update size of next addition module!
        # It cannot happen automatically because
        # there is no layer extension recorded
        # next_node_module.update_size()

        # Evaluate on train and development sets
        acc_train, loss_train = self.evaluate(x, y, verbose=False)
        acc_dev, loss_dev = self.evaluate(x1, y1, verbose=False)

        return loss_train, loss_dev, acc_train, acc_dev, loss_history

    def find_input_amplitude_factor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        next_module: LinearAdditionGrowingModule,
        verbose: bool = True,
    ) -> float:
        """Find amplitude factor with line search for a single layer edge with extended updates

        Parameters
        ----------
        x : torch.Tensor
            input features batch
        y : torch.Tensor
            true labels batch
        next_module : LinearAdditionGrowingModule
            node module at the end of the edge
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        float
            amplitude factor that minimizes overall loss
        """

        def simulate_loss(gamma_factor, module=next_module):
            # TODO: Change with extended_forward
            for edge in module.previous_modules:
                # update = delta_W_star[edge.name]
                # weight = gamma_factor * update[0]
                # bias = gamma_factor * update[1]
                # edge.parameter_step(weight, bias)
                edge._scaling_factor_next_module[0] = gamma_factor

            with torch.no_grad():
                # pred = self(x)
                pred = self.extended_forward(x)
                loss = self.loss_fn(pred, y).item()

            # for edge in module.previous_modules:
            #     update = delta_W_star[edge.name]
            #     weight = -gamma_factor * update[0]
            #     bias = -gamma_factor * update[1]
            #     edge.parameter_step(weight, bias)

            return loss

        gamma_factor, _ = line_search(simulate_loss, verbose=verbose)
        return gamma_factor

    def execute_expansions(
        self,
        generations: list[dict],
        bottleneck: dict,
        input_B: dict,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_dev: torch.Tensor,
        Y_dev: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        amplitude_factor: bool,
        verbose: bool = False,
    ) -> None:
        """Execute all DAG expansions and save statistics

        Parameters
        ----------
        generations : list[dict]
            list of dictionaries with growth actions information
        bottleneck : dict
            dictionary of calculated expressivity bottleneck at each pre-activity
        input_B : dict
            dictionary of post-activity input of each node
        X_train : torch.Tensor
            train features
        Y_train : torch.Tensor
            train labels
        X_dev : torch.Tensor
            development features
        Y_dev : torch.Tensor
            development labels
        X_val : torch.Tensor
            validation features
        Y_val : torch.Tensor
            validation labels
        amplitude_factor : bool
            use amplitude factor on new neurons
        verbose : bool, optional
            print info, by default False
        """
        # Execute all graph growth options
        for gen in generations:
            # Create a new edge
            if gen.get("type") == "edge":
                attributes = gen.get("attributes", {})
                prev_node = attributes.get("previous_node")
                next_node = attributes.get("next_node")

                if verbose:
                    print(f"Adding direct edge from {prev_node} to {next_node}")

                model_copy = copy.deepcopy(self)
                model_copy.to(self.device)
                model_copy.dag.add_direct_edge(
                    prev_node, next_node, attributes.get("edge_attributes", {})
                )

                model_copy.growth_history_step(neurons_added=[(prev_node, next_node)])

                # Update weight of next_node's incoming edge
                loss_train, loss_dev, acc_train, acc_dev, _ = (
                    model_copy.update_edge_weights(
                        prev_node=prev_node,
                        next_node=next_node,
                        bottlenecks=bottleneck,
                        activities=input_B,
                        x=X_train,
                        y=Y_train,
                        x1=X_dev,
                        y1=Y_dev,
                        amplitude_factor=amplitude_factor,
                        verbose=verbose,
                    )
                )

                # TODO: save updates weight tensors
                # gen[] =

            # Create/Expand node
            elif gen.get("type") == "node":
                attributes = gen.get("attributes", {})
                new_node = attributes.get("new_node")
                prev_nodes = attributes.get("previous_node")
                next_nodes = attributes.get("next_node")
                new_edges = attributes.get("new_edges")

                # copy.deepcopy(self.dag)
                model_copy = copy.deepcopy(self)
                model_copy.to(self.device)

                if new_node not in model_copy.dag.nodes:
                    model_copy.dag.add_node_with_two_edges(
                        prev_nodes,
                        new_node,
                        next_nodes,
                        attributes.get("node_attributes"),
                        attributes.get("edge_attributes", {}),
                    )
                    prev_nodes = [prev_nodes]
                    next_nodes = [next_nodes]

                model_copy.growth_history_step(
                    nodes_added=new_node, neurons_added=new_edges
                )

                # Update weights of new edges
                loss_train, loss_dev, acc_train, acc_dev, _ = model_copy.expand_node(
                    node=new_node,
                    prev_nodes=prev_nodes,
                    next_nodes=next_nodes,
                    bottlenecks=bottleneck,
                    activities=input_B,
                    x=X_train,
                    y=Y_train,
                    x1=X_dev,
                    y1=Y_dev,
                    amplitude_factor=amplitude_factor,
                    verbose=verbose,
                )

                # TODO: save update weight tensors
                # gen[] =

            # Evaluate
            acc_val, loss_val = model_copy.evaluate(X_val, Y_val, verbose=False)

            # TODO: return all info instead of saving
            gen["loss_train"] = loss_train
            gen["loss_dev"] = loss_dev
            gen["loss_val"] = loss_val
            gen["acc_train"] = acc_train
            gen["acc_dev"] = acc_dev
            gen["acc_val"] = acc_val
            gen["nb_params"] = model_copy.dag.count_parameters_all()
            gen["BIC"] = model_copy.BIC(loss_val, n=len(X_val))

            # TEMP: save DAG
            gen["dag"] = model_copy.dag
            gen["growth_history"] = model_copy.growth_history

        del model_copy

    def calculate_bottleneck(
        self, generations: list[dict], X_train: torch.Tensor, Y_train: torch.Tensor
    ) -> tuple[dict, dict]:
        """Calculate expressivity bottleneck on important nodes
        Assign hooks where necessary and update tensors with a single forward-backward
        Keep track of bottleneck and post-activities

        Parameters
        ----------
        generations : list[dict]
            list of dictionaries with growth actions information
        X_train : torch.Tensor
            train features
        Y_train : torch.Tensor
            train labels

        Returns
        -------
        tuple[dict, dict]
            bottleneck of nodes, input of nodes
        """
        # Handle empty graph case
        constant_module = False
        if self.dag.is_empty():
            # Create constant module if the graph is empty
            constant_module = True
            edge_attributes = {"type": "L", "use_bias": self.use_bias, "constant": True}
            self.dag.add_direct_edge("start", "end", edge_attributes)

        # Find nodes of interest
        prev_node_modules = set()
        next_node_modules = set()
        for gen in generations:
            attributes = gen.get("attributes", {})

            prev_node = attributes.get("previous_node")
            next_node = attributes.get("next_node")
            if not isinstance(prev_node, list):
                prev_node = [prev_node]
            if not isinstance(next_node, list):
                next_node = [next_node]

            prev_node_modules.update(prev_node)
            next_node_modules.update(next_node)

        # Add hooks on node modules of interest
        prev_node_modules = self.dag.get_node_modules(prev_node_modules)
        next_node_modules = self.dag.get_node_modules(next_node_modules)
        for node_module in prev_node_modules:
            node_module.store_activity = True
        for node_module in next_node_modules:
            node_module.init_computation()

        # Forward - Backward step
        pred = self(X_train)
        loss = self.loss_fn(pred, Y_train)
        loss.backward()

        input_B = {}
        bottleneck = {}

        # Update tensors
        for node_module in next_node_modules:
            assert node_module.previous_tensor_s is not None
            assert node_module.previous_tensor_m is not None
            node_module.previous_tensor_s.update()
            node_module.previous_tensor_m.update()

            # Compute optimal possible updates
            deltas = node_module.compute_optimal_delta(update=True, return_deltas=True)

            # Compute expressivity bottleneck
            bottleneck[node_module._name] = (
                node_module.projected_v_goal().clone().detach()
            )  # (batch_size, out_features)

            del deltas
            # TODO: separate to functions that add the hooks and remove them

            if constant_module:
                assert torch.all(
                    bottleneck[node_module._name] == node_module.pre_activity.grad
                ), "Graph is empty and the bottleneck should be the same as the pre_activity gradient. Expected: {node_module.pre_activity.grad} Found: {bottleneck[node_module._name]}"

            # Reset tensors and remove hooks
            node_module.reset_computation()

        # Retrieve input activities
        for node_module in prev_node_modules:
            assert node_module.activity is not None
            # Save input activity of input layers
            input_B[node_module._name] = node_module.activity.clone().detach()

            # Reset tensors and remove hooks
            node_module.store_activity = False
            # node_module.delete_update()

        # Reset all hooks
        for next_node_module in next_node_modules:
            for parallel_module in next_node_module.previous_modules:
                parallel_module.reset_computation()
                # DO NOT delete updates
                # parallel_module.delete_update(include_previous=False)
            # Delete activities
            next_node_module.delete_update()

        if constant_module:
            # Remove constant module if needed
            self.dag.remove_direct_edge("start", "end")

        return bottleneck, input_B

    def restrict_action_space(
        self, generations: list[dict], chosen_position: str
    ) -> list[dict]:
        """Reduce action space to contribute only to specific node position

        Parameters
        ----------
        generations : list[dict]
            list of dictionaries with growth actions information
        chosen_position : str
            node position to restrict to

        Returns
        -------
        list[dict]
            reduced list of dictionaries with growth actions information
        """
        new_generations = []
        for gen in generations:
            new_node = gen["attributes"].get("new_node", -1)
            next_node = gen["attributes"].get("next_node", -1)
            if new_node == chosen_position:
                # Case: expand current node
                new_generations.append(gen)
            if isinstance(next_node, list) and chosen_position in next_node:
                # Case: expand immediate previous node
                new_generations.append(gen)
            elif next_node == chosen_position:
                # Case: add new previous node
                new_generations.append(gen)
        return new_generations

    def choose_growth_best_action(
        self, options: list[dict], use_bic: bool = False, verbose: bool = False
    ) -> None:
        """Choose the growth action with the minimum validation loss greedily
        Log average metrics of the current growth step
        Reconstruct chosen graph and discard the rest

        Parameters
        ----------
        options : list[dict]
            dictionary with all possible graphs and their statistics
        use_bic : bool, optional
            use BIC to select the network expansion, by default False
        verbose : bool, optional
            print info, by default False
        """
        # Greedy choice based on validation loss
        selection = {}
        if use_bic:
            for index, item in enumerate(options):
                selection[index] = item["BIC"]
        else:
            for index, item in enumerate(options):
                selection[index] = item["loss_val"]

        best_ind = min(selection.items(), key=operator.itemgetter(1))[0]

        if verbose:
            print("Chose option", best_ind)

        # Reconstruct graph
        best_option = options[best_ind]
        del options

        self.dag = copy.copy(best_option["dag"])
        self.growth_history = best_option["growth_history"]
        self.growth_loss_train = best_option["loss_train"]
        self.growth_loss_dev = best_option["loss_dev"]
        self.growth_loss_val = best_option["loss_val"]
        self.growth_acc_train = best_option["acc_train"]
        self.growth_acc_dev = best_option["acc_dev"]
        self.growth_acc_val = best_option["acc_val"]
        del best_option

    def define_next_generations(self) -> list[dict]:
        """Find all possible growth extensions for the current graph

        Returns
        -------
        list[dict]
            list of dictionaries with growth actions information
        """
        # TODO: check if they allow growing
        direct_edges, one_hop_edges = self.dag.find_possible_extensions()

        # gen_id = 0
        generations = []

        # All possible new direct edges
        for attr in direct_edges:
            previous_node = attr.get("previous_node")
            next_node = attr.get("next_node")

            edge_name = f"l{previous_node}_{next_node}"
            gen = {
                "type": "edge",
                "attributes": attr,
                "id": edge_name,
                "evolved": False,
            }
            generations.append(gen)

        # All possible one-hop connections
        for attr in one_hop_edges:
            previous_node = attr.get("previous_node")
            new_node = attr.get("new_node")
            next_node = attr.get("next_node")
            new_edges = [
                (previous_node, new_node),
                (new_node, next_node),
            ]
            attr["new_edges"] = new_edges

            gen = {
                "type": "node",
                "attributes": attr,
                "id": new_node,
                "evolved": False,
            }
            generations.append(gen)

        # All existing nodes
        for node in self.dag.nodes:
            if (node == self.dag.root) or (node == self.dag.end):
                continue

            previous_nodes = [n for n in self.dag.predecessors(node)]
            next_nodes = [n for n in self.dag.successors(node)]

            new_edges = [in_edge for in_edge in self.dag.in_edges(node)]
            new_edges.extend([out_edge for out_edge in self.dag.out_edges(node)])

            attr = {
                "new_node": node,
                "previous_node": previous_nodes,
                "next_node": next_nodes,
                "new_edges": new_edges,
            }
            gen = {
                "type": "node",
                "attributes": attr,
                "id": node,
                "evolved": False,
            }
            generations.append(gen)

        return generations

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

    def BIC(self, loss: float, n: int) -> float:
        """Bayesian Information Criterion
        BIC = k*log(n) - 2log(L), where k is the number of parameters

        Parameters
        ----------
        loss : float
            loss of the model
        n : int
            number of samples used for training

        Returns
        -------
        float
            BIC score
        """
        k = self.dag.count_parameters_all()
        return k * np.log2(n) - 2 * np.log2(loss)

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        with_f1score: bool = False,
        verbose: bool = True,
    ) -> tuple[float, float] | tuple[float, float, float]:
        """Evaluate network on batch

        Important: Assumes that the batch is already on the correct device

        Parameters
        ----------
        x : torch.Tensor
            input features tensor
        y : torch.Tensor
            true labels tensor
        with_f1score : bool, optional
            calculate f1-score, by default False
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        tuple[float, float] | tuple[float, float, float]
            accuracy and loss, optionally f1-score
        """
        with torch.no_grad():
            pred = self(x)
            loss = self.loss_fn(pred, y)

        if self.out_features > 1:
            final_pred = pred.argmax(axis=1)
            correct = (final_pred == y).int().sum()
            accuracy = (correct / pred.shape[0]).item()
        else:
            accuracy = -1

        # if verbose and self.out_features > 1:
        # TODO: replace dependency
        #     mca = classification.MulticlassAccuracy(
        #         num_classes=self.out_features, average="micro"
        #     ).to(self.device)
        #     print(f"{mca(final_pred, y)=}")
        #     confmat = classification.ConfusionMatrix(
        #         task="multiclass", num_classes=self.out_features
        #     ).to(self.device)
        #     confmat(final_pred, y)
        #     confmat.plot()

        if with_f1score:
            if self.out_features > 1:
                f1score = f1_micro(y.cpu(), final_pred.cpu())
            else:
                f1score = -1
            return accuracy, loss.item(), f1score

        return accuracy, loss.item()

    def evaluate_dataset(self, dataloader: DataLoader) -> tuple[float, float]:
        """Evaluate network on dataset

        Parameters
        ----------
        dataloader : DataLoader
            dataloader containing the data

        Returns
        -------
        tuple[float, float]
            accuracy and loss
        """
        correct, total = 0, 0

        loss = []
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                pred = self(x)
                loss.append(self.loss_fn(pred, y).item())

            final_pred = pred.argmax(axis=1)
            count_this = final_pred == y
            count_this = count_this.sum()

            correct += count_this.item()
            total += len(pred)

        return (correct / total), np.mean(loss).item()
