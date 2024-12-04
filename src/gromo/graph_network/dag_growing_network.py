import copy
import json
import operator

# import sys
import os
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

# from memory_profiler import profile as memprofile
from torch.utils.data import DataLoader, Dataset, random_split


# from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis


# from memory_profiler import LogFile

try:
    from graph_network.GrowableDAG import GrowableDAG  # type: ignore
    from linear_growing_module import LinearAdditionGrowingModule  # type: ignore
    from utils.logger import Logger  # type: ignore
    from utils.profiling import CustomProfile, profile_function  # type: ignore
    from utils.utils import (  # type: ignore;
        DAG_to_pyvis,
        batch_gradient_descent,
        f1_micro,
        global_device,
        line_search,
    )
except ModuleNotFoundError:
    from gromo.graph_network.GrowableDAG import GrowableDAG
    from gromo.linear_growing_module import LinearAdditionGrowingModule
    from gromo.utils.logger import Logger
    from gromo.utils.profiling import CustomProfile, profile_function
    from gromo.utils.utils import (
        DAG_to_pyvis,
        batch_gradient_descent,
        f1_micro,
        global_device,
        line_search,
    )


def safe_forward(self, input: torch.Tensor) -> torch.Tensor:
    """Safe Linear forward function for empty input tensors
    Resolves bug with shape transformation when using cuda

    Parameters
    ----------
    input : torch.Tensor
        input tensor

    Returns
    -------
    torch.Tensor
        F.linear forward function output
    """
    assert (
        input.shape[1] == self.in_features
    ), f"Input shape {input.shape} must match the input feature size. Expected: {self.in_features}, Found: {input.shape[1]}"
    if self.in_features == 0:
        return torch.zeros(
            input.shape[0], self.out_features, device=global_device(), requires_grad=True
        )
    return nn.functional.linear(input, self.weight, self.bias)


class GraphGrowingNetwork(torch.nn.Module):
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
    test_batch_size : int, optional
        batch size to use on the test set, by default 256
    device : str | None, optional
        default device, by default None
    exp_name : str, optional
        experiment name for logger, by default "Debug"
    with_profiler : bool, optional
        execute with profiling, by default False
    with_logger : bool, optional
        log results during execution, by default True
    """

    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 1,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        neurons: int = 20,
        test_batch_size: int = 256,
        device: str | None = None,
        exp_name: str = "Debug",
        with_profiler: bool = False,
        with_logger: bool = True,
    ) -> None:
        super(GraphGrowingNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.neurons = neurons
        self.test_batch_size = test_batch_size
        self.device = device if device else global_device()
        self.with_profiler = with_profiler
        self.global_step = 0
        self.global_epoch = 0
        self.loss_fn = nn.CrossEntropyLoss()

        self.logger = Logger(exp_name, enabled=with_logger)
        if with_logger:
            self.logger.setup_tracking()

        # sys.stdout = LogFile('temp/memory_profile_log')

        self.reset_network()

        self.hist_loss_dev = []
        self.hist_acc_dev = []

    def init_empty_graph(self) -> None:
        """Create empty DAG with start and end nodes"""
        # start_module = LinearAdditionGrowingModule(in_features=self.in_features, name="start")
        # end_module = LinearAdditionGrowingModule(in_features=self.out_features, name="end")

        start = "start"
        end = "end"
        edges = [(start, end)]
        node_attributes = {
            # TODO: change to actual modules rather than strings
            start: {
                "type": "L",  # shows what follows
                "size": self.in_features,
                # "activation": "id",
                # "module": LinearAdditionGrowingModule(in_features=self.in_features, name="start")
            },
            end: {
                "type": "L",
                "size": self.out_features,
                # "use_batch_norm": self.use_batch_norm,
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
        self.logger.clear()
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

    def log_growth_info(self, x: torch.Tensor) -> None:
        """Log important metrics at the end of each step

        Parameters
        ----------
        x : torch.Tensor
            input features batch to infer model signature
        """
        self.logger.log_metric(
            "growth train loss", self.growth_loss_train, self.global_epoch
        )
        self.logger.log_metric("growth dev loss", self.growth_loss_dev, self.global_epoch)
        self.logger.log_metric("dev loss", self.loss_dev, self.global_epoch)
        self.logger.log_metric("growth val loss", self.growth_loss_val, self.global_epoch)
        self.logger.log_metric("val loss", self.loss_val, self.global_epoch)
        self.logger.log_metric("test loss", self.loss_test, self.global_epoch)
        self.logger.log_metric(
            "growth train accuracy", self.growth_acc_train, self.global_epoch
        )
        self.logger.log_metric(
            "growth dev accuracy", self.growth_acc_dev, self.global_epoch
        )
        self.logger.log_metric("dev accuracy", self.acc_dev, self.global_epoch)
        self.logger.log_metric(
            "growth val accuracy", self.growth_acc_val, self.global_epoch
        )
        self.logger.log_metric("val accuracy", self.acc_val, self.global_epoch)
        self.logger.log_metric("test accuracy", self.acc_test, self.global_epoch)

        # model_copy = copy.deepcopy(self)
        # model_copy.to(self.device)
        # for param in model_copy.parameters():
        #     param.requires_grad = False
        # flops = FlopCountAnalysis(model_copy, x)
        # self.logger.log_metric("complexity/flops", flops.total(), self.global_epoch)
        # activations = ActivationCountAnalysis(model_copy, x)
        # self.logger.log_metric(
        #     "complexity/activations", activations.total(), self.global_epoch
        # )
        # del model_copy

        self.logger.log_metric(
            "complexity/nb of parameters",
            self.dag.count_parameters_all(),
            self.global_epoch,
        )
        # nb of parameters per edge
        for edge in self.dag.edges:
            params = self.dag.count_parameters([edge])
            self.logger.log_metric(
                f"complexity/nb of parameters at/layer {edge[0]}_{edge[1]}",
                params,
                self.global_epoch,
            )
        # in-degree and out-degree per node
        for node in self.dag.nodes:
            self.logger.log_metric(
                f"complexity/in-degree/node {node}",
                self.dag.in_degree(node),
                self.global_epoch,
            )
            self.logger.log_metric(
                f"complexity/out-degree/node {node}",
                self.dag.out_degree(node),
                self.global_epoch,
            )
            self.logger.log_metric(
                f"complexity/size/node {node}",
                self.dag.nodes[node]["size"],
                self.global_epoch,
            )

        with torch.no_grad():
            # Save model
            try:
                # TODO: save dag instead
                self.logger.log_pytorch_model(
                    self.dag, f"GrowableDAG step {self.global_step}", x
                )
            except Exception as error:
                print(f"[DAGNN Model] {error}")

        # Save growth history file
        dirname = "temp"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        try:
            with open(f"{dirname}/gh.json", "w") as f:
                json.dump(self.growth_history, f)
                f.flush()
                self.logger.log_artifact(f"{dirname}/gh.json")
        except Exception as error:
            print(f"[Growth History] {error}")

        # Save interactive graph
        try:
            graph = DAG_to_pyvis(self.dag)
            pyvis_path = f"{dirname}/graph_.html"
            graph.save_graph(pyvis_path)
            self.logger.log_artifact(pyvis_path)
        except Exception as error:
            print(f"[Interactive DAG] {error}")

    @profile_function
    def setup_train_datasets(
        self, train_dataset: Dataset, generator: torch.Generator
    ) -> tuple:
        """Split train dataset in three parts for train, development and validation

        Parameters
        ----------
        train_dataset : Dataset
            whole train dataset
        generator : torch.Generator
            random generator for dataset shuffling

        Returns
        -------
        tuple
            features and labels of train, development and validation datasets
        """
        split_length = int(len(train_dataset) / 3)
        train_dataset, dev_dataset, val_dataset = random_split(
            train_dataset, (split_length,) * 3, generator
        )

        train_loader = DataLoader(
            train_dataset, len(train_dataset), shuffle=True, generator=generator
        )
        dev_loader = DataLoader(
            dev_dataset, len(dev_dataset), shuffle=True, generator=generator
        )
        val_loader = DataLoader(
            val_dataset, len(val_dataset), shuffle=True, generator=generator
        )
        # print("Length of the train_loader:", len(train_loader))
        # print("Length of the dev_loader:", len(dev_loader))
        # print("Length of the val_loader:", len(val_loader))

        # # This way does not call the transforms!
        # X_train, Y_train = trainset.dataset.data[trainset.indices].to(self.device), trainset.dataset.targets[trainset.indices].to(self.device)
        # X_dev, Y_dev = devset.dataset.data[devset.indices].to(self.device), devset.dataset.targets[devset.indices].to(self.device)
        # X_val, Y_val = valset.dataset.data[valset.indices].to(self.device), valset.dataset.targets[valset.indices].to(self.device)

        X_train, Y_train = next(iter(train_loader))
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        X_dev, Y_dev = next(iter(dev_loader))
        X_dev, Y_dev = X_dev.to(self.device), Y_dev.to(self.device)
        X_val, Y_val = next(iter(val_loader))
        X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)

        return X_train, Y_train, X_dev, Y_dev, X_val, Y_val

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
        optimizer = torch.optim.AdamW([alpha, omega, bias], lr=1e-3, weight_decay=0)

        loss_history, _ = batch_gradient_descent(
            forward_fn=forward_fn,
            cost_fn=self.bottleneck_loss,
            target=bottleneck,
            optimizer=optimizer,
            fast=True,
            max_epochs=100,
            verbose=verbose,
            loss_name="expected bottleneck",
            title=f"[Step {self.global_step}] Adding new block",
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
        pass

    @profile_function
    # @memprofile
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

        # Enact safe forward for layers with zero in features
        nn.Linear.forward = safe_forward

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

        # ####### SANITY CHECK #######
        # # Concatenated pre-activity of parallel layers
        # existing_activity = torch.cat(list(A.values()), dim=0) # (batch_size, total_out_features)
        # # New activity of new nodes
        # new_activity = self.block_forward(alpha, omega, bias, concatenated_input_x.T, node_module.post_addition_function).T # TODO: try with extended forward, you have to set extended layers on all modules # (batch_size, total_out_features)

        # option1 = new_activity - bottleneck
        # option2 = new_activity + existing_activity - desired_update
        # assert(torch.all(option1 == option2))
        # ############################

        # TODO: Gradient descent on bottleneck
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
            prev_edge_module.scaling_factor = 1
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
                    prev_edge_module.scaling_factor = factor
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
            prev_edge_module.scaling_factor = factor
            prev_edge_module.apply_change(apply_previous=False)
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

        # # Update growth history
        # self.growth_history_step(neurons_updated=np.array([edge for edge in delta_W_star]))

        # Evaluation
        acc_train, loss_train = self.evaluate(x, y, verbose=False)
        acc_dev, loss_dev = self.evaluate(x1, y1, verbose=False)

        # TODO FUTURE : Save updates to return

        return loss_train, loss_dev, acc_train, acc_dev, loss_history

    @profile_function
    # @memprofile
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

        optimizer = torch.optim.AdamW([weight, bias], lr=1e-3, weight_decay=0)

        forward_fn = lambda: nn.functional.linear(activity, weight, bias)

        loss_history, _ = batch_gradient_descent(
            forward_fn=forward_fn,
            cost_fn=self.bottleneck_loss,
            target=bottleneck,
            optimizer=optimizer,
            fast=True,
            max_epochs=100,
            verbose=verbose,
            loss_name="expected bottleneck",
            title=f"[Step {self.global_step}] Adding direct edge ({prev_node}, {next_node})",
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

    @profile_function
    # @memprofile
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
                edge.scaling_factor = gamma_factor

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

    @profile_function
    def inter_training(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x1: torch.Tensor,
        y1: torch.Tensor,
        epochs: int = 500,
        lrate: float = 1e-3,
        return_eval: bool = False,
        verbose: bool = True,
    ) -> (
        tuple[list[float], list[float], list[float]]
        | tuple[
            list[float], list[float], list[float], list[float], list[float], list[float]
        ]
    ):
        """Perform training on the model
        Batch gradient descent with AdamW with lrate of 1e-3 and no weight decay
        Log train and validation metrics with logger

        Parameters
        ----------
        x : torch.Tensor
            train input features batch
        y : torch.Tensor
            train true labels batch
        x1 : torch.Tensor
            validation input features batch
        y1 : torch.Tensor
            validation true labels batch
        epochs : int, optional
            maximum number of epochs, by default 500
        lrate : float, optional
            learning rate, by default 1e-3
        return_eval : bool, optional
            return evaluation metrics on the validation set, by default False
        verbose : bool, optional
            print info, by default True

        Returns
        -------
        tuple[list[float], list[float], list[float]] | tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
            train loss history, train accuracy history, train f1-score history
        """

        f1score_history = []
        loss_val_history, acc_val_history, f1score_val_history = [], [], []

        def forward_fn() -> torch.Tensor:
            return self(x)

        def eval_fn() -> None:
            val_acc, val_loss, val_f1score = self.evaluate(
                x1, y1, with_f1score=True, verbose=False
            )
            _, _, dev_f1score = self.evaluate(x, y, with_f1score=True, verbose=False)
            f1score_history.append(dev_f1score)
            loss_val_history.append(val_loss)
            acc_val_history.append(val_acc)
            f1score_val_history.append(val_f1score)
            self.logger.log_metric(
                "Intermediate training/val loss",
                val_loss,
                self.global_epoch,
            )
            self.logger.log_metric(
                "Intermediate training/val accuracy",
                val_acc,
                self.global_epoch,
            )
            self.global_epoch += 1

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0)
        epochs = 500

        loss_history, acc_history = batch_gradient_descent(
            forward_fn=forward_fn,
            cost_fn=self.loss_fn,
            target=y,
            optimizer=optimizer,
            max_epochs=epochs,
            fast=False,
            eval_fn=eval_fn,
            verbose=verbose,
            loss_name="overall loss",
            title=f"[Step {self.global_step}] Intermediate training",
        )

        # Log intermediate training
        for i in range(len(loss_history)):
            self.logger.log_metric(
                "Intermediate training/dev loss",
                loss_history[i],
                self.global_epoch - epochs + i,
            )
            self.logger.log_metric(
                "Intermediate training/dev accuracy",
                acc_history[i],
                self.global_epoch - epochs + i,
            )

        if return_eval:
            return (
                loss_history,
                acc_history,
                f1score_history,
                loss_val_history,
                acc_val_history,
                f1score_val_history,
            )

        return loss_history, acc_history, f1score_history

    @profile_function
    # @memprofile
    def grow_step(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        generator: torch.Generator,
        amplitude_factor: bool = True,
        inter_train: bool = True,
        parallel: bool = True,
        verbose: bool = True,
    ) -> None:
        """Increase the size of the DAG network as one step
        Perform all possible growth actions and choose greedily the one that minimizes the validation loss
        Log important metrics

        Parameters
        ----------
        train_dataset : Dataset
            training dataset object
        test_dataset : Dataset
            test dataset object
        generator : torch.Generator
            random generator for dataset shuffling
        amplitude_factor : bool, optional
            apply amplitude factor to the new neurons, by default True
        inter_train : bool, optional
            train the network after growth, by default True
        parallel : bool, optional
            take into account parallel layers, by default True
        verbose : bool, optional
            print info, by default True
        """

        self.global_step += 1

        # Find new ways to grow the DAG
        generations = self.define_next_generations()
        if verbose:
            print(f"{generations=}")

        constant_module = False
        if self.dag.is_empty():
            # Create constant module if the graph is empty
            constant_module = True
            edge_attributes = {"type": "L", "use_bias": self.use_bias, "constant": True}
            self.dag.add_direct_edge("start", "end", edge_attributes)

        # Split train dataset into 3 parts
        X_train, Y_train, X_dev, Y_dev, X_val, Y_val = self.setup_train_datasets(
            train_dataset=train_dataset, generator=generator
        )
        # super_train_loader = DataLoader(train_dataset, self.test_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, self.test_batch_size)

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

            # Log possible optimal updates
            assert deltas is not None
            assert node_module.pre_activity is not None
            assert node_module.pre_activity.grad is not None
            for i, edge_module in enumerate(node_module.previous_modules):
                self.logger.log_metric_with_stats(
                    f"growth/possible update/edge {edge_module._name}/weight",
                    deltas[i][0],
                    self.global_step,
                )
                if edge_module.use_bias:
                    self.logger.log_metric_with_stats(
                        f"growth/possible update/edge {edge_module._name}/bias",
                        deltas[i][1],
                        self.global_step,
                    )
            del deltas
            self.logger.log_metric_with_stats(
                f"growth/desired update/node {node_module._name}",
                node_module.pre_activity.grad,
                self.global_step,
            )
            self.logger.log_metric_with_stats(
                f"growth/bottleneck/node {node_module._name}",
                bottleneck[node_module._name],
                self.global_step,
            )

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

        # bottleneck = (
        #     torch.cat((bottleneck.values()), dim=0).detach().clone()
        # )  # (batch_size, total_out_features)
        # concatenated_input_B = torch.cat(list(input_B.values()), dim=0)#.clone().detach()

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

        # We have bottleneck, activities, optimal updates
        # do we need the name for each activity? probably
        # do we save the node name?

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

            gen["loss_train"] = loss_train
            gen["loss_dev"] = loss_dev
            gen["loss_val"] = loss_val
            gen["acc_train"] = acc_train
            gen["acc_dev"] = acc_dev
            gen["acc_val"] = acc_val

            # TEMP: save DAG
            gen["dag"] = model_copy.dag
            gen["growth_history"] = model_copy.growth_history

        del model_copy

        # Find option that generates minimum loss
        self.choose_growth_best_action(
            generations, verbose=verbose
        )  # TODO: apply best option

        # Intermediate training
        # TODO with validation set?
        if inter_train:
            hist_loss_dev, hist_acc_dev, _ = self.inter_training(
                torch.cat([X_train, X_dev], dim=0),
                torch.cat([Y_train, Y_dev], dim=0),
                X_val,
                Y_val,
                verbose=verbose,
            )
            self.loss_dev = hist_loss_dev[-1]
            self.acc_dev = hist_acc_dev[-1]

            ########## TEMPORARY SOLUTION ##########
            self.hist_loss_dev.extend(hist_loss_dev)
            self.hist_acc_dev.extend(hist_acc_dev)
            #######################################

        # Evaluation
        self.acc_val, self.loss_val = self.evaluate(X_val, Y_val, verbose=False)
        self.acc_test, self.loss_test = self.evaluate_dataset(test_loader)

        # TODO: log metrics
        self.log_growth_info(X_train)

        # TODO: check that we reset all hooks and delete all updates
        # # Reset all hooks
        # for next_node_module in next_node_modules:
        #     for parallel_module in next_node_module.previous_modules:
        #         parallel_module.reset_computation()
        #         # Delete activities
        #         parallel_module.delete_update(include_previous=False)
        #     # Delete activities
        #     next_node_module.delete_update()
        # for node_module in new_node_modules:
        #     node_module.delete_update()

    def choose_growth_best_action(
        self, options: list[dict], regularization: bool = False, verbose: bool = False
    ) -> None:
        """Choose the growth action with the minimum validation loss greedily
        Log average metrics of the current growth step
        Reconstruct chosen graph and discard the rest

        Parameters
        ----------
        options : list[dict]
            dictionary with all possible graphs and their statistics
        regularization : bool, optional
            take into account the regularization term, by default False
        verbose : bool, optional
            print info, by default False
        """

        # TODO: reinit metrics
        # DF["train loss reduction"] = self.loss_train - DF["loss_train"]
        # DF["dev loss reduction"] = self.loss_dev - DF["loss_dev"]
        # DF["val loss reduction"] = self.loss_val - DF["loss_val"]
        # self.log_metric_stats(
        #     "growth actions/average train loss reduction",
        #     DF["train loss reduction"].mean(),
        # )
        # self.log_metric_stats(
        #     "growth actions/average dev loss reduction", DF["dev loss reduction"].mean()
        # )
        # self.log_metric_stats(
        #     "growth actions/average val loss reduction", DF["val loss reduction"].mean()
        # )

        # Greedy choice based on validation loss
        selection = {}
        if regularization:
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

        # for metric, value in best_option.stats.items():
        #     self.log_metric_stats(metric, value)
        # best_option.stats.clear()
        self.dag = copy.copy(best_option["dag"])
        self.growth_history = best_option["growth_history"]
        self.growth_loss_train = best_option["loss_train"]
        self.growth_loss_dev = best_option["loss_dev"]
        self.growth_loss_val = best_option["loss_val"]
        self.growth_acc_train = best_option["acc_train"]
        self.growth_acc_dev = best_option["acc_dev"]
        self.growth_acc_val = best_option["acc_val"]
        del best_option

    @profile_function
    def define_next_generations(self) -> list[dict]:
        """Find all possible growth extensions for the current graph

        Returns
        -------
        list[dict]
            list of dictionaries with growth information
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
                f1score = f1_micro(y, final_pred)
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
