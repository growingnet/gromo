import copy
import math
import string
import warnings
from collections import deque
from enum import Enum
from typing import Any, Callable, Iterator, Mapping

import networkx as nx
import torch
import torch.nn as nn

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.constant_module import ConstantModule
from gromo.modules.conv2d_growing_module import (
    Conv2dMergeGrowingModule,
    FullConv2dGrowingModule,
)
from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.tools import lecun_normal_
from gromo.utils.utils import (
    activation_fn,
    compute_BIC,
    evaluate_extended_dataset,
    f1_micro,
)


supported_layer_types = ["linear", "convolution"]


class GrowingDAG(nx.DiGraph, GrowingContainer):
    """Represents a directed acyclic graph with edges as GrowingModule and nodes as MergeGrowingModule

    Parameters
    ----------
    in_features : int
        input features
    out_features : int
        output features
    neurons : int
        number of neurons to add on each growth step
    use_bias : bool
        use bias
    use_batch_norm : bool
        use Batch Normalization
    default_layer_type : str, optional
        the type of layer operations, to choose between "linear" and "convolution", by default "linear"
    activation : str, optional
        the default activation function, by default "selu"
    name : str, optional
        name of the dag, by default ""
    root : str, optional
        name of the root node, by default "start"
    end : str, optional
        name of the end node, by default "end"
    input_shape : tuple[int, int] | None, optional
        the expected shape of the input excluding batch size and channels, by default None
    DAG_parameters : dict | None, optional
        configuration dictionary to create a custom initial dag, by default None
    device : torch.device | str | None, optional
        default device, by default None

    Raises
    ------
    ValueError
        if the reserved character "_" is used in the name of the dag
    NotImplementedError
        if the default layer type is not supported
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        neurons: int,
        use_bias: bool,
        use_batch_norm: bool,
        default_layer_type: str = "linear",
        activation: str = "selu",
        name: str = "",
        root: str = "start",
        end: str = "end",
        input_shape: tuple[int, int] | None = None,
        DAG_parameters: dict | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        nx.DiGraph.__init__(self)
        GrowingContainer.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            device=device,
        )
        self.neurons = neurons
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        if "_" in name:
            raise ValueError(
                f"The character '_' is not allowed in the name of a GrowingDAG. Found {name}."
            )
        self._name = name
        self.root = f"{root}@{name}"
        self.end = f"{end}@{name}"

        if default_layer_type not in supported_layer_types:
            raise NotImplementedError(
                f"The default layer type is not supported. Expected one of {supported_layer_types}, got {default_layer_type}"
            )
        self.layer_type = default_layer_type
        self.input_shape = input_shape

        if DAG_parameters is None:
            DAG_parameters = self.init_dag_parameters()

        edges = DAG_parameters.get("edges", [])
        edge_attributes = DAG_parameters.get("edge_attributes", {})
        node_attributes = DAG_parameters.get("node_attributes", {})
        self.ancestors = {}

        self.add_edges_from(edges)
        self.update_nodes(self.nodes, node_attributes)
        self.update_edges(edges, edge_attributes, zero_weights=False)
        self.update_connections(edges)
        self.set_growing_layers()

    # Override functions from GrowingContainer

    def set_growing_layers(self) -> None:
        """
        Reference all growable layers of the dag as all the edges and nodes
        """
        self._growing_layers = self.get_all_edge_modules() + self.get_all_node_modules()

    def init_computation(self):
        """Initialize statistics computations for all nodes"""
        for node_module in self.get_all_node_modules():
            if node_module._name == self.root:
                node_module.store_activity = True
            else:
                node_module.init_computation()

    def update_computation(self):
        """Update statistics computations for all nodes"""
        for node_module in self.get_all_node_modules():
            if node_module._name == self.root:
                continue
            # node_module.previous_tensor_s.update()
            # node_module.previous_tensor_m.update()
            node_module.update_computation()

    def reset_computation(self):
        """Reset the computation of the optimal added parameters on the whole network"""
        for edge_module in self.get_all_edge_modules():
            edge_module.reset_computation()
        for node_module in self.get_all_node_modules():
            node_module.reset_computation()

    def compute_optimal_updates(self, *args: Any, **kwargs: Any):
        """Compute optimal delta for growth procedure for all nodes"""
        self.compute_optimal_delta(*args, **kwargs)

    def compute_optimal_delta(
        self,
        update: bool = True,
        return_deltas: bool = False,
        force_pseudo_inverse: bool = False,
    ):
        """Compute optimal delta for growth procedure for all nodes

        Parameters
        ----------
        update : bool, optional
            update the optimal delta layer attribute and the first order decrease, by default True
        return_deltas: bool, optional
            placeholder argument as this function does not return anything
        force_pseudo_inverse : bool, optional
            use the pseudo-inverse to compute the optimal delta even if the
            matrix is invertible, by default False
        """
        for node_module in self.get_all_node_modules():
            if node_module._name == self.root:
                continue
            node_module.compute_optimal_delta(
                update=update,
                return_deltas=return_deltas,
                force_pseudo_inverse=force_pseudo_inverse,
            )
            assert node_module.parameter_update_decrease is not None

    def delete_update(self):
        """Delete tensor updates for all nodes"""
        for node_module in self.get_all_node_modules():
            node_module.delete_update(include_previous=True)

    # Initialize GrowingDAG and properties

    def init_dag_parameters(self) -> dict:
        """Initialize configuration parameters of the dag

        Returns
        -------
        dict
            configuration dictionary for initial dag
        """
        edges = [(self.root, self.end)]
        node_attributes = {
            self.root: {
                "type": self.layer_type,  # shows what follows
                "size": self.in_features,
                "shape": self.input_shape,
                "kernel_size": (3, 3),
            },
            self.end: {
                "type": self.layer_type,
                "size": self.out_features,
                "kernel_size": (3, 3),
                "use_batch_norm": self.use_batch_norm,
            },
        }
        edge_attributes = {
            "type": self.layer_type,
            "use_bias": self.use_bias,
            "kernel_size": (3, 3),
        }

        DAG_parameters = {}
        DAG_parameters["edges"] = edges
        DAG_parameters["node_attributes"] = node_attributes
        DAG_parameters["edge_attributes"] = edge_attributes
        return DAG_parameters

    def export_dag_parameters(self) -> dict:
        """Export detailed parameters of the dag in dictionary format

        Returns
        -------
        dict
            dictionary with parameters sizes and attributes
        """
        kernel_size = (3, 3)
        node_attributes = {
            node: {
                "type": self.layer_type,
                "size": value["size"],
                "shape": value.get("shape"),
                "kernel_size": kernel_size,
                "activation": self.activation if node != self.root else "id",
            }
            for node, value in self.nodes.items()
        }
        edge_attributes = {
            "type": self.layer_type,
            "use_bias": self.use_bias,
            "kernel_size": kernel_size,
        }
        DAG_parameters = {}
        DAG_parameters["edges"] = list(self.edges)
        # DAG_parameters["nodes"] = self.nodes
        DAG_parameters["node_attributes"] = node_attributes
        DAG_parameters["edge_attributes"] = edge_attributes
        return DAG_parameters

    @property
    def nodes(self) -> nx.reportviews.NodeView:
        """Get all nodes of dag

        Returns
        -------
        nx.reportviews.NodeView
            nodes
        """
        return super().nodes

    @property
    def edges(self) -> nx.reportviews.OutEdgeView:
        """Get all edges of dag

        Returns
        -------
        nx.reportviews.OutEdgeView
            edges
        """
        return super().edges

    @property
    def out_edges(self) -> nx.reportviews.OutEdgeView:
        """Get output edges of dag

        Returns
        -------
        nx.reportviews.OutEdgeView
            output edges
        """
        return super().out_edges

    @property
    def in_edges(self) -> nx.reportviews.InEdgeView:
        """Get input edges of dag

        Returns
        -------
        nx.reportviews.InEdgeView
            input edges
        """
        return super().in_edges

    @property
    def in_degree(self) -> nx.reportviews.InDegreeView:
        """Get in-degree of dag

        Returns
        -------
        nx.reportviews.InDegreeView
            in-degree
        """
        return super().in_degree

    @property
    def out_degree(self) -> nx.reportviews.OutDegreeView:
        """Get out-degree of dag

        Returns
        -------
        nx.reportviews.OutDegreeView
            out-degree
        """
        return super().out_degree

    # Module setters and attributes

    def __set_edge_module(
        self, prev_node: str, next_node: str, module: GrowingModule
    ) -> None:
        """Setter function for module of edge

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing node of edge
        module : GrowingModule
            growable module to set to edge
        """
        edge = str((prev_node, next_node))
        if edge in self._modules:
            del self._modules[edge]
        self[prev_node][next_node]["module"] = module
        self._modules[edge] = module

    def __set_node_module(self, node: str, module: MergeGrowingModule) -> None:
        """Setter function for module of node

        Parameters
        ----------
        node : str
            specified node name
        module : MergeGrowingModule
            growable module to set to node
        """
        if node in self._modules:
            del self._modules[node]
        self.nodes[node]["module"] = module
        self._modules[node] = module

    def toggle_edge_candidate(
        self, prev_node: str, next_node: str, candidate: bool
    ) -> None:
        """Toggle the candidate attribute of an edge

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing node of edge
        candidate : bool
            candidate value

        Raises
        ------
        ValueError
            raised if the edge does not exist
        """
        if prev_node is None or next_node is None:
            return
        if (prev_node, next_node) not in self.edges:
            raise ValueError(
                f"Edge ({prev_node}, {next_node}) is not present in the graph"
            )
        self[prev_node][next_node]["candidate"] = candidate

    def toggle_node_candidate(self, node: str, candidate: bool) -> None:
        """Toggle the candidate attribute of a node

        Parameters
        ----------
        node : str
            specified node name
        candidate : bool
            candidate value

        Raises
        ------
        ValueError
            raised if the node does not exist
        """
        if node is None:
            return
        if node not in self.nodes:
            raise ValueError(f"Node {node} is not present in the graph")
        self.nodes[node]["candidate"] = candidate

    # Module getters and attributes

    def is_edge_candidate(self, prev_node: str, next_node: str) -> bool:
        """Know if an edge is a candidate edge

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing node of edge

        Returns
        -------
        bool
            candidate attribute
        """
        if ("_" in prev_node) or ("_" in next_node):
            return True
        if (prev_node, next_node) not in self.edges:
            # default behaviour assumes only one GrowingDAG is growing at a time
            warnings.warn(
                f"Edge ({prev_node},{next_node}) does not belong in the current GrowingDAG({self._name}). All external edges are assumed to be non-candidate.",
                UserWarning,
            )
            return False
        return self[prev_node][next_node].get("candidate", False)

    def is_node_candidate(self, node: str) -> bool:
        """Know if a node is a candidate node

        Parameters
        ----------
        node : str
            specified node name

        Returns
        -------
        bool
            candidate attribute
        """
        if "_" in node:
            return True
        simple_nodes = {k.split("_")[0]: v for k, v in self.nodes.items() if node in k}
        if node not in simple_nodes:
            # default behaviour assumes only one GrowingDAG is growing at a time
            warnings.warn(
                f"Node {node} does not belong in the current GrowingDAG({self._name}). All external nodes are assumed to be non-candidate.",
                UserWarning,
            )
            return False
        return simple_nodes[node].get("candidate", False)

    def get_edge_module(self, prev_node: str, next_node: str) -> GrowingModule:
        """Getter function for module of edge

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing node of edge

        Returns
        -------
        GrowingModule
            module attached to edge
        """
        return self[prev_node][next_node]["module"]

    def get_node_module(self, node: str) -> MergeGrowingModule:
        """Getter function for module of node

        Parameters
        ----------
        node : str
            specified node name

        Returns
        -------
        MergeGrowingModule
            module attached to node
        """
        return self.nodes[node]["module"]

    def get_edge_modules(self, edges: list | set) -> list[GrowingModule]:
        """Getter function for modules attached to edges

        Parameters
        ----------
        edges : list | set
            list of edges to retrieve modules

        Returns
        -------
        list[GrowingModule]
            list of modules for each specified edge
        """
        return [self.get_edge_module(*edge) for edge in edges]

    def get_node_modules(self, nodes: list | set) -> list[MergeGrowingModule]:
        """Getter function for modules attached to nodes

        Parameters
        ----------
        nodes : list | set
            list of nodes to retrieve modules

        Returns
        -------
        list[MergeGrowingModule]
            list of modules for each specified node
        """
        return [self.get_node_module(node) for node in nodes]

    def get_all_edge_modules(self) -> list[GrowingModule]:
        """Getter function for all modules attached to edges

        Returns
        -------
        list[GrowingModule]
            list of modules for all existing edges
        """
        return self.get_edge_modules(list(self.edges))

    def get_all_node_modules(self) -> list[MergeGrowingModule]:
        """Getter function for all modules attached to nodes

        Returns
        -------
        list[MergeGrowingModule]
            list of modules for all existing nodes
        """
        return self.get_node_modules(list(self.nodes))

    def is_empty(self) -> bool:
        """Check if the dag has no connections

        Returns
        -------
        bool
            empty dag
        """
        return nx.is_empty(self)

    # Add new modules

    def add_direct_edge(
        self,
        prev_node: str,
        next_node: str,
        edge_attributes: dict = {},
        zero_weights: bool = False,
    ) -> None:
        """Add direct edge to graph, link two nodes with a new module

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing node of edge
        edge_attributes : dict, optional
            extra attributes of edge, by default {}
        zero_weights : bool, optional
            set the weights to zero, by default False
        """
        self.add_edge(prev_node, next_node)
        edges = [(prev_node, next_node)]
        self.update_edges(
            edges, edge_attributes=edge_attributes, zero_weights=zero_weights
        )
        self.update_connections(edges)
        self.set_growing_layers()

    def add_node_with_two_edges(
        self,
        prev_node: str,
        new_node: str,
        next_node: str,
        node_attributes: dict,
        edge_attributes: dict = {},
        zero_weights: bool = False,
    ) -> None:
        """Add new node to graph, create incoming and outgoing edges with new modules

        Parameters
        ----------
        prev_node : str
            incoming node for new edge
        new_node : str
            new node id
        next_node : str
            outgoing node for new edge
        node_attributes : dict
            attributes of new node
        edge_attributes : dict, optional
            extra attributes of edge, by default {}
        zero_weights : bool, optional
            set the weights to zero, by default False

        Raises
        ------
        KeyError
            if the "type" and the "size" of node is not specified in node_attributes dictionary
        """
        new_edges = [(prev_node, new_node), (new_node, next_node)]
        self.add_edges_from(new_edges)

        if "type" not in node_attributes:
            raise KeyError(
                'The type of the node should be specified at initialization. Example: key "type" in node_attributes'
            )
        if "size" not in node_attributes:
            raise KeyError(
                'The size of the node should be specified at initialization. Example: key "size" in node_attributes'
            )
        # TODO: separate functions for different modules, no need to check the type of node
        # self.nodes[new_node].update(node_attributes)
        self.update_nodes([new_node], node_attributes={new_node: node_attributes})
        self.update_edges(
            new_edges, edge_attributes=edge_attributes, zero_weights=zero_weights
        )
        self.update_connections(new_edges)
        self.set_growing_layers()

    def update_nodes(
        self, nodes: list[str] | Mapping, node_attributes: dict[str, dict]
    ) -> None:
        """Create new merge modules for nodes based on incoming and outgoing edges

        Parameters
        ----------
        nodes : list[str] | Mapping
            list of nodes to update modules
        node_attributes : dict[str, dict]
            extra attributes for nodes. Keys are node names and values are dictionaries with attributes. Keys \"type\" and \"size\" are mandatory

        Raises
        ------
        KeyError
            if the "type" and the "size" of node is not specified in node_attributes[node] dictionary
        NotImplementedError
            if the type of the node is invalid
        """
        for node in nodes:
            attributes = node_attributes.get(node, {})
            if "type" not in attributes:
                raise KeyError(
                    'The type of the node should be specified at initialization. Example: key "type" in node_attributes[new_node]'
                )
            if "size" not in attributes:
                raise KeyError(
                    'The size of the node should be specified at initialization. Example: key "size" in node_attributes[new_node]'
                )
            self.nodes[node].update(attributes)
            if attributes.get("use_batch_norm", self.use_batch_norm):
                batch_norm = nn.BatchNorm1d(
                    self.nodes[node]["size"], affine=False, device=self.device
                )
            else:
                batch_norm = nn.Identity()
            name = node.split("_")[0]
            if self.nodes[node]["type"] == "linear":
                in_features = self.nodes[node]["size"]
                self.__set_node_module(
                    node,
                    LinearMergeGrowingModule(
                        in_features=in_features,
                        post_merge_function=torch.nn.Sequential(
                            batch_norm,
                            activation_fn(self.nodes[node].get("activation")),
                        ),
                        allow_growing=True,
                        device=self.device,
                        name=f"{name}",
                    ),
                )
            elif self.nodes[node]["type"] == "convolution":
                in_channels = self.nodes[node]["size"]
                input_size = self.nodes[node].get("shape")
                kernel_size = self.nodes[node]["kernel_size"]
                input_volume = (
                    in_channels * input_size[0] * input_size[1]
                    if input_size is not None
                    else None
                )
                self.__set_node_module(
                    node,
                    Conv2dMergeGrowingModule(
                        in_channels=in_channels,
                        input_size=input_size,
                        next_kernel_size=kernel_size,
                        input_volume=input_volume,
                        post_merge_function=torch.nn.Sequential(
                            batch_norm,
                            activation_fn(self.nodes[node].get("activation")),
                        ),
                        allow_growing=True,
                        device=self.device,
                        name=f"{name}",
                    ),
                )
            else:
                raise NotImplementedError

    def update_edges(
        self,
        edges: list[tuple[str, str]],
        edge_attributes: dict = {},
        zero_weights: bool = False,
    ) -> None:
        """Create new modules for edges based on node types

        Parameters
        ----------
        edges : list[tuple[str, str]]
            list of edges to update modules
        edge_attributes : dict, optional
            extra attributes for edges, by default {}
        zero_weights : bool, optional
            set the weights to zero, by default False

        Raises
        ------
        KeyError
            if the kernel_size is not specified in edge_attributes
        NotImplementedError
            if the type of the node is invalid
        """
        for prev_node, next_node in edges:
            name = f"{prev_node.split('_')[0]}_{next_node.split('_')[0]}"

            if edge_attributes.get("constant"):
                self.__set_edge_module(
                    prev_node,
                    next_node,
                    ConstantModule(
                        in_features=self.get_node_module(prev_node).out_features,
                        out_features=self.nodes[next_node]["size"],
                        device=self.device,
                    ),
                )
                self[prev_node][next_node]["type"] = "constant"
                continue
            # If both nodes are linear
            elif (
                self.nodes[prev_node]["type"] == "linear"
                and self.nodes[next_node]["type"] == "linear"
            ):
                new_module = LinearGrowingModule(
                    in_features=self.nodes[prev_node]["size"],
                    out_features=self.nodes[next_node]["size"],
                    use_bias=edge_attributes.get("use_bias", self.use_bias),
                    device=self.device,
                    name=f"L{name}",
                )
            elif (
                self.nodes[prev_node]["type"] == "convolution"
                and self.nodes[next_node]["type"] == "convolution"
            ):
                if "kernel_size" not in edge_attributes:
                    raise KeyError(
                        'The kernel size of the edge should be specified at initialization. Example: key "kernel_size" in edge_attributes'
                    )
                kernel_size = edge_attributes["kernel_size"]
                input_size = self.get_node_module(prev_node).output_size
                default_padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
                new_module = FullConv2dGrowingModule(
                    in_channels=self.nodes[prev_node]["size"],
                    out_channels=self.nodes[next_node]["size"],
                    kernel_size=kernel_size,
                    input_size=input_size,
                    stride=edge_attributes.get("stride", 1),
                    padding=edge_attributes.get("padding", default_padding),
                    dilation=edge_attributes.get("dilation", 1),
                    use_bias=edge_attributes.get("use_bias", self.use_bias),
                    # allow_growing=True,
                    device=self.device,
                    name=f"C{name}",
                )
            elif (
                self.nodes[prev_node]["type"] == "convolution"
                and self.nodes[next_node]["type"] == "linear"
            ):
                in_features = self.nodes[prev_node]["module"].out_features
                new_module = LinearGrowingModule(
                    in_features=in_features,
                    out_features=self.nodes[next_node]["size"],
                    use_bias=edge_attributes.get("use_bias", self.use_bias),
                    device=self.device,
                    name=f"L{name}",
                )
            else:
                raise NotImplementedError

            if zero_weights:
                nn.init.zeros_(new_module.weight)
            else:
                lecun_normal_(new_module.weight)
            if new_module.use_bias:
                nn.init.zeros_(new_module.bias)

            self.__set_edge_module(
                prev_node,
                next_node,
                new_module,
            )
            self[prev_node][next_node]["type"] = self.nodes[next_node]["type"]

    def update_connections(self, edges: list) -> None:
        """Update connections to modules on specific edges and their adjacent nodes

        Parameters
        ----------
        edges : list
            list of edges to update modules
        """
        if len(edges) == 0:
            return

        for prev_node, next_node in edges:
            # prev_node, next_node = edge
            assert self.get_edge_module(prev_node, next_node)
            assert self.get_node_module(prev_node)
            assert self.get_node_module(next_node)

            self.get_edge_module(prev_node, next_node).previous_module = (
                self.get_node_module(prev_node)
            )
            self.get_edge_module(prev_node, next_node).next_module = self.get_node_module(
                next_node
            )

            self.get_node_module(prev_node).set_next_modules(
                list(self.get_edge_modules(self.out_edges(prev_node)))
            )
            self.get_node_module(next_node).set_previous_modules(
                list(self.get_edge_modules(self.in_edges(next_node)))
            )

        self._get_ancestors(self.root)

    def update_size(self) -> None:
        """Update the sizes of all the nodes and edges based on their modules"""
        super().update_size()
        for node in self.nodes():
            module = self.get_node_module(node)
            if isinstance(module, Conv2dMergeGrowingModule):
                size = module.in_channels
            elif isinstance(module, LinearMergeGrowingModule):
                size = module.in_features
            self.nodes[node].update({"size": size})
            if node == self.root:
                self.in_features = size
            elif node == self.end:
                self.out_features = size

    # Remove existing modules

    def remove_direct_edge(self, prev_node: str, next_node: str) -> None:
        """Remove direct edge from graph
        Delete module instances from the connected nodes and update their size

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing node of edge
        """
        edge = (prev_node, next_node)
        if edge in self.edges:
            edge_module = self.get_edge_module(*edge)
            edge_module.__del__()
            self.remove_edge(*edge)
            self._get_ancestors(self.root)
            self.set_growing_layers()

    def remove_node(self, node: str) -> None:
        """Remove node from dag

        Parameters
        ----------
        node : str
            node name
        """
        if node in self.nodes:
            node_module = self.get_node_module(node)
            node_module.__del__()
            super().remove_node(node)
            self._get_ancestors(self.root)
            self.set_growing_layers()

    def rename_nodes(self, mapping: dict) -> None:
        """Rename nodes in the graph.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping old node names to new node names

        Raises
        ------
        ValueError
            if the new node already exists in the graph
        """
        # nx.relabel_nodes(self, mapping, copy=True)
        for old_name, new_name in mapping.items():
            if (new_name == old_name) or (old_name not in self.nodes):
                continue
            if new_name in self.nodes:
                raise ValueError(
                    f"New node name '{new_name}' already exists in the graph."
                )
            # Move successors
            self._succ[new_name] = self._succ.pop(old_name)
            # Update predecessors of successors
            for succ in self._succ[new_name]:
                self._pred[succ][new_name] = self._pred[succ].pop(old_name)

            # Move predecessors
            self._pred[new_name] = self._pred.pop(old_name)
            # Update successors of predecessors
            for pred in self._pred[new_name]:
                self._succ[pred][new_name] = self._succ[pred].pop(old_name)

            # Move node attributes
            self._node[new_name] = self._node.pop(old_name)

        # Update ancestors
        self._get_ancestors(self.root)

    # Calculate expressivity bottleneck of GrowingDAG
    def calculate_bottleneck(
        self,
        actions: list["Expansion"],
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable = nn.CrossEntropyLoss(),
    ) -> tuple[dict, dict]:
        """Calculate expressivity bottleneck on important nodes
        Assign hooks where necessary and update tensors with a single forward-backward
        Keep track of bottleneck and post-activities

        Parameters
        ----------
        actions : list[Expansion]
            list with growth actions information
        dataloader : torch.utils.data.DataLoader
            train features and labels
        loss_fn : Callable, optional
            loss function for bottleneck calculation, by default torch.nn.CrossEntropyLoss

        Returns
        -------
        tuple[dict, dict]
            bottleneck of nodes, input of nodes
        """
        # Handle empty graph case
        constant_module = False
        if self.is_empty():
            # Create constant module if the graph is empty
            constant_module = True
            edge_attributes = {
                "type": self.layer_type,
                "use_bias": self.use_bias,
                "constant": True,
            }
            self.add_direct_edge(self.root, self.end, edge_attributes)

        # Find nodes of interest
        prev_node_modules = set()
        next_node_modules = set()
        for expansion in actions:
            prev_nodes = expansion.previous_nodes
            next_nodes = expansion.next_nodes

            prev_node_modules.update(prev_nodes)
            next_node_modules.update(next_nodes)

        prev_node_modules = self.get_node_modules(prev_node_modules)
        next_node_modules = self.get_node_modules(next_node_modules)

        # Add hooks on node modules of interest
        self.init_computation()

        pre_activities_grad = {
            node_module._name: torch.empty(0) for node_module in next_node_modules
        }
        input_B = {node: torch.empty(0) for node in self.nodes}
        bottleneck = {}

        # Forward - Backward step
        for X, Y in dataloader:
            self.zero_grad()
            pred = self(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            self.update_computation()

            # Accumulate pre-activity gradients and input tensors on cpu
            for node_module in next_node_modules:
                assert node_module.pre_activity is not None
                assert node_module.pre_activity.grad is not None
                # Save pre activiy gradients
                pre_activities_grad[node_module._name] = torch.cat(
                    (
                        pre_activities_grad[node_module._name],
                        node_module.pre_activity.grad.clone().detach().cpu(),
                    )
                )
            for node_module in self.get_all_node_modules():
                assert node_module.activity is not None
                # Save input activity of input layers
                input_B[node_module._name] = torch.cat(
                    (
                        input_B[node_module._name],
                        node_module.activity.clone().detach().cpu(),
                    )
                )

        # Compute optimal updates
        self.compute_optimal_delta()

        with torch.no_grad():
            for node_module in next_node_modules:
                # Compute expressivity bottleneck
                v_proj = pre_activities_grad[node_module._name]
                for module in node_module.previous_modules:
                    v_proj -= (
                        module.optimal_delta_layer(
                            input_B[module.previous_module._name].to(module.device)
                        )
                        .clone()
                        .detach()
                        .cpu()
                    )

                bottleneck[node_module._name] = v_proj

                if constant_module:
                    assert torch.all(
                        bottleneck[node_module._name]
                        == pre_activities_grad[node_module._name]
                    ), "Graph is empty and the bottleneck should be the same as the pre_activity gradient. Expected: {node_module.pre_activity.grad} Found: {bottleneck[node_module._name]}"

        # Reset tensors and remove hooks
        self.reset_computation()

        # Delete activities of node modules
        for node_module in self.get_all_node_modules():
            node_module.delete_update()

        if constant_module:
            # Remove constant module if needed
            self.remove_direct_edge(self.root, self.end)

        return bottleneck, input_B

    # Helper functions for managing the DAG

    def _get_ancestors(self, root: str, pre_root: int = 0) -> None:
        """Discover all eventual ancestors of nodes

        Parameters
        ----------
        root : str
            root node of graph
        pre_root : int, optional
            toy node before root, by default 0
        """
        if pre_root == 0:
            nodes_visited = {root: set(self.predecessors(root))}
            self.ancestors.setdefault(root, set()).update([root])
            q = deque()
            for edge in self.out_edges(root):
                q.append(edge)
            self.__recursiveBFS(q, nodes_visited, update=False)
        else:
            q = deque([(pre_root, root)])
            self.__recursiveBFS(q, nodes_visited={}, update=True)

    def _indirect_connection_exists(self, prev_node: str, next_node: str) -> bool:
        """Check if two nodes are connected with one-hop links

        Parameters
        ----------
        prev_node : str
            input node
        next_node : str
            output node

        Returns
        -------
        bool
            one-hop link already exists
        """
        successors = set(self.successors(prev_node))
        predecessors = set(self.predecessors(next_node))
        intermediate_nodes = successors.intersection(predecessors)
        return len(intermediate_nodes) > 0

    def _find_possible_direct_connections(
        self, direct_successors: Mapping[str, list[str]] | Mapping[str, set[str]]
    ) -> list[dict]:
        """Find all possible non-existent direct links between two nodes based on module types

        Parameters
        ----------
        direct_successors : Mapping[str, list[str]] | Mapping[str, set[str]]
            dictionary with direct successors of nodes

        Returns
        -------
        list[dict]
            list of dictionaries with all possible new direct edges and their attributes
        """
        direct_edges = []
        for prev_node, successors in direct_successors.items():
            for next_node in successors:
                # if len(list(self.predecessors(next_node))) >= 2:
                #     continue
                direct_edges.append(
                    {
                        "previous_node": prev_node,
                        "next_node": next_node,
                        "edge_attributes": {"kernel_size": (3, 3)},
                    }
                )

        return direct_edges

    def _find_possible_one_hop_connections(
        self,
        successors: Mapping[str, list[str]] | Mapping[str, set[str]],
        size: int = 0,
    ) -> list[dict]:
        """Discover all possible non-existent one-hop links between existing nodes

        Parameters
        ----------
        successors : Mapping[str, list[str]] | Mapping[str, set[str]]
            dictionary with all successors fo nodes
        size : int, optional
            size of new node to add, by default 0

        Returns
        -------
        list[dict]
            list of dictionaries with all possible new one-hop connections and their attributes
        """

        one_hop_edges = []
        new_node = f"{len(self.nodes) - 1}@{self._name}"
        for prev_node, succ in successors.items():
            for next_node in succ:
                if not self._indirect_connection_exists(prev_node, next_node):
                    # if len(list(self.predecessors(next_node))) >= 2:
                    #     continue
                    one_hop_edges.append(
                        {
                            "previous_node": prev_node,
                            "new_node": new_node,
                            "next_node": next_node,
                            "node_attributes": {
                                "type": self.nodes[prev_node]["type"],
                                "size": size,
                                "activation": self.activation,
                                "kernel_size": (3, 3),
                                "shape": self.input_shape,
                            },
                            "edge_attributes": {
                                "kernel_size": (3, 3),
                            },
                        }
                    )

        return one_hop_edges

    def find_possible_extensions(self) -> tuple[list[dict], list[dict]]:
        """Discover all possible direct and one-hop connections of the graph

        Returns
        -------
        tuple[list[dict], list[dict]]
            discovered direct connections, discovered one-hop connections
        """
        # TODO: add existing nodes growing
        nodes_set = set(self.nodes)
        possible_successors = {
            node: nodes_set.difference(self.ancestors[node]) for node in self.nodes
        }
        possible_direct_successors = {
            node: (nodes_set.difference(self.ancestors[node])).difference(
                self.successors(node)
            )
            for node in self.nodes
        }

        # Add direct edges
        direct_edges = self._find_possible_direct_connections(possible_direct_successors)

        # Add new nodes
        one_hop_edges = self._find_possible_one_hop_connections(possible_successors)

        # # Extend existing nodes
        # nodes_set.remove(self.root)
        # nodes_set.remove(self.end)
        # existing_nodes = self._find_possible_node_extensions(list(nodes_set))

        return direct_edges, one_hop_edges

    def define_next_actions(self, expand_end: bool = False) -> list["Expansion"]:
        """Find all possible growth extensions for the current graph

        Parameters
        ----------
        expand_end : bool, optional
            expand the output dimension of the last node, by default False

        Returns
        -------
        list[Expansion]
            list with growth actions information

        Raises
        ------
        NotImplementedError
            if expand_end is set to True and there are more than one next modules
        """
        # TODO: check if they allow growing
        direct_edges, one_hop_edges = self.find_possible_extensions()

        # gen_id = 0
        actions = []

        # All possible new direct edges
        for attr in direct_edges:
            previous_node = attr.get("previous_node")
            next_node = attr.get("next_node")
            edge_attributes = attr.get("edge_attributes", {})

            expansion = Expansion(
                self,
                ExpansionType.NEW_EDGE,
                previous_node=previous_node,
                next_node=next_node,
                edge_attributes=edge_attributes,
            )
            actions.append(expansion)

        # All possible one-hop connections
        for i, attr in enumerate(one_hop_edges):
            previous_node = attr.get("previous_node")
            new_node = f"{attr.get('new_node')}_{string.ascii_lowercase[i]}"
            next_node = attr.get("next_node")
            node_attributes = attr.get("node_attributes", {})
            edge_attributes = attr.get("edge_attributes", {})

            expansion = Expansion(
                self,
                ExpansionType.NEW_NODE,
                expanding_node=new_node,
                previous_node=previous_node,
                next_node=next_node,
                node_attributes=node_attributes,
                edge_attributes=edge_attributes,
            )
            actions.append(expansion)

        # All existing nodes
        for node in self.nodes:
            if (node == self.root) or (node == self.end):
                continue
            expansion = Expansion(self, ExpansionType.EXPANDED_NODE, expanding_node=node)
            actions.append(expansion)

        if expand_end:
            next_node = self.get_node_module(self.end).next_modules
            if len(next_node) > 1:
                raise NotImplementedError(
                    "Can only expand single connected inter-merge nodes"
                )
            elif len(next_node) == 1:
                expansion = InterMergeExpansion(
                    self,
                    ExpansionType.EXPANDED_NODE,
                    expanding_node=self.end,
                    adjacent_expanding_node=next_node[0]._name,
                )
                actions.append(expansion)

        return actions

    def __recursiveBFS(self, q: deque, nodes_visited: dict, update: bool) -> None:
        """Breadth First Search recursive function to find ancestors

        Parameters
        ----------
        q : deque
            queue of edges to visit
        nodes_visited : dict
            dictionary of the nodes already visited and their set of predecessors
        update : bool
            update the nodes_visited dictionary with all predecessors
        """
        if len(q) == 0:
            return

        previous_node, node = edge = q.popleft()

        self.ancestors.setdefault(node, set()).update(self.ancestors[previous_node])

        if not update:
            nodes_visited.setdefault(node, set()).update([previous_node])

        if update or (len(nodes_visited[node]) == self.in_degree(node)):
            self.ancestors[node].update([node])
            for edge in self.out_edges(node):
                q.append(edge)

        self.__recursiveBFS(q, nodes_visited, update)

    # Forward functions

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Forward function for DAG model

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        verbose : bool, optional
            print info, by default False

        Returns
        -------
        torch.Tensor
            output of model
        """
        if verbose:
            print("\nForward DAG...")
        output = {self.root: x}
        for node in nx.topological_sort(self):
            if verbose:
                print(f"{node=}")
            for previous_node in self.predecessors(node):
                module = self.get_edge_module(previous_node, node)
                if verbose:
                    print("\t-->", module.name, module)
                module_input = output[previous_node]
                activity = module(module_input)

                assert activity.shape[1] == self.nodes[node]["size"]

                if node in output:
                    output[node] = output[node].add(activity)
                else:
                    output[node] = activity
            # Pass through node
            merge_module = self.get_node_module(node)
            if verbose:
                print("\t-->", merge_module)
            output[node] = output[node] / math.sqrt(
                max(1, len(merge_module.previous_modules))
            )
            output[node] = merge_module(output[node])
        if verbose:
            print()
        return output[self.end]

    def extended_forward(
        self,
        x: torch.Tensor,
        x_ext: torch.Tensor = None,
        mask: dict = {},
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extended forward function for DAG model including extensions of the modules

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        x_ext: torch.Tensor, optional
            extension tensor, by default None
        mask : dict, optional
            extension mask for specific nodes and edges, by default {}
            example: mask["edges"] for edges and mask["nodes"] for nodes
        verbose : bool, optional
            print info, by default False

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            output of the extended model
        """
        if verbose:
            print("\nExtended Forward DAG...")
        output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {self.root: (x, x_ext)}
        for node in nx.topological_sort(self):
            # Check if node is a candidate node and is not present in the mask
            if self.is_node_candidate(node) and node not in mask.get("nodes", {}):
                continue
            if verbose:
                print(f"{node=}")
            for previous_node in self.predecessors(node):
                # Check if previous_node is a candidate node and is not present in the mask
                if self.is_node_candidate(
                    previous_node
                ) and previous_node not in mask.get("nodes", {}):
                    continue
                # Check if (previous_node, node) is a candidate edge and is not present in the mask
                if self.is_edge_candidate(previous_node, node) and (
                    previous_node,
                    node,
                ) not in mask.get("edges", {}):
                    continue
                module = self.get_edge_module(previous_node, node)
                if verbose:
                    print("\t-->", module.name, module)
                module_input = output[previous_node]
                # Perform extended_forward on the edge layer
                # if node in mask extend the output, if previous_node in mask extend the input
                activity, activity_ext = module.extended_forward(
                    *module_input,
                    use_optimal_delta=True,
                    use_extended_input=previous_node in mask.get("nodes", []),
                    use_extended_output=node in mask.get("nodes", []),
                )
                # activity_ext = (
                #     activity_ext
                #     if activity_ext is not None
                #     else torch.empty(0, x.shape[0], module.out_features, device=self.device)
                # )

                assert activity.shape[1] == self.nodes[node]["size"]

                if node in output:
                    output[node] = (
                        output[node][0].add(activity),
                        (
                            output[node][1].add(activity_ext)
                            if output[node][1] is not None
                            else activity_ext
                        ),
                    )
                else:
                    output[node] = (activity, activity_ext)
            # Pass through node
            merge_module = self.get_node_module(node)
            if verbose:
                print("\t-->", merge_module)

            output[node] = (
                output[node][0] / math.sqrt(max(1, len(merge_module.previous_modules))),
                (
                    output[node][1]
                    / math.sqrt(max(1, len(merge_module.previous_modules)))
                    if output[node][1] is not None
                    else None
                ),
            )

            output[node] = (
                merge_module(output[node][0]),
                merge_module(output[node][1]),
            )  # TODO: simplify
        if verbose:
            print()
        return output[self.end]

    # Parameters

    def parameters(self) -> Iterator:
        """Returns parameters Iterator

        Returns
        -------
        Iterator
            parameters iterator
        """
        # TODO : Temporary solution
        param = []
        for edge in self.edges:
            module = self.get_edge_module(*edge)
            param.append(module.weight)
            if module.use_bias:
                param.append(module.bias)
        return iter(param)

    def count_parameters_all(self) -> int:
        """Count the total number of parameters of the DAG model.

        Returns
        -------
        int
            parameter count
        """
        return sum(param.numel() for param in self.parameters())

    def count_parameters(self, edges: list[tuple[str, str]]) -> int:
        """Count the total number of parameters in the specified edges of the DAGNN model

        Parameters
        ----------
        edges : list[tuple[str, str]]
            list of edges to consider

        Returns
        -------
        int
            sum of number of parameters in the specified edges
        """
        return sum(
            param.numel()
            for edge in edges
            for param in self.get_edge_module(*edge).parameters()
        )

    # Evaluation functions

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        with_f1score: bool = False,
    ) -> tuple[float, float] | tuple[float, float, float]:
        """Evaluate network on batch

        Important: Assumes that the batch is already on the correct device

        Parameters
        ----------
        x : torch.Tensor
            input features tensor
        y : torch.Tensor
            true labels tensor
        loss_fn : Callable
            loss function for bottleneck calculation
        with_f1score : bool, optional
            calculate f1-score, by default False

        Returns
        -------
        tuple[float, float] | tuple[float, float, float]
            accuracy and loss, optionally f1-score
        """
        with torch.no_grad():
            pred = self(x)
            loss = loss_fn(pred, y)

        if self.out_features > 1 and y.dim() == 1:
            final_pred = pred.argmax(axis=1)
            correct = (final_pred == y).int().sum()
            accuracy = (correct / pred.shape[0]).item()
        else:
            accuracy = -1

        if with_f1score:
            if self.out_features > 1:
                f1score = f1_micro(y.cpu(), final_pred.cpu())
            else:
                f1score = -1
            return accuracy, loss.item(), f1score

        return accuracy, loss.item()

    def evaluate_extended(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        with_f1score: bool = False,
    ) -> tuple[float, float] | tuple[float, float, float]:
        """Evaluate extended network on batch

        Important: Assumes that the batch is already on the correct device

        Parameters
        ----------
        x : torch.Tensor
            input features tensor
        y : torch.Tensor
            true labels tensor
        loss_fn : Callable
            loss function for bottleneck calculation
        with_f1score : bool, optional
            calculate f1-score, by default False

        Returns
        -------
        tuple[float, float] | tuple[float, float, float]
            accuracy and loss, optionally f1-score
        """
        with torch.no_grad():
            pred, _ = self.extended_forward(x)
            loss = loss_fn(pred, y)

        if self.out_features > 1:
            final_pred = pred.argmax(axis=1)
            correct = (final_pred == y).int().sum()
            accuracy = (correct / pred.shape[0]).item()
        else:
            accuracy = -1

        if with_f1score:
            if self.out_features > 1:
                f1score = f1_micro(y.cpu(), final_pred.cpu())
            else:
                f1score = -1
            return accuracy, loss.item(), f1score

        return accuracy, loss.item()

    # String representations

    def __str__(self) -> str:
        nodes = list(self.nodes)
        edges = list(self.edges)
        lines = [f"GrowingDAG[{self._name}]("]
        lines.append(f"\tNodes ({len(nodes)}):")
        for i, n in enumerate(nodes):
            activation = list(self.nodes[n]["module"].post_merge_function)
            activation = (
                "None"
                if all(isinstance(act, torch.nn.Identity) for act in activation)
                else str(activation)
            )
            attrs = {
                "layer type": self.nodes[n]["type"],
                "hidden size": self.nodes[n]["size"],
                "activation": activation,
            }
            attr_str = ", ".join(f"{k}: {v}" for k, v in list(attrs.items()))
            lines.append(f"\t\t{n} ({attr_str if attr_str else '{}'})")

        lines.append(f"\tEdges ({len(edges)}):")
        edge_str = ", ".join(f"{u}->{v}" for u, v in edges)
        lines.append(f"\t\t{edge_str}")

        lines.append(")")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class ExpansionType(Enum):
    """Expansion types for GrowingDAG"""

    NEW_EDGE = 0
    NEW_NODE = 1
    EXPANDED_NODE = 2


class Expansion:
    """Wrapper for expansions of a GrowingDAG

    Parameters
    ----------
    dag : GrowingDAG
        enclosed GrowingDAG object that is deep-copied
    exp_type : ExpansionType
        type of expansion, can be one of [ExpansionType.NEW_EDGE, ExpansionType.NEW_NODE, ExpansionType.EXPANDED_NODE]
    growth_history : dict, optional
        expansion history of the enclosed GrowingDAG, by default {}
    expanding_node : str | None, optional
        node to be expanded, only relevant in expansion types ExpansionType.NEW_NODE and ExpansionType.EXPANDED_NODE, by default None
    previous_node : str | None, optional
        previous node for expansion, only relevant in expansion types ExpansionType.NEW_EDGE and ExpansionType.NEW_NODE, by default None
    next_node : str | None, optional
        next node for expansion, only relevant in expansion types ExpansionType.NEW_EDGE and ExpansionType.NEW_NODE, by default None
    edge_attributes : dict, optional
        attributes of new edges, by default {}
    node_attributes : dict, optional
        attributes of new nodes, by default {}

    Raises
    ------
    ValueError
        if the type is ExpansionType.NEW_EDGE and the previous_node and next_node are missing
        or the type is ExpansionType.NEW_NODE and the previous_node, next_node and new_node are missing
        or the type is ExpansionType.EXPANDED_NODE and the new_node is missing
    """

    def __init__(
        self,
        dag: GrowingDAG,
        exp_type: ExpansionType,
        growth_history: dict = {},
        expanding_node: str | None = None,
        previous_node: str | None = None,
        next_node: str | None = None,
        edge_attributes: dict = {},
        node_attributes: dict = {},
    ) -> None:
        if not isinstance(exp_type, ExpansionType):
            raise ValueError(
                f"The expansion type should be one of {['ExpansionType.'+m.name for m in ExpansionType]}. Found '{exp_type}'."
            )
        self.type = exp_type
        self.dag = dag  # reference to the original dag
        self.growth_history = copy.deepcopy(growth_history)
        self.metrics = {}

        self.expanding_node = expanding_node
        self.previous_node = previous_node
        self.next_node = next_node

        self.edge_attributes = edge_attributes
        self.node_attributes = node_attributes

        if self.type == ExpansionType.NEW_EDGE:
            if self.previous_node is None or self.next_node is None:
                raise ValueError(
                    f"When creating a new edge the previous and next nodes arguments are required. Found {previous_node=} {next_node=}."
                )
            if self.expanding_node is not None:
                self.expanding_node = None
                warnings.warn(
                    f"When creating a new edge the expanding node argument is not required. Found {expanding_node=}.",
                    UserWarning,
                )
        elif self.type == ExpansionType.NEW_NODE:
            if (
                self.expanding_node is None
                or self.previous_node is None
                or self.next_node is None
            ):
                raise ValueError(
                    f"When creating a new node the expanding, previous, and next nodes arguments are required. Found {expanding_node=} {previous_node=} {next_node=}."
                )
        elif self.type == ExpansionType.EXPANDED_NODE:
            if self.expanding_node is None:
                raise ValueError(
                    f"When expanding an existing node the expanding node argument is required. Found {expanding_node=}."
                )
            if self.previous_node is not None or self.next_node is not None:
                self.previous_node = None
                self.next_node = None
                warnings.warn(
                    f"When expanding an existing node the previous and next nodes arguments are not required. Found {previous_node=} {next_node=}.",
                    UserWarning,
                )

    @property
    def previous_nodes(self) -> list[str]:
        """Get list of previous nodes of the expansion

        Returns
        -------
        list[str]
            previous nodes
        """
        if self.type == ExpansionType.NEW_EDGE or self.type == ExpansionType.NEW_NODE:
            return [self.previous_node]  # type: ignore
        else:  # Expand existing node
            return [
                n
                for n in self.dag.predecessors(self.expanding_node)
                if not self.dag.is_node_candidate(n)
            ]

    @property
    def next_nodes(self) -> list[str]:
        """Get list of next nodes of the expansion

        Returns
        -------
        list[str]
            next nodes
        """
        if self.type == ExpansionType.NEW_EDGE or self.type == ExpansionType.NEW_NODE:
            return [self.next_node]  # type: ignore
        else:
            return [
                n
                for n in self.dag.successors(self.expanding_node)
                if not self.dag.is_node_candidate(n)
            ]

    @property
    def new_edges(self) -> list[tuple]:
        """Get list of new edges created or expanded by the expansion

        Returns
        -------
        list[tuple]
            new edges
        """
        if self.type == ExpansionType.NEW_EDGE:
            return [(self.previous_node, self.next_node)]
        elif self.type == ExpansionType.NEW_NODE:
            return [
                (self.previous_node, self.expanding_node),
                (self.expanding_node, self.next_node),
            ]
        else:
            new_edges = [
                in_edge
                for in_edge in self.dag.in_edges(self.expanding_node)
                if not self.dag.is_node_candidate(in_edge[0])
            ]
            new_edges.extend(
                [
                    out_edge
                    for out_edge in self.dag.out_edges(self.expanding_node)
                    if not self.dag.is_node_candidate(out_edge[1])
                ]
            )
            return new_edges

    @property
    def in_edges(self) -> list[GrowingModule]:
        """Get list of input edge modules create or expanded by the expansion

        Returns
        -------
        list[GrowingModule]
            new input edge modules
        """
        if self.type == ExpansionType.NEW_EDGE:
            return self.dag.get_edge_modules([(self.previous_node, self.next_node)])
        elif self.type == ExpansionType.NEW_NODE:
            return self.dag.get_edge_modules([(self.previous_node, self.expanding_node)])
        else:
            return self.dag.get_edge_modules(
                [
                    in_edge
                    for in_edge in self.dag.in_edges(self.expanding_node)
                    if not self.dag.is_node_candidate(in_edge[0])
                ]
            )

    @property
    def out_edges(self) -> list[GrowingModule]:
        """Get list of output edge modules created or expanded by the expansion

        Returns
        -------
        list[GrowingModule]
            new output edge modules
        """
        if self.type == ExpansionType.NEW_EDGE:
            return self.dag.get_edge_modules([(self.previous_node, self.next_node)])
        elif self.type == ExpansionType.NEW_NODE:
            return self.dag.get_edge_modules([(self.expanding_node, self.next_node)])
        else:
            return self.dag.get_edge_modules(
                [
                    out_edge
                    for out_edge in self.dag.out_edges(self.expanding_node)
                    if not self.dag.is_node_candidate(out_edge[1])
                ]
            )

    def expand(self) -> None:
        """Create new edge or node on the enclosed GrowingDAG"""
        if self.type == ExpansionType.NEW_EDGE:
            self.dag.add_direct_edge(self.previous_node, self.next_node, self.edge_attributes, zero_weights=True)  # type: ignore
            self.dag.toggle_edge_candidate(
                self.previous_node, self.next_node, candidate=True
            )
        elif self.type == ExpansionType.NEW_NODE:
            self.dag.add_node_with_two_edges(self.previous_node, self.expanding_node, self.next_node, self.node_attributes, self.edge_attributes, zero_weights=True)  # type: ignore
            self.dag.toggle_node_candidate(self.expanding_node, candidate=True)

    def __update_growth_history(
        self,
        current_step: int,
        neurons_added: list = [],
        neurons_updated: list = [],
        nodes_added: list = [],
    ) -> None:
        """Record recent modifications on history dictionary

        Parameters
        ----------
        current_step : int
            current growth step
        neurons_added : list, optional
            list of edges that were added or increased in dimension, by default []
        neurons_updated : list, optional
            list of edges whose weights were updated, by default []
        nodes_added : list, optional
            list of nodes that were added, by default []
        """
        if current_step not in self.growth_history:
            self.growth_history[current_step] = {}

        keep_max = lambda new_value, key: max(
            self.growth_history[current_step].get(key, 0), new_value
        )

        step_update = {}
        for edge in self.dag.edges:
            new_value = (
                2 if edge in neurons_added else 1 if edge in neurons_updated else 0
            )
            step_update[str(edge)] = keep_max(new_value, str(edge))

        for node in self.dag.nodes:
            new_value = 2 if node in nodes_added else 0
            step_update[str(node)] = keep_max(new_value, str(node))
        self.growth_history[current_step].update(step_update)

    def update_growth_history(self, current_step: int) -> None:
        """Record recent modifications on history dictionary"""
        nodes_added = [self.expanding_node]
        neurons_added = self.new_edges
        neurons_updated = list(self.dag.edges)
        self.__update_growth_history(
            current_step=current_step,
            nodes_added=nodes_added,
            neurons_added=neurons_added,
            neurons_updated=neurons_updated,
        )

    def create_mask(self) -> dict:
        """Create expansion mask for extended forward functions

        Returns
        -------
        dict
            nodes and edges to be used in extended forward
        """
        mask = {
            "nodes": [self.expanding_node],
            "edges": self.new_edges,
        }
        return mask

    def evaluate(
        self,
        model: GrowingContainer,
        train_dataloader: torch.utils.data.DataLoader,
        dev_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
    ) -> None:
        """Evaluate GrowingContainer based on GrowingDAG expansion and save metrics

        Parameters
        ----------
        model : GrowingContainer
            container to be evaluated
        train_dataloader : torch.utils.data.DataLoader
            train dataloader
        dev_dataloader : torch.utils.data.DataLoader
            development dataloader
        val_dataloader : torch.utils.data.DataLoader
            validation dataloader
        loss_fn : Callable
            loss function
        """
        mask = self.create_mask()
        acc_train, loss_train = evaluate_extended_dataset(
            model, train_dataloader, loss_fn=loss_fn, mask=mask
        )
        acc_dev, loss_dev = evaluate_extended_dataset(
            model, dev_dataloader, loss_fn=loss_fn, mask=mask
        )
        acc_val, loss_val = evaluate_extended_dataset(
            model, val_dataloader, loss_fn=loss_fn, mask=mask
        )

        self.metrics["loss_train"] = loss_train
        self.metrics["loss_dev"] = loss_dev
        self.metrics["loss_val"] = loss_val
        self.metrics["acc_train"] = acc_train
        self.metrics["acc_dev"] = acc_dev
        self.metrics["acc_val"] = acc_val
        edges = []
        for prev_node, next_node in self.dag.edges:
            if (prev_node, next_node) in self.new_edges or (
                not self.dag.is_node_candidate(prev_node)
                and not self.dag.is_node_candidate(next_node)
                and not self.dag.is_edge_candidate(prev_node, next_node)
            ):
                edges.append((prev_node, next_node))
        nb_params = self.dag.count_parameters(edges=edges)
        self.metrics["nb_params"] = nb_params
        self.metrics["BIC"] = compute_BIC(
            nb_params, loss_val, n=len(val_dataloader.dataset)
        )

    def __repr__(self) -> str:
        if self.type == ExpansionType.NEW_EDGE:
            return f"[Expansion]: New edge from {self.previous_node} to {self.next_node}"
        elif self.type == ExpansionType.NEW_NODE:
            return f"[Expansion]: New node {self.expanding_node} from {self.previous_node} to {self.next_node}"
        elif self.type == ExpansionType.EXPANDED_NODE:
            return f"[Expansion]: Expanding node {self.expanding_node}"
        return "[Expansion]: NotImplemented"


class InterMergeExpansion(Expansion):
    """Wrapper for expansions between two GrowingDAGs

    Parameters
    ----------
    dag : GrowingDAG
        enclosed GrowingDAG object that is deep-copied
    exp_type : ExpansionType
        type of expansion, can be one of [ExpansionType.NEW_EDGE, ExpansionType.NEW_NODE, ExpansionType.EXPANDED_NODE]
    growth_history : dict, optional
        expansion history of the enclosed GrowingDAG, by default {}
    expanding_node : str | None, optional
        node to be expanded, only relevant in expansion types ExpansionType.NEW_NODE and ExpansionType.EXPANDED_NODE, by default None
    previous_node : str | None, optional
        previous node for expansion, only relevant in expansion types ExpansionType.NEW_EDGE and ExpansionType.NEW_NODE, by default None
    next_node : str | None, optional
        next node for expansion, only relevant in expansion types ExpansionType.NEW_EDGE and ExpansionType.NEW_NODE, by default None
    adjacent_expanding_node: str | None, optional
        adjacent node to the expanded node belonging in different GrowingDAG, only relevant in expansion type ExpansionType.EXPANDED_NODE, by default None
    edge_attributes : dict, optional
        attributes of new edges, by default {}
    node_attributes : dict, optional
        attributes of new nodes, by default {}
    """

    def __init__(
        self,
        dag: GrowingDAG,
        exp_type: ExpansionType,
        growth_history: dict = {},
        expanding_node: str | None = None,
        previous_node: str | None = None,
        next_node: str | None = None,
        adjacent_expanding_node: str | None = None,
        edge_attributes: dict = {},
        node_attributes: dict = {},
    ) -> None:
        super().__init__(
            dag,
            exp_type,
            growth_history,
            expanding_node,
            previous_node,
            next_node,
            edge_attributes,
            node_attributes,
        )
        self.adjacent_expanding_node = adjacent_expanding_node
        self.previous_node = previous_node
        self.next_node = next_node

    @property
    def previous_nodes(self) -> list[MergeGrowingModule]:
        """Get list of previous node modules of the expansion

        Returns
        -------
        list[MergeGrowingModule]
            previous node modules
        """
        if self.type == ExpansionType.NEW_EDGE or self.type == ExpansionType.NEW_NODE:
            return [self.dag.get_node_module(self.previous_node)]
        else:
            previous_nodes = []
            for edge in self.dag.get_node_module(self.expanding_node).previous_modules:
                if isinstance(edge, GrowingModule):
                    if not self.dag.is_node_candidate(edge.previous_module._name):
                        previous_nodes.append(edge.previous_module)
                elif isinstance(edge, MergeGrowingModule):
                    for prev_edge in edge.previous_modules:
                        if (
                            not self.dag.is_node_candidate(
                                prev_edge.previous_module._name
                            )
                            or self.expanding_node == self.dag.root
                        ):  # TODO: this would not work for a different dag, assume no candidate nodes on the other one?
                            previous_nodes.append(prev_edge.previous_module)
            return previous_nodes

    @property
    def next_nodes(self) -> list[MergeGrowingModule]:
        """Get list of next node modules of the expansion

        Returns
        -------
        list[MergeGrowingModule]
            next node modules
        """
        if self.type == ExpansionType.NEW_EDGE or self.type == ExpansionType.NEW_NODE:
            return [self.dag.get_node_module(self.next_node)]
        else:
            next_nodes = []
            for edge in self.dag.get_node_module(self.expanding_node).next_modules:
                if isinstance(edge, GrowingModule):
                    if not self.dag.is_node_candidate(edge.next_module._name):
                        next_nodes.append(edge.next_module)
                elif isinstance(edge, MergeGrowingModule):
                    for next_edge in edge.next_modules:
                        if (
                            not self.dag.is_node_candidate(next_edge.next_module._name)
                            or self.expanding_node == self.dag.end
                        ):
                            next_nodes.append(next_edge.next_module)
            return next_nodes

    @property
    def new_edges(self) -> list[GrowingModule]:
        """Get list of new edge modules created or expanded by the expansion

        Returns
        -------
        list[GrowingModule]
            new edge modules
        """
        if self.type == ExpansionType.NEW_EDGE or self.type == ExpansionType.NEW_NODE:
            return self.dag.get_edge_modules(super().new_edges)
        else:
            new_edges = []
            current_node_module = self.dag.get_node_module(self.expanding_node)
            for edge in current_node_module.previous_modules:
                if isinstance(edge, GrowingModule):
                    if not self.dag.is_node_candidate(edge.previous_module._name):
                        new_edges.append(edge)
                elif isinstance(edge, MergeGrowingModule):
                    for prev_edge in edge.previous_modules:
                        if (
                            self.expanding_node == self.dag.root
                        ) or not self.dag.is_node_candidate(
                            prev_edge.previous_module._name
                        ):
                            new_edges.append(prev_edge)
            for edge in current_node_module.next_modules:
                if isinstance(edge, GrowingModule):
                    if not self.dag.is_node_candidate(edge.next_module._name):
                        new_edges.append(edge)
                elif isinstance(edge, MergeGrowingModule):
                    for next_edge in edge.next_modules:
                        if (
                            self.expanding_node == self.dag.end
                        ) or not self.dag.is_node_candidate(next_edge.next_module._name):
                            new_edges.append(next_edge)
            return new_edges

    @property
    def in_edges(self) -> list[GrowingModule]:
        """Get list of input edge modules create or expanded by the expansion

        Returns
        -------
        list[GrowingModule]
            new input edge modules
        """
        if self.type == ExpansionType.NEW_EDGE:
            return self.dag.get_edge_modules([(self.previous_node, self.next_node)])
        elif self.type == ExpansionType.NEW_NODE:
            return self.dag.get_edge_modules([(self.previous_node, self.expanding_node)])
        else:
            in_edges = []
            for edge in self.dag.get_node_module(self.expanding_node).previous_modules:
                if isinstance(edge, GrowingModule):
                    if not self.dag.is_node_candidate(edge.previous_module._name):
                        in_edges.append(edge)
                elif isinstance(edge, MergeGrowingModule):
                    for prev_edge in edge.previous_modules:
                        if (
                            self.expanding_node == self.dag.root
                        ) or not self.dag.is_node_candidate(
                            prev_edge.previous_module._name
                        ):
                            in_edges.append(prev_edge)
            return in_edges

    @property
    def out_edges(self) -> list[GrowingModule]:
        """Get list of output edge modules created or expanded by the expansion

        Returns
        -------
        list[GrowingModule]
            new output edge modules
        """
        if self.type == ExpansionType.NEW_EDGE:
            return self.dag.get_edge_modules([(self.previous_node, self.next_node)])
        elif self.type == ExpansionType.NEW_NODE:
            return self.dag.get_edge_modules([(self.expanding_node, self.next_node)])
        else:
            out_edges = []
            for edge in self.dag.get_node_module(self.expanding_node).next_modules:
                if isinstance(edge, GrowingModule):
                    if not self.dag.is_node_candidate(edge.next_module._name):
                        out_edges.append(edge)
                elif isinstance(edge, MergeGrowingModule):
                    for next_edge in edge.next_modules:
                        if (
                            self.expanding_node == self.dag.end
                        ) or not self.dag.is_node_candidate(next_edge.next_module._name):
                            out_edges.append(next_edge)
            return out_edges

    def create_mask(self) -> dict:
        """Create expansion mask for extended forward functions

        Returns
        -------
        dict
            nodes and edges to be used in extended forward
        """
        mask = {
            "nodes": [self.expanding_node, self.adjacent_expanding_node],
            "edges": [edge._name for edge in self.new_edges],
        }
        return mask
