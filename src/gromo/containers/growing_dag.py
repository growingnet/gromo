import copy
import warnings
from collections import deque
from typing import Callable, Iterator, Mapping

import networkx as nx
import torch
import torch.nn as nn

from gromo.containers.growing_container import GrowingContainer, safe_forward
from gromo.modules.constant_module import ConstantModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import activation_fn, f1_micro


supported_layer_types = ["linear", "convolution"]


class GrowingDAG(nx.DiGraph, GrowingContainer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        neurons: int,
        use_bias: bool,
        use_batch_norm: bool,
        default_layer_type: str = "linear",
        activation: str = "selu",
        root: str = "start",
        end: str = "end",
        DAG_parameters: dict = None,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> None:
        nx.DiGraph.__init__(self, **kwargs)
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
        self.root = root
        self.end = end
        self.flatten = nn.Flatten(start_dim=1)

        if default_layer_type not in supported_layer_types:
            raise NotImplementedError(
                f"The default layer type is not supported. Expected one of {supported_layer_types}, got {default_layer_type}"
            )
        self.layer_type = default_layer_type

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
        self.id_last_node_added = max(len(node_attributes.keys()) - 2, 0)

    def init_dag_parameters(self) -> dict:
        edges = [(self.root, self.end)]
        node_attributes = {
            self.root: {
                "type": self.layer_type,  # shows what follows
                "size": self.in_features,
            },
            self.end: {
                "type": self.layer_type,
                "size": self.out_features,
                "use_batch_norm": self.use_batch_norm,
            },
        }
        edge_attributes = {"type": self.layer_type, "use_bias": self.use_bias}

        DAG_parameters = {}
        DAG_parameters["edges"] = edges
        DAG_parameters["node_attributes"] = node_attributes
        DAG_parameters["edge_attributes"] = edge_attributes
        return DAG_parameters

    @property
    def nodes(self) -> nx.reportviews.NodeView:
        return super().nodes

    @property
    def edges(self) -> nx.reportviews.OutEdgeView:
        return super().edges

    @property
    def out_edges(self) -> nx.reportviews.OutEdgeView:
        return super().out_edges

    @property
    def in_edges(self) -> nx.reportviews.InEdgeView:
        return super().in_edges

    @property
    def in_degree(self) -> nx.reportviews.InDegreeView:
        return super().in_degree

    @property
    def out_degree(self) -> nx.reportviews.OutDegreeView:
        return super().out_degree

    def __set_edge_module(
        self, prev_node: str, next_node: str, module: LinearGrowingModule
    ) -> None:
        """Setter function for module of edge

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing module of edge
        module : LinearGrowingModule
            growable module to set to edge
        """
        self[prev_node][next_node]["module"] = module

    def __set_node_module(self, node: str, module: LinearMergeGrowingModule) -> None:
        """Setter function for module of node

        Parameters
        ----------
        node : str
            specified node name
        module : LinearMergeGrowingModule
            growable module to set to node
        """
        self.nodes[node]["module"] = module

    def get_edge_module(self, prev_node: str, next_node: str) -> LinearGrowingModule:
        """Getter function for module of edge

        Parameters
        ----------
        prev_node : str
            incoming node of edge
        next_node : str
            outgoing node of edge

        Returns
        -------
        LinearGrowingModule
            module attached to edge
        """
        return self[prev_node][next_node]["module"]

    def get_node_module(self, node: str) -> LinearMergeGrowingModule:
        """Getter function for module of node

        Parameters
        ----------
        node : str
            specified node name

        Returns
        -------
        LinearMergeGrowingModule
            module attached to node
        """
        return self.nodes[node]["module"]

    def get_edge_modules(self, edges: list | set) -> list[LinearGrowingModule]:
        """Getter function for modules attached to edges

        Parameters
        ----------
        edges : list
            list fo edges to retrieve modules

        Returns
        -------
        list[LinearGrowingModule]
            list of modules for each specified edge
        """
        return [self.get_edge_module(*edge) for edge in edges]

    def get_node_modules(self, nodes: list | set) -> list[LinearMergeGrowingModule]:
        """Getter function for modules attached to nodes

        Parameters
        ----------
        nodes : list
            list of nodes to retrieve modules

        Returns
        -------
        list[LinearMergeGrowingModule]
            list of modules for each specified node
        """
        return [self.get_node_module(node) for node in nodes]

    def get_all_edge_modules(self) -> list[LinearGrowingModule]:
        """Getter functions for all modules attached to edges

        Returns
        -------
        list[LinearGrowingModule]
            list of modules for all existing edges
        """
        return self.get_edge_modules(list(self.edges))

    def get_all_node_modules(self) -> list[LinearMergeGrowingModule]:
        """Getter function for all modules attached to nodes

        Returns
        -------
        list[LinearMergeGrowingModule]
            list of modules for all existing nodes
        """
        return self.get_node_modules(list(self.nodes))

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
        edge_attributes : _type_, optional
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
            when type of node is not specified in node_attributes dictionary
        KeyError
            when size of node is not specified in node_attributes dictionary
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
        self.id_last_node_added += 1

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
            edge_module.previous_module.next_modules.remove(edge_module)  # type: ignore
            edge_module.previous_module.update_size()  # type: ignore
            edge_module.next_module.previous_modules.remove(edge_module)  # type: ignore
            edge_module.next_module.update_size()  # type: ignore
            del edge_module
            self.remove_edge(*edge)

    def update_nodes(
        self, nodes: list | Mapping, node_attributes: dict[str, dict]
    ) -> None:
        """Create new merge modules for nodes based on incoming and outgoing edges

        Parameters
        ----------
        nodes : list[str]
            list of nodes to update modules
        node_attributes : dict[str, dict]
            extra attributes for nodes. Keys are node names and values are dictionaries with attributes. Keys \"type\" and \"size\" are mandatory

        Raises
        ------
        KeyError
            when type of node is not specified in node_attributes[node] dictionary
        KeyError
            when size of node is not specified in node_attributes[node] dictionary
        """
        for node in nodes:
            # attributes = node_attributes if len(nodes) == 1 else node_attributes[node]
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
            if self.nodes[node]["type"] == "linear":
                in_features = self.nodes[node]["size"]
                if attributes.get("use_batch_norm", self.use_batch_norm):
                    batch_norm = nn.BatchNorm1d(
                        in_features, affine=False, device=self.device
                    )
                else:
                    batch_norm = nn.Identity()
                self.__set_node_module(
                    node,
                    LinearMergeGrowingModule(
                        allow_growing=True,
                        in_features=in_features,
                        post_merge_function=torch.nn.Sequential(
                            batch_norm,
                            activation_fn(self.nodes[node].get("activation")),
                        ),
                        device=self.device,
                        name=f"{node}",
                    ),
                )

    def update_edges(
        self,
        edges: list[tuple[str, str]],
        edge_attributes: dict = {},
        zero_weights: bool = False,
    ) -> None:
        """Create new modules for edges based on node types

        Parameters
        ----------
        edges : list[tuple[str]]
            list of edges to update modules
        edge_attributes : dict, optional
            extra attributes for edges, by default {}
        zero_weights : bool, optional
            set the weights to zero, by default False
        """
        for prev_node, next_node in edges:
            if edge_attributes.get("constant"):
                self.__set_edge_module(
                    prev_node,
                    next_node,
                    ConstantModule(
                        in_features=self.nodes[prev_node]["size"],
                        out_features=self.nodes[next_node]["size"],
                        device=self.device,
                    ),
                )
                self[prev_node][next_node]["type"] = "constant"
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
                    name=f"l{prev_node}_{next_node}",
                )
                if zero_weights:
                    new_module.weight = nn.Parameter(torch.zeros_like(new_module.weight))
                    if new_module.use_bias:
                        new_module.bias = nn.Parameter(torch.zeros_like(new_module.bias))
                self.__set_edge_module(
                    prev_node,
                    next_node,
                    new_module,
                )
                self[prev_node][next_node]["type"] = "linear"
                # TODO: set bias to zeros

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
                list(
                    module for module in self.get_edge_modules(self.out_edges(prev_node))
                )
            )
            self.get_node_module(next_node).set_previous_modules(
                list(module for module in self.get_edge_modules(self.in_edges(next_node)))
            )

        self._get_ancestors(self.root)

    def reset_computation(self) -> None:
        """Reset the computation of the optimal added parameters on the whole network"""
        for edge_module in self.get_all_edge_modules():
            edge_module.reset_computation()
        for node_module in self.get_all_node_modules():
            node_module.reset_computation()

    def delete_update(self) -> None:
        """Delete extended input and output layers and optimal added parameters on the whole network"""
        for edge_module in self.get_all_edge_modules():
            edge_module.delete_update(include_output=True)
        for node_module in self.get_all_node_modules():
            node_module.delete_update()

    def is_empty(self) -> bool:
        return nx.is_empty(self)

    def calculate_bottleneck(
        self,
        actions: list["Expansion"],
        X: torch.Tensor,
        Y: torch.Tensor,
        loss_fn: Callable = nn.CrossEntropyLoss(),
    ) -> tuple[dict, dict]:
        """Calculate expressivity bottleneck on important nodes
        Assign hooks where necessary and update tensors with a single forward-backward
        Keep track of bottleneck and post-activities

        Parameters
        ----------
        actions : list[Expansion]
            list with growth actions information
        X : torch.Tensor
            train features
        Y : torch.Tensor
            train labels
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
            if not isinstance(prev_nodes, list):
                prev_nodes = [prev_nodes]
            if not isinstance(next_nodes, list):
                next_nodes = [next_nodes]

            prev_node_modules.update(prev_nodes)
            next_node_modules.update(next_nodes)

        # Add hooks on node modules of interest
        prev_node_modules = self.get_node_modules(prev_node_modules)
        next_node_modules = self.get_node_modules(next_node_modules)
        for node_module in prev_node_modules:
            node_module.store_activity = True
        for node_module in next_node_modules:
            node_module.init_computation()

        # Forward - Backward step
        pred = self(X)
        loss = loss_fn(pred, Y)
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
            node_module.compute_optimal_delta(update=True, return_deltas=False)

            # Compute expressivity bottleneck
            bottleneck[node_module._name] = (
                node_module.projected_v_goal().clone().detach()
            )  # (batch_size, out_features)

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
        for node_module in self.get_all_node_modules():
            if node_module in next_node_modules:
                for parallel_module in node_module.previous_modules:
                    parallel_module.reset_computation()
                    # DO NOT delete updates
                    # parallel_module.delete_update(include_previous=False)
            # Delete activities
            node_module.delete_update()

        if constant_module:
            # Remove constant module if needed
            self.remove_direct_edge(self.root, self.end)

        # TODO: Temporary solution
        for expansion in actions:
            expansion.dag = copy.deepcopy(self)

        return bottleneck, input_B

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
        direct_successors : dict[str, list[str]]
            dictionary with direct successors of nodes

        Returns
        -------
        list[dict]
            list of dictionaries with all possible new direct edges and their attributes
        """
        direct_edges = []
        for prev_node, successors in direct_successors.items():
            for next_node in successors:
                # TODO: create getter for types
                if (self.nodes[prev_node]["type"] == "linear") and (
                    self.nodes[next_node]["type"] == "linear"
                ):
                    direct_edges.append(
                        {"previous_node": prev_node, "next_node": next_node}
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
        successors : dict[str, list[str]]
            dictionary with all successors fo nodes
        size : int, optional
            size of new node to add, by default 0

        Returns
        -------
        list[dict]
            list of dictionaries with all possible new one-hop connections and their attributes
        """

        one_hop_edges = []
        new_node = str(self.id_last_node_added + 1)
        for prev_node, succ in successors.items():
            for next_node in succ:
                if (self.nodes[prev_node]["type"] == "linear") and (
                    self.nodes[next_node]["type"] == "linear"
                ):
                    if not self._indirect_connection_exists(prev_node, next_node):
                        one_hop_edges.append(
                            {
                                "previous_node": prev_node,
                                "new_node": new_node,
                                "next_node": next_node,
                                "node_attributes": {
                                    "type": self.layer_type,
                                    "size": size,
                                    "activation": self.activation,
                                },
                            }
                        )

        return one_hop_edges

    def find_possible_extensions(self) -> tuple[list[dict], list[dict]]:
        """Discover all possible direct and one-hop connections of the graph

        Returns
        -------
        tuple[list[dict]]
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

    def define_next_actions(self) -> list["Expansion"]:
        """Find all possible growth extensions for the current graph

        Returns
        -------
        list[Expansion]
            list with growth actions information
        """
        # TODO: check if they allow growing
        direct_edges, one_hop_edges = self.find_possible_extensions()

        # gen_id = 0
        actions = []

        # All possible new direct edges
        for attr in direct_edges:
            previous_node = attr.get("previous_node")
            next_node = attr.get("next_node")

            expansion = Expansion(
                self, "new edge", previous_node=previous_node, next_node=next_node
            )
            actions.append(expansion)

        # All possible one-hop connections
        for attr in one_hop_edges:
            previous_node = attr.get("previous_node")
            new_node = attr.get("new_node")
            next_node = attr.get("next_node")
            node_attributes = attr.get("node_attributes", {})

            expansion = Expansion(
                self,
                "new node",
                expanding_node=new_node,
                previous_node=previous_node,
                next_node=next_node,
                node_attributes=node_attributes,
            )
            actions.append(expansion)

        # All existing nodes
        for node in self.nodes:
            if (node == self.root) or (node == self.end):
                continue
            expansion = Expansion(self, "expanded node", expanding_node=node)
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
        x = self.flatten(x)
        output = {self.root: x}
        for node in nx.topological_sort(self):
            if verbose:
                print(f"{node=}")
            for previous_node in self.predecessors(node):
                module = self.get_edge_module(previous_node, node)
                if verbose:
                    print("\t-->", module.name, module)
                module_input = output[previous_node]
                activity = safe_forward(module, module_input)

                assert activity.shape[1] == self.nodes[node]["size"]

                if node in output:
                    output[node] = output[node].add(activity)
                else:
                    output[node] = activity
            # Pass through node
            merge_module = self.get_node_module(node)
            if verbose:
                print("\t-->", merge_module)
            output[node] = merge_module(output[node])
        if verbose:
            print()
        return output[self.end]

    def extended_forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Extended forward function for DAG model

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
            print("\nExtended Forward DAG...")
        x = self.flatten(x)
        output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {self.root: (x, None)}
        for node in nx.topological_sort(self):
            if verbose:
                print(f"{node=}")
            for previous_node in self.predecessors(node):
                module = self.get_edge_module(previous_node, node)
                if verbose:
                    print("\t-->", module.name, module)
                module_input = output[previous_node]
                activity, activity_ext = module.extended_forward(*module_input)
                # activity_ext = (
                #     activity_ext
                #     if activity_ext is not None
                #     else torch.zeros(x.shape[0], self.neurons, device=self.device)
                #     # else torch.empty(0, x.shape[0], module.out_features, device=self.device)
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
                merge_module(output[node][0]),
                merge_module(output[node][1]),
            )  # TODO: simplify
        if verbose:
            print()
        return output[self.end][0]

    def parameters(self) -> Iterator:
        # TODO : Temporary solution
        param = []
        for edge in self.edges:
            module = self.get_edge_module(*edge)
            param.append(module.weight)
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
        edges : list[tuple[str]]
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
            pred = self.extended_forward(x)
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


expansion_types = ["new edge", "new node", "expanded node"]


class Expansion:
    """Wrapper for expansions of a GrowingDAG

    Parameters
    ----------
    dag : GrowingDAG
        enclosed GrowingDAG object that is deep-copied
    type : str
        type of expansion, can be one of ["new edge", "new node", "expanded node"]
    growth_history : dict, optional
        expansion history of the enclosed GrowingDAG, by default {}
    expanding_node : str, optional
        node to be expanded, only relevant in expansion types "new node" and "expanded node", by default None
    previous_node : str, optional
        previous node for expansion, only relevant in expansion types "new edge" and "new node", by default None
    next_node : str, optional
        next node for expansion, only relevant in expansion types "new edge" and "new node", by default None
    edge_attributes : dict, optional
        attributes of new edges, by default {}
    node_attributes : dict, optional
        attributes of new nodes, by default {}
    """

    def __init__(
        self,
        dag: GrowingDAG,
        type: str,
        growth_history: dict = {},
        expanding_node: str = None,
        previous_node: str = None,
        next_node: str = None,
        edge_attributes: dict = {},
        node_attributes: dict = {},
    ) -> None:
        if type not in expansion_types:
            raise ValueError(
                f"The expansion type should be one of {expansion_types}. Found {type}."
            )
        self.type = type
        self.dag = copy.deepcopy(dag)
        self.growth_history = growth_history
        self.metrics = {}

        self.expanding_node = expanding_node
        self.previous_node = previous_node
        self.next_node = next_node

        self.edge_attributes = edge_attributes
        self.node_attributes = node_attributes

        if self.type == "new edge":
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
        elif self.type == "new node":
            if (
                self.expanding_node is None
                or self.previous_node is None
                or self.next_node is None
            ):
                raise ValueError(
                    f"When creating a new node the expanding, previous, and next nodes arguments are required. Found {expanding_node=} {previous_node=} {next_node=}."
                )
        elif self.type == "expanded node":
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
    def previous_nodes(self) -> list[str] | str:
        if self.type == "new edge":
            return self.previous_node  # type: ignore
        elif self.type == "new node":
            return [self.previous_node]  # type: ignore
        else:  # Expand existing node
            return [n for n in self.dag.predecessors(self.expanding_node)]

    @property
    def next_nodes(self) -> list[str] | str:
        if self.type == "new edge":
            return self.next_node  # type: ignore
        elif self.type == "new node":
            return [self.next_node]  # type: ignore
        else:
            return [n for n in self.dag.successors(self.expanding_node)]

    @property
    def new_edges(self) -> list[tuple] | tuple:
        if self.type == "new edge":
            return (self.previous_node, self.next_node)
        elif self.type == "new node":
            return [
                (self.previous_node, self.expanding_node),
                (self.expanding_node, self.next_node),
            ]
        else:
            new_edges = [in_edge for in_edge in self.dag.in_edges(self.expanding_node)]
            new_edges.extend(
                [out_edge for out_edge in self.dag.out_edges(self.expanding_node)]
            )
            return new_edges

    def expand(self) -> None:
        """Create new edge or node on the enclosed GrowingDAG"""
        if self.type == "new edge":
            self.dag.add_direct_edge(self.previous_node, self.next_node, self.edge_attributes, zero_weights=True)  # type: ignore
        elif self.type == "new node":
            self.dag.add_node_with_two_edges(self.previous_node, self.expanding_node, self.next_node, self.node_attributes, self.edge_attributes, zero_weights=True)  # type: ignore

    def update_growth_history(
        self,
        current_step: int,
        neurons_added: list = [],
        neurons_updated: list = [],
        nodes_added: list = [],
    ) -> None:
        """Record recent modifications on history dictionary

        Parameters
        ----------
        step : int
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

        # TODO: automate
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

    def __del__(self) -> None:
        if "dag" in self.__dict__:
            del self.dag
            del self.growth_history
            del self.metrics

    def __repr__(self) -> str:
        if self.type == "new edge":
            return f"[Expansion]: New edge from {self.previous_node} to {self.next_node}"
        elif self.type == "new node":
            return f"[Expansion]: New node {self.expanding_node} from {self.previous_node} to {self.next_node}"
        elif self.type == "expanded node":
            return f"[Expansion]: Expanding node {self.expanding_node}"
        return "[Expansion]: NotImplemented"
