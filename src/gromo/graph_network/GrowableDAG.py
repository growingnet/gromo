import copy
from collections import deque
from typing import Iterator, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn


try:
    from constant_module import ConstantModule  # type: ignore
    from linear_growing_module import (  # type: ignore
        LinearAdditionGrowingModule,
        LinearGrowingModule,
    )
    from pyvis.network import Network as Pyvisnet
    from utils.profiling import profile_function  # type: ignore
    from utils.utils import DAG_to_pyvis, activation_fn, global_device  # type: ignore
except ImportError:
    from pyvis.network import Network as Pyvisnet

    from gromo.constant_module import ConstantModule
    from gromo.linear_growing_module import (
        LinearAdditionGrowingModule,
        LinearGrowingModule,
    )
    from gromo.utils.profiling import profile_function
    from gromo.utils.utils import DAG_to_pyvis, activation_fn, global_device


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
        input.shape[-1] == self.in_features
    ), f"Input shape {input.shape} must match the input feature size. Expected: {self.in_features}, Found: {input.shape[1]}"
    if self.in_features == 0:
        return torch.zeros(
            input.shape[0], self.out_features, device=global_device(), requires_grad=True
        )
    return nn.functional.linear(input, self.weight, self.bias)


class GrowableDAG(nx.DiGraph, nn.Module):
    def __init__(
        self, DAG_parameters: dict = {}, device: str | None = None, **kwargs
    ) -> None:
        # super(GrowableDAG, self).__init__(**kwargs)
        nx.DiGraph.__init__(self, **kwargs)
        nn.Module.__init__(self, **kwargs)
        self.flatten = nn.Flatten(start_dim=1)
        self.device: torch.device = torch.device(device) if device else global_device()

        edges = DAG_parameters.get("edges", [])
        edge_attributes = DAG_parameters.get("edge_attributes", {})
        node_attributes = DAG_parameters.get("node_attributes", {})
        self.root = DAG_parameters.get("start", "start")
        self.end = DAG_parameters.get("end", "end")
        self.ancestors = {}

        self.add_edges_from(edges)
        # nx.set_node_attributes(self, node_attributes)
        self.update_nodes(self.nodes, node_attributes)
        self.update_edges(edges, edge_attributes)
        self.update_connections(edges)
        self.id_last_node_added = np.max(len(node_attributes.keys()) - 2, 0)

        # Enact safe forward for layers with zero in_features, reverted see PR #70
        # nn.Linear.forward = safe_forward

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

    def __set_node_module(self, node: str, module: LinearAdditionGrowingModule) -> None:
        """Setter function for module of node

        Parameters
        ----------
        node : str
            specified node name
        module : LinearAdditionGrowingModule
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

    def get_node_module(self, node: str) -> LinearAdditionGrowingModule:
        """Getter function for module of node

        Parameters
        ----------
        node : str
            specified node name

        Returns
        -------
        LinearAdditionGrowingModule
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

    def get_node_modules(self, nodes: list | set) -> list[LinearAdditionGrowingModule]:
        """Getter function for modules attached to nodes

        Parameters
        ----------
        nodes : list
            list of nodes to retrieve modules

        Returns
        -------
        list[LinearAdditionGrowingModule]
            list of modules for each specified node
        """
        return [self.get_node_module(node) for node in nodes]

    def add_direct_edge(
        self, prev_node: str, next_node: str, edge_attributes: dict = {}
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
        """
        self.add_edge(prev_node, next_node)
        edges = [(prev_node, next_node)]
        self.update_edges(edges, edge_attributes=edge_attributes)
        self.update_connections(edges)

    def add_node_with_two_edges(
        self,
        prev_node: str,
        new_node: str,
        next_node: str,
        node_attributes: dict,
        edge_attributes: dict = {},
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
        self.update_edges(new_edges, edge_attributes=edge_attributes)
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
        """Create new addition modules for nodes based on incoming and outgoing edges

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
            if self.nodes[node]["type"] == "L":
                in_features = self.nodes[node]["size"]
                if attributes.get("use_batch_norm", False):
                    batch_norm = nn.BatchNorm1d(
                        in_features, affine=False, device=self.device
                    )
                else:
                    batch_norm = nn.Identity()
                self.__set_node_module(
                    node,
                    LinearAdditionGrowingModule(
                        allow_growing=True,
                        in_features=in_features,
                        post_addition_function=torch.nn.Sequential(
                            batch_norm,
                            activation_fn(self.nodes[node].get("activation")),
                        ),
                        device=self.device,
                        name=f"{node}",
                    ),
                )

    def update_edges(
        self, edges: list[tuple[str, str]], edge_attributes: dict = {}
    ) -> None:
        """Create new modules for edges based on node types

        Parameters
        ----------
        edges : list[tuple[str]]
            list of edges to update modules
        edge_attributes : dict, optional
            extra attributes for edges, by default {}
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
                self.nodes[prev_node]["type"] == "L"
                and self.nodes[next_node]["type"] == "L"
            ):
                self.__set_edge_module(
                    prev_node,
                    next_node,
                    LinearGrowingModule(
                        in_features=self.nodes[prev_node]["size"],
                        out_features=self.nodes[next_node]["size"],
                        use_bias=edge_attributes.get("use_bias", True),
                        device=self.device,
                        name=f"l{prev_node}_{next_node}",
                    ),
                )
                self[prev_node][next_node]["type"] = "L"
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

    def is_empty(self) -> bool:
        return nx.is_empty(self)

    @profile_function
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

    @profile_function
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

    @profile_function
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
                if (self.nodes[prev_node]["type"] == "L") and (
                    self.nodes[next_node]["type"] == "L"
                ):
                    direct_edges.append(
                        {"previous_node": prev_node, "next_node": next_node}
                    )

        return direct_edges

    @profile_function
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
                if (self.nodes[prev_node]["type"] == "L") and (
                    self.nodes[next_node]["type"] == "L"
                ):
                    if not self._indirect_connection_exists(prev_node, next_node):
                        one_hop_edges.append(
                            {
                                "previous_node": prev_node,
                                "new_node": new_node,
                                "next_node": next_node,
                                "node_attributes": {
                                    "type": "L",
                                    "size": size,
                                    "activation": "selu",
                                },
                            }
                        )

        return one_hop_edges

    @profile_function
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

    @profile_function
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

    def draw(self) -> None:
        """
        Draw graph on plot
        """
        plt.figure()
        G = copy.deepcopy(self)

        try:
            pos = nx.planar_layout(G)
        except:
            pos = nx.shell_layout(G)

        for edge in G.edges:
            if G.edges[edge]["type"] == "C":
                G.edges[edge]["style"] = "dashed"
            if G.edges[edge]["type"] == "L":
                G.edges[edge]["style"] = "solid"

            nx.draw_networkx_edges(G, pos, [edge], style=G.edges[edge]["style"])

        for node in G.nodes:
            if G.nodes[node]["type"] == "L":
                G.nodes[node]["color"] = "#42cafd"
                G.nodes[node]["shape"] = "o"
            elif G.nodes[node]["type"] == "C":
                G.nodes[node]["color"] = "#edd83d"
                G.nodes[node]["shape"] = "s"

            if node == self.root:
                G.nodes[node]["color"] = "#09bc8a"
            elif node == self.end:
                G.nodes[node]["color"] = "#f55536"

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_shape=G.nodes[node]["shape"],
                node_color=G.nodes[node]["color"],
            )

        # nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        plt.show()

    def animate(self) -> Pyvisnet:
        """Create animated version of DAG with pyvis Network

        Returns
        -------
        Pyvisnet
            pyvis network
        """

        return DAG_to_pyvis(self)

    @profile_function
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
            addition_module = self.get_node_module(node)
            if verbose:
                print("\t-->", addition_module)
            output[node] = addition_module(output[node])
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
        output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            self.root: (x, torch.empty(x.shape[0], 0))
        }
        for node in nx.topological_sort(self):
            if verbose:
                print(f"{node=}")
            for previous_node in self.predecessors(node):
                module = self.get_edge_module(previous_node, node)
                if verbose:
                    print("\t-->", module.name, module)
                module_input = output[previous_node]
                activity, activity_ext = module.extended_forward(*module_input)
                activity_ext = (
                    activity_ext
                    if activity_ext is not None
                    else torch.empty(x.shape[0], module.out_features, device=self.device)
                )

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
            addition_module = self.get_node_module(node)
            if verbose:
                print("\t-->", addition_module)
            output[node] = (
                addition_module(output[node][0]),
                addition_module(output[node][1]),
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
        # for node in self.nodes:
        #     param.append(self.get_node_module(node).post_addition_function)
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
