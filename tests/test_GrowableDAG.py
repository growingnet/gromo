import unittest

import torch

from gromo.constant_module import ConstantModule
from gromo.graph_network.GrowableDAG import GrowableDAG
from gromo.linear_growing_module import LinearAdditionGrowingModule, LinearGrowingModule
from gromo.utils.utils import global_device


# torch.set_default_tensor_type(torch.DoubleTensor)


class TestGrowableDAG(unittest.TestCase):
    def setUp(self) -> None:
        self.in_features = 10
        self.hidden_size = 5
        self.out_features = 2
        self.single_node_attributes = {"type": "L", "size": self.hidden_size}
        self.default_node_attributes = {"type": "L", "size": 0, "activation": "selu"}
        node_attributes = {
            "start": {
                "type": "L",
                "size": self.in_features,
            },
            "end": {"type": "L", "size": self.out_features},
        }
        DAG_parameters = {}
        DAG_parameters["edges"] = [("start", "end")]
        DAG_parameters["node_attributes"] = node_attributes
        DAG_parameters["edge_attributes"] = {"type": "L", "use_bias": True}
        self.dag = GrowableDAG(DAG_parameters)
        self.dag.remove_edge("start", "end")

    def tearDown(self) -> None:
        del self.dag
        return super().tearDown()

    def test_init(self) -> None:
        assert list(self.dag.nodes) == ["start", "end"]
        assert len(self.dag.edges) == 0
        assert self.dag.in_degree("start") == 0
        assert self.dag.out_degree("end") == 0
        assert self.dag.id_last_node_added == 0

    def test_get_edge_module(self) -> None:
        self.dag.add_direct_edge("start", "end")
        assert (
            self.dag.get_edge_module("start", "end") == self.dag["start"]["end"]["module"]
        )

    def test_get_node_module(self) -> None:
        assert self.dag.get_node_module("start") == self.dag.nodes["start"]["module"]

    def test_get_edge_modules(self) -> None:
        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        edges = [("start", "1"), ("1", "end")]
        assert self.dag.get_edge_modules(edges) == [
            self.dag.get_edge_module(*edges[0]),
            self.dag.get_edge_module(*edges[1]),
        ]

    def test_get_node_modules(self) -> None:
        assert self.dag.get_node_modules(["start", "end"]) == [
            self.dag.get_node_module("start"),
            self.dag.get_node_module("end"),
        ]

    def test_add_direct_edge(self) -> None:
        self.dag.add_direct_edge(prev_node="start", next_node="end")
        assert list(self.dag.edges) == [("start", "end")]
        assert self.dag.out_degree("start") == 1
        assert self.dag.in_degree("end") == 1

        assert isinstance(self.dag.get_edge_module("start", "end"), LinearGrowingModule)
        assert self.dag.get_node_module("start").next_modules
        assert self.dag.get_edge_module("start", "end").previous_module
        assert self.dag.get_edge_module("start", "end").next_module
        assert self.dag.get_node_module("end").previous_modules

        self.dag.add_direct_edge(
            prev_node="start", next_node="end", edge_attributes={"constant": True}
        )
        assert isinstance(self.dag.get_edge_module("start", "end"), ConstantModule)

    def test_add_node_with_two_edges(self) -> None:
        assert len(self.dag.nodes) == 2
        assert self.dag.out_degree("start") == 0
        assert self.dag.in_degree("end") == 0

        params = ["start", "1", "end"]
        node_attributes = {}
        with self.assertRaises(KeyError):
            self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)
        node_attributes["type"] = "L"
        with self.assertRaises(KeyError):
            self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)
        node_attributes["size"] = self.hidden_size
        self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)

        assert len(self.dag.nodes) == 3
        assert self.dag.out_degree("start") == 1
        assert self.dag.in_degree("end") == 1

        assert self.dag.get_node_module("start").next_modules
        assert self.dag.get_edge_module("start", "1").previous_module
        assert self.dag.get_edge_module("start", "1").next_module
        assert self.dag.get_node_module("1").previous_modules
        assert self.dag.get_node_module("1").next_modules
        assert self.dag.get_edge_module("1", "end").previous_module
        assert self.dag.get_edge_module("1", "end").next_module
        assert self.dag.get_node_module("end").previous_modules

    def test_remove_direct_edge(self) -> None:
        self.dag.add_direct_edge(prev_node="start", next_node="end")
        self.dag.remove_direct_edge(prev_node="start", next_node="end")
        assert len(list(self.dag.edges)) == 0
        assert self.dag.out_degree("start") == 0
        assert self.dag.in_degree("end") == 0

        self.dag.remove_direct_edge(prev_node="start", next_node="end")
        assert len(list(self.dag.edges)) == 0
        assert self.dag.out_degree("start") == 0
        assert self.dag.in_degree("end") == 0

    def test_update_nodes(self) -> None:
        new_node = "new"
        edges = [("start", new_node), (new_node, "end")]
        self.dag.add_edges_from(edges)

        node_attributes = {new_node: {}}
        with self.assertRaises(KeyError):
            self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)
        node_attributes[new_node]["type"] = "L"
        with self.assertRaises(KeyError):
            self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)
        node_attributes[new_node]["size"] = self.hidden_size
        self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)

        self.assertIsInstance(
            self.dag.get_node_module(new_node), LinearAdditionGrowingModule
        )
        self.assertIsInstance(
            self.dag.get_node_module(new_node).post_addition_function[0],
            torch.nn.Identity,
        )
        assert self.dag.get_node_module(new_node)._allow_growing
        assert self.dag.get_node_module(new_node).in_features == self.hidden_size
        assert len(self.dag.get_node_module(new_node).previous_modules) == 0
        assert len(self.dag.get_node_module(new_node).next_modules) == 0

        node_attributes[new_node]["use_batch_norm"] = True
        self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)
        self.assertIsInstance(
            self.dag.get_node_module(new_node).post_addition_function[0],
            torch.nn.BatchNorm1d,
        )
        assert (
            self.dag.get_node_module(new_node).post_addition_function[0].num_features
            == self.hidden_size
        )

    def test_update_edges(self) -> None:
        self.dag.add_edge("start", "end")
        self.dag.update_edges([("start", "end")])

        self.assertIsInstance(
            self.dag.get_edge_module("start", "end"), LinearGrowingModule
        )
        assert self.dag.get_edge_module("start", "end").in_features == self.in_features
        assert self.dag.get_edge_module("start", "end").out_features == self.out_features
        assert isinstance(
            self.dag.get_edge_module("start", "end").post_layer_function,
            torch.nn.Identity,
        )
        assert self.dag.get_edge_module("start", "end").previous_module is None
        assert self.dag.get_edge_module("start", "end").next_module is None

    def test_update_connections(self) -> None:
        self.dag.update_connections([])
        assert self.dag.is_empty()

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )

        assert self.dag.get_node_module("start").previous_modules == []
        assert self.dag.get_node_module("start").next_modules == [
            self.dag.get_edge_module("start", "1")
        ]

        assert self.dag.get_edge_module(
            "start", "1"
        ).previous_module == self.dag.get_node_module("start")
        assert self.dag.get_edge_module(
            "start", "1"
        ).next_module == self.dag.get_node_module("1")

        assert self.dag.get_node_module("1").previous_modules == [
            self.dag.get_edge_module("start", "1")
        ]
        assert self.dag.get_node_module("1").next_modules == [
            self.dag.get_edge_module("1", "end")
        ]

        assert self.dag.get_edge_module(
            "1", "end"
        ).previous_module == self.dag.get_node_module("1")
        assert self.dag.get_edge_module(
            "1", "end"
        ).next_module == self.dag.get_node_module("end")

        assert self.dag.get_node_module("end").previous_modules == [
            self.dag.get_edge_module("1", "end")
        ]
        assert self.dag.get_node_module("end").next_modules == []

    def test_is_empty(self) -> None:
        assert self.dag.is_empty()

        self.dag.add_edge("start", "end")
        assert not self.dag.is_empty()

    def test_get_ancestors(self) -> None:
        self.dag.add_direct_edge("start", "end")
        assert self.dag.ancestors["start"] == set(["start"])
        assert self.dag.ancestors["end"] == set(["start", "end"])

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        assert self.dag.ancestors["start"] == set(["start"])
        assert self.dag.ancestors["1"] == set(["start", "1"])
        assert self.dag.ancestors["end"] == set(["start", "1", "end"])

        self.dag.add_node_with_two_edges(
            "start", "2", "1", node_attributes=self.single_node_attributes
        )
        assert self.dag.ancestors["start"] == set(["start"])
        assert self.dag.ancestors["1"] == set(["start", "2", "1"])
        assert self.dag.ancestors["2"] == set(["start", "2"])
        assert self.dag.ancestors["end"] == set(["start", "2", "1", "end"])

    def test_indirect_connection_exists(self) -> None:
        assert self.dag._indirect_connection_exists("start", "end") is False

        node_attributes = self.single_node_attributes
        self.dag.add_node_with_two_edges("start", "1", "end", node_attributes)
        self.dag.add_node_with_two_edges("start", "2", "1", node_attributes)
        assert self.dag._indirect_connection_exists("start", "end") is True
        assert self.dag._indirect_connection_exists("start", "1") is True
        assert self.dag._indirect_connection_exists("1", "start") is False
        assert self.dag._indirect_connection_exists("2", "1") is False
        assert self.dag._indirect_connection_exists("2", "end") is True
        assert self.dag._indirect_connection_exists("start", "2") is False

    def test_find_possible_direct_connections(self) -> None:
        nodes_set = set(self.dag.nodes)
        possible_direct_successors = {
            node: (nodes_set.difference(self.dag.ancestors[node])).difference(
                self.dag.successors(node)
            )
            for node in self.dag.nodes
        }
        assert self.dag._find_possible_direct_connections(possible_direct_successors) == [
            {"previous_node": "start", "next_node": "end"}
        ]

        self.dag.add_direct_edge("start", "end")
        self.dag.add_node_with_two_edges("start", "1", "end", self.single_node_attributes)
        nodes_set = set(self.dag.nodes)
        possible_direct_successors = {
            node: (nodes_set.difference(self.dag.ancestors[node])).difference(
                self.dag.successors(node)
            )
            for node in self.dag.nodes
        }
        direct_connections = self.dag._find_possible_direct_connections(
            possible_direct_successors
        )
        assert direct_connections == []

        self.dag.add_node_with_two_edges(
            "start", "2", "1", node_attributes=self.single_node_attributes
        )
        nodes_set = set(self.dag.nodes)
        possible_direct_successors = {
            node: (nodes_set.difference(self.dag.ancestors[node])).difference(
                self.dag.successors(node)
            )
            for node in self.dag.nodes
        }
        direct_connections = self.dag._find_possible_direct_connections(
            possible_direct_successors
        )
        assert direct_connections == [{"previous_node": "2", "next_node": "end"}]

    def test_find_possible_one_hop_connections(self) -> None:
        nodes_set = set(self.dag.nodes)
        possible_successors = {
            node: nodes_set.difference(self.dag.ancestors[node])
            for node in self.dag.nodes
        }
        one_hop_edges = self.dag._find_possible_one_hop_connections(possible_successors)
        assert one_hop_edges == [
            {
                "previous_node": "start",
                "new_node": "1",
                "next_node": "end",
                "node_attributes": self.default_node_attributes,
            }
        ]

        self.dag.add_node_with_two_edges("start", "1", "end", self.single_node_attributes)
        nodes_set = set(self.dag.nodes)
        possible_successors = {
            node: nodes_set.difference(self.dag.ancestors[node])
            for node in self.dag.nodes
        }
        one_hop_edges = self.dag._find_possible_one_hop_connections(possible_successors)
        assert one_hop_edges == [
            {
                "previous_node": "start",
                "new_node": "2",
                "next_node": "1",
                "node_attributes": self.default_node_attributes,
            },
            {
                "previous_node": "1",
                "new_node": "2",
                "next_node": "end",
                "node_attributes": self.default_node_attributes,
            },
        ]

    def test_find_possible_extension(self) -> None:
        direct_edges, one_hop_edges = self.dag.find_possible_extensions()
        assert direct_edges == [{"previous_node": "start", "next_node": "end"}]
        assert one_hop_edges == [
            {
                "previous_node": "start",
                "new_node": "1",
                "next_node": "end",
                "node_attributes": self.default_node_attributes,
            }
        ]

        self.dag.add_node_with_two_edges(
            "start", "hidden", "end", self.single_node_attributes
        )
        direct_edges, one_hop_edges = self.dag.find_possible_extensions()
        assert direct_edges == [{"previous_node": "start", "next_node": "end"}]
        assert one_hop_edges == [
            {
                "previous_node": "start",
                "new_node": "2",
                "next_node": "hidden",
                "node_attributes": self.default_node_attributes,
            },
            {
                "previous_node": "hidden",
                "new_node": "2",
                "next_node": "end",
                "node_attributes": self.default_node_attributes,
            },
        ]

    def test_forward(self) -> None:
        self.dag.add_direct_edge("start", "end")
        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )

        x = torch.rand((50, self.in_features), device=global_device())
        x_a = self.dag.get_edge_module("start", "end")(x)
        x_b = self.dag.get_edge_module("start", "1")(x)
        x_b = self.dag.get_node_module("1")(x_b)
        x_b = self.dag.get_edge_module("1", "end")(x_b)
        out = x_a.add(x_b)
        out = self.dag.get_node_module("end")(out)

        actual_out = self.dag(x)
        assert torch.all(out == actual_out)

    def test_extended_forward(self) -> None:
        self.dag.add_direct_edge("start", "end")
        self.dag.get_edge_module("start", "end").optimal_delta_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            device=global_device(),
        )
        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        self.dag.get_edge_module("start", "1").extended_output_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            device=global_device(),
        )
        self.dag.get_edge_module("1", "end").extended_input_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.out_features,
            device=global_device(),
        )

        x = torch.rand((50, self.in_features), device=global_device())
        x_a = self.dag.get_edge_module("start", "end").extended_forward(x)[0]
        x_b = self.dag.get_edge_module("start", "1").extended_forward(x)
        x_b = self.dag.get_node_module("1")(x_b)
        x_b = self.dag.get_edge_module("1", "end").extended_forward(*x_b)
        out = x_a.add(x_b[0])
        out = self.dag.get_node_module("end")(out)

        actual_out = self.dag.extended_forward(x)
        assert torch.all(out == actual_out)

    def test_safe_forward(self) -> None:
        in_features = 0
        out_features = 2
        batch_size = 5
        linear = torch.nn.Linear(in_features, out_features, device=global_device())
        x = torch.rand((batch_size, in_features), device=global_device())
        assert torch.all(
            linear(x) == torch.zeros((batch_size, out_features), device=global_device())
        )

        in_features = 3
        linear = torch.nn.Linear(in_features, out_features, device=global_device())
        x = torch.rand((batch_size, in_features), device=global_device())
        assert torch.all(
            linear(x) == torch.nn.functional.linear(x, linear.weight, linear.bias)
        )

    def test_parameters(self) -> None:
        assert len(list(self.dag.parameters())) == len(self.dag.edges) * 2

        self.dag.add_direct_edge("start", "end")
        assert len(list(self.dag.parameters())) == len(self.dag.edges) * 2

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        assert len(list(self.dag.parameters())) == len(self.dag.edges) * 2

    def test_count_parameters_all(self) -> None:
        assert self.dag.count_parameters_all() == 0

        self.dag.add_direct_edge("start", "end")
        numel = self.in_features * self.out_features + self.out_features
        assert self.dag.count_parameters_all() == numel

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        numel += self.in_features * self.hidden_size + self.hidden_size
        numel += self.hidden_size * self.out_features + self.out_features
        assert self.dag.count_parameters_all() == numel

    def test_count_parameters(self) -> None:
        self.dag.add_direct_edge("start", "end")
        numel = self.in_features * self.out_features + self.out_features
        assert self.dag.count_parameters([("start", "end")]) == numel

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        numel = self.in_features * self.hidden_size + self.hidden_size
        assert self.dag.count_parameters([("start", "1")]) == numel
        numel += self.hidden_size * self.out_features + self.out_features
        assert self.dag.count_parameters([("start", "1"), ("1", "end")]) == numel


if __name__ == "__main__":
    unittest.main()
