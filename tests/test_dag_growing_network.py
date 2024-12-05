import copy
import unittest

import torch

from gromo.graph_network.dag_growing_network import GraphGrowingNetwork
from gromo.utils.utils import global_device


class TestGraphGrowingNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.in_features = 5
        self.out_features = 2
        self.batch_size = 8
        self.net = GraphGrowingNetwork(
            in_features=self.in_features,
            out_features=self.out_features,
            with_logger=False,
        )
        self.net.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes={"type": "L", "size": self.net.neurons}
        )
        self.x = torch.rand((self.batch_size, self.in_features), device=global_device())
        self.y = torch.randint(
            0, self.out_features, (self.batch_size,), device=global_device()
        )
        self.x_test = torch.rand(
            (self.batch_size, self.in_features), device=global_device()
        )
        self.y_test = torch.randint(
            0, self.out_features, (self.batch_size,), device=global_device()
        )

        self.bottleneck = {
            "end": torch.rand(
                (self.batch_size, self.out_features), device=global_device()
            ),
            "1": torch.rand((self.batch_size, self.net.neurons), device=global_device()),
        }
        self.input_B = {
            "start": torch.rand(
                (self.batch_size, self.in_features), device=global_device()
            ),
            "1": torch.rand((self.batch_size, self.net.neurons), device=global_device()),
        }

    def test_init_empty_graph(self) -> None:
        self.net.init_empty_graph()
        assert len(self.net.dag.nodes) == 2
        assert len(self.net.dag.edges) == 0
        assert "start" in self.net.dag.nodes
        assert "end" in self.net.dag.nodes
        assert self.net.dag.in_degree("start") == 0
        assert self.net.dag.out_degree("start") == 0
        assert self.net.dag.in_degree("end") == 0
        assert self.net.dag.out_degree("end") == 0
        assert self.net.dag.nodes["start"]["size"] == self.in_features
        assert self.net.dag.nodes["end"]["size"] == self.out_features
        assert self.net.dag.nodes["start"]["type"] == "L"
        assert self.net.dag.nodes["end"]["type"] == "L"
        assert self.net.dag.nodes["end"]["use_batch_norm"] == False

    def test_growth_history_step(self) -> None:
        self.net.growth_history_step(
            neurons_added=[("start", "1"), ("1", "end")],
            # neurons_updated=[("start", "end")],
        )

        for edge in self.net.dag.edges:
            assert str(edge) in self.net.growth_history[self.net.global_step]
        assert self.net.growth_history[self.net.global_step][str(("start", "1"))] == 2
        assert self.net.growth_history[self.net.global_step][str(("1", "end"))] == 2
        assert self.net.growth_history[self.net.global_step]["1"] == 0

        self.net.growth_history_step(nodes_added=["1", "2"])
        assert self.net.growth_history[self.net.global_step]["1"] == 2
        assert "2" not in self.net.growth_history[self.net.global_step]

    def test_expand_node(self) -> None:
        node = "1"
        prev_nodes = ["start"]
        next_nodes = ["end"]
        self.net.expand_node(
            node,
            prev_nodes,
            next_nodes,
            self.bottleneck,
            self.input_B,
            self.x,
            self.y,
            self.x_test,
            self.y_test,
            verbose=False,
        )

        assert self.net.dag.nodes[node]["size"] == self.net.neurons * 2
        assert self.net.dag.get_edge_module("start", node).in_features == self.in_features
        assert (
            self.net.dag.get_edge_module("start", node).out_features
            == self.net.neurons * 2
        )
        assert (
            self.net.dag.get_edge_module(node, "end").in_features == self.net.neurons * 2
        )
        assert self.net.dag.get_edge_module(node, "end").out_features == self.out_features

    def test_update_edge_weights(self) -> None:
        prev_node = "start"
        next_node = "end"
        self.net.dag.add_direct_edge(prev_node, next_node)
        next_node_module = self.net.dag.get_node_module(next_node)
        edge_module = self.net.dag.get_edge_module(prev_node, next_node)
        prev_weight = copy.deepcopy(edge_module.weight)

        self.net.update_edge_weights(
            prev_node,
            next_node,
            self.bottleneck,
            self.input_B,
            self.x,
            self.y,
            self.x_test,
            self.y_test,
            amplitude_factor=False,
            verbose=False,
        )

        assert len(self.net.dag.edges) == 3
        assert (prev_node, next_node) in self.net.dag.edges
        assert self.net.dag.nodes[prev_node]["size"] == self.in_features
        assert self.net.dag.nodes[next_node]["size"] == self.out_features
        assert self.net.dag.out_degree(prev_node) == 2
        assert self.net.dag.in_degree(next_node) == 2
        assert (
            self.net.dag.get_edge_module(prev_node, next_node).in_features
            == self.in_features
        )
        assert (
            self.net.dag.get_edge_module(prev_node, next_node).out_features
            == self.out_features
        )

        # activity = torch.matmul(self.x, edge_module.weight.T) + edge_module.bias
        assert torch.all(edge_module.weight != prev_weight)

    def test_grow_step(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
