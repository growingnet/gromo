import unittest

import torch

from gromo.graph_network.GrowableDAG import GrowableDAG
from gromo.linear_growing_module import LinearAdditionGrowingModule, LinearGrowingModule
from gromo.constant_module import ConstantModule
from gromo.utils.utils import global_device

# torch.set_default_tensor_type(torch.DoubleTensor)


class TestGrowableDAG(unittest.TestCase):
    def setUp(self) -> None:
        node_attributes = {
            "start": {
                "type": "L",
                "size": 28 * 28,
            },
            "end": {"type": "L", "size": 10},
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
            "start", "1", "end", node_attributes={"type": "L", "size": 20}
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

        self.dag.add_direct_edge(prev_node="start", next_node="end", edge_attributes={"constant":True})
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
        node_attributes["size"] = 20
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
        node_attributes[new_node]["size"] = 20
        self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)

        self.assertIsInstance(
            self.dag.get_node_module(new_node), LinearAdditionGrowingModule
        )
        assert self.dag.get_node_module(new_node)._allow_growing
        assert self.dag.get_node_module(new_node).in_features == 20
        assert len(self.dag.get_node_module(new_node).previous_modules) == 0
        assert len(self.dag.get_node_module(new_node).next_modules) == 0

    def test_update_edges(self) -> None:
        self.dag.add_edge("start", "end")
        self.dag.update_edges([("start", "end")])

        self.assertIsInstance(
            self.dag.get_edge_module("start", "end"), LinearGrowingModule
        )
        assert self.dag.get_edge_module("start", "end").in_features == 28 * 28
        assert self.dag.get_edge_module("start", "end").out_features == 10
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
            "start", "1", "end", node_attributes={"type": "L", "size": 20}
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
        
        self.dag.add_node_with_two_edges("start", "1", "end", node_attributes={"type": "L", "size":10})
        assert self.dag.ancestors["start"] == set(["start"])
        assert self.dag.ancestors["1"] == set(["start", "1"])
        assert self.dag.ancestors["end"] == set(["start", "1", "end"])


        self.dag.add_node_with_two_edges("start", "2", "1", node_attributes={"type": "L", "size":10})
        assert self.dag.ancestors["start"] == set(["start"])
        assert self.dag.ancestors["1"] == set(["start", "2", "1"])
        assert self.dag.ancestors["2"] == set(["start", "2"])
        assert self.dag.ancestors["end"] == set(["start", "2", "1", "end"])

    def test_forward(self) -> None:
        self.dag.add_direct_edge("start", 'end')
        self.dag.add_node_with_two_edges("start", "1", "end", node_attributes={"type": "L", "size":10})
        
        x = torch.rand((50, 784), device=global_device())
        x_a = self.dag.get_edge_module("start", "end")(x)
        x_b = self.dag.get_edge_module("start", "1")(x)
        x_b = self.dag.get_node_module("1")(x_b)
        x_b = self.dag.get_edge_module("1", "end")(x_b)
        out = x_a.add(x_b)
        out = self.dag.get_node_module("end")(out)

        actual_out = self.dag(x)
        assert torch.all(out == actual_out)
    
    def test_extended_forward(self) -> None:
        self.dag.add_direct_edge("start", 'end')
        self.dag.get_edge_module("start", "end").optimal_delta_layer = torch.nn.Linear(in_features=784, out_features=10, device=global_device())
        self.dag.add_node_with_two_edges("start", "1", "end", node_attributes={"type": "L", "size":10})
        self.dag.get_edge_module("start", "1").extended_output_layer = torch.nn.Linear(in_features=784, out_features=5, device=global_device())
        self.dag.get_edge_module("1", "end").extended_input_layer = torch.nn.Linear(in_features=5, out_features=10, device=global_device())

        
        x = torch.rand((50, 784), device=global_device())
        x_a = self.dag.get_edge_module("start", "end").extended_forward(x)[0]
        x_b = self.dag.get_edge_module("start", "1").extended_forward(x)
        x_b = self.dag.get_node_module("1")(x_b)
        x_b = self.dag.get_edge_module("1", "end").extended_forward(*x_b)
        out = x_a.add(x_b[0])
        out = self.dag.get_node_module("end")(out)

        actual_out = self.dag.extended_forward(x)
        assert torch.all(out == actual_out)


if __name__ == "__main__":
    unittest.main()
