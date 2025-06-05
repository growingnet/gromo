import unittest

import torch

from gromo.containers.growing_dag import Expansion, GrowingDAG
from gromo.modules.constant_module import ConstantModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import global_device


# torch.set_default_tensor_type(torch.DoubleTensor)


class TestGrowingDAG(unittest.TestCase):
    def setUp(self) -> None:
        self.in_features = 10
        self.hidden_size = 5
        self.out_features = 2
        self.use_bias = True
        self.use_batch_norm = False
        self.single_node_attributes = {"type": "linear", "size": self.hidden_size}
        self.default_node_attributes = {"type": "linear", "size": 0, "activation": "selu"}
        self.dag = GrowingDAG(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.hidden_size,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            layer_type="linear",
        )
        self.dag.remove_edge(self.dag.root, self.dag.end)

    def tearDown(self) -> None:
        del self.dag
        return super().tearDown()

    def test_init(self) -> None:
        self.assertEqual(list(self.dag.nodes), ["start", "end"])
        self.assertEqual(len(self.dag.edges), 0)
        self.assertEqual(self.dag.in_degree("start"), 0)
        self.assertEqual(self.dag.out_degree("end"), 0)
        self.assertEqual(self.dag.id_last_node_added, 0)

    def test_get_edge_module(self) -> None:
        self.dag.add_direct_edge("start", "end")
        self.assertEqual(
            self.dag.get_edge_module("start", "end"), self.dag["start"]["end"]["module"]
        )

    def test_get_node_module(self) -> None:
        self.assertEqual(
            self.dag.get_node_module("start"), self.dag.nodes["start"]["module"]
        )

    def test_get_edge_modules(self) -> None:
        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        edges = [("start", "1"), ("1", "end")]
        self.assertEqual(
            self.dag.get_edge_modules(edges),
            [
                self.dag.get_edge_module(*edges[0]),
                self.dag.get_edge_module(*edges[1]),
            ],
        )

    def test_get_node_modules(self) -> None:
        self.assertEqual(
            self.dag.get_node_modules(["start", "end"]),
            [
                self.dag.get_node_module("start"),
                self.dag.get_node_module("end"),
            ],
        )

    def test_get_all_node_modules(self) -> None:
        self.assertEqual(
            set(self.dag.get_all_node_modules()),
            set(
                [
                    self.dag.get_node_module("start"),
                    self.dag.get_node_module("end"),
                ]
            ),
        )
        self.dag.add_node_with_two_edges(
            self.dag.root, "test", self.dag.end, self.single_node_attributes
        )
        self.assertEqual(
            set(self.dag.get_all_node_modules()),
            set(
                [
                    self.dag.get_node_module("start"),
                    self.dag.get_node_module("test"),
                    self.dag.get_node_module("end"),
                ]
            ),
        )

    def test_add_direct_edge(self) -> None:
        self.dag.add_direct_edge(prev_node="start", next_node="end")
        self.assertEqual(list(self.dag.edges), [("start", "end")])
        self.assertEqual(self.dag.out_degree("start"), 1)
        self.assertEqual(self.dag.in_degree("end"), 1)

        self.assertIsInstance(
            self.dag.get_edge_module("start", "end"), LinearGrowingModule
        )
        self.assertIsNotNone(self.dag.get_node_module("start").next_modules)
        self.assertIsNotNone(self.dag.get_edge_module("start", "end").previous_module)
        self.assertIsNotNone(self.dag.get_edge_module("start", "end").next_module)
        self.assertIsNotNone(self.dag.get_node_module("end").previous_modules)

        self.dag.add_direct_edge(
            prev_node="start", next_node="end", edge_attributes={"constant": True}
        )
        self.assertIsInstance(self.dag.get_edge_module("start", "end"), ConstantModule)

    def test_add_node_with_two_edges(self) -> None:
        self.assertEqual(len(self.dag.nodes), 2)
        self.assertEqual(self.dag.out_degree("start"), 0)
        self.assertEqual(self.dag.in_degree("end"), 0)

        params = ["start", "1", "end"]
        node_attributes = {}
        with self.assertRaises(KeyError):
            self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)
        node_attributes["type"] = "linear"
        with self.assertRaises(KeyError):
            self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)
        node_attributes["size"] = self.hidden_size
        self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)

        self.assertEqual(len(self.dag.nodes), 3)
        self.assertEqual(self.dag.out_degree("start"), 1)
        self.assertEqual(self.dag.in_degree("end"), 1)

        self.assertIsNotNone(self.dag.get_node_module("start").next_modules)
        self.assertIsNotNone(self.dag.get_edge_module("start", "1").previous_module)
        self.assertIsNotNone(self.dag.get_edge_module("start", "1").next_module)
        self.assertIsNotNone(self.dag.get_node_module("1").previous_modules)
        self.assertIsNotNone(self.dag.get_node_module("1").next_modules)
        self.assertIsNotNone(self.dag.get_edge_module("1", "end").previous_module)
        self.assertIsNotNone(self.dag.get_edge_module("1", "end").next_module)
        self.assertIsNotNone(self.dag.get_node_module("end").previous_modules)

    def test_remove_direct_edge(self) -> None:
        self.dag.add_direct_edge(prev_node="start", next_node="end")
        self.dag.remove_direct_edge(prev_node="start", next_node="end")
        self.assertEqual(len(list(self.dag.edges)), 0)
        self.assertEqual(self.dag.out_degree("start"), 0)
        self.assertEqual(self.dag.in_degree("end"), 0)

        self.dag.remove_direct_edge(prev_node="start", next_node="end")
        self.assertEqual(len(list(self.dag.edges)), 0)
        self.assertEqual(self.dag.out_degree("start"), 0)
        self.assertEqual(self.dag.in_degree("end"), 0)

    def test_update_nodes(self) -> None:
        new_node = "new"
        edges = [("start", new_node), (new_node, "end")]
        self.dag.add_edges_from(edges)

        node_attributes = {new_node: {}}
        with self.assertRaises(KeyError):
            self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)
        node_attributes[new_node]["type"] = "linear"
        with self.assertRaises(KeyError):
            self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)
        node_attributes[new_node]["size"] = self.hidden_size
        self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)

        self.assertIsInstance(
            self.dag.get_node_module(new_node), LinearMergeGrowingModule
        )
        self.assertIsInstance(
            self.dag.get_node_module(new_node).post_merge_function[0],
            torch.nn.Identity,
        )
        self.assertIsNotNone(self.dag.get_node_module(new_node)._allow_growing)
        self.assertEqual(self.dag.get_node_module(new_node).in_features, self.hidden_size)
        self.assertEqual(len(self.dag.get_node_module(new_node).previous_modules), 0)
        self.assertEqual(len(self.dag.get_node_module(new_node).next_modules), 0)

        node_attributes[new_node]["use_batch_norm"] = True
        self.dag.update_nodes(nodes=[new_node], node_attributes=node_attributes)
        self.assertIsInstance(
            self.dag.get_node_module(new_node).post_merge_function[0],
            torch.nn.BatchNorm1d,
        )
        self.assertEqual(
            self.dag.get_node_module(new_node).post_merge_function[0].num_features,
            self.hidden_size,
        )

    def test_update_edges(self) -> None:
        self.dag.add_edge("start", "end")
        self.dag.update_edges([("start", "end")])

        self.assertIsInstance(
            self.dag.get_edge_module("start", "end"), LinearGrowingModule
        )
        self.assertEqual(
            self.dag.get_edge_module("start", "end").in_features, self.in_features
        )
        self.assertEqual(
            self.dag.get_edge_module("start", "end").out_features, self.out_features
        )
        self.assertIsInstance(
            self.dag.get_edge_module("start", "end").post_layer_function,
            torch.nn.Identity,
        )
        self.assertIsNone(self.dag.get_edge_module("start", "end").previous_module)
        self.assertIsNone(self.dag.get_edge_module("start", "end").next_module)

    def test_update_connections(self) -> None:
        self.dag.update_connections([])
        self.assertTrue(self.dag.is_empty())

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )

        self.assertEqual(self.dag.get_node_module("start").previous_modules, [])
        self.assertEqual(
            self.dag.get_node_module("start").next_modules,
            [self.dag.get_edge_module("start", "1")],
        )

        self.assertEqual(
            self.dag.get_edge_module("start", "1").previous_module,
            self.dag.get_node_module("start"),
        )
        self.assertEqual(
            self.dag.get_edge_module("start", "1").next_module,
            self.dag.get_node_module("1"),
        )

        self.assertEqual(
            self.dag.get_node_module("1").previous_modules,
            [self.dag.get_edge_module("start", "1")],
        )
        self.assertEqual(
            self.dag.get_node_module("1").next_modules,
            [self.dag.get_edge_module("1", "end")],
        )

        self.assertEqual(
            self.dag.get_edge_module("1", "end").previous_module,
            self.dag.get_node_module("1"),
        )
        self.assertEqual(
            self.dag.get_edge_module("1", "end").next_module,
            self.dag.get_node_module("end"),
        )

        self.assertEqual(
            self.dag.get_node_module("end").previous_modules,
            [self.dag.get_edge_module("1", "end")],
        )
        self.assertEqual(self.dag.get_node_module("end").next_modules, [])

    def test_is_empty(self) -> None:
        self.assertTrue(self.dag.is_empty())

        self.dag.add_edge("start", "end")
        self.assertFalse(self.dag.is_empty())

    def test_calculate_bottleneck(self) -> None:
        expansions = [
            Expansion(
                self.dag,
                type="new edge",
                previous_node=self.dag.root,
                next_node=self.dag.end,
            )
        ]

        x = torch.rand((50, self.in_features), device=global_device())
        y = torch.rand((50, self.out_features), device=global_device())

        bottleneck, input_B = self.dag.calculate_bottleneck(
            actions=expansions,
            X=x,
            Y=y,
        )
        self.assertIn(self.dag.end, bottleneck)
        self.assertEqual(bottleneck[self.dag.end].shape, (50, self.out_features))
        self.assertIn(self.dag.root, input_B)
        self.assertEqual(input_B[self.dag.root].shape, (50, self.in_features))

        self.dag.add_node_with_two_edges(
            self.dag.root,
            "test",
            self.dag.end,
            self.single_node_attributes,
            zero_weights=True,
        )
        bottleneck, input_B = self.dag.calculate_bottleneck(
            actions=expansions,
            X=x,
            Y=y,
        )
        for node_module in self.dag.get_all_node_modules():
            self.assertIsNone(node_module.activity)
        self.assertIsNotNone(
            self.dag.get_edge_module("test", self.dag.end).optimal_delta_layer
        )
        self.assertIsNotNone(
            expansions[0].dag.get_edge_module("test", self.dag.end).optimal_delta_layer
        )
        self.assertTrue(
            torch.all(
                self.dag.get_edge_module("test", self.dag.end).optimal_delta_layer.weight
                == expansions[0]
                .dag.get_edge_module("test", self.dag.end)
                .optimal_delta_layer.weight
            )
        )
        self.assertTrue(
            torch.all(
                self.dag.get_edge_module("test", self.dag.end).optimal_delta_layer.bias
                == expansions[0]
                .dag.get_edge_module("test", self.dag.end)
                .optimal_delta_layer.bias
            )
        )

    def test_get_ancestors(self) -> None:
        self.dag.add_direct_edge("start", "end")
        self.assertEqual(self.dag.ancestors["start"], set(["start"]))
        self.assertEqual(self.dag.ancestors["end"], set(["start", "end"]))

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        self.assertEqual(self.dag.ancestors["start"], set(["start"]))
        self.assertEqual(self.dag.ancestors["1"], set(["start", "1"]))
        self.assertEqual(self.dag.ancestors["end"], set(["start", "1", "end"]))

        self.dag.add_node_with_two_edges(
            "start", "2", "1", node_attributes=self.single_node_attributes
        )
        self.assertEqual(self.dag.ancestors["start"], set(["start"]))
        self.assertEqual(self.dag.ancestors["1"], set(["start", "2", "1"]))
        self.assertEqual(self.dag.ancestors["2"], set(["start", "2"]))
        self.assertEqual(self.dag.ancestors["end"], set(["start", "2", "1", "end"]))

    def test_indirect_connection_exists(self) -> None:
        self.assertFalse(self.dag._indirect_connection_exists("start", "end"))

        node_attributes = self.single_node_attributes
        self.dag.add_node_with_two_edges("start", "1", "end", node_attributes)
        self.dag.add_node_with_two_edges("start", "2", "1", node_attributes)
        self.assertTrue(self.dag._indirect_connection_exists("start", "end"))
        self.assertTrue(self.dag._indirect_connection_exists("start", "1"))
        self.assertFalse(self.dag._indirect_connection_exists("1", "start"))
        self.assertFalse(self.dag._indirect_connection_exists("2", "1"))
        self.assertTrue(self.dag._indirect_connection_exists("2", "end"))
        self.assertFalse(self.dag._indirect_connection_exists("start", "2"))

    def test_find_possible_direct_connections(self) -> None:
        nodes_set = set(self.dag.nodes)
        possible_direct_successors = {
            node: (nodes_set.difference(self.dag.ancestors[node])).difference(
                self.dag.successors(node)
            )
            for node in self.dag.nodes
        }
        self.assertEqual(
            self.dag._find_possible_direct_connections(possible_direct_successors),
            [{"previous_node": "start", "next_node": "end"}],
        )

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
        self.assertEqual(direct_connections, [])

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
        self.assertEqual(direct_connections, [{"previous_node": "2", "next_node": "end"}])

    def test_find_possible_one_hop_connections(self) -> None:
        nodes_set = set(self.dag.nodes)
        possible_successors = {
            node: nodes_set.difference(self.dag.ancestors[node])
            for node in self.dag.nodes
        }
        one_hop_edges = self.dag._find_possible_one_hop_connections(possible_successors)
        self.assertEqual(
            one_hop_edges,
            [
                {
                    "previous_node": "start",
                    "new_node": "1",
                    "next_node": "end",
                    "node_attributes": self.default_node_attributes,
                }
            ],
        )

        self.dag.add_node_with_two_edges("start", "1", "end", self.single_node_attributes)
        nodes_set = set(self.dag.nodes)
        possible_successors = {
            node: nodes_set.difference(self.dag.ancestors[node])
            for node in self.dag.nodes
        }
        one_hop_edges = self.dag._find_possible_one_hop_connections(possible_successors)
        self.assertEqual(
            one_hop_edges,
            [
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
            ],
        )

    def test_find_possible_extension(self) -> None:
        direct_edges, one_hop_edges = self.dag.find_possible_extensions()
        self.assertEqual(direct_edges, [{"previous_node": "start", "next_node": "end"}])
        self.assertEqual(
            one_hop_edges,
            [
                {
                    "previous_node": "start",
                    "new_node": "1",
                    "next_node": "end",
                    "node_attributes": self.default_node_attributes,
                }
            ],
        )

        self.dag.add_node_with_two_edges(
            "start", "hidden", "end", self.single_node_attributes
        )
        direct_edges, one_hop_edges = self.dag.find_possible_extensions()
        self.assertEqual(direct_edges, [{"previous_node": "start", "next_node": "end"}])
        self.assertEqual(
            one_hop_edges,
            [
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
            ],
        )

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
        self.assertTrue(torch.all(out == actual_out))

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
        self.assertTrue(torch.all(out == actual_out))

    def test_safe_forward(self) -> None:
        in_features = 0
        out_features = 2
        batch_size = 5
        linear = torch.nn.Linear(in_features, out_features, device=global_device())
        x = torch.rand((batch_size, in_features), device=global_device())
        self.assertTrue(
            torch.all(
                linear(x)
                == torch.zeros((batch_size, out_features), device=global_device())
            )
        )

        in_features = 3
        linear = torch.nn.Linear(in_features, out_features, device=global_device())
        x = torch.rand((batch_size, in_features), device=global_device())
        self.assertTrue(
            torch.all(
                linear(x) == torch.nn.functional.linear(x, linear.weight, linear.bias)
            )
        )

    def test_parameters(self) -> None:
        self.assertEqual(len(list(self.dag.parameters())), len(self.dag.edges) * 2)

        self.dag.add_direct_edge("start", "end")
        self.assertEqual(len(list(self.dag.parameters())), len(self.dag.edges) * 2)

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        self.assertEqual(len(list(self.dag.parameters())), len(self.dag.edges) * 2)

    def test_count_parameters_all(self) -> None:
        self.assertEqual(self.dag.count_parameters_all(), 0)

        self.dag.add_direct_edge("start", "end")
        numel = self.in_features * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters_all(), numel)

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        numel += self.in_features * self.hidden_size + self.hidden_size
        numel += self.hidden_size * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters_all(), numel)

    def test_count_parameters(self) -> None:
        self.dag.add_direct_edge("start", "end")
        numel = self.in_features * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters([("start", "end")]), numel)

        self.dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        numel = self.in_features * self.hidden_size + self.hidden_size
        self.assertEqual(self.dag.count_parameters([("start", "1")]), numel)
        numel += self.hidden_size * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters([("start", "1"), ("1", "end")]), numel)

    def test_evaluate_extended(self) -> None:
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
        y = torch.rand((50, self.out_features), device=global_device()).argmax(axis=1)
        loss_fn = torch.nn.CrossEntropyLoss()
        actual_out = self.dag.extended_forward(x)
        actual_loss = loss_fn(actual_out, y).item()
        acc, _, f1 = self.dag.evaluate_extended(
            x, actual_out.argmax(axis=1), loss_fn, with_f1score=True
        )
        self.assertEqual(acc, 1.0)
        self.assertEqual(f1, 1.0)

        _, loss = self.dag.evaluate_extended(x, y, loss_fn, with_f1score=False)
        self.assertEqual(actual_loss, loss)

        dag = GrowingDAG(
            in_features=self.in_features,
            out_features=1,
            neurons=self.hidden_size,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            layer_type="linear",
        )
        dag.get_edge_module("start", "end").optimal_delta_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=1,
            device=global_device(),
        )
        dag.add_node_with_two_edges(
            "start", "1", "end", node_attributes=self.single_node_attributes
        )
        dag.get_edge_module("start", "1").extended_output_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            device=global_device(),
        )
        dag.get_edge_module("1", "end").extended_input_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=1,
            device=global_device(),
        )
        y = torch.rand((50, 1), device=global_device())
        loss_fn = torch.nn.MSELoss()
        actual_out = dag.extended_forward(x)
        actual_loss = loss_fn(actual_out, y).item()
        acc, loss, f1 = dag.evaluate_extended(x, y, loss_fn, with_f1score=True)
        self.assertEqual(acc, -1)
        self.assertEqual(f1, -1)
        self.assertEqual(loss, actual_loss)

    def test_expansion_init(self) -> None:
        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                type="random",
            )

        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                type="new edge",
            )
        with self.assertWarns(UserWarning):
            Expansion(
                self.dag,
                type="new edge",
                previous_node=self.dag.root,
                next_node=self.dag.end,
                expanding_node="test",
            )
        expansion = Expansion(
            self.dag,
            type="new edge",
            previous_node=self.dag.root,
            next_node=self.dag.end,
        )
        self.assertIsNot(self.dag, expansion.dag)

        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                type="new node",
            )

        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                type="expanded node",
            )
        with self.assertWarns(UserWarning):
            Expansion(
                self.dag,
                type="expanded node",
                expanding_node="test",
                previous_node=self.dag.root,
            )

    def test_expansion_new_edges(self) -> None:
        expansion = Expansion(
            self.dag,
            type="new edge",
            previous_node=self.dag.root,
            next_node=self.dag.end,
        )
        self.assertEqual(expansion.new_edges, (self.dag.root, self.dag.end))
        expansion.expand()
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag.root, self.dag.end).weight)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag.root, self.dag.end).bias)
        )

        expansion = Expansion(
            self.dag,
            type="new node",
            expanding_node="test",
            previous_node=self.dag.root,
            next_node=self.dag.end,
            node_attributes=self.single_node_attributes,
        )
        self.assertEqual(
            expansion.new_edges,
            [
                (self.dag.root, "test"),
                ("test", self.dag.end),
            ],
        )
        expansion.expand()
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag.root, "test").weight)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag.root, "test").bias)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module("test", self.dag.end).weight)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module("test", self.dag.end).bias)
        )

        self.dag.add_node_with_two_edges(
            self.dag.root,
            "test",
            self.dag.end,
            node_attributes=self.single_node_attributes,
        )
        expansion = Expansion(
            self.dag,
            type="expanded node",
            expanding_node="test",
        )
        self.assertEqual(
            expansion.new_edges,
            [
                (self.dag.root, "test"),
                ("test", self.dag.end),
            ],
        )


if __name__ == "__main__":
    unittest.main()
