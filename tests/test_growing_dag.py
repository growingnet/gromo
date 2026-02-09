import unittest

import torch

from gromo.containers.growing_dag import (
    Expansion,
    ExpansionType,
    GrowingDAG,
    InterMergeExpansion,
)
from gromo.modules.constant_module import ConstantModule
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


# torch.set_default_tensor_type(torch.DoubleTensor)


class TestGrowingDAG(TorchTestCase):
    def setUp(self) -> None:
        self.in_features = 10
        self.hidden_size = 5
        self.out_features = 2
        self.use_bias = True
        self.use_batch_norm = False
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.init_node_attributes = {"type": "linear", "size": self.hidden_size}
        self.default_node_attributes = {
            "type": "linear",
            "size": 0,
            "activation": "selu",
            "kernel_size": (3, 3),
            "shape": None,
        }
        self.init_node_conv_attributes = {
            "type": "convolution",
            "size": self.hidden_size,
            "kernel_size": (3, 3),
            "shape": (3, 3),
        }
        self.default_edge_attributes = {"kernel_size": (3, 3)}
        self.dag = GrowingDAG(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.hidden_size,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            default_layer_type="linear",
            name="dag-linear",
        )
        self.dag.remove_direct_edge(self.dag.root, self.dag.end)
        self.dag_conv = GrowingDAG(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.hidden_size,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            default_layer_type="convolution",
            input_shape=(3, 3),
            name="dag-conv",
        )
        self.dag_conv.remove_direct_edge(self.dag_conv.root, self.dag_conv.end)

    def tearDown(self) -> None:
        del self.dag
        del self.dag_conv
        return super().tearDown()

    def test_init(self) -> None:
        self.assertEqual(list(self.dag.nodes), [self.dag.root, self.dag.end])
        self.assertEqual(len(self.dag.edges), 0)
        self.assertEqual(self.dag.in_degree(self.dag.root), 0)
        self.assertEqual(self.dag.out_degree(self.dag.root), 0)
        self.assertEqual(self.dag.in_degree(self.dag.end), 0)
        self.assertEqual(self.dag.out_degree(self.dag.end), 0)
        self.assertEqual(len(self.dag._growing_layers), 2)

        with self.assertRaises(ValueError):
            # Character _ not allowed in name
            GrowingDAG(
                in_features=self.in_features,
                out_features=self.out_features,
                neurons=self.hidden_size,
                use_bias=self.use_bias,
                use_batch_norm=self.use_batch_norm,
                name="test_name",
            )
        dag = GrowingDAG(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.hidden_size,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            DAG_parameters={},
        )
        self.assertEqual(len(dag.nodes), 0)
        self.assertEqual(len(dag.edges), 0)
        self.assertEqual(dag.ancestors, {})
        self.assertEqual(len(dag._growing_layers), 0)

    def test_edge_candidate(self) -> None:
        with self.assertRaises(ValueError):
            # Edge not present in the graph
            self.dag.toggle_edge_candidate(self.dag.root, self.dag.end, True)
        with self.assertWarns(UserWarning):
            # Edge does not belong in the current graph
            self.assertFalse(self.dag.is_edge_candidate(self.dag.root, self.dag.end))

        self.dag.add_direct_edge(self.dag.root, self.dag.end)
        self.assertFalse(self.dag.is_edge_candidate(self.dag.root, self.dag.end))

        self.dag.toggle_edge_candidate(self.dag.root, self.dag.end, True)
        self.assertTrue(self.dag.is_edge_candidate(self.dag.root, self.dag.end))

        self.dag.toggle_edge_candidate(self.dag.root, self.dag.end, False)
        self.assertFalse(self.dag.is_edge_candidate(self.dag.root, self.dag.end))

    def test_node_candidate(self) -> None:
        with self.assertRaises(ValueError):
            # Node not present in the graph
            self.dag.toggle_node_candidate("test", True)
        with self.assertWarns(UserWarning):
            # Node does not belong in the current graph
            self.assertFalse(self.dag.is_node_candidate("test"))

        self.dag.add_node_with_two_edges(
            self.dag.root, "test", self.dag.end, node_attributes=self.init_node_attributes
        )
        self.assertFalse(self.dag.is_node_candidate("test"))

        self.dag.toggle_node_candidate("test", True)
        self.assertTrue(self.dag.is_node_candidate("test"))

        self.dag.toggle_node_candidate("test", False)
        self.assertFalse(self.dag.is_node_candidate("test"))

    def test_get_edge_module(self) -> None:
        self.dag.add_direct_edge(self.dag.root, self.dag.end)
        self.assertEqual(
            self.dag.get_edge_module(self.dag.root, self.dag.end),
            self.dag[self.dag.root][self.dag.end]["module"],
        )

    def test_get_node_module(self) -> None:
        self.assertEqual(
            self.dag.get_node_module(self.dag.root),
            self.dag.nodes[self.dag.root]["module"],
        )

    def test_get_edge_modules(self) -> None:
        self.dag.add_node_with_two_edges(
            self.dag.root, "1", self.dag.end, node_attributes=self.init_node_attributes
        )
        edges = [(self.dag.root, "1"), ("1", self.dag.end)]
        self.assertEqual(
            self.dag.get_edge_modules(edges),
            [
                self.dag.get_edge_module(*edges[0]),
                self.dag.get_edge_module(*edges[1]),
            ],
        )

    def test_get_node_modules(self) -> None:
        self.assertEqual(
            self.dag.get_node_modules([self.dag.root, self.dag.end]),
            [
                self.dag.get_node_module(self.dag.root),
                self.dag.get_node_module(self.dag.end),
            ],
        )

    def test_get_all_edge_modules(self) -> None:
        self.assertEqual(self.dag.get_all_edge_modules(), [])
        self.dag.add_direct_edge(prev_node=self.dag.root, next_node=self.dag.end)
        self.assertEqual(
            self.dag.get_all_edge_modules(),
            [self.dag.get_edge_module(self.dag.root, self.dag.end)],
        )
        self.dag.add_node_with_two_edges(
            self.dag.root, "test", self.dag.end, self.init_node_attributes
        )
        self.assertEqual(
            set(self.dag.get_all_edge_modules()),
            {
                self.dag.get_edge_module(self.dag.root, self.dag.end),
                self.dag.get_edge_module(self.dag.root, "test"),
                self.dag.get_edge_module("test", self.dag.end),
            },
        )

    def test_get_all_node_modules(self) -> None:
        self.assertEqual(
            set(self.dag.get_all_node_modules()),
            {
                self.dag.get_node_module(self.dag.root),
                self.dag.get_node_module(self.dag.end),
            },
        )
        self.dag.add_node_with_two_edges(
            self.dag.root, "test", self.dag.end, self.init_node_attributes
        )
        self.assertEqual(
            set(self.dag.get_all_node_modules()),
            {
                self.dag.get_node_module(self.dag.root),
                self.dag.get_node_module("test"),
                self.dag.get_node_module(self.dag.end),
            },
        )

    def test_add_direct_edge(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_direct_edge(prev_node=start, next_node=end)
        self.assertEqual(list(self.dag.edges), [(start, end)])
        self.assertEqual(self.dag.out_degree(start), 1)
        self.assertEqual(self.dag.in_degree(end), 1)

        self.assertIsInstance(self.dag.get_edge_module(start, end), LinearGrowingModule)
        self.assertIsNotNone(self.dag.get_node_module(start).next_modules)
        self.assertIsNotNone(self.dag.get_edge_module(start, end).previous_module)
        self.assertIsNotNone(self.dag.get_edge_module(start, end).next_module)
        self.assertIsNotNone(self.dag.get_node_module(end).previous_modules)

        self.dag.add_direct_edge(
            prev_node=start, next_node=end, edge_attributes={"constant": True}
        )
        self.assertIsInstance(self.dag.get_edge_module(start, end), ConstantModule)

    def test_add_node_with_two_edges(self) -> None:
        self.assertEqual(len(self.dag.nodes), 2)
        self.assertEqual(self.dag.out_degree(self.dag.root), 0)
        self.assertEqual(self.dag.in_degree(self.dag.end), 0)

        params = [self.dag.root, "1", self.dag.end]
        node_attributes = {}
        with self.assertRaises(KeyError):
            self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)
        node_attributes["type"] = "linear"
        with self.assertRaises(KeyError):
            self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)
        node_attributes["size"] = self.hidden_size
        self.dag.add_node_with_two_edges(*params, node_attributes=node_attributes)

        self.assertEqual(len(self.dag.nodes), 3)
        self.assertEqual(self.dag.out_degree(self.dag.root), 1)
        self.assertEqual(self.dag.in_degree(self.dag.end), 1)

        self.assertIsNotNone(self.dag.get_node_module(self.dag.root).next_modules)
        self.assertIsNotNone(self.dag.get_edge_module(self.dag.root, "1").previous_module)
        self.assertIsNotNone(self.dag.get_edge_module(self.dag.root, "1").next_module)
        self.assertIsNotNone(self.dag.get_node_module("1").previous_modules)
        self.assertIsNotNone(self.dag.get_node_module("1").next_modules)
        self.assertIsNotNone(self.dag.get_edge_module("1", self.dag.end).previous_module)
        self.assertIsNotNone(self.dag.get_edge_module("1", self.dag.end).next_module)
        self.assertIsNotNone(self.dag.get_node_module(self.dag.end).previous_modules)

    def test_remove_direct_edge(self) -> None:
        self.dag.add_direct_edge(prev_node=self.dag.root, next_node=self.dag.end)
        self.dag.remove_direct_edge(prev_node=self.dag.root, next_node=self.dag.end)
        self.assertEqual(len(list(self.dag.edges)), 0)
        self.assertEqual(self.dag.out_degree(self.dag.root), 0)
        self.assertEqual(self.dag.in_degree(self.dag.end), 0)

        self.dag.remove_direct_edge(prev_node=self.dag.root, next_node=self.dag.end)
        self.assertEqual(len(list(self.dag.edges)), 0)
        self.assertEqual(self.dag.out_degree(self.dag.root), 0)
        self.assertEqual(self.dag.in_degree(self.dag.end), 0)

    def test_update_nodes(self) -> None:
        new_node = "new"
        edges = [(self.dag.root, new_node), (new_node, self.dag.end)]
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
        start, end = self.dag.root, self.dag.end
        self.dag.add_edge(start, end)
        self.dag.update_edges([(start, end)])

        self.assertIsInstance(self.dag.get_edge_module(start, end), LinearGrowingModule)
        self.assertEqual(
            self.dag.get_edge_module(start, end).in_features, self.in_features
        )
        self.assertEqual(
            self.dag.get_edge_module(start, end).out_features, self.out_features
        )
        self.assertIsInstance(
            self.dag.get_edge_module(start, end).post_layer_function,
            torch.nn.Identity,
        )
        self.assertIsNone(self.dag.get_edge_module(start, end).previous_module)
        self.assertIsNone(self.dag.get_edge_module(start, end).next_module)

        self.dag.nodes[start]["type"] = "convolution"
        self.dag.use_bias = False
        self.dag.update_edges([(start, end)], zero_weights=True)
        self.assertIsInstance(self.dag.get_edge_module(start, end), LinearGrowingModule)
        self.assertTrue(torch.all(self.dag.get_edge_module(start, end).weight) == 0)
        self.assertIsNone(self.dag.get_edge_module(start, end).bias)

        start, end = self.dag_conv.root, self.dag_conv.end
        self.dag_conv.add_edge(start, end)
        with self.assertRaises(
            KeyError
        ):  # The kernel size of the edge should be specified at initialization
            self.dag_conv.update_edges([(start, end)])
        self.dag_conv.update_edges(
            [(start, end)], edge_attributes={"kernel_size": (3, 3)}
        )

        self.assertIsInstance(
            self.dag_conv.get_edge_module(start, end), Conv2dGrowingModule
        )
        self.assertEqual(
            self.dag_conv.get_edge_module(start, end).in_channels, self.in_features
        )
        self.assertEqual(
            self.dag_conv.get_edge_module(start, end).out_channels, self.out_features
        )
        self.assertIsInstance(
            self.dag_conv.get_edge_module(start, end).post_layer_function,
            torch.nn.Identity,
        )
        self.assertIsNone(self.dag_conv.get_edge_module(start, end).previous_module)
        self.assertIsNone(self.dag_conv.get_edge_module(start, end).next_module)

    def test_update_connections(self) -> None:
        self.dag.update_connections([])
        self.assertTrue(self.dag.is_empty())

        self.dag.add_node_with_two_edges(
            self.dag.root, "1", self.dag.end, node_attributes=self.init_node_attributes
        )

        self.assertEqual(self.dag.get_node_module(self.dag.root).previous_modules, [])
        self.assertEqual(
            self.dag.get_node_module(self.dag.root).next_modules,
            [self.dag.get_edge_module(self.dag.root, "1")],
        )

        self.assertEqual(
            self.dag.get_edge_module(self.dag.root, "1").previous_module,
            self.dag.get_node_module(self.dag.root),
        )
        self.assertEqual(
            self.dag.get_edge_module(self.dag.root, "1").next_module,
            self.dag.get_node_module("1"),
        )

        self.assertEqual(
            self.dag.get_node_module("1").previous_modules,
            [self.dag.get_edge_module(self.dag.root, "1")],
        )
        self.assertEqual(
            self.dag.get_node_module("1").next_modules,
            [self.dag.get_edge_module("1", self.dag.end)],
        )

        self.assertEqual(
            self.dag.get_edge_module("1", self.dag.end).previous_module,
            self.dag.get_node_module("1"),
        )
        self.assertEqual(
            self.dag.get_edge_module("1", self.dag.end).next_module,
            self.dag.get_node_module(self.dag.end),
        )

        self.assertEqual(
            self.dag.get_node_module(self.dag.end).previous_modules,
            [self.dag.get_edge_module("1", self.dag.end)],
        )
        self.assertEqual(self.dag.get_node_module(self.dag.end).next_modules, [])

    def test_update_size(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        alpha = torch.zeros_like(self.dag.get_edge_module(start, "1").weight)
        omega = torch.zeros_like(self.dag.get_edge_module("1", end).weight)
        with self.assertWarns(UserWarning):  # the size has changed
            self.dag.get_edge_module(start, "1").add_parameters(
                matrix_extension=alpha,
                bias_extension=None,
                added_out_features=self.hidden_size,
            )
            self.dag.get_edge_module("1", end).add_parameters(
                matrix_extension=omega,
                bias_extension=None,
                added_in_features=self.hidden_size,
            )
        self.assertEqual(
            self.dag.get_edge_module(start, "1").out_features, self.hidden_size * 2
        )
        self.assertEqual(
            self.dag.get_edge_module("1", end).in_features, self.hidden_size * 2
        )

        self.dag.update_size()

        self.assertEqual(self.dag.get_node_module("1").in_features, self.hidden_size * 2)
        self.assertEqual(self.dag.nodes["1"]["size"], self.hidden_size * 2)

    def test_remove_node(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        self.dag.remove_node("1")

        self.assertNotIn("1", self.dag.nodes)
        self.assertNotIn((start, "1"), self.dag.edges)
        self.assertNotIn(("1", end), self.dag.edges)
        self.assertEqual(len(self.dag.get_node_module(start).next_modules), 0)
        self.assertEqual(len(self.dag.get_node_module(end).previous_modules), 0)

        self.dag.remove_node("1")

    def test_rename_nodes(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        node_module = self.dag.get_node_module("1")
        edge_module = self.dag.get_edge_module("1", end)
        self.dag.rename_nodes({"1": "2", "test": "0", "2": "2"})

        self.assertNotIn("1", self.dag.nodes)
        self.assertNotIn("test", self.dag.nodes)
        self.assertNotIn("0", self.dag.nodes)
        self.assertIn("2", self.dag.nodes)
        self.assertNotIn((start, "1"), self.dag.edges)
        self.assertNotIn(("1", end), self.dag.edges)
        self.assertIn((start, "2"), self.dag.edges)
        self.assertIn(("2", end), self.dag.edges)
        self.assertIs(node_module, self.dag.get_node_module("2"))
        self.assertEqual(node_module._name, "1")
        self.assertIs(edge_module, self.dag.get_edge_module("2", end))

        with self.assertRaises(ValueError):  # New node name already in the graph
            self.dag.rename_nodes({"2": end})

    def test_is_empty(self) -> None:
        self.assertTrue(self.dag.is_empty())

        self.dag.add_edge(self.dag.root, self.dag.end)
        self.assertFalse(self.dag.is_empty())

    def test_reset_computation(self) -> None:
        self.dag.add_node_with_two_edges(
            self.dag.root, "1", self.dag.end, node_attributes=self.init_node_attributes
        )
        self.dag.get_node_module(self.dag.root).store_activity = True
        self.dag.get_node_module(self.dag.end).init_computation()

        x = torch.rand((50, self.in_features), device=global_device())
        y = torch.rand((50, self.out_features), device=global_device())

        pred = self.dag(x)
        loss = self.loss_fn(pred, y)
        loss.backward()

        self.dag.reset_computation()

        for edge_module in self.dag.get_all_edge_modules():
            self.assertFalse(edge_module.store_input)
            self.assertFalse(edge_module.store_pre_activity)
            self.assertIsNone(edge_module.tensor_s._tensor)
            self.assertIsNone(edge_module.tensor_m._tensor)
            self.assertIsNone(edge_module.tensor_m_prev._tensor)
            self.assertIsNone(edge_module.cross_covariance._tensor)
        for node_module in self.dag.get_all_node_modules():
            self.assertFalse(node_module.store_input)
            self.assertFalse(node_module.store_activity)
            if node_module.previous_tensor_s is not None:
                self.assertIsNone(node_module.previous_tensor_s._tensor)
            if node_module.previous_tensor_m is not None:
                self.assertIsNone(node_module.previous_tensor_m._tensor)

    def test_delete_update(self) -> None:
        self.dag.add_node_with_two_edges(
            self.dag.root, "1", self.dag.end, node_attributes=self.init_node_attributes
        )
        self.dag.get_node_module(self.dag.root).store_activity = True
        self.dag.get_node_module(self.dag.end).init_computation()

        x = torch.rand((50, self.in_features), device=global_device())
        y = torch.rand((50, self.out_features), device=global_device())

        pred = self.dag(x)
        loss = self.loss_fn(pred, y)
        loss.backward()

        self.dag.reset_computation()
        self.dag.delete_update()

        for edge_module in self.dag.get_all_edge_modules():
            self.assertIsNone(edge_module.optimal_delta_layer)
            self.assertEqual(edge_module.scaling_factor, 0.0)
            self.assertIsNone(edge_module.parameter_update_decrease)
            self.assertIsNone(edge_module.eigenvalues_extension)
            self.assertIsNone(edge_module._pre_activity)
            self.assertIsNone(edge_module._input)
            self.assertIsNone(edge_module.extended_output_layer)
            self.assertIsNone(edge_module.extended_input_layer)
        for node_module in self.dag.get_all_node_modules():
            self.assertIsNone(node_module.activity)
            self.assertIsNone(node_module.input)

    def test_calculate_bottleneck(self) -> None:
        expansions = [
            Expansion(
                self.dag,
                exp_type=ExpansionType.NEW_EDGE,
                previous_node=self.dag.root,
                next_node=self.dag.end,
            )
        ]

        x = torch.rand((50, self.in_features), device=global_device())
        y = torch.rand((50, self.out_features), device=global_device())
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=10,
        )

        bottleneck, input_B = self.dag.calculate_bottleneck(
            actions=expansions,
            dataloader=dataloader,
        )
        self.assertIn(self.dag.end, bottleneck)
        self.assertEqual(bottleneck[self.dag.end].shape, (50, self.out_features))
        self.assertIn(self.dag.root, input_B)
        self.assertEqual(input_B[self.dag.root].shape, (50, self.in_features))

        self.dag.add_node_with_two_edges(
            self.dag.root,
            "test",
            self.dag.end,
            self.init_node_attributes,
            zero_weights=True,
        )

        with self.assertMaybeWarns(
            UserWarning,
            "Using the pseudo-inverse for the computation of the optimal delta",
        ):
            bottleneck, input_B = self.dag.calculate_bottleneck(
                actions=expansions,
                dataloader=dataloader,
            )

        for node_module in self.dag.get_all_node_modules():
            self.assertIsNone(node_module.activity)
        self.assertIsNotNone(
            self.dag.get_edge_module("test", self.dag.end).optimal_delta_layer
        )
        self.assertIsNotNone(
            self.dag.get_edge_module("test", self.dag.end).optimal_delta_layer.weight
        )
        self.assertIsNotNone(
            self.dag.get_edge_module("test", self.dag.end).optimal_delta_layer.bias
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
        start, end = self.dag.root, self.dag.end
        self.dag.add_direct_edge(start, end)
        self.assertEqual(self.dag.ancestors[start], {start})
        self.assertEqual(self.dag.ancestors[end], {start, end})

        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        self.assertEqual(self.dag.ancestors[start], {start})
        self.assertEqual(self.dag.ancestors["1"], {start, "1"})
        self.assertEqual(self.dag.ancestors[end], {start, "1", end})

        self.dag.add_node_with_two_edges(
            start, "2", "1", node_attributes=self.init_node_attributes
        )
        self.assertEqual(self.dag.ancestors[start], {start})
        self.assertEqual(self.dag.ancestors["1"], {start, "2", "1"})
        self.assertEqual(self.dag.ancestors["2"], {start, "2"})
        self.assertEqual(self.dag.ancestors[end], {start, "2", "1", end})

    def test_indirect_connection_exists(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.assertFalse(self.dag._indirect_connection_exists(start, end))

        node_attributes = self.init_node_attributes
        self.dag.add_node_with_two_edges(start, "1", end, node_attributes)
        self.dag.add_node_with_two_edges(start, "2", "1", node_attributes)
        self.assertTrue(self.dag._indirect_connection_exists(start, end))
        self.assertTrue(self.dag._indirect_connection_exists(start, "1"))
        self.assertFalse(self.dag._indirect_connection_exists("1", start))
        self.assertFalse(self.dag._indirect_connection_exists("2", "1"))
        self.assertTrue(self.dag._indirect_connection_exists("2", end))
        self.assertFalse(self.dag._indirect_connection_exists(start, "2"))

    def test_find_possible_direct_connections(self) -> None:
        start, end = self.dag.root, self.dag.end
        nodes_set = set(self.dag.nodes)
        possible_direct_successors = {
            node: (nodes_set.difference(self.dag.ancestors[node])).difference(
                self.dag.successors(node)
            )
            for node in self.dag.nodes
        }
        self.assertEqual(
            self.dag._find_possible_direct_connections(possible_direct_successors),
            [
                {
                    "previous_node": start,
                    "next_node": end,
                    "edge_attributes": self.default_edge_attributes,
                }
            ],
        )

        self.dag.add_direct_edge(start, end)
        self.dag.add_node_with_two_edges(start, "1", end, self.init_node_attributes)
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
            start, "2", "1", node_attributes=self.init_node_attributes
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
        self.assertEqual(
            direct_connections,
            [
                {
                    "previous_node": "2",
                    "next_node": end,
                    "edge_attributes": self.default_edge_attributes,
                }
            ],
        )

    def test_find_possible_one_hop_connections(self) -> None:
        start, end = self.dag.root, self.dag.end
        nodes_set = set(self.dag.nodes)
        possible_successors = {
            node: nodes_set.difference(self.dag.ancestors[node])
            for node in self.dag.nodes
        }
        one_hop_edges = self.dag._find_possible_one_hop_connections(
            possible_successors, size=0
        )
        self.assertEqual(
            one_hop_edges,
            [
                {
                    "previous_node": start,
                    "new_node": "1@dag-linear",
                    "next_node": end,
                    "node_attributes": self.default_node_attributes,
                    "edge_attributes": self.default_edge_attributes,
                }
            ],
        )

        self.dag.add_node_with_two_edges(start, "1", end, self.init_node_attributes)
        nodes_set = set(self.dag.nodes)
        possible_successors = {
            node: nodes_set.difference(self.dag.ancestors[node])
            for node in self.dag.nodes
        }
        one_hop_edges = self.dag._find_possible_one_hop_connections(
            possible_successors, size=0
        )
        self.assertEqual(
            one_hop_edges,
            [
                {
                    "previous_node": start,
                    "new_node": "2@dag-linear",
                    "next_node": "1",
                    "node_attributes": self.default_node_attributes,
                    "edge_attributes": self.default_edge_attributes,
                },
                {
                    "previous_node": "1",
                    "new_node": "2@dag-linear",
                    "next_node": end,
                    "node_attributes": self.default_node_attributes,
                    "edge_attributes": self.default_edge_attributes,
                },
            ],
        )

    def test_find_possible_extension(self) -> None:
        start, end = self.dag.root, self.dag.end
        direct_edges, one_hop_edges = self.dag.find_possible_extensions()
        self.assertEqual(
            direct_edges,
            [
                {
                    "previous_node": start,
                    "next_node": end,
                    "edge_attributes": self.default_edge_attributes,
                }
            ],
        )
        self.assertEqual(
            one_hop_edges,
            [
                {
                    "previous_node": start,
                    "new_node": "1@dag-linear",
                    "next_node": end,
                    "node_attributes": self.default_node_attributes,
                    "edge_attributes": self.default_edge_attributes,
                }
            ],
        )

        self.dag.add_node_with_two_edges(start, "hidden", end, self.init_node_attributes)
        direct_edges, one_hop_edges = self.dag.find_possible_extensions()
        self.assertEqual(
            direct_edges,
            [
                {
                    "previous_node": start,
                    "next_node": end,
                    "edge_attributes": self.default_edge_attributes,
                }
            ],
        )
        self.assertEqual(
            one_hop_edges,
            [
                {
                    "previous_node": start,
                    "new_node": "2@dag-linear",
                    "next_node": "hidden",
                    "node_attributes": self.default_node_attributes,
                    "edge_attributes": self.default_edge_attributes,
                },
                {
                    "previous_node": "hidden",
                    "new_node": "2@dag-linear",
                    "next_node": end,
                    "node_attributes": self.default_node_attributes,
                    "edge_attributes": self.default_edge_attributes,
                },
            ],
        )

    def test_define_next_actions(self) -> None:
        start, end = self.dag.root, self.dag.end
        base_actions = self.dag.define_next_actions(expand_end=False)
        self.assertEqual(len(base_actions), 2)
        self.assertIsInstance(base_actions[0], Expansion)
        self.assertEqual(base_actions[0].type, ExpansionType.NEW_EDGE)
        self.assertEqual(base_actions[0].previous_node, start)
        self.assertEqual(base_actions[0].next_node, end)
        self.assertIsInstance(base_actions[1], Expansion)
        self.assertEqual(base_actions[1].type, ExpansionType.NEW_NODE)
        self.assertEqual(base_actions[1].expanding_node, "1@dag-linear_a")
        self.assertEqual(base_actions[1].previous_node, start)
        self.assertEqual(base_actions[1].next_node, end)

        actions = self.dag.define_next_actions(expand_end=True)
        self.assertEqual(len(actions), 2)

        next_module = LinearMergeGrowingModule(
            in_features=self.out_features,
            previous_modules=[self.dag.get_node_module(end)],
            name="next",
        )
        self.dag.get_node_module(end).add_next_module(next_module)
        actions = self.dag.define_next_actions(expand_end=True)
        self.assertEqual(len(actions), 3)
        self.assertIsInstance(actions[2], InterMergeExpansion)
        self.assertEqual(actions[2].type, ExpansionType.EXPANDED_NODE)
        self.assertEqual(actions[2].expanding_node, end)
        self.assertEqual(actions[2].adjacent_expanding_node, "next")  # type: ignore
        self.assertIsNone(actions[2].previous_node)
        self.assertIsNone(actions[2].next_node)

        self.dag.get_node_module(end).add_next_module(next_module)
        with self.assertRaises(NotImplementedError):
            # Can only expand single connected inter-merge nodes
            actions = self.dag.define_next_actions(expand_end=True)

    def test_forward(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_direct_edge(start, end)
        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )

        x = torch.rand((50, self.in_features), device=global_device())
        x_a = self.dag.get_edge_module(start, end)(x)
        x_b = self.dag.get_edge_module(start, "1")(x)
        x_b = self.dag.get_node_module("1")(x_b)
        x_b = self.dag.get_edge_module("1", end)(x_b)
        out = x_a.add(x_b)
        out = self.dag.get_node_module(end)(out)

        actual_out = self.dag(x)
        self.assertTrue(torch.all(out == actual_out))

    def test_extended_forward(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_direct_edge(start, end)
        self.dag.get_edge_module(start, end).optimal_delta_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            device=global_device(),
        )
        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        self.dag.get_edge_module(start, "1").extended_output_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            device=global_device(),
        )
        self.dag.get_edge_module("1", end).extended_input_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.out_features,
            device=global_device(),
        )
        self.dag.set_scaling_factor(1)

        x = torch.rand((50, self.in_features), device=global_device())
        x_a = self.dag.get_edge_module(start, end).extended_forward(x)[0]
        x_b = self.dag.get_edge_module(start, "1").extended_forward(x)
        x_b = self.dag.get_node_module("1")(x_b)
        x_b = self.dag.get_edge_module("1", end).extended_forward(*x_b)
        out = x_a.add(x_b[0])
        out = self.dag.get_node_module(end)(out)

        mask = {"nodes": self.dag.nodes, "edges": self.dag.edges}
        actual_out, _ = self.dag.extended_forward(x, mask=mask)
        self.assertTrue(torch.all(out == actual_out))

        self.dag.get_edge_module(start, end).extended_output_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            device=global_device(),
        )
        self.dag.get_edge_module("1", end).extended_output_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            device=global_device(),
        )
        x_a = self.dag.get_edge_module(start, end).extended_forward(x)[1]
        x_b = self.dag.get_edge_module(start, "1").extended_forward(x)
        x_b = self.dag.get_node_module("1")(x_b)
        x_b = self.dag.get_edge_module("1", end).extended_forward(*x_b)[1]
        self.assertIsNotNone(x_a)
        self.assertIsNotNone(x_b)
        out_ext = x_a.add(x_b)  # type: ignore
        out_ext = self.dag.get_node_module(end)(out_ext)
        actual_out, actual_out_ext = self.dag.extended_forward(x, mask=mask)
        self.assertIsNotNone(actual_out_ext)
        self.assertTrue(torch.all(out == actual_out))
        self.assertTrue(torch.all(out_ext == actual_out_ext))

    def test_safe_forward(self) -> None:
        in_features = 0
        out_features = 2
        batch_size = 5
        with self.assertWarns(UserWarning):
            # Initializing zero-element tensors is a no-op
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
        start, end = self.dag.root, self.dag.end
        self.assertEqual(len(list(self.dag.parameters())), len(self.dag.edges) * 2)

        self.dag.add_direct_edge(start, end)
        self.assertEqual(len(list(self.dag.parameters())), len(self.dag.edges) * 2)

        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        self.assertEqual(len(list(self.dag.parameters())), len(self.dag.edges) * 2)

    def test_count_parameters_all(self) -> None:
        self.assertEqual(self.dag.count_parameters_all(), 0)

        self.dag.add_direct_edge(self.dag.root, self.dag.end)
        numel = self.in_features * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters_all(), numel)

        self.dag.add_node_with_two_edges(
            self.dag.root, "1", self.dag.end, node_attributes=self.init_node_attributes
        )
        numel += self.in_features * self.hidden_size + self.hidden_size
        numel += self.hidden_size * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters_all(), numel)

    def test_count_parameters(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_direct_edge(start, end)
        numel = self.in_features * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters([(start, end)]), numel)

        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        numel = self.in_features * self.hidden_size + self.hidden_size
        self.assertEqual(self.dag.count_parameters([(start, "1")]), numel)
        numel += self.hidden_size * self.out_features + self.out_features
        self.assertEqual(self.dag.count_parameters([(start, "1"), ("1", end)]), numel)

    def test_evaluate(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_direct_edge(start, end)
        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )

        x = torch.rand((50, self.in_features), device=global_device())
        y = torch.rand((50, self.out_features), device=global_device()).argmax(axis=1)
        loss_fn = torch.nn.CrossEntropyLoss()
        actual_out = self.dag.forward(x)
        actual_loss = loss_fn(actual_out, y).item()
        acc, _, f1 = self.dag.evaluate(
            x, actual_out.argmax(axis=1), loss_fn, with_f1score=True
        )
        self.assertEqual(acc, 1.0)
        self.assertEqual(f1, 1.0)

        _, loss = self.dag.evaluate(x, y, loss_fn, with_f1score=False)
        self.assertEqual(actual_loss, loss)

        dag = GrowingDAG(
            in_features=self.in_features,
            out_features=1,
            neurons=self.hidden_size,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            default_layer_type="linear",
        )
        start, end = dag.root, dag.end
        dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        y = torch.rand((50, 1), device=global_device())
        loss_fn = torch.nn.MSELoss()
        actual_out = dag.forward(x)
        actual_loss = loss_fn(actual_out, y).item()
        acc, loss, f1 = dag.evaluate(x, y, loss_fn, with_f1score=True)
        self.assertEqual(acc, -1)
        self.assertEqual(f1, -1)
        self.assertEqual(loss, actual_loss)

    def test_evaluate_extended(self) -> None:
        start, end = self.dag.root, self.dag.end
        self.dag.add_direct_edge(start, end)
        self.dag.get_edge_module(start, end).optimal_delta_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            device=global_device(),
        )
        self.dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        self.dag.get_edge_module(start, "1").extended_output_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            device=global_device(),
        )
        self.dag.get_edge_module("1", end).extended_input_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.out_features,
            device=global_device(),
        )

        x = torch.rand((50, self.in_features), device=global_device())
        y = torch.rand((50, self.out_features), device=global_device()).argmax(axis=1)
        loss_fn = torch.nn.CrossEntropyLoss()
        actual_out, _ = self.dag.extended_forward(x)
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
            default_layer_type="linear",
        )
        start, end = dag.root, dag.end
        dag.get_edge_module(start, end).optimal_delta_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=1,
            device=global_device(),
        )
        dag.add_node_with_two_edges(
            start, "1", end, node_attributes=self.init_node_attributes
        )
        dag.get_edge_module(start, "1").extended_output_layer = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            device=global_device(),
        )
        dag.get_edge_module("1", end).extended_input_layer = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=1,
            device=global_device(),
        )
        y = torch.rand((50, 1), device=global_device())
        loss_fn = torch.nn.MSELoss()
        actual_out, _ = dag.extended_forward(x)
        actual_loss = loss_fn(actual_out, y).item()
        acc, loss, f1 = dag.evaluate_extended(x, y, loss_fn, with_f1score=True)  # type: ignore
        self.assertEqual(acc, -1)
        self.assertEqual(f1, -1)
        self.assertEqual(loss, actual_loss)

    def test_expansion_init(self) -> None:
        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                exp_type="random",  # type: ignore
            )

        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                exp_type=ExpansionType.NEW_EDGE,
            )
        with self.assertWarns(UserWarning):
            # When creating a new edge the expanding node argument is not required
            Expansion(
                self.dag,
                exp_type=ExpansionType.NEW_EDGE,
                previous_node=self.dag.root,
                next_node=self.dag.end,
                expanding_node="test",
            )
        expansion = Expansion(
            self.dag,
            exp_type=ExpansionType.NEW_EDGE,
            previous_node=self.dag.root,
            next_node=self.dag.end,
        )
        self.assertIs(self.dag, expansion.dag)

        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                exp_type=ExpansionType.NEW_NODE,
            )

        with self.assertRaises(ValueError):
            Expansion(
                self.dag,
                exp_type=ExpansionType.EXPANDED_NODE,
            )
        with self.assertWarns(UserWarning):
            # When expanding an existing node the previous and next nodes arguments are not required
            Expansion(
                self.dag,
                exp_type=ExpansionType.EXPANDED_NODE,
                expanding_node="test",
                previous_node=self.dag.root,
            )

    def test_expansion_new_edges(self) -> None:
        # Add new edge
        expansion = Expansion(
            self.dag,
            exp_type=ExpansionType.NEW_EDGE,
            previous_node=self.dag.root,
            next_node=self.dag.end,
        )
        self.assertEqual(expansion.new_edges, [(self.dag.root, self.dag.end)])
        expansion.expand()
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag.root, self.dag.end).weight)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag.root, self.dag.end).bias)
        )
        self.assertEqual(
            expansion.in_edges, [self.dag.get_edge_module(self.dag.root, self.dag.end)]
        )
        self.assertEqual(expansion.in_edges, expansion.out_edges)

        # Add new node
        expansion = Expansion(
            self.dag,
            exp_type=ExpansionType.NEW_NODE,
            expanding_node="test",
            previous_node=self.dag.root,
            next_node=self.dag.end,
            node_attributes=self.init_node_attributes,
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
        self.assertEqual(
            expansion.in_edges, [self.dag.get_edge_module(self.dag.root, "test")]
        )
        self.assertEqual(
            expansion.out_edges, [self.dag.get_edge_module("test", self.dag.end)]
        )

        # Expand existing node
        self.dag.add_node_with_two_edges(
            self.dag.root,
            "test",
            self.dag.end,
            node_attributes=self.init_node_attributes,
        )
        expansion = Expansion(
            self.dag,
            exp_type=ExpansionType.EXPANDED_NODE,
            expanding_node="test",
        )
        self.assertEqual(
            expansion.new_edges,
            [
                (self.dag.root, "test"),
                ("test", self.dag.end),
            ],
        )
        expansion.expand()
        self.assertIs(self.dag, expansion.dag)
        self.assertEqual(
            expansion.in_edges, self.dag.get_node_module("test").previous_modules
        )
        self.assertEqual(
            expansion.out_edges, self.dag.get_node_module("test").next_modules
        )

    def test_inter_merge_expansion_new_edges(self) -> None:
        dag = GrowingDAG(
            in_features=self.dag_conv.get_node_module(self.dag_conv.end).output_volume,
            out_features=self.out_features,
            neurons=self.hidden_size,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            default_layer_type="linear",
            name="dag-linear",
        )
        self.dag_conv.add_direct_edge(
            self.dag_conv.root,
            self.dag_conv.end,
            edge_attributes=self.default_edge_attributes,
        )
        self.dag_conv.get_node_module(self.dag_conv.end).add_next_module(
            dag.get_node_module(self.dag.root)
        )
        dag.get_node_module(self.dag.root).add_previous_module(
            self.dag_conv.get_node_module(self.dag_conv.end)
        )

        # Add new edge
        expansion = InterMergeExpansion(
            dag,
            exp_type=ExpansionType.NEW_EDGE,
            previous_node=dag.root,
            next_node=dag.end,
        )
        expansion.expand()
        self.assertEqual(
            expansion.new_edges,
            dag.get_edge_modules([(dag.root, dag.end)]),
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(dag.root, dag.end).weight)
        )
        self.assertFalse(torch.any(expansion.dag.get_edge_module(dag.root, dag.end).bias))
        self.assertEqual(
            expansion.previous_nodes, [dag.get_node_module(expansion.previous_node)]
        )
        self.assertEqual(expansion.next_nodes, [dag.get_node_module(expansion.next_node)])
        self.assertEqual(expansion.in_edges, [dag.get_edge_module(dag.root, dag.end)])
        self.assertEqual(expansion.in_edges, expansion.out_edges)
        self.assertEqual(
            expansion.create_mask(),
            {
                "nodes": [None, None],
                "edges": [dag.get_edge_module(dag.root, dag.end)._name],
            },
        )

        # Add new node
        expansion = InterMergeExpansion(
            self.dag_conv,
            exp_type=ExpansionType.NEW_NODE,
            expanding_node="test",
            previous_node=self.dag_conv.root,
            next_node=self.dag_conv.end,
            node_attributes=self.init_node_conv_attributes,
            edge_attributes=self.default_edge_attributes,
        )
        expansion.expand()
        self.assertEqual(
            expansion.new_edges,
            self.dag_conv.get_edge_modules(
                [
                    (self.dag_conv.root, "test"),
                    ("test", self.dag_conv.end),
                ]
            ),
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag_conv.root, "test").weight)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module(self.dag_conv.root, "test").bias)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module("test", self.dag_conv.end).weight)
        )
        self.assertFalse(
            torch.any(expansion.dag.get_edge_module("test", self.dag_conv.end).bias)
        )
        self.assertEqual(
            expansion.previous_nodes,
            [self.dag_conv.get_node_module(expansion.previous_node)],
        )
        self.assertEqual(
            expansion.next_nodes, [self.dag_conv.get_node_module(expansion.next_node)]
        )
        self.assertEqual(
            expansion.in_edges,
            [self.dag_conv.get_edge_module(self.dag_conv.root, "test")],
        )
        self.assertEqual(
            expansion.out_edges,
            [self.dag_conv.get_edge_module("test", self.dag_conv.end)],
        )
        self.assertEqual(
            expansion.create_mask(),
            {
                "nodes": ["test", None],
                "edges": [
                    self.dag_conv.get_edge_module(self.dag_conv.root, "test")._name,
                    self.dag_conv.get_edge_module("test", self.dag_conv.end)._name,
                ],
            },
        )

        # Expand existing node
        self.dag_conv.add_node_with_two_edges(
            self.dag_conv.root,
            "test",
            self.dag_conv.end,
            node_attributes=self.init_node_conv_attributes,
            edge_attributes=self.default_edge_attributes,
        )
        expansion = InterMergeExpansion(
            self.dag_conv,
            exp_type=ExpansionType.EXPANDED_NODE,
            expanding_node="test",
        )
        expansion.expand()
        self.assertTrue(self.dag_conv.is_node_candidate("test"))
        self.assertEqual(
            expansion.previous_nodes, [self.dag_conv.get_node_module(self.dag_conv.root)]
        )
        self.assertEqual(
            expansion.next_nodes, [self.dag_conv.get_node_module(self.dag_conv.end)]
        )
        self.assertEqual(
            expansion.new_edges,
            self.dag_conv.get_edge_modules(
                [
                    (self.dag_conv.root, "test"),
                    ("test", self.dag_conv.end),
                ]
            ),
        )
        self.assertEqual(
            expansion.in_edges, self.dag_conv.get_node_module("test").previous_modules
        )
        self.assertEqual(
            expansion.out_edges, self.dag_conv.get_node_module("test").next_modules
        )
        self.assertEqual(
            expansion.create_mask(),
            {
                "nodes": ["test", None],
                "edges": [
                    self.dag_conv.get_edge_module(self.dag_conv.root, "test")._name,
                    self.dag_conv.get_edge_module("test", self.dag_conv.end)._name,
                ],
            },
        )

        # Expansion with the first MergeGrowingModule
        expansion = InterMergeExpansion(
            self.dag_conv,
            exp_type=ExpansionType.EXPANDED_NODE,
            expanding_node=self.dag_conv.end,
            adjacent_expanding_node=dag.root,
        )
        expansion.expand()
        self.assertEqual(
            expansion.previous_nodes, [self.dag_conv.get_node_module(self.dag_conv.root)]
        )
        with self.assertWarns(UserWarning):
            # Node does not belong in the current dag. All external nodes are assumed to be non-candidate.
            self.assertEqual(expansion.next_nodes, [dag.get_node_module(dag.end)])
        self.assertEqual(
            expansion.new_edges,
            [
                self.dag_conv.get_edge_module(self.dag_conv.root, self.dag_conv.end),
                dag.get_edge_module(dag.root, dag.end),
            ],
        )
        self.assertEqual(
            expansion.in_edges,
            [self.dag_conv.get_edge_module(self.dag_conv.root, self.dag_conv.end)],
        )
        self.assertEqual(expansion.out_edges, [dag.get_edge_module(dag.root, dag.end)])
        self.assertEqual(
            expansion.create_mask(),
            {
                "nodes": [self.dag_conv.end, dag.root],
                "edges": [
                    self.dag_conv.get_edge_module(
                        self.dag_conv.root, self.dag_conv.end
                    )._name,
                    dag.get_edge_module(dag.root, dag.end)._name,
                ],
            },
        )

        self.dag_conv.toggle_node_candidate("test", False)
        self.assertEqual(
            expansion.previous_nodes,
            [
                self.dag_conv.get_node_module(self.dag_conv.root),
                self.dag_conv.get_node_module("test"),
            ],
        )
        self.assertEqual(
            expansion.new_edges,
            [
                self.dag_conv.get_edge_module(self.dag_conv.root, self.dag_conv.end),
                self.dag_conv.get_edge_module("test", self.dag_conv.end),
                dag.get_edge_module(dag.root, dag.end),
            ],
        )
        self.assertEqual(
            expansion.in_edges,
            [
                self.dag_conv.get_edge_module(self.dag_conv.root, self.dag_conv.end),
                self.dag_conv.get_edge_module("test", self.dag_conv.end),
            ],
        )
        self.assertEqual(
            expansion.create_mask(),
            {
                "nodes": [self.dag_conv.end, dag.root],
                "edges": [
                    self.dag_conv.get_edge_module(
                        self.dag_conv.root, self.dag_conv.end
                    )._name,
                    self.dag_conv.get_edge_module("test", self.dag_conv.end)._name,
                    dag.get_edge_module(dag.root, dag.end)._name,
                ],
            },
        )

        # Expansion with the second MergeGrowingModule
        dag.add_node_with_two_edges(
            dag.root,
            "test",
            dag.end,
            node_attributes=self.init_node_attributes,
        )
        dag.toggle_node_candidate("test", True)
        expansion = InterMergeExpansion(
            dag,
            exp_type=ExpansionType.EXPANDED_NODE,
            expanding_node=dag.root,
            adjacent_expanding_node=self.dag_conv.end,
        )
        expansion.expand()
        with self.assertWarns(UserWarning):
            # Node does not belong in the current dag. All external nodes are assumed to be non-candidate.
            self.assertEqual(
                expansion.previous_nodes,
                [
                    self.dag_conv.get_node_module(self.dag_conv.root),
                    self.dag_conv.get_node_module("test"),
                ],
            )
        self.assertEqual(expansion.next_nodes, [dag.get_node_module(dag.end)])
        self.assertEqual(
            expansion.new_edges,
            [
                self.dag_conv.get_edge_module(self.dag_conv.root, self.dag_conv.end),
                self.dag_conv.get_edge_module("test", self.dag_conv.end),
                dag.get_edge_module(dag.root, dag.end),
            ],
        )
        self.assertEqual(
            expansion.in_edges,
            [
                self.dag_conv.get_edge_module(self.dag_conv.root, self.dag_conv.end),
                self.dag_conv.get_edge_module("test", self.dag_conv.end),
            ],
        )
        self.assertEqual(expansion.out_edges, [dag.get_edge_module(dag.root, dag.end)])
        self.assertEqual(
            expansion.create_mask(),
            {
                "nodes": [dag.root, self.dag_conv.end],
                "edges": [
                    self.dag_conv.get_edge_module(
                        self.dag_conv.root, self.dag_conv.end
                    )._name,
                    self.dag_conv.get_edge_module("test", self.dag_conv.end)._name,
                    dag.get_edge_module(dag.root, dag.end)._name,
                ],
            },
        )

        dag.toggle_node_candidate("test", False)
        self.assertEqual(
            expansion.next_nodes,
            [dag.get_node_module(dag.end), dag.get_node_module("test")],
        )
        self.assertEqual(
            expansion.new_edges,
            [
                self.dag_conv.get_edge_module(self.dag_conv.root, self.dag_conv.end),
                self.dag_conv.get_edge_module("test", self.dag_conv.end),
                dag.get_edge_module(dag.root, dag.end),
                dag.get_edge_module(dag.root, "test"),
            ],
        )
        self.assertEqual(
            expansion.out_edges,
            [
                dag.get_edge_module(dag.root, dag.end),
                dag.get_edge_module(dag.root, "test"),
            ],
        )
        self.assertEqual(
            expansion.create_mask(),
            {
                "nodes": [dag.root, self.dag_conv.end],
                "edges": [
                    self.dag_conv.get_edge_module(
                        self.dag_conv.root, self.dag_conv.end
                    )._name,
                    self.dag_conv.get_edge_module("test", self.dag_conv.end)._name,
                    dag.get_edge_module(dag.root, dag.end)._name,
                    dag.get_edge_module(dag.root, "test")._name,
                ],
            },
        )

    def test_update_growth_history(self) -> None:
        expansion = Expansion(
            self.dag,
            exp_type=ExpansionType.NEW_NODE,
            expanding_node="1",
            previous_node=self.dag.root,
            next_node=self.dag.end,
            node_attributes=self.init_node_attributes,
        )
        expansion.expand()
        mock_global_step = -1

        expansion._Expansion__update_growth_history(  # type: ignore
            neurons_added=[(self.dag.root, "1"), ("1", self.dag.end)],
            current_step=mock_global_step,
        )

        for edge in self.dag.edges:
            self.assertIn(str(edge), expansion.growth_history[mock_global_step])
        self.assertEqual(
            expansion.growth_history[mock_global_step][str((self.dag.root, "1"))],
            2,
        )
        self.assertEqual(
            expansion.growth_history[mock_global_step][str(("1", self.dag.end))], 2
        )
        self.assertEqual(expansion.growth_history[mock_global_step]["1"], 0)

        expansion._Expansion__update_growth_history(nodes_added=["1", "2"], current_step=mock_global_step)  # type: ignore
        self.assertEqual(expansion.growth_history[mock_global_step]["1"], 2)
        self.assertNotIn("2", expansion.growth_history[mock_global_step])

        self.dag.add_direct_edge(self.dag.root, self.dag.end)
        expansion.update_growth_history(current_step=mock_global_step + 1)
        for edge in self.dag.edges:
            self.assertIn(str(edge), expansion.growth_history[mock_global_step + 1])
        for node in self.dag.nodes:
            self.assertIn(str(node), expansion.growth_history[mock_global_step + 1])
        self.assertEqual(expansion.growth_history[mock_global_step + 1]["1"], 2)
        self.assertEqual(expansion.growth_history[mock_global_step + 1][self.dag.root], 0)
        self.assertEqual(expansion.growth_history[mock_global_step + 1][self.dag.end], 0)
        self.assertEqual(
            expansion.growth_history[mock_global_step + 1][str((self.dag.root, "1"))], 2
        )
        self.assertEqual(
            expansion.growth_history[mock_global_step + 1][str(("1", self.dag.end))], 2
        )
        self.assertEqual(
            expansion.growth_history[mock_global_step + 1][
                str((self.dag.root, self.dag.end))
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
