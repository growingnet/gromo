import copy
import unittest

import torch
from torch.nn.functional import one_hot

from gromo.containers.growing_dag import Expansion, GrowingDAG, InterMergeExpansion
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.utils.utils import global_device


try:
    from tests.torch_unittest import TorchTestCase  # type: ignore
    from tests.unittest_tools import unittest_parametrize  # type: ignore
except ImportError:
    from torch_unittest import TorchTestCase  # type: ignore
    from unittest_tools import unittest_parametrize  # type: ignore


class TestGrowingGraphNetwork(TorchTestCase):
    def setUp(self) -> None:
        self.in_features = 3
        self.out_features = 2
        self.batch_size = 8
        self.neurons = 5

        # Linear Graph
        self.net = GrowingGraphNetwork(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.neurons,
            loss_fn=torch.nn.CrossEntropyLoss(),
            layer_type="linear",
        )
        self.net.dag.add_node_with_two_edges(
            self.net.dag.root,
            "1",
            self.net.dag.end,
            node_attributes={"type": "linear", "size": self.net.neurons},
        )
        self.x = torch.rand(
            (self.batch_size, self.in_features),
            device=global_device(),
            requires_grad=True,
        )
        self.y = torch.randint(
            0, self.out_features, (self.batch_size,), device=global_device()
        )
        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x, self.y), batch_size=self.batch_size
        )

        self.x_test = torch.rand(
            (self.batch_size, self.in_features), device=global_device()
        )
        self.y_test = torch.randint(
            0, self.out_features, (self.batch_size,), device=global_device()
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x_test, self.y_test), batch_size=4
        )

        self.bottleneck = {
            self.net.dag.end: torch.rand(
                (self.batch_size, self.out_features), device=global_device()
            ),
            "1": torch.rand((self.batch_size, self.net.neurons), device=global_device()),
        }
        self.input_B = {
            self.net.dag.root: torch.rand(
                (self.batch_size, self.in_features), device=global_device()
            ),
            "1": torch.rand((self.batch_size, self.net.neurons), device=global_device()),
        }

        self.actions = self.net.dag.define_next_actions()

        # Convolutional Graph
        self.input_shape = (9, 9)
        self.kernel_size = (3, 3)
        self.net_conv = GrowingGraphNetwork(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.neurons,
            loss_fn=torch.nn.CrossEntropyLoss(),
            input_shape=self.input_shape,
            layer_type="convolution",
        )
        self.net_conv.dag.add_node_with_two_edges(
            self.net_conv.dag.root,
            "1",
            self.net_conv.dag.end,
            node_attributes={
                "type": "convolution",
                "size": self.net.neurons,
                "kernel_size": self.kernel_size,
                "shape": self.input_shape,
            },
            edge_attributes={"kernel_size": self.kernel_size},
        )
        self.net_conv.dag.remove_direct_edge(
            self.net_conv.dag.root, self.net_conv.dag.end
        )

        self.bottleneck_conv = {
            self.net_conv.dag.end: torch.rand(
                (self.batch_size, self.out_features, *self.input_shape),
                device=global_device(),
            ),
            "1": torch.rand(
                (self.batch_size, self.net_conv.neurons, *self.input_shape),
                device=global_device(),
            ),
        }
        self.input_B_conv = {
            self.net_conv.dag.root: torch.rand(
                (self.batch_size, self.in_features, *self.input_shape),
                device=global_device(),
            ),
            "1": torch.rand(
                (self.batch_size, self.net_conv.neurons, *self.input_shape),
                device=global_device(),
            ),
        }

    def test_init_empty_graph(self) -> None:
        self.net.init_empty_graph()
        self.assertEqual(len(self.net.dag.nodes), 2)
        self.assertEqual(len(self.net.dag.edges), 0)
        self.assertIn(self.net.dag.root, self.net.dag.nodes)
        self.assertIn(self.net.dag.end, self.net.dag.nodes)
        self.assertEqual(self.net.dag.in_degree(self.net.dag.root), 0)
        self.assertEqual(self.net.dag.out_degree(self.net.dag.root), 0)
        self.assertEqual(self.net.dag.in_degree(self.net.dag.end), 0)
        self.assertEqual(self.net.dag.out_degree(self.net.dag.end), 0)
        self.assertEqual(self.net.dag.nodes[self.net.dag.root]["size"], self.in_features)
        self.assertEqual(self.net.dag.nodes[self.net.dag.end]["size"], self.out_features)
        self.assertEqual(self.net.dag.nodes[self.net.dag.root]["type"], "linear")
        self.assertEqual(self.net.dag.nodes[self.net.dag.end]["type"], "linear")
        self.assertFalse(self.net.dag.nodes[self.net.dag.end]["use_batch_norm"])

    def test_expand_node(self) -> None:
        node = "1"
        prev_nodes = self.net.dag.root
        next_nodes = self.net.dag.end
        expansion = Expansion(
            self.net.dag,
            "new node",
            expanding_node=node,
            previous_node=prev_nodes,
            next_node=next_nodes,
        )
        self.net.expand_node(
            expansion=expansion,
            bottlenecks=self.bottleneck,
            activities=self.input_B,
            verbose=False,
        )

        self.assertEqual(self.net.dag.nodes[node]["size"], self.net.neurons)
        self.assertEqual(
            self.net.dag.get_edge_module(prev_nodes, node).in_features, self.in_features
        )
        self.assertEqual(
            self.net.dag.get_edge_module(prev_nodes, node).out_features, self.net.neurons
        )
        self.assertEqual(
            self.net.dag.get_edge_module(node, next_nodes).in_features, self.net.neurons
        )
        self.assertEqual(
            self.net.dag.get_edge_module(node, next_nodes).out_features, self.out_features
        )

    def test_update_edge_weights(self) -> None:
        # Linear Graph
        prev_node = self.net.dag.root
        next_node = self.net.dag.end
        expansion = Expansion(
            self.net.dag, "new edge", previous_node=prev_node, next_node=next_node
        )
        expansion.dag.add_direct_edge(prev_node, next_node)
        edge_module = expansion.dag.get_edge_module(prev_node, next_node)
        prev_weight = copy.deepcopy(edge_module.weight)

        self.net.update_edge_weights(
            expansion=expansion,
            bottlenecks=self.bottleneck,
            activities=self.input_B,
            verbose=False,
        )
        edge_module.apply_change(scaling_factor=1, apply_previous=False)

        self.assertEqual(len(self.net.dag.edges), 3)
        self.assertIn((prev_node, next_node), self.net.dag.edges)
        self.assertEqual(self.net.dag.nodes[prev_node]["size"], self.in_features)
        self.assertEqual(self.net.dag.nodes[next_node]["size"], self.out_features)
        self.assertEqual(self.net.dag.out_degree(prev_node), 2)
        self.assertEqual(self.net.dag.in_degree(next_node), 2)
        self.assertEqual(
            self.net.dag.get_edge_module(prev_node, next_node).in_features,
            self.in_features,
        )
        self.assertEqual(
            self.net.dag.get_edge_module(prev_node, next_node).out_features,
            self.out_features,
        )

        # activity = torch.matmul(self.x, edge_module.weight.T) + edge_module.bias
        self.assertTrue(torch.all(edge_module.weight != prev_weight))
        self.assertEqual(edge_module.weight.shape, (self.out_features, self.in_features))
        self.assertEqual(edge_module.bias.shape, (self.out_features,))

        # Convolutional Graph
        prev_node = self.net_conv.dag.root
        next_node = self.net_conv.dag.end
        expansion = Expansion(
            self.net_conv.dag, "new edge", previous_node=prev_node, next_node=next_node
        )
        expansion.dag.add_direct_edge(
            prev_node, next_node, edge_attributes={"kernel_size": self.kernel_size}
        )
        edge_module = expansion.dag.get_edge_module(prev_node, next_node)
        prev_weight = copy.deepcopy(edge_module.weight)

        self.net_conv.update_edge_weights(
            expansion=expansion,
            bottlenecks=self.bottleneck_conv,
            activities=self.input_B_conv,
            verbose=False,
        )
        edge_module.apply_change(scaling_factor=1, apply_previous=False)

        self.assertEqual(len(self.net_conv.dag.edges), 3)
        self.assertIn((prev_node, next_node), self.net_conv.dag.edges)
        self.assertEqual(self.net_conv.dag.nodes[prev_node]["size"], self.in_features)
        self.assertEqual(self.net_conv.dag.nodes[next_node]["size"], self.out_features)
        self.assertEqual(self.net_conv.dag.out_degree(prev_node), 2)
        self.assertEqual(self.net_conv.dag.in_degree(next_node), 2)
        self.assertEqual(
            self.net_conv.dag.get_edge_module(prev_node, next_node).in_channels,
            self.in_features,
        )
        self.assertEqual(
            self.net_conv.dag.get_edge_module(prev_node, next_node).out_channels,
            self.out_features,
        )
        self.assertTrue(torch.all(edge_module.weight != prev_weight))
        self.assertEqual(
            edge_module.weight.shape,
            (self.out_features, self.in_features, *self.kernel_size),
        )
        self.assertEqual(edge_module.bias.shape, (self.out_features,))

    def test_find_amplitude_factor(self) -> None:
        self.net.dag.get_edge_module("1", self.net.dag.end).extended_input_layer = (
            torch.nn.Linear(
                in_features=self.net.neurons,
                out_features=self.out_features,
                device=global_device(),
            )
        )
        self.net.dag.get_edge_module(self.net.dag.root, "1").extended_output_layer = (
            torch.nn.Linear(
                in_features=self.in_features,
                out_features=self.net.neurons,
                device=global_device(),
            )
        )
        self.net.loss_fn = torch.nn.L1Loss()

        pred = one_hot(
            torch.argmax(self.net(self.x), dim=1), num_classes=self.out_features
        )
        extended_pred = one_hot(
            torch.argmax(self.net.extended_forward(self.x)[0], dim=1),
            num_classes=self.out_features,
        )

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.x, one_hot(self.y, num_classes=self.out_features)
            )
        )
        factor = self.net.find_amplitude_factor(dataloader, mask={"nodes": "1"})
        self.assertNotEqual(factor, 1.0)

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x, extended_pred)
        )
        factor = self.net.find_amplitude_factor(dataloader, mask={"nodes": "1"})
        if torch.all(pred != extended_pred):
            self.assertEqual(factor, 1.0)

    def test_inter_training(self) -> None:
        pass

    @unittest_parametrize(({"evaluate": True}, {"evaluate": False}))
    def test_execute_expansions(self, evaluate: bool) -> None:
        self.net.execute_expansions(
            self.actions,
            self.bottleneck,
            self.input_B,
            amplitude_factor=False,
            evaluate=evaluate,
            train_dataloader=self.dataloader,
            dev_dataloader=self.dataloader,
            val_dataloader=self.test_dataloader,
        )

        for expansion in self.actions:
            if evaluate:
                self.assertIsNotNone(expansion.metrics.get("loss_train"))
                self.assertIsNotNone(expansion.metrics.get("loss_dev"))
                self.assertIsNotNone(expansion.metrics.get("loss_val"))
                self.assertIsNotNone(expansion.metrics.get("acc_train"))
                self.assertIsNotNone(expansion.metrics.get("acc_dev"))
                self.assertIsNotNone(expansion.metrics.get("acc_val"))

                self.assertEqual(
                    expansion.metrics.get("loss_train"), expansion.metrics.get("loss_dev")
                )
                self.assertEqual(
                    expansion.metrics.get("acc_train"), expansion.metrics.get("acc_dev")
                )

                self.assertIsNotNone(expansion.metrics.get("nb_params"))
                self.assertIsNotNone(expansion.metrics.get("BIC"))
            else:
                self.assertIsNone(expansion.metrics.get("loss_train"))
                self.assertIsNone(expansion.metrics.get("loss_dev"))
                self.assertIsNone(expansion.metrics.get("loss_val"))
                self.assertIsNone(expansion.metrics.get("acc_train"))
                self.assertIsNone(expansion.metrics.get("acc_dev"))
                self.assertIsNone(expansion.metrics.get("acc_val"))
                self.assertIsNone(expansion.metrics.get("nb_params"))
                self.assertIsNone(expansion.metrics.get("BIC"))

            self.assertIsNotNone(expansion.metrics.get("scaling_factor"))
            self.assertIsNotNone(expansion.metrics.get("loss_bott"))

            self.assertIsNotNone(expansion.dag)
            self.assertIsInstance(expansion.dag, GrowingDAG)
            self.assertIsNotNone(expansion.growth_history)
            self.assertIsInstance(expansion.growth_history, dict)

            expansion.metrics.clear()

    def test_calculate_bottleneck(self) -> None:
        bottleneck, inputB = self.net.dag.calculate_bottleneck(
            self.actions, self.dataloader
        )

        self.assertIsNotNone(bottleneck.get(self.net.dag.end))
        self.assertEqual(
            bottleneck[self.net.dag.end].shape, (self.batch_size, self.out_features)
        )

        self.assertIsNotNone(bottleneck.get("1"))
        self.assertEqual(bottleneck["1"].shape, (self.batch_size, self.net.neurons))

        self.assertIsNotNone(inputB.get(self.net.dag.root))
        self.assertEqual(
            inputB[self.net.dag.root].shape, (self.batch_size, self.in_features)
        )

        self.assertIsNotNone(inputB.get("1"))
        self.assertEqual(inputB["1"].shape, (self.batch_size, self.net.neurons))

    def test_restrict_action_space(self) -> None:
        self.assertEqual(len(self.actions), 4)
        self.actions.append(
            InterMergeExpansion(
                self.net.dag,
                type="expanded node",
                expanding_node=self.net.dag.end,
                adjacent_expanding_node="test",
            )
        )

        # Not specifing chosen_inputs or chosen_outputs should raise a warning and return all actions
        with self.assertWarns(UserWarning):
            gens = self.net.restrict_action_space(self.actions)
        self.assertListEqual(gens, self.actions)

        # Restricting based on outputs
        gens = self.net.restrict_action_space(
            self.actions, chosen_outputs=[self.net.dag.end]
        )
        self.assertEqual(len(gens), 4)

        gens = self.net.restrict_action_space(self.actions, chosen_outputs=["1"])
        self.assertEqual(len(gens), 2)

        gens = self.net.restrict_action_space(
            self.actions, chosen_outputs=[self.net.dag.root]
        )
        self.assertEqual(len(gens), 0)

        # Restricting based on both inputs and outputs should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.net.restrict_action_space(
                self.actions,
                chosen_outputs=[self.net.dag.end],
                chosen_inputs=[self.net.dag.root],
            )

        # Restricting based on inputs
        gens = self.net.restrict_action_space(
            self.actions, chosen_inputs=[self.net.dag.root]
        )
        self.assertEqual(len(gens), 3)

        gens = self.net.restrict_action_space(self.actions, chosen_inputs=["1"])
        self.assertEqual(len(gens), 2)

        gens = self.net.restrict_action_space(
            self.actions, chosen_inputs=[self.net.dag.end]
        )
        self.assertEqual(len(gens), 0)

    @unittest_parametrize(({"use_bic": True}, {"use_bic": False}))
    def test_choose_growth_best_option(self, use_bic: bool) -> None:
        options = self.net.dag.define_next_actions()
        with self.assertRaises(KeyError):
            self.net.choose_growth_best_action(options, use_bic=use_bic)

        min_value = torch.inf
        for i, opt in enumerate(options):
            opt.expand()
            opt.growth_history = i
            opt.metrics["scaling_factor"] = 1
            opt.metrics["loss_train"] = None
            opt.metrics["loss_dev"] = None
            opt.metrics["acc_train"] = None
            opt.metrics["acc_dev"] = None
            opt.metrics["acc_val"] = None
            opt.metrics["loss_val"] = torch.rand(1)
            opt.metrics["nb_params"] = torch.randint(10, 1000, (1,))
            opt.metrics["BIC"] = torch.randint(10, 1000, (1,))
            if use_bic:
                if opt.metrics["BIC"] < min_value:
                    min_value = opt.metrics["BIC"]
                    min_index = i
            else:
                if opt.metrics["loss_val"] < min_value:
                    min_value = opt.metrics["loss_val"]
                    min_index = i

        if options[min_index].type != "new edge":
            node_module = self.net.dag.get_node_module(options[min_index].expanding_node)
            for module in node_module.previous_modules:
                weight = torch.rand(
                    (self.neurons, module.in_features), device=module.device
                )
                bias = torch.rand(self.neurons, device=module.device)
                module.extended_output_layer = module.layer_of_tensor(
                    weight=weight, bias=bias
                )
            for module in node_module.next_modules:
                weight = torch.rand(
                    (module.out_features, self.neurons), device=module.device
                )
                bias = torch.zeros(module.out_features, device=module.device)
                module.extended_input_layer = module.layer_of_tensor(
                    weight=weight, bias=bias
                )

        self.net.choose_growth_best_action(options, use_bic=use_bic)

        # self.assertEqual(self.net.growth_history, min_index)
        self.assertIsNotNone(self.net.chosen_action)
        self.assertIs(self.net.chosen_action, options[min_index])
        self.assertIsNotNone(
            edge_module.optimal_delta_layer
            for edge_module in self.net.dag.get_all_edge_modules()
        )
        if options[min_index].type != "new edge":
            self.assertIsNotNone(
                edge_module.extended_output_layer
                for edge_module in node_module.previous_modules
            )
            self.assertIsNotNone(
                edge_module.extended_input_layer
                for edge_module in node_module.next_modules
            )

    def test_clean_graph_with_chosen_action_and_apply_changes(self) -> None:
        start, end = self.net_conv.dag.root, self.net_conv.dag.end
        options = self.net_conv.dag.define_next_actions()
        for opt in options:
            opt.metrics["scaling_factor"] = 1
            opt.expand()
        self.assertIn("2@_a", self.net_conv.dag)
        self.assertIn("2@_b", self.net_conv.dag)

        # Expansion in the dag
        self.net_conv.chosen_action = next(
            opt for opt in options if opt.expanding_node == "2@_a"
        )
        self.net_conv.clean_graph_with_chosen_action(options)
        self.net_conv.apply_change()
        self.assertNotIn("2@_b", self.net_conv.dag.nodes)
        self.assertNotIn((start, end), self.net_conv.dag.edges)
        self.assertIsNone(
            self.net_conv.dag.get_edge_module(start, "1").extended_output_layer
        )
        self.assertIsNone(
            self.net_conv.dag.get_edge_module("1", end).extended_input_layer
        )

        # Expansion in non-related dag
        self.net.chosen_action = self.net_conv.chosen_action
        self.net.clean_graph_with_chosen_action(options)
        self.net.apply_change()

        # Expansion in the connection between the dags
        self.net.delete_update()
        self.net_conv.delete_update()
        start_linear = self.net.dag.root
        start_linear_module = self.net.dag.get_node_module(start_linear)
        end_conv_module = self.net_conv.dag.get_node_module(end)
        start_linear_module.in_features = 162
        end_conv_module.add_next_module(start_linear_module)
        start_linear_module.add_previous_module(end_conv_module)

        options = self.net_conv.dag.define_next_actions(expand_end=True)
        for opt in options:
            opt.metrics["scaling_factor"] = 1
            opt.expand()
        self.net_conv.dag.get_edge_module("1", end).extended_output_layer = (
            torch.nn.Conv2d(
                in_channels=self.net_conv.neurons,
                out_channels=1,
                kernel_size=self.kernel_size,
                device=self.net_conv.device,
            )
        )
        self.net.dag.get_edge_module(start_linear, "1").extended_input_layer = (
            torch.nn.Linear(
                in_features=1,
                out_features=self.net_conv.neurons,
                bias=False,
                device=self.net_conv.device,
            )
        )

        # First dag
        self.net_conv.chosen_action = next(
            opt for opt in options if isinstance(opt, InterMergeExpansion)
        )
        self.net_conv.clean_graph_with_chosen_action(options)
        self.assertIsNotNone(
            self.net_conv.dag.get_edge_module("1", end).extended_output_layer
        )
        self.assertNotIn("2@_a", self.net_conv.dag.nodes)
        self.assertNotIn("2@_b", self.net_conv.dag.nodes)
        self.assertNotIn((start, end), self.net_conv.dag.edges)
        self.assertIsNone(
            self.net_conv.dag.get_edge_module(start, "1").extended_output_layer
        )
        self.assertIsNone(
            self.net_conv.dag.get_edge_module("1", end).extended_input_layer
        )
        self.net_conv.apply_change()
        self.assertEqual(end_conv_module.in_features, self.out_features + 1)

        # Second dag
        self.net.chosen_action = self.net_conv.chosen_action
        self.net.clean_graph_with_chosen_action(options)
        self.assertIsNotNone(
            self.net.dag.get_edge_module(start_linear, "1").extended_input_layer
        )
        self.net.apply_change()
        self.assertEqual(start_linear_module.in_features, self.out_features + 1)

    def test_numeric_accuracy_when_expanding_around_pooling(self) -> None:
        # import random
        # torch.manual_seed(0)
        # random.seed(0)
        start_conv, end_conv = self.net_conv.dag.root, self.net_conv.dag.end
        self.net_conv.dag.remove_node("1")
        self.net_conv.dag.add_direct_edge(
            start_conv, end_conv, edge_attributes={"kernel_size": self.kernel_size}
        )
        end_conv_module = self.net_conv.dag.get_node_module(end_conv)
        end_conv_module.post_merge_function = torch.nn.Sequential(
            torch.nn.SELU(),
            torch.nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.net_conv.neuron_lrate = 1e-2
        self.net_conv.neuron_epochs = 2000

        net_linear = GrowingGraphNetwork(
            in_features=end_conv_module.out_channels,
            out_features=self.out_features,
            neurons=self.neurons,
            loss_fn=torch.nn.CrossEntropyLoss(),
            layer_type="linear",
            name="lin",
        )
        start_linear, end_linear = net_linear.dag.root, net_linear.dag.end
        net_linear.dag.add_direct_edge(start_linear, end_linear, zero_weights=True)
        start_linear_module = net_linear.dag.get_node_module(start_linear)

        end_conv_module.add_next_module(start_linear_module)
        start_linear_module.add_previous_module(end_conv_module)

        expansion = InterMergeExpansion(
            dag=self.net_conv.dag,
            type="expanded node",
            expanding_node=end_conv,
            adjacent_expanding_node=start_linear,
        )

        conv_guide = torch.nn.Conv2d(
            self.in_features,
            out_channels=self.neurons,
            kernel_size=self.kernel_size,
            padding="same",
            device=global_device(),
        )
        lin_guide = torch.nn.Linear(
            in_features=self.neurons,
            out_features=self.out_features,
            device=global_device(),
        )
        activation = torch.nn.SELU()
        pooling = torch.nn.AdaptiveAvgPool2d(output_size=1)
        flatten = torch.nn.Flatten()
        model_guide = torch.nn.Sequential(
            conv_guide,
            activation,
            pooling,
            flatten,
            lin_guide,
        )

        x = torch.rand(
            (self.batch_size, self.in_features, *self.input_shape), device=global_device()
        )
        with torch.no_grad():
            desired_output = model_guide(x)

        bottleneck = {
            end_linear: desired_output,
        }
        activity = {
            start_conv: x,
        }

        self.net_conv.expand_node(
            expansion=expansion,
            bottlenecks=bottleneck,
            activities=activity,
            verbose=False,
        )

        layer_alpha = self.net_conv.dag.get_edge_module(
            start_conv, end_conv
        ).extended_output_layer
        layer_omega = net_linear.dag.get_edge_module(
            start_linear, end_linear
        ).extended_input_layer
        self.assertIsNotNone(layer_alpha)
        self.assertIsNotNone(layer_alpha.weight)
        self.assertIsNotNone(layer_alpha.bias)
        self.assertIsNotNone(layer_omega)
        self.assertIsNotNone(layer_omega.weight)
        self.assertIsNone(layer_omega.bias)

        output = layer_omega(flatten(pooling(activation(layer_alpha(x)))))
        self.assertAllClose(desired_output, output, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
