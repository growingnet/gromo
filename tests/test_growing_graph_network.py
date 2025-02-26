import copy
import unittest

import torch

from gromo.containers.growing_dag import GrowingDAG
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.utils.utils import global_device


class TestGrowingGraphNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.in_features = 5
        self.out_features = 2
        self.batch_size = 8
        self.net = GrowingGraphNetwork(
            in_features=self.in_features,
            out_features=self.out_features,
        )
        self.net.dag.add_node_with_two_edges(
            "start",
            "1",
            "end",
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

        self.generations = self.net.dag.define_next_generations()

    def test_init_empty_graph(self) -> None:
        self.net.init_empty_graph()
        self.assertEqual(len(self.net.dag.nodes), 2)
        self.assertEqual(len(self.net.dag.edges), 0)
        self.assertIn("start", self.net.dag.nodes)
        self.assertIn("end", self.net.dag.nodes)
        self.assertEqual(self.net.dag.in_degree("start"), 0)
        self.assertEqual(self.net.dag.out_degree("start"), 0)
        self.assertEqual(self.net.dag.in_degree("end"), 0)
        self.assertEqual(self.net.dag.out_degree("end"), 0)
        self.assertEqual(self.net.dag.nodes["start"]["size"], self.in_features)
        self.assertEqual(self.net.dag.nodes["end"]["size"], self.out_features)
        self.assertEqual(self.net.dag.nodes["start"]["type"], "linear")
        self.assertEqual(self.net.dag.nodes["end"]["type"], "linear")
        self.assertFalse(self.net.dag.nodes["end"]["use_batch_norm"])

    def test_growth_history_step(self) -> None:
        self.net.growth_history_step(
            neurons_added=[("start", "1"), ("1", "end")],
            # neurons_updated=[("start", "end")],
        )

        for edge in self.net.dag.edges:
            self.assertIn(str(edge), self.net.growth_history[self.net.global_step])
        self.assertEqual(
            self.net.growth_history[self.net.global_step][str(("start", "1"))], 2
        )
        self.assertEqual(
            self.net.growth_history[self.net.global_step][str(("1", "end"))], 2
        )
        self.assertEqual(self.net.growth_history[self.net.global_step]["1"], 0)

        self.net.growth_history_step(nodes_added=["1", "2"])
        self.assertEqual(self.net.growth_history[self.net.global_step]["1"], 2)
        self.assertNotIn("2", self.net.growth_history[self.net.global_step])

    # def test_setup_train_datasets(self) -> None:
    #     dataset = torch.utils.data.TensorDataset(self.x, self.y)
    #     dev_len = int(len(self.x) / 3)
    #     train_len = len(self.x) - dev_len * 2
    #     global_device_type = global_device().type

    #     X_train, Y_train, X_dev, Y_dev, X_val, Y_val = self.net.setup_train_datasets(
    #         dataset, torch.Generator()
    #     )

    #     self.assertEqual(X_train.shape, (train_len, self.in_features))
    #     self.assertEqual(Y_train.shape, (train_len,))
    #     self.assertEqual(X_train.device.type, global_device_type)
    #     self.assertEqual(Y_train.device.type, global_device_type)
    #     self.assertEqual(X_dev.shape, (dev_len, self.in_features))
    #     self.assertEqual(Y_dev.shape, (dev_len,))
    #     self.assertEqual(X_dev.device.type, global_device_type)
    #     self.assertEqual(Y_dev.device.type, global_device_type)
    #     self.assertEqual(X_val.shape, (dev_len, self.in_features))
    #     self.assertEqual(Y_val.shape, (dev_len,))
    #     self.assertEqual(X_val.device.type, global_device_type)
    #     self.assertEqual(Y_val.device.type, global_device_type)

    def test_expand_node(self) -> None:
        node = "1"
        prev_nodes = ["start"]
        next_nodes = ["end"]
        with self.assertWarns(UserWarning):
            self.net.expand_node(
                node,
                prev_nodes,
                next_nodes,
                self.bottleneck,
                self.input_B,
                self.x_test,
                self.y_test,
                verbose=False,
            )

        self.assertEqual(self.net.dag.nodes[node]["size"], self.net.neurons * 2)
        self.assertEqual(
            self.net.dag.get_edge_module("start", node).in_features, self.in_features
        )
        self.assertEqual(
            self.net.dag.get_edge_module("start", node).out_features, self.net.neurons * 2
        )
        self.assertEqual(
            self.net.dag.get_edge_module(node, "end").in_features, self.net.neurons * 2
        )
        self.assertEqual(
            self.net.dag.get_edge_module(node, "end").out_features, self.out_features
        )

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
            self.x_test,
            self.y_test,
            amplitude_factor=False,
            verbose=False,
        )

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

    def test_find_amplitude_factor(self) -> None:
        pass
        # # self.net.dag.get_edge_module("start", "1").optimal_delta_layer = torch.nn.Linear(
        # #     in_features=self.in_features,
        # #     out_features=self.net.neurons,
        # #     device=global_device(),
        # # )
        # # self.net.dag.get_edge_module("start", "1").extended_output_layer = torch.nn.Linear(
        # #     in_features=self.in_features,
        # #     out_features=self.net.neurons,
        # #     device=global_device(),
        # # )
        # edge_module = self.net.dag.get_edge_module("1", "end")
        # edge_module.weight.data = torch.zeros((self.out_features, self.net.neurons), device=global_device())
        # edge_module.optimal_delta_layer = torch.nn.Linear(
        #     in_features=self.net.neurons,
        #     out_features=self.out_features,
        #     device=global_device(),
        # )
        # # edge_module.optimal_delta_layer.weight.data *= 100
        # # edge_module.optimal_delta_layer.bias.data += 10
        # print(f"{edge_module.optimal_delta_layer.weight=}")
        # # self.net.dag.get_edge_module("1", "end").extended_input_layer = torch.nn.Linear(
        # #     in_features=self.net.neurons,
        # #     out_features=self.out_features,
        # #     device=global_device(),
        # # )

        # node_module = self.net.dag.get_node_module("end")
        # pred = torch.argmax(self.net(self.x), dim=1)
        # extended_pred = torch.argmax(self.net.extended_forward(self.x), dim=1)
        # print(f"{pred=}")
        # print(f"{extended_pred=}")

        # factor = self.net.find_input_amplitude_factor(self.x, self.y, node_module)
        # self.assertNotEqual(factor, 0.0)
        # self.assertNotEqual(factor, 1.0)

        # factor = self.net.find_input_amplitude_factor(self.x, pred, node_module)
        # self.assertEqual(factor, 0.0)

        # # factor = self.net.find_input_amplitude_factor(self.x, extended_pred, node_module)
        # # self.assertEqual(factor, 1.0)

    def test_inter_training(self) -> None:
        pass

    def test_execute_expansions(self) -> None:
        self.net.execute_expansions(
            self.generations,
            self.bottleneck,
            self.input_B,
            self.x,
            self.y,
            self.x,
            self.y,
            self.x_test,
            self.y_test,
            amplitude_factor=False,
        )

        for gen in self.generations:
            self.assertIsNotNone(gen.get("loss_train"))
            self.assertIsNotNone(gen.get("loss_dev"))
            self.assertIsNotNone(gen.get("loss_val"))
            self.assertIsNotNone(gen.get("acc_train"))
            self.assertIsNotNone(gen.get("acc_dev"))
            self.assertIsNotNone(gen.get("acc_val"))

            self.assertEqual(gen.get("loss_train"), gen.get("loss_dev"))
            self.assertEqual(gen.get("acc_train"), gen.get("acc_dev"))

            self.assertIsNotNone(gen.get("nb_params"))
            self.assertIsNotNone(gen.get("BIC"))

            self.assertIsNotNone(gen.get("dag"))
            self.assertIsInstance(gen.get("dag"), GrowingDAG)
            self.assertIsNotNone(gen.get("growth_history"))
            self.assertIsInstance(gen.get("growth_history"), dict)

    def test_calculate_bottleneck(self) -> None:
        bottleneck, inputB = self.net.dag.calculate_bottleneck(
            self.generations, self.x, self.y
        )

        self.assertIsNotNone(bottleneck.get("end"))
        self.assertEqual(bottleneck["end"].shape, (self.batch_size, self.out_features))

        self.assertIsNotNone(bottleneck.get("1"))
        self.assertEqual(bottleneck["1"].shape, (self.batch_size, self.net.neurons))

        self.assertIsNotNone(inputB.get("start"))
        self.assertEqual(inputB["start"].shape, (self.batch_size, self.in_features))

        self.assertIsNotNone(inputB.get("1"))
        self.assertEqual(inputB["1"].shape, (self.batch_size, self.net.neurons))

    def test_restrict_action_space(self) -> None:
        self.assertEqual(len(self.generations), 4)

        gens = self.net.restrict_action_space(self.generations, "end")
        self.assertEqual(len(gens), 3)

        gens = self.net.restrict_action_space(self.generations, "1")
        self.assertEqual(len(gens), 2)

        gens = self.net.restrict_action_space(self.generations, "start")
        self.assertEqual(len(gens), 0)

    def test_grow_step(self) -> None:
        pass

    def test_choose_growth_best_option(self) -> None:
        options = self.net.dag.define_next_generations()
        with self.assertRaises(KeyError):
            self.net.choose_growth_best_action(options, use_bic=False)
        with self.assertRaises(KeyError):
            self.net.choose_growth_best_action(options, use_bic=True)

        min_value, min_value_bic = torch.inf, torch.inf
        for i, opt in enumerate(options):
            opt["dag"] = None
            opt["growth_history"] = i
            opt["loss_train"] = None
            opt["loss_dev"] = None
            opt["acc_train"] = None
            opt["acc_dev"] = None
            opt["acc_val"] = None
            opt["loss_val"] = torch.rand(1)
            opt["nb_params"] = torch.randint(10, 1000, (1,))
            opt["BIC"] = torch.randint(10, 1000, (1,))
            if opt["loss_val"] < min_value:
                min_value = opt["loss_val"]
                min_index = i
            if opt["BIC"] < min_value_bic:
                min_value_bic = opt["BIC"]
                min_index_bic = i

        self.net.choose_growth_best_action(options, use_bic=False)
        self.assertEqual(self.net.growth_history, min_index)

        self.net.choose_growth_best_action(options, use_bic=True)
        self.assertEqual(self.net.growth_history, min_index_bic)


if __name__ == "__main__":
    unittest.main()
