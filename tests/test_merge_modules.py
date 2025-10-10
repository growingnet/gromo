import unittest

import torch

from gromo.containers.growing_dag import InterMergeExpansion
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    Conv2dMergeGrowingModule,
)
from gromo.modules.growing_module import GrowingModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import global_device


class TestMergeGrowingModules(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 4
        self.input_shape = (12, 12)
        self.device = global_device()

        self.in_channels = 3
        self.out_channels = 2
        self.kernel_size = (3, 3)
        self.out_features = 1
        self.loss_fn = torch.nn.MSELoss()

        self.conv_in_features = (
            self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        )
        self.out_dim = self.input_shape[0] - self.kernel_size[0] + 1
        self.linear_in_features = self.out_dim * self.out_dim * self.out_channels

        self.x = torch.rand(
            self.batch_size, self.in_channels, *self.input_shape, device=self.device
        )
        self.y = torch.rand(self.batch_size, self.out_features, device=self.device)

        self.conv = Conv2dGrowingModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            input_size=self.input_shape,
            name="conv",
        )
        self.conv_merge = Conv2dMergeGrowingModule(
            in_channels=self.out_channels,
            next_kernel_size=self.kernel_size,
            input_size=(self.out_dim, self.out_dim),
            input_volume=self.linear_in_features,
            post_merge_function=torch.nn.ReLU(),
            name="conv_merge",
        )
        self.flatten = torch.nn.Flatten()
        self.linear_merge = LinearMergeGrowingModule(
            in_features=self.linear_in_features, name="linear_merge"
        )
        self.linear = LinearGrowingModule(
            in_features=self.linear_in_features,
            out_features=self.out_features,
            name="linear",
        )
        self.layers = [self.conv, self.conv_merge, self.linear_merge, self.linear]

    def test_direct_connection(self):
        # Direct connection between Conv2dMergeGrowingModule and LinearMergeGrowingModule
        self.conv.next_module = self.conv_merge
        self.conv_merge.add_previous_module(self.conv)
        self.conv_merge.add_next_module(self.linear_merge)
        self.linear_merge.add_previous_module(self.conv_merge)
        self.linear_merge.add_next_module(self.linear)
        self.linear.previous_module = self.linear_merge

        # Assert tensor shapes
        self.assertEqual(
            self.conv.tensor_s._shape,
            (
                self.conv_in_features + self.conv.use_bias,
                self.conv_in_features + self.conv.use_bias,
            ),
        )
        self.assertEqual(
            self.conv.tensor_m._shape,
            (self.conv_in_features + self.conv.use_bias, self.out_channels),
        )

        self.assertEqual(self.conv_merge.total_in_features, self.conv_in_features + 1)
        self.assertEqual(
            self.conv_merge.tensor_s._shape,
            (
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
            ),
        )
        self.assertEqual(
            self.conv_merge.previous_tensor_s._shape,
            (self.conv_merge.total_in_features, self.conv_merge.total_in_features),
        )
        self.assertEqual(
            self.conv_merge.previous_tensor_m._shape,
            (self.conv_merge.total_in_features, self.out_channels),
        )

        self.assertEqual(self.linear_merge.total_in_features, 0)
        self.assertEqual(
            self.linear_merge.tensor_s._shape,
            (self.linear_in_features + 1, self.linear_in_features + 1),
        )
        self.assertIsNone(self.linear_merge.previous_tensor_s)
        self.assertIsNone(self.linear_merge.previous_tensor_m)

        self.assertEqual(
            self.linear.tensor_s._shape,
            (
                self.linear_in_features + self.linear.use_bias,
                self.linear_in_features + self.linear.use_bias,
            ),
        )
        self.assertEqual(
            self.linear.tensor_m._shape,
            (self.linear_in_features + self.linear.use_bias, self.out_features),
        )

        # Forward and backward pass
        for layer in self.layers:
            layer.init_computation()

        x_conv = self.conv(self.x)
        x_conv_merge = self.conv_merge(x_conv)
        x_flatten = self.flatten(x_conv_merge)
        x_linear_merge = self.linear_merge(x_flatten)
        out = self.linear(x_linear_merge)

        self.assertEqual(out.shape, (self.batch_size, self.out_features))

        loss = self.loss_fn(out, self.y)
        loss.backward()

        for layer in self.layers:
            layer.update_computation()

        # Compute optimal deltas
        for layer in self.layers:
            if isinstance(layer, LinearMergeGrowingModule):
                with self.assertRaises(AssertionError):
                    layer.compute_optimal_delta()
            else:
                layer.compute_optimal_delta()
                if isinstance(layer, GrowingModule):
                    self.assertIsNotNone(layer.optimal_delta_layer)

        # self.assertTrue(torch.all(self.conv_merge.activity == x_conv_merge))

        # Reset computation
        for layer in self.layers:
            layer.reset_computation()
            layer.delete_update()

    def test_linear_merge_with_multiple_inputs(self):
        linear_intercept = LinearGrowingModule(
            in_features=2,
            out_features=self.linear_in_features,
            post_layer_function=torch.nn.ReLU(),
            name="linear_intercept",
        )
        self.layers.append(linear_intercept)
        x_intercept = torch.rand(self.batch_size, 2, device=self.device)

        # LinearMergeGrowingModule with two inputs: Conv2dMergeGrowingModule and LinearGrowingModule
        self.conv.next_module = self.conv_merge
        self.conv_merge.add_previous_module(self.conv)
        self.conv_merge.add_next_module(self.linear_merge)

        self.linear_merge.add_previous_module(self.conv_merge)
        self.linear_merge.add_previous_module(linear_intercept)
        self.linear_merge.add_next_module(self.linear)
        self.linear.previous_module = self.linear_merge

        # Assert tensor shapes
        self.assertEqual(
            self.linear_merge.total_in_features,
            linear_intercept.in_features + linear_intercept.use_bias,
        )
        self.assertEqual(
            self.linear_merge.tensor_s._shape,
            (self.linear_in_features + 1, self.linear_in_features + 1),
        )
        self.assertEqual(
            self.linear_merge.previous_tensor_s._shape,
            (self.linear_merge.total_in_features, self.linear_merge.total_in_features),
        )
        self.assertEqual(
            self.linear_merge.previous_tensor_m._shape,
            ((self.linear_merge.total_in_features, self.linear_in_features)),
        )

        self.assertEqual(
            linear_intercept.tensor_s._shape,
            (
                linear_intercept.in_features + linear_intercept.use_bias,
                linear_intercept.in_features + linear_intercept.use_bias,
            ),
        )
        self.assertEqual(
            linear_intercept.tensor_m._shape,
            (
                linear_intercept.in_features + linear_intercept.use_bias,
                self.linear_in_features,
            ),
        )

        self.assertEqual(
            self.linear.tensor_s._shape,
            (
                self.linear_in_features + self.linear.use_bias,
                self.linear_in_features + self.linear.use_bias,
            ),
        )
        self.assertEqual(
            self.linear.tensor_m._shape,
            (self.linear_in_features + self.linear.use_bias, self.out_features),
        )

        # Forward and backward pass
        for layer in self.layers:
            layer.init_computation()

        x_conv = self.conv(self.x)
        x_conv_merge = self.conv_merge(x_conv)
        x_flatten = self.flatten(x_conv_merge)
        x_intercept = linear_intercept(x_intercept)
        x_linear_merge = self.linear_merge(x_flatten + x_intercept)
        out = self.linear(x_linear_merge)

        self.assertEqual(out.shape, (self.batch_size, self.out_features))

        loss = self.loss_fn(out, self.y)
        loss.backward()

        for layer in self.layers:
            layer.update_computation()

        # Compute optimal deltas
        for layer in self.layers:
            layer.compute_optimal_delta()
            if isinstance(layer, GrowingModule):
                self.assertIsNotNone(layer.optimal_delta_layer)

        # Assert activities
        self.assertTrue(torch.all(self.linear_merge.input == x_flatten + x_intercept))
        self.assertTrue(torch.all(self.linear_merge.input == x_flatten + x_intercept))

        # Reset computation
        for layer in self.layers:
            layer.reset_computation()
            layer.delete_update()

    def test_conv2d_to_linear_and_conv2d_merge(self):
        # Create modules
        conv_early_exit = Conv2dGrowingModule(
            in_channels=self.out_channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            name="conv_early_exit",
            # input_size=self.input_shape,
            # post_layer_function=torch.nn.Sequential(
            #     torch.nn.Flatten(start_dim=1),
            #     torch.nn.Softmax(dim=1),
            # )
        )
        self.layers.append(conv_early_exit)
        y_early_exit = torch.rand(self.batch_size, 1, device=self.device)

        # Connect Conv2dMergeGrowingModule to both LinearMergeGrowingModule and Conv2dGrowingModule
        self.conv.next_module = self.conv_merge
        self.conv_merge.add_previous_module(self.conv)
        self.conv_merge.add_next_module(self.linear_merge)
        self.conv_merge.add_next_module(conv_early_exit)
        self.linear_merge.add_previous_module(self.conv_merge)
        self.linear_merge.add_next_module(self.linear)
        self.linear.previous_module = self.linear_merge

        # Assert tensor shapes
        self.assertEqual(self.conv_merge.total_in_features, self.conv_in_features + 1)
        self.assertEqual(
            self.conv_merge.tensor_s._shape,
            (
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
            ),
        )
        self.assertEqual(
            self.conv_merge.previous_tensor_s._shape,
            (self.conv_merge.total_in_features, self.conv_merge.total_in_features),
        )
        self.assertEqual(
            self.conv_merge.previous_tensor_m._shape,
            (self.conv_merge.total_in_features, self.out_channels),
        )

        self.assertEqual(self.linear_merge.total_in_features, 0)
        self.assertEqual(
            self.linear_merge.tensor_s._shape,
            (self.linear_in_features + 1, self.linear_in_features + 1),
        )
        self.assertIsNone(self.linear_merge.previous_tensor_s)
        self.assertIsNone(self.linear_merge.previous_tensor_m)

        self.assertEqual(
            conv_early_exit.tensor_s._shape,
            (
                self.out_channels * self.kernel_size[0] * self.kernel_size[1]
                + conv_early_exit.use_bias,
                self.out_channels * self.kernel_size[0] * self.kernel_size[1]
                + conv_early_exit.use_bias,
            ),
        )
        self.assertEqual(
            conv_early_exit.tensor_m._shape,
            (
                self.out_channels * self.kernel_size[0] * self.kernel_size[1]
                + conv_early_exit.use_bias,
                1,
            ),
        )

        # Forward and backward pass
        for layer in self.layers:
            layer.init_computation()

        x_conv = self.conv(self.x)
        x_conv_merge = self.conv_merge(x_conv)
        out_early_exit = conv_early_exit(x_conv_merge)
        x_flatten = self.flatten(x_conv_merge)
        x_linear_merge = self.linear_merge(x_flatten)
        out = self.linear(x_linear_merge)

        self.assertEqual(out.shape, (self.batch_size, self.out_features))
        self.assertEqual(
            out_early_exit.shape,
            (self.batch_size, 1, conv_early_exit.out_width, conv_early_exit.out_height),
        )

        out_early_exit = torch.softmax(torch.flatten(out_early_exit, start_dim=1), dim=1)
        loss_early_exit = self.loss_fn(out_early_exit, y_early_exit)

        loss = self.loss_fn(out, self.y)
        loss = loss + loss_early_exit
        loss.backward()

        for layer in self.layers:
            layer.update_computation()

        # Compute optimal deltas
        for layer in self.layers:
            if isinstance(layer, LinearMergeGrowingModule):
                with self.assertRaises(AssertionError):
                    layer.compute_optimal_delta()
            else:
                layer.compute_optimal_delta()
                if isinstance(layer, GrowingModule):
                    self.assertIsNotNone(layer.optimal_delta_layer)

        # Assert activities
        self.assertTrue(torch.all(self.conv_merge.activity == conv_early_exit.input))

        # Reset computation
        for layer in self.layers:
            layer.reset_computation()
            layer.delete_update()

    def test_conv2d_merge_to_conv2d_merge_with_pooling(self):
        # Create modules
        out_dim_after_pooling = (self.out_dim - self.kernel_size[0]) // self.kernel_size[
            0
        ] + 1
        new_linear_in_features = (
            out_dim_after_pooling * out_dim_after_pooling * self.out_channels
        )
        pooling = torch.nn.AvgPool2d(kernel_size=self.kernel_size)
        self.conv_merge.post_merge_function = torch.nn.Sequential(
            torch.nn.ReLU(),
            pooling,
        )
        conv_merge_after_pooling = Conv2dMergeGrowingModule(
            in_channels=self.out_channels,
            next_kernel_size=(1, 1),
            input_size=(out_dim_after_pooling, out_dim_after_pooling),
            name="conv2d_merge_after_pooling",
        )
        new_linear_merge = LinearMergeGrowingModule(
            in_features=new_linear_in_features, name="new_linear_merge"
        )
        new_linear = LinearGrowingModule(
            in_features=new_linear_in_features,
            out_features=self.out_features,
            name="new_linear",
        )

        self.layers.remove(self.linear_merge)
        self.layers.remove(self.linear)
        self.layers.append(conv_merge_after_pooling)
        self.layers.append(new_linear_merge)
        self.layers.append(new_linear)

        # Direct connection between Conv2dMergeGrowingModule and Conv2dMergeGrowingModule
        self.conv.next_module = self.conv_merge
        self.conv_merge.add_previous_module(self.conv)
        self.conv_merge.add_next_module(conv_merge_after_pooling)
        conv_merge_after_pooling.add_previous_module(self.conv_merge)
        conv_merge_after_pooling.add_next_module(new_linear_merge)
        new_linear_merge.add_previous_module(conv_merge_after_pooling)
        new_linear_merge.add_next_module(new_linear)
        new_linear.previous_module = new_linear_merge

        # Assert tensor shapes
        self.assertEqual(
            self.conv.tensor_s._shape,
            (
                self.conv_in_features + self.conv.use_bias,
                self.conv_in_features + self.conv.use_bias,
            ),
        )
        self.assertEqual(
            self.conv.tensor_m._shape,
            (self.conv_in_features + self.conv.use_bias, self.out_channels),
        )

        self.assertEqual(self.conv_merge.total_in_features, self.conv_in_features + 1)
        self.assertEqual(
            self.conv_merge.tensor_s._shape,
            (
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
            ),
        )
        self.assertEqual(
            self.conv_merge.previous_tensor_s._shape,
            (self.conv_merge.total_in_features, self.conv_merge.total_in_features),
        )
        self.assertEqual(
            self.conv_merge.previous_tensor_m._shape,
            (self.conv_merge.total_in_features, self.out_channels),
        )

        self.assertEqual(conv_merge_after_pooling.total_in_features, 0)
        self.assertEqual(
            conv_merge_after_pooling.tensor_s._shape,
            (
                self.out_channels
                * conv_merge_after_pooling.kernel_size[0]
                * conv_merge_after_pooling.kernel_size[1]
                + 1,
                self.out_channels
                * conv_merge_after_pooling.kernel_size[0]
                * conv_merge_after_pooling.kernel_size[1]
                + 1,
            ),
        )
        self.assertIsNone(conv_merge_after_pooling.previous_tensor_s)
        self.assertIsNone(conv_merge_after_pooling.previous_tensor_m)

        self.assertEqual(new_linear_merge.total_in_features, 0)
        self.assertEqual(
            new_linear_merge.tensor_s._shape,
            (new_linear_in_features + 1, new_linear_in_features + 1),
        )
        self.assertIsNone(new_linear_merge.previous_tensor_s)
        self.assertIsNone(new_linear_merge.previous_tensor_m)

        self.assertEqual(
            new_linear.tensor_s._shape,
            (
                new_linear_in_features + new_linear.use_bias,
                new_linear_in_features + new_linear.use_bias,
            ),
        )
        self.assertEqual(
            new_linear.tensor_m._shape,
            (new_linear_in_features + new_linear.use_bias, self.out_features),
        )

        # ---------- Average Pooling ------------

        # Forward and backward pass
        for layer in self.layers:
            layer.init_computation()

        x_conv = self.conv(self.x)
        x_conv_merge = self.conv_merge(x_conv)
        x_conv_merge_after_pooling = conv_merge_after_pooling(x_conv_merge)
        x_flatten = self.flatten(x_conv_merge_after_pooling)
        x_linear_merge = new_linear_merge(x_flatten)
        out = new_linear(x_linear_merge)

        self.assertEqual(out.shape, (self.batch_size, self.out_features))

        loss = self.loss_fn(out, self.y)
        loss.backward()

        for layer in self.layers:
            layer.update_computation()

        # Compute optimal deltas
        for layer in self.layers:
            if isinstance(layer, LinearMergeGrowingModule) or (
                layer is conv_merge_after_pooling
            ):
                with self.assertRaises(AssertionError):
                    layer.compute_optimal_delta()
            else:
                layer.compute_optimal_delta()
                if isinstance(layer, GrowingModule):
                    self.assertIsNotNone(layer.optimal_delta_layer)

        # Reset computation
        for layer in self.layers:
            layer.reset_computation()
            layer.delete_update()

        # ---------- Max Pooling ------------

        pooling = torch.nn.MaxPool2d(kernel_size=self.kernel_size)
        self.conv_merge.post_merge_function = torch.nn.Sequential(
            torch.nn.ReLU(),
            pooling,
        )

        # Forward and backward pass
        for layer in self.layers:
            layer.init_computation()

        x_conv = self.conv(self.x)
        x_conv_merge = self.conv_merge(x_conv)
        x_conv_merge_after_pooling = conv_merge_after_pooling(x_conv_merge)
        x_flatten = self.flatten(x_conv_merge_after_pooling)
        x_linear_merge = new_linear_merge(x_flatten)
        out = new_linear(x_linear_merge)

        self.assertEqual(out.shape, (self.batch_size, self.out_features))

        loss = self.loss_fn(out, self.y)
        loss.backward()

        for layer in self.layers:
            layer.update_computation()

        # Compute optimal deltas
        for layer in self.layers:
            if isinstance(layer, LinearMergeGrowingModule) or (
                layer is conv_merge_after_pooling
            ):
                with self.assertRaises(AssertionError):
                    layer.compute_optimal_delta()
            else:
                layer.compute_optimal_delta()
                if isinstance(layer, GrowingModule):
                    self.assertIsNotNone(layer.optimal_delta_layer)

        # Reset computation
        for layer in self.layers:
            layer.reset_computation()
            layer.delete_update()

    def test_container_to_container_with_expansion(self):
        # Create Containers
        hidden_channels = 10
        node_attributes = {
            "type": "convolution",
            "size": 5,
            "kernel_size": self.kernel_size,
            "shape": self.input_shape,
        }
        edge_attributes = {"kernel_size": self.kernel_size}
        dag1 = GrowingGraphNetwork(
            in_features=self.in_channels,
            out_features=hidden_channels,
            loss_fn=self.loss_fn,
            neurons=20,
            input_shape=self.input_shape,
            layer_type="convolution",
            name="dag1",
        )
        dag1.dag.add_node_with_two_edges(
            dag1.dag.root,
            "1",
            dag1.dag.end,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
        )
        node_in_features = int(
            dag1.dag.get_edge_module("1", dag1.dag.end).in_features + 1
        )
        pooling = torch.nn.AvgPool2d(kernel_size=self.kernel_size)
        out_dim_after_pooling = (
            self.input_shape[0] - self.kernel_size[0]
        ) // self.kernel_size[0] + 1
        node_attributes["shape"] = (out_dim_after_pooling, out_dim_after_pooling)
        dag2 = GrowingGraphNetwork(
            in_features=hidden_channels,
            out_features=self.out_channels,
            loss_fn=self.loss_fn,
            neurons=10,
            input_shape=(out_dim_after_pooling, out_dim_after_pooling),
            layer_type="convolution",
            name="dag2",
        )
        dag2.dag.add_node_with_two_edges(
            dag2.dag.root,
            "1",
            dag2.dag.end,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
        )

        start_of_dag1 = dag1.dag.get_node_module(dag1.dag.root)
        end_of_dag1 = dag1.dag.get_node_module(dag1.dag.end)
        start_of_dag2 = dag2.dag.get_node_module(dag2.dag.root)
        end_of_dag2 = dag2.dag.get_node_module(dag2.dag.end)
        new_linear_in_features = end_of_dag2.output_volume

        # Add pooling as post_merge_function
        end_of_dag1.post_merge_function = pooling

        new_linear_merge = LinearMergeGrowingModule(
            in_features=new_linear_in_features, name="new_linear_merge"
        )
        new_linear = LinearGrowingModule(
            in_features=new_linear_in_features,
            out_features=self.out_features,
            name="new_linear",
        )
        self.layers = [dag1, dag2, new_linear_merge, new_linear]

        # Direct connection between GrowingGraphNetwork and GrowingGraphNetwork
        end_of_dag1.add_next_module(start_of_dag2)
        start_of_dag2.add_previous_module(end_of_dag1)
        end_of_dag2.add_next_module(new_linear_merge)
        new_linear_merge.add_previous_module(end_of_dag2)
        new_linear_merge.add_next_module(new_linear)
        new_linear.previous_module = new_linear_merge

        # Assert tensor shapes
        self.assertEqual(start_of_dag1.total_in_features, 0)
        self.assertEqual(
            start_of_dag1.tensor_s._shape,
            (
                self.conv_in_features + self.conv.use_bias,
                self.conv_in_features + self.conv.use_bias,
            ),
        )
        self.assertIsNone(start_of_dag1.previous_tensor_s)
        self.assertIsNone(start_of_dag1.previous_tensor_m)

        self.assertEqual(
            end_of_dag1.total_in_features, self.conv_in_features + 1 + node_in_features
        )
        self.assertEqual(
            end_of_dag1.tensor_s._shape,
            (
                hidden_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
                hidden_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
            ),
        )
        self.assertEqual(
            end_of_dag1.previous_tensor_s._shape,
            (end_of_dag1.total_in_features, end_of_dag1.total_in_features),
        )
        self.assertEqual(
            end_of_dag1.previous_tensor_m._shape,
            (end_of_dag1.total_in_features, hidden_channels),
        )

        self.assertEqual(start_of_dag2.total_in_features, 0)
        self.assertEqual(
            start_of_dag2.tensor_s._shape,
            (
                hidden_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
                hidden_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
            ),
        )
        self.assertIsNone(start_of_dag2.previous_tensor_s)
        self.assertIsNone(start_of_dag2.previous_tensor_m)

        self.assertEqual(
            end_of_dag2.total_in_features,
            hidden_channels * self.kernel_size[0] * self.kernel_size[1]
            + 1
            + node_in_features,
        )
        self.assertEqual(
            end_of_dag2.tensor_s._shape,
            (
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
                self.out_channels * self.kernel_size[0] * self.kernel_size[1] + 1,
            ),
        )
        self.assertEqual(
            end_of_dag2.previous_tensor_s._shape,
            (end_of_dag2.total_in_features, end_of_dag2.total_in_features),
        )
        self.assertEqual(
            end_of_dag2.previous_tensor_m._shape,
            (end_of_dag2.total_in_features, self.out_channels),
        )

        self.assertEqual(new_linear_merge.total_in_features, 0)
        self.assertEqual(
            new_linear_merge.tensor_s._shape,
            (new_linear_in_features + 1, new_linear_in_features + 1),
        )
        self.assertIsNone(new_linear_merge.previous_tensor_s)
        self.assertIsNone(new_linear_merge.previous_tensor_m)

        self.assertEqual(
            new_linear.tensor_s._shape,
            (
                new_linear_in_features + new_linear.use_bias,
                new_linear_in_features + new_linear.use_bias,
            ),
        )
        self.assertEqual(
            new_linear.tensor_m._shape,
            (new_linear_in_features + new_linear.use_bias, self.out_features),
        )

        # Forward and backward pass
        for layer in self.layers:
            layer.init_computation()
        start_of_dag2.init_computation()

        x_dag1 = dag1(self.x)
        x_dag2 = dag2(x_dag1)
        x_flatten = self.flatten(x_dag2)
        x_linear_merge = new_linear_merge(x_flatten)
        out = new_linear(x_linear_merge)

        self.assertEqual(out.shape, (self.batch_size, self.out_features))

        loss = self.loss_fn(out, self.y)
        loss.backward()

        for layer in self.layers:
            layer.update_computation()

        # Compute optimal deltas
        for layer in self.layers:
            if isinstance(layer, LinearMergeGrowingModule):
                with self.assertRaises(AssertionError):
                    layer.compute_optimal_delta()
            else:
                layer.compute_optimal_delta()
                if isinstance(layer, GrowingModule):
                    self.assertIsNotNone(layer.optimal_delta_layer)
                elif isinstance(layer, GrowingGraphNetwork):
                    for edge_module in layer.dag.get_all_edge_modules():
                        self.assertIsNotNone(edge_module.optimal_delta_layer)

        # Retrieve bottleneck
        with torch.no_grad():
            # Retrieve bottleneck of next node modules
            bottleneck = {
                end_of_dag2._name: end_of_dag2.projected_v_goal().clone().detach(),
                "1": dag2.dag.get_node_module("1").projected_v_goal().clone().detach(),
            }
            # Retrieve post-activity of previous node modules
            input_B = {
                start_of_dag1._name: start_of_dag1.activity.clone().detach(),
                "1": dag1.dag.get_node_module("1").activity.clone().detach(),
            }

        # You should grow the node that has pooling as a post_merge_function
        expansion = InterMergeExpansion(
            dag=dag1.dag,
            type="expanded node",
            expanding_node=dag1.dag.end,
            adjacent_expanding_node=dag2.dag.root,
        )
        actions = [expansion]

        with self.assertWarns(UserWarning):
            # All external nodes are assumed to be non-candidate
            dag1.execute_expansions(
                actions=actions,
                bottleneck=bottleneck,
                input_B=input_B,
                amplitude_factor=False,
                evaluate=False,
                verbose=False,
            )
        expansion.metrics["loss_val"] = 1

        dag1.choose_growth_best_action(
            options=actions,
            verbose=False,
        )
        dag2.chosen_action = dag1.chosen_action
        dag1.apply_change()
        dag2.apply_change()

        # Reset computation
        for layer in self.layers:
            layer.reset_computation()
            layer.delete_update()

        hidden_channels += dag1.neurons
        self.assertEqual(
            dag1.dag.get_edge_module(dag1.dag.root, dag1.dag.end).out_channels,
            hidden_channels,
        )
        self.assertEqual(
            dag1.dag.get_edge_module("1", dag1.dag.end).out_channels,
            hidden_channels,
        )
        self.assertEqual(
            dag1.dag.get_node_module(dag1.dag.end).in_channels, hidden_channels
        )
        self.assertEqual(dag1.dag.nodes[dag1.dag.end]["size"], hidden_channels)
        self.assertEqual(
            dag2.dag.get_edge_module(dag2.dag.root, dag2.dag.end).in_channels,
            hidden_channels,
        )
        self.assertEqual(
            dag2.dag.get_edge_module(dag2.dag.root, "1").in_channels,
            hidden_channels,
        )
        self.assertEqual(
            dag2.dag.get_node_module(dag2.dag.root).in_channels, hidden_channels
        )
        self.assertEqual(dag2.dag.nodes[dag2.dag.root]["size"], hidden_channels)
        self.assertEqual(dag1.out_features, hidden_channels)
        self.assertEqual(dag2.in_features, hidden_channels)
        self.assertEqual(dag1.dag.out_features, hidden_channels)
        self.assertEqual(dag2.dag.in_features, hidden_channels)

        dag1.init_computation()
        x_dag1 = dag1(self.x)
        x_dag2 = dag2(x_dag1)
        x_flatten = self.flatten(x_dag2)
        x_linear_merge = new_linear_merge(x_flatten)
        out = new_linear(x_linear_merge)

        self.assertEqual(out.shape, (self.batch_size, self.out_features))
        self.assertEqual(
            end_of_dag1.input.shape,
            (self.batch_size, hidden_channels, *self.input_shape),
        )
        self.assertEqual(
            x_dag1.shape,
            (
                self.batch_size,
                hidden_channels,
                out_dim_after_pooling,
                out_dim_after_pooling,
            ),
        )

    def test_linear_merge_to_conv_merge(self):
        # Direct connection between LinearMergeGrowingModule and Conv2dMergeGrowingModule
        # TODO:
        pass


if __name__ == "__main__":
    unittest.main()
