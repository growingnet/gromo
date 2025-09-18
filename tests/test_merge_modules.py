import unittest

import torch

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
        self.input_shape = (9, 9)
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
        )
        self.conv_merge = Conv2dMergeGrowingModule(
            in_channels=self.out_channels,
            next_kernel_size=self.kernel_size,
            input_size=(self.out_dim, self.out_dim),
            input_volume=self.linear_in_features,
            post_merge_function=torch.nn.ReLU(),
        )
        self.flatten = torch.nn.Flatten()
        self.linear_merge = LinearMergeGrowingModule(in_features=self.linear_in_features)
        self.linear = LinearGrowingModule(
            in_features=self.linear_in_features,
            out_features=self.out_features,
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

    def test_linear_merge_to_conv_merge(self):
        # Direct connection between LinearMergeGrowingModule and Conv2dMergeGrowingModule
        # TODO:
        pass


if __name__ == "__main__":
    unittest.main()
