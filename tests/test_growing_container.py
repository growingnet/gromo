import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_mlp import GrowingMLP, Perceptron
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)


# Create synthetic data
def create_synthetic_data(
    num_samples=20,
    in_features=(3, 32, 32),
    out_features=(1,),
    batch_size=1,
):
    input_data = torch.randn(num_samples, *in_features)
    output_data = torch.randn(num_samples, *out_features)
    dataset = TensorDataset(input_data, output_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def gather_statistics(dataloader, model, loss):
    model.init_computation()
    for i, (x, y) in enumerate(dataloader):
        model.zero_grad()
        loss(model(x), y).backward()
        model.update_computation()


# Create a dummy GrowingContainer to test the base class method
class DummyGrowingContainer(GrowingContainer):
    def __init__(self, in_features, out_features):
        super().__init__(in_features=in_features, out_features=out_features)
        # Add some GrowingModule instances
        self.layer1 = LinearGrowingModule(
            in_features=in_features, out_features=4, name="layer1"
        )
        self.layer2 = LinearGrowingModule(
            in_features=4, out_features=out_features, name="layer2"
        )
        self.set_growing_layers()

    def set_growing_layers(self):
        self._growing_layers = [self.layer1, self.layer2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))

    @property
    def first_order_improvement(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TestGrowingContainer(unittest.TestCase):
    """
    Test the GrowingContainer class. We only test the function calls to
    the growing layers, as the correctness of the computations should be
    tested in the respective growing layer tests.
    """

    def setUp(self):
        # Create synthetic data
        self.in_features = 2
        self.out_features = 1
        self.num_samples = 20
        self.batch_size = 4
        self.dataloader = create_synthetic_data(
            self.num_samples, (self.in_features,), (self.out_features,), self.batch_size
        )

        # Create a simple perceptron model
        self.hidden_features = 4
        self.model = GrowingMLP(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_features,
            number_hidden_layers=3,
            device=torch.device("cpu"),
        )
        self.loss = nn.MSELoss()

    def test_init_computation(self):
        self.model.init_computation()
        for layer in self.model._growing_layers:
            # store input
            self.assertTrue(
                layer.store_input,
                "init_computation was not called on the growing layer",
            )

            # store pre-activity
            self.assertTrue(
                layer.store_pre_activity,
                "init_computation was not called on the growing layer",
            )

            # Check tensors: only tensor_s and tensor_m should be initialized with non-None values.
            # tensor_m_prev and cross_covariance should be None since their shape is not known yet.
            for tensor_name in ["tensor_s", "tensor_m"]:
                self.assertIsNotNone(
                    getattr(layer, tensor_name)._tensor,
                    f"init_computation was not called on the growing layer for {tensor_name}",
                )

    def test_update_computation(self):
        self.model.init_computation()
        for i, (x, y) in enumerate(self.dataloader):
            self.model.zero_grad()
            loss = self.loss(self.model(x), y)
            loss.backward()
            self.model.update_computation()

            for layer in self.model._growing_layers:
                # check number of samples in the tensor statistics
                for tensor_name in ["tensor_s", "tensor_m"]:
                    self.assertEqual(
                        getattr(layer, tensor_name).samples,
                        (i + 1) * self.batch_size,
                        f"update_computation was not called on the growing layer for {tensor_name}",
                    )
                if layer.previous_module is not None:
                    for tensor_name in [
                        "tensor_m_prev",
                        "cross_covariance",
                        "tensor_s_growth",
                    ]:
                        self.assertEqual(
                            getattr(layer, tensor_name).samples,
                            (i + 1) * self.batch_size,
                            f"update_computation was not called on the growing layer for {tensor_name}",
                        )

    def test_reset_computation(self):
        self.model.reset_computation()

        for layer in self.model._growing_layers:
            # store input
            self.assertFalse(
                layer.store_input,
                "reset_computation was not called on the growing layer",
            )

            # store pre-activity
            self.assertFalse(
                layer.store_pre_activity,
                "reset_computation was not called on the growing layer",
            )

            # Check tensors
            for tensor_name in [
                "tensor_s",
                "tensor_m",
                "tensor_m_prev",
                "cross_covariance",
                "tensor_s_growth",
            ]:
                self.assertIsNone(
                    getattr(layer, tensor_name)._tensor,
                    f"reset_computation was not called on the growing layer for {tensor_name}",
                )

    def test_compute_optimal_delta(self):
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_delta()

        for layer in self.model._growing_layers:
            # Check if the optimal updates are computed
            self.assertTrue(
                hasattr(layer, "optimal_delta_layer"),
                "compute_optimal_updates was not called on the growing layer",
            )
            self.assertTrue(
                hasattr(layer, "parameter_update_decrease"),
                "compute_optimal_updates was not called on the growing layer",
            )

    def test_compute_optimal_updates(self):
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()

        for layer in self.model._growing_layers:
            # Check if the optimal updates are computed
            self.assertTrue(
                hasattr(layer, "optimal_delta_layer"),
                "compute_optimal_updates was not called on the growing layer",
            )
            self.assertTrue(
                hasattr(layer, "parameter_update_decrease"),
                "compute_optimal_updates was not called on the growing layer",
            )
            if layer.previous_module is not None:
                self.assertTrue(
                    hasattr(layer, "extended_input_layer"),
                    "compute_optimal_updates was not called on the growing layer",
                )
                self.assertTrue(
                    hasattr(layer.previous_module, "extended_output_layer"),
                    "compute_optimal_updates was not called on the growing layer",
                )

    def check_has_no_update(self, layer):
        self.assertIsNone(
            layer.optimal_delta_layer,
            "select_best_update was not called on the growing layer",
        )
        self.assertIsNone(
            layer.parameter_update_decrease,
            "select_best_update was not called on the growing layer",
        )
        if layer.previous_module is not None:
            self.assertIsNone(
                layer.extended_input_layer,
                "select_best_update was not called on the growing layer",
            )
            self.assertIsNone(
                layer.previous_module.extended_output_layer,
                "select_best_update was not called on the growing layer",
            )

    def check_has_update(self, layer):
        self.assertIsNotNone(
            layer.optimal_delta_layer,
            "select_best_update was not called on the growing layer",
        )
        self.assertIsNotNone(
            layer.parameter_update_decrease,
            "select_best_update was not called on the growing layer",
        )
        if layer.previous_module is not None:
            self.assertIsNotNone(
                layer.extended_input_layer,
                "select_best_update was not called on the growing layer",
            )
            self.assertIsNotNone(
                layer.previous_module.extended_output_layer,
                "select_best_update was not called on the growing layer",
            )

    def check_update_selection(self, model, layer_index=None):
        self.assertIsNotNone(model.currently_updated_layer_index, "No layer to update")
        if layer_index is not None:
            self.assertEqual(
                model.currently_updated_layer_index,
                layer_index,
                "The selected layer index is incorrect",
            )

        # Check if the optimal updates are computed
        for i, layer in enumerate(model._growing_layers):
            if i != model.currently_updated_layer_index:
                self.check_has_no_update(layer)
            else:
                self.check_has_update(layer)

    def test_select_best_update(self):
        # computing the optimal updates
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()
        self.assertIsNone(
            self.model.currently_updated_layer_index,
            f"There should be no layer to update. "
            f"Currently updated layer index: {self.model.currently_updated_layer_index}",
        )

        max_v = 0
        max_i = -1
        for i, layer in enumerate(self.model._growing_layers):
            if layer.first_order_improvement > max_v:
                max_v = layer.first_order_improvement
                max_i = i

        # selecting the best update
        self.model.select_best_update()
        self.check_update_selection(self.model, layer_index=max_i)  # only one layer

    def test_select_update(self):
        # computing the optimal updates
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()
        self.assertIsNone(
            self.model.currently_updated_layer_index,
            f"There should be no layer to update. "
            f"Currently updated layer index: {self.model.currently_updated_layer_index}",
        )

        # selecting the first update
        layer_index = 0
        self.model.select_update(layer_index=layer_index)
        self.check_update_selection(self.model, layer_index=layer_index)

    def test_apply_change(self):
        gather_statistics(self.dataloader, self.model, self.loss)
        self.model.compute_optimal_updates()
        self.model.select_best_update()
        self.model.currently_updated_layer.scaling_factor = 1.0
        self.model.apply_change()
        self.assertIsNone(self.model.currently_updated_layer_index, "No layer to update")

    def test_number_of_parameters(self):
        theoretical_number_of_param = (
            self.in_features * self.hidden_features
            + self.hidden_features
            + self.hidden_features * self.out_features
            + self.out_features
        )

        model = Perceptron(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_feature=self.hidden_features,
            device=torch.device("cpu"),
        )

        self.assertEqual(
            model.number_of_parameters(),
            theoretical_number_of_param,
            "Number of parameters is incorrect",
        )

    def test_update_size(self):
        class TestContainer(GrowingContainer):
            def __init__(self, in_features, out_features):
                super().__init__(in_features=in_features, out_features=out_features)
                self.linear1 = LinearGrowingModule(
                    in_features=self.in_features, out_features=self.out_features
                )
                self.linear2 = LinearGrowingModule(
                    in_features=self.in_features, out_features=self.out_features
                )
                self.merge = LinearMergeGrowingModule(
                    in_features=self.out_features,
                    previous_modules=[self.linear1, self.linear2],
                )
                self.linear1.next_module = self.merge
                self.linear2.next_module = self.merge
                self.set_growing_layers()

            def set_growing_layers(self):
                self._growing_layers = [self.linear1, self.linear2, self.merge]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x1 = self.linear1(x)
                x2 = self.linear2(x)
                return self.merge(x1 + x2)

        container = TestContainer(
            in_features=self.in_features, out_features=self.out_features
        )
        assumed_total_in_features = (self.in_features + 1) * 2
        self.assertEqual(container.merge.total_in_features, assumed_total_in_features)
        self.assertEqual(
            container.merge.previous_tensor_s._shape,
            (assumed_total_in_features, assumed_total_in_features),
        )

        container.linear1.layer = torch.nn.Linear(
            in_features=container.linear1.in_features + 5,
            out_features=container.linear1.out_features,
        )
        container.update_size()
        assumed_total_in_features += 5
        self.assertEqual(container.merge.total_in_features, assumed_total_in_features)
        self.assertEqual(
            container.merge.previous_tensor_s._shape,
            (assumed_total_in_features, assumed_total_in_features),
        )

    def test_weights_statistics(self):
        """Test that weights_statistics method runs and returns a dictionary."""

        # Test the dummy container
        container = DummyGrowingContainer(
            in_features=self.in_features, out_features=self.out_features
        )

        stats = container.weights_statistics()

        # Check that the result is a dictionary
        self.assertIsInstance(
            stats, dict, "weights_statistics should return a dictionary"
        )


if __name__ == "__main__":
    unittest.main()
