import unittest
import unittest.mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_mlp import GrowingMLP, Perceptron
from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.modules.growing_module import GrowingModule
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.kfac import KFACVariant
from gromo.utils.tensor_statistic import KroneckerFactorStatistic, TensorStatistic
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize


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

    def extended_forward(self, x: torch.Tensor, mask: dict = {}) -> torch.Tensor:
        return self.forward(x)


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

    def test_dummy_select_update(self):
        container = TestContainer(
            in_features=self.in_features, out_features=self.out_features
        )
        idx = container.dummy_select_update()
        self.assertEqual(idx, 0, "dummy_select_update did not return the correct index")
        self.assertIs(container.currently_updated_layer, container.linear1)

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

    def test_compute_optimal_delta_forwards_worst_case_numerical_floor(self):
        """The flag of the numerical floor reaches every growing layer."""
        with unittest.mock.patch.object(
            LinearGrowingModule, "compute_optimal_delta"
        ) as compute_optimal_delta:
            self.model.compute_optimal_delta(worst_case_numerical_floor=False)
        self.assertEqual(
            compute_optimal_delta.call_count, len(self.model._growing_layers)
        )
        for call in compute_optimal_delta.call_args_list:
            self.assertIs(call.kwargs["worst_case_numerical_floor"], False)

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

    @unittest_parametrize(
        (
            {
                "compute_delta": True,
                "use_covariance": True,
                "alpha_zero": False,
                "use_projection": True,
            },
            {
                "compute_delta": False,
                "use_covariance": False,
                "alpha_zero": True,
                "use_projection": False,
            },
        )
    )
    def test_compute_optimal_updates(
        self,
        compute_delta: bool = True,
        use_covariance: bool = True,
        alpha_zero: bool = False,
        use_projection: bool = True,
    ):
        gather_statistics(self.dataloader, self.model, self.loss)
        # Clear any previous updates to ensure clean state for each test case
        for layer in self.model._growing_layers:
            layer.delete_update()
        self.model.compute_optimal_updates(
            compute_delta=compute_delta,
            use_covariance=use_covariance,
            alpha_zero=alpha_zero,
            use_projection=use_projection,
        )

        for layer in self.model._growing_layers:
            # Use explicit checks for configuration-specific behavior
            if not alpha_zero:
                # TINY-specific checks
                self.assertTrue(
                    hasattr(layer, "optimal_delta_layer"),
                    "compute_optimal_updates was not called on the growing layer",
                )
                self.assertTrue(
                    hasattr(layer, "parameter_update_decrease"),
                    "compute_optimal_updates was not called on the growing layer",
                )
            else:
                # GradMax-specific checks
                # optimal_delta_layer should not be set (may not exist or be None)
                self.assertFalse(
                    hasattr(layer, "optimal_delta_layer")
                    and layer.optimal_delta_layer is not None,
                    "GradMax configuration should not compute optimal_delta_layer",
                )
                # parameter_update_decrease should still be set so first_order_improvement is usable
                self.assertIsInstance(
                    layer.parameter_update_decrease,
                    torch.Tensor,
                    "GradMax configuration should still set parameter_update_decrease",
                )
                assert layer.parameter_update_decrease is not None
                self.assertTrue(torch.isfinite(layer.parameter_update_decrease))
                self.assertIsInstance(layer.first_order_improvement, torch.Tensor)

            # Common checks for all methods
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


class TestContainerPositionMask(TorchTestCase):
    """Tests of the position_mask argument of the GrowingContainer statistics."""

    n_classes = 3

    def test_update_computation_forwards_position_mask(self):
        """update_computation and update_covariance_loss_gradient forward the
        position_mask, unchanged, to every growing layer of the container."""
        container = DummyGrowingContainer(in_features=3, out_features=2)
        mask = torch.ones(4, 5, dtype=torch.bool, device=global_device())

        for method in ("update_computation", "update_covariance_loss_gradient"):
            with self.subTest(method=method):
                patchers = [
                    unittest.mock.patch.object(layer, method)
                    for layer in container._growing_layers
                ]
                mocks = []
                for patcher in patchers:
                    mocks.append(patcher.start())
                    self.addCleanup(patcher.stop)

                getattr(container, method)(position_mask=mask)

                for layer_mock in mocks:
                    layer_mock.assert_called_once()
                    self.assertIs(layer_mock.call_args.kwargs["position_mask"], mask)

    def _build(self):
        """Two-layer token-level MLP (3 -> 5 -> K) with a container around it."""
        torch.manual_seed(0)
        first = LinearGrowingModule(
            3,
            5,
            post_layer_function=torch.nn.ReLU(),
            device=global_device(),
            name="first",
        )
        second = LinearGrowingModule(
            5,
            self.n_classes,
            previous_module=first,
            device=global_device(),
            name="second",
        )
        first.next_module = second
        container = GrowingContainer(3, self.n_classes, device=global_device())
        container._growing_layers = [first, second]
        return container, first, second

    def _padded_run(self, x, labels, norm, mask):
        """True-Fisher statistics of the padded batch, optionally masked."""
        container, first, second = self._build()
        container.init_computation()
        logits = second(first(x)).flatten(0, 1)
        loss = (
            torch.nn.functional.cross_entropy(logits, labels.flatten(), reduction="sum")
            / norm
        )
        container.update_computation_true_fisher(
            loss, logits, loss_normalization=norm, position_mask=mask
        )
        return first, second

    def test_true_fisher_position_mask(self):
        """update_computation_true_fisher with position_mask gives the same
        statistics (S, M, M_prev, cross-covariance and the true-Fisher E) as
        the unpadded sequences fed one at a time without mask, while the
        padded positions carry random activations and random labels."""
        lengths = [2, 5, 1, 4]
        N, R, d = len(lengths), max(lengths), 3
        norm = float(N)

        torch.manual_seed(1)
        seqs = [torch.randn(r, d, device=global_device()) for r in lengths]
        x = torch.randn(N, R, d, device=global_device())
        mask = torch.zeros(N, R, dtype=torch.bool, device=global_device())
        for n, r in enumerate(lengths):
            x[n, :r] = seqs[n]
            mask[n, :r] = True
        labels = torch.randint(0, self.n_classes, (N, R), device=global_device())

        first, second = self._padded_run(x, labels, norm, mask)

        # Reference: one unpadded sequence at a time, no mask.
        container_r, first_r, second_r = self._build()
        container_r.init_computation()
        for n, r in enumerate(lengths):
            logits_n = second_r(first_r(seqs[n].unsqueeze(0))).flatten(0, 1)
            loss_n = (
                torch.nn.functional.cross_entropy(
                    logits_n, labels[n, :r], reduction="sum"
                )
                / norm
            )
            container_r.update_computation_true_fisher(
                loss_n, logits_n, loss_normalization=norm
            )

        for name, layer, layer_r in (
            ("first", first, first_r),
            ("second", second, second_r),
        ):
            with self.subTest(layer=name):
                self.assertEqual(layer.covariance_loss_gradient.samples, N)
                self.assertEqual(layer.tensor_s.samples, N)
                self.assertAllClose(
                    layer.covariance_loss_gradient(),
                    layer_r.covariance_loss_gradient(),
                    atol=1e-6,
                    rtol=1e-4,
                )
                self.assertAllClose(
                    layer.tensor_s(),
                    layer_r.tensor_s(),
                    atol=1e-6,
                    rtol=1e-4,
                )
                self.assertAllClose(
                    layer.tensor_m(),
                    layer_r.tensor_m(),
                    atol=1e-6,
                    rtol=1e-4,
                )
        with self.subTest(layer="second", statistic="m_prev and cross-covariance"):
            self.assertAllClose(
                second.tensor_m_prev(),
                second_r.tensor_m_prev(),
                atol=1e-6,
                rtol=1e-4,
            )
            self.assertAllClose(
                second.cross_covariance(),
                second_r.cross_covariance(),
                atol=1e-6,
                rtol=1e-4,
            )

        # Sanity check: without the mask, the padded positions leak into E.
        _, second_unmasked = self._padded_run(x, labels, norm, mask=None)
        self.assertFalse(
            torch.allclose(
                second_unmasked.covariance_loss_gradient(),
                second_r.covariance_loss_gradient(),
                atol=1e-6,
                rtol=1e-4,
            ),
            "Unmasked padded run should differ from the reference E.",
        )


class TestContainerKFACVariant(TorchTestCase):
    """Tests of the K-FAC variant propagation of the GrowingContainer."""

    n_classes = 3

    def _variant(self):
        """The implemented K-FAC variant, (reduce, count_weighted)."""
        return KFACVariant("reduce", "count_weighted")

    def _build(self):
        """Two-layer token-level MLP (3 -> 5 -> K) with a container around it."""
        torch.manual_seed(0)
        first = LinearGrowingModule(
            3,
            5,
            post_layer_function=torch.nn.ReLU(),
            device=global_device(),
            name="first",
        )
        second = LinearGrowingModule(
            5,
            self.n_classes,
            previous_module=first,
            device=global_device(),
            name="second",
        )
        first.next_module = second
        container = GrowingContainer(3, self.n_classes, device=global_device())
        container.first = first
        container.second = second
        container._growing_layers = [first, second]
        return container, first, second

    def _assert_psd(self, B, label):
        """Check that B is finite, symmetric and has non-negative eigenvalues."""
        with self.subTest(factor=label):
            self.assertTrue(torch.isfinite(B).all())
            self.assertAllClose(B, B.T, atol=1e-6, rtol=1e-6)
            eigenvalues = torch.linalg.eigvalsh(B)
            self.assertTrue(
                bool((eigenvalues >= -1e-6).all()),
                f"{label} has negative eigenvalues: {eigenvalues}",
            )

    def test_set_kfac_variant_reaches_nested_modules(self):
        """set_kfac_variant of a GrowingResidualMLP reaches the layers of its
        nested GrowingResidualBlock containers, which are not in _growing_layers,
        and None restores the plain statistics S and E on all of them."""
        variant = self._variant()
        model = GrowingResidualMLP(
            in_features=(3,),
            out_features=2,
            num_features=4,
            hidden_features=2,
            num_blocks=2,
            device=global_device(),
        )
        linear_layers = [
            module
            for module in model.modules()
            if isinstance(module, LinearGrowingModule)
        ]
        self.assertEqual(len(linear_layers), 4)

        model.set_kfac_variant(variant)
        for layer in linear_layers:
            with self.subTest(layer=layer.name):
                self.assertEqual(layer.kfac_variant, variant)
                self.assertIsInstance(layer.tensor_s, KroneckerFactorStatistic)
                self.assertIsInstance(
                    layer.covariance_loss_gradient, KroneckerFactorStatistic
                )

        model.set_kfac_variant(None)
        for layer in linear_layers:
            with self.subTest(layer=layer.name, variant=None):
                self.assertIsNone(layer.kfac_variant)
                self.assertIs(type(layer.tensor_s), TensorStatistic)
                self.assertIs(type(layer.covariance_loss_gradient), TensorStatistic)

    def test_set_kfac_variant_sequential_scheduling(self):
        """With sequential scheduling only one layer is in _growing_layers, but
        set_kfac_variant reaches every GrowingModule of the model, including the
        layers that are not currently scheduled for growth."""
        variant = self._variant()
        model = GrowingMLP(
            in_features=3,
            out_features=2,
            hidden_size=4,
            number_hidden_layers=2,
            device=global_device(),
        )
        model.set_growing_layers(scheduling_method="sequential")
        self.assertEqual(len(model._growing_layers), 1)

        growing_modules = [
            module for module in model.modules() if isinstance(module, GrowingModule)
        ]
        self.assertGreater(len(growing_modules), len(model._growing_layers))

        model.set_kfac_variant(variant)
        for module in growing_modules:
            with self.subTest(layer=module.name):
                self.assertEqual(module.kfac_variant, variant)
                self.assertIsInstance(module.tensor_s, KroneckerFactorStatistic)
                self.assertIsInstance(
                    module.covariance_loss_gradient, KroneckerFactorStatistic
                )

    def test_set_kfac_variant_with_merge_module(self):
        """A K-FAC variant is refused for a layer whose previous module is a
        MergeGrowingModule, which owns tensor_s, while the container call with
        None and the layers without a merge predecessor still work."""
        variant = self._variant()
        merge = LinearMergeGrowingModule(
            in_features=4, name="merge", device=global_device()
        )
        layer = LinearGrowingModule(
            3, 4, device=global_device(), next_module=merge, name="layer"
        )
        next_layer = LinearGrowingModule(
            4, 2, device=global_device(), previous_module=merge, name="next_layer"
        )
        merge.set_previous_modules([layer])
        merge.set_next_modules([next_layer])

        container = GrowingContainer(3, 2, device=global_device())
        container.layer = layer
        container.merge = merge
        container.next_layer = next_layer
        container._growing_layers = [layer, merge, next_layer]

        with self.assertRaises(NotImplementedError):
            container.set_kfac_variant(variant)
        container.set_kfac_variant(None)
        self.assertIsNone(next_layer.kfac_variant)

        with self.assertRaises(NotImplementedError):
            next_layer.set_kfac_variant(variant)
        layer.set_kfac_variant(variant)
        self.assertEqual(layer.kfac_variant, variant)
        self.assertIsInstance(layer.tensor_s, KroneckerFactorStatistic)

    def test_true_fisher_with_kfac_variant(self):
        """With the (reduce, count_weighted) variant on a pooled sequence
        classifier, the factors are finite, symmetric and positive semi-definite,
        and the input factor of the first layer equals the closed form
        A = sum_n u_n u_n^T / sum_n R_n^2, where u_n is the sum of the extended
        inputs over the valid positions of sequence n and R_n its length."""
        lengths = [2, 5, 1, 4]
        N, R, d = len(lengths), max(lengths), 3

        torch.manual_seed(1)
        x = torch.randn(N, R, d, device=global_device())
        mask = torch.zeros(N, R, dtype=torch.bool, device=global_device())
        for n, r in enumerate(lengths):
            x[n, r:] = torch.randn(R - r, d, device=global_device())
            mask[n, :r] = True

        container, first, second = self._build()
        container.set_kfac_variant(self._variant())
        container.init_computation()
        out = second(first(x))
        logits = (out * mask.unsqueeze(-1)).sum(dim=1)
        labels = torch.randint(0, self.n_classes, (N,), device=global_device())
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum") / N
        container.update_computation_true_fisher(
            loss, logits, loss_normalization=float(N), position_mask=mask
        )

        for layer in (first, second):
            with self.subTest(layer=layer.name):
                self.assertEqual(layer.covariance_loss_gradient.samples, N)
                self.assertEqual(layer.tensor_s.samples, N)
                self._assert_psd(layer.tensor_s(), f"{layer.name} tensor_s")
                self._assert_psd(
                    layer.covariance_loss_gradient(),
                    f"{layer.name} covariance_loss_gradient",
                )

        # Closed form for the input factor of the first layer.
        ext = first.input_extended.detach()
        u = (ext * mask.unsqueeze(-1)).sum(dim=1)  # (N, d + 1)
        denominator = float(sum(r**2 for r in lengths))
        A_expected = u.T @ u / denominator
        self.assertAllClose(first.tensor_s(), A_expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
