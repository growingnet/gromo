import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gromo.utils.optimisation import (
    BatchAdapter,
    BlockOptimizationBatch,
    ClosedFormBlockOptimizer,
    ClosedFormOptimizationConfig,
    FirstLayerMoments,
    OptimizationResult,
    SecondLayerMoments,
    SupervisedBatchAdapter,
    TwoLayerLinearBlockView,
    _compute_sample_weights,
    _inverse_activation,
    _projected_gradient_nonnegative_least_squares,
    _pseudo_inverse_from_svd,
    _solve_latent_targets,
    collect_first_layer_statistics,
    collect_second_layer_statistics,
    optimize,
    resolve_two_layer_linear_block,
    update_first_layer_from_statistics,
    update_second_layer_from_statistics,
)


class TeacherNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class StudentNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class DictBatchAdapter(BatchAdapter):
    def extract(
        self,
        block: nn.Module | TwoLayerLinearBlockView,
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> BlockOptimizationBatch:
        del block
        return BlockOptimizationBatch(
            block_input=batch["features"].to(device),
            target=batch["target"].to(device),
        )


class GrowingLikeLayer:
    def __init__(self, layer: nn.Linear, post_layer_function: nn.Module | None) -> None:
        self.layer = layer
        self.post_layer_function = post_layer_function


class GrowingLikeBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: nn.Module | None,
    ) -> None:
        super().__init__()
        self.first_layer = GrowingLikeLayer(nn.Linear(input_dim, hidden_dim), activation)
        self.second_layer = GrowingLikeLayer(nn.Linear(hidden_dim, output_dim), None)
        self.name = "growing_like"


def _build_dataset(
    *,
    seed: int = 0,
    input_dim: int = 4,
    hidden_dim: int = 6,
    output_dim: int = 3,
    train_size: int | None = None,
    test_size: int | None = None,
    batch_size: int | None = None,
) -> tuple[TeacherNetwork, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    train_size = train_size if train_size is not None else 256
    test_size = test_size if test_size is not None else 64
    batch_size = batch_size if batch_size is not None else 32

    torch.manual_seed(seed)
    teacher = TeacherNetwork(input_dim, hidden_dim, output_dim)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    x_train = torch.randn(train_size, input_dim)
    x_test = torch.randn(test_size, input_dim)
    with torch.no_grad():
        y_train = teacher(x_train)
        y_test = teacher(x_test)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return teacher, train_loader, test_loader, x_train, y_train


def _evaluate_loss(model: nn.Module, loader: DataLoader) -> float:
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            losses.append(criterion(model(x_batch), y_batch).item())
    return sum(losses) / len(losses)


class TestOptimisation(unittest.TestCase):
    def test_config_validation(self) -> None:
        ClosedFormOptimizationConfig()

        with self.assertRaisesRegex(ValueError, "z_mode"):
            ClosedFormOptimizationConfig(z_mode="bad")
        with self.assertRaisesRegex(ValueError, "z_solver"):
            ClosedFormOptimizationConfig(z_solver="bad")
        with self.assertRaisesRegex(ValueError, "delta_mode"):
            ClosedFormOptimizationConfig(delta_mode="bad")

    def test_supervised_adapter_and_block_view_properties(self) -> None:
        model = StudentNetwork(4, 5, 3)
        view = resolve_two_layer_linear_block(model)
        batch = (torch.randn(7, 4), torch.randn(7, 3))
        adapter = SupervisedBatchAdapter()
        extracted = adapter.extract(view, batch, view.device)

        self.assertEqual(extracted.block_input.shape, (7, 4))
        self.assertEqual(extracted.target.shape, (7, 3))
        self.assertEqual(view.input_dim, 4)
        self.assertEqual(view.hidden_dim, 5)
        self.assertEqual(view.output_dim, 3)
        self.assertEqual(view.hidden(extracted.block_input).shape, (7, 5))
        self.assertEqual(view.forward(extracted.block_input).shape, (7, 3))

        with self.assertRaisesRegex(TypeError, "expects batches like"):
            adapter.extract(view, {"x": batch[0], "y": batch[1]}, view.device)

    def test_resolve_two_layer_linear_block_supports_common_layouts(self) -> None:
        direct = StudentNetwork(3, 4, 2)
        direct_view = resolve_two_layer_linear_block(direct)
        self.assertEqual(direct_view.name, "StudentNetwork")

        sequential = nn.Sequential(nn.Linear(3, 4), nn.Softplus(), nn.Linear(4, 2))
        sequential_view = resolve_two_layer_linear_block(sequential)
        self.assertIsInstance(sequential_view.activation, nn.Softplus)
        self.assertEqual(sequential_view.name, "Sequential")

        growing_like = GrowingLikeBlock(3, 4, 2, nn.ReLU())
        growing_view = resolve_two_layer_linear_block(growing_like)
        self.assertEqual(growing_view.name, "growing_like")
        self.assertIsInstance(growing_view.activation, nn.ReLU)

        growing_like_identity = GrowingLikeBlock(3, 4, 2, None)
        growing_view_identity = resolve_two_layer_linear_block(growing_like_identity)
        self.assertIsInstance(growing_view_identity.activation, nn.Identity)

    def test_resolve_two_layer_linear_block_rejects_unknown_layout(self) -> None:
        invalid_blocks = [nn.Linear(3, 2)]

        bad_direct = StudentNetwork(3, 4, 2)
        bad_direct.linear1 = nn.Identity()
        invalid_blocks.append(bad_direct)

        invalid_blocks.append(nn.Sequential(nn.Identity(), nn.ReLU(), nn.Linear(3, 2)))

        bad_growing = GrowingLikeBlock(3, 4, 2, nn.ReLU())
        bad_growing.first_layer.layer = nn.Identity()
        invalid_blocks.append(bad_growing)

        for invalid_block in invalid_blocks:
            with self.subTest(block=invalid_block.__class__.__name__):
                with self.assertRaisesRegex(TypeError, "Could not resolve"):
                    resolve_two_layer_linear_block(invalid_block)

    def test_inverse_activation_supported_and_rejects_unknown(self) -> None:
        z = torch.tensor([[0.5, 1.0], [2.0, 3.0]], dtype=torch.float32)
        softplus_expected = z + torch.log(-torch.expm1(-z))
        self.assertTrue(
            torch.allclose(_inverse_activation(nn.Softplus(), z), softplus_expected)
        )
        self.assertTrue(torch.equal(_inverse_activation(nn.ReLU(), z), z))

        leaky_z = torch.tensor([[-2.0, 3.0]], dtype=torch.float32)
        expected_leaky = torch.tensor([[-10.0, 3.0]], dtype=torch.float32)
        self.assertTrue(
            torch.equal(_inverse_activation(nn.LeakyReLU(0.2), leaky_z), expected_leaky)
        )

        with self.assertRaisesRegex(TypeError, "Unsupported activation"):
            _inverse_activation(nn.Sigmoid(), z)

    def test_pseudo_inverse_and_latent_solvers_cover_branches(self) -> None:
        matrix = torch.tensor([[1.0, 0.0], [0.0, 0.01]], dtype=torch.float32)
        pseudo_inverse = _pseudo_inverse_from_svd(matrix, singular_value_floor=0.05)
        expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(pseudo_inverse, expected))

        second_linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            second_linear.weight.fill_(1.0)
        target = torch.tensor([[-0.5], [0.2]], dtype=torch.float32)
        current_prediction = torch.zeros_like(target)

        relu_config = ClosedFormOptimizationConfig(z_solver="pinv", clip=0.1)
        relu_latent = _solve_latent_targets(
            second_linear=second_linear,
            activation=nn.ReLU(),
            target=target,
            config=relu_config,
            current_prediction=current_prediction,
        )
        self.assertTrue(torch.allclose(relu_latent, torch.tensor([[0.1], [0.2]])))

        leaky_latent = _solve_latent_targets(
            second_linear=second_linear,
            activation=nn.LeakyReLU(0.1),
            target=target,
            config=relu_config,
            current_prediction=current_prediction,
        )
        self.assertTrue(torch.allclose(leaky_latent, target))

        pgd_config = ClosedFormOptimizationConfig(
            z_mode="mid",
            z_solver="pgd",
            clip=0.05,
            max_pgd_iter=20,
            pgd_tol=1e-8,
        )
        pgd_latent = _solve_latent_targets(
            second_linear=second_linear,
            activation=nn.ReLU(),
            target=torch.tensor([[2.0]], dtype=torch.float32),
            config=pgd_config,
            current_prediction=torch.tensor([[0.0]], dtype=torch.float32),
        )
        self.assertEqual(pgd_latent.shape, (1, 1))
        self.assertGreaterEqual(float(pgd_latent.min().detach()), 0.05)

        zero_iter_solution = _projected_gradient_nonnegative_least_squares(
            omega=torch.tensor([[1.0]], dtype=torch.float32),
            target=torch.tensor([[0.01]], dtype=torch.float32),
            clip=0.1,
            max_iter=0,
            tol=1e-8,
            svd_floor=1e-8,
        )
        self.assertTrue(torch.allclose(zero_iter_solution, torch.tensor([[0.1]])))

    def test_compute_sample_weights_modes_and_zero_denominator(self) -> None:
        block = resolve_two_layer_linear_block(
            nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
        )
        with torch.no_grad():
            block.first_linear.weight.fill_(1.0)
            block.first_linear.bias.zero_()

        x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        z = torch.tensor([[1.5], [2.0]], dtype=torch.float32)
        z_inv = torch.tensor([[0.5], [2.0]], dtype=torch.float32)

        one_weights = _compute_sample_weights(
            block,
            x,
            z,
            z_inv,
            ClosedFormOptimizationConfig(delta_mode="one"),
        )
        self.assertTrue(torch.equal(one_weights, torch.ones(2)))

        delta_weights = _compute_sample_weights(
            block,
            x,
            z,
            z_inv,
            ClosedFormOptimizationConfig(delta_mode="delta"),
        )
        self.assertEqual(delta_weights.shape, (2,))
        self.assertEqual(delta_weights[1].item(), 0.0)
        self.assertGreater(delta_weights[0].item(), 0.0)

    def test_collect_second_layer_statistics_can_stop_early(self) -> None:
        model = StudentNetwork(2, 3, 1)
        block = resolve_two_layer_linear_block(model)
        repeated_batch = (torch.ones(4, 2), torch.ones(4, 1))
        dataloader = [repeated_batch for _ in range(12)]

        moments = collect_second_layer_statistics(
            block=block,
            dataloader=dataloader,
            adapter=SupervisedBatchAdapter(),
            config=ClosedFormOptimizationConfig(estimate_precision=1e-6),
        )

        self.assertLess(moments.batches_used, len(dataloader))
        self.assertEqual(moments.sample_mass, 8.0)

    def test_collect_second_layer_statistics_without_early_stop_uses_full_loader(
        self,
    ) -> None:
        model = StudentNetwork(2, 3, 1)
        block = resolve_two_layer_linear_block(model)
        dataloader = [(torch.randn(4, 2), torch.randn(4, 1)) for _ in range(5)]

        moments = collect_second_layer_statistics(
            block=block,
            dataloader=dataloader,
            adapter=SupervisedBatchAdapter(),
            config=ClosedFormOptimizationConfig(estimate_precision=0.0),
        )

        self.assertEqual(moments.batches_used, len(dataloader))
        self.assertEqual(moments.sample_mass, 20.0)

    def test_collect_second_layer_statistics_raises_on_empty_loader(self) -> None:
        model = StudentNetwork(2, 3, 1)
        block = resolve_two_layer_linear_block(model)
        with self.assertRaisesRegex(ValueError, "produced no samples"):
            collect_second_layer_statistics(
                block=block,
                dataloader=[],
                adapter=SupervisedBatchAdapter(),
                config=ClosedFormOptimizationConfig(),
            )

    def test_update_second_layer_from_statistics_recovers_known_map(self) -> None:
        model = StudentNetwork(2, 3, 2)
        block = resolve_two_layer_linear_block(model)
        hidden = torch.tensor(
            [
                [1.0, 0.5, -0.5],
                [0.0, 1.0, 2.0],
                [2.0, -1.0, 1.0],
                [1.5, 2.0, 0.5],
                [-0.5, 0.25, 1.0],
            ],
            dtype=torch.float32,
        )
        omega_true = torch.tensor([[2.0, -1.0, 0.5], [0.5, 1.5, -2.0]])
        bias_true = torch.tensor([0.25, -0.75], dtype=torch.float32)
        target = hidden @ omega_true.T + bias_true

        moments = SecondLayerMoments(
            sample_mass=float(hidden.size(0)),
            batches_used=1,
            mean_hidden=hidden.mean(dim=0),
            mean_target=target.mean(dim=0),
            mean_hidden_hidden_t=hidden.transpose(0, 1) @ hidden / hidden.size(0),
            mean_target_hidden_t=target.transpose(0, 1) @ hidden / hidden.size(0),
        )
        update_second_layer_from_statistics(
            block=block,
            moments=moments,
            config=ClosedFormOptimizationConfig(reg=1e-8),
        )

        self.assertTrue(
            torch.allclose(block.second_linear.weight, omega_true, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            torch.allclose(block.second_linear.bias, bias_true, atol=1e-5, rtol=1e-4)
        )

    def test_update_statistics_support_biasless_layers(self) -> None:
        biasless_block = TwoLayerLinearBlockView(
            first_linear=nn.Linear(2, 2, bias=False),
            activation=nn.ReLU(),
            second_linear=nn.Linear(2, 1, bias=False),
            name="biasless",
        )
        second_moments = SecondLayerMoments(
            sample_mass=2.0,
            batches_used=1,
            mean_hidden=torch.tensor([0.5, 0.25]),
            mean_target=torch.tensor([1.0]),
            mean_hidden_hidden_t=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            mean_target_hidden_t=torch.tensor([[2.0, -1.0]]),
        )
        update_second_layer_from_statistics(
            block=biasless_block,
            moments=second_moments,
            config=ClosedFormOptimizationConfig(reg=1e-8),
        )
        self.assertIsNone(biasless_block.second_linear.bias)

        first_moments = FirstLayerMoments(
            sample_mass=2.0,
            batches_used=1,
            mean_x=torch.tensor([0.5, -0.5]),
            mean_z_inv=torch.tensor([1.0, 0.0]),
            mean_xx_t=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            mean_z_inv_x_t=torch.tensor([[2.0, -1.0], [0.5, 1.5]]),
        )
        update_first_layer_from_statistics(
            block=biasless_block,
            moments=first_moments,
            config=ClosedFormOptimizationConfig(reg=1e-8),
        )
        self.assertIsNone(biasless_block.first_linear.bias)

    def test_collect_first_layer_statistics_can_stop_early(self) -> None:
        model = StudentNetwork(2, 3, 1)
        block = resolve_two_layer_linear_block(model)
        repeated_batch = (torch.ones(4, 2), torch.ones(4, 1))
        dataloader = [repeated_batch for _ in range(12)]

        moments = collect_first_layer_statistics(
            block=block,
            dataloader=dataloader,
            adapter=SupervisedBatchAdapter(),
            config=ClosedFormOptimizationConfig(
                estimate_precision=1e-6,
                z_solver="pinv",
                delta_mode="one",
            ),
        )

        self.assertLess(moments.batches_used, len(dataloader))
        self.assertGreater(moments.sample_mass, 0.0)

    def test_collect_first_layer_statistics_raises_on_zero_total_weight(self) -> None:
        model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
        block = resolve_two_layer_linear_block(model)
        with torch.no_grad():
            block.first_linear.weight.fill_(1.0)
            block.first_linear.bias.zero_()
            block.second_linear.weight.fill_(1.0)
            block.second_linear.bias.zero_()

        x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        y = block.forward(x)
        with self.assertRaisesRegex(ValueError, "sample weights are zero"):
            collect_first_layer_statistics(
                block=block,
                dataloader=[(x, y)],
                adapter=SupervisedBatchAdapter(),
                config=ClosedFormOptimizationConfig(
                    delta_mode="delta",
                    z_solver="pinv",
                    clip=1e-6,
                ),
            )

    def test_update_first_layer_from_statistics_recovers_known_map(self) -> None:
        model = StudentNetwork(2, 3, 1)
        block = resolve_two_layer_linear_block(model)
        x = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, -1.0]],
            dtype=torch.float32,
        )
        a_true = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-0.75, 0.25]])
        bias_true = torch.tensor([0.3, -0.4, 0.1], dtype=torch.float32)
        z_inv = x @ a_true.T + bias_true

        moments = FirstLayerMoments(
            sample_mass=float(x.size(0)),
            batches_used=1,
            mean_x=x.mean(dim=0),
            mean_z_inv=z_inv.mean(dim=0),
            mean_xx_t=x.transpose(0, 1) @ x / x.size(0),
            mean_z_inv_x_t=z_inv.transpose(0, 1) @ x / x.size(0),
        )
        update_first_layer_from_statistics(
            block=block,
            moments=moments,
            config=ClosedFormOptimizationConfig(reg=1e-8),
        )

        self.assertTrue(
            torch.allclose(block.first_linear.weight, a_true, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            torch.allclose(block.first_linear.bias, bias_true, atol=1e-5, rtol=1e-4)
        )

    def test_closed_form_optimizer_improves_teacher_student_fit(self) -> None:
        _, train_loader, test_loader, x_train, y_train = _build_dataset(seed=0)
        x_test, y_test = test_loader.dataset.tensors

        torch.manual_seed(101)
        student = StudentNetwork(4, 6, 3)
        mse = nn.MSELoss()
        with torch.no_grad():
            initial_train_loss = mse(student(x_train), y_train).item()
            initial_test_loss = mse(student(x_test), y_test).item()

        optimizer = ClosedFormBlockOptimizer(
            config=ClosedFormOptimizationConfig(
                nb_iterations=2,
                reg=1e-8,
                z_mode="star",
                z_solver="pinv",
                clip=1e-6,
                delta_mode="one",
            )
        )
        result = optimizer.optimize(student, train_loader)

        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.block_name, "StudentNetwork")

        with torch.no_grad():
            final_train_loss = mse(student(x_train), y_train).item()
            final_test_loss = mse(student(x_test), y_test).item()

        self.assertLess(final_train_loss, initial_train_loss)
        self.assertLess(final_test_loss, initial_test_loss)
        self.assertLess(final_train_loss, 0.2)

    def test_optimize_accepts_view_and_custom_adapter(self) -> None:
        _, train_loader, _, _, _ = _build_dataset(seed=1, train_size=64, test_size=32)
        x_train, y_train = train_loader.dataset.tensors
        dict_loader = [
            {"features": x_train[i : i + 16], "target": y_train[i : i + 16]}
            for i in range(0, x_train.size(0), 16)
        ]

        torch.manual_seed(202)
        student = StudentNetwork(4, 6, 3)
        view = resolve_two_layer_linear_block(student)
        result = optimize(
            view,
            dict_loader,
            adapter=DictBatchAdapter(),
            config=ClosedFormOptimizationConfig(
                nb_iterations=2,
                reg=1e-8,
                z_solver="pgd",
                max_pgd_iter=5,
                estimate_precision=0.0,
            ),
        )

        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.block_name, "StudentNetwork")


if __name__ == "__main__":
    unittest.main()
