"""
Closed-form optimization utilities for ``Linear -> activation -> Linear`` blocks.

The optimizer alternates two closed-form updates:

1. update the second layer from hidden-feature moments,
2. update the first layer from latent targets inferred through the second layer.

For a block

``y_hat = Omega sigma(Ax + b_A) + b_Omega``

the second-layer update solves a centered least-squares problem of the form

``Omega* = Cov(y, h) Cov(h, h)^(-1)``

with ``h = sigma(Ax + b_A)``. The first-layer update solves the analogous
weighted regression on the inverse-activation targets ``z_inv``:

``A* = Cov_w(z_inv, x) Cov_w(x, x)^(-1)``.

Moments are estimated batch by batch with ``TensorStatistic`` helpers. When a
target precision is provided, the dataloader pass can stop early once the
tracked batch-average estimates have stabilized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, Union

import torch
import torch.nn as nn

from gromo.utils.tensor_statistic import (
    TensorStatistic,
    TensorStatisticWithEstimationError,
)


@dataclass(frozen=True)
class ClosedFormOptimizationConfig:
    """
    Hyper-parameters controlling the alternating closed-form updates.

    Attributes
    ----------
    reg : float
        Ridge parameter added to covariance matrices before inversion.
    nb_iterations : int
        Number of alternating ``Omega -> A`` passes.
    z_mode : str
        ``"star"`` uses the target itself, while ``"mid"`` uses the midpoint
        between the target and the current prediction.
    z_solver : str
        Method used to infer latent targets ``z`` from the second layer.
    clip : float
        Lower bound enforced on ``z`` for non-negative activations.
    delta_mode : str
        ``"one"`` uses uniform weights and ``"delta"`` uses the sample-wise
        ratio described in ``_compute_sample_weights``.
    max_pgd_iter : int
        Maximum number of projected-gradient iterations for the latent solver.
    pgd_tol : float
        Stopping tolerance on consecutive projected-gradient iterates.
    svd_floor : float
        Singular values below this threshold are discarded in pseudo-inverses.
    estimate_precision : float
        Relative precision target for statistic estimation. A value ``<= 0``
        disables early stopping and consumes the full dataloader.
    trace_precision : float
        Relative precision target used internally by
        ``TensorStatisticWithEstimationError`` to stop refining its trace
        estimate.
    convergence_tau : float
        Small positive constant used in relative-error denominators.
    verbose : int
        When non-zero, print a short optimization summary.
    """

    reg: float = 1e-6
    nb_iterations: int = 2
    z_mode: str = "star"
    z_solver: str = "pgd"
    clip: float = 1e-2
    delta_mode: str = "one"
    max_pgd_iter: int = 15
    pgd_tol: float = 1e-4
    svd_floor: float = 5e-2
    estimate_precision: float = 5e-2
    trace_precision: float = 1e-2
    convergence_tau: float = 1e-8
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.z_mode not in {"star", "mid"}:
            raise ValueError("z_mode must be 'star' or 'mid'.")
        if self.z_solver not in {"pinv", "pgd"}:
            raise ValueError("z_solver must be 'pinv' or 'pgd'.")
        if self.delta_mode not in {"one", "delta"}:
            raise ValueError("delta_mode must be 'one' or 'delta'.")


@dataclass
class BlockOptimizationBatch:
    """
    Tensors extracted from one dataloader batch for the optimized block.

    ``block_input`` is the tensor fed to the first linear layer and ``target``
    is the desired output of the full two-layer block.
    """

    block_input: torch.Tensor
    target: torch.Tensor


ResolvedTwoLayerBlock: TypeAlias = Union[nn.Module, "TwoLayerLinearBlockView"]


class BatchAdapter(Protocol):
    """
    Protocol converting raw dataloader batches into block-local tensors.
    """

    def extract(
        self,
        block: ResolvedTwoLayerBlock,
        batch: Any,
        device: torch.device,
    ) -> BlockOptimizationBatch:
        """
        Return ``(block_input, target)`` on the block device for one batch.
        """


class SupervisedBatchAdapter:
    """
    Adapter for standard supervised dataloaders yielding ``(x, y)`` batches.
    """

    def __init__(
        self,
        input_index: int = 0,
        target_index: int = 1,
    ) -> None:
        self.input_index = input_index
        self.target_index = target_index

    def extract(
        self,
        block: ResolvedTwoLayerBlock,
        batch: Any,
        device: torch.device,
    ) -> BlockOptimizationBatch:
        """
        Move the selected input and target tensors to ``device``.
        """
        del block
        if not isinstance(batch, (tuple, list)):
            raise TypeError(
                "The default SupervisedBatchAdapter expects batches like (x, y). "
                "Provide a custom adapter for other batch structures."
            )
        x = batch[self.input_index].to(device)
        y = batch[self.target_index].to(device)
        return BlockOptimizationBatch(block_input=x, target=y)


@dataclass
class TwoLayerLinearBlockView:
    """
    Normalized view of a ``Linear -> activation -> Linear`` block.
    """

    first_linear: nn.Linear
    activation: nn.Module
    second_linear: nn.Linear
    name: str = "two_layer_block"

    @property
    def device(self) -> torch.device:
        """
        Device on which the block parameters live.
        """
        return self.first_linear.weight.device

    @property
    def input_dim(self) -> int:
        """
        Input dimension ``n`` of the first linear layer.
        """
        return self.first_linear.in_features

    @property
    def hidden_dim(self) -> int:
        """
        Hidden dimension ``k`` of the intermediate representation.
        """
        return self.first_linear.out_features

    @property
    def output_dim(self) -> int:
        """
        Output dimension ``m`` of the second linear layer.
        """
        return self.second_linear.out_features

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ``h = sigma(Ax + b_A)`` for one batch.
        """
        return self.activation(self.first_linear(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ``Omega sigma(Ax + b_A) + b_Omega`` for one batch.
        """
        return self.second_linear(self.hidden(x))


def resolve_two_layer_linear_block(block: nn.Module) -> TwoLayerLinearBlockView:
    """
    Extract a common block view from supported module layouts.

    The returned object always exposes ``(first_linear, activation,
    second_linear)`` so the optimizer can work with a uniform API.
    """
    if all(hasattr(block, name) for name in ("linear1", "activation", "linear2")):
        linear1 = block.linear1
        activation = block.activation
        linear2 = block.linear2
        if isinstance(linear1, nn.Linear) and isinstance(linear2, nn.Linear):
            return TwoLayerLinearBlockView(
                first_linear=linear1,
                activation=activation,
                second_linear=linear2,
                name=block.__class__.__name__,
            )

    if isinstance(block, nn.Sequential) and len(block) == 3:
        linear1, activation, linear2 = block[0], block[1], block[2]
        if isinstance(linear1, nn.Linear) and isinstance(linear2, nn.Linear):
            return TwoLayerLinearBlockView(
                first_linear=linear1,
                activation=activation,
                second_linear=linear2,
                name=block.__class__.__name__,
            )

    if hasattr(block, "first_layer") and hasattr(block, "second_layer"):
        first_layer = block.first_layer
        second_layer = block.second_layer
        linear1 = getattr(first_layer, "layer", None)
        activation = getattr(first_layer, "post_layer_function", None)
        linear2 = getattr(second_layer, "layer", None)
        if isinstance(linear1, nn.Linear) and isinstance(linear2, nn.Linear):
            return TwoLayerLinearBlockView(
                first_linear=linear1,
                activation=nn.Identity() if activation is None else activation,
                second_linear=linear2,
                name=getattr(block, "name", block.__class__.__name__),
            )

    raise TypeError(
        "Could not resolve a two-layer linear block from the given module. "
        "Provide a block with (linear1, activation, linear2), a 3-layer "
        "nn.Sequential, or a custom wrapper and custom optimizer logic."
    )


@dataclass
class SecondLayerMoments:
    """
    Materialized expectations used in the second-layer closed-form update.

    They approximate with h = F(Ax + b_A)

    ``E[h]``, ``E[y]``, ``E[h h^T]`` and ``E[y h^T]``.
    """

    sample_mass: float
    batches_used: int
    mean_hidden: torch.Tensor
    mean_target: torch.Tensor
    mean_hidden_hidden_t: torch.Tensor
    mean_target_hidden_t: torch.Tensor


@dataclass
class FirstLayerMoments:
    """
    Materialized weighted expectations used in the first-layer update.

    They approximate

    ``E_w[x]``, ``E_w[z_inv]``, ``E_w[x x^T]`` and ``E_w[z_inv x^T]``.
    """

    sample_mass: float
    batches_used: int
    mean_x: torch.Tensor
    mean_z_inv: torch.Tensor
    mean_xx_t: torch.Tensor
    mean_z_inv_x_t: torch.Tensor


@dataclass
class OptimizationResult:
    """
    Summary returned after the in-place block update.
    """

    block_name: str
    nb_iterations: int
    omega_norm: float
    a_norm: float


def _centered_covariance(
    second_moment: torch.Tensor,
    first_moment_left: torch.Tensor,
    first_moment_right: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ``Cov(X, Y) = E[XY^T] - E[X] E[Y]^T`` from batch-estimated moments.
    """
    return second_moment - torch.outer(first_moment_left, first_moment_right)


def _pseudo_inverse_from_svd(
    matrix: torch.Tensor,
    singular_value_floor: float,
) -> torch.Tensor:
    """
    Compute a truncated Moore-Penrose pseudo-inverse from an SVD.

    If ``matrix = U diag(s) V^T``, singular values below
    ``singular_value_floor`` are discarded and the pseudo-inverse is

    ``V diag(s^{-1}) U^T``.
    """
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    s_inv = torch.where(
        s > singular_value_floor,
        1.0 / s,
        torch.zeros_like(s),
    )
    return vh.transpose(0, 1) @ torch.diag(s_inv) @ u.transpose(0, 1)


def _inverse_activation(
    activation: nn.Module,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the inverse activation needed by the first-layer regression.

    Supported cases are:

    - Softplus: ``sigma^{-1}(z) = z + log(1 - exp(-z))``
    - ReLU: the clipped latent target already lies in the positive branch
    - LeakyReLU: piecewise linear inverse
    """
    if isinstance(activation, nn.Softplus):
        return z + torch.log(-torch.expm1(-z))
    if isinstance(activation, nn.ReLU):
        return z
    if isinstance(activation, nn.LeakyReLU):
        return torch.where(z >= 0, z, z / activation.negative_slope)
    raise TypeError(
        "Unsupported activation for inverse-activation step. "
        "Supported activations are Softplus, ReLU and LeakyReLU."
    )


def _solve_latent_targets(
    second_linear: nn.Linear,
    activation: nn.Module,
    target: torch.Tensor,
    config: ClosedFormOptimizationConfig,
    current_prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Infer latent targets ``z`` for the hidden representation.

    Let ``Omega`` and ``b_Omega`` denote the parameters of the second layer.
    The solver first builds the shifted target

    ``y_goal = y - b_Omega``              if ``z_mode == "star"``
    ``y_goal = 0.5 (y + y_hat) - b_Omega`` if ``z_mode == "mid"``

    and then solves

    ``min_z ||Omega z - y_goal||_2^2``

    either by pseudo-inversion or by projected gradient descent. Non-negative
    activations use ``clip`` to enforce the feasible set ``z >= clip``.
    """
    omega = second_linear.weight
    bias = second_linear.bias
    if bias is None:
        bias = torch.zeros(omega.size(0), device=omega.device, dtype=omega.dtype)

    if config.z_mode == "star":
        target_goal = target - bias
    else:
        target_goal = 0.5 * (target + current_prediction) - bias

    omega_inv = _pseudo_inverse_from_svd(omega, config.svd_floor)

    # For LeakyReLU, allow negative latent targets and always use
    # the unconstrained least-squares solution, regardless of z_solver.
    if isinstance(activation, nn.LeakyReLU):
        z = target_goal @ omega_inv.transpose(0, 1)
        return z

    if config.z_solver == "pinv":
        z = target_goal @ omega_inv.transpose(0, 1)
        return torch.clamp(z, min=config.clip)

    return _projected_gradient_nonnegative_least_squares(
        omega=omega,
        target=target_goal,
        clip=config.clip,
        max_iter=config.max_pgd_iter,
        tol=config.pgd_tol,
        svd_floor=config.svd_floor,
    )


def _projected_gradient_nonnegative_least_squares(
    omega: torch.Tensor,
    target: torch.Tensor,
    clip: float,
    max_iter: int,
    tol: float,
    svd_floor: float,
) -> torch.Tensor:
    """
    Solve ``min_z ||Omega z - target||_2^2`` subject to ``z >= clip``.

    The quadratic objective can be written row-wise as

    ``f(z) = 0.5 z (Omega^T Omega) z^T - z (target Omega)^T + c``

    so the gradient is

    ``grad f(z) = z (Omega^T Omega) - target Omega``.

    The implementation uses a FISTA-like accelerated projected-gradient step:

    ``z_{t+1} = Proj_{z >= clip}(m_t - eta grad f(m_t))``

    with ``eta = 1 / ||Omega||_2^2`` and the classical extrapolation update on
    ``m_t``.
    """
    spectral_norm = torch.linalg.matrix_norm(omega, ord=2)
    step = 1.0 / max(spectral_norm.pow(2).item(), 1e-12)
    gram = omega.transpose(0, 1) @ omega
    bias_term = target @ omega

    omega_inv = _pseudo_inverse_from_svd(omega, svd_floor)
    z = torch.clamp(target @ omega_inv.transpose(0, 1), min=clip)
    momentum = z.clone()
    alpha = 1.0

    for _ in range(max_iter):
        z_next = torch.clamp(
            momentum - step * (momentum @ gram - bias_term),
            min=clip,
        )
        if torch.linalg.norm(z_next - z) <= tol:
            return z_next

        alpha_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * alpha * alpha))
        beta = (alpha - 1.0) / alpha_next
        momentum = z_next + beta * (z_next - z)
        z = z_next
        alpha = alpha_next

    return z


def _compute_sample_weights(
    block: TwoLayerLinearBlockView,
    x: torch.Tensor,
    z: torch.Tensor,
    z_inv: torch.Tensor,
    config: ClosedFormOptimizationConfig,
) -> torch.Tensor:
    """
    Compute the weights used by the first-layer regression.

    With ``delta_mode == "one"``, every sample has unit weight. Otherwise the
    weight of sample ``i`` is

    ``delta_i = ||sigma(Ax_i) - z_i||_2^2 / ||Ax_i - sigma^{-1}(z_i)||_2^2``.

    The weights are clipped to avoid exploding ratios when the denominator is
    very small.
    """
    if config.delta_mode == "one":
        return torch.ones(x.size(0), device=x.device, dtype=x.dtype)

    ax = block.first_linear(x)
    numerator = torch.linalg.norm(block.activation(ax) - z, dim=1).pow(2)
    denominator = torch.linalg.norm(ax - z_inv, dim=1).pow(2)
    weights = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return torch.clamp(weights, min=0.0, max=100.0)


def _new_mean_statistic(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    name: str,
    trace_precision: float,
) -> TensorStatisticWithEstimationError:
    """
    Create a statistic tracking the batch-average of one tensor quantity.
    """
    return TensorStatisticWithEstimationError(
        shape=shape,
        device=device,
        name=name,
        trace_precision=trace_precision,
        update_function=lambda batch_tensor: (
            batch_tensor.sum(dim=0),
            batch_tensor.size(0),
        ),
    )


def _new_cross_statistic(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    name: str,
    trace_precision: float,
) -> TensorStatisticWithEstimationError:
    """
    Create a statistic tracking ``E[left^T right]`` from two batched matrices.
    """
    return TensorStatisticWithEstimationError(
        shape=shape,
        device=device,
        name=name,
        trace_precision=trace_precision,
        update_function=lambda left, right: (
            left.transpose(0, 1) @ right,
            left.size(0),
        ),
    )


def _new_weighted_statistic(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    name: str,
    trace_precision: float,
    update_function: Any,
) -> TensorStatisticWithEstimationError:
    """
    Create a weighted statistic whose normalization is the batch weight mass.
    """
    return TensorStatisticWithEstimationError(
        shape=shape,
        device=device,
        name=name,
        trace_precision=trace_precision,
        update_function=update_function,
    )


def _update_statistic(statistic: TensorStatistic, **kwargs: Any) -> None:
    """
    Mark a statistic as stale and consume one batch update.
    """
    statistic.updated = False
    statistic.update(**kwargs)


def _relative_statistic_error(
    statistic: TensorStatisticWithEstimationError,
    tau: float,
) -> float:
    """
    Estimate the relative error of a statistic from its tracked trace.

    If ``mu`` is the current estimate and ``err`` is the quadratic error on the
    mean returned by ``TensorStatisticWithEstimationError.error()``, this helper
    returns

    ``sqrt(err) / (||mu||_F + tau)``.
    """
    error = statistic.error()
    if not math.isfinite(error):
        return float("inf")
    return math.sqrt(max(error, 0.0)) / (torch.linalg.norm(statistic()).item() + tau)


def _statistics_converged(
    statistics: list[TensorStatisticWithEstimationError],
    config: ClosedFormOptimizationConfig,
) -> bool:
    """
    Check whether every tracked statistic meets the configured precision target.
    """
    if config.estimate_precision <= 0:
        return False
    return all(
        _relative_statistic_error(statistic, config.convergence_tau)
        <= config.estimate_precision
        for statistic in statistics
    )


@torch.no_grad()
def collect_second_layer_statistics(
    block: TwoLayerLinearBlockView,
    dataloader: list[Any] | Any,
    adapter: BatchAdapter,
    config: ClosedFormOptimizationConfig,
) -> SecondLayerMoments:
    """
    Estimate the moments used by the second-layer regression.

    For ``h = sigma(Ax + b_A)``, the function tracks batch estimates of

    ``E[h]``, ``E[y]``, ``E[h h^T]`` and ``E[y h^T]``.

    If ``estimate_precision > 0``, the pass can stop early once all tracked
    batch-average statistics satisfy the requested relative precision.
    """
    device = block.device
    hidden_stat = _new_mean_statistic(
        (block.hidden_dim,),
        device=device,
        name="mean_hidden",
        trace_precision=config.trace_precision,
    )
    target_stat = _new_mean_statistic(
        (block.output_dim,),
        device=device,
        name="mean_target",
        trace_precision=config.trace_precision,
    )
    hidden_hidden_stat = _new_cross_statistic(
        (block.hidden_dim, block.hidden_dim),
        device=device,
        name="mean_hidden_hidden_t",
        trace_precision=config.trace_precision,
    )
    target_hidden_stat = _new_cross_statistic(
        (block.output_dim, block.hidden_dim),
        device=device,
        name="mean_target_hidden_t",
        trace_precision=config.trace_precision,
    )
    tracked_stats = [
        hidden_stat,
        target_stat,
        hidden_hidden_stat,
        target_hidden_stat,
    ]

    sample_mass = 0.0
    batches_used = 0
    for raw_batch in dataloader:
        batch = adapter.extract(block, raw_batch, device)
        hidden = block.hidden(batch.block_input)
        target = batch.target

        _update_statistic(hidden_stat, batch_tensor=hidden)
        _update_statistic(target_stat, batch_tensor=target)
        _update_statistic(hidden_hidden_stat, left=hidden, right=hidden)
        _update_statistic(target_hidden_stat, left=target, right=hidden)

        sample_mass += float(hidden.size(0))
        batches_used += 1
        if _statistics_converged(tracked_stats, config):
            break

    if sample_mass <= 0:
        raise ValueError("The dataloader produced no samples.")

    return SecondLayerMoments(
        sample_mass=sample_mass,
        batches_used=batches_used,
        mean_hidden=hidden_stat(),
        mean_target=target_stat(),
        mean_hidden_hidden_t=hidden_hidden_stat(),
        mean_target_hidden_t=target_hidden_stat(),
    )


@torch.no_grad()
def update_second_layer_from_statistics(
    block: TwoLayerLinearBlockView,
    moments: SecondLayerMoments,
    config: ClosedFormOptimizationConfig,
) -> None:
    """
    Solve the centered second-layer least-squares problem in closed form.

    With ``h = sigma(Ax + b_A)``, the regression is

    ``min_{Omega, b_Omega} E ||y - (Omega h + b_Omega)||_2^2``.

    The optimum satisfies

    ``Omega* = Cov(y, h) (Cov(h, h) + reg I)^(-1)``
    ``b_Omega* = E[y] - Omega* E[h]``.
    """
    device = block.device
    covariance_hidden = _centered_covariance(
        second_moment=moments.mean_hidden_hidden_t,
        first_moment_left=moments.mean_hidden,
        first_moment_right=moments.mean_hidden,
    )
    cross_target_hidden = _centered_covariance(
        second_moment=moments.mean_target_hidden_t,
        first_moment_left=moments.mean_target,
        first_moment_right=moments.mean_hidden,
    )

    covariance_hidden = covariance_hidden + config.reg * torch.eye(
        covariance_hidden.size(0),
        device=device,
        dtype=covariance_hidden.dtype,
    )
    omega_star = torch.linalg.solve(covariance_hidden, cross_target_hidden.T).T
    bias_star = moments.mean_target - omega_star @ moments.mean_hidden

    block.second_linear.weight.copy_(omega_star)
    if block.second_linear.bias is not None:
        block.second_linear.bias.copy_(bias_star)


@torch.no_grad()
def collect_first_layer_statistics(
    block: TwoLayerLinearBlockView,
    dataloader: list[Any] | Any,
    adapter: BatchAdapter,
    config: ClosedFormOptimizationConfig,
) -> FirstLayerMoments:
    """
    Estimate the weighted moments used in the first-layer regression.

    For each sample, the current second layer defines a latent target ``z``.
    After inverse activation, the first layer is fitted through weighted
    moments of ``x`` and ``z_inv = sigma^{-1}(z)``.
    """
    device = block.device
    x_stat = _new_weighted_statistic(
        (block.input_dim,),
        device=device,
        name="mean_x",
        trace_precision=config.trace_precision,
        update_function=lambda batch_tensor, batch_weight: (
            (batch_tensor * batch_weight.unsqueeze(1)).sum(dim=0),
            float(batch_weight.sum().item()),
        ),
    )
    z_inv_stat = _new_weighted_statistic(
        (block.hidden_dim,),
        device=device,
        name="mean_z_inv",
        trace_precision=config.trace_precision,
        update_function=lambda batch_tensor, batch_weight: (
            (batch_tensor * batch_weight.unsqueeze(1)).sum(dim=0),
            float(batch_weight.sum().item()),
        ),
    )
    xx_t_stat = _new_weighted_statistic(
        (block.input_dim, block.input_dim),
        device=device,
        name="mean_xx_t",
        trace_precision=config.trace_precision,
        update_function=lambda batch_tensor, batch_weight: (
            batch_tensor.transpose(0, 1) @ (batch_tensor * batch_weight.unsqueeze(1)),
            float(batch_weight.sum().item()),
        ),
    )
    z_inv_x_t_stat = _new_weighted_statistic(
        (block.hidden_dim, block.input_dim),
        device=device,
        name="mean_z_inv_x_t",
        trace_precision=config.trace_precision,
        update_function=lambda left, right, batch_weight: (
            left.transpose(0, 1) @ (right * batch_weight.unsqueeze(1)),
            float(batch_weight.sum().item()),
        ),
    )
    tracked_stats = [x_stat, z_inv_stat, xx_t_stat, z_inv_x_t_stat]

    sample_mass = 0.0
    batches_used = 0
    for raw_batch in dataloader:
        batch = adapter.extract(block, raw_batch, device)
        x = batch.block_input
        target = batch.target
        current_prediction = block.forward(x)
        z = _solve_latent_targets(
            second_linear=block.second_linear,
            activation=block.activation,
            target=target,
            config=config,
            current_prediction=current_prediction,
        )
        z_inv = _inverse_activation(block.activation, z)
        sample_weights = _compute_sample_weights(block, x, z, z_inv, config)
        batch_mass = float(sample_weights.sum().item())
        if batch_mass <= 0:
            continue

        _update_statistic(x_stat, batch_tensor=x, batch_weight=sample_weights)
        _update_statistic(z_inv_stat, batch_tensor=z_inv, batch_weight=sample_weights)
        _update_statistic(xx_t_stat, batch_tensor=x, batch_weight=sample_weights)
        _update_statistic(
            z_inv_x_t_stat,
            left=z_inv,
            right=x,
            batch_weight=sample_weights,
        )

        sample_mass += batch_mass
        batches_used += 1
        if _statistics_converged(tracked_stats, config):
            break

    if sample_mass <= 0:
        raise ValueError("The accumulated sample weights are zero.")

    return FirstLayerMoments(
        sample_mass=sample_mass,
        batches_used=batches_used,
        mean_x=x_stat(),
        mean_z_inv=z_inv_stat(),
        mean_xx_t=xx_t_stat(),
        mean_z_inv_x_t=z_inv_x_t_stat(),
    )


@torch.no_grad()
def update_first_layer_from_statistics(
    block: TwoLayerLinearBlockView,
    moments: FirstLayerMoments,
    config: ClosedFormOptimizationConfig,
) -> None:
    """
    Solve the weighted first-layer regression in closed form.

    Once latent targets are fixed, the update minimizes

    ``min_{A, b_A} E_w ||z_inv - (Ax + b_A)||_2^2``.

    The optimum is

    ``A* = Cov_w(z_inv, x) (Cov_w(x, x) + reg I)^(-1)``
    ``b_A* = E_w[z_inv] - A* E_w[x]``.
    """
    device = block.device
    covariance_x = _centered_covariance(
        second_moment=moments.mean_xx_t,
        first_moment_left=moments.mean_x,
        first_moment_right=moments.mean_x,
    )
    cross_z_inv_x = _centered_covariance(
        second_moment=moments.mean_z_inv_x_t,
        first_moment_left=moments.mean_z_inv,
        first_moment_right=moments.mean_x,
    )

    covariance_x = covariance_x + config.reg * torch.eye(
        covariance_x.size(0),
        device=device,
        dtype=covariance_x.dtype,
    )
    a_star = torch.linalg.solve(covariance_x, cross_z_inv_x.T).T
    bias_star = moments.mean_z_inv - a_star @ moments.mean_x

    block.first_linear.weight.copy_(a_star)
    if block.first_linear.bias is not None:
        block.first_linear.bias.copy_(bias_star)


class ClosedFormBlockOptimizer:
    """
    Alternate second-layer and first-layer closed-form updates in place.
    """

    def __init__(
        self,
        config: ClosedFormOptimizationConfig | None = None,
        adapter: BatchAdapter | None = None,
    ) -> None:
        self.config = config if config is not None else ClosedFormOptimizationConfig()
        self.adapter = adapter if adapter is not None else SupervisedBatchAdapter()

    @torch.no_grad()
    def optimize(
        self,
        block: nn.Module | TwoLayerLinearBlockView,
        dataloader: list[Any] | Any,
    ) -> OptimizationResult:
        """
        Run the alternating optimizer and update the block weights in place.
        """
        block_view = (
            block
            if isinstance(block, TwoLayerLinearBlockView)
            else resolve_two_layer_linear_block(block)
        )

        for _ in range(self.config.nb_iterations):
            second_moments = collect_second_layer_statistics(
                block=block_view,
                dataloader=dataloader,
                adapter=self.adapter,
                config=self.config,
            )
            update_second_layer_from_statistics(
                block=block_view,
                moments=second_moments,
                config=self.config,
            )

            first_moments = collect_first_layer_statistics(
                block=block_view,
                dataloader=dataloader,
                adapter=self.adapter,
                config=self.config,
            )
            update_first_layer_from_statistics(
                block=block_view,
                moments=first_moments,
                config=self.config,
            )

        second_moments = collect_second_layer_statistics(
            block=block_view,
            dataloader=dataloader,
            adapter=self.adapter,
            config=self.config,
        )
        update_second_layer_from_statistics(
            block=block_view,
            moments=second_moments,
            config=self.config,
        )

        if self.config.verbose:
            print(
                f"[{block_view.name}] optimization done: "
                f"||A||={block_view.first_linear.weight.norm().item():.4f}, "
                f"||Omega||={block_view.second_linear.weight.norm().item():.4f}"
            )

        return OptimizationResult(
            block_name=block_view.name,
            nb_iterations=self.config.nb_iterations,
            omega_norm=block_view.second_linear.weight.norm().item(),
            a_norm=block_view.first_linear.weight.norm().item(),
        )


def optimize(
    block: nn.Module | TwoLayerLinearBlockView,
    dataloader: list[Any] | Any,
    adapter: BatchAdapter | None = None,
    config: ClosedFormOptimizationConfig | None = None,
) -> OptimizationResult:
    """
    Convenience wrapper around ``ClosedFormBlockOptimizer.optimize``.
    """
    optimizer = ClosedFormBlockOptimizer(config=config, adapter=adapter)
    return optimizer.optimize(block, dataloader)
