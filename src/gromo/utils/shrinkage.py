"""Shrinkage of square matrices towards a multiple of the identity.

The shrinkage of a d x d matrix M with intensity tau in [0, 1] is the matrix::

    (1 - tau) * M + tau * tr(M) / d * I.

It keeps the trace, and for a positive semi-definite M it raises the smallest
eigenvalue to at least tau * tr(M) / d.

Provide:

- `fixed_shrinkage`, the shrinkage with a given intensity;
- `weighted_ledoit_wolf` and `LedoitWolfEstimate`, the Ledoit-Wolf estimate of the
  intensity for a weighted mean of matrices;
- `ReduceMoments` and `reduce_fourth_moment`, the one-pass accumulation of the
  weighted fourth moment that `weighted_ledoit_wolf` needs when each matrix is rank
  one (K-FAC-reduce).
"""

from dataclasses import dataclass

import torch


def fixed_shrinkage(matrix: torch.Tensor, intensity: float) -> torch.Tensor:
    """Shrink a square matrix with a given intensity.

    Return (1 - alpha) * M + alpha * tr(M) / d * I. For a positive semi-definite M
    of positive trace and alpha > 0, the result is positive definite (hence
    invertible), even when the empirical spectrum of M sits at or below the usual
    whitening threshold.

    Parameters
    ----------
    matrix: torch.Tensor
        matrix M, of shape (d, d)
    intensity: float
        shrinkage intensity alpha in [0, 1]

    Returns
    -------
    torch.Tensor
        the shrunk matrix, of the dtype and device of M

    Raises
    ------
    ValueError
        if the intensity is not in [0, 1]
    """
    if not 0.0 <= intensity <= 1.0:
        raise ValueError(f"The intensity must be in [0, 1], got {intensity}.")
    d = matrix.shape[0]
    return (1 - intensity) * matrix + intensity * (matrix.trace() / d) * torch.eye(
        d, device=matrix.device, dtype=matrix.dtype
    )


@dataclass(frozen=True)
class LedoitWolfEstimate:
    """Result of `weighted_ledoit_wolf`.

    Attributes
    ----------
    matrix: torch.Tensor
        shrunk matrix tau * lambda * I + (1 - tau) * S, of the dtype and device of S
    intensity: float
        shrinkage intensity tau, in [0, 1]
    mean_eigenvalue: float
        mean eigenvalue lambda = tr(S) / d
    beta2: float
        estimate beta^2 of the squared error of S, clipped to [0, delta^2]
    delta2: float
        squared distance delta^2 = ||S - lambda I||_F^2 of S to lambda I
    effective_samples: float
        effective number of samples n_eff = (sum_n nu_n)^2 / sum_n nu_n^2
    """

    matrix: torch.Tensor
    intensity: float
    mean_eigenvalue: float
    beta2: float
    delta2: float
    effective_samples: float


def weighted_ledoit_wolf(
    matrix: torch.Tensor,
    sum_weights: float,
    sum_squared_weights: float,
    weighted_fourth_moment: float,
    dtype: torch.dtype = torch.float64,
) -> LedoitWolfEstimate:
    """Ledoit-Wolf shrinkage of a weighted mean of symmetric matrices.

    The matrix is the weighted mean S = sum_n nu_n Z_n / sum_n nu_n of symmetric
    d x d matrices Z_n, with weights nu_n > 0. It is shrunk towards lambda * I with
    the intensity of Ledoit and Wolf (2004), generalised to weights::

        lambda = tr(S) / d,
        delta^2 = ||S||_F^2 - d * lambda^2 = ||S - lambda I||_F^2,
        beta^2 = clip((sum_n nu_n^2 ||Z_n||_F^2 - (sum_n nu_n^2) ||S||_F^2)
                      / (sum_n nu_n)^2, 0, delta^2),
        tau = beta^2 / delta^2 (and tau = 0 if delta^2 = 0),
        LW = tau * lambda * I + (1 - tau) * S.

    When the Z_n are independent with the same mean and the weights are fixed, the
    unclipped beta^2 estimates the squared error E||S - E[S]||_F^2 of S, biased low
    by the factor 1 - 1 / n_eff, where n_eff = (sum_n nu_n)^2 / sum_n nu_n^2 is the
    effective number of samples.

    With nu_n = 1, this is the estimator of Ledoit and Wolf (2004) for the N
    matrices Z_n: the numerator of beta^2 divided by N^2 is
    sum_n ||Z_n - S||_F^2 / N^2. Their norm ||.||_F^2 / d divides delta^2 and
    beta^2 by d, and leaves tau unchanged.

    Centring is the caller's job: the Z_n are centred by the caller, and S and the
    fourth moment must use the same centre.

    The trace and the squared Frobenius norm of S are computed in `dtype`, without a
    copy of S when it already has this dtype, and the rest in Python floats. In a
    dtype of precision eps, delta^2 = ||S||_F^2 - d * lambda^2 has an absolute error
    of about eps * ||S||_F^2: it matters only when S is close to lambda I.

    Parameters
    ----------
    matrix: torch.Tensor
        weighted mean S, of shape (d, d)
    sum_weights: float
        sum_n nu_n
    sum_squared_weights: float
        sum_n nu_n^2
    weighted_fourth_moment: float
        sum_n nu_n^2 ||Z_n||_F^2, e.g. from `reduce_fourth_moment`
    dtype: torch.dtype
        dtype of the trace and of the squared Frobenius norm of S

    Returns
    -------
    LedoitWolfEstimate
        the shrunk matrix, of the dtype and device of S, and the scalars of the
        estimate

    Raises
    ------
    ValueError
        if the matrix is not a non-empty square matrix, or if a sum of weights is
        not positive
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(
            f"The matrix must be a non-empty square matrix, got shape "
            f"{tuple(matrix.shape)}."
        )
    if not (sum_weights > 0 and sum_squared_weights > 0):
        raise ValueError(
            f"The sums of weights must be positive, got {sum_weights=} and "
            f"{sum_squared_weights=}."
        )
    d = matrix.shape[0]
    # `to` returns the matrix itself when it already has the dtype, and the norm is
    # a reduction: no d x d tensor is allocated in that case.
    matrix_cast = matrix.to(dtype)
    mean_eigenvalue = float(matrix_cast.trace()) / d
    squared_norm = float(torch.linalg.vector_norm(matrix_cast)) ** 2
    # ||S - lambda I||_F^2 >= 0, but the subtraction can round below 0.
    delta2 = max(squared_norm - d * mean_eigenvalue**2, 0.0)
    unclipped_beta2 = (
        weighted_fourth_moment - sum_squared_weights * squared_norm
    ) / sum_weights**2
    beta2 = min(max(unclipped_beta2, 0.0), delta2)
    intensity = beta2 / delta2 if delta2 > 0 else 0.0
    return LedoitWolfEstimate(
        matrix=fixed_shrinkage(matrix, intensity),
        intensity=intensity,
        mean_eigenvalue=mean_eigenvalue,
        beta2=beta2,
        delta2=delta2,
        effective_samples=sum_weights**2 / sum_squared_weights,
    )


@dataclass
class ReduceMoments:
    """Accumulators of the weighted fourth moment of rank-one matrices.

    For vectors x_n in R^d with weights nu_n, and w_n = nu_n^2, accumulate in one
    pass sum_n nu_n and the sums::

        s_0 = sum_n w_n,                  s_1 = sum_n w_n x_n,
        s_2 = sum_n w_n ||x_n||^2,        s_3 = sum_n w_n ||x_n||^2 x_n,
        s_4 = sum_n w_n ||x_n||^4,        Q = sum_n w_n x_n x_n^T.

    From them, `reduce_fourth_moment` gives sum_n nu_n^2 ||x_n - c||^4 for a centre
    c known only at the end of the pass. It is the weighted fourth moment
    sum_n nu_n^2 ||Z_n||_F^2 of Z_n = (x_n - c)(x_n - c)^T that `weighted_ledoit_wolf`
    needs. In K-FAC-reduce, each sequence gives one such rank-one matrix.

    The vectors can be given relative to any origin, with the centre relative to the
    same origin. An origin close to their mean (e.g. a shift estimated on the first
    batch) keeps the terms of the formula from cancelling, which matters in float32.

    Attributes
    ----------
    sum_weights: torch.Tensor
        sum_n nu_n, of shape ()
    sum_squared_weights: torch.Tensor
        s_0, of shape ()
    sum_vectors: torch.Tensor
        s_1, of shape (d,)
    sum_squared_norms: torch.Tensor
        s_2, of shape ()
    sum_squared_norm_vectors: torch.Tensor
        s_3, of shape (d,)
    sum_fourth_powers: torch.Tensor
        s_4, of shape ()
    sum_outer_products: torch.Tensor
        Q, of shape (d, d)
    """

    sum_weights: torch.Tensor
    sum_squared_weights: torch.Tensor
    sum_vectors: torch.Tensor
    sum_squared_norms: torch.Tensor
    sum_squared_norm_vectors: torch.Tensor
    sum_fourth_powers: torch.Tensor
    sum_outer_products: torch.Tensor

    @classmethod
    def zeros(
        cls,
        dimension: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> "ReduceMoments":
        """Create empty accumulators.

        Parameters
        ----------
        dimension: int
            dimension d of the vectors
        dtype: torch.dtype
            dtype of the accumulators
        device: torch.device | str | None
            device of the accumulators

        Returns
        -------
        ReduceMoments
            accumulators filled with zeros
        """
        return cls(
            sum_weights=torch.zeros((), dtype=dtype, device=device),
            sum_squared_weights=torch.zeros((), dtype=dtype, device=device),
            sum_vectors=torch.zeros(dimension, dtype=dtype, device=device),
            sum_squared_norms=torch.zeros((), dtype=dtype, device=device),
            sum_squared_norm_vectors=torch.zeros(dimension, dtype=dtype, device=device),
            sum_fourth_powers=torch.zeros((), dtype=dtype, device=device),
            sum_outer_products=torch.zeros(
                dimension, dimension, dtype=dtype, device=device
            ),
        )

    def add(self, vectors: torch.Tensor, weights: torch.Tensor) -> None:
        """Accumulate one batch of weighted vectors.

        The terms are computed in the dtype of the vectors, and added to the sums in
        the dtype of the accumulators.

        Parameters
        ----------
        vectors: torch.Tensor
            vectors x_n, of shape (n, d)
        weights: torch.Tensor
            weights nu_n (not squared), of shape (n,)

        Raises
        ------
        ValueError
            if the shapes of the vectors and of the weights are inconsistent with
            each other or with d
        """
        dimension = self.sum_vectors.shape[0]
        if vectors.ndim != 2 or vectors.shape[1] != dimension:
            raise ValueError(
                f"The vectors must be of shape (n, {dimension}), got "
                f"{tuple(vectors.shape)}."
            )
        if weights.shape != vectors.shape[:1]:
            raise ValueError(
                f"The weights must be of shape ({vectors.shape[0]},), got "
                f"{tuple(weights.shape)}."
            )
        weights = weights.to(dtype=vectors.dtype, device=vectors.device)
        squared_weights = weights.square()
        squared_norms = vectors.square().sum(dim=1)
        for accumulator, term in (
            (self.sum_weights, weights.sum()),
            (self.sum_squared_weights, squared_weights.sum()),
            (self.sum_vectors, squared_weights @ vectors),
            (self.sum_squared_norms, squared_weights @ squared_norms),
            (self.sum_squared_norm_vectors, (squared_weights * squared_norms) @ vectors),
            (self.sum_fourth_powers, squared_weights @ squared_norms.square()),
            (self.sum_outer_products, vectors.T @ (squared_weights[:, None] * vectors)),
        ):
            accumulator.add_(term.to(accumulator.dtype))


def reduce_fourth_moment(
    moments: ReduceMoments,
    centre: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Weighted fourth moment of the vectors around a centre.

    Return sum_n nu_n^2 ||x_n - c||^4, from the accumulators of `ReduceMoments`
    (with w_n = nu_n^2), by the formula::

        sum_n w_n ||x_n - c||^4 = s_4 - 4 c^T s_3 + 2 ||c||^2 s_2 + 4 c^T Q c
                                  - 4 ||c||^2 (c^T s_1) + ||c||^4 s_0.

    The vectors and the centre are relative to the same origin. The formula is
    evaluated in `dtype`, without a copy of the accumulators when they already have
    this dtype. Its terms cancel when the centre is far from the origin compared with
    the spread of the vectors: float64 and an origin close to the mean of the vectors
    limit the rounding error.

    Parameters
    ----------
    moments: ReduceMoments
        accumulators of the vectors x_n and their weights nu_n
    centre: torch.Tensor
        centre c, of shape (d,)
    dtype: torch.dtype
        dtype of the evaluation

    Returns
    -------
    float
        sum_n nu_n^2 ||x_n - c||^4

    Raises
    ------
    ValueError
        if the centre is not of shape (d,)
    """
    if centre.shape != moments.sum_vectors.shape:
        raise ValueError(
            f"The centre must be of shape {tuple(moments.sum_vectors.shape)}, got "
            f"{tuple(centre.shape)}."
        )
    c = centre.to(dtype=dtype, device=moments.sum_vectors.device)
    squared_norm_c = c @ c
    fourth_moment = (
        moments.sum_fourth_powers.to(dtype)
        - 4 * (c @ moments.sum_squared_norm_vectors.to(dtype))
        + 2 * squared_norm_c * moments.sum_squared_norms.to(dtype)
        + 4 * (c @ (moments.sum_outer_products.to(dtype) @ c))
        - 4 * squared_norm_c * (c @ moments.sum_vectors.to(dtype))
        + squared_norm_c**2 * moments.sum_squared_weights.to(dtype)
    )
    return float(fourth_moment)
