import math
from collections.abc import Callable
from typing import Any, Literal
from warnings import warn

import torch

from gromo.utils.tensor_statistic import TensorStatistic


# A threshold rule maps the statistic that produced a matrix and the spectrum of that
# matrix to a threshold. The caller that owns the statistic partially applies the rule,
# leaving a SpectrumThreshold for the numerical helpers below.
SpectrumThreshold = Callable[[torch.Tensor], float]
ThresholdRule = Callable[[TensorStatistic, torch.Tensor], float]

KnownThresholdRuleName = Literal[
    "mean_over_sqrt_n",
    "operator_norm_noise_threshold",
]


def pytorch_pinv_threshold(spectrum: torch.Tensor) -> float:
    """PyTorch's default `pinv` / `matrix_rank` tolerance for a symmetric matrix.

    For a symmetric d x d matrix, it is d * eps * max(lambda_1, 0), with eps the
    epsilon of the dtype of the spectrum: in the worst case, the eigensolver cannot
    tell an eigenvalue below it apart from zero. For a rectangular matrix, PyTorch
    uses max(m, n) instead of the number of singular values. It is the first term of
    `numerical_floor_terms`, not a `ThresholdRule`.

    Parameters
    ----------
    spectrum: torch.Tensor
        eigenvalues of the symmetric matrix

    Returns
    -------
    float
        the tolerance, 0.0 for an empty spectrum
    """
    if spectrum.numel() == 0:
        return 0.0
    eps = torch.finfo(spectrum.dtype).eps
    return spectrum.numel() * eps * max(spectrum.max().item(), 0.0)


def numerical_floor_terms(
    spectrum: torch.Tensor,
    *,
    source_dtype: torch.dtype | None = None,
    worst_case: bool = True,
) -> dict[str, float]:
    """Terms of the numerical floor under every eigenvalue threshold of S and E.

    The floor is the maximum of the terms. It is the library default (the
    `pinv` / `matrix_rank` tolerance of MATLAB, NumPy, SciPy and PyTorch) plus a
    correction for the uncertainty of the data, as NumPy's `matrix_rank` docstring
    recommends when the data are less precise than the arithmetic:

    - ``"worst_case"``: `pytorch_pinv_threshold`, the worst case of the eigensolver
      error, conservative by about d. In float32 it caps the gain of the inverse
      square root at (d * eps)^(-1/2) times that of the top direction (about 90 for
      d = 1000). 0.0 when `worst_case` is False.
    - ``"precision"``: eps * max(lambda_1, 0), one ulp of the largest eigenvalue in
      the less precise of `source_dtype` and the dtype of the spectrum. It covers
      small null spaces, where the next term has nothing to read, and statistics
      cast to a more precise dtype. Without the factor d, it never reaches lambda_1,
      even in bfloat16.
    - ``"negative_eigenvalue"``: 2 * max(-lambda_min, 0). S and E are positive
      semi-definite, so their negative eigenvalues are noise, and the noise is
      roughly symmetric around zero: this measures it on the matrix itself, whatever
      the dtypes and the accumulation history. The factor 2 bounds the relative error
      of every kept eigenvalue by 1/2.

    Parameters
    ----------
    spectrum: torch.Tensor
        eigenvalues of the symmetric matrix
    source_dtype: torch.dtype | None
        dtype the matrix was accumulated in. When None, the dtype of the spectrum
        is used.
    worst_case: bool
        whether to include the worst-case term

    Returns
    -------
    dict[str, float]
        the three terms, all 0.0 for an empty spectrum
    """
    if spectrum.numel() == 0:
        return {"worst_case": 0.0, "precision": 0.0, "negative_eigenvalue": 0.0}
    smallest, largest = torch.stack(torch.aminmax(spectrum)).tolist()  # one sync
    top = max(largest, 0.0)
    inversion_eps = torch.finfo(spectrum.dtype).eps
    eps = inversion_eps
    if source_dtype is not None:
        eps = max(eps, torch.finfo(source_dtype).eps)
    return {
        "worst_case": spectrum.numel() * inversion_eps * top if worst_case else 0.0,
        "precision": eps * top,
        "negative_eigenvalue": 2 * max(-smallest, 0.0),
    }


def _mean_over_sqrt_n_rule(statistic: TensorStatistic, spectrum: torch.Tensor) -> float:
    """Mean of the spectrum divided by the square root of the number of samples.

    Parameters
    ----------
    statistic: TensorStatistic
        statistic the thresholded matrix was estimated from
    spectrum: torch.Tensor
        spectrum of that matrix

    Returns
    -------
    float
        the threshold
    """
    return spectrum.mean().item() / math.sqrt(max(statistic.samples, 1))


def _operator_norm_noise_threshold_rule(
    statistic: TensorStatistic, spectrum: torch.Tensor
) -> float:
    """Estimate of the operator norm of the estimation error of a covariance.

    2 * sqrt(lambda_1 * Tr / n) + Tr / n estimates E||C_hat - C||_op
    (Koltchinskii-Lounici; the constants are exact for an isotropic Gaussian). It is
    a conservative absolute floor, not a recommended default: the estimation noise of
    a covariance is multiplicative, so eigenvalues below it can be well estimated.

    n is ``statistic.samples``, which counts images for convolutions and sequences
    for sequence inputs. Each counted sample then aggregates several outer products,
    so the rule over-estimates the noise, by up to sqrt(patches per image). In the
    uncentred S, the bias direction dominates lambda_1.

    The rule is meant for `numerical_threshold`, i.e. for S and E. On
    `statistical_threshold` the spectrum is made of singular values of P, for which
    the formula is meaningless.

    Parameters
    ----------
    statistic: TensorStatistic
        statistic the thresholded matrix was estimated from
    spectrum: torch.Tensor
        spectrum of that matrix

    Returns
    -------
    float
        the threshold
    """
    n = max(statistic.samples, 1)
    trace = spectrum.clamp_min(0).sum().item()
    top = max(spectrum.max().item(), 0.0)
    return 2 * math.sqrt(top * trace / n) + trace / n


KNOWN_THRESHOLD_RULES: dict[KnownThresholdRuleName, ThresholdRule] = {
    "mean_over_sqrt_n": _mean_over_sqrt_n_rule,
    "operator_norm_noise_threshold": _operator_norm_noise_threshold_rule,
}


def resolve_threshold_rule(
    rule: KnownThresholdRuleName | ThresholdRule,
) -> ThresholdRule:
    """
    Get the threshold rule designated by a name, or the rule itself.

    Parameters
    ----------
    rule: KnownThresholdRuleName | ThresholdRule
        name of a rule of `KnownThresholdRuleName`, or a rule

    Returns
    -------
    ThresholdRule
        the resolved rule

    Raises
    ------
    ValueError
        if the name is not that of a known rule
    """
    if callable(rule):
        return rule
    if rule not in KNOWN_THRESHOLD_RULES:
        raise ValueError(
            f"Unknown threshold rule '{rule}'. "
            f"Available rules are: {list(KNOWN_THRESHOLD_RULES)}."
        )
    return KNOWN_THRESHOLD_RULES[rule]


def resolve_threshold(
    threshold: float | SpectrumThreshold,
    spectrum: torch.Tensor,
    fallback: float = 0.0,
) -> float:
    """
    Get the value a threshold takes for a given spectrum.

    An empty spectrum and a non-finite rule value both resolve to a fallback, as
    either would otherwise select nothing.

    Parameters
    ----------
    threshold: float | SpectrumThreshold
        a value, or a rule already bound to its statistic
    spectrum: torch.Tensor
        spectrum the threshold is compared against
    fallback: float
        value used when the rule returns a non-finite number

    Returns
    -------
    float
        the resolved threshold
    """
    if not callable(threshold):
        return float(threshold)
    if spectrum.numel() == 0:
        return 0.0
    value: float = float(threshold(spectrum))  # type: ignore
    if not math.isfinite(value):
        warn(
            message=f"The threshold rule returned {value}, falling back to {fallback}.",
            category=RuntimeWarning,
        )
        return fallback
    return value


def _kept_eigenpairs(
    matrix: torch.Tensor,
    threshold: float | SpectrumThreshold | None,
    *,
    spectra: dict[str, Any] | None,
    source_dtype: torch.dtype | None,
    worst_case_numerical_floor: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenpairs of a symmetric matrix above a threshold raised to the numerical floor.

    See `sqrt_inverse_matrix_semi_positive` for the parameters.

    Parameters
    ----------
    matrix: torch.Tensor
        symmetric matrix
    threshold: float | SpectrumThreshold | None
        threshold, None for the numerical floor only
    spectra: dict[str, Any] | None
        filled in place when given
    source_dtype: torch.dtype | None
        dtype the matrix was accumulated in
    worst_case_numerical_floor: bool
        whether the floor includes its worst-case term

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        the kept eigenvalues and their eigenvectors (as columns)
    """
    regularized = False
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    except torch.linalg.LinAlgError:
        # Sometimes, due to numerical issues, we get an error:
        # The algorithm failed to converge because the input matrix is
        # ill-conditioned or has too many repeated eigenvalues
        regularized = True
        # Out of place, so that the caller's matrix is left untouched.
        matrix = matrix + torch.finfo(matrix.dtype).resolution * torch.eye(
            matrix.shape[0],
            device=matrix.device,
            dtype=matrix.dtype,
        )
        warn(
            message="Adding a small identity matrix to make the input matrix positive definite.",
            category=RuntimeWarning,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    floor_terms = numerical_floor_terms(
        eigenvalues, source_dtype=source_dtype, worst_case=worst_case_numerical_floor
    )
    numerical_floor = max(floor_terms.values())
    # Every threshold, value or rule, can only raise the floor.
    if threshold is None:
        threshold = numerical_floor
    else:
        threshold = max(
            resolve_threshold(threshold, eigenvalues, fallback=numerical_floor),
            numerical_floor,
        )
    selected_eigenvalues = eigenvalues > threshold
    if (
        not selected_eigenvalues.any()
        and eigenvalues.numel() > 0
        and eigenvalues.max() > 0
    ):
        if threshold > numerical_floor:
            # The texts of these warnings are constant, so that the default filter
            # shows each once per call site; the values are in `spectra`.
            warn(
                message=(
                    "A threshold drops the whole spectrum of a non-zero matrix, which "
                    "would make its inverse zero. Falling back to the numerical floor."
                ),
                category=RuntimeWarning,
            )
            threshold = numerical_floor
            selected_eigenvalues = eigenvalues > threshold
        if not selected_eigenvalues.any():
            warn(
                message=(
                    "The numerical floor drops the whole spectrum: the matrix is far "
                    "from positive semi-definite (its most negative eigenvalue is at "
                    "least half its largest one), so its inverse is set to zero."
                ),
                category=RuntimeWarning,
            )

    if spectra is not None:
        spectra.update(
            eigenvalues=eigenvalues.detach(),
            threshold=threshold,
            numerical_floor=numerical_floor,
            numerical_floor_terms=floor_terms,
            kept=int(selected_eigenvalues.sum()),
            total=int(eigenvalues.numel()),
            regularized=regularized,
        )
    return eigenvalues[selected_eigenvalues], eigenvectors[:, selected_eigenvalues]


def sqrt_inverse_matrix_semi_positive(
    matrix: torch.Tensor,
    threshold: float | SpectrumThreshold | None = None,
    *,
    spectra: dict[str, Any] | None = None,
    source_dtype: torch.dtype | None = None,
    worst_case_numerical_floor: bool = True,
) -> torch.Tensor:
    """
    Compute the square root of the inverse of a semi-positive definite matrix.

    Eigenvalues at or below the threshold are treated as zero. Every threshold is
    raised to the numerical floor (see `numerical_floor_terms`).

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix, square and semi-positive definite
    threshold: float | SpectrumThreshold | None
        threshold to consider an eigenvalue as zero: None for the numerical floor
        only, a value, or a rule already bound to its statistic (see
        `resolve_threshold`). A value or a rule can only raise the floor, and falls
        back to it when it would drop the whole spectrum of a non-zero matrix.
    spectra: dict[str, Any] | None
        if given, filled in place with the eigenvalues of the input matrix, the
        threshold applied to them, the numerical floor and its terms, the number of
        eigenvalues kept, the total number of eigenvalues, and whether the matrix
        had to be regularized. Nothing is computed when None.
    source_dtype: torch.dtype | None
        dtype the matrix was accumulated in, when it was cast before this call. The
        numerical floor uses the epsilon of the less precise of this dtype and the
        dtype of the matrix. When None, the dtype of the matrix is used.
    worst_case_numerical_floor: bool
        whether the numerical floor includes its worst-case term, the default
        tolerance of `torch.linalg.pinv` (see `numerical_floor_terms`)

    Returns
    -------
    torch.Tensor
        square root of the inverse of the input matrix
    """
    assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."
    assert torch.allclose(matrix, matrix.t()), "The input matrix must be symmetric."
    assert torch.isnan(matrix).sum() == 0, "The input matrix must not contain NaN values."
    eigenvalues, eigenvectors = _kept_eigenpairs(
        matrix,
        threshold,
        spectra=spectra,
        source_dtype=source_dtype,
        worst_case_numerical_floor=worst_case_numerical_floor,
    )
    return eigenvectors @ torch.diag(torch.rsqrt(eigenvalues)) @ eigenvectors.t()


def pseudo_inverse_matrix_semi_positive(
    matrix: torch.Tensor,
    *,
    spectra: dict[str, Any] | None = None,
    source_dtype: torch.dtype | None = None,
    worst_case_numerical_floor: bool = True,
) -> torch.Tensor:
    """
    Compute the pseudo-inverse of a semi-positive definite matrix.

    Unlike `torch.linalg.pinv`, which selects eigenvalues by magnitude, it drops
    the eigenvalues at or below the numerical floor (see `numerical_floor_terms`),
    negative ones included: a negative eigenvalue of a semi-positive definite matrix
    is noise, and inverting it would flip the sign of the result along its
    direction. Without a cast and without negative eigenvalues, the result equals
    that of `torch.linalg.pinv` up to rounding.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix, square and semi-positive definite; it is symmetrized
    spectra: dict[str, Any] | None
        if given, filled in place as by `sqrt_inverse_matrix_semi_positive`
    source_dtype: torch.dtype | None
        dtype the matrix was accumulated in, when it was cast before this call
    worst_case_numerical_floor: bool
        whether the numerical floor includes its worst-case term

    Returns
    -------
    torch.Tensor
        pseudo-inverse of the input matrix
    """
    assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."
    eigenvalues, eigenvectors = _kept_eigenpairs(
        (matrix + matrix.t()) / 2,
        None,
        spectra=spectra,
        source_dtype=source_dtype,
        worst_case_numerical_floor=worst_case_numerical_floor,
    )
    return eigenvectors @ torch.diag(eigenvalues.reciprocal()) @ eigenvectors.t()


def optimal_delta(
    tensor_s: torch.Tensor,
    tensor_m: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    force_pseudo_inverse: bool = False,
    tensor_covariance_loss_gradient: torch.Tensor | None = None,
    spectra: dict[str, Any] | None = None,
    *,
    source_dtype: torch.dtype | None = None,
    worst_case_numerical_floor: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the optimal delta for the layer using current S and M tensors.

    :math:`dW^* = (S[-1]^-1 M)^T` (if needed we use the pseudo-inverse). When the empirical
    Fisher / gradient covariance E_s is provided via
    ``tensor_covariance_loss_gradient``, the natural-gradient-like preconditioned
    update is used instead: :math:`dW^* = (S^-1 M E_s^-1)^T = E_s^-1 M^T S^-1`.

    Compute dW* (and dBias* if needed).
    L(A + gamma * B * dW) = L(A) - gamma * d + o(gamma)
    where d is the first order decrease and gamma the scaling factor.

    Parameters
    ----------
    tensor_s: torch.Tensor
        S tensor from calling layer, of shape [total_in_features, total_in_features]
    tensor_m: torch.Tensor
        M tensor from calling layer, of shape [total_in_features, in_features]
    dtype: torch.dtype
        dtype for S and M during the computation, by default torch.float32
    force_pseudo_inverse: bool
        if True, use the pseudo-inverse to compute the optimal delta even if the
        matrix is invertible, by default False
    tensor_covariance_loss_gradient: torch.Tensor | None
        empirical Fisher E_s of shape (out_features, out_features). When provided
        the preconditioned update dW* = E_s^-1 M^T S^-1 is returned. Note that
        relying on this preconditioner silently uses the independence hypothesis
        described in `first_order_optimization.typ` (`@hyp:independence`).
    spectra: dict[str, Any] | None
        if given, filled in place with the singular values of the returned optimal
        delta. This is an additional decomposition, hence opt-in: nothing is
        computed when None.
    source_dtype: torch.dtype | None
        dtype the statistics were accumulated in, when the tensors were cast before
        this call. The numerical floor of the pseudo-inverses uses the epsilon of the
        less precise of this dtype and ``dtype``, as their rounding noise is not zero
        after a cast to a more precise dtype. When None, the dtype of ``tensor_s`` is
        used.
    worst_case_numerical_floor: bool
        whether the numerical floor of the pseudo-inverses includes its worst-case
        term (see `numerical_floor_terms`). The pseudo-inverses are computed by
        `pseudo_inverse_matrix_semi_positive`, only when a matrix is singular, when
        ``force_pseudo_inverse`` is True, or in the float64 retry.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        the optimal delta weights and the first order decrease
    """
    # Ensure both tensors have the same dtype initially
    assert tensor_s.dtype == tensor_m.dtype, (
        f"Both input tensors must have the same dtype, "
        f"got tensor_s.dtype={tensor_s.dtype} and tensor_m.dtype={tensor_m.dtype}"
    )

    saved_dtype = tensor_s.dtype
    if source_dtype is None:
        source_dtype = saved_dtype
    # The cast to dtype rounds too: the less precise dtype bounds the precision. The
    # retry below receives the cast tensors, so it inherits this dtype.
    source_dtype = max(source_dtype, dtype, key=lambda t: torch.finfo(t).eps)
    if tensor_s.dtype != dtype:
        tensor_s = tensor_s.to(dtype=dtype)
    if tensor_m.dtype != dtype:
        tensor_m = tensor_m.to(dtype=dtype)
    if (
        tensor_covariance_loss_gradient is not None
        and tensor_covariance_loss_gradient.dtype != dtype
    ):
        tensor_covariance_loss_gradient = tensor_covariance_loss_gradient.to(dtype=dtype)

    delta_raw = None
    if not force_pseudo_inverse:
        try:
            delta_raw = torch.linalg.solve(tensor_s, tensor_m).t()
        except torch.linalg.LinAlgError:
            force_pseudo_inverse = True
            # self.delta_raw = torch.linalg.lstsq(tensor_s, tensor_m).solution.t()
            # do not use lstsq because it does not work with the GPU
            warn("Using the pseudo-inverse for the computation of the optimal delta.")
    if force_pseudo_inverse:
        delta_raw = (
            pseudo_inverse_matrix_semi_positive(
                tensor_s,
                source_dtype=source_dtype,
                worst_case_numerical_floor=worst_case_numerical_floor,
            )
            @ tensor_m
        ).t()

    assert delta_raw is not None, "delta_raw should be computed by now."

    if tensor_covariance_loss_gradient is not None:
        applied_pinv = force_pseudo_inverse
        if not applied_pinv:
            try:
                delta_raw = torch.linalg.solve(tensor_covariance_loss_gradient, delta_raw)
            except torch.linalg.LinAlgError:
                applied_pinv = True
                warn(
                    "Using the pseudo-inverse for the gradient covariance preconditioner."
                )
        if applied_pinv:
            delta_raw = (
                pseudo_inverse_matrix_semi_positive(
                    tensor_covariance_loss_gradient,
                    source_dtype=source_dtype,
                    worst_case_numerical_floor=worst_case_numerical_floor,
                )
                @ delta_raw
            )

    assert delta_raw.isnan().sum() == 0, (
        "The optimal delta should not contain NaN values."
    )
    parameter_update_decrease = torch.trace(tensor_m @ delta_raw)
    if parameter_update_decrease < 0:
        warn(
            "The parameter update decrease should be positive, "
            f"but got {parameter_update_decrease=} for layer."
        )
        if not force_pseudo_inverse:
            warn("Trying to use the pseudo-inverse with torch.float64.")
            # The cast to saved_dtype and the spectra below also cover the retry.
            delta_raw, parameter_update_decrease = optimal_delta(
                tensor_s,
                tensor_m,
                dtype=torch.float64,
                force_pseudo_inverse=True,
                tensor_covariance_loss_gradient=tensor_covariance_loss_gradient,
                source_dtype=source_dtype,
                worst_case_numerical_floor=worst_case_numerical_floor,
            )
        else:
            warn("Failed to compute the optimal delta, set delta to zero.")
            delta_raw.fill_(0)
            parameter_update_decrease.fill_(0)
    delta_raw = delta_raw.to(dtype=saved_dtype)
    if isinstance(parameter_update_decrease, torch.Tensor):
        parameter_update_decrease = parameter_update_decrease.to(dtype=saved_dtype)

    if spectra is not None:
        spectra["singular_values"] = torch.linalg.svdvals(delta_raw).detach()

    return delta_raw, parameter_update_decrease


def compute_optimal_added_parameters(
    matrix_s: torch.Tensor | None,
    matrix_n: torch.Tensor,
    numerical_threshold: float | SpectrumThreshold | None = None,
    statistical_threshold: float | SpectrumThreshold = 1e-3,
    maximum_added_neurons: int | None = None,
    alpha_zero: bool = False,
    omega_zero: bool = False,
    ignore_singular_values: bool = False,
    matrix_covariance_loss_gradient: torch.Tensor | None = None,
    e_numerical_threshold: float | SpectrumThreshold | None = None,
    spectra: dict[str, Any] | None = None,
    *,
    source_dtype: torch.dtype | None = None,
    worst_case_numerical_floor: bool = True,
    e_worst_case_numerical_floor: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the optimal added parameters for a given layer.

    This function operates on primitive options, not method names.

    Parameters
    ----------
    matrix_s : torch.Tensor | None
        Square matrix S of shape (s, s). If None, identity matrix is used.
    matrix_n : torch.Tensor
        Matrix N (correlation matrix) of shape (s, t).
    numerical_threshold : float | SpectrumThreshold | None
        Threshold to consider an eigenvalue as zero in square root of inverse of S:
        None for the numerical floor only; a value or a rule can only raise the
        floor (see `sqrt_inverse_matrix_semi_positive`).
    statistical_threshold : float | SpectrumThreshold
        Threshold to consider a singular value as zero in the SVD
    maximum_added_neurons : int | None
        Maximum number of added neurons, if None all significant neurons are kept
    alpha_zero : bool
        If True, set alpha (incoming weights) to zero, else compute from SVD.
    omega_zero : bool
        If True, set omega (outgoing weights) to zero, else compute from SVD.
    ignore_singular_values : bool
        If True, ignore the actual singular values and treat them as 1 for computing alpha and
        omega, effectively only using the singular vectors for the update direction.
    matrix_covariance_loss_gradient : torch.Tensor | None
        Square matrix E_s of shape (t, t). If provided, the SVD target becomes
        S^{-1/2} N E_s^{-1/2} and omega is left-multiplied by E_s^{-1/2}, which
        applies the empirical-Fisher preconditioning to the rank-k extension.
        Note that this silently uses the independence hypothesis described in
        `first_order_optimization.typ` (`@hyp:independence`).
    e_numerical_threshold : float | SpectrumThreshold | None
        Whitening threshold for E_s. When None, `numerical_threshold` is used.
        Like every threshold, it can only raise the numerical floor: 0.0 means the
        floor only.
    spectra : dict[str, Any] | None
        If given, filled in place with the keys "matrix_s", "matrix_e" and
        "extension". The first two hold the whitening spectra of S and E (None
        when the corresponding matrix is not used, see
        `sqrt_inverse_matrix_semi_positive`); "extension" holds the singular
        values of the SVD target before any selection, the threshold applied to
        them, and how many were kept by the threshold and by
        `maximum_added_neurons`. Nothing is computed when None.
    source_dtype : torch.dtype | None
        dtype S, N and E were accumulated in, when they were cast before this call.
        The numerical floor of both whitenings uses the epsilon of the less precise
        of this dtype and the dtype of the matrix. When None, the dtype of each
        matrix is used.
    worst_case_numerical_floor : bool
        Whether the numerical floor of the whitening of S includes its worst-case
        term, the default tolerance of `torch.linalg.pinv` (see
        `numerical_floor_terms`).
    e_worst_case_numerical_floor : bool | None
        The same for E_s. When None, `worst_case_numerical_floor` is used. Pass
        False when E_s has been ridge-shrunk upstream: its eigenvalues are then
        real, and the worst-case term could cut them.

    Returns
    -------
    torch.Tensor
        Optimal added weights alpha, shape (k, s).
    torch.Tensor
        Optimal added weights omega, shape (t, k).
    torch.Tensor
        Singular values s, shape (k,).

    Raises
    ------
    torch.linalg.LinAlgError
        If SVD of S^{-1/2} N fails.
    ValueError
        If maximum_added_neurons is negative.
    """
    if spectra is not None:
        # Always set every key so a caller merging this in cannot keep a stale one.
        spectra.update(matrix_s=None, matrix_e=None, extension=None)

    # Validate inputs
    n_1, n_2 = matrix_n.shape

    # Safeguard against the -1 "resolve from schedule" sentinel (or any negative
    # budget) leaking in from the caller. ``None`` means "no limit"; a negative
    # value would otherwise be silently misread as a Python negative index in the
    # singular-value selection below (``selected[maximum_added_neurons:] = False``
    # with -1 keeps rank-1 neurons instead of capping the count), so reject it.
    if maximum_added_neurons is not None and maximum_added_neurons < 0:
        raise ValueError(
            f"maximum_added_neurons must be None (no limit) or non-negative, got "
            f"{maximum_added_neurons}. A negative value (e.g. the -1 sentinel) must "
            f"be resolved to a concrete per-layer count by the caller before reaching "
            f"compute_optimal_added_parameters."
        )

    if matrix_s is not None:
        # validate S matrix
        s_1, s_2 = matrix_s.shape
        assert s_1 == s_2, "The input matrix S must be square."
        assert s_2 == n_1, (
            f"The input matrices S and N must have compatible shapes."
            f"(got {matrix_s.shape=} and {matrix_n.shape=})"
        )
        if not torch.allclose(matrix_s, matrix_s.t()):
            diff = torch.abs(matrix_s - matrix_s.t())
            warn(
                f"Warning: The input matrix S is not symmetric.\n"
                f"Max difference: {diff.max():.2e},\n"
                f"% of non-zero elements: "
                f"{100 * (diff > 1e-10).sum() / diff.numel():.2f}%"
            )
            matrix_s = (matrix_s + matrix_s.t()) / 2

        # Compute the square root of the inverse of S
        matrix_s_spectra = dict() if spectra is not None else None
        matrix_s_inverse_sqrt = sqrt_inverse_matrix_semi_positive(
            matrix_s,
            threshold=numerical_threshold,
            spectra=matrix_s_spectra,
            source_dtype=source_dtype,
            worst_case_numerical_floor=worst_case_numerical_floor,
        )
        if spectra is not None:
            spectra["matrix_s"] = matrix_s_spectra
        # Compute the product P := S^{-1/2} N
        matrix_p = matrix_s_inverse_sqrt @ matrix_n
    else:
        # GradMax path: S = Identity, so S^{-1/2} = Identity
        matrix_p = matrix_n
        matrix_s_inverse_sqrt = torch.eye(
            n_1, device=matrix_n.device, dtype=matrix_n.dtype
        )

    # Optional empirical-Fisher preconditioner on the output side.
    matrix_e_inverse_sqrt: torch.Tensor | None = None
    if matrix_covariance_loss_gradient is not None:
        e_1, e_2 = matrix_covariance_loss_gradient.shape
        assert e_1 == e_2, "The input matrix E must be square."
        assert e_2 == n_2, (
            f"The input matrices E and N must have compatible shapes."
            f"(got {matrix_covariance_loss_gradient.shape=} and {matrix_n.shape=})"
        )
        if not torch.allclose(
            matrix_covariance_loss_gradient, matrix_covariance_loss_gradient.t()
        ):
            matrix_covariance_loss_gradient = (
                matrix_covariance_loss_gradient + matrix_covariance_loss_gradient.t()
            ) / 2
        matrix_e_spectra = dict() if spectra is not None else None
        matrix_e_inverse_sqrt = sqrt_inverse_matrix_semi_positive(
            matrix_covariance_loss_gradient,
            threshold=(
                e_numerical_threshold
                if e_numerical_threshold is not None
                else numerical_threshold
            ),
            spectra=matrix_e_spectra,
            source_dtype=source_dtype,
            worst_case_numerical_floor=(
                e_worst_case_numerical_floor
                if e_worst_case_numerical_floor is not None
                else worst_case_numerical_floor
            ),
        )
        if spectra is not None:
            spectra["matrix_e"] = matrix_e_spectra
        matrix_p = matrix_p @ matrix_e_inverse_sqrt

    # Compute the SVD of the product
    try:
        u, s, v = torch.linalg.svd(matrix_p, full_matrices=False)
    except torch.linalg.LinAlgError as e:
        print("Warning: An error occurred during the SVD computation.")
        if matrix_s is not None:
            print(f"matrix_s: {matrix_s.min()=}, {matrix_s.max()=}, {matrix_s.shape=}")
        print(f"matrix_n: {matrix_n.min()=}, {matrix_n.max()=}, {matrix_n.shape=}")
        print(
            f"matrix_s_inverse_sqrt: {matrix_s_inverse_sqrt.min()=}, "
            f"{matrix_s_inverse_sqrt.max()=}, {matrix_s_inverse_sqrt.shape=}"
        )
        print(f"matrix_p: {matrix_p.min()=}, {matrix_p.max()=}, {matrix_p.shape=}")
        raise e

    # Select the singular values
    statistical_threshold = resolve_threshold(statistical_threshold, s)
    # The min(..., s.max()) keeps at least one neuron whatever the threshold.
    selected_singular_values = s >= min(statistical_threshold, s.max())
    kept_by_threshold = int(selected_singular_values.sum())
    if maximum_added_neurons is not None:
        selected_singular_values[maximum_added_neurons:] = False

    if spectra is not None:
        spectra["extension"] = {
            "singular_values": s.detach(),  # before any selection
            "threshold": statistical_threshold,
            "kept_by_threshold": kept_by_threshold,
            "kept": int(selected_singular_values.sum()),
            "maximum_added_neurons": maximum_added_neurons,
        }

    # Keep only the significant singular values but keep at least one
    s = s[selected_singular_values]
    u = u[:, selected_singular_values]
    v = v[selected_singular_values, :]

    # Compute output based on ignore_singular_values option
    if ignore_singular_values:
        sqrt_s = torch.ones_like(s)
    else:
        sqrt_s = torch.sqrt(torch.abs(s))
    alpha = sqrt_s * (matrix_s_inverse_sqrt @ u)
    omega = sqrt_s[:, None] * v
    if matrix_e_inverse_sqrt is not None:
        # omega has shape (k, t); apply E^{-1/2} on the right so the eventual
        # transposed result (t, k) is left-multiplied by E^{-1/2}.
        omega = omega @ matrix_e_inverse_sqrt

    if alpha_zero:
        alpha = torch.zeros_like(alpha)

    if omega_zero:
        omega = torch.zeros_like(omega)

    return alpha.t(), omega.t(), s


def spectrum_summary(spectrum: torch.Tensor) -> dict[str, float]:
    """
    Summarize a spectrum (eigenvalues or singular values) with a few scalars.

    Parameters
    ----------
    spectrum: torch.Tensor
        one dimensional tensor of non-negative values

    Returns
    -------
    dict[str, float]
        maximum, minimum, mean and sum of the spectrum, its condition number
        (maximum over minimum, restricted to the strictly positive values) and
        its effective rank (the exponential of the entropy of the normalized
        spectrum). Empty for an empty spectrum.
    """
    if spectrum.numel() == 0:
        return dict()
    spectrum = spectrum.detach().to(dtype=torch.float64)
    positive = spectrum[spectrum > 0]
    total = spectrum.sum()
    if positive.numel() == 0:
        condition_number = float("inf")
        effective_rank = 0.0
    else:
        condition_number = (positive.max() / positive.min()).item()
        proportions = positive / total
        effective_rank = torch.exp(-(proportions * torch.log(proportions)).sum()).item()
    return {
        "max": spectrum.max().item(),
        "min": spectrum.min().item(),
        "mean": spectrum.mean().item(),
        "sum": total.item(),
        "condition_number": condition_number,
        "effective_rank": effective_rank,
    }


def compute_output_shape_conv(
    input_shape: tuple[int, int], conv: torch.nn.Conv2d
) -> tuple[int, int]:
    """
    Compute the output shape of a convolutional layer

    Parameters
    ----------
    input_shape: tuple[int, int]
        shape of the input tensor (H, W)
    conv: torch.nn.Conv2d
        convolutional layer

    Returns
    -------
    tuple[int, int]
        output shape of the convolutional layer
    """
    h, w = input_shape
    assert isinstance(conv.padding[0], int), "The padding must be an integer."
    assert isinstance(conv.padding[1], int), "The padding must be an integer."
    h = (
        h + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1
    ) // conv.stride[0] + 1
    w = (
        w + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1
    ) // conv.stride[1] + 1

    # check the output shape, those line should be finally removed
    with torch.no_grad():
        out_shape = conv(
            torch.empty(
                (1, conv.in_channels, input_shape[0], input_shape[1]),
                device=conv.weight.device,
            )
        ).shape[2:]

    assert h == out_shape[0], f"{h=} {out_shape[0]=} should be equal"
    assert w == out_shape[1], f"{w=} {out_shape[1]=} should be equal"

    return h, w


def compute_mask_tensor_t(
    input_shape: tuple[int, int], conv: torch.nn.Conv2d
) -> torch.Tensor:
    """
    Compute the tensor T
    For:

    - input tensor: B[-1] in (S[-1], H[-1]W[-1]) and (S[-1], H'[-1]W'[-1]) after the pooling
    - output tensor: B in (S, HW)
    - conv kernel tensor: W in (S, S[-1], Hd, Wd)

    T is the tensor in (HW, HdWd, H'[-1]W'[-1]) such that:
    B = W T B[-1]

    Parameters
    ----------
    input_shape: tuple[int, int]
        shape of the input tensor B[-1] of size (H[-1], W[-1])
    conv: torch.nn.Conv2d
        convolutional layer applied to the input tensor B[-1]

    Returns
    -------
    tensor_t: torch.Tensor
        tensor T in (HW, HdWd, H[-1]W[-1])
    """
    h, w = compute_output_shape_conv(input_shape, conv)

    tensor_t = torch.zeros(
        (
            h * w,
            conv.kernel_size[0] * conv.kernel_size[1],
            input_shape[0] * input_shape[1],
        )
    )
    unfold = torch.nn.Unfold(
        kernel_size=conv.kernel_size,
        padding=conv.padding,  # type: ignore
        stride=conv.stride,
        dilation=conv.dilation,
    )
    t_info = unfold(
        torch.arange(1, input_shape[0] * input_shape[1] + 1)
        .float()
        .reshape((1, input_shape[0], input_shape[1]))
    ).int()
    for lc in range(h * w):
        for k in range(conv.kernel_size[0] * conv.kernel_size[1]):
            if t_info[k, lc] > 0:
                tensor_t[lc, k, t_info[k, lc] - 1] = 1
    return tensor_t


def create_bordering_effect_weight(
    channels: int,
    convolution: torch.nn.Conv2d,
) -> torch.Tensor:
    """
    Create the constant depthwise kernel that simulates the border effect of a
    convolution on an unfolded tensor. The weight can then be applied functionally
    in `apply_border_effect_on_unfolded`.

    The returned tensor is a fixed constant (not a learnable parameter): the
    grouped (``groups=channels``) kernel of a depthwise convolution whose only
    non-zero entry is a ``1.0`` at the center of every channel.

    Parameters
    ----------
    channels: int
        Number of input channels for the convolution, warning
        this is for the unfolded tensor, not the original tensor.
        Therefore, it should be equal to C[-1] * C1.kernel_size[0] * C1.kernel_size[1].
    convolution: torch.nn.Conv2d
        convolutional layer whose kernel size and device are matched

    Returns
    -------
    torch.Tensor
        weight of shape ``(channels, 1, kH, kW)`` simulating the border effect

    Raises
    ------
    ValueError
        if argument channels is not a positive integer
    TypeError
        if argument convolution is not of type torch.nn.Conv2d
    """
    if not isinstance(channels, int) or channels <= 0:
        raise ValueError("Input 'channels' must be a positive integer.")
    if not isinstance(convolution, torch.nn.Conv2d):
        raise TypeError("Input 'convolution' must be a torch.nn.Conv2d instance.")

    weight = torch.zeros(
        channels,
        1,
        convolution.kernel_size[0],
        convolution.kernel_size[1],
        device=convolution.weight.device,
    )
    mid = (convolution.kernel_size[0] // 2, convolution.kernel_size[1] // 2)
    weight[:, 0, mid[0], mid[1]] = 1.0

    return weight


@torch.no_grad()
def apply_border_effect_on_unfolded(
    unfolded_tensor: torch.Tensor,
    original_size: tuple[int, int],
    border_effect_conv: torch.nn.Conv2d,
    identity_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Simulate the effect of a 1x1 convolution on the size of an unfolded tensor.
    Should satisfy that for a convolution C1 and a convolution C2,
    if B is the output of C1 of shape (n, C, H, W) we get
    as unfolded tensor the unfolded input of C1 of shape
    (n, C[-1] * C1.kernel_size[0] * C1.kernel_size[1], H * W).
    Then B[+1] is the output of C2 of shape (n, C[+1], H[+1], W[+1])
    the output of this function (noted F) should be of shape
    (n, C[+1] * C2.kernel_size[0] * C2.kernel_size[1], H[+1] * W[+1])
    such that if C2 has only 1x1 centered non-zero kernel
    C2 o C1(F) should be equal to C1 o C2(B[+1]).

    Parameters
    ----------
    unfolded_tensor: torch.Tensor
        unfolded tensor to be modified
    original_size: tuple[int, int]
        original size of the tensor before unfolding
    border_effect_conv: torch.nn.Conv2d
        convolutional layer providing the convolution hyper-parameters
        (stride, padding, dilation, kernel size) used to apply the border effect.
    identity_weight: torch.Tensor | None
        constant depthwise kernel (see `create_bordering_effect_weight`) applied
        functionally with `torch.nn.functional.conv2d`. If None, it is built from
        `border_effect_conv`.

    Returns
    -------
    torch.Tensor
        modified unfolded tensor

    Raises
    ------
    TypeError
        if argument unfloded_tensor is not of type torch.Tensor
    """
    if not isinstance(unfolded_tensor, torch.Tensor):
        raise TypeError("Input 'unfolded_tensor' must be a torch.Tensor")
    assert isinstance(border_effect_conv, torch.nn.Conv2d), (
        "'border_effect_conv' must be a torch.nn.Conv2d instance."
    )
    assert all(isinstance(s, int) and s > 0 for s in original_size), (
        "'original_size' must be a tuple of positive integers."
    )

    channels = unfolded_tensor.shape[1]
    if identity_weight is None:
        identity_weight = create_bordering_effect_weight(
            channels=channels,
            convolution=border_effect_conv,
        )

    unfolded_tensor = unfolded_tensor.reshape(
        unfolded_tensor.shape[0],
        channels,
        original_size[0],
        original_size[1],
    )

    unfolded_tensor = torch.nn.functional.conv2d(
        unfolded_tensor,
        identity_weight,
        stride=border_effect_conv.stride,
        padding=border_effect_conv.padding,
        dilation=border_effect_conv.dilation,
        groups=channels,
    )
    unfolded_tensor = unfolded_tensor.flatten(start_dim=2)

    return unfolded_tensor


def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    """Initialize weight tensor with LecunNorm
    Draws samples from a truncated normal distribution centered around 0 with std = sqrt(1 / fan_in)

    Parameters
    ----------
    tensor : torch.Tensor
        weight tensor

    Returns
    -------
    torch.Tensor
        initialized weight tensor

    Raises
    ------
    ValueError
        if the shape of the tensor is not 2D or 4D
    """
    if tensor.ndim == 2:  # Linear
        fan_in = tensor.size(1)
    elif tensor.ndim == 4:  # Conv2d
        fan_in = tensor.size(1) * tensor.size(2) * tensor.size(3)
    else:
        raise ValueError(
            f"Only supports Linear (2D) or Conv2d (4D) weights, got tensor with shape {tensor.shape}"
        )
    std = 1.0 / math.sqrt(fan_in)
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)
