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

KnownThresholdRuleName = Literal["mean_over_sqrt_n"]


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


KNOWN_THRESHOLD_RULES: dict[KnownThresholdRuleName, ThresholdRule] = {
    "mean_over_sqrt_n": _mean_over_sqrt_n_rule,
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


def sqrt_inverse_matrix_semi_positive(
    matrix: torch.Tensor,
    threshold: float | SpectrumThreshold = 1e-5,
    spectra: dict[str, Any] | None = None,
) -> torch.Tensor:
    """
    Compute the square root of the inverse of a semi-positive definite matrix.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix, square and semi-positive definite
    threshold: float | SpectrumThreshold
        threshold to consider an eigenvalue as zero, either a value or a rule
        already bound to its statistic (see `resolve_threshold`)
    spectra: dict[str, Any] | None
        if given, filled in place with the eigenvalues of the input matrix, the
        threshold applied to them, the number of eigenvalues kept and the total
        number of eigenvalues. Nothing is computed when None.

    Returns
    -------
    torch.Tensor
        square root of the inverse of the input matrix
    """
    assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."
    assert torch.allclose(matrix, matrix.t()), "The input matrix must be symmetric."
    assert torch.isnan(matrix).sum() == 0, "The input matrix must not contain NaN values."

    regularized = False
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    except torch.linalg.LinAlgError:
        # Sometimes, due to numerical issues, we get an error:
        # The algorithm failed to converge because the input matrix is
        # ill-conditioned or has too many repeated eigenvalues
        regularized = True
        matrix += torch.finfo(matrix.dtype).resolution * torch.eye(
            matrix.shape[0],
            device=matrix.device,
            dtype=matrix.dtype,
        )
        warn(
            message="Adding a small identity matrix to make the input matrix positive definite.",
            category=RuntimeWarning,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # A value is the caller's explicit choice; only a rule is second-guessed.
    threshold_is_rule = callable(threshold)
    # fallback is the same as `torch.linalg.pinv` default
    fallback_threshold = matrix.shape[0] * torch.finfo(matrix.dtype).eps
    threshold = resolve_threshold(threshold, eigenvalues, fallback=fallback_threshold)
    selected_eigenvalues = eigenvalues > threshold
    if threshold_is_rule and selected_eigenvalues.sum() == 0 and eigenvalues.max() > 0:
        warn(
            message=(
                f"The threshold {threshold:.3e} drops the whole spectrum of a non-zero "
                f"matrix, which would make the inverse square root zero. "
                f"Falling back to {fallback_threshold:.3e}."
            ),
            category=RuntimeWarning,
        )
        threshold = fallback_threshold
        selected_eigenvalues = eigenvalues > threshold

    if spectra is not None:
        spectra.update(
            eigenvalues=eigenvalues.detach(),
            threshold=threshold,
            kept=int(selected_eigenvalues.sum()),
            total=int(eigenvalues.numel()),
            regularized=regularized,
        )

    eigenvalues = torch.rsqrt(eigenvalues[selected_eigenvalues])  # inverse square root
    eigenvectors = eigenvectors[:, selected_eigenvalues]
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()


def optimal_delta(
    tensor_s: torch.Tensor,
    tensor_m: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    force_pseudo_inverse: bool = False,
    tensor_covariance_loss_gradient: torch.Tensor | None = None,
    spectra: dict[str, Any] | None = None,
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
        delta_raw = (torch.linalg.pinv(tensor_s) @ tensor_m).t()

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
            delta_raw = torch.linalg.pinv(tensor_covariance_loss_gradient) @ delta_raw

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
            return optimal_delta(
                tensor_s,
                tensor_m,
                dtype=torch.float64,
                force_pseudo_inverse=True,
                tensor_covariance_loss_gradient=tensor_covariance_loss_gradient,
                spectra=spectra,
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
    numerical_threshold: float | SpectrumThreshold = 1e-6,
    statistical_threshold: float | SpectrumThreshold = 1e-3,
    maximum_added_neurons: int | None = None,
    alpha_zero: bool = False,
    omega_zero: bool = False,
    ignore_singular_values: bool = False,
    matrix_covariance_loss_gradient: torch.Tensor | None = None,
    e_numerical_threshold: float | SpectrumThreshold | None = None,
    spectra: dict[str, Any] | None = None,
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
    numerical_threshold : float | SpectrumThreshold
        Threshold to consider an eigenvalue as zero in square root of inverse of S
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
        Pass 0.0 to keep the full spectrum (e.g. when E_s has been ridge-shrunk
        upstream and is already positive definite).
    spectra : dict[str, Any] | None
        If given, filled in place with the keys "matrix_s", "matrix_e" and
        "extension". The first two hold the whitening spectra of S and E (None
        when the corresponding matrix is not used, see
        `sqrt_inverse_matrix_semi_positive`); "extension" holds the singular
        values of the SVD target before any selection, the threshold applied to
        them, and how many were kept by the threshold and by
        `maximum_added_neurons`. Nothing is computed when None.

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
            matrix_s, threshold=numerical_threshold, spectra=matrix_s_spectra
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
