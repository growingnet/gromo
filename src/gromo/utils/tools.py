from warnings import warn

import torch


def sqrt_inverse_matrix_semi_positive(
    matrix: torch.Tensor,
    threshold: float = 1e-5,
    preferred_linalg_library: None | str = None,
) -> torch.Tensor:
    """
    Compute the square root of the inverse of a semi-positive definite matrix.

    Parameters
    ----------
    matrix: torch.Tensor
        input matrix, square and semi-positive definite
    threshold: float
        threshold to consider an eigenvalue as zero
    preferred_linalg_library: None | str in ("magma", "cusolver")
        linalg library to use, "cusolver" may failed
        for non positive definite matrix if CUDA < 12.1 is used
        see: https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html

    Returns
    -------
    torch.Tensor
        square root of the inverse of the input matrix
    """
    assert matrix.shape[0] == matrix.shape[1], "The input matrix must be square."
    assert torch.allclose(matrix, matrix.t()), "The input matrix must be symmetric."
    assert torch.isnan(matrix).sum() == 0, "The input matrix must not contain NaN values."

    if preferred_linalg_library is not None:
        torch.backends.cuda.preferred_linalg_library(preferred_linalg_library)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    except torch.linalg.LinAlgError as e:
        if preferred_linalg_library == "cusolver":
            raise ValueError(
                "This is probably a bug from CUDA < 12.1"
                "Try torch.backends.cuda.preferred_linalg_library('magma')"
            )
        else:
            raise e
    selected_eigenvalues = eigenvalues > threshold
    eigenvalues = torch.rsqrt(eigenvalues[selected_eigenvalues])  # inverse square root
    eigenvectors = eigenvectors[:, selected_eigenvalues]
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()


def compute_optimal_added_parameters(
    matrix_s: torch.Tensor,
    matrix_n: torch.Tensor,
    numerical_threshold: float = 1e-15,
    statistical_threshold: float = 1e-3,
    maximum_added_neurons: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the optimal added parameters for a given layer.

    Parameters
    ----------
    matrix_s: torch.Tensor in (s, s)
        square matrix S
    matrix_n: torch.Tensor in (s, t)
        matrix N
    numerical_threshold: float
        threshold to consider an eigenvalue as zero in the square root of the inverse of S
    statistical_threshold: float
        threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
    maximum_added_neurons: int | None
        maximum number of added neurons, if None all significant neurons are kept

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] in (k, s) (t, k) (k,)
        optimal added weights alpha, omega and eigenvalues lambda
    """
    # matrix_n = matrix_n.t()
    s_1, s_2 = matrix_s.shape
    assert s_1 == s_2, "The input matrix S must be square."
    n_1, n_2 = matrix_n.shape
    assert s_2 == n_1, (
        f"The input matrices S and N must have compatible shapes."
        f"(got {matrix_s.shape=} and {matrix_n.shape=})"
    )
    if not torch.allclose(matrix_s, matrix_s.t()):
        diff = torch.abs(matrix_s - matrix_s.t())
        warn(
            f"Warning: The input matrix S is not symmetric.\n"
            f"Max difference: {diff.max():.2e},"
            f"% of non-zero elements: {100 * (diff > 1e-10).sum() / diff.numel():.2f}%"
        )
        matrix_s = (matrix_s + matrix_s.t()) / 2

    # assert torch.allclose(matrix_s, matrix_s.t()), "The input matrix S must be symmetric."

    # compute the square root of the inverse of S
    matrix_s_inverse_sqrt = sqrt_inverse_matrix_semi_positive(
        matrix_s, threshold=numerical_threshold
    )
    # compute the product P := S^{-1/2} N
    matrix_p = matrix_s_inverse_sqrt @ matrix_n
    # compute the SVD of the product
    try:
        u, s, v = torch.linalg.svd(matrix_p, full_matrices=False)
    except torch.linalg.LinAlgError:
        print(f"Warning: An error occurred during the SVD computation.")
        print(f"matrix_s: {matrix_s.min()=}, {matrix_s.max()=}, {matrix_s.shape=}")
        print(f"matrix_n: {matrix_n.min()=}, {matrix_n.max()=}, {matrix_n.shape=}")
        print(
            f"matrix_s_inverse_sqrt: {matrix_s_inverse_sqrt.min()=}, {matrix_s_inverse_sqrt.max()=}, {matrix_s_inverse_sqrt.shape=}"
        )
        print(f"matrix_p: {matrix_p.min()=}, {matrix_p.max()=}, {matrix_p.shape=}")
        u, s, v = torch.linalg.svd(matrix_p, full_matrices=False)
        # raise ValueError("An error occurred during the SVD computation.")

        # u = torch.zeros((1, matrix_p.shape[0]))
        # s = torch.zeros(1)
        # v = torch.randn((matrix_p.shape[1], 1))
        # return u, v, s

    # select the singular values
    selected_singular_values = s >= min(statistical_threshold, s.max())
    if maximum_added_neurons is not None:
        selected_singular_values[maximum_added_neurons:] = False

    # keep only the significant singular values but keep at least one
    s = s[selected_singular_values]
    u = u[:, selected_singular_values]
    v = v[selected_singular_values, :]
    # compute the optimal added weights
    sqrt_s = torch.sqrt(torch.abs(s))
    alpha = torch.sign(s) * sqrt_s * (matrix_s_inverse_sqrt @ u)
    omega = sqrt_s[:, None] * v
    return alpha.t(), omega.t(), s


def compute_output_shape_conv(
    input_shape: tuple[int, int], conv: torch.nn.Conv2d
) -> tuple[int, int]:
    """
    Compute the output shape of a convolutional layer

    Parameters
    ----------
    input_shape: tuple
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
    input_shape: tuple
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
        padding=conv.padding,
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
