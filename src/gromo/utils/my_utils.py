import torch


def my_svd_low_rank(
    Z: torch.Tensor,
    d_low: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Factor a (d_high × d_high) matrix Z into tall skinny matrices A, B with inner
    dimension d_low  (d_low « d_high) so that   A  @  B.T  ≈  Z.

    Parameters
    ----------
    Z    : torch.Tensor  shape [d_high, d_high]  (can be non-symmetric, any real dtype)
    d_low  : int           target rank  (1 ≤ d_low ≤ d_high)
    Returns
    -------
    A : [d_high, d_low]
    B : [d_high, d_low]
    """
    assert Z.dim() == 2, "Z must be a 2-D matrix"
    d_high0, d_high1 = Z.shape
    assert d_high0 == d_high1, f"Z must be square (got {d_high0} × {d_high1})"
    assert 1 <= d_low <= d_high0, f"d_k must be between 1 and {d_high0}"
    assert torch.is_floating_point(Z), "Z must have a floating-point dtype"

    U, S, Vh = torch.linalg.svd(
        Z, full_matrices=False
    )  # U:[d_high,d_high], S:[d_high], Vh:[d_high,d_high]
    U_k = U[:, :d_low]  # [d_high, d_low]
    S_k = S[:d_low]  # [d_low]
    V_k = Vh[:d_low, :].T  # [d_high, d_low]

    sqrtS = S_k.sqrt()  # [d_low]

    A = U_k * sqrtS.unsqueeze(0)  # broadcast over rows
    B = V_k * sqrtS.unsqueeze(0)

    return A, B


def check_2Dtensor_shape(tensor: torch.Tensor, row_dim: int, col_dim: int) -> None:
    expected_rows = row_dim
    expected_cols = col_dim
    assert tensor.shape == (expected_rows, expected_cols), (
        f"Tensor shape mismatch: expected ({expected_rows}, {expected_cols}), "
        f"got {tensor.shape}"
    )
