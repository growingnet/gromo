import torch


def my_svd_low_rank(
    Z: torch.Tensor,
    d_low: int,
    *,
    reconstruction_error: bool = False,
    atol=1e-7,
    rtol=1e-6,
):
    """
    Factor a (d_high × d_high) matrix Z into tall skinny matrices A, B with inner
    dimension d_low  (d_low « d_high) so that   A  @  B.T  ≈  Z.

    Parameters
    ----------
    Z    : torch.Tensor  shape [d_high, d_high]  (can be non-symmetric, any real dtype)
    d_low  : int           target rank  (1 ≤ d_low ≤ d_high)
    atol, rtol : float   tolerances for the optional consistency check

    Returns
    -------
    A : [d_high, d_low]
    B : [d_high, d_low]
    rel_err : float   ‖A Bᵀ − Z‖_F / ‖Z‖_F
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

    if reconstruction_error:
        Z_hat = A @ B.T
        rel_err = torch.linalg.norm(Z_hat - Z) / torch.linalg.norm(Z)
        # optional consistency check (mainly to catch NaNs / infs)
        assert torch.allclose(Z_hat, Z, atol=atol, rtol=rtol) or d_low < d_high0, (
            "Exact reconstruction failed; either numerical issues "
            "or d_low is strictly less than rank(Z)."
        )
    else:
        rel_err = -1
    return A, B, rel_err
