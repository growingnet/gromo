import torch
import torch.nn as nn
import torch.nn.functional as F

# from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.utils.utils import global_device


def svd_low_rank(
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

    Z_hat = A @ B.T
    rel_err = torch.linalg.norm(Z_hat - Z) / torch.linalg.norm(Z)

    if reconstruction_error:
        # optional consistency check (mainly to catch NaNs / infs)
        assert torch.allclose(Z_hat, Z, atol=atol, rtol=rtol) or d_low < d_high0, (
            "Exact reconstruction failed; either numerical issues "
            "or d_low is strictly less than rank(Z)."
        )
        return A, B, rel_err
    return A, B


class PrototypeAttention(nn.Module):
    def __init__(
        self,
        d_s: int,
        d_e: int,
        d_k: int,
        d_v: int,
        use_bias: bool = True,
        pre_attn: nn.Module | None = None,
        post_attn: nn.Module | None = None,
    ) -> None:
        """Growing module for attention
        #NOTE Curently only focusing on growing kdim, for one head, without bias

            Parameters
            ----------
            d_s : int
                dimension of the input sequence
            d_e : int
                dimension of the input and output embedding
            d_k : int
                dimension of the query and key
            d_v : int
                dimension of the value
            use_bias : bool
                use of bias
            pre_attn : nn.Module | None
                optional module to apply before attention
            post_attn : nn.Module | None
                optional module to apply after attention
        """
        super().__init__()
        self.d_s: int = d_s
        self.d_e: int = d_e
        self.d_k: int = d_k
        self.d_v: int = d_v
        self.scale = d_k**0.5
        self.pre_attn: nn.Module | None = pre_attn
        self.post_attn: nn.Module | None = post_attn
        self.use_bias: bool = use_bias
        self.S_grad = None  # Placeholder for the gradient of S

        # Linear projections for Q, K, V (with bias if requested)
        if self.use_bias:
            self.W_q = nn.Linear(d_e + 1, d_k, bias=False)
            self.W_k = nn.Linear(d_e + 1, d_k, bias=False)
            self.W_v = nn.Linear(d_e + 1, d_v, bias=False)
        else:
            self.W_q = nn.Linear(d_e, d_k, bias=False)
            self.W_k = nn.Linear(d_e, d_k, bias=False)
            self.W_v = nn.Linear(d_e, d_v, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_v, d_e, bias=use_bias)

    def save_S_grad(self, grad: torch.Tensor) -> None:
        """Hook to save the gradient of S."""
        self.S_grad = grad

    def get_S_grad(self) -> torch.Tensor:
        """Return the gradient of S (b, d_s, d_s) from the last backward pass"""
        assert self.S_grad is not None, (
            "S_grad is not available. Make sure to call forward() first."
        )
        return self.S_grad

    def add_bias(self, X: torch.Tensor, add_column: bool = True) -> torch.Tensor:
        """Add a column or row of ones to the tensor based on the argument.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, m, n)
        add_column : bool, optional
            If True, adds a column of ones. If False, adds a row of ones. Default is True.

        Returns
        -------
        torch.Tensor
            Augmented tensor of shape:
            - (batch_size, m, n + 1) if add_column is True
            - (batch_size, m + 1, n) if add_column is False
        """
        if add_column:
            ones = torch.ones(X.size(0), X.size(1), 1, device=X.device, dtype=X.dtype)
            return torch.cat((X, ones), dim=-1)
        else:
            ones = torch.ones(X.size(0), 1, X.size(2), device=X.device, dtype=X.dtype)
            return torch.cat((X, ones), dim=1)

    def forward(self, X: torch.Tensor):
        """Forward pass of the attention module

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, d_s, d_e)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, d_s, d_e)
        """
        if self.use_bias:
            X = self.add_bias(X)  # (b, d_s, d_e + 1)

        # Optional pre-attention transform
        if self.pre_attn is not None:
            X = self.pre_attn(X)

        # Compute Q, K, V
        Q = self.W_q(X)  # (b, d_s, d_k)
        K = self.W_k(X)  # (b, d_s, d_k)
        V = self.W_v(X)  # (b, d_s, d_v)

        # Scaled dot‑product attention
        S = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (b, d_s, d_s)

        # We save the gradient of S
        if S.requires_grad:
            S.register_hook(self.save_S_grad)

        A = F.softmax(S, dim=-1)  # (b, d_s, d_s)

        # Weighted sum of values
        H = torch.matmul(A, V)  # (b, d_s, d_v)

        # Final projection
        Y = self.W_o(H)  # (b, d_s, d_e)

        # Optional post-attention transform
        if self.post_attn is not None:
            Y = self.post_attn(Y)

        return Y, S, X

    def get_growed_weight_matrices(
        self, X: torch.Tensor, add_bias_before_pseudoinverse: bool
    ):
        if self.use_bias:
            Y, S, X_bias = self.forward(X)

            if add_bias_before_pseudoinverse:
                X_pinv = torch.linalg.pinv(X_bias)  # (b, d_e +1, d_s)
            else:
                X_pinv = torch.linalg.pinv(X)  # (b, d_e, d_s)
                X_pinv = self.add_bias(X_pinv, add_column=False)  # (b, d_e +1, d_s)
        else:
            Y, S, _ = self.forward(X)
            X_pinv = torch.linalg.pinv(X)  # (b,d_e,d_s)

        # WARN X has shape (b, d_s, d_e) here, the bias was only applied in the forward
        # Problem with bias, when to add it?
        # Add it before doing the pseudoinverse? After?

        loss = Y.sum()
        loss.backward()
        S_grad = self.get_S_grad()  # (b, d_s, d_s)

        Z = S_grad + S * self.scale  # (b, d_s, d_s)
        Z = torch.matmul(X_pinv, Z)  # (b, d_e (+1), d_s)
        Z = torch.matmul(Z, X_pinv.transpose(-2, -1))  # (b, d_e (+1), d_e (+1))
        Z_mean = Z.mean(dim=0)  # (d_e (+1), d_e (+1))


if __name__ == "__main__":
    torch.manual_seed(0)
    device = global_device()
    print(f"Device: {device}")

    d_s = 8
    d_e = 32
    d_k = 16
    d_v = 18
    batch_size = 4

    # model = PrototypeAttention(d_s, d_e, d_k, d_v)
    # x = torch.randn(batch_size, d_s, d_e, requires_grad=True).to(device)
    # model.get_growed_weight_matrices(x, False)
