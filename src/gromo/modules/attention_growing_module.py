import torch
import torch.nn as nn
import torch.nn.functional as F
from gromo.utils.utils import global_device
from gromo.modules.growing_module import GrowingModule, MergeGrowingModule


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
        self.pre_attn: nn.Module | None = pre_attn
        self.post_attn: nn.Module | None = post_attn

        # Linear projections for Q, K, V (with bias if requested)
        self.W_q = nn.Linear(d_e, d_k, bias=use_bias)
        self.W_k = nn.Linear(d_e, d_k, bias=use_bias)
        self.W_v = nn.Linear(d_e, d_v, bias=use_bias)

        # Output projection
        self.W_o = nn.Linear(d_v, d_e, bias=use_bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
        # Optional pre-attention transform
        if self.pre_attn is not None:
            X = self.pre_attn(X)

        # b, s, _ = X.size()

        # Compute Q, K, V
        Q = self.W_q(X)  # (b, d_s, d_k)
        K = self.W_k(X)  # (b, d_s, d_k)
        V = self.W_v(X)  # (b, d_s, d_v)

        # Scaled dot‑product attention
        S = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)  # (b, d_s, d_s)
        A = F.softmax(S, dim=-1)  # (b, d_s, d_s)

        # Weighted sum of values
        H = torch.matmul(A, V)  # (b, d_s, d_v)

        # Final projection
        Y = self.W_o(H)  # (b, d_s, d_e)

        # Optional post-attention transform
        if self.post_attn is not None:
            Y = self.post_attn(Y)

        return Y


if __name__ == "__main__":
    d_s = 8
    d_e = 32
    d_k = 16
    d_v = 16
    batch_size = 8

    model = PrototypeAttention(d_s, d_e, d_k, d_v)
    x = torch.randn(batch_size, d_s, d_e)
    output = model(x)
    print(output.shape)  # Should be (batch_size, seq_length, embed_dim)
