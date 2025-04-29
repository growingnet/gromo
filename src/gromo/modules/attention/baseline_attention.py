import torch
import torch.nn as nn
import torch.nn.functional as F

# from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.utils.utils import global_device


class BaselineAttention(nn.Module):
    def __init__(
        self,
        d_s: int,
        d_e: int,
        d_k: int,
        d_v: int,
        use_bias: bool = True,
        # pre_attn: nn.Module | None = None,
        # post_attn: nn.Module | None = None,
    ) -> None:
        """Baseline attention module
        batch_size = b

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
        # self.pre_attn: nn.Module | None = pre_attn
        # self.post_attn: nn.Module | None = post_attn
        self.use_bias: bool = use_bias

        self.W_Q = nn.Linear(d_e, d_k, bias=self.use_bias)
        self.W_K = nn.Linear(d_e, d_k, bias=self.use_bias)
        self.W_V = nn.Linear(d_e, d_v, bias=self.use_bias)
        self.W_O = nn.Linear(d_v, d_e, bias=self.use_bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Classical forward pass attention module

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (b, d_s, d_e)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (b, d_s, d_e)
        """
        # Optional pre-attention transform
        # if self.pre_attn is not None:
        #     X = self.pre_attn(X)

        Q = self.W_Q(X)  # (b, d_s, d_k)
        K = self.W_K(X)  # (b, d_s, d_k)
        V = self.W_V(X)  # (b, d_s, d_v)

        S = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (b, d_s, d_s)
        A = F.softmax(S, dim=-1)  # (b, d_s, d_s)
        H = torch.matmul(A, V)  # (b, d_s, d_v)
        Y = self.W_O(H)  # (b, d_s, d_e)

        # Optional post-attention transform
        # if self.post_attn is not None:
        #     Y = self.post_attn(Y)

        return Y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = global_device()
    print(f"Device: {device}")

    d_s = 4
    d_e = 16
    d_k = 8
    d_v = 8
    batch_size = 128
