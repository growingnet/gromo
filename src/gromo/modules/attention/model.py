import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import LayerNorm
from gromo.modules.attention.my_utils import my_svd_low_rank


@dataclass
class ModelConfig:
    d_s: int = 4
    d_e: int = 16
    d_k: int = 8
    d_k_max: int = 8
    d_v: int = 8
    bias: bool = False
    assert bias is False, "The growing algorithm is not implemented with bias"


class SelfAttentionBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_Q = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias)
        self.W_K = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias)
        self.W_V = nn.Linear(cfg.d_e, cfg.d_v, bias=cfg.bias)
        self.W_O = nn.Linear(cfg.d_v, cfg.d_e, bias=cfg.bias)
        self.scale = math.sqrt(cfg.d_k)
        self.S_grad = None

    def save_S_grad(self, grad: torch.Tensor) -> None:
        """Hook to save the gradient of S."""
        self.S_grad = grad

    def get_S_grad(self) -> torch.Tensor:
        """Return the gradient of S from the last backward pass"""
        assert self.S_grad is not None, (
            "S_grad is not available. Make sure to call forward() first."
        )
        return self.S_grad

    def forward(self, X, scaling_test: None | float = None):
        """If scaling_test is not None, compute the forward using (S + scaling_test * S_grad) instead of S"""
        Q = self.W_Q(X)  # Compute query vectors
        K = self.W_K(X)  # Compute key vectors
        V = self.W_V(X)  # Compute value vectors

        S = (Q @ K.transpose(-2, -1)) * (1 / self.scale)

        # We save the gradient of S
        if S.requires_grad:
            S.register_hook(self.save_S_grad)

        if scaling_test is not None:
            assert self.S_grad is not None
            S -= scaling_test * self.S_grad

        A = F.softmax(S, dim=-1)  # Apply softmax to get attention weights
        H = A @ V
        y = self.W_O(H)
        return y


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lin1 = nn.Linear(cfg.d_e, 4 * cfg.d_e, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(4 * cfg.d_e, cfg.d_e, bias=cfg.bias)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_e, eps=1e-5, bias=cfg.bias)
        self.attn = SelfAttentionBaseline(cfg)
        self.ln2 = LayerNorm(cfg.d_e, eps=1e-5, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, scaling_test: None | float = None):
        """If a batch size is provided, the statistics for the natural gradient will be retained. The batch size should be equal to the training batch size"""
        x = self.ln1(x)
        self.input_attention_block = x

        x = x + self.attn(x, scaling_test)
        x = x + self.mlp(self.ln2(x))
        return x

    def compute_statistics(self, outside_esp: bool):
        assert self.attn.S_grad is not None
        x = self.input_attention_block  # (b,s,e)
        xt = x.transpose(-2, -1)  # (b,e,s)

        if outside_esp:
            acc_cov = torch.linalg.pinv((xt @ x))  # (b,e,e)
            acc_cov_grad = xt @ self.attn.S_grad @ x  # (b,e,e)
            acc_x = torch.linalg.pinv(x)  # (b,e,s)
            acc_x_grad = self.attn.S_grad  # (b,s,s)

            self.P_steph = (acc_cov @ acc_cov_grad @ acc_cov).mean(dim=0)  # (e,e)
            self.P_leo = (acc_x @ acc_x_grad @ acc_x.transpose(-2, -1)).mean(
                dim=0
            )  # (e,e)
        else:
            acc_cov = torch.linalg.pinv((xt @ x).mean(dim=0))  # (e,e)
            acc_cov_grad = (xt @ self.attn.S_grad @ x).mean(dim=0)  # (e,e)
            acc_x = torch.linalg.pinv(x.mean(dim=0))  # (e,s)
            acc_x_grad = (self.attn.S_grad).mean(dim=0)  # (s,s)

            self.P_steph = acc_cov @ acc_cov_grad @ acc_cov  # (e,e)
            self.P_leo = acc_x @ acc_x_grad @ acc_x.transpose(-2, -1)  # (e,e)

    def get_P_ratios(self):
        """Return the Frobenius and Operator norm of the differences of P_steph and P_leo"""
        assert self.P_steph is not None
        assert self.P_leo is not None
        fro = torch.linalg.matrix_norm(
            self.P_steph, ord="fro", dim=(-2, -1)
        ) / torch.linalg.matrix_norm(self.P_leo, ord="fro", dim=(-2, -1))
        op = torch.linalg.matrix_norm(
            self.P_steph, ord=2, dim=(-2, -1)
        ) / torch.linalg.matrix_norm(self.P_leo, ord=2, dim=(-2, -1))
        return fro, op

    def update_WQ_WK(self, cfg):
        """Update the weights of W_Q and W_K using the natural gradient"""
        assert self.P_leo is not None
        self.WQ_copy = self.attn.W_Q.weight.clone().T
        self.WK_copy = self.attn.W_K.weight.clone().T

        lbd = 10000000  # TODO: Get this with line search
        WQ_new, WK_new = my_svd_low_rank(
            self.WQ_copy @ self.WK_copy.T - lbd * self.P_leo, cfg.d_k
        )
        WQ_new = WQ_new.T
        WK_new = WK_new.T

        new_WQ = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias)
        new_WK = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias)

        with torch.no_grad():
            new_WQ.weight.copy_(WQ_new)
            new_WK.weight.copy_(WK_new)

        self.W_Q = new_WQ
        self.W_K = new_WK
