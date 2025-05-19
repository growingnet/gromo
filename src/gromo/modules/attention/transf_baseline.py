import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import LayerNorm


@dataclass
class ModelConfig:
    d_s: int = 4
    d_e: int = 16
    d_k: int = 8
    d_v: int = 8
    bias: bool = False


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

    def forward(self, X, gamma: None | float = None):
        """If gamma is not None, compute the forward using (S + gamma * S_grad) instead of S"""
        Q = self.W_Q(X)  # Compute query vectors
        K = self.W_K(X)  # Compute key vectors
        V = self.W_V(X)  # Compute value vectors

        S = (Q @ K.transpose(-2, -1)) * (1 / self.scale)

        # We save the gradient of S
        if S.requires_grad:
            S.register_hook(self.save_S_grad)

        if gamma is not None:
            assert self.S_grad is not None
            S -= gamma * self.S_grad

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

    def forward(self, x, gamma: None | float = None):
        x = x + self.attn(self.ln1(x), gamma)
        x = x + self.mlp(self.ln2(x))
        return x
