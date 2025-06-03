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
    bias_attention: bool = False
    bias_other: bool = True
    assert bias_attention is False, "The growing algorithm is not implemented with bias"


class SelfAttentionBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_Q = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias_attention)
        self.W_K = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias_attention)
        self.W_V = nn.Linear(cfg.d_e, cfg.d_v, bias=cfg.bias_attention)
        self.W_O = nn.Linear(cfg.d_v, cfg.d_e, bias=cfg.bias_attention)
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
        self.S_keep = S

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
        self.lin1 = nn.Linear(cfg.d_e, 4 * cfg.d_e, bias=cfg.bias_other)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(4 * cfg.d_e, cfg.d_e, bias=cfg.bias_other)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_e, eps=1e-5, bias=cfg.bias_other)
        self.attn = SelfAttentionBaseline(cfg)
        self.ln2 = LayerNorm(cfg.d_e, eps=1e-5, bias=cfg.bias_other)
        self.mlp = MLP(cfg)

    def forward(self, x, scaling_test: None | float = None):
        x = self.ln1(x)
        self.input_attention_block = x

        x = x + self.attn(x, scaling_test)
        x = x + self.mlp(self.ln2(x))
        return x

    def freeze_batch_input_and_grad(self):
        assert self.attn.S_grad is not None
        self.frozen_x = self.input_attention_block.clone()
        self.frozen_S_grad = self.attn.S_grad.clone()
        self.frozen_S = self.attn.S_keep.clone()

    def reinitialize_stats(self, methods):
        self.kro_batch = 0.0
        self.vec_xtGx_batch = 0.0
        self.xtx_batch = 0.0
        self.xtGx_batch = 0.0
        self.bigf_batch = 0.0
        self.grad_S_batch = 0.0
        self.S_batch = 0.0

        self.kro_avg = []
        self.vec_xtGx_avg = []
        self.xtx_avg = []
        self.xtGx_avg = []
        self.bigf_avg = []
        self.grad_S_avg = []
        self.S_avg = []

        self.P_stat = {}
        for method in methods:
            self.P_stat[method] = []

    def compute_stats_average(self, N):
        self.kro_avg.append(self.kro_batch / N)
        self.vec_xtGx_avg.append(self.vec_xtGx_batch / N)
        self.xtx_avg.append(self.xtx_batch / N)
        self.xtGx_avg.append(self.xtGx_batch / N)
        self.bigf_avg.append(self.bigf_batch / N)
        self.grad_S_avg.append(self.grad_S_batch / N)
        self.S_avg.append(self.S_batch / N)

    def accumulate_batch_stats(self, method_li: list[str]):
        """
        Compute the intermediate statistics for a batch.
        """
        assert self.frozen_S_grad is not None
        xt = self.frozen_x.transpose(-2, -1)  # (b,e,s)
        xtx = xt @ self.frozen_x  # (b,e,e)
        xtGx = xt @ self.frozen_S_grad @ self.frozen_x
        b = xtx.size(0)
        e = xtx.size(-1)

        # TODO: Check eigeinvalues of methods
        if "kro" in method_li:
            # Kronecker product for each batch
            covi1 = xtx.unsqueeze(2).unsqueeze(4)  # (b,e,1,e,1)
            covi2 = xtx.unsqueeze(1).unsqueeze(3)  # (b,1,e,1,e)
            kro = (covi1 * covi2).reshape(b, e * e, e * e)  # (b,e*e,e*e)
            self.kro_batch += kro.mean(dim=0)  # (e*e,e*e)
            self.vec_xtGx_batch += xtGx.mean(dim=0).reshape(-1, 1)

        if "big_in" in method_li:
            self.xtx_batch += xtx.mean(dim=0)  # (e,e)
            self.xtGx_batch += xtGx.mean(dim=0)  # (e,e)

        if "big_out" in method_li:
            xtx_plus = torch.linalg.pinv(xtx, hermitian=True)  # (b,e,e)
            self.bigf_batch += (xtx_plus @ xtGx @ xtx_plus).mean(dim=0)  # (e,e)

        self.grad_S_batch += self.frozen_S_grad.mean(dim=0)  # (s,s)
        self.S_batch += self.frozen_S.mean(dim=0)  # (s,s)

    def compute_P_stat(self, method_li: list[str]):
        e = self.frozen_x.size(-1)

        # TODO: Check if hermitian to speedup
        if "kro" in method_li:
            self.P_stat["kro"].append(
                (
                    torch.linalg.pinv(self.kro_avg[-1], hermitian=True)
                    @ self.vec_xtGx_avg[-1]
                ).reshape(e, e)
            )
        if "big_in" in method_li:
            xtx_inv = torch.linalg.pinv(self.xtx_avg[-1], hermitian=True)  # (e,e)
            self.P_stat["big_in"].append(xtx_inv @ self.xtGx_avg[-1] @ xtx_inv)  # (e,e)
        if "big_out" in method_li:
            self.P_stat["big_out"].append(self.bigf_avg[-1])  # (e,e)

    def freeze_WQt_WKt(self):
        # Notation (out,in) -> (in,out)
        self.frozen_WQt = self.attn.W_Q.weight.clone().T
        self.frozen_WKt = self.attn.W_K.weight.clone().T

    def reset_layers_WQt_WKt(self, cfg):
        """
        Restore the linear layers W_Q and W_K using the saved WQt and WKt.
        """
        WQ_layer = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias_attention)
        WK_layer = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias_attention)
        with torch.no_grad():
            WQ_layer.weight.copy_(self.frozen_WQt.T)
            WK_layer.weight.copy_(self.frozen_WKt.T)
        self.attn.W_Q = WQ_layer
        self.attn.W_K = WK_layer

    def update_WQ_WK(
        self,
        cfg,
        lbd: float,
        choice_P_stat: str,
        dif: bool = False,
    ):
        """
        Update the linear layers W_Q and W_K.
        Depends on: Frozen WQt and WKt, lbd, P_stat

        dif: If True, find dWQ, dWK = SVD(-lbd * P); instead of finding directly WQ, WK
        """
        assert isinstance(self.P_stat[choice_P_stat][-1], torch.Tensor)
        temp_P = self.P_stat[choice_P_stat][-1]

        new_WQ = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias_attention)
        new_WK = nn.Linear(cfg.d_e, cfg.d_k, bias=cfg.bias_attention)

        if not dif:
            WQtplus1, WKtplus1 = my_svd_low_rank(
                self.frozen_WQt @ self.frozen_WKt.T - lbd * temp_P, cfg.d_k
            )

            # Notation (in, out) -> (out, in)
            WQtplus1 = WQtplus1.T
            WKtplus1 = WKtplus1.T
        else:
            dWQ, dWK = my_svd_low_rank(-lbd * temp_P, cfg.d_k)

            # Notation (in, out) -> (out, in)
            WQtplus1 = (self.frozen_WQt + dWQ).T
            WKtplus1 = (self.frozen_WKt + dWK).T

        with torch.no_grad():
            new_WQ.weight.copy_(WQtplus1)
            new_WK.weight.copy_(WKtplus1)

        self.attn.W_Q = new_WQ
        self.attn.W_K = new_WK
