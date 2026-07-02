import torch
import torch.nn as nn


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Apply stochastic depth (per-sample residual path dropping).

    During training, each batch element is kept independently with probability
    ``1 - drop_prob``. Kept paths are scaled by ``1 / (1 - drop_prob)`` so the
    expected output matches the input (unbiased stochastic depth).
    """
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"`drop_prob` must lie in [0, 1], got {drop_prob}.")
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    batch_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    batch_keep_mask = torch.rand(batch_shape, dtype=x.dtype, device=x.device) < keep_prob
    return x * batch_keep_mask / keep_prob


class GrowingDropPath(nn.Module):
    """Stochastic depth for growing architectures.

    Applies drop-path to the main activation during ``forward`` and
    ``extended_forward``; extension tensors are passed through unchanged.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth to the input tensor."""
        return drop_path(x, self.drop_prob, self.training)

    def extended_forward(
        self,
        x: torch.Tensor | None,
        x_ext: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply stochastic depth while preserving the extension tensor."""
        return self(x) if x is not None else None, x_ext


# Backward-compatible alias used by the growing transformer containers.
DropPath = GrowingDropPath
