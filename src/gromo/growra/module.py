"""
GrowRA module classes for gromo.

Defines :class:`GrowRALinear` and :class:`GrowRAConv2d`, which wrap
a frozen original layer and add a trainable low-rank adaptation::

    output = original(x) + scaling * B(A(x))

where A and B are growing modules from gromo.  The rank starts at 0 and grows
via the FOGRO pipeline (see :mod:`gromo.growra.container`).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch import functional as torch_functional

from gromo.containers.growing_block import Conv2dGrowingBlock, LinearGrowingBlock
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule


# Types accepted as the "original layer" for linear GrowRA
_LinearLayerType = (nn.Linear, LinearGrowingModule)
# Types accepted as the "original layer" for conv GrowRA
_Conv2dLayerType = (nn.Conv2d, Conv2dGrowingModule)


class GrowRALinear(LinearGrowingBlock):
    """GrowRA block for nn.Linear (or LinearGrowingModule).

    The decomposition is ``W_original + scaling * B @ A`` where A and B
    are the two internal layers of this ``LinearGrowingBlock``. The frozen
    original layer is used as the residual/downsample path.

    Parameters
    ----------
    linear : nn.Linear | LinearGrowingModule
        Original linear layer (will be frozen).
    rank : int
        Initial rank. Default 0 (no adaptation).
    alpha : float
        Scaling factor. Effective scaling is ``alpha / rank``.
    dropout : float
        Dropout probability applied to the input before the adapter path.
        Disabled (``p=0.0``) by default.
    use_dora : bool
        If ``True``, use DoRA-style magnitude reparameterization on top of the
        adapter direction update.
    target_rank : int | None
        Target rank for the growing block.
    activation : torch.nn.Module | None
        Activation between A and B. Default ``nn.Identity()``.
    device : torch.device | None
        Device for parameters.
    name : str
        Name for the growing block.
    """

    def __init__(
        self,
        linear: nn.Linear | LinearGrowingModule,
        rank: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
        use_dora: bool = False,
        target_rank: int | None = None,
        activation: torch.nn.Module | None = None,
        device: torch.device | None = None,
        name: str = "growra_block",
    ):
        linear.requires_grad_(False)
        self.alpha = alpha
        if device is None:
            device = linear.weight.device

        if activation is None:
            activation = nn.Identity()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Initializing zero-element tensors is a no-op",
                category=UserWarning,
            )
            super().__init__(
                in_features=linear.in_features,
                out_features=linear.out_features,
                hidden_features=rank,
                target_hidden_features=target_rank,
                activation=activation,
                name=name,
                kwargs_layer={"use_bias": False},
                downsample=linear,
                device=device,
            )
        self.linear = linear
        self.dropout = nn.Dropout(p=dropout)
        self.use_dora = False
        self.magnitude: nn.Parameter | None = None
        if use_dora:
            self.enable_dora()

    @property
    def rank(self) -> int:
        """Current rank (hidden dimension)."""
        return self.hidden_neurons

    @property
    def scaling(self) -> float:
        """Effective scaling factor ``alpha / rank``."""
        if self.rank == 0:
            return 0.0
        return self.alpha / self.rank

    @property
    def weight(self) -> torch.Tensor:
        """Original weight (read-only)."""
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Original bias (read-only)."""
        return self.linear.bias

    def _weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.norm(dim=1, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps)

    def _delta_weight(self) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros_like(self.linear.weight)
        return self.scaling * (self.second_layer.weight @ self.first_layer.weight)

    def _effective_weight(self) -> torch.Tensor:
        weight = self.linear.weight + self._delta_weight()
        if not self.use_dora:
            return weight
        assert self.magnitude is not None
        return self.magnitude[:, None] * (weight / self._weight_norm(weight))

    def enable_dora(self) -> None:
        """Enable DoRA magnitude reparameterization."""
        self.use_dora = True
        with torch.no_grad():
            magnitude = self._weight_norm(self.linear.weight).squeeze(1)
        self.magnitude = nn.Parameter(magnitude.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ``original(x) + scaling * B(A(x))``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., out_features)``.
        """
        if not self.use_dora:
            base_out = self.linear(x)
        else:
            base_out = torch_functional.F.linear(
                x, self._effective_weight(), self.linear.bias
            )
        if self.rank == 0 and not self.first_layer.store_input:
            return base_out
        if self.use_dora:
            return base_out
        x_drop = self.dropout(x)
        block_out = super().forward(x_drop)
        adapter_out = block_out - self.linear(x_drop)
        return base_out + self.scaling * adapter_out

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extended forward including computed optimal growth directions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        if not self.use_dora:
            base_out = self.linear(x)
        else:
            base_out = torch_functional.F.linear(
                x, self._effective_weight(), self.linear.bias
            )
            if self.rank == 0 and self.first_layer.extended_output_layer is None:
                return base_out
            return base_out
        x_drop = self.dropout(x)
        block_out = super().extended_forward(x_drop)
        if self.rank == 0 and self.first_layer.extended_output_layer is None:
            return base_out
        return base_out + self.scaling * (block_out - self.linear(x_drop))

    def merge(self) -> nn.Linear:
        """Merge adapter into the original layer.

        Returns
        -------
        nn.Linear
            New linear layer with merged weights.
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.linear.bias is not None,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self._effective_weight())
            if self.linear.bias is not None:
                merged.bias.copy_(self.linear.bias)
        return merged

    def growra_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable adapter parameters (A and B layers)."""
        params = list(self.first_layer.parameters()) + list(
            self.second_layer.parameters()
        )
        if self.use_dora and self.magnitude is not None:
            params.append(self.magnitude)
        return params

    def reset_adapter(self) -> None:
        """Reset adapter to zero output."""
        nn.init.kaiming_uniform_(self.first_layer.weight)
        nn.init.zeros_(self.second_layer.weight)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        dropout_p = self.dropout.p
        s = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )
        if self.use_dora:
            s += ", use_dora=True"
        if dropout_p > 0.0:
            s += f", dropout={dropout_p}"
        return s


class GrowRAConv2d(Conv2dGrowingBlock):
    """GrowRA wrapper for nn.Conv2d (or Conv2dGrowingModule).

    The decomposition is ``Conv_original(x) + scaling * B(A(x))``
    where A and B are ``Conv2dGrowingModule`` instances composed via a
    ``Conv2dGrowingBlock``. The hidden channels equal the rank and
    start at 0 by default.

    Parameters
    ----------
    conv : nn.Conv2d | Conv2dGrowingModule
        Original convolution layer (will be frozen).
    rank : int
        Initial rank (hidden channels). Default 0.
    alpha : float
        Scaling factor. Effective scaling is ``alpha / rank``.
    dropout : float
        Dropout probability applied to the input before the adapter path.
        Disabled (``p=0.0``) by default.
    use_dora : bool
        If ``True``, use DoRA-style magnitude reparameterization on top of the
        adapter direction update.
    target_rank : int | None
        Target rank for the growing block.
    activation : torch.nn.Module | None
        Activation between A and B. Default ``nn.Identity()``.
    device : torch.device | None
        Device for parameters.
    name : str
        Name for the growing block.
    """

    def __init__(
        self,
        conv: nn.Conv2d | Conv2dGrowingModule,
        rank: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
        use_dora: bool = False,
        target_rank: int | None = None,
        activation: torch.nn.Module | None = None,
        device: torch.device | None = None,
        name: str = "growra_conv_block",
    ):
        if isinstance(conv, Conv2dGrowingModule):
            underlying = conv.layer
        else:
            underlying = conv
        if device is None:
            device = underlying.weight.device
        self.in_channels = underlying.in_channels
        self.out_channels = underlying.out_channels
        self.alpha = alpha

        conv.requires_grad_(False)

        if activation is None:
            activation = nn.Identity()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Initializing zero-element tensors is a no-op",
                category=UserWarning,
            )
            super().__init__(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                hidden_channels=rank,
                target_hidden_channels=target_rank,
                activation=activation,
                name=name,
                kwargs_first_layer={
                    "use_bias": False,
                    "kernel_size": underlying.kernel_size,
                    "stride": underlying.stride,
                    "padding": underlying.padding,
                    "dilation": underlying.dilation,
                },
                kwargs_second_layer={
                    "use_bias": False,
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                },
                downsample=conv,
                device=device,
            )
        self.conv = conv
        self.dropout = nn.Dropout(p=dropout)
        self.use_dora = False
        self.magnitude: nn.Parameter | None = None
        if use_dora:
            self.enable_dora()

    @property
    def rank(self) -> int:
        """Current rank (hidden channels)."""
        return self.hidden_neurons

    @property
    def scaling(self) -> float:
        """Effective scaling factor ``alpha / rank``."""
        if self.rank == 0:
            return 0.0
        return self.alpha / self.rank

    @property
    def weight(self) -> torch.Tensor:
        """Original weight (read-only)."""
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Original bias (read-only)."""
        return self.conv.bias

    def _conv_base(self) -> nn.Conv2d:
        if isinstance(self.conv, Conv2dGrowingModule):
            return self.conv.layer
        return self.conv

    def _weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        return (
            weight.flatten(1)
            .norm(dim=1, keepdim=True)
            .clamp_min(torch.finfo(weight.dtype).eps)[:, None, None]
        )

    def _delta_weight(self) -> torch.Tensor:
        orig = self._conv_base()
        if self.rank == 0:
            return torch.zeros_like(orig.weight)
        a_w = self.first_layer.weight
        b_w = self.second_layer.weight
        b_mat = b_w.squeeze(-1).squeeze(-1)
        a_flat = a_w.view(a_w.shape[0], -1)
        delta_flat = b_mat @ a_flat
        return self.scaling * delta_flat.view_as(orig.weight)

    def _effective_weight(self) -> torch.Tensor:
        orig = self._conv_base()
        weight = orig.weight + self._delta_weight()
        if not self.use_dora:
            return weight
        assert self.magnitude is not None
        return self.magnitude[:, None, None, None] * (weight / self._weight_norm(weight))

    def enable_dora(self) -> None:
        """Enable DoRA magnitude reparameterization."""
        self.use_dora = True
        orig = self._conv_base()
        with torch.no_grad():
            magnitude = self._weight_norm(orig.weight).reshape(-1)
        self.magnitude = nn.Parameter(magnitude.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ``conv(x) + scaling * B(A(x))``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(N, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(N, C_out, H_out, W_out)``.
        """
        if not self.use_dora:
            base_out = self.conv(x)
        else:
            orig = self._conv_base()
            base_out = torch_functional.F.conv2d(
                x,
                self._effective_weight(),
                orig.bias,
                orig.stride,
                orig.padding,
                orig.dilation,
                orig.groups,
            )
        if self.rank == 0 and not self.first_layer.store_input:
            return base_out
        if self.use_dora:
            return base_out
        x_drop = self.dropout(x)
        block_out = super().forward(x_drop)
        adapter_out = block_out - self.conv(x_drop)
        return base_out + self.scaling * adapter_out

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extended forward including computed optimal growth directions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        if not self.use_dora:
            base_out = self.conv(x)
        else:
            orig = self._conv_base()
            base_out = torch_functional.F.conv2d(
                x,
                self._effective_weight(),
                orig.bias,
                orig.stride,
                orig.padding,
                orig.dilation,
                orig.groups,
            )
            if self.rank == 0 and self.first_layer.extended_output_layer is None:
                return base_out
            return base_out
        x_drop = self.dropout(x)
        block_out = super().extended_forward(x_drop)
        if self.rank == 0 and self.first_layer.extended_output_layer is None:
            return base_out
        return base_out + self.scaling * (block_out - self.conv(x_drop))

    def merge(self) -> nn.Conv2d:
        """Merge adapter into the original convolution layer.

        Note: merging is only exact when both A and B use the same
        kernel_size, stride, padding, and dilation as the original.

        Returns
        -------
        nn.Conv2d
            New conv layer with merged weights.
        """
        orig = self._conv_base()
        merged = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            dilation=orig.dilation,
            groups=orig.groups,
            bias=orig.bias is not None,
            device=orig.weight.device,
            dtype=orig.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self._effective_weight())
            if orig.bias is not None:
                merged.bias.copy_(orig.bias)
        return merged

    def growra_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable adapter parameters (A and B layers)."""
        params = list(self.first_layer.parameters()) + list(
            self.second_layer.parameters()
        )
        if self.use_dora and self.magnitude is not None:
            params.append(self.magnitude)
        return params

    def reset_adapter(self) -> None:
        """Reset adapter to zero output."""
        nn.init.kaiming_uniform_(self.first_layer.weight)
        nn.init.zeros_(self.second_layer.weight)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        dropout_p = self.dropout.p
        s = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )
        if self.use_dora:
            s += ", use_dora=True"
        if dropout_p > 0.0:
            s += f", dropout={dropout_p}"
        return s


# Union type for any GrowRA adapter
_GrowRATypes = (GrowRALinear, GrowRAConv2d)
