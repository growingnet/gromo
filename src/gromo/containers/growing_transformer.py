from typing import Any, Dict
from warnings import warn

import torch
import torch.nn as nn

from gromo.containers.growing_block import LinearGrowingBlock
from gromo.containers.growing_container import GrowingContainer
from gromo.modules.growing_drop_path import DropPath
from gromo.modules.growing_dropout import GrowingDropout
from gromo.modules.growing_module import GrowingModule
from gromo.utils.utils import compute_tensor_stats


class Attention(nn.Module):
    """CCT-style multi-head self-attention with qkv projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        projection_dropout: float = 0.1,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False, device=device)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim, device=device)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute self-attention over a token sequence."""
        batch_size, sequence_length, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                sequence_length,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, sequence_length, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class MaskedAttention(Attention):
    """CCT-style masked self-attention using token-validity masks."""

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked self-attention over a token sequence."""
        batch_size, sequence_length, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                sequence_length,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._resolve_mask(
            batch_size=batch_size,
            sequence_length=sequence_length,
            mask=mask,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn = attn.masked_fill(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, sequence_length, channels)
        x = self.proj(x)
        return self.proj_drop(x)

    def _resolve_mask(
        self,
        batch_size: int,
        sequence_length: int,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if mask is not None:
            resolved_mask = mask
        elif key_padding_mask is not None:
            resolved_mask = ~key_padding_mask.to(dtype=torch.bool)
        elif (
            attn_mask is not None
            and attn_mask.dim() == 2
            and attn_mask.shape == (batch_size, sequence_length)
        ):
            resolved_mask = attn_mask
        else:
            if attn_mask is not None:
                raise ValueError(
                    "`attn_mask` must have shape (batch_size, sequence_length) to "
                    "match CCT `MaskedAttention`. Use `mask` or `key_padding_mask`."
                )
            return None

        if resolved_mask.shape[-1] != sequence_length:
            raise ValueError("mask has incorrect dimensions")
        return resolved_mask.to(dtype=torch.bool, device=self.qkv.weight.device)


class GrowingTransformerBlock(GrowingContainer):
    """CCT-compatible transformer encoder layer with a growable MLP branch."""

    def __init__(
        self,
        d_model: int,
        num_heads: int | None = None,
        d_ff: int | None = None,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        nhead: int | None = None,
        dim_feedforward: int | None = 2048,
        attention_dropout: float | None = None,
        drop_path_rate: float = 0.0,
        projection_dropout: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if args:
            raise TypeError(f"Unexpected positional arguments: {args!r}")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        if num_heads is None:
            num_heads = nhead
        if num_heads is None:
            raise ValueError("`num_heads` or CCT-compatible `nhead` must be provided.")
        if d_ff is None:
            d_ff = dim_feedforward
        if d_ff is None:
            raise ValueError(
                "`d_ff` or CCT-compatible `dim_feedforward` must be provided."
            )
        if attention_dropout is None:
            attention_dropout = dropout
        if projection_dropout is None:
            projection_dropout = dropout

        super().__init__(
            in_features=d_model,
            out_features=d_model,
            device=device,
            name="GrowingTransformerBlock",
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate

        self.pre_norm = nn.LayerNorm(d_model, device=self.device)
        self.self_attn = MaskedAttention(
            dim=d_model,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            device=self.device,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model, device=self.device)
        dropout1 = GrowingDropout(dropout_rate=dropout) if dropout > 0 else nn.Identity()
        dropout2 = GrowingDropout(dropout_rate=dropout) if dropout > 0 else nn.Identity()
        self.mlp = LinearGrowingBlock(
            in_features=d_model,
            out_features=d_model,
            hidden_features=d_ff,
            pre_activation=nn.Identity(),
            mid_activation=nn.Sequential(
                nn.GELU(),
                dropout1,
            ),
            pre_addition_function=nn.Sequential(
                dropout2,
                DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity(),
            ),
            name="mlp",
            device=self.device,
        )
        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        """Register the submodules that participate in the growth procedure."""
        self._growing_layers = [self.mlp]

    @property
    def optimal_delta_layer(self) -> torch.nn.Module | None:
        """Expose the optimal delta layer of the growable MLP branch."""
        return self.mlp.optimal_delta_layer

    @optimal_delta_layer.setter
    def optimal_delta_layer(self, value: torch.nn.Module | None) -> None:
        """Set the optimal delta layer on the growable MLP branch."""
        self.mlp.optimal_delta_layer = value

    @property
    def hidden_features(self) -> int:
        """Backward-compatible alias for the number of hidden neurons."""
        warn(
            "hidden_features is deprecated, use hidden_neurons instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.hidden_neurons

    @property
    def hidden_neurons(self) -> int:
        """Return the number of hidden neurons in the feed-forward branch."""
        return self.mlp.hidden_neurons

    @property
    def growth_module(self) -> GrowingModule:
        """Return the leaf growable module that stores scaling tensors."""
        return self.mlp.second_layer

    @property
    def eigenvalues_extension(self) -> torch.Tensor | None:
        """Expose the current eigenvalues associated with the pending extension."""
        return self.mlp.eigenvalues_extension

    @property
    def parameter_update_decrease(self) -> torch.Tensor | None:
        """Expose the first-order decrease from the parameter-only update."""
        return self.mlp.parameter_update_decrease

    @parameter_update_decrease.setter
    def parameter_update_decrease(self, value: torch.Tensor | float) -> None:
        """Set the cached first-order decrease on the growable branch."""
        self.mlp.parameter_update_decrease = value

    @property
    def scaling_factor(self) -> torch.Tensor:
        """Expose the scaling factor used by the growable branch."""
        return self.mlp.scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, value: float) -> None:
        """Set the scaling factor used by the growable branch."""
        self.mlp.scaling_factor = value

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """Return the total first-order improvement of the current proposal."""
        return self.mlp.first_order_improvement

    def _apply_attention_branch(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the fixed attention branch before delegating to the growing MLP."""
        x = x + self.drop_path(
            self.self_attn(
                self.pre_norm(x),
                mask=mask,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        )
        return self.norm1(x)

    def forward(
        self,
        x,
        mask=None,
        attn_mask=None,
        key_padding_mask=None,
        *args,
        **kwargs,
    ):
        """Run attention, then delegate the feed-forward path to LinearGrowingBlock."""
        _ = args, kwargs
        return self.mlp(
            self._apply_attention_branch(
                x,
                mask=mask,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        )

    def extended_forward(
        self,
        x,
        mask=None,
        attn_mask=None,
        key_padding_mask=None,
        attention_mask=None,
    ):
        """Run attention, then delegate the extended path to LinearGrowingBlock."""
        return self.mlp.extended_forward(
            self._apply_attention_branch(
                x,
                mask=attention_mask,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            ),
            mask=mask,
        )

    def delete_update(self, **kwargs: Any) -> None:
        """Discard the currently cached growth proposal."""
        self.mlp.delete_update(**kwargs)

    def set_scaling_factor(self, factor: float) -> None:
        """Assign the same scaling factor to the growable branch."""
        self.mlp.set_scaling_factor(factor)

    def apply_change(
        self,
        extension_size: int | None = None,
        scaling_factor: float | torch.Tensor | None = None,
        apply_delta: bool = True,
        apply_extension: bool = True,
    ) -> None:
        """Apply the selected growth change to the MLP branch."""
        self.mlp.apply_change(
            extension_size=extension_size,
            scaling_factor=scaling_factor,
            apply_delta=apply_delta,
            apply_extension=apply_extension,
        )

    def missing_neurons(self) -> int:
        """Return how many hidden neurons remain before reaching the target width."""
        return self.mlp.missing_neurons()

    def number_of_neurons_to_add(self, **kwargs: Any) -> int:
        """Delegate the next growth-step width to the growable MLP branch."""
        return self.mlp.number_of_neurons_to_add(**kwargs)

    def complete_growth(self, extension_kwargs: Any) -> None:
        """Expand the growable MLP branch to its configured target width."""
        self.mlp.complete_growth(extension_kwargs=extension_kwargs)

    def create_layer_extensions(self, **kwargs: Any) -> None:
        """Create random growth extensions on the growable MLP branch."""
        self.mlp.create_layer_extensions(**kwargs)

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int | None = None,
        threshold: float | None = None,
        sub_select_previous: bool = True,
        zeros_if_not_enough: bool = False,
        zeros_fan_in: bool = True,
        zeros_fan_out: bool = False,
    ) -> None:
        """Keep only a subset of the proposed added neurons."""
        self.mlp.sub_select_optimal_added_parameters(
            keep_neurons=keep_neurons,
            threshold=threshold,
            sub_select_previous=sub_select_previous,
            zeros_if_not_enough=zeros_if_not_enough,
            zeros_fan_in=zeros_fan_in,
            zeros_fan_out=zeros_fan_out,
        )

    def apply_rescaling(self, **kwargs: Any) -> None:
        """Apply variance-transfer rescaling on the growable MLP branch."""
        self.mlp.apply_rescaling(**kwargs)

    def apply_neuron_pairing(self, **kwargs: Any) -> None:
        """Apply neuron pairing on the growable MLP branch."""
        self.mlp.apply_neuron_pairing(**kwargs)

    def normalize_optimal_updates(self, **kwargs: Any) -> None:
        """Normalize growth updates on the growable MLP branch."""
        self.mlp.normalize_optimal_updates(**kwargs)

    def update_information(self) -> Dict[str, Any]:
        """Summarize the current growth proposal for reporting and comparison."""
        eigenvalues = self.eigenvalues_extension
        return {
            "update_value": self.first_order_improvement,
            "parameter_improvement": self.parameter_update_decrease,
            "eigenvalues_extension": eigenvalues,
            "scaling_factor": self.scaling_factor,
            "added_neurons": 0 if eigenvalues is None else eigenvalues.shape[0],
            "d_model": self.in_features,
            "d_ff": self.hidden_neurons,
        }

    def weights_statistics(self) -> Dict[str, Any]:
        """Collect summary statistics for attention and MLP parameters."""
        statistics: Dict[str, Any] = {
            "attention": {
                "qkv_weight": compute_tensor_stats(self.self_attn.qkv.weight),
                "proj_weight": compute_tensor_stats(self.self_attn.proj.weight),
            },
            "pre_norm": {
                "weight": compute_tensor_stats(self.pre_norm.weight),
                "bias": compute_tensor_stats(self.pre_norm.bias),
            },
            "norm1": {
                "weight": compute_tensor_stats(self.norm1.weight),
                "bias": compute_tensor_stats(self.norm1.bias),
            },
            "mlp": self.mlp.weights_statistics(),
            "d_model": self.in_features,
            "d_ff": self.hidden_neurons,
        }
        if self.self_attn.qkv.bias is not None:
            statistics["attention"]["qkv_bias"] = compute_tensor_stats(
                self.self_attn.qkv.bias
            )
        if self.self_attn.proj.bias is not None:
            statistics["attention"]["proj_bias"] = compute_tensor_stats(
                self.self_attn.proj.bias
            )
        return statistics
