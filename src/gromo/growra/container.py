"""
Container-level GrowRA utilities for gromo.

Provides :class:`GrowRAModel` and :func:`get_growra_model` to inject growing
adapters into a pretrained model, as well as utilities for extracting,
saving, and loading adapter parameters.

Typical usage::

    from gromo.growra.container import get_growra_model
    import torch.nn as nn

    pretrained = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    model = get_growra_model(pretrained)
    # ... train model a few steps using model.growra_parameters() as optimizer params ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn


if TYPE_CHECKING:
    from collections.abc import Callable

from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.growra.module import (
    GrowRAConv2d,
    GrowRALinear,
    _Conv2dLayerType,
    _GrowRATypes,
    _LinearLayerType,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _infer_features(model: nn.Module) -> tuple[int, int]:
    """Infer in_features and out_features from the first and last layers.

    Parameters
    ----------
    model : nn.Module
        Model to inspect.

    Returns
    -------
    tuple[int, int]
        ``(in_features, out_features)`` inferred from the first and last layer.

    Raises
    ------
    ValueError
        If no linear or conv layer is found.
    """
    first: nn.Module | None = None
    last: nn.Module | None = None
    for m in model.modules():
        if isinstance(m, _LinearLayerType + _Conv2dLayerType):
            if first is None:
                first = m
            last = m

    if first is None or last is None:
        raise ValueError(
            "Cannot infer in_features / out_features from the model. "
            "Pass them explicitly to get_growra_model()."
        )

    in_f = first.in_features if isinstance(first, _LinearLayerType) else first.in_channels
    out_f = last.out_features if isinstance(last, _LinearLayerType) else last.out_channels
    return in_f, out_f


def _matches_target(
    name: str,
    module: nn.Module,
    target_modules: list[str] | None,
    target_types: tuple[type, ...],
) -> bool:
    """Check if a module matches the target criteria.

    Parameters
    ----------
    name : str
        Full dotted name of the module.
    module : nn.Module
        The module instance.
    target_modules : list[str] | None
        If provided, only modules whose name contains one of these strings are
        targeted. If None, all modules of target_types are targeted.
    target_types : tuple[type, ...]
        Layer types to match.

    Returns
    -------
    bool
    """
    if not isinstance(module, target_types):
        return False
    if target_modules is None:
        return True
    return any(t in name for t in target_modules)


def _inject_growra_inplace(
    model: nn.Module,
    scaling: float | Callable[[int], float],
    dropout: float,
    use_dora: bool,
    target_modules: list[str] | None,
) -> None:
    """Replace targeted layers with GrowRA wrappers in-place (rank 0).

    Parameters
    ----------
    model : nn.Module
        Model to modify.
    scaling : float | Callable[[int], float]
        Scaling factor.
    dropout : float
        Dropout probability for the adapter path.
    use_dora : bool
        Whether to enable DoRA magnitude reparameterization.
    target_modules : list[str] | None
        Name filter; ``None`` wraps all linear / conv layers.
    """
    all_types = _LinearLayerType + _Conv2dLayerType
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    wrapped_names: set[str] = set()

    all_named = list(model.named_modules())
    named_dict = dict(all_named)

    for full_name, module in all_named:
        if not isinstance(module, all_types):
            continue
        if not _matches_target(full_name, module, target_modules, all_types):
            continue
        if any(full_name.startswith(wn + ".") for wn in wrapped_names):
            continue

        parts = full_name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            attr_name = parts[0]
        else:
            parent = named_dict[parts[0]]
            attr_name = parts[1]

        if isinstance(module, _LinearLayerType):
            replacement: nn.Module = GrowRALinear(
                module,
                rank=0,
                scaling=scaling,
                dropout=dropout,
                use_dora=use_dora,
                name=f"growra_{full_name}",
            )
        else:
            replacement = GrowRAConv2d(
                module,
                rank=0,
                scaling=scaling,
                dropout=dropout,
                use_dora=use_dora,
                name=f"growra_{full_name}",
            )
        replacements.append((parent, attr_name, replacement))
        wrapped_names.add(full_name)

    for parent, attr_name, replacement in replacements:
        setattr(parent, attr_name, replacement)


# ---------------------------------------------------------------------------
# GrowRAModel
# ---------------------------------------------------------------------------


class GrowRAModel(SequentialGrowingModel):
    """Growing model wrapping a pretrained network with GrowRA adapters.

    All targeted linear and convolutional layers are replaced in-place with
    :class:`~gromo.growra.module.GrowRALinear` /
    :class:`~gromo.growra.module.GrowRAConv2d` wrappers
    whose rank starts at 0. Original weights are frozen.

    The adapter layers are registered as ``_growable_layers`` /
    ``_growing_layers`` so that all
    :class:`~gromo.containers.growing_container.GrowingContainer` growth
    methods (``init_computation``, ``compute_optimal_updates``,
    ``set_growing_layers``, …) work directly on this object.

    Prefer the factory function :func:`get_growra_model` over instantiating this
    class directly.

    Parameters
    ----------
    model : nn.Module
        Fully trained model to adapt. Modified in-place.
    scaling : float | Callable[[int], float]
        Scaling factor applied to every adapter output. A float gives a fixed
        scaling (default ``1.0``, rank-invariant). A callable receives the
        current rank and returns the scaling factor.
    dropout : float
        Dropout probability applied to the input before the adapter path.
        Default ``0.0`` (no dropout).
    use_dora : bool
        Whether to enable DoRA magnitude reparameterization.
    target_modules : list[str] | None
        If provided, only wrap layers whose full name contains one of these
        strings. Wraps all linear / conv layers when ``None``.
    in_features : int | None
        Input feature size (inferred from the first layer when ``None``).
    out_features : int | None
        Output feature size (inferred from the last layer when ``None``).
    device : torch.device | str | None
        Device for the container metadata.
    """

    def __init__(
        self,
        model: nn.Module,
        scaling: float | Callable[[int], float] = 1.0,
        dropout: float = 0.0,
        use_dora: bool = False,
        target_modules: list[str] | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        # Infer I/O dimensions before freezing so a ValueError doesn't leave
        # the caller's model silently frozen.
        if in_features is None or out_features is None:
            inf, outf = _infer_features(model)
            if in_features is None:
                in_features = inf
            if out_features is None:
                out_features = outf

        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )

        for p in model.parameters():
            p.requires_grad = False

        # Inject rank-0 GrowRA wrappers into the model
        _inject_growra_inplace(
            model,
            scaling=scaling,
            dropout=dropout,
            use_dora=use_dora,
            target_modules=target_modules,
        )
        self.model = model
        self._raw_scaling: float | Callable[[int], float] = scaling
        self.dropout = dropout
        self.use_dora = use_dora

        # Register adapter modules as growable / growing layers
        growra_mods: list[GrowRALinear | GrowRAConv2d] = [
            m for m in model.modules() if isinstance(m, _GrowRATypes)
        ]
        self._growable_layers = growra_mods  # type: ignore[assignment]
        self._growing_layers = []
        self.set_growing_layers(scheduling_method="all")

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Forward pass delegating to the wrapped model."""
        return self.model(*args, **kwargs)

    def extended_forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        mask: dict | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run the wrapped model with each GrowRA adapter in extended mode.

        Uses forward hooks to replace each adapter's ``forward`` output with its
        ``extended_forward`` output (which includes the optimal-delta correction
        and the proposed new neurons) while leaving the rest of the wrapped model
        untouched.  Hooks are always removed, even if the forward pass raises.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor forwarded to ``self.model``.
        mask : dict | None
            Unused; present for ``GrowingContainer`` interface compatibility.

        Returns
        -------
        torch.Tensor
            Model output computed with extended adapter activations.
        """
        handles = []
        for m in self.growra_modules():

            def _hook(_mod, inp, _out, _m=m):
                return _m.extended_forward(inp[0])

            handles.append(m.register_forward_hook(_hook))
        try:
            return self.model(x)
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def growra_modules(self) -> list[GrowRALinear | GrowRAConv2d]:
        """Return all adapter modules in the wrapped model."""
        return [m for m in self.model.modules() if isinstance(m, _GrowRATypes)]

    def growra_parameters(self) -> list[nn.Parameter]:
        """Return all trainable adapter parameters (A and B weight matrices)."""
        params: list[nn.Parameter] = []
        for m in self.growra_modules():
            params.extend(m.growra_parameters())
        return params

    def merge(self) -> "GrowRAModel":
        """Merge all adapters back into the original layers.

        After merging, the model no longer has any growable layers.

        Returns
        -------
        GrowRAModel
            Self, with adapter wrappers replaced by merged plain layers.
        """
        merge_all_growra(self.model)
        self._growable_layers = []
        self._growing_layers = []
        return self

    def growra_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract adapter parameters as a portable state dict."""
        return get_growra_state_dict(self.model)

    def load_growra_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load adapter parameters from a state dict."""
        load_growra_state_dict(self.model, state)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        n = len(self.growra_modules())
        scaling_str = "adaptive" if callable(self._raw_scaling) else self._raw_scaling
        s = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scaling={scaling_str}, growra_modules={n}"
        )
        if self.use_dora:
            s += ", use_dora=True"
        if self.dropout > 0.0:
            s += f", dropout={self.dropout}"
        return s


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_growra_model(
    model: nn.Module,
    scaling: float | Callable[[int], float] = 1.0,
    dropout: float = 0.0,
    use_dora: bool = False,
    target_modules: list[str] | None = None,
    in_features: int | None = None,
    out_features: int | None = None,
    device: torch.device | str | None = None,
) -> GrowRAModel:
    """Wrap a pretrained model with growing GrowRA adapters.

    Analogous to PEFT's ``get_peft_model`` but returns a
    :class:`GrowRAModel` (a
    :class:`~gromo.containers.sequential_growing_container.SequentialGrowingModel`).
    Rank starts at 0 and grows via the FOGRO pipeline — no ``rank``
    argument is needed.

    Parameters
    ----------
    model : nn.Module
        Pretrained model to adapt. Modified in-place.
    scaling : float | Callable[[int], float]
        Scaling factor applied to every adapter output. A float gives a fixed
        scaling regardless of rank (default ``1.0``, rank-invariant). A
        callable receives the current rank and returns the scaling factor.
    dropout : float
        Dropout probability applied to the input before the adapter path.
        Default ``0.0`` (no dropout).
    use_dora : bool
        Whether to enable DoRA magnitude reparameterization.
    target_modules : list[str] | None
        Name filter for which layers to wrap. ``None`` wraps all linear / conv
        layers.
    in_features : int | None
        Override for the model input dimension (inferred when ``None``).
    out_features : int | None
        Override for the model output dimension (inferred when ``None``).
    device : torch.device | str | None
        Device for the container metadata.

    Returns
    -------
    GrowRAModel

    Examples
    --------
    >>> import torch.nn as nn
    >>> from gromo.growra.container import get_growra_model
    >>> base = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    >>> model = get_growra_model(base, scaling=1.0)
    """
    return GrowRAModel(
        model=model,
        scaling=scaling,
        dropout=dropout,
        use_dora=use_dora,
        target_modules=target_modules,
        in_features=in_features,
        out_features=out_features,
        device=device,
    )


# ---------------------------------------------------------------------------
# Standalone utilities (work on any nn.Module containing GrowRA adapters)
# ---------------------------------------------------------------------------


def get_growra_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect all trainable adapter parameters from a model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    list[nn.Parameter]
    """
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, _GrowRATypes):
            params.extend(module.growra_parameters())
    return params


def get_growra_modules(
    model: nn.Module,
) -> list[GrowRALinear | GrowRAConv2d]:
    """Collect all GrowRA adapter modules from a model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    list[GrowRALinear | GrowRAConv2d]
    """
    return [m for m in model.modules() if isinstance(m, _GrowRATypes)]


def merge_all_growra(model: nn.Module) -> nn.Module:
    """Merge all adapter weights back into the original model layers.

    Replaces each adapter wrapper with a plain ``nn.Linear`` or ``nn.Conv2d``
    whose weights incorporate the learned low-rank adaptation.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    nn.Module
        Same object with adapter wrappers replaced by merged layers.
    """
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    named_dict = dict(model.named_modules())

    for full_name, module in named_dict.items():
        if isinstance(module, _GrowRATypes):
            parts = full_name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr_name = parts[0]
            else:
                parent = named_dict[parts[0]]
                attr_name = parts[1]
            replacements.append((parent, attr_name, module.merge()))

    for parent, attr_name, merged in replacements:
        setattr(parent, attr_name, merged)

    return model


def get_growra_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the adapter parameters from the model state dict.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict[str, torch.Tensor]
        State dict containing adapter-related keys (weights, rank, scaling).
    """
    growra_state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, _GrowRATypes):
            prefix = f"{name}." if name else ""
            growra_state[f"{prefix}first_layer.weight"] = (
                module.first_layer.weight.data.clone()
            )
            growra_state[f"{prefix}second_layer.weight"] = (
                module.second_layer.weight.data.clone()
            )
            growra_state[f"{prefix}rank"] = torch.tensor(module.rank)
            growra_state[f"{prefix}scaling"] = torch.tensor(
                module.scaling_fn(max(1, module.rank))
            )
            growra_state[f"{prefix}use_dora"] = torch.tensor(module.use_dora)
            if module.use_dora and module.magnitude is not None:
                growra_state[f"{prefix}magnitude"] = module.magnitude.data.clone()
    return growra_state


def load_growra_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load adapter parameters into a model.

    Parameters
    ----------
    model : nn.Module
        Model with GrowRA layers (must have matching structure).
    state : dict[str, torch.Tensor]
        State dict from :func:`get_growra_state_dict`.

    Raises
    ------
    KeyError
        If ``first_layer.weight`` is present for a module but
        ``second_layer.weight`` or ``scaling`` is missing (malformed state).
    ValueError
        If the rank stored in the state is smaller than the module's current
        rank (rank reduction is not supported).
    """
    for name, module in model.named_modules():
        if isinstance(module, _GrowRATypes):
            prefix = f"{name}." if name else ""
            key_a = f"{prefix}first_layer.weight"
            key_b = f"{prefix}second_layer.weight"
            key_s = f"{prefix}scaling"
            if key_a not in state:
                continue
            if key_b not in state or key_s not in state:
                raise KeyError(
                    f"State dict has '{key_a}' but is missing '{key_b}' or '{key_s}'. "
                    "State dict is malformed."
                )
            A_data = state[key_a]
            B_data = state[key_b]
            new_rank = A_data.shape[0]
            if new_rank < module.rank:
                raise ValueError(
                    f"Cannot load rank {new_rank} into module '{name}' which already "
                    f"has rank {module.rank}. Rank reduction is not supported."
                )
            _s = state[key_s].item()
            # NOTE: callable scaling (e.g. RSLoRA) is replaced with the scalar
            # evaluated at the saved rank. The adaptive schedule is not preserved.
            module._raw_scaling = _s
            module.scaling_fn = lambda _, s=_s: s
            module._scaling.scaling_fn = module.scaling_fn
            use_dora_key = f"{prefix}use_dora"
            if use_dora_key in state and bool(state[use_dora_key].item()):
                if not module.use_dora:
                    module.enable_dora()
            if new_rank > module.rank:
                if isinstance(module, GrowRALinear):
                    added = new_rank - module.rank
                    module.first_layer.add_parameters(
                        matrix_extension=None,
                        bias_extension=None,
                        added_out_features=added,
                    )
                    module.second_layer.add_parameters(
                        matrix_extension=None,
                        bias_extension=None,
                        added_in_features=added,
                    )
                else:
                    added = new_rank - module.rank
                    dev_a = module.first_layer.weight.device
                    dev_b = module.second_layer.weight.device
                    # A (first_layer): add output channels — new rows only
                    module.first_layer.layer_out_extension(
                        A_data[module.rank :].to(dev_a)
                    )
                    # B (second_layer): add input channels — new columns only
                    module.second_layer.layer_in_extension(
                        B_data[:, module.rank :].to(dev_b)
                    )
            with torch.no_grad():
                module.first_layer.weight.copy_(
                    A_data.to(module.first_layer.weight.device)
                )
                module.second_layer.weight.copy_(
                    B_data.to(module.second_layer.weight.device)
                )
                magnitude_key = f"{prefix}magnitude"
                if magnitude_key in state and module.magnitude is not None:
                    module.magnitude.copy_(
                        state[magnitude_key].to(module.magnitude.device)
                    )
