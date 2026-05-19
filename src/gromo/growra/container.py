"""
Container-level LoRA utilities for gromo.

Provides :class:`LoRAGrowingModel` and :func:`get_growing_lora_model` to inject growing
LoRA adapters into a pretrained model, as well as utilities for extracting,
saving, and loading LoRA parameters.

Typical usage::

    from gromo.growra.container import get_growing_lora_model
    import torch.nn as nn

    pretrained = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    model = get_growing_lora_model(pretrained, alpha=1.0)
    # ... train model a few steps using model.lora_parameters() as optimizer params ...
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.growra.module import (
    GrowingLoRAConv2d,
    GrowingLoRALinear,
    _Conv2dLayerType,
    _LinearLayerType,
    _LoRATypes,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _infer_features(model: nn.Module) -> tuple[int, int]:
    """Infer in_features and out_features from the first and last layers.

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
            "Pass them explicitly to get_growing_lora_model()."
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
    target_modules : list of str or None
        If provided, only modules whose name contains one of these strings are
        targeted. If None, all modules of target_types are targeted.
    target_types : tuple of type
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


def _inject_lora_inplace(
    model: nn.Module,
    alpha: float,
    lora_dropout: float,
    use_dora: bool,
    target_modules: list[str] | None,
) -> None:
    """Replace targeted layers with LoRA wrappers in-place (rank 0).

    Parameters
    ----------
    model : nn.Module
        Model to modify.
    alpha : float
        LoRA scaling factor.
    lora_dropout : float
        Dropout probability for the LoRA path.
    use_dora : bool
        Whether to enable DoRA magnitude reparameterization.
    target_modules : list of str or None
        Name filter; ``None`` wraps all linear / conv layers.
    """
    all_types = _LinearLayerType + _Conv2dLayerType
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    wrapped_names: set[str] = set()

    for full_name, module in model.named_modules():
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
            parent = dict(model.named_modules())[parts[0]]
            attr_name = parts[1]

        if isinstance(module, _LinearLayerType):
            replacement: nn.Module = GrowingLoRALinear(
                module,
                rank=0,
                alpha=alpha,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
                name=f"lora_{full_name}",
            )
        else:
            replacement = GrowingLoRAConv2d(
                module,
                rank=0,
                alpha=alpha,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
                name=f"lora_{full_name}",
            )
        replacements.append((parent, attr_name, replacement))
        wrapped_names.add(full_name)

    for parent, attr_name, replacement in replacements:
        setattr(parent, attr_name, replacement)


# ---------------------------------------------------------------------------
# LoRAGrowingModel
# ---------------------------------------------------------------------------


class LoRAGrowingModel(SequentialGrowingModel):
    """Growing model wrapping a pretrained network with LoRA adapters.

    All targeted linear and convolutional layers are replaced in-place with
    :class:`~gromo.growra.module.GrowingLoRALinear` /
    :class:`~gromo.growra.module.GrowingLoRAConv2d` wrappers
    whose LoRA rank starts at 0. Original weights are frozen.

    The LoRA layers are registered as ``_growable_layers`` /
    ``_growing_layers`` so that all
    :class:`~gromo.containers.growing_container.GrowingContainer` growth
    methods (``init_computation``, ``compute_optimal_updates``,
    ``set_growing_layers``, …) work directly on this object.

    Prefer the factory function :func:`get_growing_lora_model` over instantiating this
    class directly.

    Parameters
    ----------
    model : nn.Module
        Fully trained model to adapt. Modified in-place.
    alpha : float
        LoRA scaling factor (``effective_scaling = alpha / rank``).
    lora_dropout : float
        Dropout probability applied to the input before the LoRA path.
        Default ``0.0`` (no dropout).
    target_modules : list of str or None
        If provided, only wrap layers whose full name contains one of these
        strings. Wraps all linear / conv layers when ``None``.
    in_features : int or None
        Input feature size (inferred from the first layer when ``None``).
    out_features : int or None
        Output feature size (inferred from the last layer when ``None``).
    device : torch.device or str or None
        Device for the container metadata.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.0,
        lora_dropout: float = 0.0,
        use_dora: bool = False,
        target_modules: list[str] | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        # Freeze original weights before injecting LoRA
        for p in model.parameters():
            p.requires_grad = False

        # Infer I/O dimensions for SequentialGrowingModel
        if in_features is None or out_features is None:
            inf, outf = _infer_features(model)
            if in_features is None:
                in_features = inf
            if out_features is None:
                out_features = outf

        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )

        # Inject rank-0 LoRA wrappers into the model
        _inject_lora_inplace(
            model,
            alpha=alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            target_modules=target_modules,
        )
        self.model = model
        self.alpha = alpha
        self.lora_dropout = lora_dropout
        self.use_dora = use_dora

        # Register LoRA modules as growable / growing layers
        lora_mods: list[GrowingLoRALinear | GrowingLoRAConv2d] = [
            m for m in model.modules() if isinstance(m, _LoRATypes)
        ]
        self._growable_layers = lora_mods  # type: ignore[assignment]
        self._growing_layers = []
        self.set_growing_layers(scheduling_method="all")

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Forward pass delegating to the wrapped model."""
        return self.model(*args, **kwargs)

    def extended_forward(  # type: ignore[override]
        self, x: torch.Tensor, mask: dict | None = None
    ) -> torch.Tensor:
        """Not supported for arbitrary wrapped models.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "extended_forward is not supported for LoRAGrowingModel. "
            "Use the FOGRO growth pipeline (fogro_growth_step) instead."
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def lora_modules(self) -> list[GrowingLoRALinear | GrowingLoRAConv2d]:
        """Return all LoRA adapter modules in the wrapped model."""
        return [m for m in self.model.modules() if isinstance(m, _LoRATypes)]

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return all trainable LoRA parameters (A and B weight matrices)."""
        params: list[nn.Parameter] = []
        for m in self.lora_modules():
            params.extend(m.lora_parameters())
        return params

    def merge_lora(self) -> "LoRAGrowingModel":
        """Merge all LoRA adapters back into the original layers.

        After merging, the model no longer has any growable layers.

        Returns
        -------
        LoRAGrowingModel
            Self, with LoRA wrappers replaced by merged plain layers.
        """
        merge_all_lora(self.model)
        self._growable_layers = []
        self._growing_layers = []
        return self

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract LoRA parameters as a portable state dict."""
        return get_lora_state_dict(self.model)

    def load_lora_state_dict(self, lora_state: dict[str, torch.Tensor]) -> None:
        """Load LoRA parameters from a state dict."""
        load_lora_state_dict(self.model, lora_state)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        n = len(self.lora_modules())
        s = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"alpha={self.alpha}, lora_modules={n}"
        )
        if self.use_dora:
            s += ", use_dora=True"
        if self.lora_dropout > 0.0:
            s += f", lora_dropout={self.lora_dropout}"
        return s


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_growing_lora_model(
    model: nn.Module,
    alpha: float = 1.0,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    target_modules: list[str] | None = None,
    in_features: int | None = None,
    out_features: int | None = None,
    device: torch.device | str | None = None,
) -> LoRAGrowingModel:
    """Wrap a pretrained model with growing LoRA adapters.

    Analogous to PEFT's ``get_peft_model`` but returns a
    :class:`LoRAGrowingModel` (a
    :class:`~gromo.containers.sequential_growing_container.SequentialGrowingModel`).
    LoRA rank starts at 0 and grows via the FOGRO pipeline — no ``rank``
    argument is needed.

    Parameters
    ----------
    model : nn.Module
        Pretrained model to adapt. Modified in-place.
    alpha : float
        LoRA scaling factor.
    lora_dropout : float
        Dropout probability applied to the input before the LoRA path.
        Default ``0.0`` (no dropout).
    use_dora : bool
        Whether to enable DoRA magnitude reparameterization.
    target_modules : list of str or None
        Name filter for which layers to wrap. ``None`` wraps all linear / conv
        layers.
    in_features : int or None
        Override for the model input dimension (inferred when ``None``).
    out_features : int or None
        Override for the model output dimension (inferred when ``None``).
    device : torch.device or str or None
        Device for the container metadata.

    Returns
    -------
    LoRAGrowingModel

    Examples
    --------
    >>> import torch.nn as nn
    >>> from gromo.growra.container import get_growing_lora_model
    >>> base = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    >>> model = get_growing_lora_model(base, alpha=1.0)
    """
    return LoRAGrowingModel(
        model=model,
        alpha=alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        target_modules=target_modules,
        in_features=in_features,
        out_features=out_features,
        device=device,
    )


# ---------------------------------------------------------------------------
# Standalone utilities (work on any nn.Module containing LoRA wrappers)
# ---------------------------------------------------------------------------


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect all trainable LoRA parameters from a model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    list of nn.Parameter
    """
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, _LoRATypes):
            params.extend(module.lora_parameters())
    return params


def get_lora_modules(
    model: nn.Module,
) -> list[GrowingLoRALinear | GrowingLoRAConv2d]:
    """Collect all LoRA modules from a model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    list of GrowingLoRALinear | GrowingLoRAConv2d
    """
    return [m for m in model.modules() if isinstance(m, _LoRATypes)]


def merge_all_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights back into the original model layers.

    Replaces each LoRA wrapper with a plain ``nn.Linear`` or ``nn.Conv2d``
    whose weights incorporate the learned low-rank adaptation.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    nn.Module
        Same object with LoRA wrappers replaced by merged layers.
    """
    replacements: list[tuple[nn.Module, str, nn.Module]] = []

    for full_name, module in model.named_modules():
        if isinstance(module, _LoRATypes):
            parts = full_name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr_name = parts[0]
            else:
                parent = dict(model.named_modules())[parts[0]]
                attr_name = parts[1]
            replacements.append((parent, attr_name, module.merge_lora()))

    for parent, attr_name, merged in replacements:
        setattr(parent, attr_name, merged)

    return model


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA parameters from the model state dict.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict
        State dict containing LoRA-related keys (weights, rank, alpha).
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, _LoRATypes):
            prefix = f"{name}." if name else ""
            lora_state[f"{prefix}first_layer.weight"] = (
                module.first_layer.weight.data.clone()
            )
            lora_state[f"{prefix}second_layer.weight"] = (
                module.second_layer.weight.data.clone()
            )
            lora_state[f"{prefix}rank"] = torch.tensor(module.rank)
            lora_state[f"{prefix}alpha"] = torch.tensor(module.alpha)
            lora_state[f"{prefix}use_dora"] = torch.tensor(module.use_dora)
            if module.use_dora and module.magnitude is not None:
                lora_state[f"{prefix}magnitude"] = module.magnitude.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: dict[str, torch.Tensor]) -> None:
    """Load LoRA parameters into a model.

    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers (must have matching structure).
    lora_state : dict
        State dict from :func:`get_lora_state_dict`.
    """
    for name, module in model.named_modules():
        if isinstance(module, _LoRATypes):
            prefix = f"{name}." if name else ""
            key_a = f"{prefix}first_layer.weight"
            if key_a in lora_state:
                A_data = lora_state[key_a]
                B_data = lora_state[f"{prefix}second_layer.weight"]
                new_rank = A_data.shape[0]
                module.alpha = lora_state[f"{prefix}alpha"].item()
                use_dora_key = f"{prefix}use_dora"
                if use_dora_key in lora_state and bool(lora_state[use_dora_key].item()):
                    if not module.use_dora:
                        module.enable_dora()
                if new_rank > module.rank and isinstance(module, GrowingLoRALinear):
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
                with torch.no_grad():
                    module.first_layer.weight.copy_(
                        A_data.to(module.first_layer.weight.device)
                    )
                    module.second_layer.weight.copy_(
                        B_data.to(module.second_layer.weight.device)
                    )
                    magnitude_key = f"{prefix}magnitude"
                    if magnitude_key in lora_state and module.magnitude is not None:
                        module.magnitude.copy_(
                            lora_state[magnitude_key].to(module.magnitude.device)
                        )
