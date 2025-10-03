from typing import Any

import torch

from gromo.config.loader import load_config
from gromo.modules.growing_module import GrowingModule, MergeGrowingModule
from gromo.utils.utils import get_correct_device


class GrowingContainer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
    ) -> None:
        super(GrowingContainer, self).__init__()
        self._config_data, _ = load_config()
        self.device = get_correct_device(self, device)

        self.in_features = in_features
        self.out_features = out_features

        self._growing_layers: list[
            "GrowingModule | MergeGrowingModule | GrowingContainer"
        ] = list()
        self.currently_updated_layer_index = None

    def set_growing_layers(self) -> None:
        """
        Reference all growable layers of the model in the _growing_layers private attribute. This method should be implemented
        in the child class and called in the __init__ method.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        raise NotImplementedError

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extended forward pass through the network"""
        raise NotImplementedError

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """Get the first order improvement of the current update."""
        raise NotImplementedError

    def init_computation(self) -> None:
        """Initialize statistics computations for growth procedure"""
        for layer in self._growing_layers:
            layer.init_computation()

    def update_computation(self) -> None:
        """Update statistics computations for growth procedure"""
        for layer in self._growing_layers:
            layer.update_computation()

    def reset_computation(self) -> None:
        """Reset statistics computations for growth procedure"""
        for layer in self._growing_layers:
            layer.reset_computation()

    def compute_optimal_delta(
        self,
        update: bool = True,
        force_pseudo_inverse: bool = False,
    ) -> None:
        """Compute optimal delta for growth procedure

        Parameters
        ----------
        update : bool, optional
            update the optimal delta layer attribute and the first order decrease, by default True
        force_pseudo_inverse : bool, optional
            use the pseudo-inverse to compute the optimal delta even if the
            matrix is invertible, by default False
        """
        for layer in self._growing_layers:
            layer.compute_optimal_delta(
                update=update,
                force_pseudo_inverse=force_pseudo_inverse,
            )

    def compute_optimal_updates(self, *args, **kwargs) -> None:
        """Compute optimal updates for growth procedure"""
        for layer in self._growing_layers:
            if isinstance(layer, (GrowingModule, GrowingContainer)):
                layer.compute_optimal_updates(*args, **kwargs)

    def select_best_update(self) -> None:
        """Select the best update for growth procedure"""
        first_order_improvements: list[torch.Tensor] = [
            layer.first_order_improvement for layer in self._growing_layers
        ]
        best_layer_idx = torch.argmax(torch.stack(first_order_improvements))
        self.currently_updated_layer_index = best_layer_idx

        for idx, layer in enumerate(self._growing_layers):
            if idx != best_layer_idx:
                layer.delete_update()

    def select_update(self, layer_index: int, verbose: bool = False) -> int:
        self.currently_updated_layer_index = layer_index
        for i, layer in enumerate(self._growing_layers):
            if verbose:
                print(f"Layer {i} update: {layer.first_order_improvement}")
                print(
                    f"Layer {i} parameter improvement: {layer.parameter_update_decrease}"
                )
                print(f"Layer {i} eigenvalues extension: {layer.eigenvalues_extension}")
            if i != layer_index:
                layer.delete_update()
                if verbose:
                    print(f"Deleting layer {i}")
        return self.currently_updated_layer_index

    @property
    def currently_updated_layer(
        self,
    ) -> "GrowingModule | MergeGrowingModule | GrowingContainer":
        """Get the currently updated layer"""
        assert self.currently_updated_layer_index is not None, "No layer to update"
        return self._growing_layers[self.currently_updated_layer_index]

    def apply_change(self) -> None:
        """Apply changes to the model"""
        assert self.currently_updated_layer is not None, "No layer to update"
        self.currently_updated_layer.apply_change()
        self.currently_updated_layer.delete_update()
        self.currently_updated_layer_index = None

    def number_of_parameters(self) -> int:
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def update_size(self) -> None:
        """Update sizes of the individual modules"""
        for layer in self._growing_layers:
            if isinstance(layer, (MergeGrowingModule, GrowingContainer)):
                layer.update_size()

    def weights_statistics(self) -> dict[str, Any]:
        """Get the statistics of the weights in the growing layers.
        Due to the recursive nature of the containers, the returned dictionary
        contains nested dictionaries for each layer.
        """
        stats = {}
        for module in self.modules():
            if isinstance(module, GrowingModule):
                stats[module.name] = module.weights_statistics()
        return stats
