from typing import Any

import torch

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.growing_module import GrowingModule


class SequentialGrowingContainer(GrowingContainer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
    ) -> None:
        super(SequentialGrowingContainer, self).__init__(
            in_features, out_features, device
        )
        assert all(
            isinstance(layer, (GrowingModule, GrowingContainer))
            for layer in self._growing_layers
        ), "All layers in _growing_layers must be of type GrowingModule"
        self._growing_layers: list[GrowingModule | GrowingContainer]
        self._growable_layers: list[GrowingModule | GrowingContainer] = []
        self.layer_to_grow_index = -1  # index inside _growable_layers

    def set_growing_layers(
        self, scheduling_method: str = "all", index: int | None = None
    ) -> None:
        """
        Update the list of growable layers.

        This method should be called after a growth step is performed.

        Parameters
        ----------
        scheduling_method : str
            Method to use for scheduling the growth. Options are "sequential" and "all".
            "sequential": only the next layer in the _growable_layers list is added to the
            growing_layers list.
            "all": all layers in the _growable_layers list are added to the
            _growing_layers list.
        index : int, optional
            If scheduling_method is "sequential", this index specifies which layer to
            grow next.
        """
        if isinstance(index, int):
            if index < 0 or index >= len(self._growable_layers):
                raise IndexError(
                    f"Index {index} is out of bounds for _growable_layers with length "
                    f"{len(self._growable_layers)}."
                )
            else:
                self.layer_to_grow_index = index
                self._growing_layers = [self._growable_layers[self.layer_to_grow_index]]
        elif scheduling_method == "sequential":
            self.layer_to_grow_index = (self.layer_to_grow_index + 1) % len(
                self._growable_layers
            )
            self._growing_layers = [self._growable_layers[self.layer_to_grow_index]]
        elif scheduling_method == "all":
            self._growing_layers = (  # pyright: ignore[reportIncompatibleVariableOverride]
                self._growable_layers
            )
            # The above ignore is needed because we do not allow MergeGrowingModule in
            # SequentialGrowingContainer, but it is allowed in GrowingContainer.
        else:
            raise ValueError(
                f"Invalid scheduling method: {scheduling_method}. Supported methods are "
                f"'sequential' and 'all'."
            )

    def number_of_neurons_to_add(self, **kwargs) -> int:
        """Get the number of neurons to add in the next growth step."""
        raise NotImplementedError

    def update_information(self) -> dict[str, Any]:
        """Get information about the current state of the growing layers."""
        information = {}
        for i, layer in enumerate(self._growing_layers):
            assert isinstance(
                layer.parameter_update_decrease, torch.Tensor
            ), "parameter_update_decrease should be a tensor"
            layer_information = {
                "update_value": layer.first_order_improvement.item(),
                "parameter_improvement": layer.parameter_update_decrease.item(),
                "eigenvalues_extension": layer.eigenvalues_extension,
            }
            information[i] = layer_information
        return information
