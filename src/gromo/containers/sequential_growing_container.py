from typing import Any

import torch

from gromo.containers.growing_container import GrowingContainer
from gromo.modules.growing_module import GrowingModule


class SequentialGrowingContainer(GrowingContainer):
    """Container for sequential model architectures

    Parameters
    ----------
    in_features : int
        input features, to be interpreted based on current needs
    out_features : int
        output features, to be interpreted based on current needs
    device : torch.device | str | None, optional
        default device, by default None
    """

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
        index : int | None, optional
            If scheduling_method is "sequential", this index specifies which layer to
            grow next.

        Raises
        ------
        IndexError
            if index is out of bounds for the number og growing_layers in the module
        ValueError
            if argument scheduling_method is not 'sequential' or 'all'
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
            self.layer_to_grow_index = -1
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

    def number_of_neurons_to_add(self, **kwargs: Any) -> int:
        """Get the number of neurons to add in the next growth step."""
        if self.layer_to_grow_index < 0:
            raise RuntimeError(
                "number_of_neurons_to_add is only supported when a single layer is being "
                "grown (e.g. with scheduling_method='sequential'). A negative "
                "layer_to_grow_index usually indicates that multiple layers are being "
                "grown at once, which is not supported by this method."
            )
        return self._growable_layers[self.layer_to_grow_index].number_of_neurons_to_add(
            **kwargs
        )

    def update_information(self) -> dict[str, Any]:
        """Update information for all growing layers including first order improvement

        Returns
        -------
        dict[str, Any]
            information dictionary
        """
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

    def missing_neurons(self) -> int:
        """Get the number of missing neurons to reach the target size."""
        return self.currently_updated_layer.missing_neurons()

    def complete_growth(self, extension_kwargs: dict) -> None:
        """Complete the growth to the target size."""
        for layer in self._growable_layers:
            layer.complete_growth(extension_kwargs=extension_kwargs)
