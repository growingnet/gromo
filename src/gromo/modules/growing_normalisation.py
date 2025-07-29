"""
Growing Batch Normalization module for extending batch normalization layers dynamically.
"""

from typing import Optional

import torch
import torch.nn as nn


class GrowingBatchNorm(nn.modules.batchnorm._BatchNorm):
    """
    Base class for growing batch normalization layers.

    This class provides the common functionality for growing batch normalization
    layers by adding new parameters with default or custom values.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        name: str = "growing_batch_norm",
    ):
        """
        Initialize the base growing batch norm functionality.

        Parameters
        ----------
        num_features : int
            Number of features (channels) in the input
        eps : float, default=1e-5
            A value added to the denominator for numerical stability
        momentum : float, default=0.1
            The value used for the running_mean and running_var computation
        affine : bool, default=True
            Whether to learn affine parameters (weight and bias)
        track_running_stats : bool, default=True
            Whether to track running statistics
        device : torch.device, optional
            Device to place the layer on
        dtype : torch.dtype, optional
            Data type for the parameters
        name : str, default="growing_batch_norm"
            Name of the layer for debugging
        """
        super(GrowingBatchNorm, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self.name = name

    def _extend_parameter(
        self,
        param_name: str,
        additional_features: int,
        new_values: Optional[torch.Tensor],
        default_value_fn,
        device: torch.device,
        as_parameter: bool = True,
    ) -> None:
        """
        Helper method to extend a parameter or buffer.

        Parameters
        ----------
        param_name : str
            Name of the parameter/buffer to extend
        additional_features : int
            Number of additional features to add
        new_values : torch.Tensor, optional
            Custom values for the new features. If None, uses default_value_fn.
        default_value_fn : callable
            Function to generate default values: fn(additional_features, device, dtype) -> torch.Tensor
        device : torch.device
            Device to place new parameters on
        as_parameter : bool, default=True
            Whether to treat as nn.Parameter (True) or buffer (False)
        """
        current_param = getattr(self, param_name, None)
        if current_param is None:
            return

        if new_values is None:
            new_values = default_value_fn(
                additional_features, device=device, dtype=current_param.dtype
            )
        else:
            if new_values.shape[0] != additional_features:
                raise ValueError(
                    f"new_{param_name} must have {additional_features} elements, got {new_values.shape[0]}"
                )
            # Ensure new_values is on the correct device
            if new_values.device != device:
                new_values = new_values.to(device)

        # Concatenate old and new values
        assert new_values is not None  # Type hint for mypy
        with torch.no_grad():
            extended_param = torch.cat([current_param.detach(), new_values])

        if as_parameter:
            setattr(self, param_name, nn.Parameter(extended_param))
        else:
            self.register_buffer(param_name, extended_param)

    def grow(
        self,
        additional_features: int,
        new_weights: Optional[torch.Tensor] = None,
        new_biases: Optional[torch.Tensor] = None,
        new_running_mean: Optional[torch.Tensor] = None,
        new_running_var: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Grow the batch normalization layer by adding more features.

        Parameters
        ----------
        additional_features : int
            Number of additional features to add
        new_weights : torch.Tensor, optional
            Custom weights for the new features. If None, defaults to ones.
        new_biases : torch.Tensor, optional
            Custom biases for the new features. If None, defaults to zeros.
        new_running_mean : torch.Tensor, optional
            Custom running mean for new features. If None, defaults to zeros.
        new_running_var : torch.Tensor, optional
            Custom running variance for new features. If None, defaults to ones.
        device : torch.device, optional
            Device to place new parameters on. If None, uses current device.
        """
        if additional_features <= 0:
            raise ValueError(
                f"additional_features must be positive, got {additional_features}"
            )

        # Update num_features
        self.num_features += additional_features

        # Extend affine parameters if enabled
        if getattr(self, "affine", False):
            device = self.weight.device
            self._extend_parameter(
                "weight",
                additional_features,
                new_weights,
                torch.ones,
                device,
                as_parameter=True,
            )
            self._extend_parameter(
                "bias",
                additional_features,
                new_biases,
                torch.zeros,
                device,
                as_parameter=True,
            )

        # Extend running statistics if enabled
        if getattr(self, "track_running_stats", False):
            device = self.running_mean.device
            self._extend_parameter(
                "running_mean",
                additional_features,
                new_running_mean,
                torch.zeros,
                device,
                as_parameter=False,
            )
            self._extend_parameter(
                "running_var",
                additional_features,
                new_running_var,
                torch.ones,
                device,
                as_parameter=False,
            )

        # Note: num_batches_tracked is just a counter, so no need to extend

    def get_growth_info(self) -> dict:
        """
        Get information about the growth of this layer.

        Returns
        -------
        dict
            Dictionary containing growth information
        """
        return {
            "num_features": self.num_features,
            "name": self.name,
        }

    def extra_repr(self) -> str:
        """
        Extra representation string for the layer.
        """
        return f"{super().extra_repr()}, name={self.name}"


class GrowingBatchNorm2d(GrowingBatchNorm, nn.BatchNorm2d):
    """
    A batch normalization layer that can grow in the number of features.

    This class extends torch.nn.BatchNorm2d to allow dynamic growth of the
    number of features by adding new parameters with default or custom values.
    """


class GrowingBatchNorm1d(GrowingBatchNorm, nn.BatchNorm1d):
    """
    A 1D batch normalization layer that can grow in the number of features.

    Similar to GrowingBatchNorm2d but for 1D inputs.
    """
