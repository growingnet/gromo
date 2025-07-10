"""
Growing Batch Normalization module for extending batch norm layers dynamically.
"""

from typing import Optional

import torch
import torch.nn as nn


class GrowingBatchNorm:
    """
    Base class for growing batch normalization layers.

    This class provides the common functionality for growing batch normalization
    layers by adding new parameters with default or custom values.
    """

    def __init__(self, name: str = "growing_batch_norm"):
        """
        Initialize the base growing batch norm functionality.

        Parameters
        ----------
        name : str
            Name of the layer for debugging
        """
        self.name = name
        self.original_num_features = self.num_features

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

        # Store old num_features
        old_num_features = self.num_features
        self.num_features = old_num_features + additional_features

        # Determine device
        if device is None:
            if self.weight is not None:
                device = self.weight.device
            elif self.running_mean is not None:
                device = self.running_mean.device
            else:
                device = torch.device("cpu")
                return  # No parameters to extend

        # Extend weight parameter if affine=True
        if getattr(self, "affine", False) and self.weight is not None:
            if new_weights is None:
                new_weights = torch.ones(
                    additional_features, device=device, dtype=self.weight.dtype
                )
            elif new_weights.shape[0] != additional_features:
                raise ValueError(
                    f"new_weights must have {additional_features} elements, got {new_weights.shape[0]}"
                )

            # Concatenate old and new weights
            extended_weight = torch.cat([self.weight.data, new_weights.to(device)])
            self.weight = nn.Parameter(extended_weight)

        # Extend bias parameter if affine=True
        if getattr(self, "affine", False) and self.bias is not None:
            if new_biases is None:
                new_biases = torch.zeros(
                    additional_features, device=device, dtype=self.bias.dtype
                )
            elif new_biases.shape[0] != additional_features:
                raise ValueError(
                    f"new_biases must have {additional_features} elements, got {new_biases.shape[0]}"
                )

            # Concatenate old and new biases
            extended_bias = torch.cat([self.bias.data, new_biases.to(device)])
            self.bias = nn.Parameter(extended_bias)

        # Extend running statistics if track_running_stats=True
        if getattr(self, "track_running_stats", False):
            # Extend running_mean
            if self.running_mean is not None:
                if new_running_mean is None:
                    new_running_mean = torch.zeros(
                        additional_features, device=device, dtype=self.running_mean.dtype
                    )
                elif new_running_mean.shape[0] != additional_features:
                    raise ValueError(
                        f"new_running_mean must have {additional_features} elements, got {new_running_mean.shape[0]}"
                    )

                extended_running_mean = torch.cat(
                    [self.running_mean, new_running_mean.to(device)]
                )
                self.running_mean = extended_running_mean

            # Extend running_var
            if self.running_var is not None:
                if new_running_var is None:
                    new_running_var = torch.ones(
                        additional_features, device=device, dtype=self.running_var.dtype
                    )
                elif new_running_var.shape[0] != additional_features:
                    raise ValueError(
                        f"new_running_var must have {additional_features} elements, got {new_running_var.shape[0]}"
                    )

                extended_running_var = torch.cat(
                    [self.running_var, new_running_var.to(device)]
                )
                self.running_var = extended_running_var

            # Extend num_batches_tracked (this is just a counter, so no need to extend)
            # It will continue tracking from where it left off

    def get_growth_info(self) -> dict:
        """
        Get information about the growth of this layer.

        Returns
        -------
        dict
            Dictionary containing growth information
        """
        return {
            "original_num_features": self.original_num_features,
            "current_num_features": self.num_features,
            "total_growth": self.num_features - self.original_num_features,
            "growth_ratio": self.num_features / self.original_num_features,
            "name": self.name,
        }


class GrowingBatchNorm2d(nn.BatchNorm2d, GrowingBatchNorm):
    """
    A batch normalization layer that can grow in the number of features.

    This class extends torch.nn.BatchNorm2d to allow dynamic growth of the
    number of features by adding new parameters with default or custom values.
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
        name: str = "growing_batch_norm_2d",
    ):
        """
        Initialize the GrowingBatchNorm2d layer.

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
        name : str, default="growing_batch_norm_2d"
            Name of the layer for debugging
        """
        nn.BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        GrowingBatchNorm.__init__(self, name=name)

    def extra_repr(self) -> str:
        """
        Extra representation string for the layer.
        """
        return (
            f"{self.num_features}, eps={self.eps:.1e}, momentum={self.momentum:.1e}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}, name={self.name}"
        )


class GrowingBatchNorm1d(nn.BatchNorm1d, GrowingBatchNorm):
    """
    A 1D batch normalization layer that can grow in the number of features.

    Similar to GrowingBatchNorm2d but for 1D inputs.
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
        name: str = "growing_batch_norm_1d",
    ):
        """
        Initialize the GrowingBatchNorm1d layer.

        Parameters are the same as GrowingBatchNorm2d.
        """
        nn.BatchNorm1d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        GrowingBatchNorm.__init__(self, name=name)

    def extra_repr(self) -> str:
        """
        Extra representation string for the layer.
        """
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}, name={self.name}"
        )
