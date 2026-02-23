"""
Growing Batch Normalization module for extending batch normalization layers dynamically.
"""

from typing import Callable

import torch
import torch.nn as nn


class GrowingBatchNorm(nn.modules.batchnorm._BatchNorm):
    """
    Base class for growing batch normalization layers.

    This class provides the common functionality for growing batch normalization
    layers by adding new parameters with default or custom values.

    Parameters
    ----------
    num_features : int
        Number of features (channels) in the input
    eps : float, optional
        A value added to the denominator for numerical stability, by default=1e-5
    momentum : float, optional
        The value used for the running_mean and running_var computation, by default=0.1
    affine : bool, optional
        Whether to learn affine parameters (weight and bias), by default=True
    track_running_stats : bool, optional
        Whether to track running statistics, by default=True
    device : torch.device | None, optional
        Device to place the layer on
    dtype : torch.dtype | None, optional
        Data type for the parameters
    name : str, optional
        Name of the layer for debugging, by default="growing_batch_norm"
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        name: str = "growing_batch_norm",
    ):
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
        new_values: torch.Tensor | None,
        default_value_fn: Callable,
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
        new_values : torch.Tensor | None, optional
            Custom values for the new features. If None, uses default_value_fn.
        default_value_fn : Callable
            Function to generate default values:
            fn(additional_features, device, dtype) -> torch.Tensor
        device : torch.device
            Device to place new parameters on
        as_parameter : bool, optional
            Whether to treat as nn.Parameter (True) or buffer (False), by default=True

        Raises
        ------
        ValueError
            if the parameter does not have additional_features as a number of elements
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
                    f"new_{param_name} must have {additional_features} elements, "
                    f"got {new_values.shape[0]}"
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
        new_weights: torch.Tensor | None = None,
        new_biases: torch.Tensor | None = None,
        new_running_mean: torch.Tensor | None = None,
        new_running_var: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Grow the batch normalization layer by adding more features.

        Parameters
        ----------
        additional_features : int
            Number of additional features to add
        new_weights : torch.Tensor | None, optional
            Custom weights for the new features. If None, defaults to ones.
        new_biases : torch.Tensor | None, optional
            Custom biases for the new features. If None, defaults to zeros.
        new_running_mean : torch.Tensor | None, optional
            Custom running mean for new features. If None, defaults to zeros.
        new_running_var : torch.Tensor | None, optional
            Custom running variance for new features. If None, defaults to ones.
        device : torch.device | None, optional
            Device to place new parameters on. If None, uses current device.

        Raises
        ------
        ValueError
            if the additional_features argument is not positive
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
            assert isinstance(
                self.running_mean, torch.Tensor
            ), "running_mean is not initialized while track_running_stats is True"
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


class GrowingLayerNorm(nn.LayerNorm):
    """
    Growable LayerNorm implemented by subclassing nn.LayerNorm and using super().__init__.

    Growth extends the last dimension of normalized_shape:
        (..., D) -> (..., D + additional_last_dim)

    Note: LayerNorm has no running stats.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        name: str = "growing_layer_norm",
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.name = name

    def _extend_parameter(
        self,
        param_name: str,
        additional_last_dim: int,
        new_values: torch.Tensor | None,
        default_value_fn: Callable,
        device: torch.device,
        as_parameter: bool = True,
    ) -> None:
        current_param = getattr(self, param_name, None)
        if current_param is None:
            return

        required_shape = tuple(current_param.shape[:-1]) + (additional_last_dim,)

        if new_values is None:
            new_values = default_value_fn(
                required_shape, device=device, dtype=current_param.dtype
            )
        else:
            if tuple(new_values.shape) != required_shape:
                raise ValueError(
                    f"new_{param_name} must have shape {required_shape}, got {tuple(new_values.shape)}"
                )
            if new_values.device != device:
                new_values = new_values.to(device)

        assert new_values is not None
        with torch.no_grad():
            extended_param = torch.cat([current_param.detach(), new_values], dim=-1)

        if as_parameter:
            setattr(self, param_name, nn.Parameter(extended_param))
        else:
            self.register_buffer(param_name, extended_param)

    def grow(
        self,
        additional_last_dim: int,
        new_weights: torch.Tensor | None = None,
        new_biases: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        if additional_last_dim <= 0:
            raise ValueError(
                f"additional_last_dim must be positive, got {additional_last_dim}"
            )

        # Update normalized_shape metadata used by forward
        old = (
            (self.normalized_shape,)
            if isinstance(self.normalized_shape, int)
            else tuple(int(v) for v in self.normalized_shape)
        )
        self.normalized_shape = tuple(old[:-1]) + (old[-1] + additional_last_dim,)

        # Extend affine parameters if enabled
        if getattr(self, "elementwise_affine", False):
            if device is None:
                assert isinstance(self.weight, torch.Tensor)
                device = self.weight.device

            self._extend_parameter(
                "weight",
                additional_last_dim,
                new_weights,
                torch.ones,
                device,
                as_parameter=True,
            )
            if getattr(self, "bias", None) is not None:
                self._extend_parameter(
                    "bias",
                    additional_last_dim,
                    new_biases,
                    torch.zeros,
                    device,
                    as_parameter=True,
                )

    def get_growth_info(self) -> dict:
        return {
            "normalized_shape": tuple(self.normalized_shape),
            "name": self.name,
        }

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, name={self.name}"


class GrowingGroupNorm(nn.GroupNorm):
    """
    Growable GroupNorm implemented by subclassing nn.GroupNorm and using super().__init__.

    Growth extends num_channels:
        C -> C + additional_channels

    Note: GroupNorm has no running stats.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        name: str = "growing_group_norm",
    ):
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )
        self.name = name

    def _extend_parameter(
        self,
        param_name: str,
        additional_channels: int,
        new_values: torch.Tensor | None,
        default_value_fn: Callable,
        device: torch.device,
        as_parameter: bool = True,
    ) -> None:
        current_param = getattr(self, param_name, None)
        if current_param is None:
            return

        if new_values is None:
            new_values = default_value_fn(
                additional_channels, device=device, dtype=current_param.dtype
            )
        else:
            if new_values.ndim != 1 or new_values.shape[0] != additional_channels:
                raise ValueError(
                    f"new_{param_name} must have shape ({additional_channels},), got {tuple(new_values.shape)}"
                )
            if new_values.device != device:
                new_values = new_values.to(device)

        assert new_values is not None
        with torch.no_grad():
            extended_param = torch.cat([current_param.detach(), new_values], dim=0)

        if as_parameter:
            setattr(self, param_name, nn.Parameter(extended_param))
        else:
            self.register_buffer(param_name, extended_param)

    def grow(
        self,
        additional_channels: int,
        new_weights: torch.Tensor | None = None,
        new_biases: torch.Tensor | None = None,
        device: torch.device | None = None,
        new_num_groups: int | None = None,
    ) -> None:
        if additional_channels <= 0:
            raise ValueError(
                f"additional_channels must be positive, got {additional_channels}"
            )

        if new_num_groups is not None:
            self.num_groups = int(new_num_groups)

        # Update num_channels metadata used by forward
        self.num_channels += int(additional_channels)

        if self.num_channels % self.num_groups != 0:
            raise ValueError(
                f"After growth: num_channels ({self.num_channels}) must be divisible by "
                f"num_groups ({self.num_groups})."
            )

        if getattr(self, "affine", False):
            if device is None:
                assert isinstance(self.weight, torch.Tensor)
                device = self.weight.device

            self._extend_parameter(
                "weight",
                additional_channels,
                new_weights,
                torch.ones,
                device,
                as_parameter=True,
            )
            self._extend_parameter(
                "bias",
                additional_channels,
                new_biases,
                torch.zeros,
                device,
                as_parameter=True,
            )

    def get_growth_info(self) -> dict:
        return {
            "num_channels": self.num_channels,
            "num_groups": self.num_groups,
            "name": self.name,
        }

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, name={self.name}"
