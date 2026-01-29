from typing import Any, Callable

import numpy as np
import torch

from gromo.utils.utils import global_device


class TensorStatistic:
    """
    Class to store a tensor statistic and update it with a given function.
    A tensor statistic is a tensor that is an average of a given tensor over
    multiple samples. It is typically computed by batch.

    When computing the new source data, the tensor statistic should be
    informed that it is not updated. Then the update function should be called
    to update the tensor statistic.

    Example:
        We want to compute the average of a set of tensors of shape (2, 3) in data
        loader `data_loader`. We can use the following code:

            ```python
            tensor_statistic = TensorStatistic(
                shape=(2, 3),
                update_function=lambda data: (data.sum(dim=0), data.size(0)),
                name="Average",
            )
            for data_batch in data_loader:
                tensor_statistic.updated = False
                tensor_statistic.update(data_batch)

            print(tensor_statistic())
            ```

    Parameters
    ----------
    shape: tuple[int, ...] | None
        shape of the tensor to compute, if None use the shape of the first update
    update_function: Callable[[Any], tuple[torch.Tensor, int]] | Callable[[], tuple[torch.Tensor, int]]
        function to update the tensor
    device : torch.device | str | None, optional
        default device, by default None
    name: str | None, optional
        used for debugging, by default None
    """

    def __init__(
        self,
        shape: tuple[int, ...] | None,
        update_function: (
            Callable[[Any], tuple[torch.Tensor, int]]
            | Callable[[], tuple[torch.Tensor, int]]
        ),
        device: torch.device | str | None = None,
        name: str | None = None,
    ) -> None:
        assert shape is None or all(
            i >= 0 and isinstance(i, (int, np.int64)) for i in shape  # type: ignore
        ), f"The shape must be a tuple of positive integers. {type(shape)}, {shape}"
        self._shape = shape
        self._update_function = update_function
        self.name = name if name is not None else "TensorStatistic"
        self._tensor: torch.Tensor | None = None
        self.samples = 0
        self.updated = True
        self.device = device if device else global_device()

    def __str__(self):
        return f"{self.name} tensor of shape {self._shape} with {self.samples} samples"

    @torch.no_grad()
    def update(self, **kwargs: Any) -> tuple[torch.Tensor, int] | None:
        """Update tensor based on update_function

        Parameters
        ----------
        **kwargs : Any

        Returns
        -------
        tuple[torch.Tensor, int] | None
            the update tensor, number of samples used to compute the update
        """
        if self.updated is False:
            update, nb_sample = self._update_function(**kwargs)  # type: ignore
            assert (self._shape is None or self._shape == update.size()) and (
                self._tensor is None or self._tensor.size() == update.size()
            ), (
                f"The update tensor has a different size than the tensor statistic "
                f"{self.name} : {self._shape=}, {update.size()=}, "
                f"{None if self._tensor is None else self._tensor.size()=}"
            )
            if self._tensor is None:
                self._tensor = update
            else:
                self._tensor += update
            self.samples += nb_sample
            self.updated = True
            return update, nb_sample

    def init(self):
        """Reset the tensor"""
        self.reset()

    def reset(self):
        """Reset the tensor"""
        self._tensor = None
        self.samples = 0

    def __call__(self) -> torch.Tensor:
        """Get the average of the precomputed tensor over the number of samples

        Returns
        -------
        torch.Tensor
            averaged tensor

        Raises
        ------
        ValueError
            if the tensor has not been computed
        """
        if self.samples == 0:
            raise ValueError("The tensor statistic has not been computed.")
        else:
            assert (
                self._tensor is not None
            ), "If the number of samples is not zero the tensor should not be None."
            return self._tensor / self.samples


class TensorStatiticWithEstimationError(TensorStatistic):
    """
    Extends TensorStatistic with estimated quadratic error.

    Extends TensorStatistic to compute an estimation of the quadratic error of the current
    estimate to the true expectation. This is done by computing the trace of the
    covariance matrix of the random variable averaged on a batch. The trace is computed
    incrementally using a stopping criterion based on a relative precision.

    Note that the precision of the trace computation can be controlled by the user, and
    the true precision of the trace
    will not be guaranteed to be below this value, indeed if trace_precision is set to
    eps, the expected relative precision
    on the trace computation will be of order sqrt(eps).

    Example:
        We want to compute the average of a set of tensors of shape (2, 3) in data
        loader `data_loader`. We can use the following code:

            ```python
            tensor_statistic = TensorStatisticWithEstimationError(
                update_function=lambda data: (data.sum(dim=0), data.size(0)),
                name="Average",
            )
            for data_batch in data_loader:
                tensor_statistic.updated = False
                tensor_statistic.update(data_batch)
                if tensor_statistic.error() < 0.01:
                    break
            print(tensor_statistic())
            print(tensor_statistic.error())
            ```

    Parameters
    ----------
    shape: tuple[int, ...] | None
        shape of the tensor to compute, if None use the shape of the first update
    update_function: Callable[[Any], tuple[torch.Tensor, int]] | Callable[[], tuple[torch.Tensor, int]]
        function to update the tensor and compute the batch covariance
    device : torch.device | str | None, optional
        default device, by default None
    name: str | None, optional
        used for debugging, by default None
    trace_precision: float
        relative precision for the trace computation, default 1e-3
    """

    def __init__(
        self,
        shape: tuple[int, ...] | None,
        update_function: (
            Callable[[Any], tuple[torch.Tensor, int]]
            | Callable[[], tuple[torch.Tensor, int]]
        ),
        device: torch.device | str | None = None,
        name: str | None = None,
        trace_precision: float = 1e-3,
    ) -> None:
        super().__init__(shape, update_function, device, name)
        self._square_norm_sum = 0
        self.trace_precision = trace_precision
        # relative precision stopping criterion for the trace computation
        self._compute_trace = True  # whether to continue computing the trace covariance
        self._batches = 0
        self._trace = None  # trace of the covariance matrix
        # (of the random variable obtain when averaging on a batch)

    def reset(self):
        """Reset the tensor"""
        super().reset()
        self._square_norm_sum = 0
        self._compute_trace = True
        self._batches = 0
        self._trace = None

    def error(self) -> float:
        """
        Returns an estimation of the quadratic error of the current estimate to the true
        expectation.
        If the trace has not been computed accurately enough, NO warning is
        raised and the error estimate may be inaccurate.
        If only one batch has been used to compute the statistic, returns infinity.

        Returns
        -------
        float
            estimation of the quadratic error of the current estimate to the true
            expectation
        """
        assert self._trace is not None, "The trace has not been computed yet."
        if self._batches == 1:
            return float("inf")
        return self._trace / self._batches

    @torch.no_grad()
    def update(self, **kwargs: Any) -> tuple[torch.Tensor, int] | None:
        """Update tensor based on update_function

        Parameters
        ----------
        **kwargs : Any

        Returns
        -------
        tuple[torch.Tensor, int] | None
            the update tensor, number of samples used to compute the update
        """
        if self.updated is False:
            update, nb_sample = super().update(**kwargs)  # type: ignore (we are sure updated is False here)
            assert isinstance(
                self._tensor, torch.Tensor
            )  # self._tensor should not be None here
            self._batches += 1

            if self._compute_trace:
                self._square_norm_sum += update.pow(2).sum().item() / (nb_sample**2)
                mu_n_norm = self._tensor.pow(2).sum().item() / (self.samples**2)
                trace_covariance = (self._square_norm_sum) / (self._batches) - mu_n_norm
                # trace_covariance := Ê[||X||^2] - ||mu_n||^2
                # where X is the random variable obtained by averaging on a batch
                # and mu_n is the current estimate of the expectation
                if self._trace is not None:
                    delta_trace_covariance = trace_covariance - self._trace
                    if abs(delta_trace_covariance) < self.trace_precision * abs(
                        trace_covariance
                    ):
                        self._compute_trace = False
                self._trace = trace_covariance

            return update, nb_sample
