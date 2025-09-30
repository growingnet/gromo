from typing import Any, Callable
from warnings import warn

import numpy as np
import torch

from gromo.utils.utils import global_device


class TensorStatistic:
    """
    Class to store a tensor statistic and update it with a given function.
    A tensor statistic is a tensor that is an average of a given tensor over
    multiple samples. It is typically computed by batch.

    When computing the new source data, the tensor statistic should be
    informed that it is not updated. Then The update function should be called
    to update the tensor statistic.

    Example:
        We want to compute the average of a set of tensors of shape (2, 3) in data
        loader `data_loader`. We can use the following code:

            ```
            python
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
        """
        Initialise the tensor information.

        Parameters
        ----------
        shape: tuple[int, ...] | None
            shape of the tensor to compute, if None use the shape of the first update
        update_function: Callable[[Any], tuple[torch.Tensor, int]]
            function to update the tensor
        name: str
            used for debugging
        """
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
    def update(self, **kwargs):
        if self.updated is False:
            update, nb_sample = self._update_function(**kwargs)
            if self._shape is None:
                self._shape = update.size()
            assert self._shape == update.size(), (
                f"The update tensor has a different size than the tensor statistic {self.name}"
                f" {update.size()=}, {self._shape=}"
            )
            if self._tensor is None:
                self._tensor = update
            else:
                self._tensor += update
            self.samples += nb_sample
            self.updated = True

    def init(self):
        pass

    def reset(self):
        self._tensor = None
        self.samples = 0

    def __call__(self):
        if self.samples == 0:
            raise ValueError("The tensor statistic has not been computed.")
        else:
            assert (
                self._tensor is not None
            ), f"If the number of samples is not zero the tensor should not be None."
            return self._tensor / self.samples


class TensorStatisticWithError(TensorStatistic):
    """
    Extends TensorStatistic to compute an estimation of the quadratic error of the current estimate to the true expectation.
    This is done by computing the trace of the covariance matrix of the random variable averaged on a batch.
    The trace is computed incrementally using a stopping criterion based on a relative precision.

    Note that the precision of the trace computation can be controlled by the user, and the true precision of the trace
    will not be guaranteed to be below this value, indeed if trace_precision is set to eps, the expected relative precision
    on the trace computation will be of order sqrt(eps).

    Example:
        We want to compute the average of a set of tensors of shape (2, 3) in data
        loader `data_loader`. We can use the following code:

            ```
            python
            tensor_statistic = TensorStatistic(
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
    """

    def __init__(
        self,
        shape: tuple[int, ...] | None,
        update_function: (
            Callable[[Any], tuple[torch.Tensor, int, torch.Tensor]]
            | Callable[[], tuple[torch.Tensor, int, torch.Tensor]]
        ),
        device: torch.device | str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the tensor information.

        Parameters
        ----------
        shape: tuple[int, ...] | None
            shape of the tensor to compute, if None use the shape of the first update
        update_function: Callable[[Any], tuple[torch.Tensor, int, torch.Tensor]]
            function to update the tensor and compute the batch covariance
        name: str
            used for debugging
        **kwargs: Any
            additional arguments to pass to the update function, such as relative precision for the trace computation
        """
        super().__init__(shape, update_function, device, name)
        self._square_norm_sum = 0
        self.trace_precision = kwargs.get(
            "trace_precision", 1e-3
        )  # relative precision stopping criterion for the trace computation
        self._compute_trace = True  # whether to continue computing the trace covariance
        self._batches = 0
        self._trace = None  # trace of the covariance matrix (of the random variable obtain when averaging on a batch)

    def reset(self):
        super().reset()
        self._square_norm_sum = 0
        self._compute_trace = True
        self._batches = 0
        self._trace = None

    def error(self):
        """
        Returns an estimation of the quadratic error of the current estimate to the true expectation.

        Returns
        -------
        float
            estimation of the quadratic error of the current estimate to the true expectation

        """
        assert self._trace is not None, "The trace has not been computed yet."
        assert self._batches > 1, "The trace has not been computed yet."
        if self._compute_trace:
            warn(
                f"The desired trace precision has not been reached for the tensor statistic {self.name}. The error estimate may be inaccurate."
            )
        return self._trace / self._batches

    def update(self, **kwargs):
        if self.updated is False:
            update, nb_sample = self._update_function(**kwargs)
            if self._shape is None:
                self._shape = update.size()
            assert self._shape == update.size(), (
                f"The update tensor has a different size than the tensor statistic {self.name}"
                f" {update.size()=}, {self._shape=}"
            )
            if self._tensor is None:
                self._tensor = update
            else:
                self._tensor += update

            self._batches += 1
            self.samples += nb_sample
            self.updated = True

            if self._compute_trace:
                batch_sum_norm = (update / nb_sample).norm().pow(2).item()
                self._square_norm_sum += batch_sum_norm
                mu_n_norm = torch.norm(self._tensor / (self.samples)).item()
                trace_covariance = (self._square_norm_sum) / (
                    self._batches
                ) - mu_n_norm**2
                if self._trace is not None:
                    delta_trace_covariance = trace_covariance - self._trace
                    if abs(delta_trace_covariance) < self.trace_precision * abs(
                        trace_covariance
                    ):
                        self._compute_trace = False
                self._trace = trace_covariance
