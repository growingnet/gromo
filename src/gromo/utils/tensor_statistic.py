from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch

from gromo.utils.kfac import KFACVariant
from gromo.utils.utils import global_device, sum_positions


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
    device : torch.device | str | None
        default device, by default None
    name: str | None
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
            i >= 0 and isinstance(i, (int, np.int64))
            for i in shape  # type: ignore
        ), f"The shape must be a tuple of positive integers. {type(shape)}, {shape}"
        self._shape = shape
        self._update_function = update_function
        self.name = name if name is not None else "TensorStatistic"
        self._tensor: torch.Tensor | None = None
        self.samples = 0
        self.updated = True
        self.device = device if device else global_device()
        # dtype of the running sum (None: the dtype of the updates), and the dtype of
        # the updates, in which the average is handed back
        self._accumulation_dtype: torch.dtype | None = None
        self._update_dtype: torch.dtype | None = None

    def __str__(self):
        return f"{self.name} tensor of shape {self._shape} with {self.samples} samples"

    @property
    def dtype(self) -> torch.dtype | None:
        """Dtype the statistic is accumulated in, None before the first update.

        Without an accumulation dtype (see `init`), it is the dtype of the updates,
        which bounds the precision of the statistic, even when the averaged tensor
        is later cast to a more precise one. With one, the running sum is kept in
        that dtype and the average is rounded once to the dtype of the updates.

        Returns
        -------
        torch.dtype | None
            dtype of the accumulated tensor
        """
        return None if self._tensor is None else self._tensor.dtype

    @torch.no_grad()
    def update(
        self, count_samples: bool = True, **kwargs: Any
    ) -> tuple[torch.Tensor, int] | None:
        """Update tensor based on update_function

        Parameters
        ----------
        count_samples : bool
            if False, accumulate the update tensor without increasing the
            sample counter. Used when several updates are computed from the
            same samples (e.g. one backward pass per class for the true
            Fisher) and should be averaged as a single-pass contribution.
        **kwargs : Any

        Returns
        -------
        tuple[torch.Tensor, int] | None
            the update tensor, number of samples used to compute the update
        """
        if self.updated is False:
            update, nb_sample = self._update_function(**kwargs)  # type: ignore
            if not count_samples:
                nb_sample = 0
            assert (self._shape is None or self._shape == update.size()) and (
                self._tensor is None or self._tensor.size() == update.size()
            ), (
                f"The update tensor has a different size than the tensor statistic "
                f"{self.name} : {self._shape=}, {update.size()=}, "
                f"{None if self._tensor is None else self._tensor.size()=}"
            )
            if self._tensor is None:
                self._update_dtype = update.dtype
                self._tensor = (
                    update
                    if self._accumulation_dtype is None
                    else update.to(self._accumulation_dtype)
                )
            else:
                self._tensor += update
            self.samples += nb_sample
            self.updated = True
            return update, nb_sample

    def init(self, accumulation_dtype: torch.dtype | None = None) -> None:
        """Reset the tensor and choose the dtype of the running sum

        Parameters
        ----------
        accumulation_dtype : torch.dtype | None
            dtype in which the updates are summed, e.g. ``torch.float64`` so that
            the rounding of a long accumulation does not degrade the statistic.
            The average is still returned in the dtype of the updates. None sums
            in the dtype of the updates.
        """
        self.reset()
        self._accumulation_dtype = accumulation_dtype

    def reset(self):
        """Reset the tensor"""
        self._tensor = None
        self.samples = 0
        self._update_dtype = None

    def __call__(self) -> torch.Tensor:
        """Get the average of the precomputed tensor over the number of samples

        Returns
        -------
        torch.Tensor
            averaged tensor, in the dtype of the updates

        Raises
        ------
        ValueError
            if the tensor has not been computed
        """
        if self.samples == 0:
            raise ValueError("The tensor statistic has not been computed.")
        else:
            assert self._tensor is not None, (
                "If the number of samples is not zero the tensor should not be None."
            )
            average = self._tensor / self.samples
            # _update_dtype is set with the first update, so whenever _tensor is
            if self._accumulation_dtype is not None and self._update_dtype is not None:
                average = average.to(self._update_dtype)
            return average


class KroneckerFactorStatistic(TensorStatistic):
    """
    Kronecker factor A or B of a K-FAC variant, accumulated over batches.

    It replaces the `TensorStatistic` of the input second moment S (input side) or of
    the gradient second moment E (gradient side) of a layer. A batch has N
    sequences; sequence n has R_n valid positions, given by the position mask. At
    position m, a_{n,m} is the input of the layer (bias column included) and g_{n,m}
    the gradient of the loss with respect to its pre-activation. With the means over
    the valid positions x̄_n = (sum_m x_{n,m}) / R_n, rho_n = ``variant.rho(R_n)``
    and the weights omega_n of the side (see `KFACVariant`), the factors are::

        A = sum_n omega_n ā_n ā_n^T / D_A,    D_A = sum_n omega_n (count-weighted),
        B = sum_n omega_n ḡ_n ḡ_n^T / N,

    the sums running over every batch accumulated. For K-FAC-reduce with
    count-weighted lengths, with u_n = sum_m a_{n,m} and g_n = sum_m g_{n,m}::

        A = sum_n u_n u_n^T / sum_n R_n^2,    B = sum_n g_n g_n^T / N.

    A sequence without valid position adds nothing to the sums, but counts in N.

    The sums are accumulated about a shift s::

        W_0 = sum_n omega_n,    w_1 = sum_n omega_n (x̄_n - s),
        W_2 = sum_n omega_n (x̄_n - s)(x̄_n - s)^T,

    and the numerator is recovered exactly, for any s, by::

        sum_n omega_n x̄_n x̄_n^T = W_2 + w_1 s^T + s w_1^T + W_0 s s^T.

    With ``assume_centered=False``, s is the weighted mean
    sum omega_n x̄_n / sum omega_n of the first update with a valid position, which
    keeps the accumulators close to zero mean. With ``assume_centered=True``, the
    means x̄_n are assumed to be close to zero mean already, and s = 0. Unlike the
    ``assume_centered`` of scikit-learn, it does not change the factor, which is the
    uncentred one in both cases, only how its sums are accumulated.

    The number of samples `samples` is N, the number of sequences, as for
    `TensorStatistic`.

    Parameters
    ----------
    variant: KFACVariant
        K-FAC variant giving rho_n, the weights and the denominator of A
    side: Literal["input", "gradient"]
        factor computed: A for ``"input"``, B for ``"gradient"``
    positions_function: Callable[[], torch.Tensor]
        function returning the tensor of the current batch, of shape (N, *, d): the
        input of the layer on the input side, the gradient of the loss with respect
        to the pre-activation on the gradient side. Use a bound method of the module
        rather than a closure, so that ``copy.deepcopy`` binds it to the copy.
    assume_centered: bool
        if True, accumulate the sums about s = 0; if False, by default, about the
        weighted mean of the first update with a valid position
    device: torch.device | str | None
        default device, by default None
    name: str | None
        used for debugging, by default None

    Raises
    ------
    ValueError
        if the side is neither ``"input"`` nor ``"gradient"``
    """

    def __init__(
        self,
        variant: KFACVariant,
        side: Literal["input", "gradient"],
        positions_function: Callable[[], torch.Tensor],
        assume_centered: bool = False,
        device: torch.device | str | None = None,
        name: str | None = None,
    ) -> None:
        if side not in ("input", "gradient"):
            raise ValueError(f"The side must be 'input' or 'gradient', got {side!r}.")
        # `update` is overridden: it reads the positions with `_read_positions`.
        super().__init__(
            None,
            update_function=self._read_positions,
            device=device,
            name=name if name is not None else "KroneckerFactorStatistic",
        )
        self.variant = variant
        self.side = side
        self.assume_centered = assume_centered
        self._positions_function = positions_function
        # s, in the dtype of the updates: set by the first update with a valid
        # position unless assume_centered (None means s = 0), cleared by reset
        self._shift: torch.Tensor | None = None
        self._weight_sum: torch.Tensor | None = None  # W_0
        self._first_moment: torch.Tensor | None = None  # w_1

    def __str__(self) -> str:
        return (
            f"{self.name} Kronecker factor ({self.side} side of {self.variant}) "
            f"with {self.samples} samples"
        )

    def _read_positions(self) -> tuple[torch.Tensor, int]:
        """Read the tensor of the current batch.

        Returns
        -------
        torch.Tensor
            tensor of shape (N, *, d) returned by the positions function
        int
            number of sequences N
        """
        positions = self._positions_function()
        return positions, positions.shape[0]

    @torch.no_grad()
    def update(self, count_samples: bool = True, **kwargs: Any) -> None:
        """Add the current batch to the factor.

        Does nothing when `updated` is True. Otherwise, adds the batch to W_0, w_1
        and W_2, adds N to `samples` when ``count_samples`` is True, and sets
        `updated` to True. Unless ``assume_centered``, the first batch with a valid
        position sets the shift.

        Parameters
        ----------
        count_samples : bool
            if False, accumulate the batch without increasing the sample counter
            (see `TensorStatistic.update`)
        **kwargs : Any
            ``position_mask``, the mask of the valid positions, of shape
            ``positions.shape[:-1]`` (see `gromo.utils.utils.sum_positions`), by
            default None (every position is valid). No other keyword is accepted.

        Raises
        ------
        TypeError
            if a keyword other than ``position_mask`` is given
        """
        position_mask = kwargs.pop("position_mask", None)
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.update got unexpected keyword arguments "
                f"{sorted(kwargs)}, only position_mask is accepted."
            )
        if self.updated:
            return
        positions, n_sequences = self._read_positions()
        sums, lengths = sum_positions(positions, position_mask)
        valid = lengths > 0
        sums, lengths = sums[valid], lengths[valid]
        means = sums / lengths.unsqueeze(1)
        rho = self.variant.rho(lengths)
        if self.side == "input":
            weights = self.variant.input_weights(rho)
        else:
            weights = self.variant.gradient_weights(rho)
        if not self.assume_centered and self._shift is None and lengths.numel() > 0:
            self._shift = weights @ means / weights.sum()
        centred = means if self._shift is None else means - self._shift
        weighted = weights.unsqueeze(1) * centred
        self._accumulate(weights.sum(), weighted.sum(dim=0), weighted.T @ centred)
        if count_samples:
            self.samples += n_sequences
        self.updated = True

    def _accumulate(
        self,
        weight_sum: torch.Tensor,
        first_moment: torch.Tensor,
        second_moment: torch.Tensor,
    ) -> None:
        """Add the terms of a batch to W_0, w_1 and W_2.

        Parameters
        ----------
        weight_sum: torch.Tensor
            term of W_0, a scalar
        first_moment: torch.Tensor
            term of w_1, of shape (d,)
        second_moment: torch.Tensor
            term of W_2, of shape (d, d)
        """
        if self._tensor is None:
            self._update_dtype = second_moment.dtype
            dtype = (
                second_moment.dtype
                if self._accumulation_dtype is None
                else self._accumulation_dtype
            )
            self._weight_sum = weight_sum.to(dtype)
            self._first_moment = first_moment.to(dtype)
            self._tensor = second_moment.to(dtype)
        else:
            assert self._weight_sum is not None and self._first_moment is not None
            self._weight_sum += weight_sum
            self._first_moment += first_moment
            self._tensor += second_moment

    def _second_moment(self) -> torch.Tensor:
        """Numerator sum_n omega_n x̄_n x̄_n^T, in the accumulation dtype.

        Returns
        -------
        torch.Tensor
            W_2 + w_1 s^T + s w_1^T + W_0 s s^T, of shape (d, d)
        """
        assert (
            self._tensor is not None
            and self._first_moment is not None
            and self._weight_sum is not None
        )
        if self._shift is None:
            return self._tensor
        shift = self._shift.to(self._tensor.dtype)
        cross = torch.outer(self._first_moment, shift)
        return (
            self._tensor + cross + cross.T + self._weight_sum * torch.outer(shift, shift)
        )

    def __call__(self) -> torch.Tensor:
        """Get the factor A (input side) or B (gradient side).

        Returns
        -------
        torch.Tensor
            the factor, of shape (d, d), in the dtype of the updates

        Raises
        ------
        ValueError
            if no sample has been counted, or, on the input side, if no valid
            position has been seen
        """
        if self.samples == 0:
            raise ValueError(f"The Kronecker factor {self.name} has not been computed.")
        assert self._weight_sum is not None and self._update_dtype is not None
        if self.side == "input":
            denominator = self.variant.input_denominator(self._weight_sum, self.samples)
            if denominator == 0:
                raise ValueError(
                    f"The Kronecker factor {self.name} has no valid position yet."
                )
        else:
            denominator = self.samples
        return (self._second_moment() / denominator).to(self._update_dtype)

    def reset(self) -> None:
        """Reset the accumulators and the shift, keeping the accumulation dtype."""
        super().reset()
        self._shift = None
        self._weight_sum = None
        self._first_moment = None


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
    device : torch.device | str | None
        default device, by default None
    name: str | None
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
    def update(
        self, count_samples: bool = True, **kwargs: Any
    ) -> tuple[torch.Tensor, int] | None:
        """Update tensor based on update_function

        Parameters
        ----------
        count_samples : bool
            if False, accumulate the update tensor without increasing the
            sample counter (see TensorStatistic.update) and without treating
            this call as a new batch for the trace-error estimation below.
        **kwargs : Any

        Returns
        -------
        tuple[torch.Tensor, int] | None
            the update tensor, number of samples used to compute the update
        """
        if self.updated is False:
            update, nb_sample = super().update(count_samples=count_samples, **kwargs)  # type: ignore (we are sure updated is False here)
            assert isinstance(
                self._tensor, torch.Tensor
            )  # self._tensor should not be None here

            if not count_samples:
                return update, nb_sample

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
