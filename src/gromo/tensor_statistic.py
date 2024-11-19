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
    informed that it is not updated. Then The update function should be called
    to update the tensor statistic.

    Example:
        We want to compute the average of a set of tensors of shape (2, 3) in data
        loader `data_loader`. We can use the following code:

            ```python
            data = torch.zeros(2, 3)
            tensor_statistic = TensorStatistic(
                shape=(2, 3),
                update_function=lambda: (data.sum(dim=0), data.size(0)),
                name="Average",
            )
            for data_batch in data_loader:
                data = data_batch
                tensor_statistic.updated = False
                tensor_statistic.update()

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

    def update(self, **kwargs):
        assert (
            not self._shape or self._tensor is not None
        ), f"The tensor statistic {self.name} has not been initialized."
        if self.updated is False:
            update, nb_sample = self._update_function(**kwargs)
            if self._tensor is not None:
                assert update.size() == self._tensor.size(), (
                    f"The update tensor has a different size than the tensor statistic {self.name}"
                    f" {update.size()=}, {self._tensor.size()=}"
                )
                self._tensor += update
            else:
                assert (
                    self._shape is None
                ), "If self._shape is not None, self_.tensor should be initialised in init"
                self._tensor = update
            self.samples += nb_sample
            self.updated = True

    def init(self):
        if self._shape is None:
            self._tensor = None
        else:
            self._tensor = torch.zeros(self._shape, device=self.device)
        self.samples = 0

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
