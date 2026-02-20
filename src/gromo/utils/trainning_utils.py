from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.data
from torch import nn


if TYPE_CHECKING:
    from torchmetrics import Metric

    from gromo.containers.growing_container import GrowingContainer


class AverageMeter(object):
    """Computes and stores an average"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter to initial state."""
        self.sum: float = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Updates the average with a new value.

        Parameters
        ----------
        val : float
            The new value to include in the average.
        n : int, optional
            The number of samples that the value represents. Default is 1.
        """
        if val != torch.nan and val != torch.inf:
            self.sum += val * n
            self.count += n

    def __call__(self) -> float:
        """Returns the current average.

        Returns
        -------
        float
            The average of the values seen so far. Returns 0.0 if no values have been
            added.

        Raises
        ------
        TypeError
            If the type of `sum` is not supported for division.
        """
        if self.count == 0:
            return 0.0
            # raise ValueError("AverageMeter has no values to compute average")
        else:
            if isinstance(self.sum, torch.Tensor):
                return (self.sum / self.count).item()  # type: ignore
            elif isinstance(self.sum, (float, int)):
                return self.sum / self.count
            else:
                raise TypeError(f"Unsupported type for sum: {type(self.sum)}")


class DummyMetric(Metric):
    """A dummy metric that always returns 0.0."""

    def update(self, *_, **__):
        """No-op for updating the metric."""
        return

    def compute(self) -> float:
        """Returns the computed metric value.

        Returns
        -------
        float
            Always returns 0.0.
        """
        return 0.0


def enumerate_dataloader(
    dataloader: torch.utils.data.DataLoader,
    dataloader_seed: int | None = None,
    batch_limit: int | None = None,
    epochs: float | None = None,
) -> Generator[tuple[int, Any]]:
    """
    A generator that yields batches from a dataloader with an optional batch limit.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader to iterate over.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to yield after `epochs` epochs.
        Use -1 for no limit. Default is None.
    epochs : float | None, optional
        Proportion of the dataloader to iterate over.
        Is incompatible with non None `batch_limit`.

    Yields
    ------
    Generator[tuple[int, Any]]
        A generator yielding tuples of (batch_index, batch_data).

    Raises
    ------
    AttributeError
        If `dataloader_seed` is provided but the dataloader does not have a random
        number generator attribute.
    TypeError
        If `epochs` and `batch_limit` are both None or
        if they are both provided.
    """
    if not (epochs is None) ^ (batch_limit is None):
        msg = f"Exactly one of `epochs` and `batch_limit` must be provided, but got {epochs=} and {batch_limit=}"
        raise TypeError(msg)
    assert (epochs is None) or (epochs >= 0), "Epochs must be non-negative"
    assert (batch_limit is None) or (
        batch_limit == -1 or batch_limit >= 0
    ), "Batch limit must be -1 or non-negative"
    if dataloader_seed is not None:
        if hasattr(dataloader, "generator") and isinstance(
            dataloader.generator, torch.Generator
        ):
            dataloader.generator.manual_seed(dataloader_seed)
        else:
            raise AttributeError(
                "The dataloader does not have a 'generator' attribute of type torch.Generator, "
                "so the seed cannot be set."
            )
    if batch_limit is None:
        assert isinstance(epochs, float)
        batch_limit = int(len(dataloader) * epochs)
    for i, batch in enumerate(dataloader):
        if i >= batch_limit:
            break
        yield i, batch


@torch.no_grad()
def evaluate_model(
    model: nn.Module | GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    use_extended_model: bool = False,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Evaluate the model on a dataloader.

    Parameters
    ----------
    model : nn.Module | GrowingContainer
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader for evaluation data.
    loss_function : nn.Module
        The loss function to use. Must have reduction="mean".
    use_extended_model : bool, optional
        Whether to use the extended model for evaluation. Default is False.
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to evaluate. Use -1 for no limit. Default is None.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    device : torch.device, optional
        Device to use. Default is torch.device("cpu").

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, metrics_value).

    Raises
    ------
    TypeError
        If the model is not an instance of GrowingContainer when
        `use_extended_model` is True.
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    # metrics meters
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()

    # prediction function
    if use_extended_model:
        if not isinstance(model, GrowingContainer):
            raise TypeError(
                "Model must be an instance of GrowingContainer when use_extended_model is True"
            )
        predict_fn = lambda x: model.extended_forward(x)
    else:
        predict_fn = lambda x: model(x)

    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x, y = x.to(device), y.to(device)
        y_pred: torch.Tensor = predict_fn(x)
        loss = loss_function(y_pred, y)
        loss_meter.update(loss.item(), x.size(0))
        metrics.update(y_pred, y)

    return loss_meter(), metrics()


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    loss_function: nn.Module,
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Train the model on the train_dataloader

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for training data.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    scheduler : Any | None, optional
        Learning rate scheduler. Default is None.
    loss_function : nn.Module
        The loss function to use. Must have reduction="mean".
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        Maximum number of batches to train. Use -1 for no limit. Default is None.
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    device : torch.device, optional
        Device to use. Default is torch.device("cpu").

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, aux_loss_function_value).
    """
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"

    # metrics meters
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()

    model.train()
    for i, (x, y) in enumerate_dataloader(
        train_dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y_pred, y)
        assert (
            loss.isnan().sum() == 0
        ), f"During training of {model}, loss is NaN: {loss}, sample index: {i / len(train_dataloader)}"

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # update metrics
        loss_meter.update(loss.item(), x.size(0))
        metrics.update(y_pred.detach(), y)

    if scheduler is not None:
        scheduler.epoch_step()

    return loss_meter(), metrics()


def compute_statistics(
    model: GrowingContainer,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = nn.MSELoss(reduction="sum"),
    metrics: Metric | None = None,
    batch_limit: int | None = None,
    dataloader_seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Compute the tensor of statistics of the model on the dataloader
    with a limit of batch_limit batches.

    Parameters
    ----------
    model : GrowingContainer
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use.
    loss_function : nn.Module
        The loss function to use. Must have reduction="sum".
    metrics : Metric | None, optional
        A Metric instance to track auxiliary metrics (e.g., accuracy).
        Will be reset at the start and updated each batch. Default is None.
    batch_limit : int | None, optional
        The maximum number of batches to use. Default is None (no limit).
    dataloader_seed : int | None, optional
        An optional seed to set for the dataloader's random number generator (if it has
        one). This can be used to ensure reproducibility when shuffling is involved.
        Default is None.
    device : torch.device, optional
        The device to use. Default is torch.device("cpu").

    Returns
    -------
    tuple[float, float]
        A tuple containing (average_loss, metrics_value).
    """
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"
    loss_meter = AverageMeter()
    if metrics is None:
        metrics = DummyMetric()
    else:
        metrics.reset()

    model.init_computation()
    model.eval()
    for _, (x, y) in enumerate_dataloader(
        dataloader, dataloader_seed=dataloader_seed, batch_limit=batch_limit
    ):
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        model.update_computation()
        loss_meter.update(loss.item() / x.size(0), x.size(0))
        metrics.update(y_pred.detach(), y)

    return loss_meter(), metrics()
