from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gromo.utils.utils import global_device


class Accuracy(nn.Module):
    def __init__(self, k: int = 1, reduction: str = "sum"):
        super(Accuracy, self).__init__()
        assert reduction in [
            "mean",
            "sum",
            "none",
        ], "reduction should be in ['mean', 'sum', 'none']"
        self.reduction = reduction
        self.k = k

    def forward(self, y_pred, y):
        result = y_pred.topk(self.k, dim=1).indices == y.unsqueeze(1)
        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return result.mean()
        elif self.reduction == "sum":
            return result.sum()
        else:
            raise ValueError("reduction should be in ['mean', 'sum', 'none']")


class AxisMSELoss(nn.Module):
    def __init__(self, reduction: str = "sum"):
        super(AxisMSELoss, self).__init__()
        assert reduction in [
            "mean",
            "sum",
            "none",
        ], "reduction should be in ['mean', 'sum', 'none']"
        self.reduction = reduction

    def forward(self, y_pred, y):
        result = ((y_pred - y) ** 2).sum(dim=1)
        if self.reduction == "none":
            return result
        elif self.reduction == "mean":
            return result.mean()
        elif self.reduction == "sum":
            return result.sum()
        else:
            raise ValueError("reduction should be in ['mean', 'sum', 'none']")


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = AxisMSELoss(),
    aux_loss_function: nn.Module | None = Accuracy(k=1),
    batch_limit: int = -1,
    device: torch.device = global_device(),
) -> tuple[float, float]:
    """
    /!/ The loss function should not be averaged over the batch
    """
    assert loss_function.reduction in [
        "mean",
        "sum",
    ], "The loss function should be averaged over the batch"
    normalized_loss = loss_function.reduction == "mean"
    # assert loss_function.reduction == "sum", "The loss function should not be averaged over the batch"
    assert (
        aux_loss_function is None or aux_loss_function.reduction == "sum"
    ), "The aux loss function should not be averaged over the batch"
    model.eval()
    n_batch = 0
    nb_sample = 0
    total_loss = torch.tensor(0.0, device=device)
    aux_total_loss = torch.tensor(0.0, device=device)
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        loss = loss_function(y_pred, y)
        if normalized_loss:
            loss *= y.size(0)
        total_loss += loss
        if aux_loss_function is not None:
            aux_loss = aux_loss_function(y_pred, y)
            aux_total_loss += aux_loss

        nb_sample += x.size(0)
        n_batch += 1
        if 0 <= batch_limit <= n_batch:
            break
    total_loss /= nb_sample
    aux_total_loss /= nb_sample
    return total_loss.item(), aux_total_loss.item()


def extended_evaluate_model(
    growing_model: "GrowingMLP",
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = AxisMSELoss(),
    batch_limit: int = -1,
    device: torch.device = global_device(),
) -> float:
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"
    growing_model.eval()
    n_batch = 0
    nb_sample = 0
    total_loss = torch.tensor(0.0, device=device)
    for x, y in dataloader:
        growing_model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = growing_model.extended_forward(x)
        loss = loss_function(y_pred, y)
        total_loss += loss
        nb_sample += x.size(0)
        n_batch += 1
        if 0 <= batch_limit <= n_batch:
            break
    return total_loss.item() / nb_sample


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader | None = None,
    loss_function=AxisMSELoss(reduction="mean"),
    aux_loss_function: nn.Module | None = Accuracy(k=1),
    optimizer=None,
    lr: float = 1e-2,
    weight_decay: float = 0,
    nb_epoch: int = 10,
    show: bool = False,
    device: torch.device = global_device(),
):
    assert (
        loss_function.reduction == "mean"
    ), "The loss function should be averaged over the batch"
    assert (
        aux_loss_function is None or aux_loss_function.reduction == "sum"
    ), "The aux loss function should not be averaged over the batch"
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_loss_train = []
    epoch_accuracy_train = []
    epoch_loss_val = []
    epoch_accuracy_val = []

    iterator = range(nb_epoch)
    if show:
        iterator = tqdm(iterator)

    for epoch in iterator:
        this_epoch_loss_train = torch.tensor(0.0, device=device)
        this_epoch_accuracy_train = torch.tensor(0.0, device=device)
        nb_examples = 0
        for x, y in train_dataloader:
            x = x.to(global_device())
            y = y.to(global_device())
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            assert (
                loss.isnan().sum() == 0
            ), f"During training of {model}, loss is NaN: {loss}"
            loss.backward()
            optimizer.step()
            this_epoch_loss_train += loss * y.shape[0]
            if aux_loss_function:
                this_epoch_accuracy_train += aux_loss_function(y_pred, y)
            nb_examples += y.shape[0]

        this_epoch_accuracy_train /= nb_examples
        this_epoch_loss_train /= nb_examples
        epoch_loss_train.append(this_epoch_loss_train.item())
        epoch_accuracy_train.append(this_epoch_accuracy_train.item())

        this_epoch_loss_val = 0
        this_epoch_accuracy_val = 0
        if val_dataloader is not None:
            this_epoch_loss_val, this_epoch_accuracy_val = evaluate_model(
                model=model,
                dataloader=val_dataloader,
                loss_function=loss_function,
                aux_loss_function=aux_loss_function,
                device=device,
            )
            epoch_loss_val.append(this_epoch_loss_val)
            epoch_accuracy_val.append(this_epoch_accuracy_val)
            model.train()

        if show and epoch % max(1, (nb_epoch // 10)) == 0:
            print(
                f"Epoch {epoch}:\t",
                f"Train: loss={this_epoch_loss_train:.3e}, accuracy={this_epoch_accuracy_train:.2f}\t",
                (
                    f"Val: loss={this_epoch_loss_val:.3e}, accuracy={this_epoch_accuracy_val:.2f}"
                    if val_dataloader is not None
                    else ""
                ),
            )
    return epoch_loss_train, epoch_accuracy_train, epoch_loss_val, epoch_accuracy_val


def compute_statistics(
    growing_model: "GrowingMLP",
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = AxisMSELoss(),
    aux_loss_function: nn.Module | None = Accuracy(k=1),
    batch_limit: int = 1_000_000,
    device: torch.device = global_device(),
    show: bool = False,
) -> tuple[float, float]:
    """
    Compute the tensor of statistics of the model on the dataloader
    with a limit of batch_limit batches.

    Parameters
    ----------
    growing_model: GrowingMLP
        The model to evaluate
    dataloader: torch.utils.data.DataLoader
        The dataloader to use
    loss_function: nn.Module
        The loss function to use.
        /!/ The loss function should not be averaged over the batch
    aux_loss_function: nn.Module | None
        The auxiliary loss function to use.
    batch_limit: int
        The maximum number of batches to use
    device: torch.device
        The device to use
    show: bool
        If True, display a progress bar

    Returns
    -------
    tuple[float, float]
        loss, auxiliary loss
    """
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"
    assert (
        aux_loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"

    growing_model.init_computation()
    n_batch = 0
    nb_sample = 0
    total_loss = torch.tensor(0.0, device=device)
    total_aux_loss = torch.tensor(0.0, device=device)

    if show:
        dataloader = tqdm(dataloader)

    for x, y in dataloader:
        growing_model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = growing_model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        growing_model.update_computation()
        total_loss += loss
        if aux_loss_function is not None:
            aux_loss = aux_loss_function(y_pred, y)
            total_aux_loss += aux_loss
        nb_sample += x.size(0)
        n_batch += 1
        if 0 <= batch_limit <= n_batch:
            break
    return total_loss.item() / nb_sample, total_aux_loss.item() / nb_sample


def line_search(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = AxisMSELoss(),
    batch_limit: int = -1,
    initial_loss: float | None = None,
    first_order_improvement: float = 1,
    alpha: float = 0.1,
    beta: float = 0.5,
    t0: float | None = None,
    extended_search: bool = True,
    max_iter: int = 100,
    epsilon: float = 1e-7,
    verbose: bool = False,
    device: torch.device = global_device(),
) -> tuple[float, float, list[float], list[float]]:
    gammas = []
    losses = []
    beta = np.sqrt(beta)
    epsilon = np.sqrt(epsilon)
    if isinstance(first_order_improvement, torch.Tensor):
        first_order_improvement = first_order_improvement.item()
    if isinstance(initial_loss, torch.Tensor):
        initial_loss = initial_loss.item()

    def test_gamma(sqrt_gamma):
        model.amplitude_factor = sqrt_gamma
        loss = extended_evaluate_model(
            growing_model=model,
            loss_function=loss_function,
            dataloader=dataloader,
            batch_limit=batch_limit,
            device=device,
        )
        gammas.append(sqrt_gamma**2)
        losses.append(loss)
        if verbose:
            print(f"gamma n° {len(gammas)}: {sqrt_gamma**2:.3e} -> Loss: {loss:.3e}")
        return loss

    def under_bound(sqrt_gamma: float, loss: float):
        return loss < initial_loss - alpha * sqrt_gamma**2 * first_order_improvement

    if initial_loss is None:
        warn("Initial loss is not provided, computing it")
        initial_loss = test_gamma(0.0)
        print(f"Initial loss: {initial_loss:.3e}")
    else:
        gammas.append(0.0)
        losses.append(initial_loss)

    # gamma = t ** 2
    if t0 is None:
        t = np.sqrt(2 * (initial_loss / first_order_improvement))
    else:
        t = np.sqrt(t0)
    l0 = test_gamma(t)
    l1 = l0
    i = 0
    if under_bound(t, l0):
        if extended_search:
            go = True
            while go:
                l0 = l1
                t /= beta
                l1 = test_gamma(t)
                go = l1 < l0 and i < max_iter
                i += 1
            t *= beta
        model.amplitude_factor = t
        return t**2, l0, gammas, losses
    else:
        go = True
        while go:
            l0 = l1
            t *= beta
            l1 = test_gamma(t)
            go = (
                ((not under_bound(t, l1)) or (l1 < l0 and extended_search))
                and i < max_iter
                and t > epsilon
            )
            i += 1
        t /= beta
        model.amplitude_factor = t
        return t**2, l0, gammas, losses


def full_search(
    model: nn.Module,
    loss: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    batch_limit: int = 1_000_000,
    initial_loss: float = None,
    first_order_improvement: float = 1,
    min_value: float = -100,
    max_value: float = 100,
    nb_points: int = 100,
):
    xs = np.linspace(min_value, max_value, nb_points)
    values = []
    for v in tqdm(xs):
        model.amplitude_factor = np.sign(v) * np.sqrt(abs(v))
        values.append(
            extended_evaluate_model(
                growing_model=model,
                loss_function=loss,
                dataloader=dataloader,
                batch_limit=batch_limit,
            )
        )

    return xs, values
