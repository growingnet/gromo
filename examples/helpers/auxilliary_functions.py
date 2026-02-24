from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from gromo.utils.utils import global_device


if TYPE_CHECKING:
    from gromo.containers.growing_mlp import GrowingMLP  # noqa: TC004


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
    nb_sample = 0
    total_loss = torch.tensor(0.0, device=device)
    aux_total_loss = torch.tensor(0.0, device=device)
    for n_batch, (x, y) in enumerate(dataloader, start=1):
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
        if 0 <= batch_limit <= n_batch:
            break
    total_loss /= nb_sample
    aux_total_loss /= nb_sample
    return total_loss.item(), aux_total_loss.item()


def extended_evaluate_model(
    growing_model: GrowingMLP,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module = AxisMSELoss(),
    batch_limit: int = -1,
    device: torch.device = global_device(),
) -> float:
    assert (
        loss_function.reduction == "sum"
    ), "The loss function should not be averaged over the batch"
    growing_model.eval()
    nb_sample = 0
    total_loss = torch.tensor(0.0, device=device)
    for n_batch, (x, y) in enumerate(dataloader, start=1):
        growing_model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = growing_model.extended_forward(x)
        loss = loss_function(y_pred, y)
        total_loss += loss
        nb_sample += x.size(0)
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
