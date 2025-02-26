from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn


__global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device: str | torch.device) -> None:
    """Set default global device

    Parameters
    ----------
    device : str | torch.device
        device choice
    """
    global __global_device
    __global_device = torch.device(device)


def reset_device() -> None:
    """Reset global device"""
    global __global_device
    __global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def global_device() -> torch.device:
    """Get global device for whole codebase

    Returns
    -------
    torch.device
        global device
    """
    global __global_device
    return __global_device


def get_correct_device(self, device: torch.device | str | None) -> torch.device:
    """Get and set the correct device as global
    Precedence works as follows:
        argument > config file > global_device

    Parameters
    ----------
    device : torch.device | str | None
        chosen device argument, leave empty to use config file

    Returns
    -------
    torch.device
        selected correct device
    """
    device = torch.device(
        device
        if device is not None
        else set_from_conf(self, "device", global_device(), setter=False)
    )
    set_device(device)
    return device


def torch_zeros(*size: tuple[int, int], **kwargs) -> torch.Tensor:
    """Create zero tensors on global selected device

    Parameters
    ----------
    size : tuple[int, int]
        size of tensor

    Returns
    -------
    torch.Tensor
        zero-initialized tensor of defined size on global device
    """
    global __global_device
    try:
        return torch.zeros(size=size, device=__global_device, **kwargs)  # type: ignore
    except TypeError:
        return torch.zeros(*size, device=__global_device, **kwargs)


def torch_ones(*size: tuple[int, int], **kwargs) -> torch.Tensor:
    """Create one tensors on global selected device

    Parameters
    ----------
    size : tuple[int, int]
        size of tensor

    Returns
    -------
    torch.Tensor
        one-initialized tensor of defined size on global device
    """
    global __global_device
    try:
        return torch.ones(size=size, device=__global_device, **kwargs)  # type: ignore
    except TypeError:
        return torch.ones(*size, device=__global_device, **kwargs)


def set_from_conf(self, name: str, default: Any = None, setter: bool = True) -> Any:
    """Standardize private argument setting from config file

    Parameters
    ----------
    name : str
        name of variable
    default : Any, optional
        default value in case config does not provide one, by default None
    setter : bool, optional
        set the retrieved value as argument in the object, by default True

    Returns
    -------
    Any
        value set to variable
    """
    # Check that config file has been found and read
    assert hasattr(self, "_config_data")
    assert isinstance(self._config_data, dict)

    value = self._config_data.get(name, default)

    if setter:
        setattr(self, f"{name}", value)

    return value


def activation_fn(fn_name: str) -> nn.Module:
    """Create activation function module by name

    Parameters
    ----------
    fn_name : str
        name of activation function

    Returns
    -------
    torch.nn.Module
        activation function module
    """
    if fn_name is None:
        return nn.Identity()
    fn_name = fn_name.strip().lower()
    if fn_name == "id":
        return nn.Identity()
    elif fn_name == "selu":
        return nn.SELU()
    elif fn_name == "relu":
        return nn.ReLU()
    elif fn_name == "softmax":
        return nn.Softmax(dim=1)
    else:
        return nn.Identity()


def line_search(cost_fn: Callable) -> tuple[float, float]:
    """Line search for black-box convex function

    Parameters
    ----------
    cost_fn : Callable
        black-box convex function

    Returns
    -------
    tuple[float, float]
        return minima and min value
    """
    losses = []
    n_points = 100
    f_min = 1e-6
    f_max = 1
    f_test = np.concatenate(
        [np.zeros(1), np.logspace(np.log10(f_min), np.log10(f_max), n_points)]
    )

    decrease = True
    min_loss = np.inf
    f_full = np.array([])

    while decrease:
        for factor in f_test:
            loss = cost_fn(factor)
            losses.append(loss)

        f_full = np.concatenate([f_full, f_test])

        new_min = np.min(losses)
        decrease = new_min < min_loss
        min_loss = new_min

        f_min = f_max
        f_max = f_max * 10
        f_test = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    factor = f_full[np.argmin(losses)]
    min_loss = np.min(losses)

    # TODO: access full loss history
    # if verbose:
    #     plt.figure()
    #     plt.plot(f_full, losses)
    #     plt.xlabel(f"factor $\gamma$")  # type: ignore
    #     plt.ylabel("loss")
    #     plt.title(f"Minima at {factor=} with loss={min_loss}")
    #     plt.show()

    return factor, min_loss


def mini_batch_gradient_descent(
    model: nn.Module | Callable,
    cost_fn: Callable,
    X: torch.Tensor,
    Y: torch.Tensor,
    lrate: float,
    max_epochs: int,
    batch_size: int,
    parameters: Iterable | None = None,
    fast: bool = False,
    eval_fn: Callable | None = None,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:
    """Mini-batch gradient descent implementation
    Uses AdamW with no weight decay and shuffled DataLoader

    Parameters
    ----------
    model : nn.Module
        pytorch model or forwards function
    cost_fn : Callable
        cost function
    X : torch.Tensor
        input features
    Y : torch.Tensor
        true labels
    lrate : float
        learning rate
    max_epochs : int
        maximum epochs
    batch_size : int
        batch size
    parameters: iterable | None, optional
        list of torch parameters in case the model is just a forward function, by default None
    fast : bool, optional
        fast implementation without evaluation, by default False
    eval_fn : Callable | None, optional
        evaluation function, by default None
    verbose : bool, optional
        print info, by default True

    Returns
    -------
    tuple[list[float], list[float]]
        train loss history, train accuracy history
    """
    loss_history, acc_history = [], []
    full_loss = []
    gradients = []

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if not isinstance(model, nn.Module):
        if (parameters is None) or (len(parameters) < 1):
            raise AttributeError(
                "When the model is just a forward function, the parameters argument must not be None or empty"
            )
    else:
        parameters = model.parameters()
        saved_parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lrate, weight_decay=0)

    for epoch in range(max_epochs):
        correct, total, epoch_loss = 0, 0, 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            output = model(x_batch)
            loss = cost_fn(output, y_batch)
            epoch_loss += loss.item()
            full_loss.append(loss.item())

            if not fast:
                correct += (output.argmax(axis=1) == y_batch).int().sum().item()
                total += len(output)

            loss.backward()

            if isinstance(model, nn.Module):
                avg_grad_norm = 0.0
                for param in model.parameters():
                    avg_grad_norm += param.grad.norm()
                avg_grad_norm /= len(saved_parameters)
                gradients.append(avg_grad_norm.cpu())
            optimizer.step()

        loss_history.append(epoch_loss / len(dataloader))
        if not fast:
            accuracy = correct / total
            acc_history.append(accuracy)
            if eval_fn is not None:
                eval_fn()

        if verbose and epoch % 10 == 0:
            if fast:
                print(f"Epoch {epoch}: Train loss {loss_history[-1]}")
            else:
                print(
                    f"Epoch {epoch}: Train loss {loss_history[-1]} Train Accuracy {accuracy}"
                )

    return loss_history, acc_history


def batch_gradient_descent(
    forward_fn: Callable,
    cost_fn: Callable,
    target: torch.Tensor,
    optimizer,
    max_epochs: int = 100,
    tol: float = 1e-5,
    fast: bool = True,
    eval_fn: Callable | None = None,
) -> tuple[list[float], list[float]]:
    """Batch gradient descent implementation

    Parameters
    ----------
    forward_fn : Callable
        Forward function
    cost_fn : Callable
        _description_
    target : torch.Tensor
        target tensor
    optimizer : torch.optim.Optimizer
        optimizer
    max_epochs : int, optional
        max number of epochs, by default 100
    tol : float, optional
        tolerance, by default 1e-5
    fast : bool, optional
        fast implementation without evaluation, by default True
    eval_fn : Callable | None, optional
        evaluation function, by default None

    Returns
    -------
    list[float]
        _description_
    """
    # print(target, target.shape)
    # temp = (target**2).sum()
    # print(temp)
    loss_history, acc_history = [], []
    min_loss = np.inf
    prev_loss = np.inf

    for _ in range(max_epochs):
        output = forward_fn()
        loss = cost_fn(output, target)
        loss_history.append(loss.item())

        if not fast:
            correct = (output.argmax(axis=1) == target).int().sum().item()
            accuracy = correct / len(output)
            if eval_fn:
                eval_fn()
            acc_history.append(accuracy)

        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()

        # Early stopping
        # if np.abs(prev_loss - loss.item()) <= tol:
        #     break
        if loss.item() < min_loss:
            min_loss = loss.item()
        prev_loss = loss.item()
        # target.detach_()

    return loss_history, acc_history


def calculate_true_positives(
    actual: torch.Tensor, predicted: torch.Tensor, label: int
) -> tuple[float, float, float]:
    """Calculate true positives, false positives and false negatives of a specific label

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels
    label : int
        target label to calculate metrics

    Returns
    -------
    tuple[float, float, float]
        true positives, false positives, false negatives
    """
    true_positives = torch.sum((actual == label) & (predicted == label)).item()
    false_positives = torch.sum((actual != label) & (predicted == label)).item()
    false_negatives = torch.sum((predicted != label) & (actual == label)).item()

    return true_positives, false_positives, false_negatives


def f1(actual: torch.Tensor, predicted: torch.Tensor, label: int) -> float:
    """Calculate f1 score of specific label

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels
    label : int
        target label to calculate f1 score

    Returns
    -------
    float
        f1 score of label
    """
    # F1 = 2 * (precision * recall) / (precision + recall)
    tp, fp, fn = calculate_true_positives(actual, predicted, label)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def f1_micro(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    """Calculate f1 score with micro average

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels

    Returns
    -------
    float
        micro-average f1 score
    """
    true_positives, false_positives, false_negatives = {}, {}, {}
    for label in np.unique(actual):
        tp, fp, fn = calculate_true_positives(actual, predicted, label)
        true_positives[label] = tp
        false_positives[label] = fp
        false_negatives[label] = fn

    all_true_positives = np.sum(list(true_positives.values()))
    all_false_positives = np.sum(list(false_positives.values()))
    all_false_negatives = np.sum(list(false_negatives.values()))

    micro_precision = all_true_positives / (all_true_positives + all_false_positives)
    micro_recall = all_true_positives / (all_true_positives + all_false_negatives)

    f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    return f1


def f1_macro(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    """Calculate f1 score with macro average

    Parameters
    ----------
    actual : torch.Tensor
        true labels
    predicted : torch.Tensor
        predicted labels

    Returns
    -------
    float
        macro-average f1 score
    """
    return float(np.mean([f1(actual, predicted, label) for label in np.unique(actual)]))
