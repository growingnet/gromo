from typing import Callable, Iterable

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pyvis.network import Network
from torch.types import _int


__global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device: str | torch.device) -> None:
    """Set default global device

    Parameters
    ----------
    device : str | torch.device
        device choice
    """
    global __global_device
    if isinstance(device, str):
        __global_device = torch.device(device)
    else:
        __global_device = device


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


def line_search(cost_fn: Callable, verbose: bool = True) -> tuple[float, float]:
    """Line search for black-box convex function

    Parameters
    ----------
    cost_fn : Callable
        black-box convex function
    verbose : bool, optional
        create plot, by default True

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

    if verbose:
        plt.figure()
        plt.plot(f_full, losses)
        plt.xlabel(f"factor $\gamma$")  # type: ignore
        plt.ylabel("loss")
        plt.title(f"Minima at {factor=} with loss={min_loss}")
        plt.show()

    return factor, min_loss


def batch_gradient_descent(
    forward_fn: Callable,
    cost_fn: Callable,
    target: torch.Tensor,
    optimizer,
    max_epochs: int = 100,
    tol: float = 1e-5,
    fast: bool = True,
    eval_fn: Callable | None = None,
    verbose: bool = True,
    loss_name: str = "loss",
    title: str = "",
) -> tuple[list[float], list[float]]:
    """Batch gradient descent implementation

    Parameters
    ----------
    output : torch.Tensor
        current output
    target : torch.Tensor
        target tensor
    cost_fn : Callable
        _description_
    lrate : float, optional
        _description_, by default 0.01
    max_epochs : int, optional
        _description_, by default 100
    tol : float, optional
        _description_, by default 1e-5
    verbose : bool, optional
        _description_, by default True

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

    if verbose:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("epochs")
        plt.ylabel(f"{loss_name}")
        plt.title(f"{title}")
        plt.show()

        if not fast:
            labels = ["train"]
            plt.figure()
            plt.plot(acc_history, label=labels)
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.title(f"{title}")
            plt.legend()
            plt.show()

    return loss_history, acc_history


def DAG_to_pyvis(dag):
    """Create pyvis graph based on GrowableDAG

    Parameters
    ----------
    dag : GrowableDAG
        growable dag object

    Returns
    -------
    _type_
        pyvis object
    """
    # nt = Network('500px', '500px', directed=True, notebook=True, cdn_resources='remote')
    nt = Network(directed=True)

    default_offset_x = 150.0
    default_offset_y = 0.0

    for node in dag.nodes:
        size = dag.nodes[node]["size"]
        attrs = {
            "x": None,
            "y": None,
            "physics": True,
            "label": node,
            "title": str(size),
            "color": size_to_color(size),
            "size": np.sqrt(size),
            "mass": 4,
        }
        if node == "start":
            attrs.update(
                {"x": -default_offset_x, "y": -default_offset_y, "physics": False}
            )
        elif node == "end":
            attrs.update({"x": default_offset_x, "y": default_offset_y, "physics": False})
        nt.add_node(node, **attrs)
    for edge in dag.edges:
        prev_node, next_node = edge
        module = dag.get_edge_module(prev_node, next_node)
        nt.add_edge(
            prev_node, next_node, title=module.name, label=str(module.weight.shape)
        )

    # nt.toggle_physics(False)
    return nt


def size_to_color(size):
    cmap = mpl_cm.Reds
    norm = mpl_colors.Normalize(vmin=0, vmax=784)
    rgba = cmap(norm(size))
    return mpl_colors.rgb2hex(rgba)


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
    true_positives = np.sum((actual == label) & (predicted == label))
    false_positives = np.sum((actual != label) & (predicted == label))
    false_negatives = np.sum((predicted != label) & (actual == label))

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
