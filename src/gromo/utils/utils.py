from typing import Callable

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
    """
    Create zero tensors on selected device
    :param tuple[int, int] size: size of tensor
    :returns torch.Tensor: zero-initialized tensor of defined size on device
    """
    global __global_device
    try:
        return torch.zeros(size=size, device=__global_device, **kwargs)  # type: ignore
    except TypeError:
        return torch.zeros(*size, device=__global_device, **kwargs)


def torch_ones(*size: tuple[int, int], **kwargs) -> torch.Tensor:
    """
    Create zero tensors on selected device
    :param tuple[int, int] size: size of tensor
    :returns torch.Tensor: zero-initialized tensor of defined size on device
    """
    global __global_device
    try:
        return torch.ones(size=size, device=__global_device, **kwargs)  # type: ignore
    except TypeError:
        return torch.ones(*size, device=__global_device, **kwargs)


def activation_fn(fn_name: str) -> nn.Module:
    """_summary_

    Parameters
    ----------
    fn_name : str
        _description_

    Returns
    -------
    torch.nn.Module
        _description_
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
    """_summary_

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
            if eval_fn:
                labels.extend(["test"] * (len(acc_history[0]) - 1))
            plt.figure()
            plt.plot(acc_history, label=labels)
            plt.xlabel("epochs")
            plt.ylabel("accuracy")
            plt.title(f"{title}")
            plt.legend()
            plt.show()

    return loss_history, acc_history


def DAG_to_pyvis(dag):
    # nt = Network('500px', '500px', directed=True, notebook=True, cdn_resources='remote')
    nt = Network(directed=True)

    for node in dag.nodes:
        size = dag.nodes[node]["size"]
        attrs = {
            "x": None,
            "y": None,
            "physics": True,
            "label": node,
            "title": str(size),
            "color": size_to_color(size),
            "mass": 4,
        }
        if node == "start":
            attrs.update({"x": -20.0, "y": 0.0, "physics": False})
        elif node == "end":
            attrs.update({"x": 20.0, "y": 0.0, "physics": False})
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
