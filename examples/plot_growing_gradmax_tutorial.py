r"""
GradMax tutorial
=========================

This example shows how to use GroMo's boolean-flag API to reproduce the
GradMax-style growth path described in the GradMax paper.

The core configuration is:

- ``compute_delta=False``: skip the natural-gradient delta update
- ``use_covariance=False``: do not precondition with ``S``
- ``alpha_zero=True``: keep the incoming weights of newly added neurons at zero
- ``use_projection=False``: use the raw gradient signal instead of the projected one

In practice, this gives a simple and robust growth loop:

1. train a small network,
2. gather growth statistics,
3. compute a GradMax update,
4. run a short line search on the scaling factor,
5. apply the growth.

Because GradMax keeps the incoming weights of the new neurons at zero, the
proposal is function-preserving at insertion time. In this example the line
search is therefore mostly a sanity check; the new capacity becomes useful
after subsequent training.

Tutorial Overview
-----------------

In this notebook, we will:

1. Set up the environment
2. Create a small ``GrowingMLP`` model
3. Define training, evaluation, and growth functions
4. Iteratively train and grow the network

This process is presented for the ``GrowingMLP`` structure but it can be
generalized to other structures like a ResNet, a MLP Mixer, etc.
The model is intentionally small so the example stays readable and fast
enough for documentation builds.

Let's get started!

Step 1: Environment Setup and Imports
--------------------------------------

First, we import the necessary libraries:
"""

###############################################################################

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from helpers.synthetic_data import MultiSinDataloader

from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.utils import global_device
from gromo.utils.training_utils import (
    compute_statistics,
    evaluate_model,
    gradient_descent,
)


###############################################################################
# Step 2: Helper Functions
# -------------------------
#
# We need three key operations to work with our growing network:
#
# 1. Counting parameters: A simple utility to count the number of trainable parameters in the model.
# 2. **Growth**: The core GroMo logic — we use ``compute_statistics`` to
#    accumulate gradient information, then perform a line search locally
# 3. Plotting: Visualizing the evolution of loss and parameter count across training.
###############################################################################

def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def grow_with_gradmax(
    model: GrowingMLP,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    layer_to_grow: int,
    neurons_to_add: int = 2,
) -> float:
    """Run one GradMax growth step and return the chosen scaling factor."""
    model.set_growing_layers(index=layer_to_grow)
    compute_statistics(model, train_loader, loss_function=torch.nn.MSELoss(reduction="sum"), device=device)


    # Compute the optimal updates for the new neurons using GradMax flags
    # - ``compute_delta=False``: skip the natural-gradient delta update
    # - ``use_covariance=False``: do not precondition with ``S``
    # - ``alpha_zero=True``: keep the incoming weights of newly added neurons at zero
    # - ``use_projection=False``: use the raw gradient signal instead of the projected one
    model.compute_optimal_updates(
        maximum_added_neurons=neurons_to_add,
        compute_delta=False,
        use_covariance=False,
        alpha_zero=True,
        use_projection=False,
        ignore_singular_values=True,
    )
    model.dummy_select_update()

    best_loss = float("inf")
    best_scale = 0.0
    for scale in [0.0, 0.1, 0.5, 1.0]:
        model.set_scaling_factor(scale)
        loss,_ = evaluate_model(model, train_loader, torch.nn.MSELoss(reduction="mean"), device=device, use_extended_model=True)
        print(f"Scaling factor: {scale}, loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_scale = scale

    model.set_scaling_factor(best_scale)
    model.apply_change()
    model.reset_computation()
    return best_scale


def plot_history(history: dict[str, list[float] | list[int] | list[str]]) -> None:
    """Plot the evolution of loss and parameter count across training."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    init_indices = [
        i for i, step_type in enumerate(history["step_type"]) if step_type == "INIT"
    ]
    sgd_indices = [
        i for i, step_type in enumerate(history["step_type"]) if step_type == "SGD"
    ]
    gro_indices = [
        i for i, step_type in enumerate(history["step_type"]) if step_type == "GRO"
    ]

    ax1.set_xlabel("Training step", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test loss", color="tab:blue", fontsize=12, fontweight="bold")
    ax1.plot(
        history["step"],
        history["test_loss"],
        color="tab:blue",
        alpha=0.3,
        linewidth=2,
        linestyle="--",
    )
    ax1.scatter(
        [history["step"][i] for i in init_indices],
        [history["test_loss"][i] for i in init_indices],
        color="gray",
        marker="s",
        s=100,
        label="Initial",
        zorder=3,
        edgecolors="black",
        linewidths=1,
    )
    ax1.scatter(
        [history["step"][i] for i in sgd_indices],
        [history["test_loss"][i] for i in sgd_indices],
        color="tab:blue",
        marker="o",
        s=80,
        label="SGD",
        zorder=3,
        alpha=0.7,
    )
    ax1.scatter(
        [history["step"][i] for i in gro_indices],
        [history["test_loss"][i] for i in gro_indices],
        color="tab:blue",
        marker="*",
        s=300,
        label="GradMax growth",
        zorder=4,
        edgecolors="darkblue",
        linewidths=1.5,
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3, linestyle=":")

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        "Number of parameters",
        color="tab:orange",
        fontsize=12,
        fontweight="bold",
    )
    ax2.plot(
        history["step"],
        history["num_params"],
        color="tab:orange",
        alpha=0.3,
        linewidth=2,
        linestyle="--",
    )
    ax2.scatter(
        [history["step"][i] for i in init_indices],
        [history["num_params"][i] for i in init_indices],
        color="gray",
        marker="s",
        s=100,
        zorder=3,
        edgecolors="black",
        linewidths=1,
    )
    ax2.scatter(
        [history["step"][i] for i in sgd_indices],
        [history["num_params"][i] for i in sgd_indices],
        color="tab:orange",
        marker="o",
        s=80,
        label="SGD parameters",
        zorder=3,
        alpha=0.7,
    )
    ax2.scatter(
        [history["step"][i] for i in gro_indices],
        [history["num_params"][i] for i in gro_indices],
        color="tab:orange",
        marker="*",
        s=300,
        label="GradMax parameters",
        zorder=4,
        edgecolors="darkorange",
        linewidths=1.5,
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)
    plt.title(
        "GradMax: model performance and capacity evolution",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


def main() -> None:
    """Train a small model and grow it with the GradMax configuration."""
    device = global_device()
    print(f"Using device: {device}")

    ###############################################################################
    # Step 3: Define the data loaders
    # --------------------------------
    #
    # We use a custom dataloader with synthetic data for training and testing.
    #
    # The input :math:`x \sim \mathcal{N}(0_k, 1_k)` and the target is defined as:
    #
    # .. math:: y[d] = \sum_{i=1}^{k} \sin(i x[i] + d)

    ###############################################################################

    in_features = 10
    out_features = 3
    hidden_size = 2
    number_hidden_layers = 2
    growth_steps = 2
    intermediate_epochs = 2
    neurons_to_add = 2

    train_loader = MultiSinDataloader(
        nb_sample=4,
        batch_size=256,
        seed=0,
        in_features=in_features,
        out_features=out_features,
        device=device,
    )
    test_loader = MultiSinDataloader(
        nb_sample=4,
        batch_size=256,
        seed=1,
        in_features=in_features,
        out_features=out_features,
        device=device,
    )

    ###############################################################################
    # Step 4: Create the Initial Model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Now we create our ``GrowingMLP`` model with:
    #
    # - **Input size**: 10 (features)
    # - **Output size**: 3 (targets)
    # - **Hidden size**: 2 neurons (intentionally small - we'll grow it!)
    # - **Number of hidden layers**: 2
    #
    # Starting with such a small network demonstrates how GroMo can grow the
    # architecture from minimal capacity.

    ###############################################################################
    model = GrowingMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_size=hidden_size,
        number_hidden_layers=number_hidden_layers,
        activation=torch.nn.GELU(),
        use_bias=True,
        device=device,
    )

    history: dict[str, list[float] | list[int] | list[str]] = {
        "step": [],
        "test_loss": [],
        "num_params": [],
        "step_type": [],
    }

    ###############################################################################
    # Step 5: Training Loop with Growth
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Now comes the exciting part! We run an iterative process that alternates
    # between:
    #
    # 1. **Training**: Standard gradient descent to improve the current network
    # 2. **Growing**: Adding new neurons to increase capacity
    #
    # **What to observe:**
    #
    # - The model starts very small (only 4 hidden neurons, 2 per hidden layer)
    # - After each growth step, the model architecture changes (more neurons are
    #   added)
    # - Test loss should decrease as the network gains capacity
    # - The line search output shows how different scaling factors affect
    #   performance
    #
    # Watch how the network progressively grows and improves!

    ###############################################################################

    criterion = torch.nn.MSELoss()
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    initial_loss,_ = evaluate_model(model, test_loader, criterion, device=device)
    print(f"[N/A] Step {0}, Initial Loss: {initial_loss:.4f}")
    history["step"].append(0)
    history["test_loss"].append(initial_loss)
    history["num_params"].append(count_parameters(model))
    history["step_type"].append("INIT")

    last_loss = initial_loss
    for growth_step in range(growth_steps):
        print(f"\nGrowth cycle {growth_step + 1}/{growth_steps}")

        for epoch in range(intermediate_epochs):
            gradient_descent(
                        model,
                        train_loader,
                        optimizer,
                        scheduler=None,
                        loss_function=criterion,
                        device=device,
                    )
            train_loss,_ = evaluate_model(model, train_loader, criterion, device=device)
            test_loss,_ = evaluate_model(model, test_loader, criterion, device=device)
            current_step = growth_step * (intermediate_epochs + 1) + epoch + 1
            print(
                f"[SGD] step {current_step}, train loss: {train_loss:.4f}, "
                f"test loss: {test_loss:.4f} ({test_loss - last_loss:+.4f})"
            )
            last_loss = test_loss
            history["step"].append(current_step)
            history["test_loss"].append(test_loss)
            history["num_params"].append(count_parameters(model))
            history["step_type"].append("SGD")

        layer_to_grow: int = growth_step % max(1, number_hidden_layers)
        print(f"Growing layer {layer_to_grow} with GradMax flags")
        scale = grow_with_gradmax(
            model,
            device=device,
            train_loader=train_loader,
            layer_to_grow=layer_to_grow,
            neurons_to_add=neurons_to_add,
        )
        test_loss,_ = evaluate_model(model, test_loader, criterion, device=device)
        current_step = (growth_step + 1) * (intermediate_epochs + 1)
        print(
            f"[GRO] step {current_step}, scale: {scale:.1f}, test loss: {test_loss:.4f}"
        )
        last_loss = test_loss
        history["step"].append(current_step)
        history["test_loss"].append(test_loss)
        history["num_params"].append(count_parameters(model))
        history["step_type"].append("GRO")

    print("\nFinal model:")
    print(model)
    print(f"Final test loss: {last_loss:.4f}")
    print(f"Improvement: {history['test_loss'][0] - last_loss:.4f}")

    plot_history(history)


if __name__ == "__main__":
    main()