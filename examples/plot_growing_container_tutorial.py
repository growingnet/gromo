r"""
GrowingContainer tutorial
=========================

A step-by-step guide to neural network growing using the **GroMo** (Growing
Modules) library.

What is GroMo?
--------------

GroMo is a library that enables **progressive neural network growth** during
training. Instead of defining a fixed architecture upfront, you start with a
small network and **dynamically add neurons** to layers based on gradient
information. This approach can lead to **Informed growth**: New neurons are
added in directions that most reduce the loss.

Tutorial Overview
-----------------

In this notebook, we will:

1. Set up the environment
2. Create a small ``GrowingMLP`` model
3. Define training, evaluation, and growth functions
4. Iteratively train and grow the network

This process is presented for the ``GrowingMLP`` structure but it can be
generalized to other structures like a ResNet, a MLP Mixer, etc.

Let's get started!

Step 1: Environment Setup and Imports
--------------------------------------

First, we import the necessary libraries:
"""

###############################################################################
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from helpers.synthetic_data import MultiSinDataloader

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_mlp import GrowingMLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# Step 2: Define the data loaders
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

train_data_loader = MultiSinDataloader(
    nb_sample=10,
    batch_size=1_000,
    in_features=in_features,
    out_features=out_features,
    seed=0,
    device=device,
)

test_data_loader = MultiSinDataloader(
    nb_sample=1,
    batch_size=1_000,
    in_features=in_features,
    out_features=out_features,
    seed=1,
    device=device,
)

###############################################################################
# Defining the GrowingMLP Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Before we use ``GrowingMLP``, let's look at its
# `implementation <https://github.com/growingnet/gromo/blob/main/src/gromo/containers/growing_mlp.py>`_
# to understand how it works.
#
# Key Components
# ~~~~~~~~~~~~~~
#
# - **LinearGrowingModule**: A special linear layer that supports dynamic
#   growth. Each layer can have neurons added to it during training.
#
# - **GrowingContainer**: The base class that provides the growth infrastructure
#
# Important Methods
# ~~~~~~~~~~~~~~~~~
#
# +-------------------------------+----------------------------------------+
# | Method                        | Description                            |
# +===============================+========================================+
# | ``__init__``                  | Define the layers, carefully link them |
# |                               | together                               |
# +-------------------------------+----------------------------------------+
# | ``forward(x)``                | Standard forward pass using current    |
# |                               | weights                                |
# +-------------------------------+----------------------------------------+
# | ``extended_forward``          | Forward pass that includes proposed    |
# |                               | new neurons                            |
# +-------------------------------+----------------------------------------+
# | ``set_growing_layers(index)`` | Select which layer(s) to grow          |
# +-------------------------------+----------------------------------------+
#

###############################################################################
# Step 3: Define Helper Functions
# --------------------------------
#
# We need three key functions to work with our growing network:
#
# 1. **Training function**: Standard training loop using SGD
# 2. **Evaluation function**: Compute loss, with support for extended mode
# 3. **Growth function**: The core GroMo logic to add new neurons intelligently

###############################################################################
# 3.1 Training Function
# ^^^^^^^^^^^^^^^^^^^^^
#
# A standard PyTorch training loop that performs one epoch of training using
# SGD optimizer with MSE loss.


###############################################################################
def train(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
) -> None:
    """
    Train the model for one epoch using SGD optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    device : torch.device
        The device (CPU or CUDA) to run computations on.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing batches of (input, target) pairs.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


###############################################################################
# 3.2 Evaluation Function
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The evaluation function computes the loss on a dataset. It supports two
# modes:
#
# - **Standard mode** (``extended=False``): Uses the current model weights
# - **Extended mode** (``extended=True``): Evaluates the model *as if* the
#   proposed new neurons were added
#
# This allows us to preview the effect of growth before committing to it.


###############################################################################
@torch.no_grad()
def evaluate(
    model: GrowingContainer,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    extended: bool = False,
) -> float:
    """
    Evaluate the model on a dataset.

    Parameters
    ----------
    model : GrowingContainer
        The neural network model to evaluate.
    device : torch.device
        The device (CPU or CUDA) to run computations on.
    test_loader : torch.utils.data.DataLoader
        DataLoader providing batches of (input, target) pairs.
    extended : bool, optional
        If True, use extended_forward which includes proposed new neurons.
        If False, use standard forward pass. Default is False.

    Returns
    -------
    loss : float
        The average mean squared error loss per sample.
    """
    model.eval()
    criterion = torch.nn.MSELoss(reduction="mean")
    loss = 0.0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if extended:
            output = model.extended_forward(data)
        else:
            output = model(data)
        loss += criterion(output, target).item()

    loss /= len(test_loader)
    return loss


###############################################################################
# 3.3 Growth Function
# ^^^^^^^^^^^^^^^^^^^
#
# This is the **core of GroMo** - the function that intelligently grows the
# network. Here's what happens step by step:
#
# 1. **set_growing_layers(layer_to_grow)**: Specify which layer(s) will be grown
# 2. **init_computation()**: Initialize internal buffers to accumulate gradient
#    statistics
# 3. **Forward/backward pass loop**: Process the entire dataset, accumulating
#    information about optimal growth directions via ``update_computation()``
# 4. **compute_optimal_updates()**: Solve for the optimal new neuron weights
#    based on accumulated statistics
# 5. **dummy_select_update()**: Here it's trivial as we selected the layer
#    before computing the statistics but we could have done it the other way
#    around and select the layer to grow by looking at the different proposed
#    updates
# 6. **Line search**: Try different scaling factors to find the best magnitude
#    for the new neurons
# 7. **apply_change()**: Permanently add the new neurons to the network
#
# The **line search** is crucial: it determines how much to "trust" the
# computed optimal neurons. A scaling factor of 0 means no growth, while
# larger values add stronger contributions from new neurons.


###############################################################################
def grow(
    model: GrowingMLP,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    layer_to_grow: int,
) -> None:
    """
    Grow the specified layer of the model by adding new neurons.

    Parameters
    ----------
    model : GrowingMLP
        The neural network model to grow.
    device : torch.device
        The device (CPU or CUDA) to run computations on.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing batches of (input, target) pairs.
    layer_to_grow : int
        Index of the layer to grow (1-indexed).

    Notes
    -----
    The growth procedure:
    1. Accumulate gradient statistics over the training set
    2. Compute optimal weights for new neurons
    3. Find the best scaling factor via line search
    4. Apply the growth to the model
    """
    model.eval()
    # /!/ We use the reduction="sum" as the averaging is already
    # done in the GrowingMLP methods
    criterion = torch.nn.MSELoss(reduction="sum")

    model.set_growing_layers(layer_to_grow)
    model.init_computation()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        model.update_computation()

    model.compute_optimal_updates()
    model.reset_computation()

    model.dummy_select_update()

    # Line search to find the best scaling factor
    best_loss = float("inf")
    best_value = 0.0
    for value in [0.0, 0.1, 0.5, 1.0]:
        model.set_scaling_factor(value)
        loss = evaluate(model, device, train_loader, extended=True)
        print(f"Scaling factor: {value}, Loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_value = value
    print(f"Best scaling factor: {best_value}, loss: {best_loss:.4f}")

    model.set_scaling_factor(best_value)
    model.apply_change()


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
number_hidden_layers = 2

torch.manual_seed(0)
model = GrowingMLP(
    in_features=in_features,
    out_features=out_features,
    hidden_size=2,
    number_hidden_layers=number_hidden_layers,
    device=device,
)
model

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
growth_steps = 4
intermediate_epochs = 3

# Data collection for plotting
history = {
    "step": [],
    "test_loss": [],
    "num_params": [],
    "step_type": [],  # "SGD" or "GRO"
}


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Original model:")
print(model)

last_test_loss = test_loss = evaluate(model, device, test_data_loader)
print(f"[N/A] Step {0}, Test Loss: {test_loss:.4f}")

# Record initial state
history["step"].append(0)
history["test_loss"].append(test_loss)
history["num_params"].append(count_parameters(model))
history["step_type"].append("SGD")

for step in range(growth_steps):
    for epoch in range(1, intermediate_epochs + 1):
        train(model, device, train_data_loader)
        test_loss = evaluate(model, device, test_data_loader)
        current_step = epoch + step * (intermediate_epochs + 1)
        print(
            f"[SGD] Step {current_step}, "
            f"Test Loss: {test_loss:.4f} ({test_loss - last_test_loss:.4f})"
        )
        last_test_loss = test_loss

        # Record SGD step
        history["step"].append(current_step)
        history["test_loss"].append(test_loss)
        history["num_params"].append(count_parameters(model))
        history["step_type"].append("SGD")

    layer_to_grow: int = step % max(1, number_hidden_layers) + 1
    print(f"Growing layer {layer_to_grow}")
    grow(model, device, train_data_loader, layer_to_grow=layer_to_grow)
    print("Model after growing:")
    print(model)
    test_loss = evaluate(model, device, test_data_loader)
    current_step = (step + 1) * (intermediate_epochs + 1)
    print(
        f"[GRO], Step {current_step}, "
        f"Test Loss: {test_loss:.4f} ({test_loss - last_test_loss:.4f})"
    )
    last_test_loss = test_loss

    # Record growth step (update the last entry to mark it as GRO)
    history["step"].append(current_step)
    history["test_loss"].append(test_loss)
    history["num_params"].append(count_parameters(model))
    history["step_type"].append("GRO")

###############################################################################
# Step 6: Visualize Training Progress
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's plot the evolution of the model's performance:
#
# - **Test loss** (left y-axis): Shows how well the model generalizes
# - **Number of parameters** (right y-axis): Shows the model capacity growth
# - **Markers**: Circles (○) for SGD steps, stars (★) for growth steps

###############################################################################
fig, ax1 = plt.subplots(figsize=(10, 6))

# Separate data by step type
sgd_indices = [i for i, t in enumerate(history["step_type"]) if t == "SGD"]
gro_indices = [i for i, t in enumerate(history["step_type"]) if t == "GRO"]

# Left y-axis: Test Loss
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Test Loss", color="tab:blue", fontsize=12)
ax1.plot(history["step"], history["test_loss"], color="tab:blue", alpha=0.5, linewidth=1)
ax1.scatter(
    [history["step"][i] for i in sgd_indices],
    [history["test_loss"][i] for i in sgd_indices],
    color="tab:blue",
    marker="o",
    s=80,
    label="SGD (Loss)",
    zorder=3,
)
ax1.scatter(
    [history["step"][i] for i in gro_indices],
    [history["test_loss"][i] for i in gro_indices],
    color="tab:blue",
    marker="*",
    s=200,
    label="Growth (Loss)",
    zorder=3,
)
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Right y-axis: Number of Parameters
ax2 = ax1.twinx()
ax2.set_ylabel("Number of Parameters", color="tab:orange", fontsize=12)
ax2.plot(
    history["step"], history["num_params"], color="tab:orange", alpha=0.5, linewidth=1
)
ax2.scatter(
    [history["step"][i] for i in sgd_indices],
    [history["num_params"][i] for i in sgd_indices],
    color="tab:orange",
    marker="o",
    s=80,
    label="SGD (Params)",
    zorder=3,
)
ax2.scatter(
    [history["step"][i] for i in gro_indices],
    [history["num_params"][i] for i in gro_indices],
    color="tab:orange",
    marker="*",
    s=200,
    label="Growth (Params)",
    zorder=3,
)
ax2.tick_params(axis="y", labelcolor="tab:orange")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Model Performance and Capacity Evolution", fontsize=14)
fig.tight_layout()
plt.show()

###############################################################################
