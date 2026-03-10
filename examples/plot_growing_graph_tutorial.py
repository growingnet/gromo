r"""
GrowingGraphNetwork tutorial
============================

A step-by-step guide to neural network growing on a **directed acyclic graph
(DAG)** using the **GroMo** (Growing Modules) library.

What is GrowingGraphNetwork?
-----------------------------

While :class:`~gromo.containers.GrowingMLP` grows neurons inside a fixed
sequence of layers, :class:`~gromo.containers.GrowingGraphNetwork` operates on
a **DAG** topology where:

- **Nodes** are merge modules that receive one or more incoming activations.
- **Edges** are linear (or convolutional) layers connecting nodes.
- New edges can be inserted between any pair of nodes, giving the network the
  freedom to grow its *connectivity* as well as its *width*.

Tutorial Overview
-----------------

In this tutorial, we will:

1. Set up the environment and data loaders
2. Define a ``GraphModel`` wrapping ``GrowingGraphNetwork``
3. Implement the growth procedure: bottleneck detection, action selection,
   candidate evaluation, and weight application
4. Iteratively train and grow the DAG network
5. Inspect the evolving DAG topology
6. Visualise the evolution of loss and model capacity

Let's get started!

Step 1: Environment Setup and Imports
--------------------------------------

First, we import the necessary libraries:
"""

###############################################################################
import operator
import random

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.utils.data
from helpers.synthetic_data import MultiSinDataloader

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.modules.growing_module import MergeGrowingModule
from gromo.utils.training_utils import evaluate_model, gradient_descent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
# Step 2: Define the data loaders
# --------------------------------
#
# We use a custom dataloader with synthetic data for training, validation, and
# testing.
#
# The input :math:`x \sim \mathcal{N}(0_k, 1_k)` and the target is defined as:
#
# .. math:: y[d] = \sum_{i=1}^{k} \sin(i x[i] + d)
#
# We need a **validation** set in addition to train / test because the growth
# procedure selects the best candidate action using held-out data (to avoid
# over-fitting to the training signal used for statistics accumulation).

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

val_data_loader = MultiSinDataloader(
    nb_sample=10,
    batch_size=500,
    in_features=in_features,
    out_features=out_features,
    seed=1,
    device=device,
)

test_data_loader = MultiSinDataloader(
    nb_sample=1,
    batch_size=1_000,
    in_features=in_features,
    out_features=out_features,
    seed=2,
    device=device,
)

###############################################################################
# Step 3: Define the GraphModel Architecture
# -------------------------------------------
#
# ``GraphModel`` is a thin wrapper around :class:`~gromo.containers.GrowingContainer`
# that delegates all computation to an internal
# :class:`~gromo.containers.GrowingGraphNetwork` (``self.growing_dag``).
#
# Key design points
# ^^^^^^^^^^^^^^^^^
#
# - **GrowingGraphNetwork** maintains a ``GrowingDAG`` internally.  Nodes are
#   merge modules; edges are linear layers.
# - We attach a :class:`torch.nn.SELU` activation to the *output* merge node so
#   that the network has non-linearity before the final projection.
# - ``set_growing_layers`` registers the DAG with the container so that the
#   standard growth bookkeeping (statistics, deltas, …) propagates correctly.
#
# +---------------------------+--------------------------------------------------+
# | Method                    | Description                                      |
# +===========================+==================================================+
# | ``forward(x)``            | Flatten input, pass through DAG                  |
# +---------------------------+--------------------------------------------------+
# | ``extended_forward(x)``   | Same as forward but also returns the "growth"    |
# |                           | output used during candidate evaluation          |
# +---------------------------+--------------------------------------------------+
# | ``set_growing_layers()``  | Register ``growing_dag`` with the container      |
# +---------------------------+--------------------------------------------------+


###############################################################################
class GraphModel(GrowingContainer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        neurons: int,
        neuron_epochs: int,
        neuron_lrate: float,
        neuron_batch_size: int,
        loss_fn: torch.nn.Module,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(in_features, out_features, device)

        self.growing_dag = GrowingGraphNetwork(
            in_features=in_features,
            out_features=out_features,
            neurons=neurons,
            neuron_epochs=neuron_epochs,
            neuron_lrate=neuron_lrate,
            neuron_batch_size=neuron_batch_size,
            loss_fn=loss_fn,
            layer_type="linear",
            name="dag",
        )

        # Attach a SELU activation to the output merge node so the final
        # representation is non-linear before the loss is computed.
        self.growing_dag.dag.get_node_module(
            self.growing_dag.dag.end
        ).post_merge_function = torch.nn.Sequential(
            torch.nn.SELU(),
        )

        self.set_growing_layers()

    def set_growing_layers(self) -> None:
        self._growing_layers.append(self.growing_dag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.growing_dag(x)

    def extended_forward(self, x: torch.Tensor, mask: dict = {}):
        x = torch.flatten(x, 1)
        return self.growing_dag.extended_forward(x, mask=mask)


###############################################################################
# Step 4: Helper Functions
# -------------------------
#
# The graph growth procedure is more involved than the MLP case because we must
# decide **where** in the DAG to add capacity, not only **how much**.
#
# Three helper functions handle the statistics and bottleneck logic:
#
# 1. :func:`update_computation` — runs one pass over the training set,
#    accumulating pre-activity gradients and input activations for every node.
# 2. :func:`calculate_bottleneck` — for each node, computes a residual vector
#    that measures how much of the gradient signal is *not* explained by the
#    current edges.  Nodes with a large residual norm are bottlenecks.
# 3. :func:`grow` — orchestrates the full growth step:
#
#    a. Enumerate candidate actions (possible new edges / expansions).
#    b. Accumulate statistics and compute optimal weight increments.
#    c. Identify the most important node via bottleneck norms.
#    d. Restrict the action space to that node and execute candidate
#       expansions.
#    e. Line-search the best scaling factor on the *training* set, then rank
#       candidates by *validation* loss.
#    f. Apply the winning action permanently.


###############################################################################
def update_computation(
    model: GraphModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
) -> tuple[dict]:
    """Run a forward-backward pass and collect per-node statistics.

    Parameters
    ----------
    model: GraphModel
        The graph model whose statistics buffers are already initialised.
    dataloader: torch.utils.data.DataLoader
        Training batches ``(X, Y)``.
    criterion: torch.nn.Module
        Loss function (must support ``reduction="mean"``).

    Returns
    -------
    tuple[dict]
        pre_activities_grad : dict[str, Tensor]
            Concatenated pre-activation gradients for every non-root node.
        inputs : dict[str, Tensor]
            Concatenated input activations for every node.
    """
    all_nodes = list(model.growing_dag.dag.nodes)

    pre_activities_grad = {
        node: torch.empty(0)
        for node in all_nodes
        if (node != model.growing_dag.dag.root) and ("start" not in node)
    }
    inputs = {node: torch.empty(0) for node in all_nodes}

    for X, Y in dataloader:
        X, Y = X.to(model.device), Y.to(model.device)
        model.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        model.update_computation()

        # Accumulate per-node activations and pre-activity gradients on CPU to
        # avoid running out of GPU memory across batches.
        for node_module in set(model.growing_dag.dag.get_all_node_modules()):
            assert node_module.activity is not None

            activity = node_module.activity.clone().detach().cpu()
            inputs[node_module._name] = torch.cat((inputs[node_module._name], activity))

            if node_module._name == model.growing_dag.dag.root:
                continue
            assert node_module.pre_activity is not None
            assert node_module.pre_activity.grad is not None

            pre_activities_grad[node_module._name] = torch.cat(
                (
                    pre_activities_grad[node_module._name],
                    node_module.pre_activity.grad.clone().detach().cpu(),
                )
            )

    return pre_activities_grad, inputs


###############################################################################
def calculate_bottleneck(
    model: GraphModel,
    pre_activities_grad: dict,
    inputs: dict,
) -> dict[str, torch.Tensor]:
    """Compute the expressivity bottleneck for each node.

    For a node :math:`v`, the bottleneck vector is:

    .. math::

        b_v = \\nabla_{z_v} \\mathcal{L}
              - \\sum_{e \\in \\text{in}(v)} \\Delta W_e^* \\, a_{\\text{src}(e)}

    where :math:`\\Delta W_e^*` is the optimal weight increment for edge
    :math:`e` computed by ``compute_optimal_delta()``.  A large
    :math:`\\|b_v\\|` means that no currently proposed weight change can
    explain the gradient at :math:`v`, i.e. the node is a bottleneck.

    Parameters
    ----------
    model: GraphModel
        Graph model after ``compute_optimal_delta()`` has been called.
    pre_activities_grad: dict
        saved gradient of the pre-activities of each node.
    inputs: dict
        saved input of each node.

    Returns
    -------
    bottleneck : dict[str, torch.Tensor]
        Residual gradient vector for each node.

    Raises
    ------
    KeyError
        if the activity gradient or the input of a node was not recorded
    """
    bottleneck = {}
    with torch.no_grad():
        for node_module in set(model.growing_dag.dag.get_all_node_modules()):
            if node_module._name == model.growing_dag.dag.root:
                continue

            if node_module._name not in pre_activities_grad:
                raise KeyError(f"Activity gradient {node_module._name} was not recorded")

            v_proj = pre_activities_grad[node_module._name]
            for module in node_module.previous_modules:
                prev_module = module.previous_module
                if prev_module._name not in inputs:
                    if isinstance(prev_module.previous_modules[0], MergeGrowingModule):
                        prev_module = prev_module.previous_modules[0]

                if prev_module._name not in inputs:
                    raise KeyError(f"Input activity {prev_module._name} was not recorded")

                input_activity = inputs[prev_module._name]
                v_proj = (
                    v_proj
                    - module.optimal_delta_layer(
                        input_activity.to(module.device)  # type: ignore
                    ).cpu()
                )

            bottleneck[node_module._name] = v_proj

    return bottleneck


###############################################################################
def grow(
    model: GraphModel,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
) -> None:
    """Grow the graph model by one step.

    The procedure is:

    1. Enumerate candidate actions (possible new edges / node expansions).
    2. Run a forward-backward pass to accumulate statistics.
    3. Compute optimal weight increments (:math:`\\Delta W^* = S^{-1} M`).
    4. Identify the DAG node with the largest bottleneck norm.
    5. Restrict candidates to that node and execute their weight expansions.
    6. Line-search the scaling factor on the training set.
    7. Score candidates on the validation set and keep the best one.
    8. Apply the winning expansion permanently.

    Parameters
    ----------
    model: GraphModel
        The ``GraphModel`` to grow.
    train_dataloader: torch.utils.data.DataLoader
        Used for statistics accumulation and line search.
    val_dataloader: torch.utils.data.DataLoader
        Used for final candidate ranking to avoid over-fitting.
    criterion: torch.nn.Module
        Loss function (``reduction="mean"``).
    """
    # Enumerate what expansions are possible in the current DAG.
    actions = model.growing_dag.dag.define_next_actions(expand_end=True)

    model.init_computation()
    pre_activities_grad, inputs = update_computation(model, train_dataloader, criterion)

    # Solve for optimal weight increments for every candidate edge.
    model.compute_optimal_delta()
    bottleneck = calculate_bottleneck(model, pre_activities_grad, inputs)
    del pre_activities_grad

    model.reset_computation()

    # Select the DAG node whose bottleneck norm is largest — that is where
    # adding capacity will help the most.
    bott_norms = {
        key: torch.linalg.norm(val)
        for key, val in bottleneck.items()
        if key in model.growing_dag.dag.nodes
    }
    most_important_node = max(bott_norms.items(), key=operator.itemgetter(1))[0]
    print(
        f"Most important node: {most_important_node} "
        f"(bottleneck norm = {bott_norms[most_important_node]:.4f})"
    )

    # Keep only actions that target the selected node.
    actions = model.growing_dag.restrict_action_space(
        actions, chosen_outputs=[most_important_node]
    )

    # Instantiate the candidate weight extensions (not yet applied).
    model.growing_dag.execute_expansions(
        actions=actions,
        bottleneck=bottleneck,
        input_B=inputs,
        amplitude_factor=False,
        evaluate=False,
    )

    # For each candidate, find the best scaling factor via line search on the
    # training set, then record the validation loss for final ranking.
    for action in actions:
        print(f"Evaluating candidate: {action}")
        mask = action.create_mask()

        best_loss = float("inf")
        best_value = 0.0
        for value in [0.0, 0.05, 0.1, 0.5, 1.0]:
            model.set_scaling_factor(value)
            loss, _ = evaluate_model(
                model=model,
                dataloader=val_dataloader,
                loss_function=criterion,
                use_extended_model=True,
                mask=mask,
                device=device,
            )
            print(f"  scaling={value:.1f}  val_loss={loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                best_value = value
        print(f"  => best scaling={best_value:.1f}  val_loss={best_loss:.4f}")

        model.set_scaling_factor(best_value)
        action.metrics["scaling_factor"] = best_value

        action.metrics["loss_val"] = best_loss

    # Pick the candidate with the lowest validation loss and apply it.
    model.growing_dag.choose_growth_best_action(actions)
    print(f"Chose option {model.growing_dag.chosen_action}")
    model.growing_dag.apply_change()


###############################################################################
# Step 5: Create the Initial Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We initialise a ``GraphModel`` with:
#
# - **Input size**: 10 features
# - **Output size**: 3 targets
# - **50 neurons** per internal node (the DAG starts with a single hidden node)
# - A SELU activation on the output merge node
#
# Setting fixed random seeds ensures reproducibility.

###############################################################################
criterion = torch.nn.MSELoss()

torch.manual_seed(1)
random.seed(1)

model = GraphModel(
    in_features=in_features,
    out_features=out_features,
    neurons=10,
    neuron_epochs=100,
    neuron_lrate=1e-2,
    neuron_batch_size=256,
    loss_fn=criterion,
    device=device,
)

print("Initial model:")
print(model)

###############################################################################
# Step 6: Training Loop with Growth
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We alternate between:
#
# 1. **SGD training** — standard gradient descent to optimise current weights.
# 2. **Growing** — the DAG identifies its bottleneck node and adds a new edge
#    (or expands an existing node) to relieve it.
#
# **What to observe:**
#
# - Each growth step may add a new edge between distant nodes, increasing both
#   the connectivity and the parameter count.
# - The validation-based candidate ranking prevents the growth from over-fitting
#   to the training statistics.
# - Test loss should decrease monotonically across growth steps.

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


test_loss, _ = evaluate_model(model, test_data_loader, criterion, device=device)
last_test_loss = test_loss
print(f"[N/A] Step 0  Test Loss: {test_loss:.4f}")

history["step"].append(0)
history["test_loss"].append(test_loss)
history["num_params"].append(count_parameters(model))
history["step_type"].append("SGD")

for step in range(growth_steps):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # --- SGD phase ---
    for epoch in range(1, intermediate_epochs + 1):
        gradient_descent(
            model,
            train_data_loader,
            optimizer,
            scheduler=None,
            loss_function=criterion,
            device=device,
        )
        test_loss, _ = evaluate_model(
            model,
            test_data_loader,
            criterion,
            device=device,
        )
        current_step = epoch + step * (intermediate_epochs + 1)
        print(
            f"[SGD] Step {current_step}  "
            f"Test Loss: {test_loss:.4f} ({test_loss - last_test_loss:+.4f})"
        )
        last_test_loss = test_loss

        history["step"].append(current_step)
        history["test_loss"].append(test_loss)
        history["num_params"].append(count_parameters(model))
        history["step_type"].append("SGD")

    # --- Growth phase ---
    grow(model, train_data_loader, val_data_loader, criterion)
    print("Model after growing:")
    print(model)
    nx.draw(model.growing_dag.dag, pos=nx.planar_layout(model.growing_dag.dag))
    plt.show()

    test_loss, _ = evaluate_model(model, test_data_loader, criterion, device=device)
    current_step = (step + 1) * (intermediate_epochs + 1)
    print(
        f"[GRO] Step {current_step}  "
        f"Test Loss: {test_loss:.4f} ({test_loss - last_test_loss:+.4f})"
    )
    last_test_loss = test_loss

    history["step"].append(current_step)
    history["test_loss"].append(test_loss)
    history["num_params"].append(count_parameters(model))
    history["step_type"].append("GRO")


###############################################################################
# Step 7: Visualise Training Progress
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The figure below tracks two quantities across training steps:
#
# - **Test loss** (left y-axis, blue): how well the model generalises.
# - **Number of parameters** (right y-axis, orange): model capacity.
#
# Vertical dotted lines mark each **growth event**.  Circles (●) denote SGD
# steps; stars (★) denote growth steps.

###############################################################################
fig, ax1 = plt.subplots(figsize=(10, 6))

sgd_indices = [i for i, t in enumerate(history["step_type"]) if t == "SGD"]
gro_indices = [i for i, t in enumerate(history["step_type"]) if t == "GRO"]

# --- Left y-axis: Test Loss ---
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Test Loss", color="tab:blue", fontsize=12)
ax1.plot(
    history["step"],
    history["test_loss"],
    color="tab:blue",
    alpha=0.4,
    linewidth=1.5,
    linestyle="--",
)
ax1.scatter(
    [history["step"][i] for i in sgd_indices],
    [history["test_loss"][i] for i in sgd_indices],
    color="tab:blue",
    marker="o",
    s=70,
    zorder=4,
    label="SGD (Loss)",
)
ax1.scatter(
    [history["step"][i] for i in gro_indices],
    [history["test_loss"][i] for i in gro_indices],
    color="tab:blue",
    marker="*",
    s=250,
    zorder=5,
    edgecolors="white",
    linewidths=0.5,
    label="Growth (Loss)",
)
ax1.tick_params(axis="y", labelcolor="tab:blue")

# --- Right y-axis: Number of Parameters ---
ax2 = ax1.twinx()
ax2.set_ylabel("Number of Parameters", color="tab:orange", fontsize=12)
ax2.plot(
    history["step"],
    history["num_params"],
    color="tab:orange",
    alpha=0.4,
    linewidth=1.5,
    linestyle="--",
)
ax2.scatter(
    [history["step"][i] for i in sgd_indices],
    [history["num_params"][i] for i in sgd_indices],
    color="tab:orange",
    marker="o",
    s=70,
    zorder=4,
    label="SGD (Params)",
)
ax2.scatter(
    [history["step"][i] for i in gro_indices],
    [history["num_params"][i] for i in gro_indices],
    color="tab:orange",
    marker="*",
    s=250,
    zorder=5,
    edgecolors="white",
    linewidths=0.5,
    label="Growth (Params)",
)
ax2.tick_params(axis="y", labelcolor="tab:orange")

# Mark growth events with vertical dotted lines
for i, idx in enumerate(gro_indices):
    ax1.axvline(
        x=history["step"][idx],
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label="Growth event" if i == 0 else None,
    )

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    framealpha=0.9,
    fontsize=10,
)

plt.title(
    "GrowingGraphNetwork — Test Loss and Capacity Evolution",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax1.grid(axis="x", linestyle="--", alpha=0.3)
fig.tight_layout()
plt.show()
print()

###############################################################################
