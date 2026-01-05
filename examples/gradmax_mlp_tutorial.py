"""
GradMax Initialization Tutorial
================================

A step-by-step guide to neural network growing using GradMax initialization.

This tutorial demonstrates how to use the GradMax initialization method for
progressive neural network growth, as an alternative to the TINY method.

GradMax initializes new neurons by maximizing gradient information:
- Alpha (input weights): Set to zeros
- Omega (output weights): Top-k left singular vectors of the
  gradient-input correlation matrix

This results in simpler initialization that focuses on output weight
optimization.
"""

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Custom Synthetic Dataset
# ============================================================================


class MultiSinDataset(Dataset):
    """
    Synthetic dataset where y[d] = sum_i sin(i * x[i] + d)

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    in_features : int
        Input dimension
    out_features : int
        Output dimension
    seed : int
        Random seed for reproducibility
    device : torch.device
        Device to place tensors on
    """

    def __init__(
        self,
        num_samples: int,
        in_features: int,
        out_features: int,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        torch.manual_seed(seed)
        self.device = device

        # Generate random input data: x ~ N(0, 1)
        self.data = torch.randn(num_samples, in_features, device=device)

        # Generate targets: y[d] = sum_i sin(i * x[i] + d)
        self.targets = torch.zeros(num_samples, out_features, device=device)
        for d in range(out_features):
            for i in range(in_features):
                self.targets[:, d] += torch.sin((i + 1) * self.data[:, i] + d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ============================================================================
# Training Function
# ============================================================================


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    learning_rate: float = 0.01,
) -> None:
    """
    Train the model for one epoch using SGD optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    device : torch.device
        The device (CPU or CUDA) to run computations on.
    train_loader : DataLoader
        DataLoader providing batches of (input, target) pairs.
    learning_rate : float
        Learning rate for SGD optimizer.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


# ============================================================================
# Evaluation Function
# ============================================================================


@torch.no_grad()
def evaluate(
    model,
    device: torch.device,
    test_loader: DataLoader,
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
    test_loader : DataLoader
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


# ============================================================================
# Growth Function (GradMax)
# ============================================================================


def grow(
    model,
    device: torch.device,
    train_loader: DataLoader,
    layer_to_grow: int,
    initialization_method: str = "gradmax",
    num_neurons_to_add: int = 2,
) -> None:
    """
    Grow the network by adding neurons using GradMax initialization.

    Steps:
    1. Set which layer to grow
    2. Initialize computation buffers
    3. Accumulate gradient statistics over the dataset
    4. Compute optimal updates using GradMax
    5. Select the update (here: the single layer we chose)
    6. Line search for optimal scaling factor
    7. Apply the changes

    Parameters
    ----------
    model : GrowingMLP
        The growing MLP model.
    device : torch.device
        The device to run computations on.
    train_loader : DataLoader
        DataLoader for accumulating gradient statistics.
    layer_to_grow : int
        Index of the layer to grow.
    initialization_method : str
        Initialization method: "gradmax" or "tiny".
    num_neurons_to_add : int
        Number of neurons to add.
    """
    model.eval()
    criterion = torch.nn.MSELoss()

    # Step 1: Set the layer to grow
    model.set_growing_layers(layer_to_grow)

    # Step 2: Initialize computation buffers
    model.init_computation()

    # Step 3: Accumulate gradient statistics
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        model.update_computation()

    # Step 4: Compute optimal updates using specified initialization method
    model.compute_optimal_updates(
        maximum_added_neurons=num_neurons_to_add,
        initialization_method=initialization_method,
    )

    # Step 5: Select which update to apply (here: the single layer we chose)
    model.dummy_select_update()

    # Step 6: Line search for optimal scaling factor
    scaling_factors = [0.0, 0.1, 0.5, 1.0]
    best_loss = float("inf")
    best_scale = 0.0

    for scale in scaling_factors:
        model.set_scaling_factor(scale)
        loss = evaluate(model, device, train_loader, extended=True)
        print(f"Scaling factor: {scale}, Loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_scale = scale

    print(f"Best scaling factor: {best_scale}, loss: {best_loss:.4f}")

    # Step 7: Apply the change with the best scaling factor
    model.set_scaling_factor(best_scale)
    model.apply_change()
    model.reset_computation()  # Reset computation state after growth


# ============================================================================
# Utility Functions
# ============================================================================


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Main Training Loop
# ============================================================================


def main():
    """Main function to demonstrate GradMax initialization."""

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data parameters
    in_features = 10
    out_features = 3

    # Training parameters
    initial_hidden_size = 2
    number_hidden_layers = 2
    growth_steps = 4
    intermediate_epochs = 3
    neurons_to_add = 2

    # Create datasets and dataloaders
    train_dataset = MultiSinDataset(
        num_samples=1000,
        in_features=in_features,
        out_features=out_features,
        seed=0,
        device=device,
    )
    test_dataset = MultiSinDataset(
        num_samples=1000,
        in_features=in_features,
        out_features=out_features,
        seed=1,
        device=device,
    )

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Import here to ensure it's after setup
    from gromo.containers.growing_mlp import GrowingMLP

    # Create initial model
    model = GrowingMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_size=initial_hidden_size,
        number_hidden_layers=number_hidden_layers,
        device=device,
        use_bias=True,
    )

    print("\n" + "=" * 70)
    print("GRADMAX INITIALIZATION TUTORIAL")
    print("=" * 70)
    print("\nOriginal model:")
    print(model)
    print(f"Parameters: {count_parameters(model)}")

    # History tracking
    history = {
        "step": [],
        "test_loss": [],
        "num_params": [],
        "step_type": [],
    }

    # Initial evaluation
    test_loss = evaluate(model, device, test_loader)
    print(f"\n[N/A] Step 0, Test Loss: {test_loss:.4f}")
    last_test_loss = test_loss

    history["step"].append(0)
    history["test_loss"].append(test_loss)
    history["num_params"].append(count_parameters(model))
    history["step_type"].append("INIT")

    # Training loop
    for step in range(growth_steps):
        print(f"\n{'-' * 70}")
        print(f"GROWTH CYCLE {step + 1}/{growth_steps}")
        print(f"{'-' * 70}")

        # Train for several epochs
        for epoch in range(1, intermediate_epochs + 1):
            train(model, device, train_loader)
            test_loss = evaluate(model, device, test_loader)
            current_step = epoch + step * (intermediate_epochs + 1)
            print(
                f"[SGD] Step {current_step}, "
                f"Test Loss: {test_loss:.4f} ({test_loss - last_test_loss:+.4f})"
            )
            last_test_loss = test_loss

            history["step"].append(current_step)
            history["test_loss"].append(test_loss)
            history["num_params"].append(count_parameters(model))
            history["step_type"].append("SGD")

        # Grow the network
        layer_to_grow = step % max(1, number_hidden_layers) + 1
        print(f"\nGrowing layer {layer_to_grow} using GradMax initialization...")
        grow(
            model,
            device,
            train_loader,
            layer_to_grow=layer_to_grow,
            initialization_method="gradmax",
            num_neurons_to_add=neurons_to_add,
        )

        print("\nModel after growing:")
        print(model)
        print(f"Parameters: {count_parameters(model)}")

        test_loss = evaluate(model, device, test_loader)
        current_step = (step + 1) * (intermediate_epochs + 1)
        print(
            f"[GRO] Step {current_step}, "
            f"Test Loss: {test_loss:.4f} ({test_loss - last_test_loss:+.4f})"
        )
        last_test_loss = test_loss

        history["step"].append(current_step)
        history["test_loss"].append(test_loss)
        history["num_params"].append(count_parameters(model))
        history["step_type"].append("GRO")

    # Final summary
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("Final model:")
    print(model)
    print(f"Final parameters: {count_parameters(model)}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Initial test loss: {history['test_loss'][0]:.4f}")
    print(f"Improvement: {history['test_loss'][0] - test_loss:.4f}")

    # Visualization
    visualize_training(history)


def visualize_training(history: dict) -> None:
    """Visualize the training progress."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Separate data by step type
    init_indices = [i for i, t in enumerate(history["step_type"]) if t == "INIT"]
    sgd_indices = [i for i, t in enumerate(history["step_type"]) if t == "SGD"]
    gro_indices = [i for i, t in enumerate(history["step_type"]) if t == "GRO"]

    # Left y-axis: Test Loss
    ax1.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Loss (MSE)", color="tab:blue", fontsize=12, fontweight="bold")
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
        label="SGD (Loss)",
        zorder=3,
        alpha=0.7,
    )
    ax1.scatter(
        [history["step"][i] for i in gro_indices],
        [history["test_loss"][i] for i in gro_indices],
        color="tab:blue",
        marker="*",
        s=300,
        label="GradMax Growth (Loss)",
        zorder=4,
        edgecolors="darkblue",
        linewidths=1.5,
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle=":")

    # Right y-axis: Number of Parameters
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        "Number of Parameters", color="tab:orange", fontsize=12, fontweight="bold"
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
        label="SGD (Params)",
        zorder=3,
        alpha=0.7,
    )
    ax2.scatter(
        [history["step"][i] for i in gro_indices],
        [history["num_params"][i] for i in gro_indices],
        color="tab:orange",
        marker="*",
        s=300,
        label="GradMax Growth (Params)",
        zorder=4,
        edgecolors="darkorange",
        linewidths=1.5,
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=10)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )

    plt.title(
        "GradMax: Model Performance and Capacity Evolution",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()
    plt.savefig("gradmax_training_progress.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: gradmax_training_progress.png")
    plt.show()


if __name__ == "__main__":
    main()
