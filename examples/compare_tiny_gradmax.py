"""
TINY vs GradMax Comparison
===========================

Side-by-side comparison of TINY and GradMax initialization methods.

This script trains two identical MLPs, one using TINY initialization and
one using GradMax initialization, to compare their performance and behavior.

Key Differences:
- TINY: Uses projected gradients and optimizes both input (alpha) and
  output (omega) weights
- GradMax: Simpler approach with alpha=0, optimizes only output weights
  (omega)
"""

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


class MultiSinDataset(Dataset):
    """Synthetic dataset: y[d] = sum_i sin(i * x[i] + d)"""

    def __init__(
        self,
        num_samples,
        in_features,
        out_features,
        seed=0,
        device=torch.device("cpu"),
    ):
        torch.manual_seed(seed)
        self.device = device
        self.data = torch.randn(num_samples, in_features, device=device)
        self.targets = torch.zeros(num_samples, out_features, device=device)
        for d in range(out_features):
            for i in range(in_features):
                self.targets[:, d] += torch.sin((i + 1) * self.data[:, i] + d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def train_epoch(model, device, train_loader, learning_rate=0.01):
    """Train for one epoch."""
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


@torch.no_grad()
def evaluate(model, device, test_loader, extended=False):
    """Evaluate the model."""
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

    return loss / len(test_loader)


def grow_network(
    model, device, train_loader, layer_to_grow, initialization_method, num_neurons=2
):
    """Grow the network using specified initialization method."""
    model.eval()
    criterion = torch.nn.MSELoss()

    model.set_growing_layers(layer_to_grow)
    model.init_computation()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        model.update_computation()

    model.compute_optimal_updates(
        maximum_added_neurons=num_neurons,
        initialization_method=initialization_method,
    )
    model.dummy_select_update()

    # Line search
    best_loss = float("inf")
    best_scale = 0.0
    for scale in [0.0, 0.1, 0.5, 1.0]:
        model.set_scaling_factor(scale)
        loss = evaluate(model, device, train_loader, extended=True)
        if loss < best_loss:
            best_loss = loss
            best_scale = scale

    model.set_scaling_factor(best_scale)
    model.apply_change()
    model.reset_computation()
    return best_scale


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment(device, train_loader, test_loader, initialization_method, config):
    """Run a single experiment with given initialization method."""
    from gromo.containers.growing_mlp import GrowingMLP

    # Create model
    model = GrowingMLP(
        in_features=config["in_features"],
        out_features=config["out_features"],
        hidden_size=config["initial_hidden_size"],
        number_hidden_layers=config["number_hidden_layers"],
        device=device,
        use_bias=True,
    )

    # History tracking
    history = {
        "step": [0],
        "test_loss": [evaluate(model, device, test_loader)],
        "num_params": [count_parameters(model)],
        "step_type": ["INIT"],
    }

    print(f"\n{'=' * 70}")
    print(f"{initialization_method.upper()} INITIALIZATION")
    print("=" * 70)
    print(
        f"Initial loss: {history['test_loss'][0]:.4f}, "
        f"Params: {history['num_params'][0]}"
    )

    # Training loop
    for step in range(config["growth_steps"]):
        # SGD epochs
        for epoch in range(1, config["intermediate_epochs"] + 1):
            train_epoch(model, device, train_loader)
            test_loss = evaluate(model, device, test_loader)
            current_step = epoch + step * (config["intermediate_epochs"] + 1)

            history["step"].append(current_step)
            history["test_loss"].append(test_loss)
            history["num_params"].append(count_parameters(model))
            history["step_type"].append("SGD")

        # Growth step
        layer_to_grow = step % max(1, config["number_hidden_layers"]) + 1
        scale = grow_network(
            model,
            device,
            train_loader,
            layer_to_grow,
            initialization_method,
            config["neurons_to_add"],
        )

        test_loss = evaluate(model, device, test_loader)
        current_step = (step + 1) * (config["intermediate_epochs"] + 1)

        history["step"].append(current_step)
        history["test_loss"].append(test_loss)
        history["num_params"].append(count_parameters(model))
        history["step_type"].append("GRO")

        print(
            f"Step {current_step} [GRO Layer {layer_to_grow}, scale={scale:.1f}]: "
            f"Loss={test_loss:.4f}, Params={count_parameters(model)}"
        )

    print(f"\nFinal - Loss: {test_loss:.4f}, Params: {count_parameters(model)}")
    print(f"Improvement: {history['test_loss'][0] - test_loss:.4f}")

    return history, model


def compare_methods():
    """Main comparison function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    config = {
        "in_features": 10,
        "out_features": 3,
        "initial_hidden_size": 2,
        "number_hidden_layers": 2,
        "growth_steps": 4,
        "intermediate_epochs": 3,
        "neurons_to_add": 2,
    }

    # Create datasets
    train_dataset = MultiSinDataset(
        1000, config["in_features"], config["out_features"], seed=0, device=device
    )
    test_dataset = MultiSinDataset(
        1000, config["in_features"], config["out_features"], seed=1, device=device
    )
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Run both experiments
    tiny_history, _ = run_experiment(device, train_loader, test_loader, "tiny", config)
    gradmax_history, _ = run_experiment(
        device, train_loader, test_loader, "gradmax", config
    )

    # Visualize comparison
    visualize_comparison(tiny_history, gradmax_history)

    # Print summary
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(
        f"{'Method':<15} {'Final Loss':<12} {'Final Params':<15} " f"{'Improvement':<12}"
    )
    print("-" * 70)
    print(
        f"{'TINY':<15} {tiny_history['test_loss'][-1]:<12.4f} "
        f"{tiny_history['num_params'][-1]:<15} "
        f"{tiny_history['test_loss'][0] - tiny_history['test_loss'][-1]:<12.4f}"
    )
    print(
        f"{'GradMax':<15} {gradmax_history['test_loss'][-1]:<12.4f} "
        f"{gradmax_history['num_params'][-1]:<15} "
        f"{gradmax_history['test_loss'][0] - gradmax_history['test_loss'][-1]:<12.4f}"
    )


def visualize_comparison(tiny_history, gradmax_history):
    """Create comparison plots."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Test Loss Comparison
    ax1.plot(
        tiny_history["step"],
        tiny_history["test_loss"],
        color="blue",
        linewidth=2,
        label="TINY",
        marker="o",
        markersize=4,
    )
    ax1.plot(
        gradmax_history["step"],
        gradmax_history["test_loss"],
        color="red",
        linewidth=2,
        label="GradMax",
        marker="s",
        markersize=4,
    )

    # Mark growth steps
    tiny_gro = [i for i, t in enumerate(tiny_history["step_type"]) if t == "GRO"]
    gradmax_gro = [i for i, t in enumerate(gradmax_history["step_type"]) if t == "GRO"]

    ax1.scatter(
        [tiny_history["step"][i] for i in tiny_gro],
        [tiny_history["test_loss"][i] for i in tiny_gro],
        color="blue",
        marker="*",
        s=300,
        zorder=5,
        edgecolors="darkblue",
        linewidths=1.5,
        label="TINY Growth",
    )
    ax1.scatter(
        [gradmax_history["step"][i] for i in gradmax_gro],
        [gradmax_history["test_loss"][i] for i in gradmax_gro],
        color="red",
        marker="*",
        s=300,
        zorder=5,
        edgecolors="darkred",
        linewidths=1.5,
        label="GradMax Growth",
    )

    ax1.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Test Loss (MSE)", fontsize=12, fontweight="bold")
    ax1.set_title("Test Loss Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=":")

    # Plot 2: Parameter Growth Comparison
    ax2.plot(
        tiny_history["step"],
        tiny_history["num_params"],
        color="blue",
        linewidth=2,
        label="TINY",
        marker="o",
        markersize=4,
    )
    ax2.plot(
        gradmax_history["step"],
        gradmax_history["num_params"],
        color="red",
        linewidth=2,
        label="GradMax",
        marker="s",
        markersize=4,
    )

    ax2.scatter(
        [tiny_history["step"][i] for i in tiny_gro],
        [tiny_history["num_params"][i] for i in tiny_gro],
        color="blue",
        marker="*",
        s=300,
        zorder=5,
        edgecolors="darkblue",
        linewidths=1.5,
    )
    ax2.scatter(
        [gradmax_history["step"][i] for i in gradmax_gro],
        [gradmax_history["num_params"][i] for i in gradmax_gro],
        color="red",
        marker="*",
        s=300,
        zorder=5,
        edgecolors="darkred",
        linewidths=1.5,
    )

    ax2.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of Parameters", fontsize=12, fontweight="bold")
    ax2.set_title("Model Capacity Growth", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=":")

    plt.suptitle("TINY vs GradMax Initialization", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("tiny_vs_gradmax_comparison.png", dpi=150, bbox_inches="tight")
    print("\nComparison plot saved to: tiny_vs_gradmax_comparison.png")
    plt.show()


if __name__ == "__main__":
    compare_methods()
