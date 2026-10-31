r"""
GrowRA Adapter Tutorial
=======================

This tutorial shows how **GrowRA** (Growing LoRA) adapters efficiently adapt
a frozen pretrained model to a new, harder target task.

**The scenario.**

* **Task 1 (pre-training)**: 6-class Gaussian-cluster classification in a
  clean, well-separated feature space.  The base model converges to ≈ 99 %.
* **Task 2 (adaptation)**: *same 6 classes* but the clusters are *rotated
  30 °* and *more overlapping* (higher noise std).  The frozen base model
  scores only ≈ 53 % because the rotation places every Task-2 centre on a
  Task-1 decision boundary.

With GrowRA we attach rank-0 adapters, grow them once using Fisher budget
guidance, then fine-tune **only the small adapter matrices**:

* GrowRA at **rank 1** uses just ≈ 5 % of the base parameter count and
  recovers ≈ 78 % accuracy — matching full fine-tuning.
* Full fine-tuning reaches ≈ 78 % but updates *all* parameters.
* Neither method reaches 100 %, so the learning curves are clearly visible
  and the efficiency trade-off is easy to compare.
"""

###############################################################################
# Imports
# -------

import copy
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.data
from matplotlib.lines import Line2D

from gromo.growra.container import get_growra_model


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Structured synthetic datasets
# ------------------------------
#
# Each sample has 10 features: the first 2 are the *signal* (2-D cluster
# position), the rest are negligible noise.  The 6 class centres form a
# regular hexagon of radius 2.5 on the (x₀, x₁) plane.
#
# * **Task 1** - clean clusters (std = 0.4), easy to learn.
# * **Task 2** - same hexagon **rotated 30 °** and **noisier** (std = 0.9).

NC = 6  # number of classes
IN_DIM = 10  # input dimension (2 signal + 8 noise)
HIDDEN = 64
R = 2.5

_angles = [i * 2 * math.pi / NC for i in range(NC)]
_shift = math.pi / NC  # 30 ° = half step of the hexagon

CENTERS_T1 = [(R * math.cos(a), R * math.sin(a)) for a in _angles]
CENTERS_T2 = [(R * math.cos(a + _shift), R * math.sin(a + _shift)) for a in _angles]

STD_T1 = 0.4  # Task 1: clean, well-separated
STD_T2 = 0.9  # Task 2: noisier, more overlap


def make_dataloader(
    centers: list[tuple[float, float]],
    std: float,
    n: int,
    seed: int,
    batch_size: int = 32,
) -> torch.utils.data.DataLoader:
    """Gaussian-cluster dataloader; first 2 features carry the class signal."""
    torch.manual_seed(seed)
    nc = len(centers)
    n_per = n // nc
    Xs, ys = [], []
    for label, (cx, cy) in enumerate(centers):
        signal = torch.randn(n_per, 2) * std + torch.tensor([cx, cy])
        noise = torch.randn(n_per, IN_DIM - 2) * 0.1
        Xs.append(torch.cat([signal, noise], dim=1))
        ys.append(torch.full((n_per,), label, dtype=torch.long))
    X = torch.cat(Xs)
    y = torch.cat(ys)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    ds = torch.utils.data.TensorDataset(X[perm].to(device), y[perm].to(device))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


train1 = make_dataloader(CENTERS_T1, STD_T1, n=800, seed=0)
test1 = make_dataloader(CENTERS_T1, STD_T1, n=300, seed=1)
train2 = make_dataloader(CENTERS_T2, STD_T2, n=600, seed=2)
test2 = make_dataloader(CENTERS_T2, STD_T2, n=300, seed=3)

###############################################################################
# Model and training utilities
# ----------------------------


class MLP(nn.Module):
    """Two-hidden-layer ReLU MLP for NC-class classification."""

    def __init__(self, in_dim: int = IN_DIM, nc: int = NC, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


def eval_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    """Compute accuracy on an evaluation split (model.eval(), no grad)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
) -> None:
    model.train()
    for X, y in loader:
        opt.zero_grad()
        F.cross_entropy(model(X), y).backward()
        opt.step()


###############################################################################
# Step 1 - Pre-train base model on Task 1
# ----------------------------------------

base_model = MLP().to(device)
opt = torch.optim.Adam(base_model.parameters(), lr=1e-2)
for _ in range(40):
    train_epoch(base_model, train1, opt)

task1_acc = eval_accuracy(base_model, test1)
print(f"Base model - Task 1 eval accuracy: {task1_acc:.1%}")

###############################################################################
# Step 2 - Frozen base model on Task 2
# --------------------------------------
#
# The 30 ° rotation places every Task-2 cluster centre on a Task-1 decision
# boundary, causing frequent confusion → accuracy drops well below Task 1.

frozen_acc = eval_accuracy(base_model, test2)
print(f"Frozen base model - Task 2 eval accuracy: {frozen_acc:.1%}")

###############################################################################
# Step 3 - Wrap with GrowRA and grow the adapters
# -------------------------------------------------
#
# :func:`~gromo.growra.container.get_growra_model` replaces every linear layer
# with a rank-0 GrowRA adapter (zero extra trainable parameters; base frozen).
# We accumulate Fisher statistics over a pass through Task-2 training data to
# find the most informative low-rank growth directions, then grow once.

# Keep a copy of the frozen model for the full fine-tuning comparison (Step 5).
base_for_ft = copy.deepcopy(base_model)

# Wrap with rank-0 adapters (base_model is frozen in-place).
growra_model = get_growra_model(base_model)


def grow_adapters(
    growra_model,
    loader: torch.utils.data.DataLoader,
    added_rank: int = 1,
) -> None:
    """One GrowRA growth step: accumulate Fisher → compute directions → grow → reset."""
    mods = growra_model.growra_modules()
    for m in mods:
        m.init_computation()

    growra_model.eval()
    for X, y in loader:
        growra_model.zero_grad()
        F.cross_entropy(growra_model(X), y).backward()
        for m in mods:
            m.update_computation()

    for m in mods:
        m.compute_optimal_updates(
            compute_delta=False,
            use_covariance=True,
            use_projection=False,
            alpha_zero=False,
            omega_zero=False,
            ignore_singular_values=True,
            use_fisher=True,
        )
        m.sub_select_optimal_added_parameters(keep_neurons=added_rank)
        m.apply_change(scaling_factor=1.0, extension_size=added_rank)
        m.reset_computation()


grow_adapters(growra_model, train2, added_rank=1)

adapter_params = sum(p.numel() for p in growra_model.growra_parameters())
base_params = sum(p.numel() for p in base_for_ft.parameters())
ranks = [m.rank for m in growra_model.growra_modules()]
print(f"\nAdapter ranks per layer: {ranks}")
print(
    f"Adapter params: {adapter_params:,}  ({100 * adapter_params / base_params:.1f}% of {base_params:,} base params)"
)

###############################################################################
# Step 4 - Fine-tune adapter parameters on Task 2
# -------------------------------------------------
#
# Only the small A and B matrices (one pair per linear layer) are updated;
# the base model weights remain completely frozen throughout.

N_EPOCHS = 30
growra_opt = torch.optim.Adam(growra_model.growra_parameters(), lr=3e-3)
growra_accs: list[float] = []
for _ in range(N_EPOCHS):
    train_epoch(growra_model, train2, growra_opt)
    growra_accs.append(eval_accuracy(growra_model, test2))

growra_final = growra_accs[-1]
print(f"\nGrowRA - Task 2 eval accuracy: {growra_final:.1%}")

###############################################################################
# Step 5 - Full fine-tuning baseline
# ------------------------------------
#
# Unfreeze all base-model parameters and fine-tune on Task 2 for the same
# number of epochs.  This shows what updating *all* parameters achieves with
# the same training budget.

for p in base_for_ft.parameters():
    p.requires_grad = True

ft_opt = torch.optim.Adam(base_for_ft.parameters(), lr=3e-3)
ft_accs: list[float] = []
for _ in range(N_EPOCHS):
    train_epoch(base_for_ft, train2, ft_opt)
    ft_accs.append(eval_accuracy(base_for_ft, test2))

ft_final = ft_accs[-1]
print(f"Full fine-tune - Task 2 eval accuracy: {ft_final:.1%}")

###############################################################################
# Results summary
# ---------------

print(f"\n{'=' * 55}")
print(f"  Frozen base model               : {frozen_acc:.1%}")
print(
    f"  GrowRA ({adapter_params:,} adapter params) : {growra_final:.1%}  (+{growra_final - frozen_acc:.1%})"
)
print(
    f"  Full fine-tune ({base_params:,} params): {ft_final:.1%}  (+{ft_final - frozen_acc:.1%})"
)
print(f"{'=' * 55}")
print(
    f"\n  GrowRA uses {100 * adapter_params / base_params:.0f}% of the base parameter count"
)
print(
    f"  and achieves {100 * growra_final / ft_final:.0f}% of full fine-tuning accuracy."
)

###############################################################################
# Visualisation
# -------------
#
# * **Left** - data scatter (first 2 features) showing Task-1 and Task-2
#   cluster positions.  The 30 ° rotation and extra spread are clearly visible.
# * **Centre** - eval accuracy on Task 2 per fine-tuning epoch.
# * **Right** - final eval accuracy bar chart with parameter counts.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "GrowRA: efficient task adaptation with Fisher-budget low-rank adapters",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

# ---- Panel 1: data scatter ----
ax = axes[0]
colors6 = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
torch.manual_seed(0)
for label, (cx, cy) in enumerate(CENTERS_T1):
    pts = torch.randn(40, 2) * STD_T1 + torch.tensor([cx, cy])
    ax.scatter(
        pts[:, 0].numpy(),
        pts[:, 1].numpy(),
        c=colors6[label],
        alpha=0.4,
        s=15,
        marker="o",
    )
    ax.scatter(
        cx,
        cy,
        c=colors6[label],
        s=100,
        marker="o",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )

for label, (cx, cy) in enumerate(CENTERS_T2):
    pts = torch.randn(40, 2) * STD_T2 + torch.tensor([cx, cy])
    ax.scatter(
        pts[:, 0].numpy(),
        pts[:, 1].numpy(),
        c=colors6[label],
        alpha=0.25,
        s=15,
        marker="^",
    )
    ax.scatter(
        cx,
        cy,
        c=colors6[label],
        s=100,
        marker="^",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markersize=8,
        markeredgecolor="black",
        label="Task 1 (clean, std=0.4)",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor="gray",
        markersize=8,
        markeredgecolor="black",
        label="Task 2 (rotated 30°, std=0.9)",
    ),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
ax.set_title("Task 1 vs Task 2 data", fontsize=11, fontweight="bold")
ax.set_xlabel("x₀ (signal dim 1)", fontsize=10)
ax.set_ylabel("x₁ (signal dim 2)", fontsize=10)
ax.set_aspect("equal")
ax.grid(alpha=0.2)

# ---- Panel 2: learning curves ----
ax = axes[1]
epochs = range(N_EPOCHS + 1)
ax.plot(
    epochs,
    [frozen_acc, *ft_accs],
    color="#4ECDC4",
    linewidth=2.5,
    marker="o",
    markersize=4,
    label=f"Full fine-tune ({ft_final:.0%}, {base_params:,} params)",
)
ax.plot(
    epochs,
    [frozen_acc, *growra_accs],
    color="#FF6B6B",
    linewidth=2.5,
    marker="s",
    markersize=4,
    label=f"GrowRA ({growra_final:.0%}, {adapter_params:,} params)",
)
ax.set_xlabel("Fine-tuning epoch", fontsize=11)
ax.set_ylabel("Eval accuracy on Task 2", fontsize=11)
ax.set_title("Eval accuracy during adaptation", fontsize=11, fontweight="bold")
ax.set_ylim(0.2, 1.0)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- Panel 3: bar chart ----
ax = axes[2]
methods = [
    f"Frozen base\n({base_params:,} params,\nnone updated)",
    f"Full fine-tune\n({base_params:,} params\nupdated)",
    f"GrowRA adapters\n({adapter_params:,} params\nupdated)",
]
accs = [frozen_acc, ft_final, growra_final]
bar_colors = ["#BBBBBB", "#4ECDC4", "#FF6B6B"]
bars = ax.bar(
    methods,
    accs,
    color=bar_colors,
    edgecolor="black",
    linewidth=1.5,
    alpha=0.85,
    width=0.5,
)
for bar, acc in zip(bars, accs, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{acc:.0%}",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
ax.set_ylabel("Eval accuracy on Task 2", fontsize=11)
ax.set_title("Final eval accuracy by method", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
