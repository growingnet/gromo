import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from gromo.utils.utils import global_device
from gromo.modules.attention.attention_modules import (
    AttentionBaselineModule,
    AttentionGrowingModule,
    AttentionDataset,
)

# TODO: Old file mostly obsolete, to refactor/delete a lot of stuff

MAKE_PLOT = True  # Plot the loss or not
FILEPATH_DATASET = (
    "src/gromo/modules/attention/attention_dataset.pt"  # Path to the dataset file
)

# --- Hyperparameters
torch.manual_seed(0)
device = global_device()

d_s = 4
d_e = 16
d_k_regular = 8
d_k_grow = 2
d_v = 8

train_batch = 64
num_epochs = 200
lr = 1e-3
test_ratio = 0.2

# --- Creation of a dataset by a teacher network if it doesn't exist yet
if not os.path.exists(FILEPATH_DATASET):
    N_samples = 5000
    gen_batch = 128

    # Create and freeze the teacher model ###
    teacher = AttentionBaselineModule(d_e, d_k_regular, d_v).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Generate the dataset
    all_X, all_Y = [], []
    with torch.no_grad():
        for _ in range(0, N_samples, gen_batch):
            Xb = torch.randn(gen_batch, d_s, d_e, device=device)
            Yb = teacher.forward(Xb)
            all_X.append(Xb.cpu())
            all_Y.append(Yb.cpu())

    X = torch.cat(all_X, dim=0)
    Y = torch.cat(all_Y, dim=0)
    torch.save({"X": X, "Y": Y}, FILEPATH_DATASET)

    print(f"Saved dataset with {X.size(0)} samples to {FILEPATH_DATASET}")


# --- Get the dataset
dataset = AttentionDataset(FILEPATH_DATASET)
test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=train_batch)

# --- Get the models
model_baseline = AttentionBaselineModule(d_e, d_k_regular, d_v).to(device)
model_growing = AttentionGrowingModule(
    d_e, d_k_grow, d_v, use_bias=True, add_bias_before_pseudoinverse_calc=True
).to(device)

optimizer_baseline = torch.optim.SGD(model_baseline.parameters(), lr=lr)
optimizer_growing = torch.optim.SGD(model_growing.parameters(), lr=lr)

loss_fn = nn.MSELoss()
train_losses_baseline, test_losses_baseline = [], []
train_losses_growing, test_losses_growing = [], []

# --- Training loop
# For the regular model:
for epoch in range(1, num_epochs + 1):
    # Train
    model_baseline.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer_baseline.zero_grad()
        y_pred = model_baseline.forward(xb)
        loss = loss_fn(y_pred, yb)
        loss.backward()
        optimizer_baseline.step()
        running_train += loss.item() * xb.size(0)
    epoch_train_loss = running_train / train_size
    train_losses_baseline.append(epoch_train_loss)

    # Test
    model_baseline.eval()
    running_test = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model_baseline.forward(xb)
            running_test += loss_fn(y_pred, yb).item() * xb.size(0)
    epoch_test_loss = running_test / test_size
    test_losses_baseline.append(epoch_test_loss)

    if epoch % 10 == 0:
        print(
            f"Baseline Epoch {epoch}/{num_epochs} — "
            f"Baseline Train Loss: {epoch_train_loss:.6f}, "
            f"Baseline Test Loss: {epoch_test_loss:.6f}"
        )

# For the growing model:


growing_iteration = False
# for epoch in range(1, num_epochs + 1):
#     # Train
#     model_growing.train()
#     running_train = 0.0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#
#         optimizer_growing.zero_grad()
#         if growing_iteration:
#             model_growing.update_weights(p=1)
#         else:
#             y_pred = model_growing.forward(xb)
#             loss = loss_fn(y_pred, yb)
#             loss.backward()
#
#             optimizer_growing.step()
#             running_train += loss.item() * xb.size(
#                 0
#             )
#
#         # Manual weight update
#         # with torch.no_grad():
#         #     for layer in model.modules():
#         #         if isinstance(layer, nn.Linear):
#         #             layer.weight += some_custom_update
#
#     epoch_train_loss = running_train / train_size
#     train_losses_growing.append(epoch_train_loss)
#
#     # Test
#     model_growing.eval()
#     running_test = 0.0
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             y_pred = model_growing.forward(xb)
#             running_test += loss_fn(y_pred, yb).item() * xb.size(0)
#     epoch_test_loss = running_test / test_size
#     test_losses_growing.append(epoch_test_loss)
#
#     if epoch % 10 == 0:
#         print(
#             f"Growing Epoch {epoch}/{num_epochs} — "
#             f"Growing Train Loss: {epoch_train_loss:.6f}, "
#             f"Growing Test Loss: {epoch_test_loss:.6f}"
#         )

# --- Plotting
if MAKE_PLOT:
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses_baseline)
    plt.plot(range(1, num_epochs + 1), test_losses_baseline)
    # plt.plot(range(1, num_epochs + 1), train_losses_growing)
    # plt.plot(range(1, num_epochs + 1), test_losses_growing)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    # plt.legend(["Train baseline", "Test baseline", "Train growing", "Test growing"])
    plt.legend(["Train baseline", "Test baseline"])
    plt.title("Training vs. Testing Loss")
    plt.show()
