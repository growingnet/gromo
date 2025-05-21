import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from gromo.modules.attention.model import Block, ModelConfig
from gromo.modules.attention.my_utils import AttentionDataset, generate_teacher_dataset
from gromo.utils.utils import global_device


torch.manual_seed(0)
device = global_device()
print(f"Device:{device}")

# Hyperparams dataset
DATA_PATH = "src/gromo/modules/attention/transf_teacher.pt"
test_ratio = 0.2

# Hyperparams training
num_epochs = 10
log_every_x_epochs = num_epochs // 10 if num_epochs > 10 else 1
train_batch = 64
lr = 1e-3

config = ModelConfig(
    d_s=4,
    d_e=16,
    d_k=2,
    d_k_max=8,
    d_v=8,
    bias=False,
)

if not os.path.exists(DATA_PATH) or True:
    generate_teacher_dataset(Block, config, DATA_PATH, device)
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}."

dataset = AttentionDataset(DATA_PATH)
test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=train_batch)


model = Block(config).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
train_losses = []
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    fro_mean_outside, op_mean_outside = 0.0, 0.0
    fro_mean_inside, op_mean_inside = 0.0, 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model.forward(xb)
        loss = loss_fn(y_pred, yb)
        loss.backward()  # Get S_grad
        # TODO: Check ordre algo, notamment calcul S_grad pour bien verif pas bug

        growing = True
        if epoch != 1:
            with torch.no_grad():
                model.compute_statistics(outside_esp=False)
                fro_inside, op_inside = model.get_P_ratios()
                fro_mean_inside += fro_inside.item() * xb.size(0)
                op_mean_inside += op_inside.item() * xb.size(0)

                model.compute_statistics(outside_esp=True)
                fro_outside, op_outside = model.get_P_ratios()
                fro_mean_outside += fro_outside.item() * xb.size(0)
                op_mean_outside += op_outside.item() * xb.size(0)

                model.update_WQ_WK(config)
        else:
            optimizer.step()

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / train_size
    epoch_fro_outside = fro_mean_outside / train_size
    epoch_op_outside = op_mean_outside / train_size
    epoch_fro_inside = fro_mean_inside / train_size
    epoch_op_inside = op_mean_inside / train_size
    print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss}")
    print(f"Outside Fro: \t{epoch_fro_outside:.4f}, Op: \t{epoch_op_outside:.4f}")
    print(f"Inside Fro: \t{epoch_fro_inside:.4f}, Op: \t{epoch_op_inside:.4f}")
    train_losses.append(epoch_loss)


# Evaluate on test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        y_pred = model(xb)
        loss = loss_fn(y_pred, yb)
        test_loss += loss.item() * xb.size(0)
test_loss /= test_size
print(f"Test Loss: {test_loss:.4f}")


# model = Block(config).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# loss_fn = nn.MSELoss()
#
# train_losses = []
#
# for epoch in range(1, num_epochs + 1):
#     model.train()
#     running_loss = 0.0
#
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         y_pred = model(xb)
#         loss = loss_fn(y_pred, yb)
#         loss.backward()  # Get S_grad
#         optimizer.step()
#         running_loss += loss.item() * xb.size(0)
#
#     epoch_loss = running_loss / train_size
#     train_losses.append(epoch_loss)
#
#     if epoch % log_every_x_epochs == 0 or epoch == 1 or epoch == num_epochs:
#         print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
#
# # Evaluate on test set
# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for xb, yb in test_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         y_pred = model(xb)
#         loss = loss_fn(y_pred, yb)
#         test_loss += loss.item() * xb.size(0)
# test_loss /= test_size
# print(f"Test Loss: {test_loss:.4f}")
