import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os

import torch
import torch.nn as nn
import numpy as np
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
test_stat_formula = False
tol_minimize = 1e-6
lbds = [i for i in np.linspace(0, 1e7, 16 + 1)]
num_epochs = 4
log_every_x_epochs = num_epochs // 10 if num_epochs > 10 else 1
train_batch = 64
lr = 1e-3

config = ModelConfig(
    d_s=16 * 4,
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
lbds_epoch_losses = []
lbds_epoch_losses_dif = []
lbds_epoch_losses_dif2 = []
lbds_min = []


def loss_SVD(lbd, config, model, choice_P_stat, xb, loss_fn):
    model.update_WQ_WK(config, lbd, choice_P_stat)
    y_pred = model.forward(xb)
    return loss_fn(y_pred, yb)


for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    running_ratio_norm_inside, running_ratio_norm_bigf = 0.0, 0.0
    running_lbds_train_losses = {lbd: 0.0 for lbd in lbds}
    running_lbds_train_losses_dif = {lbd: 0.0 for lbd in lbds}
    running_lbds_train_losses_dif2 = {lbd: 0.0 for lbd in lbds}
    running_lbd_min = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model.forward(xb)  # Retain the attention block input
        loss = loss_fn(y_pred, yb)
        loss.backward()  # Retain S_grad
        model.freeze_input_and_grad()  # Freeze the input and gradient of S_grad

        if epoch % 2 == 0:
            with torch.no_grad():
                model.compute_statistics()
                temp_ratio_norm_inside, temp_ratio_norm_bigf = model.get_P_ratios()
                running_ratio_norm_inside += temp_ratio_norm_inside.item() * xb.size(0)
                running_ratio_norm_bigf += temp_ratio_norm_bigf.item() * xb.size(0)

                # lbd search
                model.freeze_WQt_WKt()
                for lbd in lbds:
                    model.update_WQ_WK(
                        config, lbd=lbd, choice_P_stat=("small_f", "out_e")
                    )
                    y_pred_search = model.forward(xb)
                    loss_search = loss_fn(y_pred_search, yb)
                    running_lbds_train_losses[lbd] += loss_search.item() * xb.size(0)
                lbd_min = minimize_scalar(
                    loss_SVD,
                    args=(config, model, ("small_f", "out_e"), xb, loss_fn),
                    options={
                        "xtol": tol_minimize,
                        # "maxiter": 1e10,
                    },
                )
                running_lbd_min += lbd_min.x * xb.size(0)
                if test_stat_formula:
                    for lbd in lbds:
                        model.update_WQ_WK(
                            config, lbd=lbd, choice_P_stat=("small_f", "in_e"), dif=False
                        )
                        y_pred_search = model.forward(xb)
                        loss_search = loss_fn(y_pred_search, yb)
                        running_lbds_train_losses_dif[lbd] += (
                            loss_search.item() * xb.size(0)
                        )
                    for lbd in lbds:
                        model.update_WQ_WK(
                            config, lbd=lbd, choice_P_stat=("big_f", "in_e"), dif=False
                        )
                        y_pred_search = model.forward(xb)
                        loss_search = loss_fn(y_pred_search, yb)
                        running_lbds_train_losses_dif2[lbd] += (
                            loss_search.item() * xb.size(0)
                        )
                model.reset_layers_WQt_WKt(config)

        else:
            optimizer.step()

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / train_size
    epoch_ratio_norm_inside = running_ratio_norm_inside / train_size
    epoch_ratio_norm_bigf = running_ratio_norm_bigf / train_size
    epoch_lbd_min = running_lbd_min / train_size

    print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss}")
    train_losses.append(epoch_loss)
    lbds_epoch_losses.append(running_lbds_train_losses)
    lbds_min.append(epoch_lbd_min)
    if test_stat_formula:
        lbds_epoch_losses_dif.append(running_lbds_train_losses_dif)
        lbds_epoch_losses_dif2.append(running_lbds_train_losses_dif2)

    if epoch % 2 == 0:
        for lbd in lbds:
            running_lbds_train_losses[lbd] /= train_size
            print(f"lbd: {lbd}, Loss: {running_lbds_train_losses[lbd]}")
            if test_stat_formula:
                running_lbds_train_losses_dif[lbd] /= train_size
                running_lbds_train_losses_dif2[lbd] /= train_size
        print(f"Ratio Norm Inside smallf/bigf: \t{epoch_ratio_norm_inside:.4f}")
        print(f"Ratio Norm Bigf out_e/in_e: \t{epoch_ratio_norm_bigf:.4f}")
        print(f"Opti: {epoch_lbd_min:.4f}")

plt.figure()
legend = []
for epoch in range(1, num_epochs + 1):
    if epoch % 2 == 0:
        plt.plot(lbds, [lbds_epoch_losses[epoch - 1][lbd] for lbd in lbds])
        legend.append(f"Epoch {epoch} small_f/big_g, out_e")
        plt.axvline(lbds_min[epoch - 1], linestyle="--", label=f"Opti Epoch {epoch}")
        legend.append(f"Lbd min Epoch {epoch}")

        if test_stat_formula:
            plt.plot(lbds, [lbds_epoch_losses_dif[epoch - 1][lbd] for lbd in lbds])
            legend.append(f"Epoch {epoch} small_f, in_e")
            plt.plot(lbds, [lbds_epoch_losses_dif2[epoch - 1][lbd] for lbd in lbds])
            legend.append(f"Epoch {epoch} big_f, in_e")
plt.yscale("log")
plt.xlabel("Lambda")
plt.ylabel("Loss")
plt.title("Train Loss (lambda)")
plt.legend(legend)
plt.show()


# Evaluate on test set
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
