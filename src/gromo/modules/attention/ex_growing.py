import matplotlib.pyplot as plt
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
lbds = [i for i in np.linspace(-1e7, 1e7, 16 + 1)]
num_epochs = 30
log_every_x_epochs = num_epochs // 10 if num_epochs > 10 else 1
train_batch = 64
lr = 1e-3

config = ModelConfig(
    d_s=16 * 4,
    d_e=16,
    d_k=2,
    d_k_max=8,
    d_v=8,
    bias_attention=False,
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


def armijo_line_search(
    phi_fn,
    config: ModelConfig,
    model: nn.Module,
    choice_P_stat: tuple,
    xb: torch.Tensor,
    loss_fn,
    expand: bool = False,
    alpha: float = 1e-4,
    beta: float = 0.5,
    min_step: float = 1e-7,
    max_iters: int = 50,
    verbose: int = 0,
) -> float:
    """
    Perform Armijo line search to find a step size that satisfies the Armijo condition.

    Parameters
    ----------
    phi_fn : callable
        Function that computes the objective value for a given step size.
    config : ModelConfig
        Model configuration object.
    model : nn.Module
        The neural network model.
    choice_P_stat : tuple
        Tuple specifying which P_stat to use.
    xb : torch.Tensor
        Input tensor for the WHOLE model.
    loss_fn : callable
        Loss function to evaluate.
    expand : bool, optional
        Whether to perform expansion before backtracking (default is False).
    alpha : float, optional
        Armijo condition parameter (default is 1e-4).
    beta : float, optional
        Step size reduction factor (default is 0.5).
    min_step : float, optional
        Minimum allowable step size (default is 1e-7).
    max_iters : int, optional
        Maximum number of iterations for line search (default is 50).
    verbose : int, optional
        Verbosity level (default is 0).

    Returns
    -------
    float
        Step size that satisfies the Armijo condition.
    """

    # Be careful, difference between xb input of the whole model,
    # and model.frozen_x which is just for the self attention block (just before the matmul with W_Q and W_K)
    def phi(t):
        return phi_fn(t, config, model, choice_P_stat, xb, loss_fn)

    phi0 = phi(0)  # TODO: Can be more efficient because already computed
    # TODO: Compare with the function no SVD

    # Scalar product <S_grad, X @ P_stat @ X.T >
    # S_grad and X are 3D, P_stat is 2D, so we get a scalar product for each element of
    # the minibatch, then average them
    tmp = model.frozen_x.transpose(-2, -1) @ model.frozen_S_grad @ model.frozen_x
    inner = torch.sum(tmp * model.P_stat[choice_P_stat] * (-1), dim=(-2, -1))
    dphi0 = inner.mean().item()

    if dphi0 < 0:
        t = -2 * phi0 / dphi0
    else:
        print("Warning: dphi0 is not negative, using 0 instead.")
        if verbose == 2:
            print(f"dphi0: {dphi0}")
        return 0
    if verbose == 2:
        print(f"t0: {t}")

    # EXPANSION (optional)
    i = 0
    if expand:
        i += 1
        for _ in range(max_iters):
            if phi(t) > phi0 + alpha * t * dphi0:
                break  # violation → bracket found
            t /= beta  # enlarge step
        t *= beta  # step back to last good value
        if verbose == 1:
            print(f"Armijo expand: Ran {i} iterations")

    # 2) BACKTRACKING
    i = 0
    for _ in range(max_iters):
        i += 1
        if phi(t) <= phi0 + alpha * t * dphi0:
            if verbose == 1:
                print(f"Armijo backtrack: Ran {i} iterations")
            return t
        t *= beta
        if t < min_step:
            break
    if verbose == 1:
        print(f"Armijo backtrack: Ran {i} iterations")
    return t


hist_li_lbd_min_epoch = []  # to plot hist
proportion_lbdmin_0 = []
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    running_ratio_norm_inside, running_ratio_norm_bigf = 0.0, 0.0
    running_lbds_train_losses = {lbd: 0.0 for lbd in lbds}
    running_lbds_train_losses_dif = {lbd: 0.0 for lbd in lbds}
    running_lbds_train_losses_dif2 = {lbd: 0.0 for lbd in lbds}
    running_lbd_min = 0.0
    hist_li_lbd_min_batch = []
    nb_lbdmin_0 = 0
    nb_lbdmin_not0 = 0

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
                lbd_min = armijo_line_search(
                    loss_SVD,
                    config,
                    model,
                    choice_P_stat=("big_f", "in_e"),
                    xb=xb,
                    loss_fn=loss_fn,
                )
                if int(lbd_min) == 0:
                    nb_lbdmin_0 += 1
                else:
                    nb_lbdmin_not0 += 1
                hist_li_lbd_min_batch.append(lbd_min)
                running_lbd_min += lbd_min * xb.size(0)

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

    hist_li_lbd_min_epoch.append(hist_li_lbd_min_batch)
    if epoch % 2 == 0:
        proportion_lbdmin_0.append(nb_lbdmin_0 / (nb_lbdmin_not0 + nb_lbdmin_0))

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
            # print(f"lbd: {lbd}, Loss: {running_lbds_train_losses[lbd]}")
            if test_stat_formula:
                running_lbds_train_losses_dif[lbd] /= train_size
                running_lbds_train_losses_dif2[lbd] /= train_size
        print(f"Ratio Norm Inside smallf/bigf: \t{epoch_ratio_norm_inside:.4f}")
        print(f"Ratio Norm Bigf out_e/in_e: \t{epoch_ratio_norm_bigf:.4f}")
        print(f"Opti: {epoch_lbd_min:.4f}")
        # TODO: comparaison de taille S et lbd * dS
        # print(f"lbd * P {epoch_lbd_min * model.P_stat[('big_f', 'in_e')]}")
        # print(f"S {model.frozen_S}")
        # print(f"lbd * S_grad {epoch_lbd_min * model.frozen_S_grad}")
print(f"Proportion lbd_min = 0 by epoch: {proportion_lbdmin_0}")
# print(f"Average for all epochs of proportion lbd_min = 0: {np.mean(proportion_lbdmin_0)}")

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

# plt.hist(
#     hist_li_lbd_min_epoch,
#     # bins=30,
#     stacked=True,
#     # color=colors,
#     # label=["iter 1", "iter 2"],
#     edgecolor="black",
# )
# plt.xlabel("x")
# plt.ylabel("count")
# plt.show()


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
