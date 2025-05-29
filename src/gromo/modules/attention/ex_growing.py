import time
import matplotlib.pyplot as plt
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
start_file = time.time()

# Hyperparams dataset
DATA_PATH = "src/gromo/modules/attention/transf_teacher.pt"
test_ratio = 0.2

# Hyperparams training ----------------------------
num_epochs = 2
methods = ["kro", "big_in"]
lbds = [i for i in torch.linspace(-1e8, 1e8, 16 + 1).tolist()]
train_batch_size = 64
stats_batch_size = 64
lr = 1e-3

armijo_method = "kro"
alpha_armijo = 0.1
beta_armijo = 0.5
plot_armijo = True

# config = ModelConfig(
#     d_k=4,
#     d_k_max=8,
#     d_v=8,
#     d_e=16,
#     d_s=32,
#     bias_attention=False,
# )
config = ModelConfig(
    d_k=8,
    d_k_max=16,
    d_v=16,
    d_e=32,
    d_s=64,
    bias_attention=False,
)
# config = ModelConfig(
#     d_k=32,
#     d_k_max=64,
#     d_v=64,
#     d_e=128,
#     d_s=128 * 2,
#     bias_attention=False,
# )

# Dataset generation ---------------------
if not os.path.exists(DATA_PATH) or True:
    generate_teacher_dataset(Block, config, DATA_PATH, device)
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}."
dataset = AttentionDataset(DATA_PATH)
test_obs_size = int(test_ratio * len(dataset))
train_obs_size = len(dataset) - test_obs_size  # Number of observations in the train set
train_ds, test_ds = random_split(dataset, [train_obs_size, test_obs_size])

# Data loaders ---------------------
train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
stats_loader = DataLoader(train_ds, batch_size=stats_batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=train_batch_size)

# Model ---------------------------
model = Block(config).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


def loss_SVD(lbd, config, model, choice_P_stat: str, xb, loss_fn):
    model.update_WQ_WK(config, lbd, choice_P_stat)
    y_pred = model.forward(xb)
    return loss_fn(y_pred, yb)


def derivative(f, x, h=1e-5, method="central"):
    """
    Approximate the derivative of a function f at point x.

    Parameters
    ----------
    f : callable
        The function to differentiate. Must accept a float and return a float.
    x : float
        The point at which to evaluate the derivative.
    h : float, optional
        The step size. Smaller values give more accuracy but may introduce numerical errors. Default is 1e-5.
    method : {'forward', 'backward', 'central'}, optional
        The finite difference method to use:
        - 'forward': (f(x + h) - f(x)) / h
        - 'backward': (f(x) - f(x - h)) / h
        - 'central': (f(x + h) - f(x - h)) / (2*h) (default)

    Returns
    -------
    float
        The approximate derivative f'(x).
    """
    if method == "forward":
        return (f(x + h) - f(x)) / h
    elif method == "backward":
        return (f(x) - f(x - h)) / h
    elif method == "central":
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError("method must be 'forward', 'backward', or 'central'")


def armijo_line_search(
    phi_fn,
    config: ModelConfig,
    model: nn.Module,
    choice_P_stat: str,
    xb: torch.Tensor,
    loss_fn,
    alpha: float,
    beta: float,
    expand: bool = False,
    min_step: float = 1e-7,
    max_iters: int = 50,
    verbose: int = 0,
) -> tuple[float, float, float]:
    # Be careful, difference between xb input of the whole model,
    # and model.frozen_x which is just for the self attention block (just before the matmul with W_Q and W_K)
    def phi(t):
        return phi_fn(t, config, model, choice_P_stat, xb, loss_fn)

    phi0 = phi(0)  # TODO: Can be more efficient because already computed
    # TODO: Compare with the function no SVD

    # TODO: Make first eval work?
    # Scalar product <S_grad, X @ P_stat @ X.T >
    # S_grad and X are 3D, P_stat is 2D, so we get a scalar product for each element of
    # the minibatch, then average them
    # tmp = model.frozen_x.transpose(-2, -1) @ model.frozen_S_grad @ model.frozen_x
    # inner = torch.sum(tmp * model.P_stat[choice_P_stat] * (-1), dim=(-2, -1))
    # dphi0_2 = inner.mean().item()
    # print(f"dphi0_2 no norm: \t{dphi0_2}")
    # normalize = (
    #     model.frozen_x @ model.P_stat[choice_P_stat] @ model.frozen_x.transpose(-2, -1)
    # )
    # norms_per_batch = torch.linalg.norm(normalize, ord="fro", dim=(-2, -1)).mean().item()
    # dphi0_2 /= norms_per_batch
    # print(f"dphi0_2: \t{dphi0_2}")

    dphi0 = derivative(phi, 0, h=1e7, method="central").item()
    # print(f"dphi0: \t{dphi0}")

    if dphi0 < 0:
        t = -2 * phi0 / dphi0
    else:
        print(
            "Warning: dphi0 is not negative, using 0 instead. Try using a higher h parameter for the derivative."
        )
        if verbose >= 2:
            print(f"dphi0: {dphi0}")
        return 0, dphi0, phi0
    if verbose >= 2:
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
        if verbose >= 1:
            print(f"Armijo expand: Ran {i} iterations")

    # 2) BACKTRACKING
    i = 0
    for _ in range(max_iters):
        i += 1
        if phi(t) <= phi0 + alpha * t * dphi0:
            if verbose >= 1:
                print(f"Armijo backtrack: Ran {i} iterations")
            return t, dphi0, phi0
        t *= beta
        if t < min_step:
            break
    if verbose >= 1:
        print(f"Armijo backtrack: Ran {i} iterations")
    print("Armijo got to max_iters")
    return t, dphi0, phi0


# lbd_min = None
train_losses = []
all_train_losses = {method: {} for method in methods}
running_all_losses = {method: {} for method in methods}
# lbds_min = []

for method in methods:
    for lbd in lbds:
        all_train_losses[method][lbd] = []

for epoch in range(1, num_epochs + 1):
    start_epoch = time.time()
    model.train()
    running_loss = 0.0
    for method in methods:
        for lbd in lbds:
            running_all_losses[method][lbd] = 0.0

    if epoch % 2 != 0:
        for xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            running_loss += loss.item() * xb.size(0)
            optimizer.step()
        # lbds_min.append(0.0)  # Just for epoch number consistency
    else:
        model.reinitialize_batch_stats()
        cpt = 0
        # First pass on the batches to accumulate the intermediate stats
        for xb, yb in stats_loader:
            cpt += 1
            optimizer.zero_grad()
            y_pred = model.forward(xb)  # Retain attention block input
            loss = loss_fn(y_pred, yb)
            loss.backward()  # Retain S_grad
            running_loss += loss.item() * xb.size(0)
            model.freeze_batch_input_and_grad()  # Freeze the input and gradient of S_grad
            model.accumulate_batch_stats(methods)

        # Computing the P_stat
        with torch.no_grad():
            model.average_batch_stats(cpt)
            model.compute_P_stat(methods)
            model.freeze_WQt_WKt()
            for method in methods:
                print(
                    f"Norm P {method}: \t{torch.linalg.norm(model.P_stat[method], ord='fro')}"
                )
            print(f"Norm S_grad: \t{torch.linalg.norm(model.grad_S_batch, ord='fro')}")

            # Second pass on the batches to test the forward
        for xb, yb in stats_loader:
            for method in methods:
                for lbd in lbds:
                    model.update_WQ_WK(config, lbd=lbd, choice_P_stat=method)
                    y_pred_search = model.forward(xb)
                    loss_search = loss_fn(y_pred_search, yb)
                    running_all_losses[method][lbd] += loss_search.item() * xb.size(0)

            # WARN: Line search not method agnostic
            # lbd_min, dphi0, phi0 = armijo_line_search(
            #     loss_SVD,
            #     config,
            #     model,
            #     choice_P_stat=armijo_method,
            #     xb=xb,
            #     loss_fn=loss_fn,
            #     alpha=alpha_armijo,
            #     beta=beta_armijo,
            #     expand=True,
            # )
        model.reset_layers_WQt_WKt(config)
        # lbds_min.append(lbd_min)

    end_epoch = time.time()

    running_loss /= train_obs_size
    print(
        f"Epoch {epoch}/{num_epochs}, Time {end_epoch - start_epoch:.2f}s, Loss: {running_loss}"
    )
    train_losses.append(running_loss)
    if epoch % 2 == 0:
        for method in methods:
            for lbd in lbds:
                running_all_losses[method][lbd] /= train_obs_size
                all_train_losses[method][lbd].append(running_all_losses[method][lbd])
        # print(f"Lbd*: \t{lbd_min:e}")
        # TODO: Comparaison S et lbd*dS

        # for method in methods:
        #     print(
        #         f"Norm (lbd * P {method}): \t{torch.linalg.norm(lbd_min * model.P_stat[method], ord='fro')}"
        #     )
        #
        # TODO: Change the frozen batch version to all version

        # print(
        #     f"Norm (lbd * S_grad): \t{torch.linalg.norm(lbd_min * model.frozen_S_grad, ord='fro')}"
        # )
        # print(f"Norm S: \t{torch.linalg.norm(model.S_batch, ord='fro')}")

end_file = time.time()
print(f"Time taken: {end_file - start_file:.2f} seconds")


def testalph(lbd, phi0, dphi0, alpha):
    return phi0 + lbd * dphi0 * alpha


plt.figure()
legend = []
for epoch in range(1, num_epochs + 1):
    if epoch % 2 == 0:
        for method in methods:
            plt.plot(
                lbds, [all_train_losses[method][lbd][epoch // 2 - 1] for lbd in lbds]
            )
            legend.append(f"Epoch {epoch}, {method}")

        # plt.axvline(lbds_min[epoch - 1], linestyle="--")
        # legend.append(f"lbd_min Epoch {epoch} {armijo_method}")

        # if plot_armijo:
        #     plt.plot(
        #         lbds,
        #         [
        #             testalph(lbd, li_epoch_phi0[epoch - 1], li_epoch_dphi0[epoch - 1], 1)
        #             for lbd in lbds
        #         ],
        #     )
        #     legend.append("phi0 + lbd * dphi0")
        #     plt.plot(
        #         lbds,
        #         [
        #             testalph(
        #                 lbd, li_epoch_phi0[epoch - 1], li_epoch_dphi0[epoch - 1], 1 / 2
        #             )
        #             for lbd in lbds
        #         ],
        #     )
        #     legend.append("phi0 + 1/2 * lbd * dphi0")
        #     plt.plot(
        #         lbds,
        #         [
        #             testalph(
        #                 lbd,
        #                 li_epoch_phi0[epoch - 1],
        #                 li_epoch_dphi0[epoch - 1],
        #                 alpha_armijo,
        #             )
        #             for lbd in lbds
        #         ],
        #     )
        #     legend.append("phi0 + alpha * lbd * dphi0")
plt.yscale("log")
plt.xlabel("Lambda")
plt.ylabel("Loss")
plt.title("Train Loss(lambda)")
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
