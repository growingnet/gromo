import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from gromo.modules.attention.transf_baseline import Block, ModelConfig
from gromo.utils.utils import global_device


def generate_teacher_dataset(cfg, path, device, N_samples=5000, gen_batch=128):
    # Create and freeze the teacher model
    teacher = Block(cfg).to(device)
    teacher.eval()
    for z in teacher.parameters():
        z.requires_grad = False

    # Generate the dataset
    all_X, all_Y = [], []
    with torch.no_grad():
        for _ in range(0, N_samples, gen_batch):
            Xb = torch.randn(gen_batch, cfg.d_s, cfg.d_e, device=device)
            Yb = teacher.forward(Xb)
            all_X.append(Xb.cpu())
            all_Y.append(Yb.cpu())

    X = torch.cat(all_X, dim=0)
    Y = torch.cat(all_Y, dim=0)
    torch.save({"X": X, "Y": Y}, path)

    print(f"Saved dataset with {X.size(0)} samples to {path}")


class AttentionDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.X, self.Y = data["X"], data["Y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == "__main__":
    torch.manual_seed(0)
    device = global_device()
    print(f"Device:{device}")

    # Hyperparams dataset
    DATA_PATH = "src/gromo/modules/attention/transf_teacher.pt"
    test_ratio = 0.2
    train_batch = 64

    # Hyperparams training
    num_epochs = 10
    log_every_x_epochs = num_epochs // 10
    lr = 1e-3
    gammas = [round(g, 2) for g in np.linspace(0, 32000, 16 + 1)]

    config = ModelConfig(
        d_s=4,
        d_e=16,
        d_k=8,
        d_v=8,
        bias=False,
    )

    if not os.path.exists(DATA_PATH) or True:
        generate_teacher_dataset(config, DATA_PATH, device)
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
    gammas_train_losses = {gamma: [] for gamma in gammas}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_gammas_train_losses = {gamma: 0.0 for gamma in gammas}
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()  # Get S_grad

            # Eval train Loss(S + S_grad), on the training set
            # Eval done here because otherwise S_grad would be associated to the wrong data batch
            with torch.no_grad():
                for gamma in gammas:
                    assert gamma is not None

                    y_pred = model.forward(xb, gamma=gamma)  # Forward with (S + S_grad)
                    running_gammas_train_losses[gamma] += loss_fn(
                        y_pred, yb
                    ).item() * xb.size(0)

            optimizer.step()  # Weight update
            running_loss += loss.item() * xb.size(0)

        # For each gamma, accumulate the running batch loss into an epoch loss
        for gamma in gammas:
            running_gammas_train_losses[gamma] /= train_size
            gammas_train_losses[gamma].append(running_gammas_train_losses[gamma])

        epoch_train_loss = running_loss / train_size
        train_losses.append(epoch_train_loss)

        # WARN: Old code, wrong because the S_grad is not the one associated with the data batch
        # model.eval()
        # for gamma in gammas:
        #     assert gamma is not None
        #     running_test = 0.0
        #     with torch.no_grad():
        #         for xb, yb in test_loader:
        #             xb, yb = xb.to(device), yb.to(device)
        #             y_pred = model.forward(xb, gamma=gamma)
        #             running_test += loss_fn(y_pred, yb).item() * xb.size(0)
        #     epoch_test_loss = running_test / test_size
        #     gammas_test_losses[gamma].append(epoch_test_loss)

        if (epoch == 1) or (epoch % log_every_x_epochs == 0):
            print(
                f"Epoch {epoch}/{num_epochs} — Train Loss: \t{epoch_train_loss:.6f}, "
                # f"Baseline Test Loss: {epoch_test_loss:.6f}"
            )
            for gamma in gammas:
                print(
                    f"Train Loss (gamma={gamma:.2f}): \t{gammas_train_losses[gamma][-1]:.6f}"
                )

    if True:
        plt.figure()
        legend = []
        for epoch in range(1, num_epochs + 1):
            if (epoch == 1) or (epoch % log_every_x_epochs == 0):
                plt.plot(
                    gammas, [gammas_train_losses[gamma][epoch - 1] for gamma in gammas]
                )
                legend.append(f"Epoch {epoch}")
        plt.yscale("log")
        plt.xlabel("Gamma")
        plt.ylabel("Loss")
        plt.title("Train Loss (gamma)")
        plt.legend(legend)
        # plt.plot(range(1, num_epochs + 1), train_losses)
        # legend = ["Train"]
        # for gamma in gammas:
        #     plt.plot(range(1, num_epochs + 1), gammas_train_losses[gamma])
        #     legend.append("gamma=" + str(gamma))
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.yscale("log")
        # plt.legend(legend)
        # plt.title("Training loss")
        plt.show()
