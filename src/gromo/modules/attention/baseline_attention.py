import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Teacher and student attention netwoks
# ─── Attention Module ──────────────────────────────────────────────────────────


class BaselineAttention(nn.Module):
    def __init__(self, d_s, d_e, d_k, d_v, use_bias=True):
        super().__init__()
        self.scale = d_k**0.5
        self.W_Q = nn.Linear(d_e, d_k, bias=use_bias)
        self.W_K = nn.Linear(d_e, d_k, bias=use_bias)
        self.W_V = nn.Linear(d_e, d_v, bias=use_bias)
        self.W_O = nn.Linear(d_v, d_e, bias=use_bias)

    def forward(self, X):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        S = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        A = F.softmax(S, dim=-1)
        H = torch.matmul(A, V)
        return self.W_O(H)


# ─── 1) Hyperparameters & Device ──────────────────────────────────────────────

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_s, d_e, d_k, d_v = 4, 16, 8, 8
N_samples = 5000
gen_batch_sz = 128
train_batch_sz = 64
num_epochs = 100
lr = 1e-3
test_ratio = 0.2

# ─── 2) Instantiate & Freeze Teacher ──────────────────────────────────────────

teacher = BaselineAttention(d_s, d_e, d_k, d_v).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# ─── 3) Generate & Save Synthetic Dataset ────────────────────────────────────

all_X, all_Y = [], []
with torch.no_grad():
    for i in range(0, N_samples, gen_batch_sz):
        Xb = torch.randn(gen_batch_sz, d_s, d_e, device=device)
        # Yb = torch.sin(Xb) + 0.02 * torch.randn_like(Xb)
        # Yb = teacher(Xb) + 0.02 * torch.randn_like(Xb)
        Yb = teacher(Xb)
        all_X.append(Xb.cpu())
        all_Y.append(Yb.cpu())

X = torch.cat(all_X, dim=0)
Y = torch.cat(all_Y, dim=0)
torch.save({"X": X, "Y": Y}, "attention_dataset.pt")

# ─── 4) Dataset & Split ───────────────────────────────────────────────────────


class AttentionDataset(Dataset):
    def __init__(self, path="attention_dataset.pt"):
        data = torch.load(path)
        self.X, self.Y = data["X"], data["Y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset = AttentionDataset("attention_dataset.pt")
test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=train_batch_sz, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=train_batch_sz)

# ─── 5) Student & Training Setup ──────────────────────────────────────────────

student = BaselineAttention(d_s, d_e, d_k, d_v).to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=lr)
loss_fn = nn.MSELoss()

train_losses, test_losses = [], []

# ─── 6) Training Loop with Test Evaluation ───────────────────────────────────

for epoch in range(1, num_epochs + 1):
    # Training
    student.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = student(xb)
        loss = loss_fn(y_pred, yb)
        loss.backward()
        optimizer.step()
        running_train += loss.item() * xb.size(0)
    epoch_train_loss = running_train / train_size
    train_losses.append(epoch_train_loss)

    # Testing
    student.eval()
    running_test = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = student(xb)
            loss = loss_fn(y_pred, yb)
            running_test += loss.item() * xb.size(0)
    epoch_test_loss = running_test / test_size
    test_losses.append(epoch_test_loss)

    print(
        f"Epoch {epoch}/{num_epochs} — Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}"
    )

# ─── 7) Plot Train & Test Loss ────────────────────────────────────────────────

epochs = list(range(1, num_epochs + 1))
MAKE_PLOT = True
if MAKE_PLOT:
    plt.figure()
    plt.plot(epochs, train_losses)
    plt.plot(epochs, test_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend(["Train", "Test"])
    plt.title("Training vs. Testing Loss")
    plt.show()
