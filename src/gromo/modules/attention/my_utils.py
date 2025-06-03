import torch
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal


# TODO: Explain why custom svd function is needed
def my_svd_low_rank(
    Z: torch.Tensor,
    d_low: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Factor a (d_high × d_high) matrix Z into tall skinny matrices A, B with inner
    dimension d_low  (d_low « d_high) so that   A  @  B.T  ≈  Z.

    Parameters
    ----------
    Z    : torch.Tensor  shape [d_high, d_high]  (can be non-symmetric, any real dtype)
    d_low  : int           target rank  (1 ≤ d_low ≤ d_high)
    Returns
    -------
    A : [d_high, d_low]
    B : [d_high, d_low]
    """
    assert Z.dim() == 2, "Z must be a 2-D matrix"
    d_high0, d_high1 = Z.shape
    assert d_high0 == d_high1, f"Z must be square (got {d_high0} × {d_high1})"
    assert 1 <= d_low <= d_high0, f"d_k must be between 1 and {d_high0}"
    assert torch.is_floating_point(Z), "Z must have a floating-point dtype"

    U, S, Vh = torch.linalg.svd(
        Z, full_matrices=False
    )  # U:[d_high,d_high], S:[d_high], Vh:[d_high,d_high]
    U_k = U[:, :d_low]  # [d_high, d_low]
    S_k = S[:d_low]  # [d_low]
    V_k = Vh[:d_low, :].T  # [d_high, d_low]

    sqrtS = S_k.sqrt()  # [d_low]

    A = U_k * sqrtS.unsqueeze(0)  # broadcast over rows
    B = V_k * sqrtS.unsqueeze(0)

    return A, B


def assert_2Dtensor_shape(tensor: torch.Tensor, row_dim: int, col_dim: int) -> None:
    expected_rows = row_dim
    expected_cols = col_dim
    assert tensor.shape == (expected_rows, expected_cols), (
        f"Tensor shape mismatch: expected ({expected_rows}, {expected_cols}), "
        f"got {tensor.shape}"
    )


def generate_teacher_dataset(
    teacher_model, cfg, path, device, N_samples=5000, gen_batch=128
):
    assert isinstance(cfg.d_s, int)
    # Create and freeze the teacher model
    teacher = teacher_model(cfg).to(device)
    teacher.eval()
    for z in teacher.parameters():
        z.requires_grad = False

    # Generate the dataset
    all_X, all_Y = [], []
    with torch.no_grad():
        for _ in range(0, N_samples, gen_batch):
            A = torch.randn(cfg.d_e, cfg.d_e, device=device)
            sigma = A @ A.T + 1e-3 * torch.eye(cfg.d_e, device=device)  # make it SPD

            # 2) create the distribution
            mvn = MultivariateNormal(
                loc=torch.zeros(cfg.d_e, device=device), covariance_matrix=sigma
            )

            # 3) draw batch of sequence vectors
            #    shape: (gen_batch, d_s, d_e)
            batch_shape = torch.Size((gen_batch, cfg.d_s))
            Xb = mvn.sample(batch_shape)

            # Xb = torch.randn(gen_batch, cfg.d_s, cfg.d_e, device=device)
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
