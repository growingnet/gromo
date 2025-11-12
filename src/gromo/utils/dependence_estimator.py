import torch


def linear_kernel(X):
    """Computes the linear kernel matrix: K = X X^T"""
    return X @ X.T


def gaussian_kernel(X, sigma=None):
    """Computes the gaussian kernel matrix"""
    dist = torch.cdist(X, X) ** 2
    if sigma is None:
        var = torch.median(dist)
    else:
        var = sigma**2
    return torch.exp(-dist / (2 * var))


def slow_gaussian_kernel(X, sigma_sq=None):
    pairwise_sq_dists = torch.sum((X[:, None] - X) ** 2, axis=-1)  # (n,n)

    if sigma_sq is None:
        sigma_sq = torch.median(pairwise_sq_dists)

    K = torch.exp(-pairwise_sq_dists / (2 * sigma_sq))
    return K


def center_kernel_matrix(K):
    """Centers the kernel matrix using the centering matrix H."""
    n = K.shape[0]  # Number of samples
    H = torch.eye(n, device=K.device) - (1 / n) * torch.ones(
        (n, n), device=K.device
    )  # Centering matrix
    K_centered = H @ K @ H  # Centered kernel matrix
    return K_centered


def HSIC(K_centered, L_centered):
    """Computes the HSIC between two centered kernel matrices."""
    n = K_centered.shape[0]
    return torch.trace(K_centered @ L_centered) / ((n - 1) ** 2)


def calculate_dependency(
    X_inputs: dict, Y: torch.Tensor, n_samples: int, normalize: bool = True
) -> dict:
    indices = torch.randperm(len(Y))[:n_samples]

    Y_matrix = Y[indices]
    Y_matrix = gaussian_kernel(Y_matrix)
    Y_matrix = center_kernel_matrix(Y_matrix)
    hsicY = HSIC(Y_matrix, Y_matrix)

    hsic = {}

    for name, X_matrix in X_inputs.items():
        X_matrix = X_matrix[indices]
        X_matrix = gaussian_kernel(X_matrix)

        X_matrix = center_kernel_matrix(X_matrix)
        hsicX = HSIC(X_matrix, X_matrix)

        hsic[name] = HSIC(X_matrix, Y_matrix)
        if normalize:
            hsic[name] = hsic[name] / torch.sqrt(hsicX * hsicY)

    return hsic
