import torch


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """Computes the linear kernel matrix: K = X X^T

    Parameters
    ----------
    X : torch.Tensor
        input tensor X

    Returns
    -------
    torch.Tensor
        linear kernel of X
    """
    return X @ X.T


def gaussian_kernel(X: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Computes the gaussian kernel matrix

    Parameters
    ----------
    X : torch.Tensor
        input tensor X
    sigma : float | None, optional
        standard deviation, is None the median heuristic is used, by default None

    Returns
    -------
    torch.Tensor
        gaussin kernel of X
    """
    dist = torch.cdist(X, X) ** 2
    if sigma is None:
        var = torch.median(dist) + 1e-7
    else:
        var = sigma**2
    return torch.exp(-dist / (2 * var))


def center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
    """Centers the kernel matrix using the centering matrix H

    Parameters
    ----------
    K : torch.Tensor
        input tensor

    Returns
    -------
    torch.Tensor
        centered matrix
    """
    n = K.shape[0]  # Number of samples
    H = torch.eye(n, device=K.device) - (1 / n) * torch.ones(
        (n, n), device=K.device
    )  # Centering matrix
    K_centered = H @ K @ H  # Centered kernel matrix
    return K_centered


def HSIC(K_centered: torch.Tensor, L_centered: torch.Tensor) -> torch.Tensor:
    """Computes the Hilbert-Schmidt idependece Criterion (HSIC) between two centered kernel matrices

    Parameters
    ----------
    K_centered : torch.Tensor
        centered kernel matrix K
    L_centered : torch.Tensor
        cetnered kernel matrix L

    Returns
    -------
    torch.Tensor
        HSIC estimation
    """
    n = K_centered.shape[0]
    return torch.trace(K_centered @ L_centered) / ((n - 1) ** 2)


def calculate_dependency(
    X_inputs: dict[str, torch.Tensor],
    Y: torch.Tensor,
    n_samples: int,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """Calculate Hilbert-Schmidt non-linear dependency between multiple input tensors and a target output

    Parameters
    ----------
    X_inputs : dict[str, torch.Tensor]
        input tensors
    Y : torch.Tensor
        target tensor
    n_samples : int
        maximum number of samples to use for estimation
    normalize : bool, optional
        normalize the criterion, by default True

    Returns
    -------
    dict[str, torch.Tensor]
        HSIC between every input and the target
    """
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
