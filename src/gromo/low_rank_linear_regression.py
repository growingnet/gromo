import tensorly as tl
import torch


tl.set_backend("pytorch")

try:
    from src.gromo.tools import compute_optimal_added_parameters
except ImportError:
    from gromo.tools import compute_optimal_added_parameters


def value(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    Compute the value of the objective function

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    y : torch.Tensor
        Vector of shape (n, d)
    a : torch.Tensor
        Matrix of shape (p, k)
    b : torch.Tensor
        Matrix of shape (d, k)

    Returns
    -------
    torch.Tensor
        Value of the objective function
    """
    return torch.norm(y - x @ a @ b)


def solve_tiny(
    x: torch.Tensor, y: torch.Tensor, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Goal of the function is to solve the following optimization problem:
    argmin_{a, b} ||y - x @ a @ b||_2
    with (a[:k], b[:k]) optimal solution of the problem for k

    Use the solution form the paper

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    y : torch.Tensor
        Vector of shape (n, d)
    k : int
        Number of components to keep

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the optimal solutions (a, b)
    """
    tensor_s = torch.einsum("ij,ik->jk", x, x)
    tensor_n = torch.einsum("ij,ik->kj", y, x)
    alpha, omega, _ = compute_optimal_added_parameters(
        tensor_s,
        tensor_n,
        maximum_added_neurons=k,
        numerical_threshold=1e-15,
        statistical_threshold=1e-15,
    )
    return alpha, omega


def solve_tiny_normalized(
    x: torch.Tensor, y: torch.Tensor, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Goal of the function is to solve the following optimization problem:
    argmin_{a, b} ||y - x @ a @ b||_2
    with (a[:k], b[:k]) optimal solution of the problem for k

    Use the solution form the paper

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    y : torch.Tensor
        Vector of shape (n, d)
    k : int
        Number of components to keep

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the optimal solutions (a, b)
    """
    tensor_s = torch.einsum("ij,ik->jk", x, x) / x.shape[0]
    tensor_n = torch.einsum("ij,ik->kj", y, x) / x.shape[0]
    alpha, omega, _ = compute_optimal_added_parameters(
        tensor_s,
        tensor_n,
        maximum_added_neurons=k,
        numerical_threshold=1e-15,
        statistical_threshold=1e-15,
    )
    return alpha, omega


def nih_w_handling(w: torch.Tensor, r: int, k: int, option: int = 0) -> torch.Tensor:
    """
    Process the matrix w to get a matrix of r lines and of rank <= k

    Parameters
    ----------
    w : torch.Tensor
        Matrix of shape (p, d)
    r : int
        Number of lines to keep
    k : int
        Rank of the matrix to keep
    option : int
        which method to use

    Returns
    -------
    torch.Tensor
        Matrix of shape (r, d)
    """
    if option == 0:
        return svd_projection(w[:r, :], k)  # pho_s(w_r)
    elif option == 1:
        return w[:r, :]  # w_r
    elif option == 2:
        return svd_projection(w, k)[:r, :]  # pho_s(w)_r
    else:
        raise ValueError(f"Option {option} is not valid")


def solve_nih(
    x: torch.Tensor, y: torch.Tensor, k: int, option: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Goal of the function is to solve the following optimization problem:
    argmin_{a, b} ||y - x @ a @ b||_2
    with (a[:k], b[:k]) optimal solution of the problem for k

    Use the algorithm form literature

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    y : torch.Tensor
        Vector of shape (n, d)
    k : int
        Number of components to keep

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the optimal solutions (a, b)
    """
    n, p = x.shape
    _, d = y.shape
    assert _ == n, "x and y must have the same number of rows"
    m = min(n, p)

    zero = 1e-15
    u, diag, v = torch.svd(x)
    assert diag.shape == (m,), f"{diag.shape=}"
    assert u.shape == (n, m), f"{u.shape=}"
    assert v.shape == (p, m), f"{v.shape=}"

    r = (diag > zero).sum()
    diag[:r] = 1 / diag[:r]
    w = u.T @ y
    assert w.shape == (m, d), f"{w.shape=}"

    diag = diag[:r]
    assert diag.shape == (r,), f"{diag.shape=}"

    w = nih_w_handling(w, r, k, option)

    assert w.shape == (r, d), f"{w.shape=}"
    dw = torch.diag(diag) @ w
    assert dw.shape == (r, d), f"{dw.shape=}"
    theta = v[:, :r] @ dw
    assert theta.shape == (p, d), f"{theta.shape=}"

    u, s, v = torch.linalg.svd(theta, full_matrices=False)
    s = torch.sqrt(s[:k])
    return u[:, :k] * s, v[:k, :] * s.unsqueeze(1)


def svd_projection(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute the projection of x on the first k components

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    k : int
        Number of components to keep

    Returns
    -------
    torch.Tensor
        Matrix of shape (n, k)
    """
    u, s, v = torch.linalg.svd(x, full_matrices=False)
    return u[:, :k] @ (s[:k].unsqueeze(1) * v[:k, :])


def solve_svd_l2(
    x: torch.Tensor, y: torch.Tensor, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Goal of the function is to solve the following optimization problem:
    argmin_{a, b} ||y - x @ a @ b||_2
    with (a[:k], b[:k]) optimal solution of the problem for k

    Use SVD of (S^{-1}N)

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    y : torch.Tensor
        Vector of shape (n, d)
    k : int
        Number of components to keep

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the optimal solutions (a, b)
    """
    n, p = x.shape
    _, d = y.shape
    assert _ == n, "x and y must have the same number of rows"
    assert k <= min(
        p, d
    ), f"k must be less than or equal to min(p, d) but k={k} and min(p, d)={min(p, d)}"
    tensor_s = torch.einsum("ij,ik->jk", x, x)
    tensor_n = torch.einsum("ij,ik->kj", y, x)
    # torch.linalg.lstsq(tensor_s, tensor_n).solution == tensor_s.pinv() @ tensor_n
    u, s, v = torch.linalg.svd(
        torch.linalg.lstsq(tensor_s, tensor_n).solution, full_matrices=False
    )
    assert u.shape == (
        p,
        min(p, d),
    ), f"u must have the shape ({p}, {min(p, d)}) but has shape {u.shape}"
    assert v.shape == (
        min(d, p),
        d,
    ), f"v must have the shape ({min(d, p)}, {p}) but has shape {v.shape}"
    s = torch.sqrt(s[:k])
    return u[:, :k] * s, v[:k, :] * s.unsqueeze(1)


def solve_svd_l2_pinv(
    x: torch.Tensor, y: torch.Tensor, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Goal of the function is to solve the following optimization problem:
    argmin_{a, b} ||y - x @ a @ b||_2
    with (a[:k], b[:k]) optimal solution of the problem for k

    Use SVD of (S^{-1}N)

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    y : torch.Tensor
        Vector of shape (n, d)
    k : int
        Number of components to keep

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the optimal solutions (a, b)
    """
    n, p = x.shape
    _, d = y.shape
    assert _ == n, "x and y must have the same number of rows"
    assert k <= min(
        p, d
    ), f"k must be less than or equal to min(p, d) but k={k} and min(p, d)={min(p, d)}"
    tensor_s = torch.einsum("ij,ik->jk", x, x)
    tensor_n = torch.einsum("ij,ik->kj", y, x)
    # torch.linalg.lstsq(tensor_s, tensor_n).solution == tensor_s.pinv() @ tensor_n
    u, s, v = torch.linalg.svd(
        torch.linalg.pinv(tensor_s) @ tensor_n, full_matrices=False
    )
    assert u.shape == (
        p,
        min(p, d),
    ), f"u must have the shape ({p}, {min(p, d)}) but has shape {u.shape}"
    assert v.shape == (
        min(d, p),
        d,
    ), f"v must have the shape ({min(d, p)}, {p}) but has shape {v.shape}"
    s = torch.sqrt(s[:k])
    return u[:, :k] * s, v[:k, :] * s.unsqueeze(1)


def solve_svd_l2_tl(
    x: torch.Tensor, y: torch.Tensor, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Goal of the function is to solve the following optimization problem:
    argmin_{a, b} ||y - x @ a @ b||_2
    with (a[:k], b[:k]) optimal solution of the problem for k

    Use SVD of (S^{-1}N)

    Parameters
    ----------
    x : torch.Tensor
        Matrix of shape (n, p)
    y : torch.Tensor
        Vector of shape (n, d)
    k : int
        Number of components to keep

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing the optimal solutions (a, b)
    """
    n, p = x.shape
    _, d = y.shape
    assert _ == n, "x and y must have the same number of rows"
    assert k <= min(
        p, d
    ), f"k must be less than or equal to min(p, d) but k={k} and min(p, d)={min(p, d)}"
    tensor_s = torch.einsum("ij,ik->jk", x, x)
    tensor_n = torch.einsum("ij,ik->kj", y, x)

    u, s, v = tl.truncated_svd(
        torch.linalg.lstsq(tensor_s, tensor_n).solution, n_eigenvecs=k
    )
    assert u.shape == (
        p,
        k,
    ), f"u must have the shape ({p}, {k}) but has shape {u.shape}"
    assert v.shape == (
        k,
        d,
    ), f"v must have the shape ({k}, {p}) but has shape {v.shape}"
    s = torch.sqrt(s)
    return u * s, v * s.unsqueeze(1)
