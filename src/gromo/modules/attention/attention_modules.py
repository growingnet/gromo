import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

from gromo.modules.attention.my_utils import assert_2Dtensor_shape, my_svd_low_rank
from gromo.utils.utils import global_device


def generate_teacher_dataset(
    d_s, d_e, d_k, d_v, path, device, N_samples=5000, gen_batch=128
):
    """
    Generate a dataset using a teacher attention model.

    Parameters
    ----------
    d_s : int
        Sequence length of the input.
    d_e : int
        Embedding dimension of the input.
    d_k : int
        Dimension of the query and key vectors.
    d_v : int
        Dimension of the value vectors.
    path : str
        Path to save the generated dataset.
    device : torch.device
        Device to run the computations on.
    N_samples : int, optional
        Total number of samples to generate (default is 5000).
    gen_batch : int, optional
        Batch size for dataset generation (default is 128).

    Returns
    -------
    None
    """
    # Create and freeze the teacher model
    teacher = AttentionBaselineModule(d_e, d_k, d_v).to(device)
    teacher.eval()
    for z in teacher.parameters():
        z.requires_grad = False

    # Generate the dataset
    all_X, all_Y = [], []
    with torch.no_grad():
        for _ in range(0, N_samples, gen_batch):
            Xb = torch.randn(gen_batch, d_s, d_e, device=device)
            Yb = teacher.forward(Xb)
            all_X.append(Xb.cpu())
            all_Y.append(Yb.cpu())

    X = torch.cat(all_X, dim=0)
    Y = torch.cat(all_Y, dim=0)
    torch.save({"X": X, "Y": Y}, path)

    print(f"Saved dataset with {X.size(0)} samples to {path}")


class AttentionDataset(Dataset):
    """
    Dataset class to help load the data.
    """

    def __init__(self, path):
        data = torch.load(path)
        self.X, self.Y = data["X"], data["Y"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class AttentionBaselineModule(nn.Module):
    """
    A baseline implementation of self-attention mechanism.
    Used to create a teacher network to create a dataset,
    and to compare the growing model to a baseline.

    Attributes
    ----------
    W_Q : nn.Linear
        Linear layer to compute query vectors.
    W_K : nn.Linear
        Linear layer to compute key vectors.
    W_V : nn.Linear
        Linear layer to compute value vectors.
    W_O : nn.Linear
        Linear layer to compute the output representation.
    scale : float
        Scaling factor for the dot-product attention scores.
    """

    def __init__(self, d_e, d_k, d_v, use_bias=True):
        """
        Initialize the AttentionBaselineModule.

        Parameters
        ----------
        d_e : int
            Embedding dimension of the input.
        d_k : int
            Dimension of the query and key vectors.
        d_v : int
            Dimension of the value vectors.
        use_bias : bool, optional
            Whether to use bias in the linear layers (default is True).
        """
        super().__init__()
        self.W_Q = nn.Linear(d_e, d_k, bias=use_bias)
        self.W_K = nn.Linear(d_e, d_k, bias=use_bias)
        self.W_V = nn.Linear(d_e, d_v, bias=use_bias)
        self.W_O = nn.Linear(d_v, d_e, bias=use_bias)
        self.scale = d_k**0.5

    def forward(self, X):
        """
        Forward pass of the attention mechanism.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_e).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_e).
        """
        Q = self.W_Q(X)  # Compute query vectors
        K = self.W_K(X)  # Compute key vectors
        V = self.W_V(X)  # Compute value vectors
        S = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        )  # Compute scaled dot-product scores
        A = F.softmax(S, dim=-1)  # Apply softmax to get attention weights
        H = torch.matmul(A, V)  # Compute the weighted sum of values
        return X + self.W_O(
            H
        )  # Transform the result and return, with residual connection


class AttentionGrowingModule(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_k_initial: int,
        d_v: int,
        d_k_max: int | None = None,
        use_bias: bool = True,
        add_bias_before_pseudoinverse_calc: bool = True,
        # pre_attn: nn.Module | None = None,
        # post_attn: nn.Module | None = None,
    ) -> None:
        """Growing module for attention
        Currently only focusing on growing d_k, for one head
        Abbreviations: batch_size = b, sequence_length = d_s

            Parameters
            ----------
            d_e : int
                dimension of the input and output embedding
            d_k_initial : int
                initial dimension of the query and key, before any growing
            d_v : int
                dimension of the value
            d_k_max : int | None
                maximum dimension of the query and key
                the growing phases will be skipped if d_k_max is reached
            use_bias : bool
                use of bias
            pre_attn : nn.Module | None
                optional module to apply before attention
            post_attn : nn.Module | None
                optional module to apply after attention
        """
        super().__init__()
        self.d_e: int = d_e
        self.d_k: int = d_k_initial
        self.d_v: int = d_v
        self.scale = d_k_initial**0.5
        self.d_k_max = d_k_max
        # self.pre_attn: nn.Module | None = pre_attn
        # self.post_attn: nn.Module | None = post_attn
        self.use_bias: bool = use_bias
        self.add_bias_before_pseudoinverse_calc: bool = add_bias_before_pseudoinverse_calc

        self.W_Q = nn.Linear(d_e, d_k_initial, bias=self.use_bias)
        self.W_K = nn.Linear(d_e, d_k_initial, bias=self.use_bias)
        self.W_V = nn.Linear(d_e, d_v, bias=self.use_bias)
        self.W_O = nn.Linear(d_v, d_e, bias=self.use_bias)

    def save_S_grad(self, grad: torch.Tensor) -> None:
        """Hook to save the gradient of S."""
        self.S_grad = grad

    def get_S_grad(self) -> torch.Tensor:
        """Return the gradient of S (b, d_s, d_s) from the last backward pass"""
        assert self.S_grad is not None, (
            "S_grad is not available. Make sure to call forward() first."
        )
        return self.S_grad

    @staticmethod
    def _add_bias(X: torch.Tensor, add_column: bool = True) -> torch.Tensor:
        """Add a column or row of ones to the 3D tensor based on the argument.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, m, n)
        add_column : bool, optional
            If True, adds a column of ones. If False, adds a row of ones. Default is True.

        Returns
        -------
        torch.Tensor
            Augmented tensor of shape:
            - (batch_size, m, n + 1) if add_column is True
            - (batch_size, m + 1, n) if add_column is False
        """
        # TODO: Assert the input is a 3D tensor?
        if add_column:
            ones = torch.ones(X.size(0), X.size(1), 1, device=X.device, dtype=X.dtype)
            return torch.cat((X, ones), dim=-1)
        else:
            ones = torch.ones(X.size(0), 1, X.size(2), device=X.device, dtype=X.dtype)
            return torch.cat((X, ones), dim=1)

    def forward(self, X: torch.Tensor, pre_growing: bool = False) -> torch.Tensor:
        """Classical forward pass of the attention module
        During the growing iteration, this forward pass is used to get the S matrix and its gradient.
        Be careful when adding custom pre/post transforms, to not mess things up with the growing phase.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (b, d_s, d_e)
        pre_growing : bool, optional
            Set to False in the regular forward.
            Set to True if this is a forward used in a growing phase, to get the matrices S and the gradient of S.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (b, d_s, d_e)
        """
        # Optional pre-attention transform # WARN: pre and post-attention transforms would probably mess things up for the growing? like gradient calculation, other?
        # if self.pre_attn is not None:
        #     X = self.pre_attn(X)

        # if self.use_bias:  # WARN: Add bias after or before the pre-attention transform?
        #     X = self.add_bias(X)  # (b, d_s, d_e + 1)

        Q = self.W_Q(X)  # (b, d_s, d_k)
        K = self.W_K(X)  # (b, d_s, d_k)
        V = self.W_V(X)  # (b, d_s, d_v)

        if pre_growing:
            self.S = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (b, d_s, d_s)
            if self.S.requires_grad:
                self.S.register_hook(self.save_S_grad)  # Save the gradient of S
            A = F.softmax(self.S, dim=-1)  # (b, d_s, d_s)
        else:
            S = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (b, d_s, d_s)
            A = F.softmax(S, dim=-1)  # (b, d_s, d_s)

        H = torch.matmul(A, V)  # (b, d_s, d_v)
        Y = self.W_O(H)  # (b, d_s, d_e)

        # Optional post-attention transform # WARN: Same problem as pre-attention transform
        # if self.post_attn is not None:
        #     Y = self.post_attn(Y)

        return X + Y  # Residual connection

    def compute_split_breve(self) -> None:
        # TODO: Old implementation, to refactor
        assert self.breve_W_Q is not None, "breve_W_Q is None"
        assert self.breve_W_K is not None, "breve_W_K is None"

        self.W_Q_temp = self.breve_W_Q[:, : self.d_k]  # (d_e (+1), d_k)
        self.dW_Q = self.W_Q_temp - self.W_Q  # (d_e (+1), d_k)
        self.W_Q_new = self.breve_W_Q[:, self.d_k :]  # (d_e (+1), p)
        assert torch.equal(
            torch.cat((self.W_Q_temp, self.W_Q_new), dim=1), self.breve_W_Q
        ), (
            "Concatenation of W_Q_temp and W_Q_new does not match breve_W_Q"
        )  # OPTIMIZE: Put the assertion in a test?

        self.W_K_temp = self.breve_W_K[:, : self.d_k]  # (d_e (+1), d_k)
        self.dW_K = self.W_K_temp - self.W_K  # (d_e (+1), d_k)
        self.W_K_new = self.breve_W_K[:, self.d_k :]  # (d_e (+1), p)
        assert torch.equal(
            torch.cat((self.W_K_temp, self.W_K_new), dim=1), self.breve_W_K
        ), (
            "Concatenation of W_K_temp and W_K_new does not match breve_W_K"
        )  # OPTIMIZE: Put the assertion in a test?

    @staticmethod
    def _compute_reconstruction_error(
        A: torch.Tensor,
        B: torch.Tensor,
        Z: torch.Tensor,
        atol=1e-7,
        rtol=1e-6,
    ) -> None:
        """
        atol, rtol : float   tolerances for the optional consistency check
        rel_err : float   ‖A Bᵀ − Z‖_F / ‖Z‖_F
        """
        Z_hat = A @ B.T
        rel_err = torch.linalg.norm(Z_hat - Z) / torch.linalg.norm(Z)
        # optional consistency check (mainly to catch NaNs / infs)
        assert (
            torch.allclose(Z_hat, Z, atol=atol, rtol=rtol) or A.shape[1] < Z.shape[1]
        ), (
            "Exact reconstruction failed; either numerical issues "
            "or d_low is strictly less than rank(Z)."
        )
        return rel_err

    def grow_WQ_WK(
        self, X: torch.Tensor, p: int = 1, compute_reconstruction_error: bool = False
    ) -> None:
        """
        Not fully implemented yet. The goal is to compute update the old W_Q and W_K matrices, of shape (d_e, d_k), into new growed matrices W_Q and W_K, of shape (d_e, d_k + p).

        p: neuron to add

        Need to be called after a forward in the training loop, see the example.

        ```
        y_pred = model_growing.forward(xb, pre_growing=True)  # Creates self.S
        loss = loss_fn( y_pred, yb)
        loss.backward()  # Creates self.S_grad
        model_growing.grow_WQ_WK(xb, p=1)
        ```
        """
        # Compute the pseudoinverse of X
        # WARN: If a bias was used with X, as in W_Q, W_K, W_V has biases,
        # there could be a problem here. We need to compute Z = X^+ B (X^+)^T
        # So we probably need to take into account the bias of X.
        # There are two ways: Either we concatenate a column of ones to X, then compute the pseudoinverse,
        # Or the inverse where we compute the pseudoinverse, then concatenate to it a column of one.
        # The first way seems to be the good one, but not sure.
        # Implemented the two ways with a class variable to set the option wanted.

        if self.use_bias:
            if self.add_bias_before_pseudoinverse_calc:
                X_bias = self._add_bias(X)  # (b, d_s, d_e + 1)
                X_pinv = torch.linalg.pinv(X_bias)  # (b, d_e +1, d_s)
            else:
                X_pinv = torch.linalg.pinv(X)  # (b, d_e, d_s)
                X_pinv = self._add_bias(X_pinv, add_column=False)  # (b, d_e +1, d_s)
        else:
            X_pinv = torch.linalg.pinv(X)  # (b, d_e, d_s)

        assert self.S is not None, (
            "S is not available. Make sure to call forward() first."
        )
        assert self.S_grad is not None, (
            "S_grad is not available. Make sure to call forward() first."
        )

        Z = self.S_grad + self.S * self.scale  # (b, d_s, d_s)
        Z = torch.matmul(X_pinv, Z)  # (b, d_e (+1), d_s)
        Z = torch.matmul(Z, X_pinv.transpose(-2, -1))  # (b, d_e (+1), d_e (+1))
        Z_mean: torch.Tensor = Z.mean(dim=0)  # (d_e (+1), d_e (+1))
        Z_mean = Z_mean.detach()

        breve_W_Q, breve_W_K = my_svd_low_rank(Z_mean, self.d_k + p)

        if self.use_bias:  # OPTIMIZE: Put asserts in a test instead of here?
            assert_2Dtensor_shape(breve_W_Q, self.d_e + 1, self.d_k + p)
            assert_2Dtensor_shape(breve_W_K, self.d_e + 1, self.d_k + p)
        else:
            assert_2Dtensor_shape(breve_W_Q, self.d_e, self.d_k + p)
            assert_2Dtensor_shape(breve_W_K, self.d_e, self.d_k + p)

        if compute_reconstruction_error:
            self.reconstruction_error = self._compute_reconstruction_error(
                breve_W_Q, breve_W_K, Z_mean
            )

        self.breve_W_Q = breve_W_Q
        self.breve_W_K = breve_W_K

    @staticmethod
    def _tensor_to_linear(
        mat: torch.Tensor, in_features: int, bias_flag: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Convert a (d_e(+1), d_k_new) tensor coming from SVD into weight / bias
        compatible with nn.Linear(in_features=d_e, out_features=d_k_new).
        Caution! We mainly used for matrices the shape (in_features, out_feautres),
        but pytorch stores the weights in the linear layer in the shape (out_features, in_features).

        If `bias_flag` is True, the last row of `mat` is treated as the bias
            mat = [ w^T
                    b    ]  shape (d_e+1, d_k_new)
            with w of shape (out_features, in_features) the weights in a format
            expected by pytorch.

        Returns
        -------
        weight :  (d_k_new, d_e)
        bias   :  (d_k_new,)  or  None
        """
        if bias_flag:
            weight = mat[:-1, :].T.contiguous()  # (out, in)
            bias = mat[-1, :].contiguous()  # (out,)
        else:
            weight = mat.T.contiguous()
            bias = None
        assert weight.shape == (mat.shape[1], in_features)
        return weight, bias

    def update_WQ_WK(self, optimizer: torch.optim.Optimizer, p: int = 1):
        """
        Convert `self.breve_W_(Q or K)` into proper nn.Linear layers of size
        (d_k + p) and replace `self.W_Q`, `self.W_K` in-place.

        Call after `grow_WQ_WK()`.
        Pass the current optimizer so its parameter list is refreshed.
        """
        # --- Sanity checks
        assert hasattr(self, "breve_W_Q") and hasattr(self, "breve_W_K"), (
            "Call grow_WQ_WK() before update_WQ_WK()."
        )
        d_k_new = self.d_k + p
        device = self.W_Q.weight.device
        dtype = self.W_Q.weight.dtype

        # --- Slice the augmented matrices into weight and bias
        WQ_w, WQ_b = self._tensor_to_linear(self.breve_W_Q, self.d_e, self.use_bias)
        WK_w, WK_b = self._tensor_to_linear(self.breve_W_K, self.d_e, self.use_bias)

        # --- Build fresh nn.Linear layers
        new_W_Q = nn.Linear(
            self.d_e, d_k_new, bias=self.use_bias, device=device, dtype=dtype
        )
        new_W_K = nn.Linear(
            self.d_e, d_k_new, bias=self.use_bias, device=device, dtype=dtype
        )

        with torch.no_grad():  # copy the numbers
            new_W_Q.weight.copy_(WQ_w)
            new_W_K.weight.copy_(WK_w)
            if self.use_bias:
                assert WQ_b is not None
                assert WK_b is not None
                new_W_Q.bias.copy_(WQ_b)
                new_W_K.bias.copy_(WK_b)

        # --- Swap the modules in the enclosing nn.Module
        #     (this should automatically re-registers the new parameters)
        self.W_Q = new_W_Q
        self.W_K = new_W_K

        # Keep class variables consistent
        self.d_k = d_k_new
        self.scale = d_k_new**0.5

        # --- Refresh the optimizer so it sees the new parameters
        # Rebuild the optimizer with identical hyper-params
        # TODO: Check if this is right
        if optimizer is not None:
            # cls = type(optimizer) # Saves the class of the optimizer

            # Grab the hyper-params (lr, betas, momentum, ...) stored in the
            # optimizer's `defaults` dict
            defaults = optimizer.defaults

            # Erase every attribute of the optimizer instance, including
            # `param_groups`, `state`, `defaults`, ...
            optimizer.__dict__.clear()

            # Call the optimiser’s own constructor again, passing the new model
            # parameters and the hyper-params
            optimizer.__init__(self.parameters(), **defaults)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = global_device()
    print(f"Device: {device}")

    use_bias = True
    add_bias_before_pseudoinverse_calc = True
    d_s = 4
    d_e = 16
    d_k_grow = 2
    d_k_teacher = 8
    p = 2
    d_v = 8
    batch_size = 64

    test_ratio = 0.2
    train_batch = 64
    lr = 1e-3
    num_epochs = 50

    FILEPATH_DATASET = (
        "src/gromo/modules/attention/attention_dataset.pt"  # Path to the dataset file
    )

    if not os.path.exists(FILEPATH_DATASET):
        generate_teacher_dataset(d_s, d_e, d_k_teacher, d_v, FILEPATH_DATASET, device)

    assert os.path.exists(FILEPATH_DATASET), f"Dataset not found at {FILEPATH_DATASET}."

    dataset = AttentionDataset(FILEPATH_DATASET)
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=train_batch)

    model_growing = AttentionGrowingModule(
        d_e,
        d_k_grow,
        d_v,
        use_bias=use_bias,
        add_bias_before_pseudoinverse_calc=add_bias_before_pseudoinverse_calc,
    ).to(device)

    optimizer_growing = torch.optim.SGD(model_growing.parameters(), lr=lr)

    loss_fn = nn.MSELoss()
    train_losses_growing, test_losses_growing = [], []

    for epoch in range(1, num_epochs + 1):
        # Train
        model_growing.train()
        running_train = 0.0
        flag_first_batch_of_epoch = True

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer_growing.zero_grad()
            if (
                (epoch != 1)
                and (epoch % 10 == 0)
                and (flag_first_batch_of_epoch)
                and (
                    model_growing.d_k_max is None
                    or (
                        model_growing.d_k_max is not None
                        and model_growing.d_k < model_growing.d_k_max
                    )
                )
            ):  # Growing iteration case
                # Grow p=1 every first batch of every 10 epoch, except for the first epoch

                y_pred = model_growing.forward(xb, pre_growing=True)  # Creates self.S
                loss = loss_fn(y_pred, yb)
                loss.backward()  # Creates self.S_grad
                model_growing.grow_WQ_WK(xb, p=p, compute_reconstruction_error=True)
                # print(f"rec_error: {model_growing.reconstruction_error}")
                model_growing.update_WQ_WK(optimizer=optimizer_growing, p=p)
                # print(model_growing.W_Q)

                optimizer_growing.zero_grad()  # WARN: Useless?

            else:  # Regular training iteration case
                y_pred = model_growing.forward(xb)
                loss = loss_fn(y_pred, yb)
                loss.backward()

                optimizer_growing.step()
                running_train += (
                    loss.item() * xb.size(0)
                )  # WARN: Need to do a running loss also with the growing iteration or not?

            flag_first_batch_of_epoch = False

        epoch_train_loss = running_train / train_size
        train_losses_growing.append(epoch_train_loss)

        # Test
        model_growing.eval()
        running_test = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model_growing(xb)
                running_test += loss_fn(y_pred, yb).item() * xb.size(0)
        epoch_test_loss = running_test / test_size
        test_losses_growing.append(epoch_test_loss)

        if epoch % 10 == 0:
            print(
                f"Growing Epoch {epoch}/{num_epochs} — "
                f"Growing Train Loss: {epoch_train_loss:.6f}, "
                f"Growing Test Loss: {epoch_test_loss:.6f}"
            )
