import torch

from gromo.modules.linear_growing_module import LinearGrowingModule


class ConstantModule(LinearGrowingModule):
    """Placeholder Module with constant zero output.
    Used to simulate zero connection between modules

    Parameters
    ----------
    in_features : int
        input features
    out_features : int
        output features
    device : torch.device | None, optional
        default device, by default None
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None
    ) -> None:
        super(ConstantModule, self).__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=False,
            device=device,
        )
        # Store the constant tensor as a buffer (non-trainable parameter)
        # self.register_buffer('constant', torch.zeros(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Placeholder function that ignores the input and always returns the constant tensor

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            constant tensor
        """
        # Ignore the input x and always return the constant tensor
        self.register_buffer(
            "constant",
            torch.zeros(
                len(x), self.out_features, requires_grad=True, device=self.device
            ),
        )
        return self.constant

    def __setattr__(self, key, value):
        if key == "optimal_delta_layer":
            torch.nn.Module.__setattr__(self, "_hidden_optimal_delta_layer", value)
        else:
            super().__setattr__(key, value)

    @property
    def optimal_delta_layer(self) -> torch.nn.Linear:
        """Placeholder function that returns a zero weight module as a mock update to the constant module

        Returns
        -------
        torch.nn.Linear
            zero-weight linear layer
        """
        return self.layer_of_tensor(
            torch.zeros_like(self._hidden_optimal_delta_layer.weight, device=self.device)
        )
