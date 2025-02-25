import torch

from gromo.modules.linear_growing_module import LinearGrowingModule


class ConstantModule(LinearGrowingModule):
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

    def forward(self, x):
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
    def optimal_delta_layer(self):
        return self.layer_of_tensor(
            torch.zeros_like(self._hidden_optimal_delta_layer.weight, device=self.device)
        )
