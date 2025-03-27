import torch

from gromo.config.loader import load_config
from gromo.modules.growing_module import AdditionGrowingModule, GrowingModule
from gromo.utils.utils import get_correct_device, global_device


def safe_forward(self, input: torch.Tensor) -> torch.Tensor:
    """Safe Linear forward function for empty input tensors
    Resolves bug with shape transformation when using cuda

    Parameters
    ----------
    input : torch.Tensor
        input tensor

    Returns
    -------
    torch.Tensor
        F.linear forward function output
    """
    assert (
        input.shape[-1] == self.in_features
    ), f"Input shape {input.shape} must match the input feature size. Expected: {self.in_features}, Found: {input.shape[1]}"
    if self.in_features == 0:
        return torch.zeros(
            input.shape[0], self.out_features, device=global_device(), requires_grad=True
        )  # TODO: change to self.device?
    return torch.nn.functional.linear(input, self.weight, self.bias)


class GrowingContainer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
    ) -> None:
        super(GrowingContainer, self).__init__()
        self._config_data, _ = load_config()
        self.device = get_correct_device(self, device)

        self.in_features = in_features
        self.out_features = out_features

        self.growing_layers = torch.nn.ModuleList()

    def set_growing_layers(self):
        """
        Reference all growable layers of the model in the growing_layers attribute. This method should be implemented
        in the child class and called in the __init__ method.
        """
        raise NotImplementedError

    def init_computation(self):
        """Initialize statistics computations for growth procedure"""
        for layer in self.growing_layers:
            if isinstance(layer, (GrowingModule, AdditionGrowingModule)):
                layer.init_computation()

    def reset_computation(self):
        """Reset statistics computations for growth procedure"""
        for layer in self.growing_layers:
            if isinstance(layer, (GrowingModule, AdditionGrowingModule)):
                layer.reset_computation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        raise NotImplementedError

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extended forward pass through the network"""
        raise NotImplementedError
