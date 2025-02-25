import torch

from gromo.config.loader import load_config
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


supported_layer_types = ["linear", "convolution"]


class GrowingContainer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool,
        layer_type: str,
        activation: torch.nn.Module | str | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super(GrowingContainer, self).__init__()
        self._config_data, _ = load_config()
        self.device = get_correct_device(self, device)

        if layer_type not in supported_layer_types:
            raise NotImplementedError(
                f"The layer type is not supported. Expected one of {supported_layer_types}, got {layer_type}"
            )
        self.layer_type = layer_type

        if seed is not None:
            torch.manual_seed(seed)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
