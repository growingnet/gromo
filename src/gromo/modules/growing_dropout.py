import torch
import torch.nn as nn


class GrowingDropout(nn.modules.dropout._DropoutNd):
    """
    Base class for dropout layers on growing architectures.

    This class provides dropout without penalyzing growth.
    The growth of the previous layer is never zeroed.

    Parameters
    ----------
    dropout_rate : float
        Probability of an element to be zeroed.
        Set to non-zero to activate.
        Default: 0.0 for no dropout.
    name : str, optional
        Name of the layer for debugging, by default="growing_droupout"
    """

    def __init__(
        self,
        dropout_rate: float = 0.0,
        name: str = "growing_dropout",
    ):

        super(GrowingDropout, self).__init__(
            p=dropout_rate,
            inplace=False,  # Dropout should not be inplace to avoid modifying the input tensor directly
        )
        self.name = name

    def extended_forward(
        self, x: torch.Tensor | None, x_ext: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply dropout to the main tensor; pass the extension through unchanged.


        Parameters
        ----------
        x: torch.Tensor | None
            Main pre-activation tensor (N channels / features), or ``None``
            when the main path is irrelevant.
        x_ext: torch.Tensor | None
            Extension pre-activation tensor (M channels / features), or ``None``
            when there is no extension.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            ``(self(x), x_ext)`` — batch-normalised main tensor and unmodified
            extension tensor.  ``None`` inputs propagate as ``None`` outputs.
        """
        return self(x) if x is not None else None, x_ext

    def extra_repr(self) -> str:
        """
        Extra representation string for the layer.
        """
        return f"{super().extra_repr()}, name={self.name}"


class GrowingDropout2d(GrowingDropout, nn.Dropout2d):
    """
    A 2D dropout layer that do not penalize the growth.

    This class extends torch.nn.Dropout2d and never zero the growth of the previous layer.
    """


class GrowingDropout1d(GrowingDropout, nn.Dropout1d):
    """
    A 1D dropout layer that do not penalize the growth.

    This class extends torch.nn.Dropout1d and never zero the growth of the previous layer.
    """
