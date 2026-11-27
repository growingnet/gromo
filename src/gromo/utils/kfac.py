"""K-FAC variants of the Kronecker factors of the curvature of a layer.

The growth computations of a layer use two Kronecker factors of the curvature: the
input second moment S (``tensor_s``) and the gradient second moment E
(``covariance_loss_gradient``). For sequence inputs of shape (N, R, d), both sum over
the valid positions and divide by the number of sequences N.

A K-FAC variant replaces them by the factors A and B of a K-FAC approximation suited
to a given weight-sharing setting. With ``a_{n,m}`` the input of the layer and
``g_{n,m}`` the gradient of the loss with respect to its pre-activation at the valid
position m of sequence n, R_n the number of valid positions of sequence n, and the
per-sequence means ``x̄_n = (sum_m x_{n,m}) / R_n``, every variant has the form::

    A = sum_n omega_n ā_n ā_n^T / D_A,
    B = sum_n omega_n ḡ_n ḡ_n^T / N,

where a variant sets the weights omega_n, from rho_n, and the denominator D_A.

Provide `KFACVariant`. Only K-FAC-reduce with count-weighted lengths is implemented:
rho_n = R_n^2, omega_n = rho_n for both factors and D_A = sum_n omega_n, so that::

    A = sum_n u_n u_n^T / sum_n R_n^2,    B = sum_n g_n g_n^T / N,

with ``u_n = sum_m a_{n,m}`` and ``g_n = sum_m g_{n,m}``. It suits models that pool
the positions of a sequence before the loss, e.g. a sequence classifier.
"""

from dataclasses import dataclass
from typing import Literal, get_args

import torch


Setting = Literal["expand", "reduce"]
Placement = Literal["count_weighted", "mean_sum", "square_root", "sum_mean"]


@dataclass(frozen=True)
class KFACVariant:
    """Variant of the K-FAC factors: the weights and the denominator of A and B.

    Only ``KFACVariant("reduce", "count_weighted")`` is implemented: rho_n = R_n^2,
    the number of pairs of positions of sequence n, the weights of both factors are
    omega_n = rho_n, and the denominator of A is D_A = sum_n omega_n.

    Attributes
    ----------
    setting: Setting
        weight-sharing setting, ``"expand"`` or ``"reduce"``
    placement: Placement
        placement of the lengths R_n in the factors, ``"count_weighted"``,
        ``"mean_sum"``, ``"square_root"`` or ``"sum_mean"``

    Raises
    ------
    ValueError
        if the setting or the placement is unknown
    NotImplementedError
        for a known combination other than ``("reduce", "count_weighted")``
    """

    setting: Setting
    placement: Placement

    def __post_init__(self) -> None:
        if self.setting not in get_args(Setting):
            raise ValueError(
                f"Unknown K-FAC setting {self.setting!r}, expected one of "
                f"{get_args(Setting)}."
            )
        if self.placement not in get_args(Placement):
            raise ValueError(
                f"Unknown K-FAC placement {self.placement!r}, expected one of "
                f"{get_args(Placement)}."
            )
        if (self.setting, self.placement) != ("reduce", "count_weighted"):
            raise NotImplementedError(
                f"The K-FAC variant ({self.setting!r}, {self.placement!r}) is not "
                f"implemented, only ('reduce', 'count_weighted') is."
            )

    def rho(self, lengths: torch.Tensor) -> torch.Tensor:
        """Count rho_n of each sequence, from which the weights are computed.

        For K-FAC-reduce, rho_n = R_n^2 is the number of pairs of positions.

        Parameters
        ----------
        lengths: torch.Tensor
            numbers of valid positions R_n, of shape (N,)

        Returns
        -------
        torch.Tensor
            rho_n, of shape (N,)
        """
        return lengths.square()

    def input_weights(self, rho: torch.Tensor) -> torch.Tensor:
        """Weights omega_n of the input factor A.

        Count-weighted: omega_n = rho_n.

        Parameters
        ----------
        rho: torch.Tensor
            rho_n, of shape (N,)

        Returns
        -------
        torch.Tensor
            omega_n, of shape (N,)
        """
        return rho

    def gradient_weights(self, rho: torch.Tensor) -> torch.Tensor:
        """Weights omega_n of the gradient factor B.

        Count-weighted: omega_n = rho_n.

        Parameters
        ----------
        rho: torch.Tensor
            rho_n, of shape (N,)

        Returns
        -------
        torch.Tensor
            omega_n, of shape (N,)
        """
        return rho

    def input_denominator(
        self,
        sum_input_weights: torch.Tensor,
        n_sequences: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Denominator D_A of the input factor A.

        Count-weighted: D_A = sum_n omega_n.

        Parameters
        ----------
        sum_input_weights: torch.Tensor
            sum of the input weights sum_n omega_n
        n_sequences: int
            number of sequences N, used by other placements

        Returns
        -------
        torch.Tensor
            D_A
        """
        return sum_input_weights
