from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor


def mdn_loss(
    pi: TorchTensor, mu: TorchTensor, sigma: TorchTensor, target: TorchTensor
) -> TorchTensor:
    """Compute negative log-likelihood loss for Mixture Density Network.

    This loss assumes a spherical Gaussian for each mixture component (i.e.,
    the same standard deviation across all output dimensions within each
    component).

    Parameters
    ----------
    pi : Tensor of shape (batch_size, n_components)
        Mixture weights.
    mu : Tensor of shape (batch_size, n_components, output_dim)
        Component means.
    sigma : Tensor of shape (batch_size, n_components)
        Component standard deviations.
    target : Tensor of shape (batch_size, output_dim)
        Target values.

    Returns
    -------
    loss : Tensor
        Scalar negative log-likelihood loss. This value can be negative when
        the model assigns high probability density to the targets, which is
        valid behavior for *continuous* distributions.
    """
    output_dim = mu.shape[-1]
    assert target.shape[-1] == output_dim, "Output dimension mismatch."

    target_expanded = target.unsqueeze(1).expand_as(mu)

    sigma_clamped = sigma.clamp(min=1e-7)
    log_probs = (
        -0.5 * output_dim * math.log(2.0 * math.pi)
        - output_dim * sigma_clamped.log()
        - (target_expanded - mu).pow(2).sum(dim=-1) / (2.0 * sigma_clamped.pow(2))
    )
    weighted_log_probs = pi.clamp(min=1e-8).log() + log_probs

    # Negative log-likelihood
    return -1.0 * weighted_log_probs.logsumexp(dim=-1).mean()
