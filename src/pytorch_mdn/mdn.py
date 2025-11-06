from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor as TorchTensor

LOG_2PI = math.log(2.0 * math.pi)

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "silu": nn.SiLU,
}


class ELUPlus1(nn.ELU):
    """ELU activation shifted upwards by 1 to get strictly positive outputs."""

    def __init__(self):
        super().__init__(alpha=1.0)

    def forward(self, input: TorchTensor) -> TorchTensor:
        return super().forward(input) + 1.0


class MixtureDensityNetwork(nn.Module):
    """Mixture Density Network for modeling multimodal distributions.

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dims : list[int] | int
        Hidden layer dimensions. Can be a single int or list of ints
    output_dim : int
        Dimension of output targets
    n_components : int
        Number of mixture components in the output distribution
    activation : {"gelu", "relu", "mish", "silu"}
        Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | int,
        output_dim: int,
        n_components: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.n_components: int = n_components

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims: list[int] = hidden_dims

        activation_fn = ACTIVATION_FUNCTIONS.get(activation.lower(), nn.GELU)

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            block = [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation_fn(),
            ]

            layers.extend(block)
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.pi_layer = nn.Sequential(
            nn.Linear(prev_dim, self.n_components), nn.Softmax(dim=-1)
        )
        self.mu_layer = nn.Sequential(
            nn.Linear(prev_dim, self.n_components * output_dim), nn.Softplus()
        )
        self.sigma_layer = nn.Sequential(
            nn.Linear(prev_dim, self.n_components), ELUPlus1()
        )

    def forward(self, x: TorchTensor) -> tuple[TorchTensor, TorchTensor, TorchTensor]:
        """Forward pass to compute mixture distribution parameters.

        Parameters
        ----------
        x : TorchTensor, shape (batch_size, input_dim)
            Input tensor.

        Returns
        -------
        pi : TorchTensor, shape (batch_size, n_components)
            Mixture weights.
        mu : TorchTensor, shape (batch_size, n_components, output_dim)
            Component means.
        sigma : TorchTensor, shape (batch_size, n_components)
            Component standard deviations.
        """
        h = self.backbone(x)
        pi = self.pi_layer(h)
        mu = self.mu_layer(h).view(-1, self.n_components, self.output_dim)
        sigma = self.sigma_layer(h).view(-1, self.n_components)
        return pi, mu, sigma

    @torch.inference_mode()
    def sample(self, x: TorchTensor, n_samples: int = 100) -> TorchTensor:
        """Sample from the mixture distribution.

        Parameters
        ----------
        x : TorchTensor, shape (batch_size, input_dim)
            Input tensor.
        n_samples : int, default=100
            Number of samples to draw per batch element.

        Returns
        -------
        samples : TorchTensor, shape (batch_size, n_samples, output_dim)
            Samples from the mixture distribution.
        """
        pi, mu, sigma = self.forward(x)
        batch_size = x.shape[0]

        # Vectorized sampling
        component_dist = torch.distributions.Categorical(pi)
        component_idxs = component_dist.sample((n_samples,)).T  # (B, N)

        batch_indices = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, n_samples)
        )
        sampled_mu = mu[batch_indices, component_idxs]  # (B, N, D_out)
        sampled_sigma = sigma[batch_indices, component_idxs]  # (B, N)

        noise = torch.randn_like(sampled_mu)
        return sampled_mu + sampled_sigma.unsqueeze(-1) * noise

    @torch.inference_mode()
    def predict(
        self,
        x: TorchTensor,
        inference_type: Literal["mode", "mean", "sample"],
    ) -> TorchTensor:
        """Generate predictions from the mixture distribution.

        Parameters
        ----------
        x : TorchTensor, shape (batch_size, input_dim)
            Input tensor.
        inference_type : {"mode", "mean", "sample"}
            Type of prediction to generate:
            - "mode": Mean of the most probable component
            - "mean": Weighted average of all component means
            - "sample": Single sample from the distribution

        Returns
        -------
        predictions : TorchTensor, shape (batch_size, output_dim)
            Predicted values based on the selected inference type.
        """
        if inference_type == "sample":
            return self.sample(x, n_samples=1).squeeze(1)

        pi, mu, _ = self.forward(x)

        if inference_type == "mean":
            # Weighted average of all component means
            return torch.sum(pi.unsqueeze(-1) * mu, dim=1)
        else:
            # inference_type == "mode":
            # Mean of the most probable component
            most_probable_idxs = pi.argmax(dim=-1)  # (B,)
            batch_idxs = torch.arange(most_probable_idxs.shape[0], device=x.device)
            return mu[batch_idxs, most_probable_idxs]

    @torch.inference_mode()
    def predict_quantiles(
        self, x: TorchTensor, quantiles: list[float], n_samples: int = 100
    ) -> dict[str, TorchTensor]:
        """Compute quantile predictions from the mixture distribution.

        Parameters
        ----------
        x : TorchTensor, shape (batch_size, input_dim)
            Input tensor.
        quantiles : list[float]
            List of quantile values to compute. Each value must be in the
            range (0, 1).
        n_samples : int, default=100
            Number of samples to draw for quantile estimation.

        Returns
        -------
        results : dict[str, TorchTensor]
            Dictionary with key-value pairs:
            - "samples" -> TorchTensor, shape (batch_size, n_samples, output_dim)
              All samples from the distribution
            - "q_{q}" -> TorchTensor, shape (batch_size, output_dim)
              Quantile predictions for each quantile value in `quantiles`.
        """  # noqa: W505
        q_vals = np.array(quantiles)
        if not np.all((q_vals > 0) & (q_vals < 1)):
            raise ValueError("Quantiles must be between 0 and 1.")

        samples = self.sample(x, n_samples)  # (B, N, D_out)
        results = {"samples": samples}

        for q in q_vals:
            results[f"q_{q}"] = samples.quantile(q, dim=1)  # (B, D_out)

        return results


def mdn_loss(
    pi: TorchTensor, mu: TorchTensor, sigma: TorchTensor, target: TorchTensor
) -> TorchTensor:
    """Compute negative log-likelihood loss for Mixture Density Network.

    This loss assumes a spherical Gaussian for each mixture component (i.e.,
    the same standard deviation across all output dimensions within each
    component).

    Parameters
    ----------
    pi : TorchTensor, shape (batch_size, n_components)
        Mixture weights.
    mu : TorchTensor, shape (batch_size, n_components, output_dim)
        Component means.
    sigma : TorchTensor, shape (batch_size, n_components)
        Component standard deviations.
    target : TorchTensor, shape (batch_size, output_dim)
        Target values.

    Returns
    -------
    loss : TorchTensor
        Scalar negative log-likelihood loss. This value can be negative when
        the model assigns high probability density to the targets, which is
        valid behavior for *continuous* distributions.
    """
    output_dim = mu.shape[-1]
    assert target.shape[-1] == output_dim, "Output dimension mismatch."

    target_expanded = target.unsqueeze(1).expand_as(mu)

    sigma_clamped = sigma.clamp(min=1e-7)
    log_probs = (
        -0.5 * output_dim * LOG_2PI
        - output_dim * sigma_clamped.log()
        - (target_expanded - mu).pow(2).sum(dim=-1) / (2.0 * sigma_clamped.pow(2))
    )
    weighted_log_probs = pi.clamp(min=1e-8).log() + log_probs

    # Negative log-likelihood
    return -1.0 * weighted_log_probs.logsumexp(dim=-1).mean()
