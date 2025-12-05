from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor as TorchTensor

from .activations import ElevatedELU
from .backbone import MLPBackbone

if TYPE_CHECKING:
    from collections.abc import Sequence


class MixtureDensityNetwork(nn.Module):
    """Mixture Density Network for modeling multimodal distributions.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dims : list[int] or int
        Hidden layer dimensions. Can be a single int or list of ints.
    output_dim : int
        Dimension of output targets.
    n_components : int
        Number of mixture components in the output distribution.
    activation_type : {"gelu", "relu", "mish", "silu"}
        Activation function name.
    norm_type : {"batch", "layer"} or None, default=None
        Type of normalisation layer to use. "batch" for batch normalisation,
        "layer" for layer normalisation, or None for no normalisation.
    norm_after_act : bool, default=False
        Whether to apply normalisation after activation functions. If False,
        normalisation is applied before activation.
    dropout_rate : float, default=0.0
        Dropout probability for regularisation. Must be between 0.0 and 1.0.
        No dropout is applied if 0.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | int,
        output_dim: int,
        n_components: int,
        activation_type: str,
        norm_type: Literal["batch", "layer"] | None = None,
        norm_after_act: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims: list[int] = hidden_dims

        # Create MLP backbone
        self.backbone = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_type=activation_type,
            norm_type=norm_type,
            norm_after_act=norm_after_act,
            dropout_rate=dropout_rate,
        )

        backbone_output_dim = self.backbone.output_dim

        self.pi_layer = nn.Sequential(
            nn.Linear(backbone_output_dim, self.n_components), nn.LogSoftmax(dim=-1)
        )
        self.mu_layer = nn.Linear(backbone_output_dim, self.n_components * output_dim)
        self.sigma_layer = nn.Sequential(
            nn.Linear(backbone_output_dim, self.n_components), ElevatedELU()
        )

    def forward(self, x: TorchTensor) -> tuple[TorchTensor, TorchTensor, TorchTensor]:
        """Forward pass to compute mixture distribution parameters.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Input tensor.

        Returns
        -------
        log_pi : Tensor of shape (batch_size, n_components)
            Log mixture weights (log probabilities).
        mu : Tensor of shape (batch_size, n_components, output_dim)
            Component means (i.e., centres).
        sigma : Tensor of shape (batch_size, n_components)
            Component standard deviations.
        """
        h = self.backbone(x)
        log_pi = self.pi_layer(h)
        mu = self.mu_layer(h).view(-1, self.n_components, self.output_dim)
        sigma = self.sigma_layer(h).view(-1, self.n_components)
        return log_pi, mu, sigma

    @torch.inference_mode()
    def generate_samples(self, x: TorchTensor, n_samples: int = 100) -> TorchTensor:
        """Generate random samples from the mixture distribution.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Input tensor.
        n_samples : int, default=100
            Number of samples to draw per batch element.

        Returns
        -------
        samples : Tensor of shape (batch_size, n_samples, output_dim)
            Samples from the mixture distribution.
        """
        log_pi, mu, sigma = self.forward(x)
        batch_size = x.shape[0]

        # Vectorized sampling
        component_dist = torch.distributions.Categorical(logits=log_pi)
        component_idxs = component_dist.sample((n_samples,)).T  # (B, N)

        batch_idxs = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, n_samples)
        )
        sampled_mu = mu[batch_idxs, component_idxs]  # (B, N, D_out)
        sampled_sigma = sigma[batch_idxs, component_idxs]  # (B, N)

        noise = torch.randn_like(sampled_mu)
        return sampled_mu + sampled_sigma.unsqueeze(-1) * noise

    @torch.inference_mode()
    def predict(
        self,
        x: TorchTensor,
        inference_type: Literal[
            "sample_mean", "sample_median", "weighted_mean", "argmax_mean"
        ],
        n_samples: int = 100,
    ) -> TorchTensor:
        """Generate predictions from the mixture distribution.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Input tensor.

        inference_type : {"sample_mean", "sample_median", "weighted_mean", \
"argmax_mean"}
            Type of prediction to generate:
            - "sample_mean":
                Mean of samples drawn from the mixture distribution.
            - "sample_median":
                Median of samples drawn from the mixture distribution.
            - "weighted_mean":
                Expected value E[Y|X] computed as weighted average of component
                means of the mixture distribution.
            - "argmax_mean":
                Mean of the most probable component (i.e., component with
                highest mixing weight). Fast approximation that works well when
                one component dominates, but may be inaccurate when components
                overlap significantly.

        n_samples : int, default=100
            Number of samples to draw when using "sample_mean" or
            "sample_median" inference types.

        Returns
        -------
        preds : Tensor of shape (batch_size, output_dim)
            Predicted values based on the selected inference type.
        """
        if inference_type in ("sample_mean", "sample_median"):
            samples = self.generate_samples(x, n_samples=n_samples)  # (B, N, D_out)
            return (
                samples.mean(dim=1)
                if inference_type == "sample_mean"
                else samples.median(dim=1).values
            )  # (B, D_out)

        log_pi, mu, _ = self.forward(x)

        if inference_type == "weighted_mean":
            # Weighted average of all component means, E[Y|X=x]
            pi = log_pi.exp()
            return torch.sum(pi.unsqueeze(-1) * mu, dim=1)
        else:
            # inference_type == "argmax_mean"
            # Mean of the most probable component
            most_probable_idxs = log_pi.argmax(dim=-1)  # (B,)
            batch_idxs = torch.arange(most_probable_idxs.shape[0], device=x.device)
            return mu[batch_idxs, most_probable_idxs]

    @torch.inference_mode()
    def predict_quantiles(
        self, x: TorchTensor, quantiles: Sequence[float], n_samples: int = 100
    ) -> dict[str, TorchTensor]:
        """Compute quantile predictions from the mixture distribution.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Input tensor.
        quantiles : sequence of float
            Sequence of quantile values to compute, each between 0 and 1.
        n_samples : int, default=100
            Number of samples to draw for quantile estimation.

        Returns
        -------
        results : dict[str, Tensor]
            Dictionary with keys as "q_{quantile}" and values as tensors of
            shape (batch_size, output_dim) containing the estimated quantiles.
        """
        quantiles_arr = np.asarray(quantiles)
        if not np.all((quantiles_arr > 0) & (quantiles_arr < 1)):
            raise ValueError("Quantiles must be between 0 and 1.")

        samples = self.generate_samples(x, n_samples)  # (B, N, D_out)
        return {f"q_{q}": samples.quantile(q, dim=1) for q in quantiles}
