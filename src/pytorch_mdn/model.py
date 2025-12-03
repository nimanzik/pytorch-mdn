from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator
from torch import Tensor as TorchTensor

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "silu": nn.SiLU,
}


class ElevatedELU(nn.ELU):
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
    activation_type : {"gelu", "relu", "mish", "silu"}
        Activation function name.
    bn_after_act : bool, default=False
        Whether to apply batch normalization after activation functions.
    """

    # ToDo: Add option for different activation functions for mu layer

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | int,
        output_dim: int,
        n_components: int,
        activation_type: str,
        bn_after_act: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims: list[int] = hidden_dims

        activation_fn = ACTIVATION_FUNCTIONS.get(activation_type.lower(), nn.GELU)

        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            block = [nn.Linear(prev_dim, hidden_dim), activation_fn()]
            if bn_after_act:
                block.append(nn.BatchNorm1d(hidden_dim))
            else:
                block.insert(1, nn.BatchNorm1d(hidden_dim))

            layers.extend(block)
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.pi_layer = nn.Sequential(
            nn.Linear(prev_dim, self.n_components), nn.LogSoftmax(dim=-1)
        )
        self.mu_layer = nn.Sequential(
            nn.Linear(prev_dim, self.n_components * output_dim), nn.Softplus()
        )
        self.sigma_layer = nn.Sequential(
            nn.Linear(prev_dim, self.n_components), ElevatedELU()
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
            Component means (i.e., centers).
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
        self, x: TorchTensor, quantiles: ArrayLike[float], n_samples: int = 100
    ) -> dict[str, TorchTensor]:
        """Compute quantile predictions from the mixture distribution.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Input tensor.
        quantiles : list[float]
            List of quantile values to compute, each between 0 and 1.
        n_samples : int, default=100
            Number of samples to draw for quantile estimation.

        Returns
        -------
        results : dict[str, Tensor]
            Dictionary with keys as "q_{quantile}" and values as Tensors of
            shape (batch_size, output_dim) containing the estimated quantiles.
        """
        quantiles = np.asarray(quantiles)
        if not np.all((quantiles > 0) & (quantiles < 1)):
            raise ValueError("Quantiles must be between 0 and 1.")

        samples = self.generate_samples(x, n_samples)  # (B, N, D_out)
        return {f"q_{q}": samples.quantile(q, dim=1) for q in quantiles}


class MDNConfig(BaseModel, extra="forbid"):
    """Configuration for MDN model architecture."""

    input_dim: int = Field(..., gt=0, description="Dimension of input features")
    hidden_dims: list[int] = Field(..., description="Hidden layer dimensions")
    output_dim: int = Field(..., gt=0, description="Dimension of output targets")
    n_components: int = Field(..., gt=0, description="Number of mixture components")
    activation_type: str = Field(..., description="Activation function name")
    bn_after_act: bool = Field(
        default=False, description="Apply BatchNorm after activations."
    )

    @field_validator("hidden_dims", mode="before")
    @classmethod
    def ensure_hidden_dims_list(cls, v: list[int] | int) -> list[int]:
        """Ensure `hidden_dims` is always a list of integers."""
        if isinstance(v, int):
            return [v]
        return v

    def create_model(self) -> MixtureDensityNetwork:
        """Create Mixture Density Network model based on the configuration."""
        return MixtureDensityNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            n_components=self.n_components,
            activation_type=self.activation_type,
            bn_after_act=self.bn_after_act,
        )
