from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from .mixup import BaseMixup, KDEMixup, RandomMixup
    from .model import MixtureDensityNetwork


class MDNConfig(BaseModel, extra="forbid"):
    """Configuration for MDN model architecture."""

    input_dim: int = Field(..., gt=0, description="Dimension of input features")
    hidden_dims: list[int] = Field(..., description="Hidden layer dimensions")
    output_dim: int = Field(..., gt=0, description="Dimension of output targets")
    n_components: int = Field(..., gt=0, description="Number of mixture components")
    activation_type: str = Field(..., description="Activation function name")
    norm_type: Literal["batch", "layer"] | None = Field(
        None, description="Type of normalisation layer to use"
    )
    norm_after_act: bool = Field(
        False, description="Whether to apply normalisation after activation"
    )
    dropout_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Dropout rate for regularisation"
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
        from .model import MixtureDensityNetwork

        return MixtureDensityNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            n_components=self.n_components,
            activation_type=self.activation_type,
            norm_type=self.norm_type,
            norm_after_act=self.norm_after_act,
            dropout_rate=self.dropout_rate,
        )


class BaseMixupConfig(BaseModel, ABC):
    """Base class for all MixUp configurations."""

    @abstractmethod
    def create_mixup(self) -> BaseMixup:
        """Convert configuration to MixUp instance.

        Returns
        -------
        BaseMixup
            MixUp instance created from this configuration.
        """
        ...


class RandomMixupConfig(BaseMixupConfig, extra="forbid"):
    """Configuration for random MixUp augmentation."""

    type: Literal["random"] = "random"
    alpha: float = Field(
        ..., gt=0, description="Beta-distribution parameter for lambda sampling"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    def create_mixup(self) -> RandomMixup:
        """Convert configuration to RandomMixup instance."""
        from .mixup import RandomMixup

        return RandomMixup(alpha=self.alpha, seed=self.seed)


class KDEMixupConfig(BaseMixupConfig, extra="forbid"):
    """Configuration for KDE-based MixUp augmentation.

    Uses kernel density estimation to find similar samples in *target space*.
    """

    type: Literal["kde"] = "kde"
    bandwidth: float = Field(..., description="Kernel bandwidth")
    kernel: Literal["gaussian", "tophat", "epanechnikov"] = Field(
        default="gaussian", description="Kernel type to use"
    )
    metric: Literal["euclidean", "cosine", "manhattan"] = Field(
        default="euclidean", description="Metric to use for distance computation"
    )
    alpha: float = Field(
        ..., gt=0, description="Beta-distribution parameter for lambda sampling"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    def create_mixup(self) -> KDEMixup:
        """Convert configuration to KDEMixup instance."""
        from .mixup import KDEMixup

        return KDEMixup(
            bandwidth=self.bandwidth,
            alpha=self.alpha,
            kernel=self.kernel,
            metric=self.metric,
            seed=self.seed,
        )


# Union type for all mixup configs
MixupConfig = RandomMixupConfig | KDEMixupConfig
