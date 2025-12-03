from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.neighbors import KernelDensity

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor


class BaseMixUpConfig(BaseModel, ABC):
    """Base class for all MixUp configurations."""

    @abstractmethod
    def apply_mixup(
        self, x: TorchTensor, y: TorchTensor
    ) -> tuple[TorchTensor, TorchTensor]:
        """Apply MixUp to input and target tensors.

        Parameters
        ----------
        x : TorchTensor
            Input tensor of shape (batch_size, ...).
        y : TorchTensor
            Target tensor of shape (batch_size, ...).

        Returns
        -------
        x_mixed : TorchTensor
            Mixed input tensor.
        y_mixed : TorchTensor
            Mixed target tensor.
        """
        ...

    def __call__(
        self, x: TorchTensor, y: TorchTensor
    ) -> tuple[TorchTensor, TorchTensor]:
        return self.apply_mixup(x, y)


class RandomMixUpConfig(BaseMixUpConfig, extra="forbid"):
    """Configuration for random MixUp augmentation."""

    type: Literal["random"] = "random"
    alpha: float = Field(
        ..., gt=0, description="Beta-distribution parameter for lambda sampling"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    def model_post_init(self, _) -> None:
        self._rng = np.random.default_rng(seed=self.seed)

    def apply_mixup(
        self, x: TorchTensor, y: TorchTensor
    ) -> tuple[TorchTensor, TorchTensor]:
        """Apply random MixUp to input and target tensors."""
        lambda_ = float(self._rng.beta(self.alpha, self.alpha))

        batch_size = x.shape[0]
        mixup_idxs = self._rng.permutation(batch_size).tolist()

        # Mix inputs and targets with the same lambda for all samples
        x_mixed = lambda_ * x + (1.0 - lambda_) * x[mixup_idxs]
        y_mixed = lambda_ * y + (1.0 - lambda_) * y[mixup_idxs]

        return x_mixed, y_mixed


class KDEMixUpConfig(BaseMixUpConfig, extra="forbid"):
    """Configuration for KDE-based MixUp augmentation.

    Uses kernel density estimation to find similar samples in *target space*.
    """

    type: Literal["kde"] = "kde"
    bandwidth: float = Field(
        ..., description="Kernel bandwidth or method to estimate it"
    )
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

    def model_post_init(self, _) -> None:
        self._rng = np.random.default_rng(seed=self.seed)

    def apply_mixup(
        self, x: TorchTensor, y: TorchTensor
    ) -> tuple[TorchTensor, TorchTensor]:
        """Apply KDE-guided MixUp to input and target tensors.

        Uses KDE on target space to find similar samples and mixes them.
        """
        lambda_ = float(self._rng.beta(self.alpha, self.alpha))

        # Get mixup indices using KDE on target space
        # Convert targets to numpy for KDE computation
        y_np = y.detach().cpu().numpy()
        mixup_idxs = self._get_mixup_idxs(y_np).tolist()

        # Mix inputs and targets with the same lambda for all samples
        x_mixed = lambda_ * x + (1 - lambda_) * x[mixup_idxs]
        y_mixed = lambda_ * y + (1 - lambda_) * y[mixup_idxs]

        return x_mixed, y_mixed

    def _get_mixup_idxs(self, data: np.ndarray) -> np.ndarray:
        """Compute KDE to find similar samples for mixing.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, n_features)
            Input data. Each row corresponds to a single data point.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Array of selected indices for each sample based on KDE similarity.

        Notes
        -----
        - Data must be in sklearn format: NumPy array of shape
          (n_samples, n_features) or (n_samples, n_targets).

        - From the C-Mixup paper (https://arxiv.org/abs/2210.05775):

          *C-Mixup proposes to sample closer pairs of examples with higher
          probability. Specifically, given an example (xi, yi), C-Mixup
          introduces a symmetric Gaussian kernel to calculate the sampling
          probability P((xj , yj) | (xi, yi)) for another (xj , yj) example
          to be mixed.*

        - When fitting KDE on single samples, if bandwidth is a method (i.e.,
          string), sklearn falls back to the default value of 1.0. That's why
          'scott' and 'silverman' are not used here.
        """
        batch_size = data.shape[0]

        mixup_idxs: list[int] = []
        for i in range(batch_size):
            kde = KernelDensity(
                bandwidth=self.bandwidth, kernel=self.kernel, metric=self.metric
            )
            kde.fit(data[[i]])
            mixup_rates = np.exp(kde.score_samples(data))  # shape: (n_samples,)
            mixup_rates[i] = 0.0  # Don't mix with itself
            mixup_rates = mixup_rates / np.sum(mixup_rates)

            # Get mixup index for sample i
            mixup_idx: int = self._rng.choice(
                np.arange(batch_size), size=None, p=mixup_rates
            )
            mixup_idxs.append(mixup_idx)

        return np.array(mixup_idxs)


# Union type for all mixup configs
MixUpConfig = RandomMixUpConfig | KDEMixUpConfig
