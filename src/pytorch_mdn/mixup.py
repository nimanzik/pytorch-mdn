from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.neighbors import KernelDensity


def get_mixup_idxs(
    data: np.ndarray,
    bandwidth: Literal["scott", "silverman"] = "scott",
    seed: int | None = None,
) -> np.ndarray:
    """Compute sampling rates based on kernel density estimation.

    Data must be in sklearn format: (n_samples, n_features) or
    (n_samples, n_targets)

    Returns
    -------
    np.ndarray of shape (n_samples, n_samples)
        Array of sampling probabilities that sum to 1.0. Each row corresponds
        to the sampling distribution for the respective sample.
    """
    rng = np.random.default_rng(seed=seed)
    batch_size = data.shape[0]

    mixup_idxs: list[int] = []

    for i in range(data.shape[0]):
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian", metric="euclidean")
        kde.fit(data[[i]])
        mixup_rates = np.exp(kde.score_samples(data))  # shape: (n_samples,)
        mixup_rates = mixup_rates / np.sum(mixup_rates)

        # Get mixup indices
        mixup_idx: int = rng.choice(np.arange(batch_size), size=None, p=mixup_rates)
        mixup_idxs.append(mixup_idx)

    return np.array(mixup_idxs)
