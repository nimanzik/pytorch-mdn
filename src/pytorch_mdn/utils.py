from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    r2_score,
    spearman_corrcoef,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor as TorchTensor


def concatenate_batch_predictions(
    predictions: Sequence[dict[str, TorchTensor]],
) -> dict[str, TorchTensor]:
    """Concatenate predictions from multiple batches into single tensors.

    Parameters
    ----------
    predictions : Sequence[dict[str, Tensor]]
        Sequence of prediction dictionaries from multiple batches. Each
        dictionary should have the same keys.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with concatenated tensors for each key. Returns an
        empty dictionary if predictions is empty.
    """
    if not predictions:
        return {}

    return {
        key: torch.cat([batch[key] for batch in predictions], dim=0)
        for key in predictions[0].keys()
    }


def compute_prediction_metrics(
    predictions: dict[str, TorchTensor],
    quantiles: Sequence[float],
) -> dict[str, float]:
    """Compute prediction metrics (MAE and coverage) from MDN predictions.

    Parameters
    ----------
    predictions : dict[str, Tensor]
        Dictionary containing prediction quantiles and ground truth. Must
        contain key "y_true" and quantile keys corresponding to the values
        in the quantiles sequence. If quantiles contains 0.5, must also
        contain key "q_0.5".
    quantiles : Sequence[float]
        Sequence of quantile values used for predictions (e.g., [0.05, 0.5,
        0.95]).

    Returns
    -------
    dict[str, float]
        Dictionary containing prediction metrics. Always includes "coverage"
        (prediction interval coverage). Additionally includes "mae" (Mean
        Absolute Error) if 0.5 is in quantiles. Returns an empty dictionary
        if "y_true" is not present in predictions.
    """
    y_true = predictions.get("y_true")
    if y_true is None:
        return {}

    metrics_to_log = {}
    n_targets = y_true.shape[1] if y_true.ndim > 1 else 1

    # Compute regression metrics for median predictions if available
    if 0.5 in quantiles:
        y_median = predictions["q_0.5"]
        per_target_metrics = {
            "MAE": mean_absolute_error(y_median, y_true, num_outputs=n_targets),
            "RMSE": mean_squared_error(y_median, y_true, num_outputs=n_targets).sqrt(),
            "R2": r2_score(y_median, y_true, multioutput="raw_values"),
            # ToDo: Compute correlation coefficients if we have enough variance
            # ToDo: tiny = 1e-10, var(y_true) > tiny and var(y_median) > tiny
            "Spearman-R": spearman_corrcoef(y_median, y_true),
            "Pearson-R": pearson_corrcoef(y_median, y_true),
        }

        for metric_name, values in per_target_metrics.items():
            metrics_to_log[f"{metric_name}_avg"] = values.mean().item()
            for i_target in range(n_targets):
                metrics_to_log[f"{metric_name}_{i_target}"] = values[i_target].item()

    # Compute prediction interval coverage
    y_lower = predictions[f"q_{min(quantiles)}"]
    y_upper = predictions[f"q_{max(quantiles)}"]
    coverage_per_target = (
        torch.logical_and(y_true >= y_lower, y_true <= y_upper).float().mean(dim=0)
    )

    # Log coverage metrics following the same pattern as other metrics
    metrics_to_log["coverage_avg"] = coverage_per_target.mean().item()
    for i_target in range(n_targets):
        metrics_to_log[f"coverage_{i_target}"] = coverage_per_target[i_target].item()

    return metrics_to_log
