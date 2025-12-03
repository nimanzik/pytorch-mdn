from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import polars as pl
import safetensors.torch as st
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from lightning.pytorch.cli import SaveConfigCallback

from .utils import compute_prediction_metrics, concatenate_batch_predictions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning.pytorch import LightningModule, Trainer

    from .lit_data_modules import TabularDataModule


class ConfigLoggerForMLflow(SaveConfigCallback):
    """Custom callback to log LightningCLI config file to MLflow."""

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None
    ) -> None:
        """Save config file and log it as an artifact in MLflow."""
        if trainer.logger is None:
            return

        # Temporarily save to a file -> log as artifact -> remove file
        config_fpath = Path(trainer.default_root_dir) / "config.yaml"
        self.parser.save(
            self.config, path=config_fpath, skip_none=False, overwrite=True
        )

        trainer.logger.experiment.log_artifact(
            local_path=config_fpath.as_posix(), run_id=trainer.logger.run_id
        )

        config_fpath.unlink()


class ScalerLoggerForMLflow(Callback):
    """Callback to log sklearn scalers to MLflow after training."""

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log scalers to MLflow as artifacts after training completes."""
        # Access the data module
        datamodule: Any = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        if trainer.logger is None:
            return

        # Log x_scaler
        if hasattr(datamodule, "x_scaler") and datamodule.x_scaler is not None:
            x_scaler_fpath = Path(trainer.default_root_dir) / "x_scaler.joblib"
            joblib.dump(datamodule.x_scaler, x_scaler_fpath)

            trainer.logger.experiment.log_artifact(
                local_path=x_scaler_fpath.as_posix(),
                run_id=trainer.logger.run_id,
            )

            x_scaler_fpath.unlink()

        # Log y_scaler
        if hasattr(datamodule, "y_scaler") and datamodule.y_scaler is not None:
            y_scaler_fpath = Path(trainer.default_root_dir) / "y_scaler.joblib"
            joblib.dump(datamodule.y_scaler, y_scaler_fpath)

            trainer.logger.experiment.log_artifact(
                local_path=y_scaler_fpath.as_posix(),
                run_id=trainer.logger.run_id,
            )

            y_scaler_fpath.unlink()


class MDNPredictionLoggerForMLflow(BasePredictionWriter):
    """Callback to log MDN quantile predictions as MLflow artifacts.

    This callback concatenates predictions from all batches, saves them as a
    safetensors file, logs the file as an MLflow artifact, and computes
    and logs prediction metrics.
    """

    output_file = "predictions.safetensors"

    def __init__(self) -> None:
        super().__init__(write_interval="epoch")

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        """Concatenate predictions, compute metrics, and log to MLflow."""
        if not predictions or trainer.logger is None:
            return

        # Use utility function for concatenation
        scaled_preds = concatenate_batch_predictions(predictions)

        # Inverse transform predictions back to original scale
        dm: TabularDataModule = trainer.datamodule
        final_preds = {}
        for key, tensor in scaled_preds.items():
            df_trans: pl.DataFrame = dm.inverse_transform_y(tensor)
            final_preds[key] = df_trans.to_torch(dtype=pl.Float32).contiguous()

        # Compute and log metrics
        metrics = compute_prediction_metrics(final_preds, pl_module.quantiles)  # ty: ignore[invalid-argument-type]
        if metrics:
            trainer.logger.log_metrics(metrics)

        # Log final predictions as a safetensors file
        output_fpath = Path(trainer.default_root_dir) / self.output_file
        st.save_file(final_preds, output_fpath)

        trainer.logger.experiment.log_artifact(
            local_path=output_fpath.as_posix(),
            run_id=trainer.logger.run_id,
        )

        output_fpath.unlink()
