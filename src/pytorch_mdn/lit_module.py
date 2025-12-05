from __future__ import annotations

from typing import TYPE_CHECKING

from lightning import LightningModule

from .loss import mdn_loss

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor as TorchTensor

    from .configs import MDNConfig, MixupConfig


class MDNLitModule(LightningModule):
    """PyTorch Lightning module for training a Mixture Density Network."""

    def __init__(
        self,
        model_config: MDNConfig,
        mixup_config: MixupConfig | None = None,
        quantiles: Sequence[float] | None = None,
        n_samples: int = 1000,
    ) -> None:
        super().__init__()

        # Save all the hyperparameters passed to the constructor
        self.save_hyperparameters()

        # Initialize MDN model
        self.model = model_config.create_model()
        self.mixup = mixup_config.create_mixup() if mixup_config is not None else None
        self.quantiles = (
            sorted(quantiles) if quantiles is not None else (0.05, 0.5, 0.95)
        )
        self.n_samples = n_samples

    def forward(self, x: TorchTensor) -> tuple[TorchTensor, TorchTensor, TorchTensor]:
        """Perform a forward pass through the MDN model.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        log_pi : Tensor
            Log of mixture weights (log probabilities).
        mu : Tensor
            Component means.
        sigma : Tensor
            Component standard deviations.
        """
        return self.model(x)

    def _compute_loss(self, x: TorchTensor, y: TorchTensor) -> TorchTensor:
        """Compute MDN loss for given inputs and targets."""
        pi, mu, sigma = self.forward(x)
        return mdn_loss(pi, mu, sigma, y)

    def training_step(
        self, batch: tuple[TorchTensor, TorchTensor], batch_idx: int
    ) -> TorchTensor:
        """Compute and log training loss for a single batch.

        Returns
        -------
        loss : Tensor
            Training loss for the batch.
        """
        x, y = batch

        # Apply MixUp if configured
        if self.mixup is not None:
            x, y = self.mixup(x, y)

        loss = self._compute_loss(x, y)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            logger=False,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        """Log training loss to MLflow at the end of each epoch."""
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is None or self.logger is None:
            return
        # Log train_loss to MLflow at epoch end
        self.logger.log_metrics(
            {"train_loss": train_loss.item()}, step=self.current_epoch
        )

    def validation_step(
        self, batch: tuple[TorchTensor, TorchTensor], batch_idx: int
    ) -> TorchTensor:
        """Compute and log validation loss for a single batch.

        Returns
        -------
        loss : Tensor
            Validation loss for the batch.
        """
        x, y = batch
        loss = self._compute_loss(x, y)
        self.log(
            "val_loss", loss, on_epoch=True, on_step=False, logger=False, prog_bar=True
        )
        return loss

    def on_validation_epoch_end(self):
        """Log validation loss to MLflow at the end of each epoch."""
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is None or self.logger is None:
            return
        self.logger.log_metrics({"val_loss": val_loss.item()}, step=self.current_epoch)

    def test_step(self, batch: tuple[TorchTensor, TorchTensor]) -> TorchTensor:
        """Compute and log test loss for a single batch.

        Returns
        -------
        loss : Tensor
            Test loss for the batch.
        """
        x, y = batch
        loss = self._compute_loss(x, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False, logger=False)
        return loss

    def on_test_end(self):
        """Log test loss to MLflow at the end of testing."""
        test_loss = self.trainer.callback_metrics.get("test_loss")
        if test_loss is None or self.logger is None:
            return
        # Log test_loss to MLflow at test end
        self.logger.log_metrics({"test_loss": test_loss.item()})

    def predict_step(
        self,
        batch: TorchTensor | tuple[TorchTensor, TorchTensor] | list[TorchTensor],
        batch_idx: int,
    ) -> dict[str, TorchTensor]:
        """Predict quantiles for the given batch.

        Returns
        -------
        predictions : dict[str, Tensor]
            Dictionary containing predictions for the specified quantiles. The
            keys are "q_{quantile}" where quantile is the quantile value, e.g.,
            "q_0.05", "q_0.5", "q_0.95". The values are Tensors of shape
            (batch_size, output_dim). If ground-truth y is available in the
            batch, it is also included with key "y_true".
        """
        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            x = batch
            y = None

        preds = self.model.predict_quantiles(x, self.quantiles, self.n_samples)

        # Include ground-truth y if available
        if y is not None:
            preds["y_true"] = y

        return preds
