from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning import LightningModule
from timm.optim import create_optimizer_v2

from .loss import mdn_loss
from .model import MixtureDensityNetwork

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor
    from torch.optim import Optimizer

    from ..scheduler_config import SchedulerConfig
    from .config import MDNTrainingConfig


class MDNLitModule(LightningModule):
    """PyTorch Lightning module for training a Mixture Density Network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | int,
        output_dim: int,
        n_components: int,
        activation_type: str,
        bn_after_act: bool = False,
        optimizer_type: str = "AdamW",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        super().__init__()

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Initialize MDN model
        self.model = MixtureDensityNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            n_components=n_components,
            activation_type=activation_type,
            bn_after_act=bn_after_act,
        )

        self.optimizer_type: str = optimizer_type
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.scheduler_config: SchedulerConfig | None = scheduler_config

    @classmethod
    def from_config(cls, config: MDNTrainingConfig) -> MDNLitModule:
        """Create an MDNLitModule instance from a configuration object.

        Parameters
        ----------
        config : MDNTrainingConfig
            Configuration object containing all training parameters.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            n_components=config.n_components,
            activation_type=config.activation_type,
            bn_after_act=config.bn_after_act,
            optimizer_type=config.optimizer_type,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            scheduler_config=config.scheduler_config,
        )

    def forward(self, x: TorchTensor) -> tuple[TorchTensor, TorchTensor, TorchTensor]:
        """Forward pass through the MDN model.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        pi : Tensor
            Mixture weights.
        mu : Tensor
            Component means.
        sigma : Tensor
            Component standard deviations.
        """
        return self.model(x)

    def _compute_loss(self, x: TorchTensor, y: TorchTensor) -> TorchTensor:
        pi, mu, sigma = self.forward(x)
        return mdn_loss(pi, mu, sigma, y)

    def training_step(
        self, batch: tuple[TorchTensor, TorchTensor], batch_idx: int
    ) -> TorchTensor:
        """Training step for a single batch.

        Returns
        -------
        loss : Tensor
            Training loss for the batch.
        """
        x, y = batch
        loss = self._compute_loss(x, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(
        self, batch: tuple[TorchTensor, TorchTensor], batch_idx: int
    ) -> TorchTensor:
        """Compute validation loss for a single batch.

        Returns
        -------
        loss : Tensor
            Validation loss for the batch.
        """
        x, y = batch
        loss = self._compute_loss(x, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(
        self, batch: tuple[TorchTensor, TorchTensor], batch_idx: int
    ) -> TorchTensor:
        """Test step for a single batch.

        Returns
        -------
        loss : Tensor
            Test loss for the batch.
        """
        x, y = batch
        loss = self._compute_loss(x, y)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def predict_step(
        self,
        batch: TorchTensor | tuple[TorchTensor, TorchTensor] | list[TorchTensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, TorchTensor]:
        """Prediction step for a single batch.

        Returns
        -------
        predictions : dict[str, Tensor]
            Dictionary containing predictions with different inference types:
            - 'weighted_mean': Expected value E[Y|X]
            - 'argmax_mean': Mean of most probable component
            - 'pi': Mixture weights
            - 'mu': Component means
            - 'sigma': Component standard deviations
        """
        # Handle different batch formats
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        pi, mu, sigma = self.forward(x)

        # Generate predictions using different inference types
        weighted_mean_pred = self.model.predict(x, inference_type="weighted_mean")
        argmax_mean_pred = self.model.predict(x, inference_type="argmax_mean")

        return {
            "weighted_mean": weighted_mean_pred,
            "argmax_mean": argmax_mean_pred,
            "pi": pi,
            "mu": mu,
            "sigma": sigma,
        }

    def configure_optimizers(self) -> dict[str, Any] | Optimizer:
        """Configure optimizer and optional learning rate scheduler.

        Uses timm's create_optimizer_v2 for flexible optimizer creation.
        Supported optimizers: adam, adamw, sgd, momentum, rmsprop, adagrad,
        adabelief, lamb, lars, madgrad, etc.

        Returns
        -------
        config : dict or Optimizer
            If scheduler is configured, returns a dictionary with optimizer and
            scheduler configuration. Otherwise, returns the optimizer only.
        """
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.optimizer_type,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_config is None:
            return optimizer

        scheduler = self.scheduler_config.create_scheduler(optimizer)
        lit_lr_sched_cfg = self.scheduler_config.get_lightning_config(scheduler)

        return {"optimizer": optimizer, "lr_scheduler": lit_lr_sched_cfg}
