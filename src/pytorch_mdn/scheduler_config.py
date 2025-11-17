from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class BaseSchedulerConfig(BaseModel, ABC):
    """Base class for all scheduler configurations."""

    @abstractmethod
    def create_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Create scheduler instance."""
        ...

    @abstractmethod
    def get_lightning_config(self, scheduler: LRScheduler) -> dict:
        """Get Lightning-specific configuration for this scheduler."""
        ...


class ReduceLROnPlateauConfig(BaseSchedulerConfig, extra="forbid"):
    """Configuration for ReduceLROnPlateau scheduler."""

    type: Literal["reduce_on_plateau"] = "reduce_on_plateau"
    monitor: str = Field(..., description="Metric to monitor")
    mode: Literal["min", "max"] = Field(
        ..., description="'min' for loss, 'max' for accuracy"
    )
    factor: float = Field(default=0.1, gt=0, lt=1, description="Factor to reduce LR by")
    patience: int = Field(
        default=10, ge=0, description="Number of epochs with no improvement"
    )
    threshold: float = Field(
        default=1e-4, ge=0, description="Threshold for measuring improvement"
    )
    threshold_mode: Literal["rel", "abs"] = Field(
        default="rel", description="Relative or absolute threshold"
    )
    cooldown: int = Field(
        default=0, ge=0, description="Epochs to wait before resuming normal operation"
    )
    min_lr: float = Field(default=1e-6, ge=0, description="Minimum learning rate")

    def create_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Create ReduceLROnPlateau scheduler instance."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
        )

    def get_lightning_config(self, scheduler: LRScheduler) -> dict:
        """Get Lightning-specific configuration for this scheduler."""
        return {
            "scheduler": scheduler,
            "monitor": self.monitor,
            "strict": True,
            "interval": "epoch",
            "frequency": 1,
        }


class StepLRConfig(BaseSchedulerConfig, extra="forbid"):
    """Configuration for StepLR scheduler."""

    type: Literal["step"] = "step"
    step_size: int = Field(..., gt=0, description="Period of learning rate decay")
    gamma: float = Field(
        default=0.1, gt=0, lt=1, description="Multiplicative factor of LR decay"
    )

    def create_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Create StepLR scheduler instance."""
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )

    def get_lightning_config(self, scheduler: LRScheduler) -> dict:
        """Get Lightning-specific configuration for this scheduler."""
        return {"scheduler": scheduler}


class CosineAnnealingLRConfig(BaseSchedulerConfig, extra="forbid"):
    """Configuration for CosineAnnealingLR scheduler."""

    type: Literal["cosine"] = "cosine"
    T_max: int = Field(..., gt=0, description="Maximum number of iterations")
    eta_min: float = Field(default=0.0, ge=0, description="Minimum learning rate")

    def create_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Create CosineAnnealingLR scheduler instance."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
        )

    def get_lightning_config(self, scheduler: LRScheduler) -> dict:
        """Get Lightning-specific configuration for this scheduler."""
        return {"scheduler": scheduler}


class CosineAnnealingWarmRestartsConfig(BaseSchedulerConfig, extra="forbid"):
    """Configuration for CosineAnnealingWarmRestarts scheduler."""

    type: Literal["cosine_warm_restarts"] = "cosine_warm_restarts"
    T_0: int = Field(
        ..., gt=0, description="Number of iterations for the first restart"
    )
    T_mult: int = Field(
        default=1, ge=1, description="Factor to increase T_i after a restart"
    )
    eta_min: float = Field(default=0.0, ge=0, description="Minimum learning rate")

    def create_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """Create CosineAnnealingWarmRestarts scheduler instance."""
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
        )

    def get_lightning_config(self, scheduler: LRScheduler) -> dict:
        """Get Lightning-specific configuration for this scheduler."""
        return {"scheduler": scheduler}


# Union type for all scheduler configs
SchedulerConfig = (
    ReduceLROnPlateauConfig
    | StepLRConfig
    | CosineAnnealingLRConfig
    | CosineAnnealingWarmRestartsConfig
)
