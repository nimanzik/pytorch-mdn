from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, field_validator

from ..mixup_config import MixUpConfig
from ..scheduler_config import SchedulerConfig

if TYPE_CHECKING:
    from typing import Self


class MDNTrainingConfig(BaseModel, extra="forbid"):
    """Configuration for MDN training with PyTorch Lightning."""

    input_dim: int = Field(..., gt=0, description="Dimension of input features")
    hidden_dims: list[int] = Field(..., description="Hidden layer dimensions")
    output_dim: int = Field(..., gt=0, description="Dimension of output targets")
    n_components: int = Field(..., gt=0, description="Number of mixture components")
    activation_type: str = Field(..., description="Activation function name")
    bn_after_act: bool = Field(
        default=False, description="Apply BatchNorm after activations."
    )
    optimizer_type: str = Field(
        default="adamw", description="Optimizer type (e.g., adamw, sgd)"
    )
    learning_rate: float = Field(..., gt=0, description="Learning rate")
    weight_decay: float = Field(
        default=0.0, ge=0, description="Weight decay coefficient"
    )
    scheduler_config: SchedulerConfig | None = Field(
        default=None, description="Learning-rate scheduler configuration"
    )
    mixup_config: MixUpConfig | None = Field(
        default=None, description="MixUp augmentation configuration"
    )

    @field_validator("hidden_dims", mode="before")
    @classmethod
    def ensure_hidden_dims_list(cls, v: list[int] | int) -> list[int]:
        """Ensure `hidden_dims` is always a list of integers."""
        if isinstance(v, int):
            return [v]
        return v

    @classmethod
    def from_yaml(cls, config_fpath: str | Path) -> Self:
        """Load configuration from a YAML file."""
        config_fpath = Path(config_fpath)

        with config_fpath.open("r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_json(cls, config_fpath: str | Path) -> Self:
        """Load configuration from a JSON file."""
        config_fpath = Path(config_fpath)

        with config_fpath.open("r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with Path(path).open("w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Save configuration to a JSON file."""
        with Path(path).open("w") as f:
            json.dump(self.model_dump(), f, indent=indent)
