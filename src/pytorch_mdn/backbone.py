from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torch import Tensor as TorchTensor

from .activations import ACTIVATION_FUNCTIONS


class MLPBackbone(nn.Module):
    """Multi-layer perceptron backbone with configurable normalisation.

    This class serves as a feature extractor, transforming inputs through
    multiple hidden layers. It does not include an output layer, making it
    suitable as a shared backbone for models with multiple output heads.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dims : list[int]
        List of hidden layer dimensions.
    activation_type : str
        Activation function name. Must be one of "relu", "gelu", "mish",
        "silu".
    norm_type : {"batch", "layer"} or None, default=None
        Type of normalisation layer to use. "batch" for batch normalisation,
        "layer" for layer normalisation, or None for no normalisation.
    norm_after_act : bool, default=False
        Whether to apply normalisation after activation functions. If False,
        normalisation is applied before activation.
    dropout_rate : float, default=0.0
        Dropout probability for regularisation. Must be between 0.0 and 1.0.
        No dropout is applied if 0.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation_type: str,
        norm_type: Literal["batch", "layer"] | None = None,
        norm_after_act: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

        activation_fn = ACTIVATION_FUNCTIONS.get(activation_type.lower(), nn.GELU)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            block = [nn.Linear(prev_dim, hidden_dim), activation_fn()]

            if norm_type == "batch":
                norm_layer = nn.BatchNorm1d(hidden_dim)
            elif norm_type == "layer":
                norm_layer = nn.LayerNorm(hidden_dim)
            else:
                norm_layer = None

            if norm_layer is not None:
                if norm_after_act:
                    block.append(norm_layer)
                else:
                    block.insert(1, norm_layer)

            if dropout_rate > 0.0:
                block.append(nn.Dropout(dropout_rate))

            layers.extend(block)
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)

    def forward(self, x: TorchTensor) -> TorchTensor:
        """Forward pass through the MLP backbone.

        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Input tensor.

        Returns
        -------
        output : Tensor of shape (batch_size, output_dim)
            Output features after passing through all hidden layers.
            `output_dim` is equal to the last element in `hidden_dims`.
        """
        return self.layers(x)
