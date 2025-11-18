# PyTorch MDN

***A PyTorch implementation of Mixture Density Networks (MDN) for modelling multi-modal distributions in regression tasks.***

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A Mixture Density Network (MDN) is a neural network that predicts the parameters of a mixture of distributions rather than a single point estimate. This makes MDN particularly useful for regression problems where:

- The relationship between inputs and outputs is one-to-many (inverse problems).
- The target distribution is multi-modal or highly uncertain.

This implementation uses Gaussian mixture models where each component shares the same standard deviation across output dimensions (isotropic Gaussians).

## Features

- **Multiple inference modes**:
  - `weighted_mean`: Expected value E[Y|X] computed as weighted average of component means.
  - `argmax_mean`: Mean of the most probable component (it's a fast approximation).
  - `sample_[mean|median]`: Mean/median of samples drawn from the mixture distribution.
- **PyTorch Lightning integration**: Built-in training module for easy training and validation.
- **Pydantic-based configuration**: Type-safe model and training setup.
- **C-Mixup support**: Data augmentation that improves generalisation on regression tasks.

## Installation

You can add (install) the package to your project using `uv`:

```bash
uv add git+https://github.com/nimanzik/pytorch-mdn.git
```

## Quick Start

```python
import torch
from pytorch_mdn import MixtureDensityNetwork, mdn_loss

input_dim = 12  # Number of input features
output_dim = 3  # Number of output dimensions
batch_size = 32

model = MixtureDensityNetwork(
    input_dim=input_dim,
    hidden_dims=[128, 64, 32],
    output_dim=output_dim,
    n_components=5,
    activation_type="gelu",
)

# Forward pass: get mixture parameters
x = torch.randn(batch_size, input_dim)
log_pi, mu, sigma = model(x)

# Compute loss
y_true = torch.randn(batch_size, output_dim)
loss = mdn_loss(log_pi, mu, sigma, y_true)

# Generate predictions
predictions = model.predict(x, inference_type="sample_median")

# Sample from the mixture distribution
samples = model.generate_samples(x, n_samples=1_000)

# Predict specific quantiles
quantiles = model.predict_quantiles(x, quantiles=[0.05, 0.5, 0.95])
```

## Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.9.0
- PyTorch Lightning ≥ 2.5.6
- timm ≥ 1.0.22
- Pydantic ≥ 2.12.4
- NumPy ≥ 2.3.4
- scikit-learn ≥ 1.7.2

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
