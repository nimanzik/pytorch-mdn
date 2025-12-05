"""Tests for activation functions and classes."""

from __future__ import annotations

import pytest
import torch

from pytorch_mdn.activations import ElevatedELU


class TestElevatedELU:
    """Tests for ElevatedELU activation function."""

    @pytest.fixture
    def activation(self):
        """Fixture providing an ElevatedELU instance for tests."""
        return ElevatedELU()

    def test_initialization(self, activation):
        """Test that ElevatedELU initialises with alpha=1.0."""
        assert activation.alpha == 1.0

    @pytest.mark.parametrize(
        ("input_values", "expected_values"),
        [
            # Positive inputs: ELU(x) = x, so ElevatedELU(x) = x + 1
            ([1.0, 2.0, 3.0, 5.0], [2.0, 3.0, 4.0, 6.0]),  # x > 0 -> Act = x + 1
            ([0.0], [1.0]),  # x = 0 -> Act = 1
            ([1.5], [2.5]),
        ],
    )
    def test_positive_and_zero_inputs(self, activation, input_values, expected_values):
        """Test ElevatedELU with positive and zero inputs.

        For x >= 0, ELU(x) = x, so ElevatedELU(x) = x + 1.
        """
        x = torch.tensor(input_values)
        expected = torch.tensor(expected_values)

        output = activation(x)

        assert torch.allclose(output, expected)

    @pytest.mark.parametrize(
        "input_values", [[-1.0, -2.0, -0.5], [-0.1], [-5.0, -3.5, -1.2]]
    )
    def test_negative_inputs(self, activation, input_values):
        """Test ElevatedELU with negative inputs.

        For negative inputs, ELU(x) = alpha * (exp(x) - 1).
        With alpha=1.0, ELU(x) = exp(x) - 1.
        So ElevatedELU(x) = exp(x) - 1 + 1 = exp(x).
        """
        x = torch.tensor(input_values)
        expected = torch.exp(x)
        output = activation(x)
        assert torch.allclose(output, expected, rtol=1e-5)

    def test_mixed_inputs(self, activation):
        """Test ElevatedELU with mixed positive and negative inputs."""
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Expected: [exp(-2), exp(-1), 1, 2, 3]
        expected = torch.tensor(
            [
                torch.exp(torch.tensor(-2.0)).item(),
                torch.exp(torch.tensor(-1.0)).item(),
                1.0,
                2.0,
                3.0,
            ]
        )
        output = activation(x)
        assert torch.allclose(output, expected, rtol=1e-5)

    def test_strictly_positive_output(self, activation):
        """Test that ElevatedELU always produces strictly positive outputs.

        This is the key property: all outputs should be > 0.
        """
        # Test with a wide range of inputs
        x = torch.linspace(-10, 10, 100)
        output = activation(x)
        assert torch.all(output > 0)

    def test_minimum_output_value(self, activation):
        """Test that the minimum output approaches 0 as input → -∞.

        For very negative inputs, ElevatedELU(x) = exp(x) → 0.
        Due to floating point precision, extremely negative values may
        underflow to exactly 0, but moderately negative values should be
        close to 0 but positive.
        """
        # Moderately negative input (not so extreme to cause underflow)
        x = torch.tensor([-10.0])
        output = activation(x)

        # Should be very close to 0 but positive
        assert output > 0
        assert output < 0.01

    def test_batch_processing(self, activation):
        """Test ElevatedELU with batched inputs."""
        # 2D tensor with batch_size=3, n_features=4
        x = torch.tensor(
            [[-1.0, 0.0, 1.0, 2.0], [-2.0, -0.5, 0.5, 1.5], [0.0, 1.0, 2.0, 3.0]]
        )

        output = activation(x)

        assert output.shape == x.shape
        assert torch.all(output > 0)

    def test_gradient_flow(self, activation):
        """Test that gradients flow correctly through ElevatedELU."""
        x = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
        output = activation(x)
        loss = output.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype_preservation(self, activation, dtype):
        """Test that ElevatedELU preserves input dtype."""
        x = torch.tensor([1.0, -1.0], dtype=dtype)
        output = activation(x)
        assert output.dtype == dtype

    def test_device_compatibility(self, activation):
        """Test that ElevatedELU works on different devices."""
        # CPU
        x_cpu = torch.tensor([1.0, -1.0, 0.0])
        output_cpu = activation(x_cpu)
        assert output_cpu.device.type == "cpu"
        assert torch.all(output_cpu > 0)

        # GPU (if available)
        if torch.cuda.is_available():
            x_cuda = x_cpu.cuda()
            activation_cuda = ElevatedELU().cuda()
            output_cuda = activation_cuda(x_cuda)
            assert output_cuda.device.type == "cuda"
            assert torch.all(output_cuda > 0)

    @pytest.mark.parametrize(
        ("x_value", "expected_value", "tolerance"),
        [
            (-1e-6, 1.0, 1e-5),  # Left of zero
            (0.0, 1.0, 1e-7),  # At zero
            (1e-6, 1.0, 1e-5),  # Right of zero
        ],
    )
    def test_continuity_at_zero(self, activation, x_value, expected_value, tolerance):
        """Test that ElevatedELU is continuous at x=0.

        The function should be continuous, especially at x=0 where ELU
        transitions from exponential to linear.
        """
        x = torch.tensor([x_value])
        y = activation(x)

        assert torch.allclose(y, torch.tensor([expected_value]), atol=tolerance)

    def test_comparison_with_manual_computation(self, activation):
        """Test against manual computation of ElevatedELU."""
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

        elu_output = torch.where(x > 0, x, 1.0 * (torch.exp(x) - 1))  # alpha=1.0
        expected = elu_output + 1.0

        output = activation(x)

        assert torch.allclose(output, expected, rtol=1e-5)
