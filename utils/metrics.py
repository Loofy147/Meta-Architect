import torch
import numpy as np
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Utility functions for computing telemetry metrics."""

    @staticmethod
    def compute_condition_number(matrix: torch.Tensor) -> float:
        """Compute condition number κ = σ_max / σ_min."""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return (S.max() / (S.min() + 1e-8)).item()

    @staticmethod
    def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # attention_weights: [batch, seq_len, seq_len]
        entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(
            dim=-1
        )
        return entropy.mean().item()

    @staticmethod
    def compute_gradient_norm(parameters: List[torch.nn.Parameter]) -> float:
        """Compute global gradient norm."""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    @staticmethod
    def compute_spectral_radius(matrix: torch.Tensor) -> float:
        """Compute spectral radius (largest singular value)."""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return S.max().item()

    @staticmethod
    def compute_rank_estimate(matrix: torch.Tensor, threshold: float = 1e-5) -> int:
        """Estimate rank of matrix."""
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return (S > threshold).sum().item()
