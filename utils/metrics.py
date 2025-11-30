import torch
import numpy as np
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Provides a collection of static methods for computing telemetry metrics.

    This class centralizes the calculation of various metrics used by the
    `TelemetryCollector` to monitor the health and performance of the HAMHA
    model. All methods are static, so no instance of the class is required.
    """

    @staticmethod
    def compute_condition_number(matrix: torch.Tensor) -> float:
        """Computes the condition number (κ) of a matrix.

        The condition number is the ratio of the largest singular value to the
        smallest singular value. A high condition number can indicate that the
        matrix is ill-conditioned and close to being singular, which is a sign
        of potential rank collapse.

        Args:
            matrix (torch.Tensor): The input matrix.

        Returns:
            float: The condition number of the matrix.
        """
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return (S.max() / (S.min() + 1e-8)).item()

    @staticmethod
    def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
        """Computes the entropy of an attention distribution.

        Entropy measures the uncertainty or "peakedness" of the attention
        distribution. A low entropy indicates that a head is "fixated" on a
        small number of tokens, while a high entropy means it attends more
        uniformly.

        Args:
            attention_weights (torch.Tensor): The attention weights tensor,
                typically of shape [batch_size, seq_len, seq_len].

        Returns:
            float: The mean entropy of the attention distribution.
        """
        # attention_weights: [batch, seq_len, seq_len]
        entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(
            dim=-1
        )
        return entropy.mean().item()

    @staticmethod
    def compute_gradient_norm(parameters: List[torch.nn.Parameter]) -> float:
        """Computes the total L2 norm of the gradients for a list of parameters.

        This is used to monitor for vanishing or exploding gradients.

        Args:
            parameters (List[torch.nn.Parameter]): A list of model parameters
                for which to compute the gradient norm.

        Returns:
            float: The total L2 norm of the gradients.
        """
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    @staticmethod
    def compute_spectral_radius(matrix: torch.Tensor) -> float:
        """Computes the spectral radius of a matrix.

        The spectral radius is the largest singular value of the matrix.

        Args:
            matrix (torch.Tensor): The input matrix.

        Returns:
            float: The spectral radius of the matrix.
        """
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return S.max().item()

    @staticmethod
    def compute_rank_estimate(matrix: torch.Tensor, threshold: float = 1e-5) -> int:
        """Estimates the numerical rank of a matrix.

        The rank is estimated by counting the number of singular values that
        are greater than a small threshold.

        Args:
            matrix (torch.Tensor): The input matrix.
            threshold (float, optional): The tolerance threshold for singular
                values. Defaults to 1e-5.

        Returns:
            int: The estimated rank of the matrix.
        """
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return (S > threshold).sum().item()
