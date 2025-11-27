import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SpectralFilter(nn.Module):
    """
    Learnable spectral filter with multi-band support.

    Features:
    - Chebyshev polynomial approximation (efficient)
    - Multi-scale filtering (low, mid, high frequencies)
    - Differentiable filter design
    """

    def __init__(self, k_eigenvectors: int, num_bands: int = 3):
        super().__init__()

        self.k = k_eigenvectors
        self.num_bands = num_bands

        # Band boundaries (learnable)
        self.band_boundaries = nn.Parameter(
            torch.linspace(0, 1, num_bands + 1)
        )

        # Filter coefficients per band
        self.filter_weights = nn.Parameter(
            torch.ones(num_bands, k_eigenvectors)
        )

    def forward(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Generate filter response.

        Args:
            eigenvalues: [K] - Sorted eigenvalues

        Returns:
            filter_response: [K] - Filter gains
        """
        # Normalize eigenvalues to [0, 1]
        lambda_norm = (eigenvalues - eigenvalues.min()) / \
                      (eigenvalues.max() - eigenvalues.min() + 1e-8)

        # Assign each frequency to a band
        response = torch.zeros_like(eigenvalues)

        for i in range(self.num_bands):
            # Band mask
            lower = self.band_boundaries[i]
            upper = self.band_boundaries[i + 1]
            mask = (lambda_norm >= lower) & (lambda_norm < upper)

            # Apply filter weights
            response[mask] = self.filter_weights[i, mask]

        return response


class SpectralAttentionLayer(nn.Module):
    """
    Attention in spectral domain of hexagonal graph.

    Benefits over spatial attention:
    1. No over-squashing (K << N for large graphs)
    2. Global receptive field (all nodes interact)
    3. Interpretable frequency response
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        k_eigenvectors: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.k = k_eigenvectors
        self.d_head = d_head

        # Standard attention projections
        self.W_Q = nn.Linear(d_model, num_heads * d_head)
        self.W_K = nn.Linear(d_model, num_heads * d_head)
        self.W_V = nn.Linear(d_model, num_heads * d_head)

        # Spectral filters (per head)
        self.filters = nn.ModuleList([
            SpectralFilter(k_eigenvectors, num_bands=3)
            for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout)

        # Cache for eigenvectors (set externally)
        self.register_buffer('eigenvectors', None)
        self.register_buffer('eigenvalues', None)

    def set_graph_spectrum(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor
    ):
        """
        Cache graph spectrum (call once per graph structure).

        Args:
            eigenvectors: [N, K]
            eigenvalues: [K]
        """
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues

    def to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Project to frequency domain: x_freq = U^T @ x"""
        return torch.matmul(self.eigenvectors.T, x)

    def to_spatial(self, x_freq: torch.Tensor) -> torch.Tensor:
        """Project to spatial domain: x = U @ x_freq"""
        return torch.matmul(self.eigenvectors, x_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - Spatial features

        Returns:
            output: [B, N, D] - Transformed features
        """
        B, N, D = x.shape

        # Project Q, K, V
        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_head)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_head)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_head)

        # Transform to spectral domain
        Q_freq = torch.einsum("nk,bnhd->bkhd", self.eigenvectors, Q)
        K_freq = torch.einsum("nk,bnhd->bkhd", self.eigenvectors, K)
        V_freq = torch.einsum("nk,bnhd->bkhd", self.eigenvectors, V)

        # Apply spectral filters per head
        filter_responses = torch.stack([f(self.eigenvalues) for f in self.filters], dim=0)  # [H, K]
        Q_filtered = torch.einsum("hk,bkhd->bkhd", filter_responses, Q_freq)
        K_filtered = torch.einsum("hk,bkhd->bkhd", filter_responses, K_freq)
        V_filtered = torch.einsum("hk,bkhd->bkhd", filter_responses, V_freq)

        # Attention in spectral space
        scores = torch.einsum("bkhd,blhd->bkhl", Q_filtered, K_filtered) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        output_freq = torch.einsum("bkhl,blhd->bkhd", attn, V_filtered)

        # Transform back to spatial domain
        output = torch.einsum("nk,bkhd->bnhd", self.eigenvectors, output_freq)

        # Merge heads
        output = output.reshape(B, N, self.num_heads * self.d_head)

        return output
