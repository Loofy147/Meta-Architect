import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
import numpy as np


class SpectralMixingLayer(nn.Module):
    """
    Applies a spectral filter to the head outputs using the graph Laplacian.
    """

    def __init__(
        self,
        d_head: int,
        num_heads: int,
        adjacency_matrix: torch.Tensor,
    ):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads

        # --- Eigendecomposition of the Graph Laplacian ---
        L, U, eigenvalues = self._compute_laplacian_eigen(adjacency_matrix)
        self.register_buffer("L", L)  # Laplacian
        self.register_buffer("U", U)  # Eigenvectors
        self.register_buffer("eigenvalues", eigenvalues)

        # --- Learnable Spectral Filter ---
        # Initialize filters close to identity (pass-through)
        self.spectral_filter = nn.Parameter(torch.ones(num_heads, 1))

        self.norm = nn.LayerNorm(d_head)

    def _compute_laplacian_eigen(self, adj):
        """Computes the normalized graph Laplacian and its eigendecomposition."""
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        return L, eigenvectors, eigenvalues

    def forward(self, head_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply spectral graph filtering.

        Args:
            head_outputs: List of [batch, seq_len, d_head] tensors

        Returns:
            List of mixed outputs with same shapes
        """
        # Stack all heads: [num_heads, batch, seq_len, d_head]
        stacked = torch.stack(head_outputs, dim=0)

        # --- Graph Fourier Transform ---
        # Project inputs onto eigenvectors
        # [heads, b, s, d] -> [b, s, d, heads]
        x_reshaped = stacked.permute(1, 2, 3, 0)
        # [b, s, d, heads] @ [heads, heads] -> [b, s, d, heads]
        x_spectral = torch.matmul(x_reshaped, self.U)

        # --- Apply Spectral Filter ---
        # [b, s, d, heads] * [heads, 1] -> [b, s, d, heads]
        filtered_spectral = x_spectral * self.spectral_filter.T

        # --- Inverse Graph Fourier Transform ---
        # Project back to node domain
        # [b, s, d, heads] @ [heads, heads] -> [b, s, d, heads]
        mixed_reshaped = torch.matmul(filtered_spectral, self.U.T)

        # [b, s, d, heads] -> [heads, b, s, d]
        mixed = mixed_reshaped.permute(3, 0, 1, 2)

        # Apply normalization
        mixed = self.norm(mixed)

        return [mixed[i] for i in range(self.num_heads)]

    def get_mixing_statistics(self) -> dict:
        """Return diagnostic information about mixing patterns."""
        return {
            "spectral_filter_weights": self.spectral_filter.detach().cpu().numpy().flatten(),
            "eigenvalues": self.eigenvalues.detach().cpu().numpy()
        }

def benchmark_mixing_layers():
    """Compare performance of original vs optimized mixing."""
    import time

    class GNNMixingLayer(nn.Module):
        """Vectorized GNN mixing using sparse matrix operations."""

        def __init__(
            self,
            d_head: int,
            num_heads: int,
            adjacency_matrix: torch.Tensor,
        ):
            super().__init__()
            self.d_head = d_head
            self.num_heads = num_heads
            self.register_buffer("adjacency_dense", adjacency_matrix)
            self.W_self = nn.Parameter(torch.randn(d_head, d_head) / math.sqrt(d_head))
            self.W_neighbor = nn.Parameter(torch.randn(d_head, d_head) / math.sqrt(d_head))
            self.lambda_self = nn.Parameter(torch.ones(num_heads) * 0.7)
            self.g_ij = nn.Parameter(torch.ones(num_heads, num_heads) * 0.05)
            self.norm = nn.LayerNorm(d_head)

        def forward(self, head_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
            stacked = torch.stack(head_outputs, dim=0)
            self_contrib = torch.einsum('hbsd,de->hbse', stacked, self.W_self)
            self_contrib = self_contrib * self.lambda_self.view(-1, 1, 1, 1)
            transformed = torch.einsum('hbsd,de->hbse', stacked, self.W_neighbor)
            edge_weights = self.g_ij * self.adjacency_dense
            neighbor_contrib = torch.einsum(
                'ts,sbde->tbde',
                edge_weights,
                transformed
            )
            mixed = F.relu(self_contrib + neighbor_contrib)
            mixed = self.norm(mixed)
            return [mixed[i] for i in range(self.num_heads)]

    d_head = 64
    num_heads = 19  # radius=2
    batch_size = 4
    seq_len = 128

    # Create dummy adjacency
    adjacency = torch.randn(num_heads, num_heads).abs()
    adjacency = (adjacency > 0.5).float()
    adjacency = (adjacency + adjacency.T) / 2 # Symmetrize

    # Create dummy inputs
    head_outputs = [torch.randn(batch_size, seq_len, d_head) for _ in range(num_heads)]

    # GNN Layer
    gnn_layer = GNNMixingLayer(d_head, num_heads, adjacency)

    # Spectral Layer
    spectral_layer = SpectralMixingLayer(d_head, num_heads, adjacency)

    # Benchmark
    iterations = 100

    # GNN
    start = time.time()
    for _ in range(iterations):
        _ = gnn_layer(head_outputs)
    gnn_time = time.time() - start

    # Spectral
    start = time.time()
    for _ in range(iterations):
        _ = spectral_layer(head_outputs)
    spectral_time = time.time() - start

    print(f"GNN Layer: {gnn_time:.4f}s")
    print(f"Spectral Layer: {spectral_time:.4f}s")
    print(f"Performance Ratio (GNN/Spectral): {gnn_time / spectral_time:.2f}x")


if __name__ == "__main__":
    benchmark_mixing_layers()
