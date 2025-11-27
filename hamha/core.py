import torch
import torch.nn as nn
import math
from hamha.topology import generate_hex_grid, build_adjacency_matrix
from hamha.heads import CoordinateBiasFunction, HyperNetwork, AttentionHead
from hamha.mixing import GNNMixingLayer
from hamha.spectral.attention import SpectralAttentionLayer


# Module-level cache for eigendecomposition
EIGEN_CACHE = {}


class HexagonalMultiHeadAttention(nn.Module):
    """Complete HAMHA mechanism with hexagonal topology."""

    def __init__(
        self,
        d_model: int,
        grid_radius: int = 2,
        d_head: int = 64,
        use_hypernet: bool = False,
        use_spectral: bool = False,
        k_eigenvectors: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.use_spectral = use_spectral
        self.grid_coords = generate_hex_grid(grid_radius)
        self.num_heads = len(self.grid_coords)
        self.coord_to_idx = {coord: i for i, coord in enumerate(self.grid_coords)}

        if use_spectral:
            self.spectral_attention = SpectralAttentionLayer(
                d_model, self.num_heads, d_head, k_eigenvectors
            )
            if grid_radius not in EIGEN_CACHE:
                adj = build_adjacency_matrix(self.grid_coords)
                degree = torch.diag(adj.sum(dim=1))
                laplacian = degree - adj
                eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
                EIGEN_CACHE[grid_radius] = (
                    eigenvectors[:, :k_eigenvectors],
                    eigenvalues[:k_eigenvectors],
                )
            eigenvectors, eigenvalues = EIGEN_CACHE[grid_radius]
            self.spectral_attention.set_graph_spectrum(eigenvectors, eigenvalues)
        else:
            self.bias_function = CoordinateBiasFunction(d_model, d_head)
            self.hypernet = HyperNetwork(d_model, d_head) if use_hypernet else None

            self.heads = nn.ModuleList(
                [
                    AttentionHead(
                        coord,
                        d_model,
                        d_head,
                        use_hypernet,
                        self.bias_function,
                        self.hypernet,
                    )
                    for coord in self.grid_coords
                ]
            )

            adjacency = build_adjacency_matrix(self.grid_coords)
            self.gnn_mixing = GNNMixingLayer(d_head, self.num_heads, adjacency)

        self.W_O = nn.Parameter(
            torch.randn(self.num_heads * d_head, d_model) / math.sqrt(d_model)
        )

        # Entropy regularization coefficient (controlled by LMA)
        self.entropy_reg = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension

        if self.use_spectral:
            # Spectral attention provides a global context, making the local GNN mixing layer redundant
            spectral_output = self.spectral_attention(x)
            return torch.matmul(spectral_output, self.W_O)
        else:
            x_global = x.mean(dim=1)
            head_outputs = [head(x, x_global, self.entropy_reg) for head in self.heads]
            mixed_outputs = self.gnn_mixing(head_outputs)
            concatenated = torch.cat(mixed_outputs, dim=-1)
            return torch.matmul(concatenated, self.W_O)
