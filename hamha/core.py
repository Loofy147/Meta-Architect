import torch
import torch.nn as nn
import math
import copy
from typing import Dict
import torch.func
from hamha.topology import generate_hex_grid, build_adjacency_matrix
from hamha.heads import CoordinateBiasFunction, HyperNetwork, AttentionHead
from hamha.mixing import GNNMixingLayer
from hamha.spectral.attention import SpectralAttentionLayer


# Module-level cache for eigendecomposition
EIGEN_CACHE = {}


class HexagonalMultiHeadAttention(nn.Module):
    """Implements the Hexagonal Multi-Head Attention (HAMHA) mechanism.

    This module arranges attention heads in a hexagonal grid topology, allowing
    for efficient spatial reasoning and information flow between heads. It can
    operate in two modes:
    1.  **Standard Mode**: Each head computes attention independently, and their
        outputs are mixed using a Graph Neural Network (GNN).
    2.  **Spectral Mode**: The attention mechanism is reformulated in the graph
        Fourier domain, using the eigenvectors of the graph Laplacian for a
        more global and efficient computation.

    The module also supports dynamic weight generation via a HyperNetwork and
    is designed to be governed by the Lead Meta-Architect (LMA), which can
    adjust parameters like entropy regularization.

    Attributes:
        d_model (int): The total dimensionality of the input and output.
        d_head (int): The dimensionality of each attention head.
        use_spectral (bool): If True, operates in spectral mode.
        grid_coords (list): A list of HexCoordinate objects representing the grid.
        num_heads (int): The total number of attention heads.
        coord_to_idx (dict): A mapping from HexCoordinate to an integer index.
        spectral_attention (SpectralAttentionLayer): The spectral attention module.
        bias_function (CoordinateBiasFunction): Generates coordinate-based biases.
        hypernet (HyperNetwork): Generates head weights dynamically.
        heads (nn.ModuleList): A list of standard AttentionHead modules.
        gnn_mixing (GNNMixingLayer): The GNN layer for mixing head outputs.
        W_O (nn.Parameter): The output projection matrix.
        entropy_reg (float): A coefficient for entropy regularization, controlled
            externally by the LMA.
    """

    def __init__(
        self,
        d_model: int,
        grid_radius: int = 2,
        d_head: int = 64,
        use_hypernet: bool = False,
        use_spectral: bool = False,
        k_eigenvectors: int = 16,
    ):
        """Initializes the HexagonalMultiHeadAttention module.

        Args:
            d_model (int): The total dimensionality of the input feature space.
            grid_radius (int, optional): The radius of the hexagonal grid. The
                total number of heads will be 3*r^2 + 3*r + 1. Defaults to 2.
            d_head (int, optional): The dimensionality of each individual
                attention head. Defaults to 64.
            use_hypernet (bool, optional): If True, a HyperNetwork is used to
                generate query, key, and value projection matrices based on
                head coordinates. Not compatible with spectral mode.
                Defaults to False.
            use_spectral (bool, optional): If True, the module operates in
                spectral attention mode. This mode is more efficient for large
                grids. Defaults to False.
            k_eigenvectors (int, optional): The number of eigenvectors to use
                for the spectral attention mechanism. This value is capped by
                the total number of heads. Defaults to 16.
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.use_spectral = use_spectral
        self.grid_coords = generate_hex_grid(grid_radius)
        self.num_heads = len(self.grid_coords)
        self.coord_to_idx = {coord: i for i, coord in enumerate(self.grid_coords)}

        if use_spectral:
            # The number of eigenvectors cannot exceed the number of nodes (heads)
            actual_k = min(k_eigenvectors, self.num_heads)

            self.spectral_attention = SpectralAttentionLayer(
                d_model, self.num_heads, d_head, actual_k
            )
            if grid_radius not in EIGEN_CACHE:
                adj = build_adjacency_matrix(self.grid_coords)
                degree = torch.diag(adj.sum(dim=1))
                laplacian = degree - adj
                eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
                EIGEN_CACHE[grid_radius] = (
                    eigenvectors[:, :actual_k],
                    eigenvalues[:actual_k],
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
        """Performs the forward pass for the HAMHA module.

        Args:
            x (torch.Tensor): The input tensor of shape
                [batch_size, seq_len, d_model]. If the input is 2D, it is
                automatically unsqueezed to add a batch dimension of 1.

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, seq_len, d_model].
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if self.use_spectral:
            spectral_output = self.spectral_attention(x)
            return torch.matmul(spectral_output, self.W_O)
        else:
            x_global = x.mean(dim=1)
            head_outputs = [head(x, x_global, self.entropy_reg) for head in self.heads]
            mixed_outputs = self.gnn_mixing(head_outputs)
            concatenated = torch.cat(mixed_outputs, dim=-1)
            return torch.matmul(concatenated, self.W_O)

    def forward_functional(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x: torch.Tensor
    ) -> torch.Tensor:
        """A functional version of the forward pass for use with `torch.func`.

        This method allows the model to be used in functional programming
        paradigms, such as meta-learning with higher-order gradients, by
        uncoupling the model's parameters from its definition.

        Args:
            params (Dict[str, torch.Tensor]): A dictionary of the model's
                parameters, typically obtained from `clone_for_functional`.
            buffers (Dict[str, torch.Tensor]): A dictionary of the model's
                buffers.
            x (torch.Tensor): The input tensor for the forward pass.

        Returns:
            torch.Tensor: The output of the model for the given input and
                parameters.
        """
        return torch.func.functional_call(self, (params, buffers), (x,))

    def clone_for_functional(self):
        """Prepares the model for functional programming.

        This method captures the model's current state (parameters and buffers)
        in dictionaries, which can then be passed to `forward_functional`.

        Returns:
            tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple
                containing a dictionary of named parameters and a dictionary of
                named buffers.
        """
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        return params, buffers
