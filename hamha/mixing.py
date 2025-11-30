import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

class GNNMixingLayer(nn.Module):
    """Mixes attention head outputs using a Graph Neural Network (GNN).

    This layer treats the attention heads as nodes in a graph and updates each
    head's output by aggregating information from its neighbors. The mixing
    is performed using a vectorized approach with batch matrix multiplication
    for efficiency, avoiding explicit loops over heads.

    Each head's new state is a combination of its own transformed state (self-
    contribution) and a weighted sum of its neighbors' transformed states
    (neighbor contribution).

    Attributes:
        d_head (int): The dimensionality of each attention head's output.
        num_heads (int): The total number of attention heads.
        adjacency_dense (torch.Tensor): A buffer storing the dense adjacency
            matrix of the head graph.
        W_self (nn.Parameter): The learnable weight matrix for the self-
            contribution.
        W_neighbor (nn.Parameter): The learnable weight matrix for the neighbor
            contributions.
        lambda_self (nn.Parameter): A learnable vector of coefficients for the
            self-contribution of each head.
        g_ij (nn.Parameter): A learnable matrix of weights for the contribution
            of each neighbor `j` to head `i`.
        norm (nn.LayerNorm): Layer normalization for stabilizing the output.
    """

    def __init__(
        self,
        d_head: int,
        num_heads: int,
        adjacency_matrix: torch.Tensor,
    ):
        """Initializes the GNNMixingLayer.

        Args:
            d_head (int): The dimensionality of each attention head.
            num_heads (int): The total number of attention heads.
            adjacency_matrix (torch.Tensor): A dense tensor of shape
                [num_heads, num_heads] representing the graph connectivity.
        """
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads

        self.register_buffer("adjacency_dense", adjacency_matrix)

        # Learnable parameters
        self.W_self = nn.Parameter(torch.randn(d_head, d_head) / math.sqrt(d_head))
        self.W_neighbor = nn.Parameter(torch.randn(d_head, d_head) / math.sqrt(d_head))

        # Per-head mixing coefficients
        self.lambda_self = nn.Parameter(torch.ones(num_heads) * 0.7)

        # Edge-specific weights (vectorized)
        self.g_ij = nn.Parameter(torch.ones(num_heads, num_heads) * 0.05)

        # Optional: Layer normalization for stability
        self.norm = nn.LayerNorm(d_head)

    def forward(self, head_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Performs the GNN mixing operation.

        Args:
            head_outputs (List[torch.Tensor]): A list of tensors, where each
                tensor is the output of an attention head. Each tensor should
                have the shape [batch_size, seq_len, d_head].

        Returns:
            List[torch.Tensor]: A list of mixed output tensors, with the same
                shape as the input tensors.
        """
        # Stack all heads: [batch, num_heads, seq_len, d_head]
        stacked_heads = torch.stack(head_outputs, dim=1)

        # Self-contribution
        self_contrib = torch.einsum('bjnd,de->bjne', stacked_heads, self.W_self)
        self_contrib *= self.lambda_self.view(1, -1, 1, 1)

        # Neighbor contribution
        transformed_neighbor = torch.einsum('bjnd,de->bjne', stacked_heads, self.W_neighbor)

        # Adjacency-weighted sum
        # A_ij is adj matrix from j to i, so we need to sum over j
        adj_and_g = self.adjacency_dense * self.g_ij

        # Reshape for matmul: move the 'j' dimension to the end for contraction
        # (batch, n_heads_j, seq, d) -> (batch, seq, d, n_heads_j)
        transformed_neighbor_reshaped = transformed_neighbor.permute(0, 2, 3, 1)

        # Matmul: (batch, seq, d, n_heads_j) @ (n_heads_j, n_heads_i) -> (batch, seq, d, n_heads_i)
        # The einsum 'ji' means we need to transpose the adjacency matrix before multiplying.
        neighbor_contrib_reshaped = torch.matmul(transformed_neighbor_reshaped, adj_and_g.T)

        # Reshape back to original format
        # (batch, seq, d, n_heads_i) -> (batch, n_heads_i, seq, d)
        neighbor_contrib = neighbor_contrib_reshaped.permute(0, 3, 1, 2)

        # Combine, activate, and normalize
        mixed = self.norm(F.relu(self_contrib + neighbor_contrib))

        # Unstack back to list
        return [mixed[:, i, :, :] for i in range(self.num_heads)]

    def get_mixing_statistics(self) -> dict:
        """Returns diagnostic information about the mixing patterns.

        This method provides insights into the learned mixing behavior, which
        can be used for analysis and monitoring by the LMA.

        Returns:
            dict: A dictionary containing key statistics:
                - 'self_weights': The learned self-contribution weights for
                  each head.
                - 'neighbor_weights': The learned neighbor-contribution weights.
                - 'avg_self_weight': The average self-contribution weight.
                - 'avg_neighbor_weight': The average neighbor-contribution
                  weight.
                - 'effective_neighbors': The number of neighbors for each head.
        """
        return {
            'self_weights': self.lambda_self.detach().cpu().numpy(),
            'neighbor_weights': self.g_ij.detach().cpu().numpy(),
            'avg_self_weight': self.lambda_self.mean().item(),
            'avg_neighbor_weight': self.g_ij.mean().item(),
            'effective_neighbors': (self.adjacency_dense.sum(dim=1)).cpu().numpy()
        }

def benchmark_mixing_layers():
    """Compares the performance of the original and optimized GNN mixing layers.

    This function sets up a benchmark to measure the speedup gained from
    vectorizing the mixing operation. It creates instances of both the
    original (loop-based) and the current (vectorized) mixing layers,
    runs them for a fixed number of iterations on dummy data, and prints
    the execution times and the speedup factor.
    """
    import time

    class OriginalGNNMixingLayer(nn.Module):
        """Graph Neural Network-based mixing of attention head outputs."""

        def __init__(self, d_head: int, num_heads: int, adjacency_matrix: torch.Tensor):
            super().__init__()
            self.d_head = d_head
            self.num_heads = num_heads
            self.register_buffer("adjacency", adjacency_matrix)

            self.W_self = nn.Parameter(torch.randn(d_head, d_head) / math.sqrt(d_head))
            self.W_neighbor = nn.Parameter(torch.randn(d_head, d_head) / math.sqrt(d_head))
            self.lambda_self = nn.Parameter(torch.ones(num_heads) * 0.7)
            self.g_ij = nn.Parameter(torch.ones(num_heads, num_heads) * 0.05)

        def forward(self, head_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
            mixed_outputs = []
            for i, H_i in enumerate(head_outputs):
                self_contrib = self.lambda_self[i] * torch.matmul(H_i, self.W_self)
                neighbor_contrib = torch.zeros_like(H_i)
                for j, H_j in enumerate(head_outputs):
                    if self.adjacency[i, j] > 0:
                        neighbor_contrib += self.g_ij[i, j] * torch.matmul(
                            H_j, self.W_neighbor
                        )
                mixed = F.relu(self_contrib + neighbor_contrib)
                mixed_outputs.append(mixed)
            return mixed_outputs

    d_head = 64
    num_heads = 19  # radius=2
    batch_size = 4
    seq_len = 128

    # Create dummy adjacency
    adjacency = torch.randn(num_heads, num_heads).abs()
    adjacency = (adjacency > 0.5).float()

    # Create dummy inputs
    head_outputs = [torch.randn(batch_size, seq_len, d_head) for _ in range(num_heads)]

    # Original
    original = OriginalGNNMixingLayer(d_head, num_heads, adjacency)

    # Optimized
    optimized = GNNMixingLayer(d_head, num_heads, adjacency)

    # Benchmark
    iterations = 100

    # Original
    start = time.time()
    for _ in range(iterations):
        _ = original(head_outputs)
    original_time = time.time() - start

    # Optimized
    start = time.time()
    for _ in range(iterations):
        _ = optimized(head_outputs)
    optimized_time = time.time() - start

    print(f"Original: {original_time:.4f}s")
    print(f"Optimized: {optimized_time:.4f}s")
    print(f"Speedup: {original_time / optimized_time:.2f}x")


if __name__ == "__main__":
    benchmark_mixing_layers()
