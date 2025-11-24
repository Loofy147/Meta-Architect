import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class GNNMixingLayer(nn.Module):
    """Graph Neural Network-based mixing of attention head outputs."""

    def __init__(self, d_head: int, num_heads: int, adjacency_matrix: torch.Tensor):
        super().__init__()
        self.d_head = d_head
        self.num_heads = num_heads
        self.register_buffer('adjacency', adjacency_matrix)

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
                    neighbor_contrib += self.g_ij[i, j] * torch.matmul(H_j, self.W_neighbor)
            mixed = F.relu(self_contrib + neighbor_contrib)
            mixed_outputs.append(mixed)
        return mixed_outputs
