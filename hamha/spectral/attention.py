import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralFilter(nn.Module):
    """A learnable spectral filter."""

    def __init__(self, k_eigenvectors: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(k_eigenvectors))

    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """Apply the filter in the frequency domain.

        Args:
            x_freq: [B, N, K, D] - Batch in frequency domain

        Returns:
            [B, N, K, D] - Filtered batch
        """
        return x_freq * self.weights.view(1, 1, -1, 1)


class SpectralAttentionLayer(nn.Module):
    """Attention in the spectral domain of the graph of heads."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        k_eigenvectors: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.k = k_eigenvectors
        self.d_head = d_head

        self.W_Q = nn.Linear(d_model, num_heads * d_head)
        self.W_K = nn.Linear(d_model, num_heads * d_head)
        self.W_V = nn.Linear(d_model, num_heads * d_head)

        self.spectral_filter = None
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('eigenvectors', None)
        self.register_buffer('eigenvalues', None)

    def set_graph_spectrum(self, eigenvectors: torch.Tensor, eigenvalues: torch.Tensor):
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues

        # k is determined by the provided graph spectrum
        self.k = eigenvectors.shape[1]
        self.spectral_filter = SpectralFilter(self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        if self.spectral_filter is None:
            raise RuntimeError("Graph spectrum must be set before forward pass.")

        # Project and reshape Q, K, V
        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_head)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_head)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_head)

        # Project to spectral domain: einsum('hk,bnhd->bnkd', U, Q)
        Q_freq = torch.einsum('hk,bnhd->bnkd', self.eigenvectors, Q)
        K_freq = torch.einsum('hk,bnhd->bnkd', self.eigenvectors, K)
        V_freq = torch.einsum('hk,bnhd->bnkd', self.eigenvectors, V)

        # Apply spectral filter
        Q_filtered = self.spectral_filter(Q_freq)
        K_filtered = self.spectral_filter(K_freq)
        V_filtered = self.spectral_filter(V_freq)

        # Attention in spectral space: einsum('bnkd,bnld->bnkl', Q, K)
        scores = torch.einsum("bnkd,bnld->bnkl", Q_filtered, K_filtered) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output_freq = torch.einsum("bnkl,bnld->bnkd", attn, V_filtered)

        # Project back to spatial domain: einsum('hk,bnkd->bnhd', U.T, out_freq)
        output = torch.einsum("hk,bnkd->bnhd", self.eigenvectors, output_freq)

        # Merge heads
        return output.reshape(B, N, self.num_heads * self.d_head)
