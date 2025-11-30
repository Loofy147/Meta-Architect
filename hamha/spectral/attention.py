import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpectralFilter(nn.Module):
    """A learnable spectral filter for graph Fourier modes.

    This module learns a set of weights, one for each eigenvector of the graph
    Laplacian, to modulate the importance of different frequency components
    of the input signal on the graph.

    Attributes:
        weights (nn.Parameter): A learnable parameter vector of shape
            [k_eigenvectors].
    """

    def __init__(self, k_eigenvectors: int):
        """Initializes the SpectralFilter.

        Args:
            k_eigenvectors (int): The number of eigenvectors (frequency
                components) to be filtered.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.ones(k_eigenvectors))

    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """Applies the filter in the frequency domain.

        Args:
            x_freq (torch.Tensor): The input tensor in the frequency domain,
                with shape [batch_size, seq_len, k_eigenvectors, d_head].

        Returns:
            torch.Tensor: The filtered tensor, with the same shape as the input.
        """
        return x_freq * self.weights.view(1, 1, -1, 1)


class SpectralAttentionLayer(nn.Module):
    """Performs attention in the spectral domain of the head graph.

    This layer transforms the query, key, and value tensors into the graph
    Fourier domain using the eigenvectors of the graph Laplacian. It then
    applies a learnable spectral filter, performs scaled dot-product attention
    in the spectral domain, and transforms the result back to the spatial
    domain. This approach allows the model to capture global relationships
    between heads more efficiently than with spatial message passing.

    Attributes:
        d_model (int): The dimensionality of the input.
        num_heads (int): The number of attention heads.
        k (int): The number of eigenvectors used.
        d_head (int): The dimensionality of each head.
        W_Q (nn.Linear): The linear layer for the query projection.
        W_K (nn.Linear): The linear layer for the key projection.
        W_V (nn.Linear): The linear layer for the value projection.
        spectral_filter (SpectralFilter): The learnable filter for the
            frequency domain.
        dropout (nn.Dropout): Dropout layer for the attention weights.
        eigenvectors (torch.Tensor): A buffer storing the eigenvectors of the
            graph Laplacian.
        eigenvalues (torch.Tensor): A buffer storing the eigenvalues of the
            graph Laplacian.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        k_eigenvectors: int,
        dropout: float = 0.1
    ):
        """Initializes the SpectralAttentionLayer.

        Args:
            d_model (int): The dimensionality of the input feature space.
            num_heads (int): The total number of attention heads.
            d_head (int): The dimensionality of each attention head.
            k_eigenvectors (int): The number of eigenvectors to use for the
                spectral transformation.
            dropout (float, optional): The dropout rate for the attention
                weights. Defaults to 0.1.
        """
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
        """Sets the graph spectrum for the layer.

        This method must be called before the forward pass. It provides the
        eigenvectors and eigenvalues of the graph Laplacian, which are used for
        the spectral transformations. It also initializes the `SpectralFilter`
        with the correct number of eigenvectors.

        Args:
            eigenvectors (torch.Tensor): A tensor of shape [num_heads, k],
                where k is the number of eigenvectors.
            eigenvalues (torch.Tensor): A tensor of shape [k] containing the
                corresponding eigenvalues.
        """
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues

        # k is determined by the provided graph spectrum
        self.k = eigenvectors.shape[1]
        self.spectral_filter = SpectralFilter(self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass for the spectral attention.

        Args:
            x (torch.Tensor): The input tensor of shape
                [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: The output tensor of shape
                [batch_size, seq_len, num_heads * d_head].

        Raises:
            RuntimeError: If the graph spectrum has not been set by calling
                `set_graph_spectrum` before the forward pass.
        """
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
