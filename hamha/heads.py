import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from hamha.topology import HexCoordinate


class CoordinateBiasFunction(nn.Module):
    """Generates a learnable, coordinate-dependent bias vector.

    This module takes a 2D hexagonal grid coordinate (q, r) and maps it to a
    bias vector of dimension `d_head` through a small neural network. This
    allows the attention mechanism to incorporate spatial information by adding
    a unique bias to each head's value projection based on its position in
    the grid.

    Attributes:
        d_model (int): The dimensionality of the main model.
        d_head (int): The dimensionality of the head and the output bias vector.
        coord_embed (nn.Sequential): The neural network that maps coordinates
            to bias vectors.
    """

    def __init__(self, d_model: int, d_head: int, hidden_dim: int = 64):
        """Initializes the CoordinateBiasFunction.

        Args:
            d_model (int): The dimensionality of the main model.
            d_head (int): The dimensionality of the output bias vector.
            hidden_dim (int, optional): The dimensionality of the hidden layer
                in the coordinate embedding network. Defaults to 64.
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.coord_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_head),
        )

    def forward(self, q: int, r: int) -> torch.Tensor:
        """Computes the bias vector for a given coordinate.

        Args:
            q (int): The 'q' coordinate in the hexagonal grid.
            r (int): The 'r' coordinate in the hexagonal grid.

        Returns:
            torch.Tensor: The computed bias vector of shape [d_head].
        """
        coord_tensor = torch.tensor([[float(q), float(r)]], dtype=torch.float32)
        bias_flat = self.coord_embed(coord_tensor)
        return bias_flat.view(self.d_head)


class HyperNetwork(nn.Module):
    """Generates attention head projection matrices dynamically.

    This module, known as a HyperNetwork, creates the query, key, and value
    projection matrices for an attention head based on its hexagonal grid
    coordinates (q, r) and a global context vector derived from the input.
    This allows each head to have specialized projection matrices that are
    sensitive to both its spatial position and the overall content of the
    input sequence.

    Attributes:
        d_model (int): The dimensionality of the main model.
        d_head (int): The dimensionality of the attention head.
        context_encoder (nn.Sequential): A network to process the global
            context vector.
        coord_embed (nn.Embedding): An embedding layer for the coordinates.
        weight_gen (nn.Sequential): The main network that generates the
            flattened projection matrix.
    """

    def __init__(self, d_model: int, d_head: int, context_dim: int = 128):
        """Initializes the HyperNetwork.

        Args:
            d_model (int): The dimensionality of the main model.
            d_head (int): The dimensionality of the attention head.
            context_dim (int, optional): The dimensionality of the encoded
                context vector. Defaults to 128.
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, context_dim), nn.LayerNorm(context_dim), nn.ReLU()
        )
        self.coord_embed = nn.Embedding(200, 32)
        input_dim = context_dim + 64
        self.weight_gen = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, d_model * d_head)
        )

    def forward(self, q: int, r: int, x_global: torch.Tensor) -> torch.Tensor:
        """Generates the projection matrix.

        Args:
            q (int): The 'q' coordinate in the hexagonal grid.
            r (int): The 'r' coordinate in the hexagonal grid.
            x_global (torch.Tensor): The global context vector, typically the
                mean of the input sequence, of shape [batch_size, d_model].

        Returns:
            torch.Tensor: The generated projection matrix of shape
                [batch_size, d_model, d_head].
        """
        context = self.context_encoder(x_global)
        q_embed = self.coord_embed(torch.tensor(q + 100).clamp(0, 199))
        r_embed = self.coord_embed(torch.tensor(r + 100).clamp(0, 199))
        coord_features = torch.cat([q_embed, r_embed]).unsqueeze(0).expand(context.shape[0], -1)
        combined = torch.cat([context, coord_features], dim=1)
        W_flat = self.weight_gen(combined)
        return W_flat.view(-1, self.d_model, self.d_head)


class AttentionHead(nn.Module):
    """A single attention head with coordinate-aware projection capabilities.

    This module performs the core scaled dot-product attention operation. It
    can generate its projection matrices (W_Q, W_K, W_V) in two ways:
    1.  **Static**: Uses a set of base projection matrices that are learned
        during training.
    2.  **Dynamic (HyperNetwork)**: Uses a `HyperNetwork` to generate the
        projection matrices based on the head's coordinates and the global
        input context.

    In the static mode, it can also add a coordinate-dependent spatial bias
    to the value tensor, provided by a `CoordinateBiasFunction`.

    Attributes:
        coord (HexCoordinate): The hexagonal grid coordinate of this head.
        d_model (int): The dimensionality of the input.
        d_head (int): The dimensionality of the head's output.
        use_hypernet (bool): Flag indicating whether to use the HyperNetwork.
        W_Q_base (nn.Parameter): The base query projection matrix.
        W_K_base (nn.Parameter): The base key projection matrix.
        W_V_base (nn.Parameter): The base value projection matrix.
        bias_function (CoordinateBiasFunction, optional): The module for
            generating spatial biases.
        hypernet (HyperNetwork, optional): The module for generating dynamic
            projection matrices.
        attention_weights (torch.Tensor): A detached copy of the attention
            weights from the last forward pass, for telemetry purposes.
        head_output (torch.Tensor): A detached copy of the head's output from
            the last forward pass, for telemetry purposes.
    """

    def __init__(
        self,
        coord: HexCoordinate,
        d_model: int,
        d_head: int,
        use_hypernet: bool = False,
        bias_function: Optional[CoordinateBiasFunction] = None,
        hypernet: Optional[HyperNetwork] = None,
    ):
        """Initializes the AttentionHead.

        Args:
            coord (HexCoordinate): The hexagonal coordinate of this head.
            d_model (int): The dimensionality of the input feature space.
            d_head (int): The dimensionality of this attention head.
            use_hypernet (bool, optional): If True, enables the use of the
                HyperNetwork for dynamic weight generation. Defaults to False.
            bias_function (CoordinateBiasFunction, optional): A pre-initialized
                module to generate coordinate-based biases. Used only if
                `use_hypernet` is False. Defaults to None.
            hypernet (HyperNetwork, optional): A pre-initialized HyperNetwork
                module. Used only if `use_hypernet` is True. Defaults to None.
        """
        super().__init__()
        self.coord = coord
        self.d_model = d_model
        self.d_head = d_head
        self.use_hypernet = use_hypernet

        # Base projection matrices
        self.W_Q_base = nn.Parameter(torch.randn(d_model, d_head) / math.sqrt(d_model))
        self.W_K_base = nn.Parameter(torch.randn(d_model, d_head) / math.sqrt(d_model))
        self.W_V_base = nn.Parameter(torch.randn(d_model, d_head) / math.sqrt(d_model))

        self.bias_function = bias_function
        self.hypernet = hypernet

        # Telemetry storage
        self.attention_weights = None
        self.head_output = None

    def get_projection_matrices(
        self, x_global: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves or generates the Q, K, and V projection matrices.

        If `use_hypernet` is True, it calls the HyperNetwork to generate the
        matrices. Otherwise, it returns the stored base matrices.

        Args:
            x_global (torch.Tensor, optional): The global context vector,
                required if `use_hypernet` is True. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                the W_Q, W_K, and W_V projection matrices, each of shape
                [batch_size, d_model, d_head].
        """
        if self.use_hypernet and self.hypernet is not None:
            W_Q = self.hypernet(self.coord.q, self.coord.r, x_global)
            W_K = self.hypernet(self.coord.q, self.coord.r, x_global)
            W_V = self.hypernet(self.coord.q, self.coord.r, x_global)
        else:
            W_Q = self.W_Q_base.unsqueeze(0)
            W_K = self.W_K_base.unsqueeze(0)
            W_V = self.W_V_base.unsqueeze(0)
        return W_Q, W_K, W_V

    def forward(
        self,
        x: torch.Tensor,
        x_global: Optional[torch.Tensor] = None,
        entropy_reg: float = 0.0,
    ) -> torch.Tensor:
        """Performs the forward pass for the attention head.

        Args:
            x (torch.Tensor): The input tensor of shape
                [batch_size, seq_len, d_model].
            x_global (torch.Tensor, optional): The global context vector,
                required if `use_hypernet` is True. Defaults to None.
            entropy_reg (float, optional): The coefficient for entropy
                regularization. A small amount of Gaussian noise is added to
                the attention scores to prevent fixation. Defaults to 0.0.

        Returns:
            torch.Tensor: The output tensor of the attention head, of shape
                [batch_size, seq_len, d_head].
        """
        W_Q, W_K, W_V = self.get_projection_matrices(x_global)
        Q = torch.einsum("bnd,bdh->bnh", x, W_Q)
        K = torch.einsum("bnd,bdh->bnh", x, W_K)
        V = torch.einsum("bnd,bdh->bnh", x, W_V)

        # Apply spatial bias to the value tensor
        if not self.use_hypernet and self.bias_function is not None:
            spatial_bias = self.bias_function(self.coord.q, self.coord.r)
            V = V + spatial_bias.view(1, 1, self.d_head)

        scores = torch.einsum("bnh,bmh->bnm", Q, K) / math.sqrt(self.d_head)

        # Entropy regularization (for diversity enforcement)
        if entropy_reg > 0:
            scores = scores + torch.randn_like(scores) * entropy_reg

        attn_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attn_weights.detach()

        output = torch.einsum("bnm,bmh->bnh", attn_weights, V)
        self.head_output = output.detach()
        return output
