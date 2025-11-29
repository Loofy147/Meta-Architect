import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from hamha.topology import HexCoordinate


class CoordinateBiasFunction(nn.Module):
    """Learnable coordinate-dependent bias for projection matrices."""

    def __init__(self, d_model: int, d_head: int, hidden_dim: int = 64):
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
        coord_tensor = torch.tensor([[float(q), float(r)]], dtype=torch.float32)
        bias_flat = self.coord_embed(coord_tensor)
        return bias_flat.view(self.d_head)


class HyperNetwork(nn.Module):
    """Generate projection matrices dynamically based on coordinates and context."""

    def __init__(self, d_model: int, d_head: int, context_dim: int = 128):
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
        context = self.context_encoder(x_global)
        q_embed = self.coord_embed(torch.tensor(q + 100).clamp(0, 199))
        r_embed = self.coord_embed(torch.tensor(r + 100).clamp(0, 199))
        coord_features = torch.cat([q_embed, r_embed]).unsqueeze(0).expand(context.shape[0], -1)
        combined = torch.cat([context, coord_features], dim=1)
        W_flat = self.weight_gen(combined)
        return W_flat.view(-1, self.d_model, self.d_head)


class AttentionHead(nn.Module):
    """Single attention head with coordinate-aware projections."""

    def __init__(
        self,
        coord: HexCoordinate,
        d_model: int,
        d_head: int,
        use_hypernet: bool = False,
        bias_function: Optional[CoordinateBiasFunction] = None,
        hypernet: Optional[HyperNetwork] = None,
    ):
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
