import torch
import time
from typing import List, Dict
from hamha.core import HexagonalMultiHeadAttention


class EmergencyProtocols:
    """Emergency response protocols for system degradation."""

    def __init__(self, hamha_model: HexagonalMultiHeadAttention):
        self.model = hamha_model
        self.protocol_history: List[Dict] = []

    def trigger_aap_ad_phase1(
        self, target_head_idx: int, entropy_reg_increment: float = 0.01
    ):
        """Attention Diversity Protocol - Phase 1: Soft Intervention."""
        self.model.entropy_reg += entropy_reg_increment

        # Decay self-mixing coefficient
        self.model.gnn_mixing.lambda_self.data[target_head_idx] *= 0.95

        # Boost neighbor influence
        adj = self.model.gnn_mixing.adjacency
        for j in range(self.model.num_heads):
            if adj[target_head_idx, j] > 0:
                self.model.gnn_mixing.g_ij.data[target_head_idx, j] *= 1.05

        self.protocol_history.append(
            {
                "protocol": "AAP_AD_PHASE1",
                "target_head": target_head_idx,
                "entropy_reg": self.model.entropy_reg,
                "timestamp": time.time(),
            }
        )

        return f"AAP_AD_PHASE1 executed on head {target_head_idx}"

    def trigger_aap_ad_phase2(self, target_head_idx: int):
        """Attention Diversity Protocol - Phase 2: Hard Intervention."""
        head = self.model.heads[target_head_idx]

        # Add random perturbation to projection matrices
        with torch.no_grad():
            head.W_Q_base.data += torch.randn_like(head.W_Q_base) * 1e-3
            head.W_K_base.data += torch.randn_like(head.W_K_base) * 1e-3
            head.W_V_base.data += torch.randn_like(head.W_V_base) * 1e-3

        self.protocol_history.append(
            {
                "protocol": "AAP_AD_PHASE2",
                "target_head": target_head_idx,
                "perturbation": 1e-3,
                "timestamp": time.time(),
            }
        )

        return f"AAP_AD_PHASE2 executed on head {target_head_idx}"

    def reset_head_projections(
        self, target_head_idx: int, strategy: str = "orthogonal"
    ):
        """Reset projection matrices for a head."""
        head = self.model.heads[target_head_idx]

        with torch.no_grad():
            if strategy == "orthogonal":
                head.W_Q_base.data = torch.nn.init.orthogonal_(head.W_Q_base.data)
                head.W_K_base.data = torch.nn.init.orthogonal_(head.W_K_base.data)
                head.W_V_base.data = torch.nn.init.orthogonal_(head.W_V_base.data)
            elif strategy == "xavier":
                torch.nn.init.xavier_uniform_(head.W_Q_base)
                torch.nn.init.xavier_uniform_(head.W_K_base)
                torch.nn.init.xavier_uniform_(head.W_V_base)

        self.protocol_history.append(
            {
                "protocol": "RESET_PROJECTIONS",
                "target_head": target_head_idx,
                "strategy": strategy,
                "timestamp": time.time(),
            }
        )

        return f"Projections reset for head {target_head_idx} using {strategy}"
