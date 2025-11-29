# FILE: lma/protocols.py (FIXED VERSION)

import torch
import time
import logging
from typing import List, Dict, Optional
from hamha.core import HexagonalMultiHeadAttention
from lma.search_space import is_valid_architecture
from lma.task_encoder import TaskEncoder
from lma.meta_nas import MetaNASController

# Set up logging
logger = logging.getLogger(__name__)

class EmergencyProtocols:
    """Emergency response protocols for system degradation."""

    def __init__(
        self,
        hamha_model: HexagonalMultiHeadAttention,
        task_encoder: Optional[TaskEncoder] = None,
        meta_nas_controller: Optional[MetaNASController] = None,
    ):
        self.model = hamha_model
        self.task_encoder = task_encoder
        self.meta_nas_controller = meta_nas_controller
        self.protocol_history: List[Dict] = []

        # Check if Meta-NAS is available
        self.meta_nas_enabled = (task_encoder is not None and
                                 meta_nas_controller is not None)

    def trigger_aap_ad_phase1(
        self, target_head_idx: int, entropy_reg_increment: float = 0.01
    ):
        """Attention Diversity Protocol - Phase 1: Soft Intervention."""
        self.model.entropy_reg += entropy_reg_increment

        # Only apply GNN mixing adjustments if not using spectral attention
        if not self.model.use_spectral and hasattr(self.model, 'gnn_mixing'):
            # Decay self-mixing coefficient
            self.model.gnn_mixing.lambda_self.data[target_head_idx] *= 0.95

            # Boost neighbor influence
            adj = self.model.gnn_mixing.adjacency_dense
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
        # Only works with non-spectral HAMHA
        if self.model.use_spectral:
            return "AAP_AD_PHASE2 not applicable for spectral attention"

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
        # Only works with non-spectral HAMHA
        if self.model.use_spectral:
            return "Head reset not applicable for spectral attention"

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

    def adapt_architecture(self, sample_data: torch.Tensor):
        """
        Generates a new HAMHA architecture for a new task using Meta-NAS.

        Args:
            sample_data: Sample batch of data from the new task

        Returns:
            A tuple containing the new HAMHA model instance and the new architecture dict,
            or (None, None) on failure.
        """
        if not self.task_encoder or not self.meta_nas_controller:
            return None, None

        # 1. Encode the task description into an embedding
        task_embedding = self.task_encoder(sample_data)

        # 2. Use Meta-NAS controller to generate new architecture
        new_arch = self.meta_nas_controller(task_embedding)

        # 3. Validate the new architecture
        if not is_valid_architecture(new_arch):
            return None, None

        # 4. Create a new HAMHA model instance
        new_model = HexagonalMultiHeadAttention(
            d_model=self.model.d_model,
            **new_arch
        )

        # 5. Transfer weights from the old model to the new one
        self._transfer_weights(self.model, new_model)

        self.protocol_history.append(
            {
                "protocol": "ADAPT_ARCHITECTURE",
                "new_architecture": new_arch,
                "timestamp": time.time(),
            }
        )

        return new_model, new_arch

    def _transfer_weights(self, old_model: HexagonalMultiHeadAttention, new_model: HexagonalMultiHeadAttention):
        """
        Transfers weights from an old model to a new, potentially different, architecture.
        This preserves learned knowledge for overlapping parts of the architecture.
        """
        logger.info("Initiating weight transfer between HAMHA models...")

        # 0. Direct copy of simple attributes
        new_model.entropy_reg = old_model.entropy_reg

        # 1. Create a coordinate-to-head mapping for the old model for efficient lookup
        old_model_coord_map = {
            head.coord: head for head in getattr(old_model, 'heads', [])
        }

        # 2. Identify overlapping and new head coordinates
        old_coords = set(old_model.coord_to_idx.keys())
        new_coords = set(new_model.coord_to_idx.keys())
        overlapping_coords = old_coords.intersection(new_coords)

        if not overlapping_coords:
            logger.warning("No overlapping coordinates found. New model will have random initialization.")
            return

        logger.info(f"Found {len(overlapping_coords)} overlapping heads to transfer.")

        # 3. Transfer weights based on mode (spectral vs. non-spectral)
        if new_model.use_spectral and old_model.use_spectral:
            old_sa = old_model.spectral_attention
            new_sa = new_model.spectral_attention
            for coord in overlapping_coords:
                old_idx = old_model.coord_to_idx[coord]
                new_idx = new_model.coord_to_idx[coord]
                new_sa.filters[new_idx].load_state_dict(old_sa.filters[old_idx].state_dict())
            logger.info("Transferred spectral filters.")

        elif not new_model.use_spectral and not old_model.use_spectral:
            use_hypernet_old = getattr(old_model, 'hypernet', None) is not None
            use_hypernet_new = getattr(new_model, 'hypernet', None) is not None
            if old_model.d_head == new_model.d_head and use_hypernet_old == use_hypernet_new:
                for coord in overlapping_coords:
                    if coord in old_model_coord_map:
                        old_head = old_model_coord_map[coord]
                        new_head = new_model.heads[new_model.coord_to_idx[coord]]
                        new_head.load_state_dict(old_head.state_dict())
                logger.info("Transferred standard attention heads.")

                if hasattr(old_model, 'gnn_mixing') and hasattr(new_model, 'gnn_mixing'):
                    new_model.gnn_mixing.W_self.data.copy_(old_model.gnn_mixing.W_self.data)
                    new_model.gnn_mixing.W_neighbor.data.copy_(old_model.gnn_mixing.W_neighbor.data)
                    logger.info("Transferred GNN mixing weights.")
                    with torch.no_grad():
                        for coord in overlapping_coords:
                            old_idx = old_model.coord_to_idx[coord]
                            new_idx = new_model.coord_to_idx[coord]
                            new_model.gnn_mixing.lambda_self.data[new_idx] = old_model.gnn_mixing.lambda_self.data[old_idx]
                            for other_coord in overlapping_coords:
                                old_j = old_model.coord_to_idx[other_coord]
                                new_j = new_model.coord_to_idx[other_coord]
                                new_model.gnn_mixing.g_ij.data[new_idx, new_j] = old_model.gnn_mixing.g_ij.data[old_idx, old_j]
                    logger.info("Transferred GNN mixing parameters.")
            else:
                logger.warning(f"WARNING: d_head or hypernet configuration has changed. Cannot transfer attention head weights.")

        # 4. Transfer the final projection layer (W_O) by slicing
        with torch.no_grad():
            if old_model.d_head == new_model.d_head:
                d_head = new_model.d_head
                for coord in overlapping_coords:
                    old_idx = old_model.coord_to_idx[coord]
                    new_idx = new_model.coord_to_idx[coord]
                    old_slice = old_model.W_O[old_idx * d_head : (old_idx + 1) * d_head, :]
                    new_model.W_O.data[new_idx * d_head : (new_idx + 1) * d_head, :] = old_slice.clone()
                logger.info("Transferred final projection layer W_O.")
            else:
                logger.warning("d_head differs. Cannot transfer W_O. It will be randomly initialized.")

        logger.info("Weight transfer complete.")
