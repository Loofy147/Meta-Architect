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
    """A collection of protocols for responding to system degradation.

    This class implements a set of pre-defined actions that the LMA can take
    to correct undesirable behaviors in the HAMHA model, such as attention
    fixation or rank collapse. It also includes the protocol for adapting the
    model's architecture using Meta-NAS.

    Attributes:
        model (HexagonalMultiHeadAttention): The HAMHA model to be acted upon.
        task_encoder (TaskEncoder, optional): The task encoder for Meta-NAS.
        meta_nas_controller (MetaNASController, optional): The controller for
            Meta-NAS.
        protocol_history (List[Dict]): A log of all executed protocols.
        meta_nas_enabled (bool): A flag indicating if Meta-NAS components are
            available.
    """

    def __init__(
        self,
        hamha_model: HexagonalMultiHeadAttention,
        task_encoder: Optional[TaskEncoder] = None,
        meta_nas_controller: Optional[MetaNASController] = None,
    ):
        """Initializes the EmergencyProtocols.

        Args:
            hamha_model (HexagonalMultiHeadAttention): The HAMHA model.
            task_encoder (TaskEncoder, optional): The task encoder for Meta-NAS.
                Defaults to None.
            meta_nas_controller (MetaNASController, optional): The controller
                for Meta-NAS. Defaults to None.
        """
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
        """Executes the Attention Diversity Protocol, Phase 1 (Soft Intervention).

        This protocol is a gentle nudge to encourage a potentially fixated
        attention head to explore a more diverse attention distribution. It
        increases the global entropy regularization and adjusts the GNN mixing
        coefficients to reduce the head's self-influence and increase its
        receptiveness to its neighbors.

        Args:
            target_head_idx (int): The index of the head to be targeted.
            entropy_reg_increment (float, optional): The amount to increase
                the entropy regularization by. Defaults to 0.01.
        """
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
        """Executes the Attention Diversity Protocol, Phase 2 (Hard Intervention).

        This protocol is a more forceful intervention for persistently fixated
        heads. It adds a small random perturbation to the head's projection
        matrices, knocking it out of a potential local minimum. This protocol is
        only applicable to non-spectral models.

        Args:
            target_head_idx (int): The index of the head to be targeted.

        Returns:
            str: A message describing the result of the operation.
        """
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
        """Resets the projection matrices for a specific attention head.

        This protocol is typically used to address rank collapse. It re-initializes
        the W_Q, W_K, and W_V matrices of a head using a specified strategy.
        This protocol is only applicable to non-spectral models.

        Args:
            target_head_idx (int): The index of the head to be reset.
            strategy (str, optional): The re-initialization strategy. Can be
                'orthogonal' or 'xavier'. Defaults to "orthogonal".

        Returns:
            str: A message describing the result of the operation.
        """
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
        """Generates a new HAMHA architecture for a new task using Meta-NAS.

        This protocol orchestrates the Meta-NAS pipeline:
        1. Encodes the task from sample data using the `TaskEncoder`.
        2. Generates a new architecture using the `MetaNASController`.
        3. Validates the proposed architecture.
        4. Creates a new `HexagonalMultiHeadAttention` model instance.
        5. Transfers weights from the old model to the new one to preserve
           learned knowledge.

        Args:
            sample_data (torch.Tensor): A sample batch of data from the new task.

        Returns:
            tuple[Optional[HexagonalMultiHeadAttention], Optional[Dict]]: A
                tuple containing the new HAMHA model instance and the new
                architecture dictionary, or (None, None) on failure.
        """
        if not self.task_encoder or not self.meta_nas_controller:
            return None, None

        # 1. Encode the task description into an embedding
        task_embedding = self.task_encoder(sample_data)

        # 2. Use Meta-NAS controller to generate new architecture
        new_arch, _ = self.meta_nas_controller.sample_architecture(task_embedding)

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
        """Transfers weights from an old model to a new one.

        This method intelligently copies weights for the parts of the
        architecture that overlap between the old and new models. This is
        crucial for preserving learned knowledge during architecture adaptation.
        It handles weight transfer for both spectral and standard models, and
        can even transfer weights between models with different grid sizes or
        head dimensions by slicing and copying the relevant parts of the
        weight tensors.

        Args:
            old_model (HexagonalMultiHeadAttention): The source model.
            new_model (HexagonalMultiHeadAttention): The destination model.
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

            # Transfer W_Q, W_K, W_V by slicing, similar to W_O
            d_head = new_model.d_head
            if old_model.d_head == d_head:
                with torch.no_grad():
                    for coord in overlapping_coords:
                        old_idx = old_model.coord_to_idx[coord]
                        new_idx = new_model.coord_to_idx[coord]

                        # Slice and copy weights for each projection matrix
                        old_slice_q = old_sa.W_Q.weight.data[old_idx * d_head : (old_idx + 1) * d_head, :]
                        new_sa.W_Q.weight.data[new_idx * d_head : (new_idx + 1) * d_head, :] = old_slice_q.clone()
                        old_slice_q_bias = old_sa.W_Q.bias.data[old_idx * d_head : (old_idx + 1) * d_head]
                        new_sa.W_Q.bias.data[new_idx * d_head : (new_idx + 1) * d_head] = old_slice_q_bias.clone()

                        old_slice_k = old_sa.W_K.weight.data[old_idx * d_head : (old_idx + 1) * d_head, :]
                        new_sa.W_K.weight.data[new_idx * d_head : (new_idx + 1) * d_head, :] = old_slice_k.clone()
                        old_slice_k_bias = old_sa.W_K.bias.data[old_idx * d_head : (old_idx + 1) * d_head]
                        new_sa.W_K.bias.data[new_idx * d_head : (new_idx + 1) * d_head] = old_slice_k_bias.clone()

                        old_slice_v = old_sa.W_V.weight.data[old_idx * d_head : (old_idx + 1) * d_head, :]
                        new_sa.W_V.weight.data[new_idx * d_head : (new_idx + 1) * d_head, :] = old_slice_v.clone()
                        old_slice_v_bias = old_sa.W_V.bias.data[old_idx * d_head : (old_idx + 1) * d_head]
                        new_sa.W_V.bias.data[new_idx * d_head : (new_idx + 1) * d_head] = old_slice_v_bias.clone()

                logger.info("Transferred spectral W_Q, W_K, W_V weights.")

            # Transfer spectral filter weights, handling different k
            if old_sa.spectral_filter and new_sa.spectral_filter:
                old_weights = old_sa.spectral_filter.weights.data
                new_weights = new_sa.spectral_filter.weights.data

                # Copy weights for the minimum number of overlapping eigenvectors
                k_to_transfer = min(len(old_weights), len(new_weights))
                new_weights[:k_to_transfer] = old_weights[:k_to_transfer]

                logger.info(f"Transferred {k_to_transfer} spectral filter weights.")

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
