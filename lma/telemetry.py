# FILE: lma/telemetry.py (FIXED VERSION)

from dataclasses import dataclass, field
from typing import Dict, List
import time
import torch
from hamha.core import HexagonalMultiHeadAttention


@dataclass
class TelemetrySnapshot:
    """A container for all telemetry data collected at a single timestep.

    This dataclass acts as a structured container for various metrics
    capturing the state of the HAMHA model at a specific point in time.

    Attributes:
        step (int): The current training step.
        timestamp (float): The wall-clock time when the snapshot was taken.
        condition_numbers (Dict[str, float]): The condition number of projection
            matrices for each head, indicating spectral stability.
        min_singular_values (Dict[str, float]): The smallest singular value of
            projection matrices, related to rank collapse.
        gradient_norms (Dict[str, float]): The norm of the gradients for each
            head's projection matrices.
        global_gradient_norm (float): The overall gradient norm of the model.
        attention_entropy (Dict[str, float]): The entropy of the attention
            distribution for each head, measuring specialization.
        entropy_derivatives (Dict[str, float]): The change in entropy from the
            previous step, indicating drift.
        t_proj (float): Time spent in the projection phase.
        t_attn (float): Time spent in the attention calculation phase.
        t_mix (float): Time spent in the GNN mixing phase.
        t_grad (float): Time spent in the gradient computation phase.
        t_total (float): Total time for the forward and backward pass.
        throughput_tps (float): Throughput in tokens per second.
        alerts (List[str]): A list of alert messages generated based on the
            collected metrics.
    """

    step: int
    timestamp: float

    # Spectral Analysis
    condition_numbers: Dict[str, float] = field(default_factory=dict)
    min_singular_values: Dict[str, float] = field(default_factory=dict)

    # Gradient Flow
    gradient_norms: Dict[str, float] = field(default_factory=dict)
    global_gradient_norm: float = 0.0

    # Attention Entropy
    attention_entropy: Dict[str, float] = field(default_factory=dict)
    entropy_derivatives: Dict[str, float] = field(default_factory=dict)

    # Computational Profile
    t_proj: float = 0.0
    t_attn: float = 0.0
    t_mix: float = 0.0
    t_grad: float = 0.0
    t_total: float = 0.0

    # System Performance
    throughput_tps: float = 0.0

    # Alerts
    alerts: List[str] = field(default_factory=list)


class TelemetryCollector:
    """Collects real-time telemetry data from a HAMHA model.

    This class is responsible for interfacing with a `HexagonalMultiHeadAttention`
    model to extract a wide range of metrics at each training step. It can
    handle both standard (non-spectral) and spectral attention models,
    collecting the appropriate metrics for each.

    The collected data is stored in a history of `TelemetrySnapshot` objects.

    Attributes:
        model (HexagonalMultiHeadAttention): The model to be monitored.
        history (List[TelemetrySnapshot]): A chronological list of all collected
            snapshots.
        current_step (int): The current step count.
        is_spectral (bool): A flag indicating whether the model is in spectral
            mode.
    """

    def __init__(self, hamha_model: HexagonalMultiHeadAttention):
        """Initializes the TelemetryCollector.

        Args:
            hamha_model (HexagonalMultiHeadAttention): The HAMHA model to monitor.
        """
        self.model = hamha_model
        self.history: List[TelemetrySnapshot] = []
        self.current_step = 0
        self.is_spectral = hamha_model.use_spectral

        # Detect if model uses spectral attention
        self.is_spectral = hamha_model.use_spectral

    def collect(self) -> TelemetrySnapshot:
        """Collects and returns a new telemetry snapshot.

        This is the main method of the collector. It creates a new
        `TelemetrySnapshot`, populates it with data by calling the appropriate
        helper methods based on the model's mode (spectral or standard),
        appends the snapshot to its history, and increments the step counter.

        Returns:
            TelemetrySnapshot: The newly collected telemetry data.
        """
        snapshot = TelemetrySnapshot(step=self.current_step, timestamp=time.time())

        if self.is_spectral:
            self._collect_spectral_telemetry(snapshot)
        else:
            self._collect_standard_telemetry(snapshot)

        self.history.append(snapshot)
        self.current_step += 1
        return snapshot

    def _collect_standard_telemetry(self, snapshot: TelemetrySnapshot):
        """Collects telemetry for a standard (non-spectral) HAMHA model.

        This method iterates over each attention head in the model to collect
        head-specific metrics like condition numbers, gradient norms, and
        attention entropy.

        Args:
            snapshot (TelemetrySnapshot): The snapshot object to be populated
                with data.
        """
        # Spectral analysis
        for i, head in enumerate(self.model.heads):
            coord = head.coord
            W_Q, W_K, W_V = head.get_projection_matrices()

            for name, W in [("Q", W_Q), ("K", W_K), ("V", W_V)]:
                # Handle both batched and non-batched projection matrices
                if W.dim() == 3:
                    W = W[0]  # Take first batch element for analysis

                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                kappa = S.max() / (S.min() + 1e-8)
                key = f"H{coord}_{name}"
                snapshot.condition_numbers[key] = kappa.item()
                snapshot.min_singular_values[key] = S.min().item()

                if kappa > 100:
                    snapshot.alerts.append(f"RANK_COLLAPSE: {key} κ={kappa:.2f}")

        # Gradient norms
        for i, head in enumerate(self.model.heads):
            coord = head.coord
            if head.W_Q_base.grad is not None:
                grad_norm = torch.norm(head.W_Q_base.grad).item()
                snapshot.gradient_norms[str(coord)] = grad_norm

                if grad_norm < 1e-6:
                    snapshot.alerts.append(f"VANISHING_GRADIENT: {coord}")
                elif grad_norm > 1e3:
                    snapshot.alerts.append(f"EXPLODING_GRADIENT: {coord}")

        # Attention entropy
        for i, head in enumerate(self.model.heads):
            coord = head.coord
            if head.attention_weights is not None:
                attn = head.attention_weights
                entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean().item()
                snapshot.attention_entropy[str(coord)] = entropy

                # Compute derivative if history exists
                if len(self.history) > 0:
                    prev_entropy = self.history[-1].attention_entropy.get(
                        str(coord), entropy
                    )
                    snapshot.entropy_derivatives[str(coord)] = entropy - prev_entropy

                if entropy < 0.3:
                    snapshot.alerts.append(f"FIXATION: {coord} H={entropy:.3f}")
                elif (
                    entropy < 0.9
                    and snapshot.entropy_derivatives.get(str(coord), 0) < 0
                ):
                    snapshot.alerts.append(f"DRIFT: {coord} H={entropy:.3f}")

    def _collect_spectral_telemetry(self, snapshot: TelemetrySnapshot):
        """Collects telemetry for a spectral HAMHA model.

        This method collects metrics from the global projection matrices of the
        `SpectralAttentionLayer`, as head-specific matrices do not exist in
        this mode.

        Args:
            snapshot (TelemetrySnapshot): The snapshot object to be populated
                with data.
        """
        spectral_layer = self.model.spectral_attention
        for name, layer in [
            ("Q", spectral_layer.W_Q),
            ("K", spectral_layer.W_K),
            ("V", spectral_layer.W_V),
        ]:
            if hasattr(layer, "weight"):
                W = layer.weight
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                kappa = S.max() / (S.min() + 1e-8)
                key = f"Spectral_{name}"
                snapshot.condition_numbers[key] = kappa.item()
                snapshot.min_singular_values[key] = S.min().item()

                if kappa > 100:
                    snapshot.alerts.append(f"RANK_COLLAPSE: {key} κ={kappa:.2f}")
