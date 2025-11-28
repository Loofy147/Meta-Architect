from dataclasses import dataclass, field
from typing import Dict, List
import time
import torch
from hamha.core import HexagonalMultiHeadAttention


@dataclass
class TelemetrySnapshot:
    """Single timestep telemetry data."""

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
    """Real-time telemetry collection from HAMHA system."""

    def __init__(self, hamha_model: HexagonalMultiHeadAttention):
        self.model = hamha_model
        self.history: List[TelemetrySnapshot] = []
        self.current_step = 0
        self.is_spectral = hamha_model.use_spectral

    def collect(self) -> TelemetrySnapshot:
        """Collect current telemetry snapshot."""
        snapshot = TelemetrySnapshot(step=self.current_step, timestamp=time.time())

        if self.is_spectral:
            self._collect_spectral_telemetry(snapshot)
        else:
            self._collect_standard_telemetry(snapshot)

        self.history.append(snapshot)
        self.current_step += 1
        return snapshot

    def _collect_standard_telemetry(self, snapshot: TelemetrySnapshot):
        # Spectral analysis
        for i, head in enumerate(self.model.heads):
            coord = head.coord
            W_Q, W_K, W_V = head.get_projection_matrices()

            for name, W in [("Q", W_Q), ("K", W_K), ("V", W_V)]:
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
