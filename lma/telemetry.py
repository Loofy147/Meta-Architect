# FILE: lma/telemetry.py (FIXED VERSION)

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

        # Detect if model uses spectral attention
        self.is_spectral = hamha_model.use_spectral

    def collect(self) -> TelemetrySnapshot:
        """Collect current telemetry snapshot."""
        snapshot = TelemetrySnapshot(step=self.current_step, timestamp=time.time())

        if self.is_spectral:
            # Spectral attention telemetry
            self._collect_spectral_telemetry(snapshot)
        else:
            # Standard HAMHA telemetry
            self._collect_standard_telemetry(snapshot)

        self.history.append(snapshot)
        self.current_step += 1
        return snapshot

    def _collect_standard_telemetry(self, snapshot: TelemetrySnapshot):
        """Collect telemetry for standard (non-spectral) HAMHA."""
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
        """Collect telemetry for spectral HAMHA."""
        # For spectral attention, we analyze the spectral layer weights
        spectral_layer = self.model.spectral_attention

        # Analyze Q, K, V projection matrices
        for name, layer in [("Q", spectral_layer.W_Q),
                           ("K", spectral_layer.W_K),
                           ("V", spectral_layer.W_V)]:
            if hasattr(layer, 'weight'):
                W = layer.weight
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                kappa = S.max() / (S.min() + 1e-8)
                key = f"Spectral_{name}"
                snapshot.condition_numbers[key] = kappa.item()
                snapshot.min_singular_values[key] = S.min().item()

                if kappa > 100:
                    snapshot.alerts.append(f"RANK_COLLAPSE: {key} κ={kappa:.2f}")

        # Gradient norms for spectral layer
        if spectral_layer.W_Q.weight.grad is not None:
            grad_norm = torch.norm(spectral_layer.W_Q.weight.grad).item()
            snapshot.gradient_norms["Spectral_Q"] = grad_norm

            if grad_norm < 1e-6:
                snapshot.alerts.append("VANISHING_GRADIENT: Spectral_Q")
            elif grad_norm > 1e3:
                snapshot.alerts.append("EXPLODING_GRADIENT: Spectral_Q")

        # For spectral attention, we can analyze filter responses
        for head_idx, filter in enumerate(spectral_layer.filters):
            # Get filter response
            filter_response = filter(spectral_layer.eigenvalues)
            avg_response = filter_response.mean().item()

            # Use filter response as a proxy for "attention entropy"
            # High variance = diverse filtering, low variance = fixation
            response_std = filter_response.std().item()

            coord_str = f"SpectralHead_{head_idx}"
            snapshot.attention_entropy[coord_str] = response_std

            # Compute derivative if history exists
            if len(self.history) > 0:
                prev_entropy = self.history[-1].attention_entropy.get(
                    coord_str, response_std
                )
                snapshot.entropy_derivatives[coord_str] = response_std - prev_entropy

            # Alerts based on filter diversity
            if response_std < 0.1:
                snapshot.alerts.append(f"FIXATION: {coord_str} σ={response_std:.3f}")
            elif (
                response_std < 0.3
                and snapshot.entropy_derivatives.get(coord_str, 0) < 0
            ):
                snapshot.alerts.append(f"DRIFT: {coord_str} σ={response_std:.3f}")
