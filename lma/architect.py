# FILE: lma/architect.py (FIXED VERSION)

from hamha.core import HexagonalMultiHeadAttention
from lma.telemetry import TelemetryCollector, TelemetrySnapshot
from lma.cmcg import CrossModalCausalGraph
from lma.hge import HypothesisGenerationEngine
from lma.adp import ArchitecturalDynamicsPredictor
from lma.protocols import EmergencyProtocols
from lma.evolutionary import EvolutionaryModules
from lma.task_encoder import TaskEncoder
from lma.meta_nas import MetaNASController
from hamha.topology import HexCoordinate
from typing import Dict, List, Optional
import numpy as np


class LeadMetaArchitect:
    """
    Lead Meta-Architect (LMA) - Central Intelligence and Control Unit

    Implements perceptive omniscience and prescriptive agency over HAMHA.
    """

    def __init__(
        self,
        hamha_model: HexagonalMultiHeadAttention,
        enable_meta_nas: bool = True
    ):
        self.model = hamha_model
        self.enable_meta_nas = enable_meta_nas

        # Core subsystems
        self.telemetry = TelemetryCollector(hamha_model)
        self.cmcg = CrossModalCausalGraph()
        self.hge = HypothesisGenerationEngine(self.cmcg)
        self.adp = ArchitecturalDynamicsPredictor()
        self.evolutionary = EvolutionaryModules()

        # Initialize Meta-NAS components if enabled
        if enable_meta_nas:
            self.task_encoder = TaskEncoder(d_in=hamha_model.d_model, embedding_dim=64)
            self.meta_nas_controller = MetaNASController(task_embedding_dim=64)
            self.protocols = EmergencyProtocols(
                hamha_model,
                self.task_encoder,
                self.meta_nas_controller
            )
        else:
            self.task_encoder = None
            self.meta_nas_controller = None
            # Use simplified protocols without Meta-NAS
            self.protocols = EmergencyProtocols(hamha_model, None, None)

        # Monitoring state
        self.monitoring_sectors: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []

        print("═" * 70)
        print("LEAD META-ARCHITECT INITIALIZED")
        print("═" * 70)
        print(f"Grid Size: {hamha_model.num_heads} heads")
        print(f"Topology: Hexagonal (radius {max(abs(c.q) for c in hamha_model.grid_coords)})")
        print(f"Meta-NAS: {'ENABLED' if enable_meta_nas else 'DISABLED'}")
        print(f"Telemetry Streams: ACTIVE")
        print(f"CMCG Nodes: {self.cmcg.graph.number_of_nodes()}")
        print(f"CMCG Edges: {self.cmcg.graph.number_of_edges()}")
        print("═" * 70)

    def process_step(self) -> Dict:
        """Main LMA processing loop - call after each training step."""
        # 1. Collect telemetry
        snapshot = self.telemetry.collect()

        # 2. Generate hypotheses if alerts present
        hypotheses = []
        if snapshot.alerts:
            hypotheses = self.hge.generate_from_snapshot(snapshot)

        # 3. Make predictions
        predictions = {}
        for coord_str in snapshot.attention_entropy.keys():
            pred = self.adp.predict_entropy_trajectory(
                self.telemetry.history, coord_str, steps_ahead=20
            )
            if "error" not in pred:
                predictions[coord_str] = pred

        # 4. Check for intervention triggers
        interventions = self._evaluate_intervention_triggers(snapshot)

        # 5. Update CMCG based on observations
        self._update_cmcg(snapshot)

        return {
            "snapshot": snapshot,
            "hypotheses": hypotheses,
            "predictions": predictions,
            "interventions": interventions,
            "status": self._generate_status_report(snapshot),
        }

    def _evaluate_intervention_triggers(self, snapshot: TelemetrySnapshot) -> List[str]:
        """Evaluate if any emergency protocols should be triggered."""
        interventions = []

        for coord_str, entropy in snapshot.attention_entropy.items():
            deriv = snapshot.entropy_derivatives.get(coord_str, 0)

            # Trigger AAP_AD Phase 1 if entropy < 0.85 or drift detected
            if entropy < 0.85 or (entropy < 0.9 and deriv < -0.008):
                q, r = map(int, coord_str[2:-1].split(","))
                coord = HexCoordinate(q, r)
                head_idx = self.model.coord_to_idx[coord]

                # Check if already in monitoring
                if coord_str not in self.monitoring_sectors:
                    self.monitoring_sectors[coord_str] = {
                        "trigger_step": snapshot.step,
                        "initial_entropy": entropy,
                    }

                    result = self.protocols.trigger_aap_ad_phase1(head_idx)
                    interventions.append(result)

                    self.alert_history.append(
                        {
                            "step": snapshot.step,
                            "type": "AAP_AD_PHASE1",
                            "target": coord_str,
                            "reason": f"H={entropy:.3f}, ΔH={deriv:.4f}",
                        }
                    )

        return interventions

    def _update_cmcg(self, snapshot: TelemetrySnapshot):
        """Update causal graph based on observations."""
        for alert in snapshot.alerts:
            if "RANK_COLLAPSE" in alert:
                self.cmcg.update_edge("kappa_increase", "rank_collapse", True)
            elif "DRIFT" in alert:
                self.cmcg.update_edge("entropy_decline", "drift", True)
            elif "FIXATION" in alert:
                self.cmcg.update_edge("drift", "fixation", True)

    def _generate_status_report(self, snapshot: TelemetrySnapshot) -> Dict:
        """Generate comprehensive status report."""
        # Compute grid-wide statistics
        avg_entropy = (
            np.mean(list(snapshot.attention_entropy.values()))
            if snapshot.attention_entropy
            else 0
        )
        max_kappa = (
            max(snapshot.condition_numbers.values())
            if snapshot.condition_numbers
            else 0
        )
        avg_grad = (
            np.mean(list(snapshot.gradient_norms.values()))
            if snapshot.gradient_norms
            else 0
        )

        return {
            "step": snapshot.step,
            "health": (
                "OPTIMAL"
                if not snapshot.alerts
                else "DEGRADED" if len(snapshot.alerts) < 3 else "CRITICAL"
            ),
            "avg_entropy": avg_entropy,
            "max_kappa": max_kappa,
            "avg_gradient_norm": avg_grad,
            "t_mix": snapshot.t_mix,
            "throughput": snapshot.throughput_tps,
            "active_alerts": len(snapshot.alerts),
            "monitoring_sectors": len(self.monitoring_sectors),
            "active_modules": self.evolutionary.get_active_modules(),
            "meta_nas_enabled": self.enable_meta_nas,
        }

    def command_activate_module(self, module_name: str, parameters: Dict = None):
        """LMA Command: Activate evolutionary module."""
        return self.evolutionary.activate_module(module_name, parameters)

    def command_adjust_entropy_regularization(self, delta: float):
        """LMA Command: Adjust global entropy regularization."""
        self.model.entropy_reg += delta
        return f"Entropy regularization: {self.model.entropy_reg}"

    def command_reset_head(self, coord: HexCoordinate, strategy: str = "orthogonal"):
        """LMA Command: Reset specific head projections."""
        head_idx = self.model.coord_to_idx[coord]
        return self.protocols.reset_head_projections(head_idx, strategy)

    def command_adapt_architecture(self, sample_data):
        """LMA Command: Trigger Meta-NAS architecture adaptation."""
        if not self.enable_meta_nas:
            return "Meta-NAS is not enabled. Initialize LMA with enable_meta_nas=True"

        new_model, new_arch = self.protocols.adapt_architecture(sample_data)

        if new_model is None:
            return "ADAPT_ARCHITECTURE failed. See logs for details."

        # Update the main model reference
        self.model = new_model

        # Re-initialize telemetry with the new model to ensure consistency
        self.telemetry = TelemetryCollector(self.model)

        # Also update the protocols' internal model reference
        self.protocols.model = new_model

        return f"ADAPT_ARCHITECTURE complete. New architecture: {new_arch}"

    def generate_report(self) -> str:
        """Generate detailed LMA report."""
        if not self.telemetry.history:
            return "No telemetry data available"

        latest = self.telemetry.history[-1]
        status = self._generate_status_report(latest)

        report = f"""
╔═══════════════════════════════════════════════════════════════════╗
║           LEAD META-ARCHITECT OPERATIONAL REPORT                  ║
╚═══════════════════════════════════════════════════════════════════╝

STEP: {status['step']}
SYSTEM HEALTH: {status['health']}
META-NAS: {'ENABLED' if status['meta_nas_enabled'] else 'DISABLED'}

TELEMETRY SUMMARY:
  • Average Entropy: {status['avg_entropy']:.3f}
  • Max Condition Number: {status['max_kappa']:.2f}
  • Average Gradient Norm: {status['avg_gradient_norm']:.4f}
  • T_mix: {status['t_mix']:.1f}µs
  • Throughput: {status['throughput']:.0f} TPS

ACTIVE MONITORING:
  • Alert Count: {status['active_alerts']}
  • Monitoring Sectors: {status['monitoring_sectors']}
  • Active Modules: {', '.join(status['active_modules']) or 'None'}

RECENT INTERVENTIONS:
"""
        for intervention in self.protocols.protocol_history[-5:]:
            report += f"  • {intervention['protocol']} on head {intervention.get('target_head', 'N/A')}\n"

        report += f"\nACTIVE HYPOTHESES: {len(self.hge.active_hypotheses)}\n"
        for hyp in self.hge.active_hypotheses[-3:]:
            report += (
                f"  • {hyp.id}: {hyp.description} (confidence: {hyp.confidence:.2f})\n"
            )

        report += "\n" + "═" * 70
        return report
