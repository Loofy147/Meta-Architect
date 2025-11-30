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
import torch


class LeadMetaArchitect:
    """The central intelligence and control unit for a HAMHA model.

    The Lead Meta-Architect (LMA) provides governance over a `HexagonalMultiHeadAttention`
    model. It collects real-time telemetry, performs causal reasoning, generates
    hypotheses about the model's state, predicts future dynamics, and executes
    interventions to maintain stability and performance. It can also perform
    meta-neural architecture search (Meta-NAS) to adapt the HAMHA model to new
    tasks.

    Attributes:
        model (HexagonalMultiHeadAttention): The HAMHA model being governed.
        enable_meta_nas (bool): Flag to enable Meta-NAS capabilities.
        telemetry (TelemetryCollector): The subsystem for collecting data from the model.
        cmcg (CrossModalCausalGraph): The causal graph for reasoning about the model.
        hge (HypothesisGenerationEngine): The subsystem for generating hypotheses.
        adp (ArchitecturalDynamicsPredictor): The subsystem for predicting future states.
        evolutionary (EvolutionaryModules): Manages long-term evolutionary adaptations.
        task_encoder (TaskEncoder, optional): Encodes task data for Meta-NAS.
        meta_nas_controller (MetaNASController, optional): Proposes new architectures.
        protocols (EmergencyProtocols): Handles automated interventions.
        monitoring_sectors (Dict[str, Dict]): Tracks heads under active monitoring.
        alert_history (List[Dict]): A history of all triggered alerts.
    """

    def __init__(self, hamha_model: HexagonalMultiHeadAttention, enable_meta_nas: bool = False):
        """Initializes the LeadMetaArchitect.

        Args:
            hamha_model (HexagonalMultiHeadAttention): The HAMHA model to be
                governed.
            enable_meta_nas (bool, optional): If True, enables the Meta-NAS
                subsystems for architecture adaptation. Defaults to False.
        """
        self.model = hamha_model
        self.enable_meta_nas = enable_meta_nas

        # Core subsystems
        self.telemetry = TelemetryCollector(hamha_model)
        self.cmcg = CrossModalCausalGraph()
        self.hge = HypothesisGenerationEngine(self.cmcg)
        self.adp = ArchitecturalDynamicsPredictor()
        self.evolutionary = EvolutionaryModules()

        if self.enable_meta_nas:
            self.task_encoder = TaskEncoder(d_in=self.model.d_model, embedding_dim=64)
            self.meta_nas_controller = MetaNASController(task_embedding_dim=64)
            self.protocols = EmergencyProtocols(
                hamha_model, self.task_encoder, self.meta_nas_controller
            )
        else:
            self.task_encoder = None
            self.meta_nas_controller = None
            self.protocols = EmergencyProtocols(hamha_model)

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
        """The main processing loop for the LMA, intended to be called after each training step.

        This method orchestrates the core functions of the LMA:
        1.  Collects a `TelemetrySnapshot` from the model.
        2.  Generates causal hypotheses if any alerts are present in the snapshot.
        3.  Generates predictions about future model dynamics.
        4.  Evaluates and triggers any necessary emergency interventions.
        5.  Updates the internal causal graph based on the new observations.

        Returns:
            Dict: A dictionary containing the results of the processing step,
                including the telemetry snapshot, any generated hypotheses,
                predictions, a list of interventions performed, and a status report.
        """
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
        """Evaluates telemetry data to determine if any emergency protocols should be triggered.

        This method checks for conditions such as low attention entropy or rapid
        entropy drift, which are indicative of potential model instability. If
        a trigger condition is met, it initiates the appropriate intervention
        protocol.

        Args:
            snapshot (TelemetrySnapshot): The latest telemetry data from the model.

        Returns:
            List[str]: A list of strings describing the interventions that were
                triggered.
        """
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
        """Updates the Cross-Modal Causal Graph (CMCG) based on new observations.

        This method strengthens the causal links in the graph based on the
        alerts present in the latest telemetry snapshot. For example, if a
        rank collapse alert is present, it strengthens the edge from
        "kappa_increase" to "rank_collapse".

        Args:
            snapshot (TelemetrySnapshot): The latest telemetry data from the model.
        """
        for alert in snapshot.alerts:
            if "RANK_COLLAPSE" in alert:
                self.cmcg.update_edge("kappa_increase", "rank_collapse", True)
            elif "DRIFT" in alert:
                self.cmcg.update_edge("entropy_decline", "drift", True)
            elif "FIXATION" in alert:
                self.cmcg.update_edge("drift", "fixation", True)

    def _generate_status_report(self, snapshot: TelemetrySnapshot) -> Dict:
        """Generates a comprehensive status report from a telemetry snapshot.

        This method aggregates key metrics from the snapshot to provide a high-level
        overview of the model's current state, including its overall health,
        average entropy, maximum condition number, and other vital signs.

        Args:
            snapshot (TelemetrySnapshot): The telemetry data to be summarized.

        Returns:
            Dict: A dictionary containing the summarized status report.
        """
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
        """LMA Command: Activates an evolutionary module.

        Args:
            module_name (str): The name of the evolutionary module to activate.
            parameters (Dict, optional): Configuration parameters for the module.
                Defaults to None.

        Returns:
            str: A message indicating the result of the activation command.
        """
        return self.evolutionary.activate_module(module_name, parameters)

    def command_adjust_entropy_regularization(self, delta: float):
        """LMA Command: Adjusts the global entropy regularization coefficient.

        Args:
            delta (float): The amount to add to the current entropy regularization
                coefficient.

        Returns:
            str: A message confirming the new regularization value.
        """
        self.model.entropy_reg += delta
        return f"Entropy regularization: {self.model.entropy_reg}"

    def command_reset_head(self, coord: HexCoordinate, strategy: str = "orthogonal"):
        """LMA Command: Resets the projection matrices of a specific attention head.

        Args:
            coord (HexCoordinate): The coordinate of the head to reset.
            strategy (str, optional): The re-initialization strategy to use
                ('orthogonal', 'xavier', etc.). Defaults to "orthogonal".

        Returns:
            str: A message confirming that the head was reset.
        """
        head_idx = self.model.coord_to_idx[coord]
        return self.protocols.reset_head_projections(head_idx, strategy)

    def command_adapt_architecture(self, sample_data: torch.Tensor):
        """LMA Command: Triggers a Meta-NAS architecture adaptation cycle.

        This command uses the provided sample data to encode a task embedding,
        which is then used by the Meta-NAS controller to propose a new,
        potentially more suitable, architecture for the HAMHA model. The LMA
        then replaces the current model with the new one.

        Args:
            sample_data (torch.Tensor): A small batch of sample data
                representative of the target task.

        Returns:
            str: A message indicating the result of the adaptation, including
                the new architecture specification if successful.
        """
        if not self.enable_meta_nas:
            return "Meta-NAS is not enabled."

        new_model, new_arch = self.protocols.adapt_architecture(sample_data)

        if new_model is None:
            return "ADAPT_ARCHITECTURE failed. See logs for details."

        self.model = new_model
        self.telemetry = TelemetryCollector(self.model)
        self.protocols.model = self.model

        return f"ADAPT_ARCHITECTURE complete. New architecture: {new_arch}"

    def generate_report(self) -> str:
        """Generates a detailed, human-readable operational report.

        The report summarizes the latest telemetry, system health, active
        monitoring sectors, recent interventions, and active hypotheses,
        providing a comprehensive overview of the LMA's status and actions.

        Returns:
            str: A formatted string containing the full report.
        """
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
