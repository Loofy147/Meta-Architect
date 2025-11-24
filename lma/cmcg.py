import networkx as nx
from typing import List


class CrossModalCausalGraph:
    """Dynamic causal graph linking telemetry metrics to system behaviors."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_base_structure()

    def _initialize_base_structure(self):
        """Initialize known causal relationships."""
        nodes = [
            "rank_collapse",
            "vanishing_gradient",
            "exploding_gradient",
            "fixation",
            "drift",
            "high_t_mix",
            "low_throughput",
            "entropy_decline",
            "kappa_increase",
            "gradient_norm_low",
        ]
        self.graph.add_nodes_from(nodes)

        # Known causal edges
        edges = [
            ("kappa_increase", "rank_collapse", 0.95),
            ("rank_collapse", "vanishing_gradient", 0.85),
            ("entropy_decline", "drift", 0.90),
            ("drift", "fixation", 0.80),
            ("high_t_mix", "low_throughput", 0.92),
        ]

        for src, dst, confidence in edges:
            self.graph.add_edge(src, dst, confidence=confidence, observations=1)

    def update_edge(self, source: str, target: str, observed: bool):
        """Update causal edge based on observation."""
        if not self.graph.has_node(source):
            self.graph.add_node(source)
        if not self.graph.has_node(target):
            self.graph.add_node(target)

        if self.graph.has_edge(source, target):
            data = self.graph[source][target]
            data["observations"] += 1
            if observed:
                data["confirmations"] = data.get("confirmations", 0) + 1
            data["confidence"] = data.get("confirmations", 0) / data["observations"]
        else:
            self.graph.add_edge(
                source,
                target,
                confidence=1.0 if observed else 0.0,
                observations=1,
                confirmations=1 if observed else 0,
            )

    def get_likely_causes(self, effect: str, threshold: float = 0.7) -> List[str]:
        """Find likely causes for an observed effect."""
        causes = []
        for pred in self.graph.predecessors(effect):
            if self.graph[pred][effect].get("confidence", 0) >= threshold:
                causes.append(pred)
        return causes

    def get_likely_effects(self, cause: str, threshold: float = 0.7) -> List[str]:
        """Predict likely effects from a cause."""
        effects = []
        for succ in self.graph.successors(cause):
            if self.graph[cause][succ].get("confidence", 0) >= threshold:
                effects.append(succ)
        return effects
