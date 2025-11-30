import networkx as nx
from typing import List


class CrossModalCausalGraph:
    """A dynamic causal graph representing relationships in the HAMHA system.

    This class uses a directed graph (DiGraph) from the `networkx` library to
    model the causal relationships between different telemetry metrics and
    observed system behaviors (e.g., "rank_collapse", "fixation"). The edges
    in the graph are weighted by a confidence score that is updated based on
    real-time observations.

    This allows the LMA to perform causal reasoning, such as identifying the
    likely causes of an observed effect or predicting the likely effects of a
    potential cause.

    Attributes:
        graph (nx.DiGraph): The directed graph representing the causal model.
            Edges have 'confidence', 'observations', and 'confirmations'
            attributes.
    """

    def __init__(self):
        """Initializes the CrossModalCausalGraph."""
        self.graph = nx.DiGraph()
        self._initialize_base_structure()

    def _initialize_base_structure(self):
        """Initializes the graph with a set of known causal relationships.

        This method populates the graph with a foundational set of nodes and
        edges that represent well-understood causal links within the attention
        mechanism. These serve as a starting point for the dynamic updates.
        """
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
        """Updates the confidence of a causal edge based on a new observation.

        If the `observed` flag is True, it increases the confirmation count for
        the edge, strengthening the belief in the causal link. The confidence
        is recalculated as the ratio of confirmations to total observations.

        Args:
            source (str): The name of the source (cause) node.
            target (str): The name of the target (effect) node.
            observed (bool): A flag indicating whether the causal link was
                observed in the latest telemetry.
        """
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
        """Finds the likely causes for an observed effect.

        This method queries the graph to find all predecessor nodes of the given
        `effect` node whose edge confidence exceeds the specified threshold.

        Args:
            effect (str): The name of the observed effect node.
            threshold (float, optional): The minimum confidence level for a cause
                to be considered likely. Defaults to 0.7.

        Returns:
            List[str]: A list of node names representing the likely causes.
        """
        causes = []
        for pred in self.graph.predecessors(effect):
            if self.graph[pred][effect].get("confidence", 0) >= threshold:
                causes.append(pred)
        return causes

    def get_likely_effects(self, cause: str, threshold: float = 0.7) -> List[str]:
        """Predicts the likely effects of a given cause.

        This method queries the graph to find all successor nodes of the given
        `cause` node whose edge confidence exceeds the specified threshold.

        Args:
            cause (str): The name of the cause node.
            threshold (float, optional): The minimum confidence level for an
                effect to be considered likely. Defaults to 0.7.

        Returns:
            List[str]: A list of node names representing the likely effects.
        """
        effects = []
        for succ in self.graph.successors(cause):
            if self.graph[cause][succ].get("confidence", 0) >= threshold:
                effects.append(succ)
        return effects
