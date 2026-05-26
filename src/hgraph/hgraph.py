"""Hypergraph data structure for HyperHiGraAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import dill


class Hypergraph:
    """# FROM HGRAG

    Lightweight hypergraph where nodes are entities and each hyperedge stores
    a set of entities, a source passage, and a weight.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, dict] = {}
        self.hyperedges: List[dict] = []
        self._incidence: Dict[str, List[int]] = {}

    def add_node(self, entity: str, metadata: dict | None = None) -> None:
        """Add or update an entity node."""
        if entity not in self.nodes:
            self.nodes[entity] = metadata or {}
            self._incidence[entity] = []
        elif metadata:
            self.nodes[entity].update(metadata)

    def add_hyperedge(self, entities: List[str], passage: str, weight: float = 1.0) -> None:
        """Add a hyperedge connecting one or more entities."""
        unique_entities = []
        for entity in entities:
            if entity and entity not in unique_entities:
                self.add_node(entity)
                unique_entities.append(entity)
        if not unique_entities:
            return
        hyperedge_id = len(self.hyperedges)
        self.hyperedges.append(
            {
                "entities": unique_entities,
                "passage": passage,
                "weight": float(weight),
            }
        )
        for entity in unique_entities:
            self._incidence.setdefault(entity, []).append(hyperedge_id)

    def diffuse(
        self,
        seed_scores: Dict[str, float],
        alpha: float = 0.85,
        steps: int = 3,
    ) -> Dict[str, float]:
        """Run personalized hypergraph diffusion over the entity nodes."""
        scores = {node: float(seed_scores.get(node, 0.0)) for node in self.nodes}
        seeds = dict(scores)
        if not scores:
            return {}

        for _ in range(max(steps, 0)):
            hyperedge_scores: List[float] = []
            for edge in self.hyperedges:
                members = edge["entities"]
                hyperedge_scores.append(sum(scores.get(node, 0.0) for node in members) / len(members))

            updated: Dict[str, float] = {node: 0.0 for node in self.nodes}
            for node, edge_ids in self._incidence.items():
                weighted_sum = 0.0
                weight_total = 0.0
                for edge_id in edge_ids:
                    edge = self.hyperedges[edge_id]
                    edge_weight = float(edge.get("weight", 1.0))
                    weighted_sum += edge_weight * hyperedge_scores[edge_id]
                    weight_total += edge_weight
                propagated = weighted_sum / weight_total if weight_total else 0.0
                updated[node] = alpha * seeds.get(node, 0.0) + (1.0 - alpha) * propagated
            scores = updated
        return scores

    def get_top_nodes(self, scores: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        """Return the highest-scoring nodes."""
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def save(self, path: str) -> None:
        """Serialize the hypergraph with dill."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            dill.dump(self, handle)

    @classmethod
    def load(cls, path: str) -> "Hypergraph":
        """Load a serialized hypergraph."""
        with Path(path).open("rb") as handle:
            graph = dill.load(handle)
        if not isinstance(graph, cls):
            raise TypeError("Serialized object is not a Hypergraph")
        return graph
