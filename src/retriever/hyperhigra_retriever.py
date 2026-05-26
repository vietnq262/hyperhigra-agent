"""Hypergraph-aware retrieval for HyperHiGraAgent."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.hgraph.hgraph import Hypergraph


class HyperHiGraRetriever:
    """# FROM HGRAG
    # FROM HiGraAgent

    Combine dense retrieval, hypergraph diffusion, and reciprocal rank fusion.
    """

    def __init__(self, config: dict, hgraph: Hypergraph, dense_retriever) -> None:
        self.config = config
        self.hgraph = hgraph
        self.dense_retriever = dense_retriever
        self.passage_lookup = {
            doc.get("text"): doc for doc in getattr(dense_retriever, "documents", []) if doc.get("text")
        }

    def retrieve(self, query: str, query_entities: List[str], top_k: int) -> List[dict]:
        """Retrieve passages using dense seeds and hypergraph diffusion."""
        retrieval_cfg = self.config.get("retrieval", {})
        seed_top_k = int(retrieval_cfg.get("similarity_passage_top_k", top_k))
        dense_scores = self.dense_retriever.retrieve(query, seed_top_k)
        seed_scores = self._passages_to_node_scores(dense_scores)
        for entity in query_entities:
            seed_scores[entity] = max(seed_scores.get(entity, 0.0), 1.0)

        diffused_nodes = self.hgraph.diffuse(
            seed_scores,
            alpha=float(retrieval_cfg.get("ppr_alpha", 0.85)),
            steps=int(retrieval_cfg.get("diffusion_steps", 3)),
        )
        diffused_passages = self._node_scores_to_passages(diffused_nodes)
        fused = self._rrf_fusion(
            dense_scores,
            diffused_passages,
            k=int(retrieval_cfg.get("rrf_ppr_constant", 10)),
        )
        return fused[:top_k]

    def _rrf_fusion(self, list1: List[dict], list2: List[dict], k: int = 10) -> List[dict]:
        """Fuse ranked passage lists with Reciprocal Rank Fusion."""
        scores: Dict[str, dict] = {}
        for ranked_list in (list1, list2):
            for rank, item in enumerate(ranked_list, start=1):
                passage_id = str(item.get("passage_id", item.get("text", rank)))
                if passage_id not in scores:
                    scores[passage_id] = {
                        "passage_id": item.get("passage_id", passage_id),
                        "text": item.get("text", ""),
                        "entities": item.get("entities", []),
                        "score": 0.0,
                    }
                scores[passage_id]["score"] += 1.0 / (k + rank)
        return sorted(scores.values(), key=lambda item: item["score"], reverse=True)

    def _passages_to_node_scores(self, passage_scores: List[dict]) -> Dict[str, float]:
        """Map dense passage scores into entity seed scores."""
        node_scores: Dict[str, float] = defaultdict(float)
        for passage in passage_scores:
            entities = passage.get("entities", [])
            if not entities:
                lookup = self.passage_lookup.get(passage.get("text"))
                entities = lookup.get("entities", []) if lookup else []
            for entity in entities:
                node_scores[entity] += float(passage.get("score", 0.0))
        return dict(node_scores)

    def _node_scores_to_passages(self, node_scores: Dict[str, float]) -> List[dict]:
        """Project diffused entity scores back onto supporting passages."""
        by_text: Dict[str, dict] = {}
        for edge in self.hgraph.hyperedges:
            entities = edge.get("entities", [])
            if not entities:
                continue
            passage_text = edge.get("passage", "")
            score = sum(node_scores.get(entity, 0.0) for entity in entities) / len(entities)
            if passage_text not in by_text or score > by_text[passage_text]["score"]:
                base_doc = self.passage_lookup.get(passage_text, {})
                by_text[passage_text] = {
                    "passage_id": base_doc.get("passage_id", passage_text[:32]),
                    "text": passage_text,
                    "entities": base_doc.get("entities", entities),
                    "score": float(score),
                }
        return sorted(by_text.values(), key=lambda item: item["score"], reverse=True)
