"""Hybrid lexical+dense retrieval for ablation experiments."""

from __future__ import annotations

from typing import List


class HybridRetriever:
    """# FROM HiGraAgent

    Fuse BM25 and dense retrieval with Reciprocal Rank Fusion.
    """

    def __init__(self, dense_retriever, documents: List[dict] | None = None) -> None:
        self.dense_retriever = dense_retriever
        self.documents = documents or getattr(dense_retriever, "documents", [])
        self.bm25 = None
        if self.documents:
            self._build_bm25(self.documents)

    def _build_bm25(self, documents: List[dict]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("rank-bm25 is required for HybridRetriever") from exc
        tokenized = [str(doc.get("text") or "").lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.documents = documents

    def _rrf(self, dense_results: List[dict], bm25_results: List[dict], k: int = 10) -> List[dict]:
        fused = {}
        for ranked_list in (dense_results, bm25_results):
            for rank, item in enumerate(ranked_list, start=1):
                passage_id = str(item.get("passage_id", item.get("text", rank)))
                if passage_id not in fused:
                    fused[passage_id] = {
                        "passage_id": item.get("passage_id", passage_id),
                        "text": item.get("text", ""),
                        "entities": item.get("entities", []),
                        "score": 0.0,
                    }
                fused[passage_id]["score"] += 1.0 / (k + rank)
        return sorted(fused.values(), key=lambda item: item["score"], reverse=True)

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        """Retrieve passages with dense and BM25 fusion."""
        dense_results = self.dense_retriever.retrieve(query, top_k)
        if self.bm25 is None:
            self._build_bm25(self.documents)
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(bm25_scores), key=lambda item: item[1], reverse=True)[:top_k]
        bm25_results = []
        for idx, score in ranked:
            doc = dict(self.documents[idx])
            doc["score"] = float(score)
            bm25_results.append(doc)
        return self._rrf(dense_results, bm25_results, k=10)[:top_k]
