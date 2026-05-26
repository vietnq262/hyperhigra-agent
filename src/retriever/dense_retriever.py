"""Dense retrieval backed by sentence-transformers and FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import List

import dill


class DenseRetriever:
    """# FROM HGRAG

    Dense retriever that stores passage metadata alongside a FAISS index.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents: List[dict] = []

    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency gated
                raise ImportError("sentence-transformers is required for DenseRetriever") from exc
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_index(self, documents: List[dict], embedding_path: str) -> None:
        """Encode passages and persist a FAISS similarity index."""
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("faiss-cpu is required for DenseRetriever") from exc

        texts = [str(doc.get("text") or doc.get("passage") or "") for doc in documents]
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        normalized_documents = []
        for idx, doc in enumerate(documents):
            normalized_documents.append(
                {
                    "passage_id": doc.get("passage_id", doc.get("id", idx)),
                    "text": str(doc.get("text") or doc.get("passage") or ""),
                    "entities": list(doc.get("entities", [])),
                }
            )

        target_dir = Path(embedding_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(target_dir / "index.faiss"))
        with (target_dir / "metadata.dill").open("wb") as handle:
            dill.dump({"documents": normalized_documents, "model_name": self.model_name}, handle)

        self.index = index
        self.documents = normalized_documents

    def load_index(self, embedding_path: str) -> None:
        """Load the FAISS index and stored passage metadata."""
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("faiss-cpu is required for DenseRetriever") from exc

        source_dir = Path(embedding_path)
        self.index = faiss.read_index(str(source_dir / "index.faiss"))
        with (source_dir / "metadata.dill").open("rb") as handle:
            payload = dill.load(handle)
        self.documents = payload.get("documents", [])
        self.model_name = payload.get("model_name", self.model_name)
        self.model = None

    def retrieve(self, query: str, top_k: int) -> List[dict]:
        """Return top passages with scores and attached entities."""
        if self.index is None:
            raise RuntimeError("Dense index is not loaded")
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("faiss-cpu is required for DenseRetriever") from exc

        model = self._load_model()
        query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[dict] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.documents):
                continue
            doc = dict(self.documents[index])
            doc["score"] = float(score)
            results.append(doc)
        return results
