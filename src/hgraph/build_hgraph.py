"""Hypergraph construction and embedding pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

import dill

from src.hgraph.ent_extraction import EntityExtractor
from src.hgraph.hgraph import Hypergraph


class HypergraphBuilder:
    """# FROM HGRAG

    Build a sentence-level and passage-level hypergraph from documents.
    """

    def __init__(self, sentencizer_model: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("spaCy is required for HypergraphBuilder") from exc

        try:
            self.sentencizer = spacy.load(sentencizer_model, disable=["tagger", "parser", "lemmatizer", "ner"])
        except OSError:
            self.sentencizer = spacy.blank("en")
        if "sentencizer" not in self.sentencizer.pipe_names:
            self.sentencizer.add_pipe("sentencizer")

    def build(self, documents: List[dict], entity_extractor: EntityExtractor) -> Hypergraph:
        """Build a hypergraph from sentence and passage entity groupings."""
        hypergraph = Hypergraph()
        for doc in documents:
            text = str(doc.get("text") or doc.get("passage") or "")
            if not text.strip():
                continue
            sentence_doc = self.sentencizer(text)
            for sentence in sentence_doc.sents:
                sentence_text = sentence.text.strip()
                if not sentence_text:
                    continue
                sentence_entities = entity_extractor.extract_from_doc(sentence_text)
                hypergraph.add_hyperedge(sentence_entities, passage=sentence_text, weight=1.0)
            passage_entities = entity_extractor.extract_from_doc(text)
            hypergraph.add_hyperedge(passage_entities, passage=text, weight=0.5)
        return hypergraph

    def build_embeddings(self, documents: List[dict], embedding_path: str) -> None:
        """Build and save sentence-transformer embeddings with FAISS."""
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("sentence-transformers and faiss-cpu are required") from exc

        texts = [str(doc.get("text") or doc.get("passage") or "") for doc in documents]
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        target_dir = Path(embedding_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(target_dir / "index.faiss"))
        with (target_dir / "metadata.dill").open("wb") as handle:
            dill.dump({"documents": documents, "model_name": "sentence-transformers/all-MiniLM-L6-v2"}, handle)
