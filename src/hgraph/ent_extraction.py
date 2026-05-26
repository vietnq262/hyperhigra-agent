"""Entity extraction utilities for hypergraph construction."""

from __future__ import annotations

from typing import List

from src.utils import deduplicate, normalize_entity


class EntityExtractor:
    """Extract entities from documents and queries using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("spaCy is required for EntityExtractor") from exc

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def _extract(self, text: str) -> List[str]:
        doc = self.nlp(text or "")
        entities = [normalize_entity(ent.text) for ent in doc.ents if normalize_entity(ent.text)]
        return deduplicate(entities)

    def extract_from_doc(self, text: str) -> List[str]:
        """Extract normalized entities from a document passage."""
        return self._extract(text)

    def extract_from_query(self, text: str) -> List[str]:
        """Extract normalized entities from a user query."""
        return self._extract(text)
