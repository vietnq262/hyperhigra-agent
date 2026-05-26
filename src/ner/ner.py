"""spaCy NER and fastcoref wrapper."""

from __future__ import annotations

from typing import List

from src.utils import deduplicate, normalize_entity


class NERModel:
    """# FROM HiGraAgent

    Provide coreference resolution and named entity extraction.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - dependency gated
            raise ImportError("spaCy is required for NERModel") from exc

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        try:
            from fastcoref import FCoref
            self.coref_model = FCoref()
        except Exception:  # pragma: no cover - optional runtime dependency
            self.coref_model = None

    def resolve_coreferences(self, text: str) -> str:
        """Resolve coreferences when the model is available, else return the input."""
        if not text or self.coref_model is None:
            return text
        try:
            prediction = self.coref_model.predict([text])[0]
            if hasattr(prediction, "get_resolved_text"):
                return prediction.get_resolved_text()
            resolved_text = getattr(prediction, "resolved_text", None)
            return resolved_text or text
        except Exception:
            return text

    def extract_entities(self, text: str) -> List[str]:
        """Extract normalized spaCy entities."""
        doc = self.nlp(text or "")
        entities = [normalize_entity(ent.text) for ent in doc.ents if normalize_entity(ent.text)]
        return deduplicate(entities)

    def extract_from_query(self, query: str) -> List[str]:
        """Extract entities from a query."""
        return self.extract_entities(query)

    def extract_from_doc(self, doc: str) -> List[str]:
        """Extract entities from a document."""
        return self.extract_entities(doc)
