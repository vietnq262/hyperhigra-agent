"""Evidence retrieval agent."""

from __future__ import annotations

from typing import List


class Seeker:
    """# FROM HiGraAgent

    Evidence retrieval agent that turns a sub-question into passages.
    """

    def __init__(self, config: dict, retriever, ner, llm_client) -> None:
        self.config = config
        self.retriever = retriever
        self.ner = ner
        self.llm_client = llm_client

    def seek(self, sub_question: str, context: List[dict]) -> List[dict]:
        """Retrieve fresh evidence for a sub-question."""
        entities = self.ner.extract_from_query(sub_question)
        top_k = int(self.config.get("retrieval", {}).get("passage_top_k", 20))
        passages = self.retriever.retrieve(sub_question, entities, top_k)
        existing_ids = {item.get("passage_id") for item in context}
        return [item for item in passages if item.get("passage_id") not in existing_ids]
