"""Evidence accumulation agent."""

from __future__ import annotations

from typing import List

from src.prompts.prompts import SUFFICIENCY_CHECK_PROMPT


class Librarian:
    """# FROM HiGraAgent

    Manage deduplicated evidence and ask an LLM whether it is sufficient.
    """

    def __init__(self, config: dict, llm_client) -> None:
        self.config = config
        self.llm_client = llm_client
        self._passages: List[dict] = []

    def update(self, new_passages: List[dict]) -> None:
        """Add new evidence while deduplicating by passage_id."""
        known = {item.get("passage_id") for item in self._passages}
        for passage in new_passages:
            if passage.get("passage_id") in known:
                continue
            self._passages.append(passage)
            known.add(passage.get("passage_id"))

    def is_sufficient(self, question: str) -> bool:
        """Judge whether the current evidence is enough to answer the question."""
        if not self._passages:
            return False
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            question=question,
            context="\n\n".join(item.get("text", "") for item in self._passages),
        )
        response = self.llm_client.complete_sync(prompt).strip().lower()
        return response.startswith("yes") or response.startswith("true")

    def get_context(self) -> List[dict]:
        """Return accumulated evidence."""
        return list(self._passages)

    def reset(self) -> None:
        """Reset the evidence buffer."""
        self._passages = []
