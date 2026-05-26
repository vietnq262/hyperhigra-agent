"""Question clarification module."""

from __future__ import annotations

from src.prompts.prompts import QUESTION_CLARIFICATION_PROMPT


class QuestionClarifier:
    """# FROM HiGraAgent

    Clarify user questions before retrieval and reasoning.
    """

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def clarify(self, question: str) -> str:
        """Return a clarified question string."""
        response = self.llm_client.complete_sync(
            QUESTION_CLARIFICATION_PROMPT.format(question=question)
        ).strip()
        return response or question
