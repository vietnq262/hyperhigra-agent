"""Question decomposition module."""

from __future__ import annotations

from typing import List

from src.prompts.prompts import QUESTION_DECOMPOSITION_PROMPT


class QuestionDecomposer:
    """# FROM HiGraAgent

    Decompose a multi-hop question into smaller sub-questions.
    """

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def decompose(self, question: str) -> List[str]:
        """Return one sub-question per line, falling back to the original question."""
        response = self.llm_client.complete_sync(
            QUESTION_DECOMPOSITION_PROMPT.format(question=question)
        )
        questions = [line.strip("- •	 ") for line in response.splitlines() if line.strip()]
        return questions or [question]
