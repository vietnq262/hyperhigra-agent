"""Merged prompt templates for HyperHiGraAgent."""

QUESTION_CLARIFICATION_PROMPT = """You are a question clarifier. Rewrite the question so that it is explicit and self-contained.

Question:
{question}

Clarified question:"""

QUESTION_DECOMPOSITION_PROMPT = """You are a multi-hop question decomposer. Break the question into minimal answerable sub-questions, one per line.

Question:
{question}

Sub-questions:"""

SUFFICIENCY_CHECK_PROMPT = """You are an evidence sufficiency judge. Answer Yes or No only.

Question:
{question}

Evidence:
{context}

Is the evidence sufficient to answer the question?"""

QA_SINGLE_STEP_PROMPT = """Answer the question using the provided context. If the answer is not supported, say so briefly.

Question:
{question}

Context:
{context}

Answer:"""

QA_ADAPTIVE_PROMPT = """You are the final answering agent in HyperHiGraAgent. Use the accumulated evidence to answer concisely and faithfully.

Question:
{question}

Evidence:
{context}

Answer:"""

ENTITY_EXTRACTION_PROMPT = """Extract the salient named entities from the text below as a newline-separated list.

Text:
{text}

Entities:"""
