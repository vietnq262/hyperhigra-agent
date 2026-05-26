"""Async OpenAI client wrapper."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List


class OpenAIClient:
    """# FROM HiGraAgent

    Async-first OpenAI client with batched completion helpers.
    """

    def __init__(self, api_key: str = "", model_name: str = "gpt-4o-mini", batch_size: int = 200) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if not self.api_key:
            return None
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - dependency gated
                raise ImportError("openai>=1.0.0 is required for OpenAIClient") from exc
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def complete(self, prompt: str) -> str:
        """Generate a single completion."""
        client = self._get_client()
        if client is None:
            return "LLM client is not configured."
        response = await client.responses.create(model=self.model_name, input=prompt)
        return getattr(response, "output_text", "") or ""

    async def batch_complete(self, prompts: List[str]) -> List[str]:
        """Generate completions with bounded concurrency."""
        semaphore = asyncio.Semaphore(10)

        async def run(prompt: str) -> str:
            async with semaphore:
                return await self.complete(prompt)

        return await asyncio.gather(*(run(prompt) for prompt in prompts))

    def complete_sync(self, prompt: str) -> str:
        """Sync wrapper for async completion."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.complete(prompt))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(self.complete(prompt)))
            return future.result()
