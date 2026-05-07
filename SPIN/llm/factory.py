"""
LLM client factory (OpenAI-compatible endpoints only in this checkout).
"""

import os
from typing import Optional

from llm.base import BaseLLMClient
from llm.openai_client import OpenAIClient


def get_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> BaseLLMClient:
    """Return a client for ``openai`` or ``openrouter`` (both use OpenAI-compatible HTTP API)."""
    provider = (provider or "").lower().strip()
    key = api_key
    base = api_base

    if provider == "openai":
        key = key or os.getenv("OPENAI_API_KEY")
        return OpenAIClient(model=model, api_key=key, api_base=base)
    if provider == "openrouter":
        key = key or os.getenv("OPENROUTER_API_KEY")
        base = base or "https://openrouter.ai/api/v1"
        return OpenAIClient(model=model, api_key=key, api_base=base)

    raise ValueError(
        f"Unknown provider: {provider}. This SPIN checkout supports: openai, openrouter"
    )
