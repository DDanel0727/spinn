"""OpenAI-compatible LLM client (used by SPIN testing)."""

from llm.base import BaseLLMClient, Message, ContentPart
from llm.factory import get_client

__all__ = [
    "BaseLLMClient",
    "Message",
    "ContentPart",
    "get_client",
]
