"""
OpenAI API client (also used for OpenAI-compatible endpoints like xAI).
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llm.base import BaseLLMClient
from llm.openai_chat_params import add_max_output_to_kwargs, maybe_inject_azure_foundry_deployment


def _image_to_data_url(image: Union[str, bytes, Path], mime: Optional[str] = None) -> str:
    """Convert image path/bytes/base64 to data URL for OpenAI content."""
    if isinstance(image, Path):
        image = str(image)
    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        path = Path(image)
        if path.exists():
            mime = mime or "image/png"
            if path.suffix.lower() in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif path.suffix.lower() == ".webp":
                mime = "image/webp"
            data = path.read_bytes()
            b64 = base64.standard_b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        # Assume base64 string
        b64 = image
        mime = mime or "image/png"
        return f"data:{mime};base64,{b64}"
    if isinstance(image, bytes):
        b64 = base64.standard_b64encode(image).decode("ascii")
        mime = mime or "image/png"
        return f"data:{mime};base64,{b64}"
    raise TypeError(f"image must be path, bytes, or base64 str, got {type(image)}")


def _normalize_content(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize content to OpenAI format: list of {type, text} or {type, image_url}."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for part in content:
        if part.get("type") == "text":
            out.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image":
            url = _image_to_data_url(
                part["image"],
                mime=part.get("mime"),
            )
            out.append({"type": "image_url", "image_url": {"url": url}})
        else:
            out.append(part)
    return out


def _messages_to_openai(messages: List[Dict[str, Any]], system: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert unified messages to OpenAI API format."""
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system" and not result and not system:
            result.append({"role": "system", "content": content if isinstance(content, str) else content[0].get("text", "")})
            continue
        normalized = _normalize_content(content) if isinstance(content, list) else [{"type": "text", "text": content}]
        result.append({"role": role, "content": normalized})
    return result


def _is_full_chat_completions_url(url: Optional[str]) -> bool:
    text = (url or "").strip().lower()
    return "/chat/completions" in text


def _build_full_endpoint_headers(api_key: str, url: Optional[str]) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if url and ".services.ai.azure.com" in url:
        headers["api-key"] = api_key
    return headers


def _extract_content_from_chat_response(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "".join(texts)
    return str(content or "")


def _load_env_from_optional_dotenv() -> None:
    """
    If SPIN_DOTENV_PATH is set, load KEY=VALUE lines into os.environ for keys
    not already defined. Use this locally for Azure/OpenAI secrets; keep unset
    in anonymous/public bundles.
    """
    raw = (os.environ.get("SPIN_DOTENV_PATH") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions API (and OpenAI-compatible endpoints, including Azure OpenAI)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        # Optional: SPIN_DOTENV_PATH points at a local .env (never commit secrets or personal paths).
        _load_env_from_optional_dotenv()

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self._azure_api_version = azure_version

        effective_base = (api_base or os.getenv("OPENAI_BASE_URL") or "").strip()
        # Prefer explicit OpenAI-compatible base so AZURE_* alone does not force Azure routing
        if effective_base:
            self._use_azure = False
            api_base = effective_base
            api_key = api_key or os.getenv("OPENAI_API_KEY")
        elif azure_endpoint and azure_key:
            self._use_azure = True
            api_key = azure_key
            api_base = azure_endpoint
            if not self._azure_api_version:
                self._azure_api_version = "2025-04-01-preview"
        else:
            self._use_azure = False
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            # api_base kept as passed (None => official api.openai.com)

        if not api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY, or for Azure set "
                "AZURE_OPENAI_ENDPOINT and OPENAI_API_KEY (or AZURE_OPENAI_API_KEY) when OPENAI_BASE_URL is unset."
            )

        api_base = maybe_inject_azure_foundry_deployment(api_base, model)

        super().__init__(model=model, api_key=api_key, api_base=api_base)

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Import here so that openai is only required when actually used
        from openai import OpenAI, AzureOpenAI
        import httpx

        openai_messages = _messages_to_openai(messages, system=system)
        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        add_max_output_to_kwargs(kwargs, self.model, max_tokens)

        if _is_full_chat_completions_url(self.api_base):
            response = httpx.post(
                self.api_base,
                headers=_build_full_endpoint_headers(self.api_key, self.api_base),
                json=kwargs,
                timeout=45.0,
            )
            response.raise_for_status()
            return _extract_content_from_chat_response(response.json()) or ""
        if self._use_azure:
            # Azure OpenAI: use AzureOpenAI client with endpoint + api_version
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self._azure_api_version,
                azure_endpoint=self.api_base,
            )
        else:
            # Standard OpenAI / OpenRouter-style base (empty => default URL)
            oc_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.api_base:
                oc_kwargs["base_url"] = self.api_base
            client = OpenAI(**oc_kwargs)

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
