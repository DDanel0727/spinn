"""
Shared utilities: JSON parsing, LLM wrapper, timestamp helper.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def extract_json_block(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            inner = "\n".join(lines[1:-1]).strip()
            if inner:
                t = inner
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    payload = extract_json_block(text)
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


class StructuredLLMCall:
    """Thin wrapper around the base agent's _call_llm that enforces JSON output."""

    def __init__(self, agent: Any, temperature: float, default_max_tokens: int):
        self.agent = agent
        self.temperature = temperature
        self.default_max_tokens = default_max_tokens

    def call_json(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        retry_on_parse_error: bool = True,
        call_label: str = "call",
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any], Dict[str, Any]]:
        """
        Returns (parsed_dict | None, raw_text, usage, full_api_response).
        If the first attempt fails JSON parse and retry is enabled, sends a
        one-shot repair request.
        """
        system = (
            "Return JSON only. No markdown fences, no commentary. "
            "If uncertain, still return best-effort valid JSON."
        )
        mt = int(max_tokens) if max_tokens is not None else self.default_max_tokens
        result = self.agent._call_llm(system, prompt, max_tokens=mt)
        raw = result.get("response_text", "") or ""
        usage = result.get("usage", {}) or {}
        full_api = result.get("full_api_response", {}) or {}
        parsed = parse_json_text(raw)
        if parsed is not None:
            return parsed, raw, usage, _safe_dict(full_api)

        if not retry_on_parse_error:
            return None, raw, usage, _safe_dict(full_api)

        repair_prompt = (
            f"The previous output for '{call_label}' was not valid JSON.\n"
            "Fix it and return strict JSON only.\n\n"
            f"Original output:\n{raw}"
        )
        r2 = self.agent._call_llm(system, repair_prompt, max_tokens=mt)
        raw2 = r2.get("response_text", "") or ""
        usage2 = r2.get("usage", {}) or {}
        full2 = r2.get("full_api_response", {}) or {}
        parsed2 = parse_json_text(raw2)
        merged_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)) + int(usage2.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)) + int(usage2.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)) + int(usage2.get("total_tokens", 0)),
        }
        merged_full: Dict[str, Any] = {}
        if isinstance(full_api, dict) and full_api:
            merged_full["primary"] = full_api
        if isinstance(full2, dict) and full2:
            merged_full["retry"] = full2
        return parsed2, raw2, merged_usage, merged_full


def _safe_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}
