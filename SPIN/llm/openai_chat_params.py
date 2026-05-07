"""
OpenAI Chat Completions: some models (e.g. GPT-5 family, parts of the o-series) require
`max_completion_tokens` instead of `max_tokens`. Azure deployments of the same names behave the same.

Azure AI Foundry inference URLs (*.services.ai.azure.com) must include the deployment segment:
  .../models/<deployment>/chat/completions
If the URL is written as .../models/chat/completions (missing the deployment segment), calls 404 with DeploymentNotFound.
"""

import os
import re
from typing import Any, Dict, Optional
from urllib.parse import quote


def maybe_inject_azure_foundry_deployment(
    api_base: Optional[str],
    model: Optional[str],
) -> Optional[str]:
    """
    When OPENAI_BASE_URL points at Foundry and the path is .../models/chat/completions,
    insert the deployment: .../models/<deployment>/chat/completions.

    Deployment name: env AZURE_INFERENCE_DEPLOYMENT_NAME (export before run) if set, else the request `model` name.
    """
    url = (api_base or "").strip()
    if not url or "services.ai.azure.com" not in url.lower():
        return api_base
    if not re.search(r"/models/chat/completions", url, re.IGNORECASE):
        return api_base
    if re.search(r"/models/[^/]+/chat/completions", url, re.IGNORECASE):
        return api_base
    dep = (os.getenv("AZURE_INFERENCE_DEPLOYMENT_NAME") or "").strip() or (model or "").strip()
    if not dep:
        return api_base
    dep_seg = quote(dep, safe="!$&'()*+,;=:@-._~")
    return re.sub(
        r"/models/chat/completions",
        f"/models/{dep_seg}/chat/completions",
        url,
        count=1,
        flags=re.IGNORECASE,
    )


def model_prefers_max_completion_tokens(model: Optional[str]) -> bool:
    """
    Whether chat.completions should use max_completion_tokens.

    Override with env:
      OPENAI_USE_MAX_COMPLETION_TOKENS=1|0  force on/off
    """
    override = os.getenv("OPENAI_USE_MAX_COMPLETION_TOKENS", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False

    m = (model or "").lower()
    if "gpt-5" in m:
        return True
    for prefix in ("o1", "o3", "o4"):
        if m == prefix or m.startswith(f"{prefix}-") or m.startswith(f"{prefix}/") or f"/{prefix}" in m:
            return True
    return False


def add_max_output_to_kwargs(
    kwargs: Dict[str, Any],
    model: Optional[str],
    max_tokens: Optional[int],
) -> None:
    """Set exactly one of max_tokens or max_completion_tokens on kwargs (mutually exclusive)."""
    if max_tokens is None:
        return
    kwargs.pop("max_tokens", None)
    kwargs.pop("max_completion_tokens", None)
    if model_prefers_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
