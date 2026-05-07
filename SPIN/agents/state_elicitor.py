"""
Phase 2A: State Elicitation.

Given a pre-compiled PersonalityCore and the current trial, uses a single LLM
call to produce an ElicitedState — the trial-level psychological activation.

The LLM prompt receives:
  - personality_core (dict form)
  - trial stimulus (trial_prompt + trial_info summary)

It MUST NOT receive the raw participant profile text.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from agents.prompts import state_elicitation_prompt
from agents.schemas import (
    ElicitedState,
    PersonalityCore,
    SpinConfig,
)
from agents.utils import StructuredLLMCall


def elicit_state(
    llm: StructuredLLMCall,
    personality_core: PersonalityCore,
    trial_prompt: str,
    trial_info: Dict[str, Any],
    config: SpinConfig,
    trial_stimulus: Dict[str, Any] | None = None,
) -> Tuple[ElicitedState, Dict[str, Any]]:
    """
    Returns (ElicitedState, call_record).
    """
    prompt = state_elicitation_prompt(
        personality_core.to_dict(),
        trial_prompt,
        trial_info,
        trial_stimulus=trial_stimulus,
    )
    parsed, raw, usage, full_api = llm.call_json(
        prompt,
        max_tokens=config.elicitation_max_tokens,
        retry_on_parse_error=config.parse_error_retry,
        call_label="state_elicitation",
    )

    call_record: Dict[str, Any] = {
        "prompt": prompt if config.save_raw_prompts else None,
        "raw": raw if config.save_raw_responses else None,
        "parsed": parsed,
        "usage": usage,
        "full_api_response": full_api,
    }

    if parsed is None:
        state = _fallback_state()
        call_record["fallback"] = True
        return state, call_record

    state = ElicitedState.from_dict(parsed)
    return state, call_record


def _fallback_state() -> ElicitedState:
    """Neutral fallback when LLM parse fails."""
    return ElicitedState(
        salient_cues=[],
        activated_dispositions=[],
        state_variables={
            "arousal": 0.5,
            "conflict": 0.5,
            "social_concern": 0.5,
            "trust_in_other": 0.5,
            "perceived_exploitation_risk": 0.5,
            "self_protection_drive": 0.5,
        },
        cooperation_drivers=[],
        cooperation_inhibitors=[],
        dominant_mode="deliberative",
        decision_rationale="Fallback: no state was successfully elicited.",
    )
