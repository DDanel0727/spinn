"""
Phase 1: Personality Compilation.

Builds a PersonalityCore from the participant profile using a single LLM call.
This is a participant-level operation: called ONCE per participant, before any
trial is seen. The resulting PersonalityCore is cached and reused for every
subsequent trial.

The LLM prompt receives ONLY the profile — no trial, scenario, or item content
is allowed to leak into this phase.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from agents.prompts import personality_compilation_prompt
from agents.schemas import SpinConfig, PersonalityCore
from agents.utils import StructuredLLMCall


def compile_personality(
    llm: StructuredLLMCall,
    participant_id: int,
    profile: Dict[str, Any],
    config: SpinConfig,
) -> Tuple[PersonalityCore, Dict[str, Any]]:
    """
    Returns (PersonalityCore, call_record).
    call_record contains prompt/raw/parsed/usage/full_api_response for logging.
    """
    prompt = personality_compilation_prompt(profile)
    parsed, raw, usage, full_api = llm.call_json(
        prompt,
        max_tokens=config.personality_max_tokens,
        retry_on_parse_error=config.parse_error_retry,
        call_label="personality_compilation",
    )

    call_record: Dict[str, Any] = {
        "prompt": prompt if config.save_raw_prompts else None,
        "raw": raw if config.save_raw_responses else None,
        "parsed": parsed,
        "usage": usage,
        "full_api_response": full_api,
    }

    if parsed is None:
        core = _fallback_core(participant_id, profile)
        call_record["fallback"] = True
        return core, call_record

    parsed["participant_id"] = participant_id
    core = PersonalityCore.from_dict(parsed)
    return core, call_record


def _fallback_core(participant_id: int, profile: Dict[str, Any]) -> PersonalityCore:
    """Deterministic fallback when LLM parse fails — neutral midpoint personality."""
    return PersonalityCore(
        participant_id=participant_id,
        trait_dimensions={
            "fairness_sensitivity": 0.5,
            "norm_sensitivity": 0.5,
            "reciprocity_orientation": 0.5,
            "uncertainty_aversion": 0.5,
            "self_protection_tendency": 0.5,
        },
        disposition_rules=[],
        default_decision_style="deliberative",
        natural_language_summary=f"Fallback neutral personality for participant {participant_id}.",
    )
