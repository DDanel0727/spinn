"""
Phase 2B: Decision Readout.

Supports two modes:
  - single: one ElicitedState for the whole trial
  - per_item: one elicited state per question/item

The LLM prompt receives either:
  - elicited_state (dict form)
  - per-item elicited states + trial question text/options

It MUST NOT receive the raw participant profile or the personality_core.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agents.prompts import (
    answer_repair_prompt,
    decision_readout_prompt,
    decision_readout_prompt_per_item,
)
from agents.response_finalize import (
    compose_answer_line,
    merge_decision_answers,
    parse_q_assignments,
    required_q_labels,
)
from agents.schemas import DecisionProcess, ElicitedState, SpinConfig, normalize_per_question_choice
from agents.utils import StructuredLLMCall
from evaluation.sanity_check import MIN_RESPONSE_TEXT_LEN_FOR_RAW_SANITY

MAX_ANSWER_REPAIR_ATTEMPTS = 2


def _too_short_for_sanity(text: str) -> bool:
    return not text or len(text.strip()) < MIN_RESPONSE_TEXT_LEN_FOR_RAW_SANITY


def _build_req_instruction(trial_info: Dict[str, Any]) -> str:
    items = trial_info.get("items") or []
    req_q = [f"Q{i + 1}" for i in range(len(items))]
    if not req_q:
        return ""
    return (
        f"\nCRITICAL: This trial requires ALL of {req_q} answered.\n"
        "per_question_choice must include every key.\n"
        "final_answer_line must be ONE line, comma+space separated, "
        f"listing every key (exactly {len(req_q)} assignments).\n"
    )


def _decision_signal_summary_from_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    sv = state.get("state_variables") or {}
    return {
        "dominant_mode": state.get("dominant_mode"),
        "trust_in_other": sv.get("trust_in_other"),
        "perceived_exploitation_risk": sv.get("perceived_exploitation_risk"),
        "self_protection_drive": sv.get("self_protection_drive"),
        "cooperation_drivers": list(state.get("cooperation_drivers") or []),
        "cooperation_inhibitors": list(state.get("cooperation_inhibitors") or []),
    }


def _decision_signal_summary(elicited_state: ElicitedState) -> Dict[str, Any]:
    return _decision_signal_summary_from_dict(elicited_state.to_dict())


def _per_item_decision_signal_summary(per_item_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in per_item_states:
        item_index = int(entry.get("item_index", 0))
        item_id = str(entry.get("item_id", f"Q{item_index + 1}"))
        state = entry.get("state") or {}
        summary = _decision_signal_summary_from_dict(state)
        out.append(
            {
                "item_index": item_index,
                "item_id": item_id,
                "question_key": f"Q{item_index + 1}",
                **summary,
            }
        )
    return out


def read_decision(
    llm: StructuredLLMCall,
    elicited_state_or_list: Any,
    trial_prompt: str,
    trial_info: Dict[str, Any],
    config: SpinConfig,
    mode: str = "single",
) -> Tuple[DecisionProcess, str, Dict[str, Any]]:
    """
    Returns (DecisionProcess, final_answer_text, calls_record).
    calls_record is a dict of {call_name: {prompt, raw, parsed, usage, full_api}}.

    DecisionProcess:
      - per_item_reasoning is populated from the model JSON on successful `call_json`
        (see DecisionProcess.from_dict); keys are question labels (e.g. Q1) -> reasoning text.
      - On parse failure, per_item_reasoning is empty; answer_repair may update
        final_answer_line / per_question_choice only (reasoning is not re-generated).
    """
    req_instruction = _build_req_instruction(trial_info)
    if mode == "per_item":
        per_item_states = list(elicited_state_or_list or [])
        prompt = decision_readout_prompt_per_item(
            per_item_states,
            trial_prompt,
            trial_info,
            req_instruction=req_instruction,
        )
        decision_signals: Any = _per_item_decision_signal_summary(per_item_states)
    else:
        elicited_state = elicited_state_or_list
        assert isinstance(elicited_state, ElicitedState)
        prompt = decision_readout_prompt(
            elicited_state.to_dict(),
            trial_prompt,
            trial_info,
            req_instruction=req_instruction,
        )
        decision_signals = _decision_signal_summary(elicited_state)

    parsed, raw, usage, full_api = llm.call_json(
        prompt,
        max_tokens=config.decision_max_tokens,
        retry_on_parse_error=config.parse_error_retry,
        call_label="decision_readout",
    )

    calls: Dict[str, Any] = {
        "decision_readout": {
            "prompt": prompt if config.save_raw_prompts else None,
            "raw": raw if config.save_raw_responses else None,
            "parsed": parsed,
            "usage": usage,
            "full_api_response": full_api,
            "mode": mode,
            "decision_signals": decision_signals,
        }
    }

    req = required_q_labels(trial_info)

    if parsed is None:
        dp = DecisionProcess(
            per_question_choice={},
            final_answer_line=str(raw or ""),
            per_item_reasoning={},
        )
        merged: Dict[str, str] = {}
        composed = str(raw or "").strip()
        missing = list(req)
    else:
        parsed = dict(parsed)
        parsed["per_question_choice"] = normalize_per_question_choice(parsed.get("per_question_choice"))
        pq = parsed["per_question_choice"]
        fal = str(parsed.get("final_answer_line", ""))
        dp = DecisionProcess.from_dict(parsed)
        composed, merged, missing = merge_decision_answers(trial_info, pq, fal)
        dp.final_answer_line = composed

    # Repair when Q labels are missing OR line is too short for benchmark sanity_check (avoids 20% "other" raw failures).
    attempt = 0
    while (
        config.enable_answer_repair
        and req
        and attempt < MAX_ANSWER_REPAIR_ATTEMPTS
        and (missing or _too_short_for_sanity(composed))
    ):
        key = "answer_repair" if attempt == 0 else f"answer_repair_retry_{attempt}"
        composed, merged, missing, repair_call = _run_repair(
            llm, trial_prompt, trial_info, req, dict(merged), config
        )
        calls[key] = repair_call
        dp.final_answer_line = composed
        attempt += 1

    out_text = (composed or "").strip() or (raw or "").strip()
    return dp, out_text, calls


def _run_repair(
    llm: StructuredLLMCall,
    trial_prompt: str,
    trial_info: Dict[str, Any],
    required: List[str],
    merged: Dict[str, str],
    config: SpinConfig,
) -> Tuple[str, Dict[str, str], List[str], Dict[str, Any]]:
    prompt = answer_repair_prompt(trial_prompt, required, merged, trial_info)
    parsed_r, raw_r, usage_r, full_r = llm.call_json(
        prompt,
        retry_on_parse_error=config.parse_error_retry,
        call_label="answer_repair",
        max_tokens=config.answer_repair_max_tokens,
    )
    call_record = {
        "prompt": prompt if config.save_raw_prompts else None,
        "raw": raw_r if config.save_raw_responses else None,
        "parsed": parsed_r,
        "usage": usage_r,
        "full_api_response": full_r,
    }
    line_r = ((parsed_r or {}).get("final_answer_line") or "").strip()
    for k, v in parse_q_assignments(line_r).items():
        if k in required and v:
            merged[k] = v
    missing = [k for k in required if k not in merged or not str(merged.get(k, "")).strip()]
    composed = compose_answer_line(required, merged) if not missing else (line_r or compose_answer_line([k for k in required if k in merged], merged))
    return composed, merged, missing, call_record
