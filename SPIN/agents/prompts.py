"""
All LLM prompt templates for SPIN.

Strict boundary enforcement:
- personality_compilation_prompt: sees ONLY profile, never trial content.
- state_elicitation_prompt: sees ONLY personality_core + current trial, never raw profile text.
- decision_readout_prompt: sees ONLY elicited_state + trial question/options, never profile or core.
- answer_repair_prompt: repair only, no identity/state re-generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.schemas import (
    ALLOWED_TRIGGERS,
    DECISION_STYLES,
    TRAIT_DIMENSIONS,
)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _compact_text(text: str, max_chars: int = 1200, tail_chars: int = 0) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    if tail_chars > 0 and max_chars > tail_chars + 5:
        head_chars = max_chars - tail_chars - 5
        return cleaned[:head_chars].rstrip() + " ... " + cleaned[-tail_chars:].lstrip()
    return cleaned[:max_chars].rstrip()


def _compact_trial_overview(trial_info: Dict[str, Any], trial_prompt: str) -> str:
    lines: List[str] = []
    instructions = _compact_text(trial_info.get("instructions") or "", max_chars=700, tail_chars=180)
    if instructions:
        lines.append(f"Instructions: {instructions}")

    items = trial_info.get("items") or []
    for i, item in enumerate(items):
        q = f"Q{i + 1}"
        item_id = str(item.get("id") or q)
        question = _compact_text(item.get("question") or "", max_chars=220)
        lines.append(f"{q} ({item_id}): {question}")
        options = item.get("options") or []
        if options:
            opt_text = ", ".join(
                f"{chr(65 + idx)}) {_compact_text(opt, max_chars=60)}"
                for idx, opt in enumerate(options[:6])
            )
            lines.append(f"Options: {opt_text}")

    response_spec = _compact_text(trial_info.get("response_spec") or "", max_chars=160)
    if response_spec:
        lines.append(f"Response spec: {response_spec}")

    if not lines:
        return _compact_text(trial_prompt, max_chars=1800, tail_chars=300)
    return "\n".join(lines)


def _fmt_state_value(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.50"


def _trial_has_open_ended_items(trial_info: Dict[str, Any]) -> bool:
    return any(
        str(it.get("type", "")).lower() == "open_ended" for it in (trial_info.get("items") or [])
    )


def _decision_output_schema(trial_info: Dict[str, Any]) -> Dict[str, Any]:
    items = trial_info.get("items") or []
    req_q = [f"Q{i + 1}" for i in range(len(items))]
    pq_example = {q: "option_letter" for q in req_q} if req_q else {"Q1": "a"}
    answer_line_example = ", ".join(f"{q}=a" for q in req_q) if req_q else "Q1=a"
    reasoning_example = {q: "1-2 sentences explaining this choice, citing specific state variables (e.g. trust_in_other=0.3)" for q in req_q} if req_q else {"Q1": "Because trust_in_other=0.3 and self_protection_drive=0.7, I chose to defect."}
    return {
        "per_question_choice": pq_example,
        "final_answer_line": answer_line_example,
        "per_item_reasoning": reasoning_example,
        "response_confidence": "float [0,1]",
        "notes": "optional short explanation",
    }


def _default_req_instruction(trial_info: Dict[str, Any]) -> str:
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


# ── Phase 1: Personality Compilation ────────────────────────────────
# Input: participant profile (demographics, traits, background).
# Output: PersonalityCore JSON.
# MUST NOT contain any trial/scenario/item information.

def personality_compilation_prompt(profile: Dict[str, Any]) -> str:
    schema = {
        "trait_dimensions": {k: "float in [0,1]" for k in TRAIT_DIMENSIONS},
        "disposition_rules": [
            {
                "trigger": "one of: " + ", ".join(ALLOWED_TRIGGERS),
                "tendency": "short description of behavioral tendency",
                "strength": "float in [0,1]",
            }
        ],
        "default_decision_style": "one of: " + ", ".join(DECISION_STYLES),
        "natural_language_summary": "2-3 sentence personality portrait",
    }
    return (
        "You are a personality assessment system.\n"
        "Given a participant profile, infer a stable personality core.\n\n"
        "RULES:\n"
        "- Base your inference ONLY on the profile provided below.\n"
        "- Do NOT assume any specific task, scenario, or experimental trial.\n"
        "- Produce exactly 5 trait dimensions, each a float in [0, 1].\n"
        "- Produce at least 2 and at most 5 disposition rules.\n"
        "- Each disposition rule trigger must be one of: "
        + ", ".join(ALLOWED_TRIGGERS) + ".\n"
        "- Output strict JSON only. No markdown, no commentary.\n\n"
        f"Output schema:\n{_json(schema)}\n\n"
        f"Participant profile:\n{_json(profile)}"
    )


# ── Phase 2A: State Elicitation ──────────────────────────────────────
# Input: personality_core (dict) + trial (trial_prompt string + trial_info dict).
# Output: ElicitedState JSON.
# MUST NOT contain raw profile text. Only the pre-compiled personality_core.

def state_elicitation_prompt(
    personality_core: Dict[str, Any],
    trial_prompt: str,
    trial_info: Dict[str, Any],
    trial_stimulus: Dict[str, Any] | None = None,
) -> str:
    sub_study = trial_info.get("sub_study_id", "")
    items = trial_info.get("items") or []
    scenario = trial_info.get("scenario") or trial_info.get("stimulus") or ""

    trial_summary = {
        "sub_study_id": sub_study,
        "scenario": scenario[:2000] if isinstance(scenario, str) else "",
        "n_items": len(items),
        "trial_prompt_excerpt": trial_prompt[:3000],
    }
    stimulus_payload = trial_stimulus if trial_stimulus is not None else trial_summary

    return (
        "You are a psychological state elicitation system.\n\n"
        "You are given:\n"
        "1. A pre-compiled PERSONALITY CORE (stable traits and dispositions of this person).\n"
        "2. A TRIAL STIMULUS (the current experimental scenario).\n\n"
        "Your task: determine how this personality core reacts to this specific trial.\n"
        "Produce the elicited psychological state that emerges from this interaction.\n\n"
        "RULES:\n"
        "- Use ONLY the personality core to represent who this person is.\n"
        "- Do NOT re-read or re-infer the original participant profile.\n"
        "- Do NOT produce the final answer/decision yet.\n"
        "- Focus on what cues become salient, which dispositions activate, "
        "and what internal state emerges in this specific trial.\n"
        "- This phase is about how the personality core RESPONDS to the trial, not about final answering.\n"
        "- Consider the scenario from the participant's subjective perspective: what emotions, concerns, calculations, and social considerations would arise?\n"
        "- If the scenario involves strategic interaction with another party, carefully assess: How trustworthy is the other party based on available information? What are the risks of being exploited? What are the potential gains from cooperation vs. self-interested action? How much uncertainty exists, and how does this person handle uncertainty?\n"
        "- When the scenario has a payoff structure where different actions lead to different outcomes, explicitly analyze the INCENTIVE created by that structure. Even when the other party has acted positively or cooperatively, the payoff structure may still make self-interested action more profitable. Include payoff-driven incentives in both cooperation_drivers and cooperation_inhibitors so the decision phase can weigh them.\n"
        "- Produce balanced cooperation_drivers and cooperation_inhibitors that reflect the genuine tensions in the scenario, rather than defaulting to either prosocial or self-interested framing.\n"
        "- Keep the JSON concise: use at most 2 salient_cues, at most 2 activated_dispositions, at most 2 cooperation_drivers, at most 2 cooperation_inhibitors, and keep decision_rationale to one short sentence.\n"
        "- Output strict JSON only.\n\n"
        "Output schema (strict JSON, all floats in [0,1]):\n"
        '{"salient_cues":[str],"activated_dispositions":[{"rule_trigger":str,"activation_strength":float,"note":str}],"state_variables":{"arousal":float,"conflict":float,"social_concern":float,"trust_in_other":float,"perceived_exploitation_risk":float,"self_protection_drive":float},"cooperation_drivers":[str],"cooperation_inhibitors":[str],"dominant_mode":str,"decision_rationale":str}\n\n'
        f"PERSONALITY CORE:\n{_compact_json(personality_core)}\n\n"
        f"TRIAL STIMULUS:\n{_compact_json(stimulus_payload)}"
    )


# ── Phase 2B: Decision Readout ───────────────────────────────────────
# Input: elicited_state (dict) + trial question/options.
# Output: DecisionProcess JSON with per_question_choice and final_answer_line.
# MUST NOT contain raw profile text or personality_core.

def decision_readout_prompt(
    elicited_state: Dict[str, Any],
    trial_prompt: str,
    trial_info: Dict[str, Any],
    req_instruction: str | None = None,
) -> str:
    schema = _decision_output_schema(trial_info)
    req_instruction = _default_req_instruction(trial_info) if req_instruction is None else req_instruction

    open_ended_rule = (
        "- For open-ended questions that ask for dollar amounts or quantities: put the number in final_answer_line (e.g. Q1=2.50), not option letters.\n"
        if _trial_has_open_ended_items(trial_info)
        else ""
    )

    return (
        "You are a decision output system.\n\n"
        "You are given:\n"
        "1. An ELICITED STATE that describes the person's current psychological activation.\n"
        "2. The TRIAL with its questions and options.\n\n"
        "Your task: based on the elicited state, make the final decision for each question.\n\n"
        "RULES:\n"
        "- Do NOT re-infer personality or re-analyze the scenario from scratch.\n"
        "- Use the elicited state as-is to drive the decision.\n"
        "- Use cooperation_drivers and cooperation_inhibitors (if present) explicitly when deciding.\n"
        "- Let the elicited state's internal tensions guide the response naturally — do not override them with external reasoning.\n"
        "- For per_item_reasoning: write 1-2 sentences for EACH question explaining WHY you chose that option. You MUST cite at least one specific state variable with its numeric value (e.g. 'trust_in_other=0.30 suggests low trust, so I defect').\n"
        "- Output the decision in the exact format required by the benchmark.\n"
        "- For multiple-choice: use option letters (a, b, c, ...).\n"
        f"{open_ended_rule}"
        "- final_answer_line format: Q1=a, Q2=b, ... (one line, comma+space separated).\n"
        f"{req_instruction}\n"
        "- Output strict JSON only.\n\n"
        f"Output schema:\n{_compact_json(schema)}\n\n"
        f"ELICITED STATE:\n{_json(elicited_state)}\n\n"
        f"TRIAL:\n{trial_prompt[:6000]}"
    )


def decision_readout_prompt_per_item(
    per_item_states: List[Dict[str, Any]],
    trial_prompt: str,
    trial_info: Dict[str, Any],
    req_instruction: str | None = None,
) -> str:
    schema = _decision_output_schema(trial_info)
    req_instruction = _default_req_instruction(trial_info) if req_instruction is None else req_instruction
    compact_trial = _compact_trial_overview(trial_info, trial_prompt)

    open_ended_rule = (
        "- For open-ended questions that ask for dollar amounts or quantities: put the number in final_answer_line (e.g. Q1=2.50), not option letters.\n"
        if _trial_has_open_ended_items(trial_info)
        else ""
    )

    states_text = ""
    for entry in per_item_states:
        idx = int(entry.get("item_index", 0))
        item_id = str(entry.get("item_id", f"item_{idx + 1}"))
        s = entry.get("state") or {}
        sv = s.get("state_variables") or {}
        compact = (
            f"\nQ{idx + 1} ({item_id}):\n"
            f"  variables: arousal={_fmt_state_value(sv.get('arousal', 0.5))}, "
            f"conflict={_fmt_state_value(sv.get('conflict', 0.5))}, "
            f"social_concern={_fmt_state_value(sv.get('social_concern', 0.5))}, "
            f"trust={_fmt_state_value(sv.get('trust_in_other', 0.5))}, "
            f"exploit_risk={_fmt_state_value(sv.get('perceived_exploitation_risk', 0.5))}, "
            f"self_protect={_fmt_state_value(sv.get('self_protection_drive', 0.5))}\n"
            f"  mode: {s.get('dominant_mode', 'deliberative')}\n"
            f"  drivers: {json.dumps((s.get('cooperation_drivers', []) or [])[:2], ensure_ascii=False, separators=(',', ':'))}\n"
            f"  inhibitors: {json.dumps((s.get('cooperation_inhibitors', []) or [])[:2], ensure_ascii=False, separators=(',', ':'))}\n"
            f"  rationale: {_compact_text(s.get('decision_rationale', ''), max_chars=180)}\n"
        )
        states_text += compact

    return (
        "You are a decision output system.\n\n"
        "You are given:\n"
        "1. MULTIPLE ELICITED STATES, one per question, each capturing how this person\n"
        "   reacts to that specific scenario condition.\n"
        "2. The TRIAL with all questions and options.\n\n"
        "Your task: for each question, use ONLY its corresponding elicited state\n"
        "to make the final decision.\n\n"
        "RULES:\n"
        "- You are given MULTIPLE elicited states, one for each question in this trial.\n"
        "- Each elicited state was generated independently for its specific scenario condition.\n"
        "- For EACH question, use ONLY its corresponding elicited state to drive your decision. Do NOT blend states across questions.\n"
        "- The elicited states may differ significantly from each other — this is expected and correct, because each question presents a different situation.\n"
        "- Let each state's internal tensions (cooperation_drivers vs cooperation_inhibitors, trust vs risk, etc.) guide that question's response naturally.\n"
        "- For per_item_reasoning: write 1-2 sentences for EACH question explaining WHY you chose that option. You MUST cite at least one specific state variable with its numeric value from that question's elicited state (e.g. 'trust_in_other=0.30 and self_protection_drive=0.71 lead me to defect').\n"
        "- Output the decision in the exact format required by the benchmark.\n"
        "- For multiple-choice: use option letters (a, b, c, ...).\n"
        f"{open_ended_rule}"
        "- final_answer_line format: Q1=a, Q2=b, ... (one line, comma+space separated).\n"
        f"{req_instruction}\n"
        "- Output strict JSON only.\n\n"
        f"Output schema:\n{_compact_json(schema)}\n\n"
        f"PER-ITEM ELICITED STATES:{states_text}\n"
        f"TRIAL:\n{compact_trial}"
    )


# ── Answer Repair ────────────────────────────────────────────────────
# Only fixes missing Q assignments. Does NOT re-generate personality or state.

def answer_repair_prompt(
    trial_prompt: str,
    required_q: List[str],
    merged_so_far: Dict[str, str],
    trial_info: Dict[str, Any],
) -> str:
    excerpt = (trial_prompt or "")[:6000]
    sub = trial_info.get("sub_study_id", "")
    return (
        "You complete a psychology experiment answer line.\n"
        'Output strict JSON: {"final_answer_line":"Q1=..., Q2=..., ..."}\n'
        f"The line MUST include every key: {', '.join(required_q)}.\n"
        "Use option letters (a/b) matching the trial.\n"
        "Format: one line, comma+space separated, e.g. Q1=a, Q2=b, Q3=a\n\n"
        f"sub_study_id: {sub}\n"
        f"Already known: {merged_so_far}\n\n"
        f"Trial text:\n{excerpt}\n"
    )
