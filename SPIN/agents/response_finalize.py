"""
Answer merging, Q-line parsing, and benchmark-compatible formatting.

Mirrors the contract used by S_T evaluators:
  - items in trial_info → Q1..Qn required labels.
  - final_answer_line: "Q1=a, Q2=b, Q3=a"
  - map to item ids via option index.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def required_q_labels(trial_info: Optional[Dict[str, Any]]) -> List[str]:
    if not trial_info:
        return []
    items = trial_info.get("items") or []
    return [f"Q{i + 1}" for i in range(len(items))]


def parse_q_assignments(text: str) -> Dict[str, str]:
    if not text or not isinstance(text, str):
        return {}
    pat = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\s]+)", re.IGNORECASE)
    return {k.strip(): v.strip().strip('"').strip("'") for k, v in pat.findall(text)}


def normalize_choice_token(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if s.lower() in ("a", "b", "c", "d", "e", "f"):
        return s.lower()
    if len(s) == 1 and s.isalpha():
        return s.lower()
    return s


def _q_sort_key(label: str) -> int:
    m = re.match(r"^Q(\d+)$", label.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else 9999


def compose_answer_line(required: List[str], values: Dict[str, str]) -> str:
    parts = []
    for k in sorted(required, key=_q_sort_key):
        v = values.get(k)
        if v is None or str(v).strip() == "":
            continue
        parts.append(f"{k}={normalize_choice_token(v)}")
    return ", ".join(parts)


def merge_decision_answers(
    trial_info: Optional[Dict[str, Any]],
    per_question_choice: Dict[str, Any],
    final_answer_line: str,
) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Merge per_question_choice dict + final_answer_line text.
    Returns (composed_line, merged_dict, missing_q_labels).
    """
    req = required_q_labels(trial_info)
    from_line = parse_q_assignments(final_answer_line or "")

    pq_normalized: Dict[str, str] = {}
    for raw_k, raw_v in (per_question_choice or {}).items():
        k = str(raw_k).strip()
        m = re.match(r"^Q?(\d+)$", k, re.IGNORECASE)
        if m:
            k = f"Q{m.group(1)}"
        if raw_v is not None:
            pq_normalized[k] = str(raw_v).strip()

    merged: Dict[str, str] = {}
    for k in req:
        if k in from_line and from_line[k]:
            merged[k] = from_line[k]
        elif k in pq_normalized and pq_normalized[k]:
            merged[k] = pq_normalized[k]

    missing = [k for k in req if k not in merged or not str(merged.get(k, "")).strip()]
    if not req:
        return final_answer_line or "", {}, []
    if not missing:
        return compose_answer_line(req, merged), merged, []
    partial = compose_answer_line([k for k in req if k in merged], merged)
    return partial or final_answer_line or "", merged, missing


def map_answer_text_to_item_values(
    trial_info: Optional[Dict[str, Any]],
    answer_text: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse Qk assignments and resolve to item-id → option-text mapping.
    Returns (q_assignments, parsed_response).
    """
    q_assignments = {k: normalize_choice_token(v) for k, v in parse_q_assignments(answer_text).items()}
    if not trial_info:
        return q_assignments, {}

    items = trial_info.get("items") or []
    parsed_response: Dict[str, str] = {}
    for idx, item in enumerate(items):
        q_label = f"Q{idx + 1}"
        raw_value = q_assignments.get(q_label, "")
        item_id = str(item.get("id") or "").strip()
        if not raw_value or not item_id:
            continue
        options = item.get("options") or []
        normalized = normalize_choice_token(raw_value)
        resolved = raw_value
        if len(normalized) == 1 and normalized.isalpha():
            option_idx = ord(normalized.upper()) - ord("A")
            if 0 <= option_idx < len(options):
                resolved = str(options[option_idx])
        parsed_response[item_id] = resolved
    return q_assignments, parsed_response
