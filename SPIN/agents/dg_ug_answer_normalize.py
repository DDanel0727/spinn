"""
Normalize SPIN final answers for legacy dictator/ultimatum open-ended trials.

Some materials use `sub_study_id` keys shared with the old study_011 layout; the LLM may
emit Qk=a/b/c instead of dollars. This module rewrites those lines to numeric dollars so
downstream parsing matches RESPONSE_SPEC. No dependency on study evaluators or scipy.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Pie sizes ($) for sub_study_id values that use letter-coded open-ended offers.
_DG_UG_PIE_BY_SUB_STUDY: Dict[str, float] = {
    "DG-P_April_Sept": 5.0,
    "UG-P_April_Sept": 5.0,
    "DG-NP_April_Sept": 5.0,
    "UG-NP_April_Sept": 5.0,
    "DG-P_10_dollars": 10.0,
    "UG-P_10_dollars": 10.0,
}


def _infer_pie_dollars(trial_info: Optional[Dict[str, Any]]) -> float:
    if not trial_info:
        return 5.0
    sid = str(trial_info.get("sub_study_id") or "")
    if sid in _DG_UG_PIE_BY_SUB_STUDY:
        return float(_DG_UG_PIE_BY_SUB_STUDY[sid])
    if "10_dollars" in sid:
        return 10.0
    for item in trial_info.get("items") or []:
        meta = item.get("metadata") or {}
        tp = str(meta.get("total_pie", ""))
        if "$10" in tp or "10" in tp.replace("$", ""):
            return 10.0
    return 5.0


def _letter_offer_to_receiver_dollars(letter: str, pie: float) -> float:
    L = (letter or "").strip().lower()
    if L == "a":
        return 0.0
    if L == "b":
        return pie / 2.0
    if L == "c":
        return pie
    if L == "d":
        return min(pie, max(0.0, pie * 0.75))
    return pie / 2.0


def _parse_q_values_for_dg_ug(response_text: str, trial_info: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Parse Qk=numeric or Qk=letter from response text (letter → receiver $)."""
    results: Dict[str, float] = {}
    patterns = [
        re.compile(r"Q(\d+(?:\.\d+)?)\s*[:=]\s*(\d*\.?\d+)"),
        re.compile(r"[*]{1,2}Q(\d+(?:\.\d+)?)\s*[:=]\s*(\d*\.?\d+)[*]{1,2}"),
        re.compile(r"[\[\(]Q(\d+(?:\.\d+)?)\s*[:=]\s*(\d*\.?\d+)[\]\)]"),
    ]
    for pattern in patterns:
        for q_idx, val in pattern.findall(response_text):
            try:
                q_key = f"Q{q_idx}"
                if q_key not in results:
                    results[q_key] = float(val)
            except ValueError:
                continue

    letter_pat = re.compile(r"Q(\d+(?:\.\d+)?)\s*[:=]\s*([a-zA-Z])")
    pie = _infer_pie_dollars(trial_info)
    for q_idx, letter in letter_pat.findall(response_text):
        q_key = f"Q{q_idx}"
        if q_key in results:
            continue
        results[q_key] = _letter_offer_to_receiver_dollars(letter, pie)

    if not results:
        dollar_patterns = [
            r"\$(\d+\.?\d*)",
            r"\b(\d+\.\d{2})\b",
            r"\b(\d+)\b",
        ]
        amounts_found: list[float] = []
        for pattern_str in dollar_patterns:
            for match in re.finditer(pattern_str, response_text):
                try:
                    amount = float(match.group(1))
                    if 0 <= amount <= 10 and amount not in amounts_found:
                        amounts_found.append(amount)
                except (ValueError, IndexError):
                    continue
        if amounts_found:
            reasonable = [a for a in amounts_found if a >= 0.10] or amounts_found
            if reasonable:
                results["Q1"] = reasonable[0]

    return results


def normalize_if_dg_ug_substudy(
    trial_info: Optional[Dict[str, Any]], response_text: str
) -> str:
    """If `sub_study_id` is a known DG/UG pie variant, rewrite Q-lines to dollar floats."""
    if not trial_info or not response_text:
        return response_text or ""
    sid = trial_info.get("sub_study_id")
    if sid not in _DG_UG_PIE_BY_SUB_STUDY:
        return response_text
    parsed = _parse_q_values_for_dg_ug(response_text, trial_info)
    if not parsed:
        return response_text

    def _q_order(k: str) -> int:
        m = re.match(r"^Q(\d+)", str(k), re.IGNORECASE)
        return int(m.group(1)) if m else 0

    keys = sorted(parsed.keys(), key=_q_order)
    parts = [f"{k}={parsed[k]:.2f}" for k in keys]
    return ", ".join(parts)
