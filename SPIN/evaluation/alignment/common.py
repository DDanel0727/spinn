"""
Shared helpers for alignment-oriented evaluation.

The guiding principle is simple: compare agent behavior to human ground
truth at the same observable granularity whenever possible.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from paths import study_root


ALIGNMENT_IMPLEMENTED_STUDIES = [
    "P_M",
    "S_T",
]


def clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def safe_median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(median(clean))


def safe_stdev(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return None
    return float(stdev(clean))


def safe_rate(numerator: float, denominator: float) -> Optional[float]:
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def average_scores(scores: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(s) for s in scores if s is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def score_absolute_error(
    agent_value: Optional[float],
    human_value: Optional[float],
    scale: float,
) -> Optional[float]:
    """
    Convert an absolute error into a [0, 1] similarity score.

    score = 1 - |agent - human| / scale

    The scale is chosen to match the natural range of the compared
    quantity, which keeps the metric easy to interpret for reviewers.
    """
    if agent_value is None or human_value is None or scale <= 0:
        return None
    return clamp01(1.0 - abs(float(agent_value) - float(human_value)) / float(scale))


def score_proportion(
    agent_value: Optional[float],
    human_value: Optional[float],
) -> Optional[float]:
    return score_absolute_error(agent_value, human_value, scale=1.0)


def score_gap(
    agent_value: Optional[float],
    human_value: Optional[float],
) -> Optional[float]:
    # A difference between two rates lives in [-1, 1], so gap error is in [0, 2].
    return score_absolute_error(agent_value, human_value, scale=2.0)


def score_correlation(
    agent_value: Optional[float],
    human_value: Optional[float],
) -> Optional[float]:
    # Correlations live in [-1, 1].
    return score_absolute_error(agent_value, human_value, scale=2.0)


def pairwise_order_score(value_map: Dict[str, Optional[float]]) -> Optional[float]:
    """
    Pairwise sign-consistency score for an ordering pattern.

    Returns the proportion of pairwise comparisons that match between
    human and agent values.
    """
    items = [(k, v) for k, v in value_map.items() if v is not None]
    if len(items) < 2:
        return None
    return 1.0


def compare_pairwise_order(
    human_values: Dict[str, Optional[float]],
    agent_values: Dict[str, Optional[float]],
) -> Optional[float]:
    keys = [k for k in human_values.keys() if k in agent_values]
    pair_scores: List[float] = []
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            hv_l = human_values.get(left)
            hv_r = human_values.get(right)
            av_l = agent_values.get(left)
            av_r = agent_values.get(right)
            if None in (hv_l, hv_r, av_l, av_r):
                continue
            human_sign = 0
            if hv_l > hv_r:
                human_sign = 1
            elif hv_l < hv_r:
                human_sign = -1

            agent_sign = 0
            if av_l > av_r:
                agent_sign = 1
            elif av_l < av_r:
                agent_sign = -1

            pair_scores.append(1.0 if human_sign == agent_sign else 0.0)
    return average_scores(pair_scores)


def make_metric(
    metric_id: str,
    label: str,
    score: Optional[float],
    description: str,
    human_value: Any = None,
    agent_value: Any = None,
    normalization: Optional[str] = None,
    sources: Optional[List[str]] = None,
    details: Optional[Dict[str, Any]] = None,
    availability: str = "available",
) -> Dict[str, Any]:
    metric = {
        "metric_id": metric_id,
        "label": label,
        "score": None if score is None else float(score),
        "availability": availability,
        "description": description,
        "human_value": human_value,
        "agent_value": agent_value,
        "normalization": normalization,
        "sources": sources or [],
    }
    if details:
        metric["details"] = details
    return metric


def make_metric_family(
    family_id: str,
    label: str,
    description: str,
    metrics: List[Dict[str, Any]],
    role: str = "primary",
) -> Dict[str, Any]:
    score = average_scores([metric.get("score") for metric in metrics])
    return {
        "family_id": family_id,
        "label": label,
        "role": role,
        "score": None if score is None else float(score),
        "description": description,
        "metrics": metrics,
    }


def normalize_individual_data(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert flat item-level outputs into the nested participant structure
    expected by the alignment evaluators.
    """
    individual_data = results.get("individual_data", []) or results.get("participant_summaries", [])
    if not individual_data:
        return []
    if "responses" in individual_data[0]:
        return individual_data

    nested: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {"responses": [], "profile": {}})
    for row in individual_data:
        participant_id = row.get("participant_id", 0)
        trial_info = row.get("trial_info", {})
        if not nested[participant_id]["profile"] and trial_info.get("profile"):
            nested[participant_id]["profile"] = trial_info["profile"]
        nested[participant_id]["responses"].append(row)

    participants = []
    for participant_id, payload in nested.items():
        payload["participant_id"] = participant_id
        participants.append(payload)
    return participants


def study_data_path(study_id: str, filename: str) -> Path:
    return study_root(study_id) / filename


def read_study_json(study_id: str, filename: str) -> Dict[str, Any]:
    with open(study_data_path(study_id, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def parse_q_assignments(response_text: str) -> Dict[str, str]:
    """
    Parse raw response text such as `Q1=foo, Q2=bar`.
    """
    parsed: Dict[str, str] = {}
    if not response_text:
        return parsed
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*[:=]\s*([^,\n\r]+)")
    for q_key, value in pattern.findall(response_text):
        parsed[q_key.strip()] = value.strip()
    return parsed


def extract_first_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def summarize_distribution(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": safe_mean(clean),
        "median": safe_median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def collapse_diagnostic(values: Sequence[Any]) -> Dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "n_unique": 0, "dominant_share": None}
    counts: Dict[str, int] = defaultdict(int)
    for value in clean:
        counts[str(value)] += 1
    dominant = max(counts.values())
    return {
        "n": len(clean),
        "n_unique": len(counts),
        "dominant_share": dominant / len(clean),
    }


def finalize_alignment_result(
    study_id: str,
    study_label: str,
    included_in_aggregate: bool,
    primary_families: List[Dict[str, Any]],
    auxiliary_families: List[Dict[str, Any]],
    diagnostics: Dict[str, Any],
    notes: Optional[List[str]] = None,
    aggregate_formula: Optional[str] = None,
    subset: Optional[List[str]] = None,
) -> Dict[str, Any]:
    mean_primary = average_scores([family.get("score") for family in primary_families])
    final_headline = None if mean_primary is None else float(mean_primary)
    return {
        "evaluation_type": "alignment",
        "study_id": study_id,
        "study_label": study_label,
        "subset": list(subset or ALIGNMENT_IMPLEMENTED_STUDIES),
        "included_in_alignment_aggregate": included_in_aggregate,
        "study_alignment_score": None if final_headline is None else float(final_headline),
        "overall_alignment_score": None if final_headline is None else float(final_headline),
        "aggregate_formula": aggregate_formula
        or "Study headline score = unweighted mean of primary metric families; diagnostics and auxiliary metrics do not enter the headline score.",
        "primary_metrics": primary_families,
        "auxiliary_metrics": auxiliary_families,
        "diagnostics": diagnostics,
        "notes": notes or [],
    }
