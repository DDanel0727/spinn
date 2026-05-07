"""
Dispatcher for alignment-oriented evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import ALIGNMENT_IMPLEMENTED_STUDIES, average_scores
from .P_M_alignment import evaluate as evaluate_pm_alignment
from .S_T_alignment import evaluate as evaluate_st_alignment


DEFAULT_ALIGNMENT_SUBSET = list(ALIGNMENT_IMPLEMENTED_STUDIES)


def evaluate_alignment_study(study_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    if study_id == "P_M":
        return evaluate_pm_alignment(results)
    if study_id == "S_T":
        return evaluate_st_alignment(results)
    return {
        "evaluation_type": "alignment",
        "study_id": study_id,
        "study_alignment_score": None,
        "overall_alignment_score": None,
        "included_in_alignment_aggregate": False,
        "subset": DEFAULT_ALIGNMENT_SUBSET,
        "primary_metrics": [],
        "auxiliary_metrics": [],
        "diagnostics": {},
        "notes": [
            "No alignment evaluator is currently implemented for this study.",
            "The current alignment workflow is implemented for P_M and S_T only.",
        ],
    }


def compute_alignment_subset_aggregate(
    study_results: Dict[str, Dict[str, Any]],
    subset: Optional[List[str]] = None,
) -> Dict[str, Any]:
    subset = subset or list(DEFAULT_ALIGNMENT_SUBSET)
    scores = []
    for sid in subset:
        block = (study_results.get(sid) or {}).get("alignment_evaluation") or {}
        s = block.get("study_alignment_score")
        if s is not None:
            scores.append(float(s))
    overall = average_scores(scores) if scores else None
    return {
        "subset": subset,
        "overall_alignment_score": overall,
        "per_study": {sid: (study_results.get(sid) or {}).get("alignment_evaluation", {}).get("study_alignment_score") for sid in subset},
    }
