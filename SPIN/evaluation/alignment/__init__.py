"""
Alignment-oriented evaluation for HumanStudy-Bench.

Implemented subset for this repository layout (see ``material/data`` registry):

- **P_M**: pluralistic-ignorance level/gap structure
- **S_T**: option proportions / condition effects
"""

from .runner import (
    DEFAULT_ALIGNMENT_SUBSET,
    compute_alignment_subset_aggregate,
    evaluate_alignment_study,
)

__all__ = [
    "DEFAULT_ALIGNMENT_SUBSET",
    "evaluate_alignment_study",
    "compute_alignment_subset_aggregate",
]
