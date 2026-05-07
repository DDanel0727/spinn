"""Testing run, evaluation, stats, alignment, and benchmark data (SPIN).

Entry: ``python -m evaluation.cli`` (see ``cli.py``).
"""

from typing import Any

__all__ = ["GenerationPipeline"]


def __getattr__(name: str) -> Any:
    if name == "GenerationPipeline":
        from evaluation.pipeline_core import GenerationPipeline

        return GenerationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
