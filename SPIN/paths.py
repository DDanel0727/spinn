"""
Resolve benchmark asset directories when data lives outside the SPIN repo.

Default: ``<SPIN repo parent>/data`` (e.g. ``material/data`` next to ``material/SPIN``).

Override: set environment variable ``SPIN_DATA_DIR`` to the directory that contains
``registry.json``, ``studies/``, and ``schemas/``.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """SPIN project root (directory containing ``agents/``, ``evaluation/``, ``llm/``)."""
    return Path(__file__).resolve().parent


def spin_data_dir() -> Path:
    env = (os.environ.get("SPIN_DATA_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root().parent / "data").resolve()


def studies_root() -> Path:
    return spin_data_dir() / "studies"


def study_root(study_id: str) -> Path:
    return studies_root() / study_id
