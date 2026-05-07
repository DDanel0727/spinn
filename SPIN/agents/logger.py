"""
Thread-safe logger for SPIN run artifacts.

Output directory (legacy): results/spin/<run_name>/<study_id>/<config_folder>/
With RESULTS_METHOD_NAME shell wrappers: same folder as testing benchmark outputs (flat under results/<method>/<run_name>/).
Files written:
  - trajectories.jsonl
  - predictions.jsonl
  - token_usage.jsonl
  - token_summary.json
  - personality_cores.jsonl
  - elicited_states.jsonl
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from evaluation.io_utils import atomic_write_json


_LOCKS: Dict[str, threading.Lock] = {}


def _get_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    if key not in _LOCKS:
        _LOCKS[key] = threading.Lock()
    return _LOCKS[key]


class SpinLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trajectory_path = self.output_dir / "trajectories.jsonl"
        self.prediction_path = self.output_dir / "predictions.jsonl"
        self.token_usage_path = self.output_dir / "token_usage.jsonl"
        self.token_summary_path = self.output_dir / "token_summary.json"
        self.personality_core_path = self.output_dir / "personality_cores.jsonl"
        self.elicited_state_path = self.output_dir / "elicited_states.jsonl"

        self._summary = {
            "total_calls": 0,
            "by_phase": {},
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
        }
        self._summary_lock = threading.Lock()

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        lock = _get_lock(path)
        line = json.dumps(record, ensure_ascii=False)
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_trajectory(self, record: Dict[str, Any]) -> None:
        self._append_jsonl(self.trajectory_path, record)

    def log_prediction(self, record: Dict[str, Any]) -> None:
        self._append_jsonl(self.prediction_path, record)

    def log_personality_core(self, record: Dict[str, Any]) -> None:
        self._append_jsonl(self.personality_core_path, record)

    def log_elicited_state(self, record: Dict[str, Any]) -> None:
        self._append_jsonl(self.elicited_state_path, record)

    def log_token_usage(self, phase: str, usage: Dict[str, Any], model: str = "") -> None:
        record = {"phase": phase, "model": model, "usage": usage}
        self._append_jsonl(self.token_usage_path, record)
        with self._summary_lock:
            self._summary["total_calls"] += 1
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            tt = int(usage.get("total_tokens", 0) or 0)
            self._summary["total_prompt_tokens"] += pt
            self._summary["total_completion_tokens"] += ct
            self._summary["total_tokens"] += tt
            by_phase = self._summary["by_phase"].setdefault(
                phase, {"count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            by_phase["count"] += 1
            by_phase["prompt_tokens"] += pt
            by_phase["completion_tokens"] += ct
            by_phase["total_tokens"] += tt
            atomic_write_json(self.token_summary_path, self._summary, indent=2, ensure_ascii=False)
