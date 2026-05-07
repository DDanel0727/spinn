"""
Core pipeline orchestration for SPIN.

Execution order for each participant:
  1. (Once) Phase 1 — compile personality core from profile.
  2. (Per trial) Phase 2A — elicit state from personality_core + trial.
  3. (Per trial) Phase 2B — decision readout from elicited_state + trial.
  4. (Per trial, optional) answer repair if Q labels are incomplete.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agents.decision_reader import read_decision
from agents.personality_compiler import compile_personality
from agents.schemas import SpinConfig, PersonalityCore
from agents.state_elicitor import elicit_state
from agents.utils import StructuredLLMCall
from evaluation.sanity_check import MIN_RESPONSE_TEXT_LEN_FOR_RAW_SANITY

S_T_SUB_STUDIES = {
    "pd_triad_tasks",
    "newcombs_computer_task",
    "pd_info_seeking_variation",
}

GENERIC_CONDITION_TOKENS = {
    "unknown",
    "known",
    "cooperate",
    "cooperated",
    "cooperation",
    "compete",
    "competed",
    "competition",
    "defect",
    "defected",
    "defection",
    "control",
    "treatment",
    "treated",
    "baseline",
    "high",
    "low",
    "gain",
    "loss",
    "certain",
    "uncertain",
    "safe",
    "risky",
    "risk",
}


class SpinPipeline:
    """
    Stateful pipeline bound to a single participant agent.

    The PersonalityCore is compiled lazily on the first trial and cached for all
    subsequent trials. This enforces the "build once, reuse always" contract.
    """

    def __init__(self, agent: Any, config: SpinConfig):
        self.agent = agent
        self.config = config
        self.llm = StructuredLLMCall(
            agent=agent,
            temperature=config.temperature,
            default_max_tokens=config.max_tokens,
        )
        # Participant-level cache — populated on first run(), never rebuilt.
        self._personality_core: Optional[PersonalityCore] = None
        self._personality_call_record: Optional[Dict[str, Any]] = None

    @property
    def personality_core(self) -> Optional[PersonalityCore]:
        return self._personality_core

    def _ensure_personality_core(self) -> None:
        """Phase 1: compile personality core if not already cached."""
        if self._personality_core is not None:
            return
        core, call_record = compile_personality(
            self.llm,
            participant_id=self.agent.participant_id,
            profile=self.agent.profile,
            config=self.config,
        )
        self._personality_core = core
        self._personality_call_record = call_record

    def _uses_rich_state_stimulus(self, info: Dict[str, Any]) -> bool:
        """
        Phase 2A single-state path: use rich trial stimulus (item_overview + long excerpt).

        S_T: PD / Newcomb / info-seeking materials are long and often multi-item.
        study_011: bargaining games likewise; materials are almost always single-state
        (per-item state is rare because condition labels are not split like PD triad).
        """
        study_id = str(info.get("study_id") or "").strip()
        sub_study_id = str(info.get("sub_study_id") or "").strip()
        # if study_id == "study_011":
        #     return True
        return study_id == "S_T" or sub_study_id in S_T_SUB_STUDIES

    def _extract_pd_conditions(self, info: Dict[str, Any]) -> list[str]:
        conditions: list[str] = []
        for item in info.get("items") or []:
            item_id = str(item.get("id") or "").strip()
            metadata = item.get("metadata") or {}
            raw = str(metadata.get("condition") or "").strip().lower()
            if item_id == "pd_unknown":
                cond = "unknown"
            elif item_id == "pd_known_compete":
                cond = "known_compete"
            elif item_id == "pd_known_cooperate":
                cond = "known_cooperate"
            elif raw.endswith("unknown"):
                cond = "unknown"
            elif raw in {"known_compete", "known_cooperate"}:
                cond = raw
            else:
                continue
            if cond not in conditions:
                conditions.append(cond)
        return conditions

    def _condition_label_from_item(self, item: Dict[str, Any]) -> str | None:
        metadata = item.get("metadata") or {}
        for raw in (metadata.get("condition"), item.get("condition")):
            label = str(raw or "").strip().lower()
            if label:
                return label

        item_id = str(item.get("id") or "").strip().lower()
        if not item_id:
            return None
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", item_id) if tok]
        matched = [tok for tok in tokens if tok in GENERIC_CONDITION_TOKENS]
        if matched:
            return ":".join(matched)
        return None

    def _items_have_distinct_conditions(self, items: List[Dict[str, Any]]) -> bool:
        conditions = {
            label
            for item in items
            if (label := self._condition_label_from_item(item)) is not None
        }
        return len(conditions) > 1

    def _build_single_item_stimulus(
        self,
        trial_info: Dict[str, Any],
        item: Dict[str, Any],
        item_index: int,
    ) -> Dict[str, Any]:
        items = trial_info.get("items") or []
        item_id = str(item.get("id") or f"item_{item_index}")
        item_scenario = str(item.get("scenario") or item.get("stimulus") or item.get("question") or "")
        stimulus: Dict[str, Any] = {
            "summary": {
                "study_id": trial_info.get("study_id", ""),
                "sub_study_id": trial_info.get("sub_study_id", ""),
                "context": "This is ONE specific scenario from a series. Focus ONLY on this scenario.",
                "item_index": item_index + 1,
                "total_items": len(items),
            },
            "item": {
                "id": item_id,
                "question": str(item.get("question") or ""),
                "options": list(item.get("options") or []),
                "scenario_description": item_scenario,
                "metadata": item.get("metadata") or {},
            },
        }

        if item.get("instructions"):
            stimulus["item"]["instructions"] = str(item.get("instructions"))[:2000]
        if item.get("condition"):
            stimulus["item"]["condition"] = item.get("condition")
        if trial_info.get("instructions"):
            stimulus["summary"]["general_instructions"] = self._compact_text(
                str(trial_info.get("instructions") or ""),
                max_chars=900,
                tail_chars=220,
            )
        if trial_info.get("scenario") or trial_info.get("stimulus"):
            shared = str(trial_info.get("scenario") or trial_info.get("stimulus") or "")
            if shared.strip() and shared.strip() != item_scenario.strip():
                stimulus["summary"]["shared_scenario_excerpt"] = self._compact_text(
                    shared,
                    max_chars=600,
                    tail_chars=160,
                )

        if stimulus["item"]["scenario_description"].strip() == stimulus["item"]["question"].strip():
            stimulus["item"].pop("scenario_description", None)

        return stimulus

    @staticmethod
    def _compact_text(text: str, max_chars: int = 800, tail_chars: int = 0) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        if tail_chars > 0 and max_chars > tail_chars + 5:
            head_chars = max_chars - tail_chars - 5
            return cleaned[:head_chars].rstrip() + " ... " + cleaned[-tail_chars:].lstrip()
        return cleaned[:max_chars].rstrip()

    @staticmethod
    def _aggregate_usage(call_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_cost = 0.0
        has_cost = False
        for record in call_records:
            usage = record.get("usage") or {}
            total["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            total["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            total["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            cost = usage.get("cost")
            if cost is not None:
                try:
                    total_cost += float(cost)
                    has_cost = True
                except (TypeError, ValueError):
                    pass
        if has_cost:
            total["cost"] = total_cost
        return total

    def _build_state_trial_stimulus(self, trial_prompt: str, info: Dict[str, Any]) -> Dict[str, Any]:
        items = info.get("items") or []
        summary = {
            "study_id": info.get("study_id"),
            "sub_study_id": info.get("sub_study_id"),
            "instructions_excerpt": str(info.get("instructions") or "")[:2000],
            "scenario_excerpt": str(info.get("scenario") or info.get("stimulus") or "")[:2000],
            "n_items": len(items),
            "item_ids": [str(item.get("id") or "") for item in items],
            "item_conditions": self._extract_pd_conditions(info),
            "trial_prompt_excerpt": trial_prompt[:3000],
        }
        if not self._uses_rich_state_stimulus(info):
            return summary

        item_overview = []
        for item in items[:5]:
            item_overview.append(
                {
                    "id": item.get("id"),
                    "question": str(item.get("question") or "")[:500],
                    "options": [str(opt)[:200] for opt in (item.get("options") or [])[:6]],
                    "metadata": item.get("metadata") or {},
                }
            )

        return {
            "summary": summary,
            "item_overview": item_overview,
            "full_trial_text_excerpt": trial_prompt[:7000],
        }

    def run(self, trial_prompt: str, trial_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full SPIN pipeline for one trial.

        Returns a dict with:
          - personality_core (dict)
          - elicited_state (dict or per-item state list)
          - decision_process (dict)
          - final_answer_text (str)
          - calls (dict of call records for logging)
          - status / errors
        """
        info = trial_info or {}
        out: Dict[str, Any] = {"status": "ok", "errors": [], "calls": {}}

        # ── Phase 1 (cached) ───────────────────────────────────────
        self._ensure_personality_core()
        assert self._personality_core is not None

        out["personality_core"] = self._personality_core.to_dict()
        if self._personality_call_record is not None:
            out["calls"]["personality_compilation"] = self._personality_call_record

        items = info.get("items") or []
        use_per_item_state = len(items) > 1 and self._items_have_distinct_conditions(items)
        out["trial_diagnostics"] = {
            "uses_rich_state_stimulus": self._uses_rich_state_stimulus(info),
            "pd_conditions_present": self._extract_pd_conditions(info),
            "item_count": len(items),
            "itemwise_state_elicitation": use_per_item_state,
            "condition_labels": [self._condition_label_from_item(item) for item in items],
        }

        # ── Phase 2A / 2B ──────────────────────────────────────────
        if use_per_item_state:
            per_item_states: List[Dict[str, Any]] = []
            per_item_calls: List[Dict[str, Any]] = []

            for i, item in enumerate(items):
                item_info = dict(info)
                item_info["items"] = [item]
                item_stimulus = self._build_single_item_stimulus(info, item, i)
                item_prompt = str(item.get("question") or item.get("scenario") or item.get("stimulus") or trial_prompt)
                state, state_call = elicit_state(
                    self.llm,
                    self._personality_core,
                    item_prompt,
                    item_info,
                    self.config,
                    trial_stimulus=item_stimulus,
                )
                item_id = str(item.get("id") or f"Q{i + 1}")
                per_item_states.append(
                    {
                        "item_index": i,
                        "item_id": item_id,
                        "state": state.to_dict(),
                    }
                )
                per_item_calls.append(
                    {
                        "item_index": i,
                        "item_id": item_id,
                        **state_call,
                    }
                )
                if state_call.get("fallback"):
                    out["errors"].append(f"state_elicitation_parse_failed_q{i + 1}")

            out["elicited_state"] = per_item_states
            out["calls"]["state_elicitation"] = {
                "mode": "per_item",
                "prompt": [record.get("prompt") for record in per_item_calls],
                "raw": [record.get("raw") for record in per_item_calls],
                "parsed": [entry["state"] for entry in per_item_states],
                "usage": self._aggregate_usage(per_item_calls),
                "full_api_response": {
                    "mode": "per_item",
                    "calls": [record.get("full_api_response") for record in per_item_calls],
                },
                "per_item": per_item_calls,
            }
            out["trial_diagnostics"]["per_item_state_ids"] = [entry["item_id"] for entry in per_item_states]
            out["trial_diagnostics"]["dominant_modes"] = {
                f"Q{entry['item_index'] + 1}": str((entry.get("state") or {}).get("dominant_mode") or "")
                for entry in per_item_states
            }

            dp, final_text, decision_calls = read_decision(
                self.llm,
                per_item_states,
                trial_prompt,
                info,
                self.config,
                mode="per_item",
            )
        else:
            state_trial_stimulus = self._build_state_trial_stimulus(trial_prompt, info)
            state, state_call = elicit_state(
                self.llm,
                self._personality_core,
                trial_prompt,
                info,
                self.config,
                trial_stimulus=state_trial_stimulus,
            )
            out["elicited_state"] = state.to_dict()
            out["calls"]["state_elicitation"] = state_call
            out["trial_diagnostics"]["dominant_mode"] = state.dominant_mode

            if state_call.get("fallback"):
                out["errors"].append("state_elicitation_parse_failed")

            dp, final_text, decision_calls = read_decision(
                self.llm,
                state,
                trial_prompt,
                info,
                self.config,
                mode="single",
            )

        out["decision_process"] = dp.to_dict()
        out["final_answer_text"] = final_text
        out["calls"].update(decision_calls)
        out["trial_diagnostics"]["final_answer_text"] = final_text

        ft = final_text.strip()
        if not ft:
            out["errors"].append("empty_final_answer")
            out["status"] = "error"
        elif len(ft) < MIN_RESPONSE_TEXT_LEN_FOR_RAW_SANITY:
            out["errors"].append("short_response_text_sanity")
            out["status"] = "error"

        return out
