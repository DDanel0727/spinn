"""
SpinParticipantAgent — participant agent wrapper for SPIN.

Inherits from LLMParticipantAgent and overrides complete_trial() to route
through the SPIN pipeline (personality compilation → state elicitation →
decision readout).

Key design:
- PersonalityCore is compiled once (on the first trial) and cached on the
  pipeline object for all subsequent trials.
- Each trial produces a fresh ElicitedState and DecisionProcess.
- Output format is compatible with the existing benchmark evaluator chain
  (Q1=a, Q2=b, ... answer lines + parsed_response dict).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from agents.logger import SpinLogger
from agents.pipeline import SpinPipeline
from agents.response_finalize import map_answer_text_to_item_values
from agents.dg_ug_answer_normalize import normalize_if_dg_ug_substudy
from agents.schemas import SpinConfig
from agents.utils import now_iso
from llm.llm_participant_agent import LLMParticipantAgent


class SpinParticipantAgent(LLMParticipantAgent):
    def __init__(
        self,
        *args: Any,
        spin_config: Optional[Dict[str, Any]] = None,
        spin_output_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.spin_config = SpinConfig.from_kwargs(spin_config or {})
        output_dir = Path(spin_output_dir) if spin_output_dir else Path("results/spin/default_run")
        self.spin_logger = SpinLogger(output_dir)
        self.spin_pipeline = SpinPipeline(self, self.spin_config)

    def _usage_meta(self, trial_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "study_id": trial_info.get("study_id"),
            "sub_study_id": trial_info.get("sub_study_id"),
            "trial_id": trial_info.get("trial_id", trial_info.get("trial_number")),
            "scenario_id": trial_info.get("scenario_id"),
        }

    def complete_trial(
        self,
        trial_prompt: str,
        trial_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        info = trial_info or {}

        if not self.use_real_llm:
            return super().complete_trial(trial_prompt, trial_info)

        run = self.spin_pipeline.run(trial_prompt, info)

        final_text = run.get("final_answer_text", "") or ""
        final_text = normalize_if_dg_ug_substudy(info, final_text)
        trial_diagnostics = run.get("trial_diagnostics") or {}
        _, parsed_response = map_answer_text_to_item_values(info, final_text)
        choice = self._parse_response(final_text, info)

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        full_api_by_call: Dict[str, Any] = {}
        for call_name, call_data in (run.get("calls", {}) or {}).items():
            usage = call_data.get("usage", {}) or {}
            total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            total_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            fa = call_data.get("full_api_response")
            if isinstance(fa, dict) and fa:
                full_api_by_call[call_name] = fa
            self.spin_logger.log_token_usage(
                phase=call_name,
                usage=usage,
                model=self.model,
            )

        pc = run.get("personality_core")
        if pc and "personality_compilation" in run.get("calls", {}):
            self.spin_logger.log_personality_core({
                "ts": now_iso(),
                **self._usage_meta(info),
                "personality_core": pc,
            })

        es = run.get("elicited_state")
        if es:
            self.spin_logger.log_elicited_state({
                "ts": now_iso(),
                **self._usage_meta(info),
                "trial_diagnostics": trial_diagnostics,
                "elicited_state": es,
            })

        traj = {
            "ts": now_iso(),
            **self._usage_meta(info),
            "status": run.get("status"),
            "errors": run.get("errors", []),
            "trial_diagnostics": trial_diagnostics,
            "personality_core": pc,
            "elicited_state": es,
            "decision_process": run.get("decision_process"),
            "final_answer_text": final_text,
            "token_usage": total_usage,
        }
        for cname in ("personality_compilation", "state_elicitation", "decision_readout", "answer_repair"):
            cd = (run.get("calls", {}) or {}).get(cname, {})
            if self.spin_config.save_raw_prompts:
                traj[f"{cname}_prompt"] = cd.get("prompt")
            if self.spin_config.save_raw_responses:
                traj[f"{cname}_raw"] = cd.get("raw")
        self.spin_logger.log_trajectory(traj)

        self.spin_logger.log_prediction({
            "ts": now_iso(),
            **self._usage_meta(info),
            "response_text": final_text,
            "parsed_response": parsed_response,
            "parsed_choice": choice,
        })

        response_data = {
            "participant_id": self.participant_id,
            "trial_number": info.get("trial_number") if info else len(self.trial_responses) + 1,
            "response": choice,
            "response_text": final_text,
            "raw_response_text": final_text,
            "parsed_response": parsed_response,
            "usage": total_usage,
            "formatter_used": False,
            "full_api_response": full_api_by_call,
            "correct_answer": info.get("correct_answer") if info else None,
            "is_correct": (
                choice == info.get("correct_answer")
                if info and info.get("correct_answer")
                else None
            ),
            "trial_info": info,
        }
        self.trial_responses.append(response_data)
        return response_data
