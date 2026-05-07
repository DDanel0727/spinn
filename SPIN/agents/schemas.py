"""
Data schemas for SPIN.

Three core objects:
- PersonalityCore: participant-level, built once from profile, reused across all trials.
- ElicitedState: trial-level, produced when a trial stimulus activates the personality core.
- DecisionProcess: trial-level, the final decision output from the elicited state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


TRAIT_DIMENSIONS = [
    "fairness_sensitivity",
    "norm_sensitivity",
    "reciprocity_orientation",
    "uncertainty_aversion",
    "self_protection_tendency",
]

ALLOWED_TRIGGERS = [
    "fairness_violation",
    "uncertainty",
    "reciprocity_signal",
    "authority_pressure",
    "social_norm",
    "betrayal_risk",
    "competitive_opponent",
]

STATE_VARIABLES = [
    "arousal",
    "conflict",
    "social_concern",
    "trust_in_other",
    "perceived_exploitation_risk",
    "self_protection_drive",
]

DECISION_STYLES = [
    "deliberative",
    "intuitive",
    "cautious",
    "accommodating",
    "defensive",
    "guarded_reciprocal",
]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def coerce01_scalar(v: Any, default: float = 0.5) -> float:
    """
    Parse a single [0,1] scalar from LLM JSON. Models sometimes emit nested objects
    (e.g. {\"value\": 0.7}) or numeric strings instead of plain floats.
    """
    if v is None:
        return clamp01(default)
    if isinstance(v, bool):
        return clamp01(1.0 if v else 0.0)
    if isinstance(v, (int, float)):
        return clamp01(float(v))
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return clamp01(default)
        try:
            return clamp01(float(s))
        except ValueError:
            return clamp01(default)
    if isinstance(v, dict):
        for key in ("value", "score", "level", "intensity", "v", "val"):
            if key in v and v[key] is not None:
                return coerce01_scalar(v[key], default)
        if len(v) == 1:
            return coerce01_scalar(next(iter(v.values())), default)
        return clamp01(default)
    if isinstance(v, (list, tuple)) and len(v) > 0:
        return coerce01_scalar(v[0], default)
    return clamp01(default)


# ── PersonalityCore ─────────────────────────────────────────────────

@dataclass
class DispositionRule:
    trigger: str
    tendency: str
    strength: float

    def to_dict(self) -> Dict[str, Any]:
        return {"trigger": self.trigger, "tendency": self.tendency, "strength": self.strength}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DispositionRule":
        return cls(
            trigger=str(d.get("trigger", "")),
            tendency=str(d.get("tendency", "")),
            strength=coerce01_scalar(d.get("strength"), 0.5),
        )


@dataclass
class PersonalityCore:
    """Participant-level object. Built once from profile, never sees trial content."""
    participant_id: int
    trait_dimensions: Dict[str, float]
    disposition_rules: List[DispositionRule]
    default_decision_style: str
    natural_language_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "trait_dimensions": self.trait_dimensions,
            "disposition_rules": [r.to_dict() for r in self.disposition_rules],
            "default_decision_style": self.default_decision_style,
            "natural_language_summary": self.natural_language_summary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonalityCore":
        traits = d.get("trait_dimensions", {})
        clamped_traits = {k: coerce01_scalar(traits.get(k), 0.5) for k in TRAIT_DIMENSIONS}
        rules_raw = d.get("disposition_rules") or []
        rules = [DispositionRule.from_dict(r) for r in rules_raw]
        return cls(
            participant_id=int(d.get("participant_id", 0)),
            trait_dimensions=clamped_traits,
            disposition_rules=rules,
            default_decision_style=str(d.get("default_decision_style", "deliberative")),
            natural_language_summary=str(d.get("natural_language_summary", "")),
        )


# ── ElicitedState ────────────────────────────────────────────────────

@dataclass
class ElicitedState:
    """Trial-level state. Produced by stimulating PersonalityCore with a trial."""
    salient_cues: List[str]
    activated_dispositions: List[Dict[str, Any]]
    state_variables: Dict[str, float]
    cooperation_drivers: List[str]
    cooperation_inhibitors: List[str]
    dominant_mode: str
    decision_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "salient_cues": self.salient_cues,
            "activated_dispositions": self.activated_dispositions,
            "state_variables": self.state_variables,
            "cooperation_drivers": self.cooperation_drivers,
            "cooperation_inhibitors": self.cooperation_inhibitors,
            "dominant_mode": self.dominant_mode,
            "decision_rationale": self.decision_rationale,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ElicitedState":
        sv = d.get("state_variables", {}) or {}
        clamped_sv = {k: coerce01_scalar(sv.get(k), 0.5) for k in STATE_VARIABLES}
        return cls(
            salient_cues=list(d.get("salient_cues") or []),
            activated_dispositions=list(d.get("activated_dispositions") or []),
            state_variables=clamped_sv,
            cooperation_drivers=list(d.get("cooperation_drivers") or []),
            cooperation_inhibitors=list(d.get("cooperation_inhibitors") or []),
            dominant_mode=str(d.get("dominant_mode", "deliberative")),
            decision_rationale=str(d.get("decision_rationale", "")),
        )


def normalize_per_question_choice(val: Any) -> Dict[str, str]:
    """
    Models sometimes emit per_question_choice as a JSON string or one line like "Q1=a, Q2=b"
    instead of an object; merge_decision_answers / dict(...) need a real dict.
    """
    if val is None:
        return {}
    if isinstance(val, dict):
        return {str(k).strip(): str(v).strip() for k, v in val.items() if v is not None}
    if isinstance(val, list):
        out: Dict[str, str] = {}
        for item in val:
            if isinstance(item, dict):
                for k, v in item.items():
                    if v is not None:
                        out[str(k).strip()] = str(v).strip()
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out[str(item[0]).strip()] = str(item[1]).strip()
        return out
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return {}
        try:
            j = json.loads(s)
            if isinstance(j, (dict, list)):
                return normalize_per_question_choice(j)
        except Exception:
            pass
        from agents.response_finalize import parse_q_assignments

        return parse_q_assignments(s)
    return {}


# ── DecisionProcess ──────────────────────────────────────────────────

@dataclass
class DecisionProcess:
    """Trial-level decision output."""
    per_question_choice: Dict[str, str]
    final_answer_line: str
    response_confidence: float = 0.5
    notes: str = ""
    per_item_reasoning: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.per_item_reasoning is None:
            self.per_item_reasoning = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_question_choice": self.per_question_choice,
            "final_answer_line": self.final_answer_line,
            "response_confidence": self.response_confidence,
            "notes": self.notes,
            "per_item_reasoning": self.per_item_reasoning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionProcess":
        raw_reasoning = d.get("per_item_reasoning") or {}
        if isinstance(raw_reasoning, dict):
            reasoning = {str(k).strip(): str(v).strip() for k, v in raw_reasoning.items() if v}
        else:
            reasoning = {}
        return cls(
            per_question_choice=normalize_per_question_choice(d.get("per_question_choice")),
            final_answer_line=str(d.get("final_answer_line", "")),
            response_confidence=coerce01_scalar(d.get("response_confidence"), 0.5),
            notes=str(d.get("notes", "")),
            per_item_reasoning=reasoning,
        )


# ── Config ───────────────────────────────────────────────────────────

@dataclass
class SpinConfig:
    method_name: str = "spin"
    temperature: float = 0.7
    max_tokens: int = 1024
    random_seed: int = 42
    save_raw_prompts: bool = True
    save_raw_responses: bool = True
    enable_answer_repair: bool = True
    answer_repair_max_tokens: int = 512
    parse_error_retry: bool = True
    personality_max_tokens: int = 1024
    elicitation_max_tokens: int = 1024
    decision_max_tokens: int = 1024

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> "SpinConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**data)
