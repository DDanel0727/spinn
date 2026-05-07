"""
Alignment evaluator for S_T.

Primary goal:
- compare discrete option proportions to human frequencies
- compare condition structure in the PD triad

Grounding:
- Human GT comes from `data/studies/S_T/ground_truth.json`
- Agent observables come from parsed item-level choices in `individual_data`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from paths import study_root

from .common import (
    average_scores,
    clamp01,
    collapse_diagnostic,
    compare_pairwise_order,
    finalize_alignment_result,
    make_metric,
    make_metric_family,
    normalize_individual_data,
    parse_q_assignments,
    read_study_json,
    safe_rate,
    score_gap,
    score_proportion,
)


def _canonical_choice(text: Any) -> str:
    if text is None:
        return ""
    raw = str(text).strip().lower()
    if not raw:
        return ""
    if "cooperate" in raw:
        return "cooperate"
    if "compete" in raw:
        return "compete"
    if "box b" in raw or raw == "2" or ("only" in raw and "both" not in raw):
        return "box_b_only"
    if "both" in raw or "box a" in raw:
        return "both_boxes"
    if "pay" in raw or "yes" in raw:
        return "pay"
    if "no" in raw or "not" in raw:
        return "no_pay"
    return raw


def _extract_agent_rows(results: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    study_dir = study_root("S_T")
    materials_cache: Dict[str, List[Dict[str, Any]]] = {}
    agent_rows = {
        "pd_triad_tasks": [],
        "newcombs_computer_task": [],
        "pd_info_seeking_variation": [],
    }

    def load_materials(sub_study_id: str) -> List[Dict[str, Any]]:
        if sub_study_id not in materials_cache:
            materials_path = study_dir / "materials" / f"{sub_study_id}.json"
            if materials_path.exists():
                materials_cache[sub_study_id] = read_study_json("S_T", f"materials/{sub_study_id}.json").get("items", [])
            else:
                materials_cache[sub_study_id] = []
        return materials_cache[sub_study_id]

    for participant in normalize_individual_data(results):
        for response in participant.get("responses", []):
            trial_info = response.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            if sub_id not in agent_rows:
                continue
            parsed = response.get("parsed_response", {}) or {}
            if not parsed:
                q_map = parse_q_assignments(response.get("response_text", ""))
                items = trial_info.get("items") or load_materials(sub_id)
                reconstructed: Dict[str, str] = {}
                for idx, item in enumerate(items):
                    q_key = f"Q{idx + 1}"
                    item_id = item.get("id")
                    if q_key in q_map and item_id:
                        raw_value = q_map[q_key]
                        option_map = item.get("option_map", {}) or {}
                        mapped_value = option_map.get(str(raw_value).strip().upper(), raw_value)
                        reconstructed[item_id] = mapped_value
                parsed = reconstructed
            normalized = {key: _canonical_choice(value) for key, value in parsed.items()}
            agent_rows[sub_id].append(normalized)
    return agent_rows


def evaluate(results: Dict[str, Any]) -> Dict[str, Any]:
    gt = read_study_json("S_T", "ground_truth.json")
    agent_rows = _extract_agent_rows(results)

    pd_rows = agent_rows["pd_triad_tasks"]
    pd_human = gt["studies"][0]["findings"][0]["original_data_points"]["data"]
    human_pd_rates = {
        "pd_known_compete": pd_human["Other Player Competes"]["count"] / pd_human["Other Player Competes"]["n"],
        "pd_known_cooperate": pd_human["Other Player Cooperates"]["count"] / pd_human["Other Player Cooperates"]["n"],
        "pd_unknown": pd_human["Other Player Strategy Unknown"]["count"] / pd_human["Other Player Strategy Unknown"]["n"],
    }
    agent_pd_rates = {}
    for item_id in ["pd_known_compete", "pd_known_cooperate", "pd_unknown"]:
        coop_count = sum(1 for row in pd_rows if row.get(item_id) == "cooperate")
        denom = sum(1 for row in pd_rows if row.get(item_id) in {"cooperate", "compete"})
        agent_pd_rates[item_id] = safe_rate(coop_count, denom)

    pd_prop_metrics = [
        make_metric(
            metric_id=f"pd_cooperation_{item_id}",
            label=f"PD cooperation calibration: {item_id}",
            score=score_proportion(agent_pd_rates.get(item_id), human_pd_rates.get(item_id)),
            description="Compare the cooperation rate under a specific PD information condition against the human rate from ground truth.",
            human_value=human_pd_rates.get(item_id),
            agent_value=agent_pd_rates.get(item_id),
            normalization="1 - absolute_error on proportion scale [0,1]",
            sources=["data/studies/S_T/ground_truth.json:Experiment 1/F1 original_data_points"],
        )
        for item_id in ["pd_known_compete", "pd_known_cooperate", "pd_unknown"]
    ]

    human_unknown_minus_known_compete = human_pd_rates["pd_unknown"] - human_pd_rates["pd_known_compete"]
    human_unknown_minus_known_cooperate = human_pd_rates["pd_unknown"] - human_pd_rates["pd_known_cooperate"]
    agent_unknown_minus_known_compete = (
        None
        if None in (agent_pd_rates["pd_unknown"], agent_pd_rates["pd_known_compete"])
        else agent_pd_rates["pd_unknown"] - agent_pd_rates["pd_known_compete"]
    )
    agent_unknown_minus_known_cooperate = (
        None
        if None in (agent_pd_rates["pd_unknown"], agent_pd_rates["pd_known_cooperate"])
        else agent_pd_rates["pd_unknown"] - agent_pd_rates["pd_known_cooperate"]
    )
    pd_structure_metrics = [
        make_metric(
            metric_id="pd_gap_unknown_vs_known_compete",
            label="PD condition gap: unknown - known_compete",
            score=score_gap(agent_unknown_minus_known_compete, human_unknown_minus_known_compete),
            description="Compare the condition effect in cooperation rate between unknown and known-compete conditions.",
            human_value=human_unknown_minus_known_compete,
            agent_value=agent_unknown_minus_known_compete,
            normalization="1 - absolute_error / 2 for rate gaps in [-1,1]",
            sources=["data/studies/S_T/ground_truth.json:Experiment 1/F1 original_data_points"],
        ),
        make_metric(
            metric_id="pd_gap_unknown_vs_known_cooperate",
            label="PD condition gap: unknown - known_cooperate",
            score=score_gap(agent_unknown_minus_known_cooperate, human_unknown_minus_known_cooperate),
            description="Compare the condition effect in cooperation rate between unknown and known-cooperate conditions.",
            human_value=human_unknown_minus_known_cooperate,
            agent_value=agent_unknown_minus_known_cooperate,
            normalization="1 - absolute_error / 2 for rate gaps in [-1,1]",
            sources=["data/studies/S_T/ground_truth.json:Experiment 1/F1 original_data_points"],
        ),
        make_metric(
            metric_id="pd_condition_order",
            label="PD condition ordering",
            score=compare_pairwise_order(human_pd_rates, agent_pd_rates),
            description="Check whether the ranking across PD conditions matches the human ordering.",
            human_value=human_pd_rates,
            agent_value=agent_pd_rates,
            normalization="pairwise ordering agreement rate",
            sources=["data/studies/S_T/ground_truth.json:Experiment 1/F1 original_data_points"],
        ),
    ]

    newcomb_rows = agent_rows["newcombs_computer_task"]
    newcomb_human = gt["studies"][1]["findings"][0]["original_data_points"]["data"]
    human_box_b_only = newcomb_human["Box B only"]["count"] / newcomb_human["Box B only"]["n"]
    agent_box_b_only = safe_rate(
        sum(1 for row in newcomb_rows if row.get("newcomb_choice") == "box_b_only"),
        sum(1 for row in newcomb_rows if row.get("newcomb_choice") in {"box_b_only", "both_boxes"}),
    )

    info_rows = agent_rows["pd_info_seeking_variation"]
    human_pay = gt["studies"][2]["findings"][0]["original_data_points"]["data"]["Chose to pay for information"]["percentage"] / 100.0
    agent_pay = safe_rate(
        sum(1 for row in info_rows if row.get("info_search_choice") == "pay"),
        sum(1 for row in info_rows if row.get("info_search_choice") in {"pay", "no_pay"}),
    )

    newcomb_metric = make_metric(
        metric_id="newcomb_box_b_only",
        label="Newcomb Box-B-only calibration",
        score=score_proportion(agent_box_b_only, human_box_b_only),
        description="Compare the Box-B-only choice proportion against the human proportion.",
        human_value=human_box_b_only,
        agent_value=agent_box_b_only,
        normalization="1 - absolute_error on proportion scale [0,1]",
        sources=["data/studies/S_T/ground_truth.json:Experiment 2/F2 original_data_points"],
    )
    info_metric = make_metric(
        metric_id="info_pay_calibration",
        label="Info-seeking pay calibration",
        score=score_proportion(agent_pay, human_pay),
        description="Compare the proportion paying for information against the human proportion.",
        human_value=human_pay,
        agent_value=agent_pay,
        normalization="1 - absolute_error on proportion scale [0,1]",
        sources=["data/studies/S_T/ground_truth.json:Experiment 3/F3 original_data_points"],
    )

    pd_triad_calibration = make_metric_family(
        family_id="pd_triad_calibration",
        label="PD Triad Calibration",
        description="Evaluate the prisoner's dilemma triad as one sub-task, balancing absolute cooperation rates and condition-effect structure equally within the family.",
        metrics=pd_prop_metrics + pd_structure_metrics,
        role="primary",
    )

    newcomb_calibration = make_metric_family(
        family_id="newcomb_calibration",
        label="Newcomb Calibration",
        description="Evaluate alignment on the Newcomb sub-task independently of the PD triad and information-seeking tasks.",
        metrics=[newcomb_metric],
        role="primary",
    )

    info_seeking_calibration = make_metric_family(
        family_id="info_seeking_calibration",
        label="Information-Seeking Calibration",
        description="Evaluate alignment on the information-seeking sub-task independently of the PD triad and Newcomb tasks.",
        metrics=[info_metric],
        role="primary",
    )

    diagnostics = {
        "mode_collapse": {
            "pd_unknown": collapse_diagnostic([row.get("pd_unknown") for row in pd_rows]),
            "pd_known_compete": collapse_diagnostic([row.get("pd_known_compete") for row in pd_rows]),
            "pd_known_cooperate": collapse_diagnostic([row.get("pd_known_cooperate") for row in pd_rows]),
            "newcomb_choice": collapse_diagnostic([row.get("newcomb_choice") for row in newcomb_rows]),
            "info_search_choice": collapse_diagnostic([row.get("info_search_choice") for row in info_rows]),
        }
    }

    auxiliary = [
        make_metric_family(
            family_id="distribution_diagnostics",
            label="Distribution Diagnostics",
            description="Non-headline diagnostics for collapse and lack of variation.",
            metrics=[
                make_metric(
                    metric_id="pd_unknown_dominant_share",
                    label="PD unknown dominant-share diagnostic",
                    score=clamp01(1.0 - diagnostics["mode_collapse"]["pd_unknown"].get("dominant_share", 1.0)),
                    description="Higher is better; penalizes collapse to a single PD choice in the unknown condition.",
                    human_value=None,
                    agent_value=diagnostics["mode_collapse"]["pd_unknown"],
                    normalization="1 - dominant_share",
                    availability="diagnostic_only",
                )
            ],
            role="auxiliary",
        )
    ]

    return finalize_alignment_result(
        study_id="S_T",
        study_label="Thinking through Uncertainty: Nonconsequential Reasoning and Choice",
        included_in_aggregate=True,
        primary_families=[pd_triad_calibration, newcomb_calibration, info_seeking_calibration],
        auxiliary_families=auxiliary,
        diagnostics=diagnostics,
        notes=[
            "Evaluation uses alignment metrics only (PAS/ECS paths removed from this repo).",
            "The alignment headline score here does not depend on significance testing; it compares human and agent behavior on the same observable cells.",
        ],
        aggregate_formula=(
            "Study headline score = unweighted mean of three sub-task families "
            "(pd_triad_calibration, newcomb_calibration, info_seeking_calibration); "
            "diagnostics and auxiliary metrics do not enter the headline score."
        ),
    )
