"""
Alignment evaluator for P_M.

Primary goal:
- calibrate mean comfort judgments in Studies 1 and 2
- compare pluralistic-ignorance gaps, ordering, and downstream action patterns
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import (
    collapse_diagnostic,
    compare_pairwise_order,
    extract_first_number,
    finalize_alignment_result,
    make_metric,
    make_metric_family,
    normalize_individual_data,
    parse_q_assignments,
    read_study_json,
    safe_mean,
    safe_stdev,
    score_absolute_error,
)


def _normalized_gender(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"female", "f"}:
        return "female"
    if raw in {"male", "m"}:
        return "male"
    return raw


def _extract_numeric_entry(response: Dict[str, Any]) -> Dict[str, Optional[float]]:
    trial_info = response.get("trial_info", {})
    items = trial_info.get("items", []) or []
    parsed = response.get("parsed_response", {}) or {}
    q_map = parse_q_assignments(response.get("response_text", ""))

    entry: Dict[str, Optional[float]] = {}
    for item_id, value in parsed.items():
        entry[item_id] = extract_first_number(value)

    for idx, item in enumerate(items, start=1):
        item_id = item.get("id")
        if not item_id or item_id in entry:
            continue
        entry[item_id] = extract_first_number(q_map.get(f"Q{idx}"))
    return entry


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    return safe_mean(row.get(key) for row in rows)


def _group_mean(rows: List[Dict[str, Any]], key: str, target_group: str) -> Optional[float]:
    return safe_mean(row.get(key) for row in rows if row.get("group") == target_group)


def _extract_rows(results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {
        "study_1_comfort_estimation": [],
        "study_2_order_and_friend_comparison": [],
        "study_4_keg_ban_alienation": [],
    }

    for participant in normalize_individual_data(results):
        participant_gender = _normalized_gender(participant.get("profile", {}).get("gender"))
        for response in participant.get("responses", []):
            trial_info = response.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            if sub_id not in grouped:
                continue

            entry = _extract_numeric_entry(response)
            entry["gender"] = participant_gender or _normalized_gender(
                trial_info.get("profile", {}).get("gender")
            )

            if sub_id == "study_2_order_and_friend_comparison":
                entry["order"] = trial_info.get("order_condition")
            elif sub_id == "study_4_keg_ban_alienation":
                q2 = entry.get("q2")
                if q2 in {1.0, 2.0}:
                    entry["group"] = "others_more_negative"
                elif q2 == 3.0:
                    entry["group"] = "others_the_same"
                elif q2 is not None:
                    entry["group"] = "others_more_positive"

            grouped[sub_id].append(entry)

    return grouped


def evaluate(results: Dict[str, Any]) -> Dict[str, Any]:
    gt = read_study_json("P_M", "ground_truth.json")
    rows = _extract_rows(results)

    study1_rows = rows["study_1_comfort_estimation"]
    study2_rows = rows["study_2_order_and_friend_comparison"]
    study4_rows_all = rows["study_4_keg_ban_alienation"]
    study4_rows_female = [row for row in study4_rows_all if row.get("gender") == "female"]
    study4_rows = study4_rows_female if study4_rows_female else study4_rows_all

    human_study1 = gt["studies"][0]["findings"][0]["original_data_points"]["data"]
    human_study2 = gt["studies"][1]["findings"][0]["original_data_points"]["data"]
    human_study4_f1 = gt["studies"][2]["findings"][0]["original_data_points"]["data"]
    human_study4_f2 = gt["studies"][2]["findings"][1]["original_data_points"]["data"]

    study2_agent = {
        "self_first": {
            "self": _mean([row for row in study2_rows if row.get("order") == "self_first"], "q1"),
            "average": _mean([row for row in study2_rows if row.get("order") == "self_first"], "q2"),
            "friend": _mean([row for row in study2_rows if row.get("order") == "self_first"], "q3"),
        },
        "average_first": {
            "self": _mean([row for row in study2_rows if row.get("order") == "average_first"], "q1"),
            "average": _mean([row for row in study2_rows if row.get("order") == "average_first"], "q2"),
            "friend": _mean([row for row in study2_rows if row.get("order") == "average_first"], "q3"),
        },
    }
    study2_human = {
        "self_first": {
            "self": human_study2["Self_First_Condition_Self"]["mean"],
            "friend": human_study2["Self_First_Condition_Friend"]["mean"],
            "average": human_study2["Self_First_Condition_Average"]["mean"],
        },
        "average_first": {
            "self": human_study2["Other_First_Condition_Self"]["mean"],
            "friend": human_study2["Other_First_Condition_Friend"]["mean"],
            "average": human_study2["Other_First_Condition_Average"]["mean"],
        },
    }

    level_metrics = [
        make_metric(
            metric_id="study1_self_mean",
            label="Study 1 self-comfort mean",
            score=score_absolute_error(_mean(study1_rows, "q1"), human_study1["Total_Self"]["mean"], scale=10.0),
            description="Compare the average self-comfort rating on the 11-point scale.",
            human_value=human_study1["Total_Self"]["mean"],
            agent_value=_mean(study1_rows, "q1"),
            normalization="1 - absolute_error / 10 on the 11-point scale",
            sources=["data/studies/P_M/ground_truth.json:Study 1 / Total_Self"],
        ),
        make_metric(
            metric_id="study1_average_mean",
            label="Study 1 average-student mean",
            score=score_absolute_error(_mean(study1_rows, "q2"), human_study1["Total_Average_Estimate"]["mean"], scale=10.0),
            description="Compare the average estimated comfort of the average student.",
            human_value=human_study1["Total_Average_Estimate"]["mean"],
            agent_value=_mean(study1_rows, "q2"),
            normalization="1 - absolute_error / 10 on the 11-point scale",
            sources=["data/studies/P_M/ground_truth.json:Study 1 / Total_Average_Estimate"],
        ),
    ]
    for order_key, label in [("self_first", "Self-first"), ("average_first", "Average-first")]:
        for target in ["self", "friend", "average"]:
            level_metrics.append(
                make_metric(
                    metric_id=f"study2_{order_key}_{target}",
                    label=f"Study 2 {label} {target} mean",
                    score=score_absolute_error(
                        study2_agent[order_key][target],
                        study2_human[order_key][target],
                        scale=10.0,
                    ),
                    description="Compare the mean comfort rating for this target under this question-order condition.",
                    human_value=study2_human[order_key][target],
                    agent_value=study2_agent[order_key][target],
                    normalization="1 - absolute_error / 10 on the 11-point scale",
                    sources=["data/studies/P_M/ground_truth.json:Study 2 / original_data_points"],
                )
            )

    level_family = make_metric_family(
        family_id="mean_level_calibration",
        label="Mean-Level Calibration",
        description="Compare mean comfort judgments in Study 1 and across both order conditions in Study 2.",
        metrics=level_metrics,
        role="primary",
    )

    structure_metrics = [
        make_metric(
            metric_id="study1_pluralistic_gap",
            label="Study 1 pluralistic-ignorance gap",
            score=score_absolute_error(
                None
                if None in (_mean(study1_rows, "q1"), _mean(study1_rows, "q2"))
                else _mean(study1_rows, "q2") - _mean(study1_rows, "q1"),
                human_study1["Total_Average_Estimate"]["mean"] - human_study1["Total_Self"]["mean"],
                scale=10.0,
            ),
            description="Compare the average-student minus self gap in Study 1.",
            human_value=human_study1["Total_Average_Estimate"]["mean"] - human_study1["Total_Self"]["mean"],
            agent_value=None
            if None in (_mean(study1_rows, "q1"), _mean(study1_rows, "q2"))
            else _mean(study1_rows, "q2") - _mean(study1_rows, "q1"),
            normalization="1 - absolute_error / 10 on the 11-point scale",
            sources=["data/studies/P_M/ground_truth.json:Study 1 / original_data_points"],
        )
    ]
    for order_key, label in [("self_first", "Self-first"), ("average_first", "Average-first")]:
        human_map = study2_human[order_key]
        agent_map = study2_agent[order_key]
        structure_metrics.extend(
            [
                make_metric(
                    metric_id=f"study2_{order_key}_self_friend_gap",
                    label=f"Study 2 {label} friend-self gap",
                    score=score_absolute_error(
                        None
                        if None in (agent_map["self"], agent_map["friend"])
                        else agent_map["friend"] - agent_map["self"],
                        human_map["friend"] - human_map["self"],
                        scale=10.0,
                    ),
                    description="Compare the friend-minus-self gap for this question-order condition.",
                    human_value=human_map["friend"] - human_map["self"],
                    agent_value=None
                    if None in (agent_map["self"], agent_map["friend"])
                    else agent_map["friend"] - agent_map["self"],
                    normalization="1 - absolute_error / 10 on the 11-point scale",
                    sources=["data/studies/P_M/ground_truth.json:Study 2 / original_data_points"],
                ),
                make_metric(
                    metric_id=f"study2_{order_key}_friend_average_gap",
                    label=f"Study 2 {label} average-friend gap",
                    score=score_absolute_error(
                        None
                        if None in (agent_map["friend"], agent_map["average"])
                        else agent_map["average"] - agent_map["friend"],
                        human_map["average"] - human_map["friend"],
                        scale=10.0,
                    ),
                    description="Compare the average-minus-friend gap for this question-order condition.",
                    human_value=human_map["average"] - human_map["friend"],
                    agent_value=None
                    if None in (agent_map["friend"], agent_map["average"])
                    else agent_map["average"] - agent_map["friend"],
                    normalization="1 - absolute_error / 10 on the 11-point scale",
                    sources=["data/studies/P_M/ground_truth.json:Study 2 / original_data_points"],
                ),
                make_metric(
                    metric_id=f"study2_{order_key}_ordering",
                    label=f"Study 2 {label} target ordering",
                    score=compare_pairwise_order(human_map, agent_map),
                    description="Check whether the target ordering self < friend < average matches the human ordering.",
                    human_value=human_map,
                    agent_value=agent_map,
                    normalization="pairwise ordering agreement rate",
                    sources=["data/studies/P_M/ground_truth.json:Study 2 / original_data_points"],
                ),
            ]
        )

    structure_family = make_metric_family(
        family_id="normative_gap_structure",
        label="Normative Gap Structure",
        description="Compare pluralistic-ignorance gaps and target ordering in Studies 1 and 2.",
        metrics=structure_metrics,
        role="primary",
    )

    dispersion_family = make_metric_family(
        family_id="dispersion_alignment",
        label="Dispersion Alignment",
        description="Compare the Study 1 variance pattern behind the illusion of universality.",
        metrics=[
            make_metric(
                metric_id="study1_self_sd",
                label="Study 1 self-rating SD",
                score=score_absolute_error(safe_stdev(row.get("q1") for row in study1_rows), human_study1["Total_Self"]["sd"], scale=10.0),
                description="Compare the dispersion of self-ratings.",
                human_value=human_study1["Total_Self"]["sd"],
                agent_value=safe_stdev(row.get("q1") for row in study1_rows),
                normalization="1 - absolute_error / 10 on the 11-point scale",
                sources=["data/studies/P_M/ground_truth.json:Study 1 / Total_Self"],
            ),
            make_metric(
                metric_id="study1_average_sd",
                label="Study 1 average-student SD",
                score=score_absolute_error(safe_stdev(row.get("q2") for row in study1_rows), human_study1["Total_Average_Estimate"]["sd"], scale=10.0),
                description="Compare the dispersion of average-student estimates.",
                human_value=human_study1["Total_Average_Estimate"]["sd"],
                agent_value=safe_stdev(row.get("q2") for row in study1_rows),
                normalization="1 - absolute_error / 10 on the 11-point scale",
                sources=["data/studies/P_M/ground_truth.json:Study 1 / Total_Average_Estimate"],
            ),
            make_metric(
                metric_id="study1_sd_gap",
                label="Study 1 self-vs-average SD gap",
                score=score_absolute_error(
                    None
                    if None in (
                        safe_stdev(row.get("q1") for row in study1_rows),
                        safe_stdev(row.get("q2") for row in study1_rows),
                    )
                    else safe_stdev(row.get("q1") for row in study1_rows)
                    - safe_stdev(row.get("q2") for row in study1_rows),
                    human_study1["Total_Self"]["sd"] - human_study1["Total_Average_Estimate"]["sd"],
                    scale=10.0,
                ),
                description="Compare the variance gap between self-ratings and estimates of others.",
                human_value=human_study1["Total_Self"]["sd"] - human_study1["Total_Average_Estimate"]["sd"],
                agent_value=None
                if None in (
                    safe_stdev(row.get("q1") for row in study1_rows),
                    safe_stdev(row.get("q2") for row in study1_rows),
                )
                else safe_stdev(row.get("q1") for row in study1_rows)
                - safe_stdev(row.get("q2") for row in study1_rows),
                normalization="1 - absolute_error / 10 on the 11-point scale",
                sources=["data/studies/P_M/ground_truth.json:Study 1 / F2"],
            ),
        ],
        role="primary",
    )

    human_study4 = {
        "others_more_negative": {
            "signatures": human_study4_f1["Others_more_negative_Women_Signatures"]["mean"],
            "hours": human_study4_f1["Others_more_negative_Women_Hours"]["mean"],
            "reunions": human_study4_f2["Others_more_negative_Women_Reunions"]["mean"],
        },
        "others_the_same": {
            "signatures": human_study4_f1["Others_the_same_Women_Signatures"]["mean"],
            "hours": human_study4_f1["Others_the_same_Women_Hours"]["mean"],
            "reunions": human_study4_f2["Others_the_same_Women_Reunions"]["mean"],
        },
    }
    agent_study4 = {
        "others_more_negative": {
            "signatures": _group_mean(study4_rows, "q3", "others_more_negative"),
            "hours": _group_mean(study4_rows, "q4", "others_more_negative"),
            "reunions": _group_mean(study4_rows, "q5", "others_more_negative"),
        },
        "others_the_same": {
            "signatures": _group_mean(study4_rows, "q3", "others_the_same"),
            "hours": _group_mean(study4_rows, "q4", "others_the_same"),
            "reunions": _group_mean(study4_rows, "q5", "others_the_same"),
        },
    }

    social_metrics: List[Dict[str, Any]] = []
    for group_key, label in [
        ("others_more_negative", "Others-more-negative"),
        ("others_the_same", "Others-the-same"),
    ]:
        social_metrics.extend(
            [
                make_metric(
                    metric_id=f"study4_{group_key}_signatures",
                    label=f"Study 4 {label} signatures mean",
                    score=score_absolute_error(agent_study4[group_key]["signatures"], human_study4[group_key]["signatures"], scale=100.0),
                    description="Compare willingness to collect signatures in this perceived-deviance group.",
                    human_value=human_study4[group_key]["signatures"],
                    agent_value=agent_study4[group_key]["signatures"],
                    normalization="1 - absolute_error / 100 on the signatures scale",
                    sources=["data/studies/P_M/ground_truth.json:Study 4 / F1"],
                ),
                make_metric(
                    metric_id=f"study4_{group_key}_hours",
                    label=f"Study 4 {label} hours mean",
                    score=score_absolute_error(agent_study4[group_key]["hours"], human_study4[group_key]["hours"], scale=10.0),
                    description="Compare willingness to spend time discussing protest in this perceived-deviance group.",
                    human_value=human_study4[group_key]["hours"],
                    agent_value=agent_study4[group_key]["hours"],
                    normalization="1 - absolute_error / 10 on the hours scale",
                    sources=["data/studies/P_M/ground_truth.json:Study 4 / F1"],
                ),
                make_metric(
                    metric_id=f"study4_{group_key}_reunions",
                    label=f"Study 4 {label} reunions mean",
                    score=score_absolute_error(agent_study4[group_key]["reunions"], human_study4[group_key]["reunions"], scale=100.0),
                    description="Compare expected reunion attendance in this perceived-deviance group.",
                    human_value=human_study4[group_key]["reunions"],
                    agent_value=agent_study4[group_key]["reunions"],
                    normalization="1 - absolute_error / 100 on the percentage scale",
                    sources=["data/studies/P_M/ground_truth.json:Study 4 / F2"],
                ),
            ]
        )

    for metric_key, scale in [("signatures", 100.0), ("hours", 10.0), ("reunions", 100.0)]:
        human_gap = human_study4["others_the_same"][metric_key] - human_study4["others_more_negative"][metric_key]
        agent_gap = (
            None
            if None in (
                agent_study4["others_the_same"][metric_key],
                agent_study4["others_more_negative"][metric_key],
            )
            else agent_study4["others_the_same"][metric_key]
            - agent_study4["others_more_negative"][metric_key]
        )
        social_metrics.append(
            make_metric(
                metric_id=f"study4_{metric_key}_gap",
                label=f"Study 4 {metric_key} group gap",
                score=score_absolute_error(agent_gap, human_gap, scale=scale),
                description="Compare the Others-the-same minus Others-more-negative gap on this outcome.",
                human_value=human_gap,
                agent_value=agent_gap,
                normalization=f"1 - absolute_error / {scale}",
                sources=["data/studies/P_M/ground_truth.json:Study 4 / original_data_points"],
            )
        )

    social_family = make_metric_family(
        family_id="social_action_alignment",
        label="Social Action and Alienation",
        description="Compare Study 4 downstream action and alienation outcomes by perceived deviance group.",
        metrics=social_metrics,
        role="primary",
    )

    diagnostics = {
        "mode_collapse": {
            "study_1_q1": collapse_diagnostic([row.get("q1") for row in study1_rows]),
            "study_1_q2": collapse_diagnostic([row.get("q2") for row in study1_rows]),
            "study_2_self_first_profile": collapse_diagnostic(
                [
                    (row.get("q1"), row.get("q2"), row.get("q3"))
                    for row in study2_rows
                    if row.get("order") == "self_first"
                ]
            ),
            "study_2_average_first_profile": collapse_diagnostic(
                [
                    (row.get("q1"), row.get("q2"), row.get("q3"))
                    for row in study2_rows
                    if row.get("order") == "average_first"
                ]
            ),
            "study_4_group": collapse_diagnostic([row.get("group") for row in study4_rows]),
        }
    }

    auxiliary = make_metric_family(
        family_id="distribution_diagnostics",
        label="Distribution diagnostics",
        description="Check for mode collapse and similar coverage issues. Does not count toward the headline score.",
        metrics=[],
        role="auxiliary",
    )
    return finalize_alignment_result(
        study_id="P_M",
        study_label="Study 8 – pluralistic ignorance and campus drinking",
        included_in_aggregate=True,
        primary_families=[level_family, structure_family, dispersion_family, social_family],
        auxiliary_families=[auxiliary],
        diagnostics=diagnostics,
        notes=[
            "Headline score compares directly observable means/gaps; it is not a replication of the paper's ANOVA/ANCOVA. Study 4 uses female ground-truth rows; the evaluator uses female agent rows when available, else all agent rows, so agents without gender or study-4 data may reduce comparability to human female-only statistics.",
        ],
    )
