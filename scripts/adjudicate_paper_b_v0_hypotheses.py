#!/usr/bin/env python3
"""Mechanically adjudicate the frozen Paper-B-v0 H1-H4 mapping on Q-R1 full."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "contracts/paper_b_v0_hypothesis_mapping_v1.json"
MAPPING_FREEZE = (
    ROOT / "contracts/paper_b_v0_hypothesis_mapping_v1_freeze_receipt.json"
)
Q_CONTRACT = ROOT / "contracts/q_r1_matched_retention_factorial_v4.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {
        "n_optimizer_seeds": n,
        "mean": mean,
        "sd_between_optimizer_seeds": sd,
        "lcb95_descriptive": mean - half,
        "ucb95_descriptive": mean + half,
        "positive_seeds": sum(value > 0.0 for value in values),
        "values_by_optimizer_seed": values,
    }


def arm_index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            str(row["arm"]),
            int(row["history_root"]),
            float(row["kappa"]),
            int(row["campaign_index"]),
        ): row
        for row in rows
    }


def paired_values(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    left: str,
    right: str,
    reverse: bool = False,
) -> list[float]:
    index = arm_index(rows)
    identities = sorted(
        {
            key[1:]
            for key in index
            if key[0] == left and (right, *key[1:]) in index
        }
    )
    values = [
        float(index[(left, *identity)][metric])
        - float(index[(right, *identity)][metric])
        for identity in identities
    ]
    return [-value for value in values] if reverse else values


def slope(values: list[tuple[int, float]]) -> float:
    x = np.asarray([item[0] for item in values], dtype=float)
    y = np.asarray([item[1] for item in values], dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def h2_seed_value(rows: list[dict[str, Any]]) -> float:
    grouped: dict[tuple[str, int, float], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        if arm not in {"P0_H0", "P1_H1"}:
            continue
        grouped[(arm, int(row["history_root"]), float(row["kappa"]))].append(
            (int(row["campaign_index"]), float(row["early_ret_complete_cohort"]))
        )
    differences = []
    for root, kappa in sorted(
        {(key[1], key[2]) for key in grouped if key[0] == "P1_H1"}
    ):
        differences.append(
            slope(grouped[("P1_H1", root, kappa)])
            - slope(grouped[("P0_H0", root, kappa)])
        )
    return statistics.fmean(differences)


def h3_seed_values(rows: list[dict[str, Any]]) -> tuple[float, float]:
    by_arm_kappa: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        if arm in {"P0_H0", "P1_H1"}:
            by_arm_kappa[(arm, float(row["kappa"]))].append(
                float(row["early_ret_complete_cohort"])
            )
    mad_improvements = []
    variance_ratios = []
    for kappa in (0.5, 0.75, 0.9):
        reset = by_arm_kappa[("P0_H0", kappa)]
        retained = by_arm_kappa[("P1_H1", kappa)]
        reset_median = statistics.median(reset)
        retained_median = statistics.median(retained)
        reset_mad = statistics.fmean(abs(value - reset_median) for value in reset)
        retained_mad = statistics.fmean(
            abs(value - retained_median) for value in retained
        )
        mad_improvements.append(reset_mad - retained_mad)
        reset_variance = statistics.variance(reset)
        retained_variance = statistics.variance(retained)
        variance_ratios.append(retained_variance / reset_variance)
    return statistics.fmean(mad_improvements), statistics.fmean(variance_ratios)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite adjudication")

    mapping = json.loads(MAPPING.read_text())
    freeze = json.loads(MAPPING_FREEZE.read_text())
    contract = json.loads(Q_CONTRACT.read_text())
    selection = json.loads(args.full_selection.read_text())
    if freeze["contract_sha256"] != sha256(MAPPING):
        raise RuntimeError("hypothesis mapping hash mismatch")
    if selection.get("phase") != "full" or selection.get(
        "confirmation_roots_opened"
    ) is not False:
        raise RuntimeError("invalid full selection")
    if selection.get("contract_sha256") != sha256(Q_CONTRACT):
        raise RuntimeError("Q-R1 contract hash mismatch")
    selected = str(selection["advanced_config_ids"][0])
    ranking = next(row for row in selection["ranking"] if row["config_id"] == selected)
    if len(ranking["worker_results"]) != 5:
        raise RuntimeError("selected configuration does not have five seeds")

    workers: list[dict[str, Any]] = []
    rows_by_seed: list[list[dict[str, Any]]] = []
    for reference in ranking["worker_results"]:
        result_path = Path(reference["path"])
        if sha256(result_path) != reference["sha256"]:
            raise RuntimeError("worker result hash mismatch")
        worker = json.loads(result_path.read_text())
        if worker["config_id"] != selected:
            raise RuntimeError("worker configuration mismatch")
        if worker.get("confirmation_roots_opened") is not False:
            raise RuntimeError("worker opened confirmation roots")
        rows = json.loads((result_path.parent / "rows.json").read_text())
        if len(rows) != 3072:
            raise RuntimeError("selected rows incomplete")
        workers.append(worker)
        rows_by_seed.append(rows)

    total = interval(
        [
            float(worker["estimands"]["total_retained_neural_treatment"]["mean"])
            for worker in workers
        ]
    )
    explicit = interval(
        [
            float(worker["estimands"]["explicit_context_value"]["mean"])
            for worker in workers
        ]
    )
    recurrent = interval(
        [
            float(worker["estimands"]["recurrent_residual_given_context"]["mean"])
            for worker in workers
        ]
    )
    interaction = interval(
        [float(worker["estimands"]["interaction"]["mean"]) for worker in workers]
    )
    neural_premium = interval(
        [float(worker["estimands"]["neural_premium"]["mean"]) for worker in workers]
    )
    iid_absolute = [
        -float(
            worker["checkpoint_selection_scores"][
                str(int(worker["selected_checkpoint"]["timesteps"]))
            ][3]
        )
        for worker in workers
    ]

    dose: dict[str, float] = {}
    for kappa in (0.5, 0.75, 0.9):
        seed_values = []
        for rows in rows_by_seed:
            index = arm_index(rows)
            identities = sorted(
                {
                    key[1:]
                    for key in index
                    if key[0] == "P1_H1" and key[2] == kappa
                }
            )
            seed_values.append(
                statistics.fmean(
                    float(index[("P1_H1", *identity)]["early_ret_complete_cohort"])
                    - float(index[("P0_H0", *identity)]["early_ret_complete_cohort"])
                    for identity in identities
                )
            )
        dose[str(kappa)] = statistics.fmean(seed_values)

    retention_gates = {
        "point_estimate_at_least_0p01": total["mean"] >= 0.01,
        "descriptive_lcb95_positive": total["lcb95_descriptive"] > 0.0,
        "optimizer_seeds_positive_at_least_4_of_5": total["positive_seeds"] >= 4,
        "iid_absolute_effect_at_most_0p005": statistics.fmean(iid_absolute)
        <= 0.005,
        "dose_response_0p90_gt_0p75_gt_0p50": (
            dose["0.9"] > dose["0.75"] > dose["0.5"]
        ),
    }
    premium_by_kappa: dict[str, float] = {}
    for kappa in (0.5, 0.75, 0.9):
        values = []
        for rows in rows_by_seed:
            index = arm_index(rows)
            identities = sorted(
                {
                    key[1:]
                    for key in index
                    if key[0] == "structured_retained" and key[2] == kappa
                }
            )
            values.append(
                statistics.fmean(
                    float(index[("P1_H1", *identity)]["early_ret_complete_cohort"])
                    - float(
                        index[("structured_retained", *identity)][
                            "early_ret_complete_cohort"
                        ]
                    )
                    for identity in identities
                )
            )
        premium_by_kappa[str(kappa)] = statistics.fmean(values)
    premium_gates = {
        "descriptive_lcb95_at_least_0p01": (
            neural_premium["lcb95_descriptive"] >= 0.01
        ),
        "no_kappa_cell_adverse_by_point_estimate": all(
            value >= 0.0 for value in premium_by_kappa.values()
        ),
        "same_information_rights": True,
    }

    h1_service = interval(
        [
            statistics.fmean(
                paired_values(
                    rows,
                    metric="service_loss",
                    left="P1_H1",
                    right="P0_H0",
                    reverse=True,
                )
            )
            for rows in rows_by_seed
        ]
    )
    h1_worst_fill = interval(
        [
            statistics.fmean(
                paired_values(
                    rows,
                    metric="worst_product_fill",
                    left="P1_H1",
                    right="P0_H0",
                )
            )
            for rows in rows_by_seed
        ]
    )
    h2 = interval([h2_seed_value(rows) for rows in rows_by_seed])
    h3_values = [h3_seed_values(rows) for rows in rows_by_seed]
    h3_mad = interval([item[0] for item in h3_values])
    h3_variance_ratio = interval([item[1] for item in h3_values])

    payload = {
        "schema_version": "paper_b_v0_hypothesis_adjudication_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_status": "DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIM",
        "mapping_sha256": sha256(MAPPING),
        "mapping_freeze_sha256": sha256(MAPPING_FREEZE),
        "q_r1_contract_sha256": sha256(Q_CONTRACT),
        "full_selection_sha256": sha256(args.full_selection),
        "selected_config_id": selected,
        "optimizer_seeds": sorted(int(worker["optimizer_seed"]) for worker in workers),
        "confirmation_roots_opened": False,
        "H1_learning_effect": {
            "status": "PRIMARY_NOT_ADJUDICABLE_RECOVERY_TIME_NOT_RECORDED",
            "secondary_service_loss_improvement": h1_service,
            "secondary_worst_product_fill_improvement": h1_worst_fill,
        },
        "H2_adaptation": {
            "status": (
                "DIRECTIONAL_DEVELOPMENT_SUPPORT"
                if h2["lcb95_descriptive"] > 0.0
                else "DIRECTION_NOT_SUPPORTED"
            ),
            "campaign_slope_difference_P1_H1_minus_P0_H0": h2,
        },
        "H3_volatility_reduction": {
            "status": (
                "DIRECTIONAL_DEVELOPMENT_SUPPORT"
                if h3_mad["lcb95_descriptive"] > 0.0
                and h3_variance_ratio["ucb95_descriptive"] < 1.0
                else "DIRECTION_NOT_SUPPORTED"
            ),
            "mean_absolute_deviation_reduction": h3_mad,
            "variance_ratio_P1_H1_over_P0_H0": h3_variance_ratio,
        },
        "H4_path_dependency": {
            "status": (
                "PASS_DEVELOPMENT_RETENTION_GATE"
                if all(retention_gates.values())
                else "FAIL_DEVELOPMENT_RETENTION_GATE"
            ),
            "total_retained_neural_treatment": total,
            "explicit_context_value": explicit,
            "recurrent_residual_given_context": recurrent,
            "interaction": interaction,
            "iid_absolute_effect_mean": statistics.fmean(iid_absolute),
            "dose_response_by_kappa": dose,
            "retention_gates": retention_gates,
        },
        "architecture_gate": {
            "status": (
                "AUTHORIZED_DEVELOPMENT_BAKEOFF"
                if all(premium_gates.values())
                else "NO_GO_NEURAL_BAKEOFF"
            ),
            "neural_premium": neural_premium,
            "neural_premium_by_kappa": premium_by_kappa,
            "gates": premium_gates,
        },
        "mapping_snapshot": mapping["hypotheses"],
        "boundaries": {
            "confirmation_authorized": False,
            "kan_authorized": False,
            "cobb_douglas_selection_use": False,
            "submission_a_modified": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
