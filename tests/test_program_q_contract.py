from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.adjudicate_program_q import CELL_IDS, adjudicate
from scripts.audit_program_q_seed_custody import scan
from scripts.benchmark_program_q_latency import benchmark_callable
from scripts.power_program_q_replication import bootstrap_effects, point_effects


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts/program_q_frozen_policy_replication_v1.json").read_text()
)
FALLBACK = (
    ROOT
    / "research/paper2_exhaustive_search/program_q_historical_recurrentppo_fallback_freeze_20260717.json"
)


def test_historical_fallback_is_hash_frozen_without_retraining() -> None:
    payload = json.loads(FALLBACK.read_text())
    assert payload["status"] == "FROZEN_FALLBACK_PENDING_DAVID_950_OR_DEFAULT_SELECTION"
    assert payload["training"]["retraining_for_program_q"] is False
    assert payload["training"]["checkpoint_selection"] == "final only"
    assert len(payload["checkpoints_sha256"]) == 10
    assert all(len(value) == 64 for value in payload["checkpoints_sha256"].values())
    assert payload["scientific_seed_state"]["7490001_7490256"] == "UNOPENED"


def result_fixture(*, h_lcb=0.02, delta_lcb=-0.005, delta_ucb=0.005):
    estimates = {}
    summaries = {}
    for cell in CELL_IDS:
        estimates[f"{cell}::H_OL"] = {"lcb95": h_lcb, "point": 0.05}
        estimates[f"{cell}::Delta_N"] = {
            "lcb95": delta_lcb,
            "ucb95": delta_ucb,
            "point": 0.0,
        }
        summaries[cell] = {
            "favorable_tapes_fraction_vs_open_loop": 0.8,
            "positive_learner_seeds_H_OL": 9,
        }
    return {
        "inference": {"estimates": estimates},
        "cell_summaries": summaries,
        "integrity_gates": {
            "feedback": True,
            "replacement_controls": True,
            "scheduled_resources_exact": True,
            "mass_partition_demand": True,
            "ret_full_noninferior": True,
            "quantity_ret_full_noninferior": True,
            "worst_product_fill_noninferior": True,
        },
    }


def test_program_q_equivalence_and_premium_are_distinct() -> None:
    equivalent = adjudicate(result_fixture(), CONTRACT)
    assert equivalent["verdict"] == "PASS_Q_LEARNED_ADAPTATION_CLASSICALLY_EQUIVALENT"
    premium = adjudicate(
        result_fixture(delta_lcb=0.011, delta_ucb=0.03), CONTRACT
    )
    assert premium["verdict"] == "PASS_Q_NEURAL_PREMIUM"


def test_program_q_adjudication_fails_closed_on_missing_integrity() -> None:
    result = result_fixture()
    del result["integrity_gates"]["feedback"]
    payload = adjudicate(result, CONTRACT)
    assert payload["verdict"] == "STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION"
    assert not payload["paper3_authorized"]


def test_power_bootstrap_reselects_comparator_families() -> None:
    panels = {}
    for cell in CELL_IDS:
        learner = np.full((3, 4), 0.75)
        open_loop = np.asarray(
            [[0.70, 0.72], [0.76, 0.69], [0.70, 0.72], [0.76, 0.69]]
        )
        classical = np.asarray([[0.73, 0.73, 0.73, 0.73], [0.74, 0.70, 0.74, 0.70]])
        panels[cell] = {"learner": learner, "open_loop": open_loop, "classical": classical}
    points = point_effects(panels)
    assert points.shape == (6,)
    draws = bootstrap_effects(
        panels, tape_count=8, replicates=12, rng=np.random.default_rng(11), batch_size=4
    )
    assert draws.shape == (12, 6)
    assert np.isfinite(draws).all()


def test_seed_custody_scan_allows_contract_declaration_only(tmp_path: Path) -> None:
    (tmp_path / "contracts").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "contracts/program_q_frozen_policy_replication_v1.json").write_text(
        '{"reserved": [7490001, 7490256]}'
    )
    (tmp_path / "scripts/audit_program_q_seed_custody.py").write_text("7_490_001")
    assert scan(tmp_path)["pass"]
    (tmp_path / "results").mkdir()
    (tmp_path / "results/result.json").write_text('{"seed": 7490001}')
    payload = scan(tmp_path)
    assert not payload["pass"]
    assert payload["status"] == "STOP_PROGRAM_Q_SEED_COLLISION"


def test_latency_benchmark_reports_batch_one_and_failures() -> None:
    payload = benchmark_callable(
        lambda observation: int(np.argmax(observation)),
        [np.asarray([0.0, 1.0], dtype=np.float32)],
        warmup=2,
        repeats=10,
    )
    assert payload["batch_size"] == 1
    assert payload["failures"] == 0
    assert payload["p95_ms"] >= 0.0
