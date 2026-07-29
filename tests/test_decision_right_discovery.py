from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from supply_chain.decision_right_discovery import (
    NumericFactor,
    adaptive_headroom_verdict,
    d_optimal_categorical,
    evaluate_design,
    morris_effects,
    morris_trajectories,
    select_candidate_families,
)


ROOT = Path(__file__).resolve().parent.parent


def test_catalog_separates_actions_from_environment_and_blocks_unimplemented() -> None:
    catalog = json.loads((ROOT / "contracts/decision_right_catalog_v1.json").read_text())
    factors = catalog["factors"]
    assert {factor["operation"] for factor in factors if factor["operation"] is not None} >= set(range(3, 14))
    assert all(factor["class"] != "decision_right" for factor in factors if factor["id"].startswith("risk_"))
    assert any(factor["status"] == "requires_adapter" for factor in factors)


def test_morris_recovers_main_effect_and_interaction() -> None:
    factors = (
        NumericFactor("x", 0.0, 1.0, "a"),
        NumericFactor("z", 0.0, 1.0, "b"),
        NumericFactor("dead", 0.0, 1.0, "c"),
    )
    design, edges = morris_trajectories(factors, trajectories=40, levels=8, seed=7)
    y = 5.0 * design[:, 0] + 8.0 * design[:, 0] * design[:, 1]
    result = morris_effects(design, y, factors, edges)
    assert result["x"]["mu_star"] > result["z"]["mu_star"] > result["dead"]["mu_star"]
    assert result["x"]["sigma"] > 0
    assert result["z"]["sigma"] > 0
    assert result["dead"]["mu_star"] == 0


def test_crn_evaluator_reuses_identical_tape_ids() -> None:
    seen: dict[tuple[float, ...], list[int]] = {}
    factors = (NumericFactor("x", 0, 1, "g"),)
    design = np.asarray([[0.0], [1.0]])

    def simulator(params, tape):
        seen.setdefault(tuple(params.values()), []).append(tape)
        return {"ret_excel": params["x"] + tape / 1000}

    evaluate_design(design, factors, [11, 12, 13], simulator)
    assert list(seen.values()) == [[11, 12, 13], [11, 12, 13]]


def test_sensitive_but_constant_optimal_fails_headroom() -> None:
    values = {
        f"s{i}": {"low": float(i), "high": float(i + 10)}
        for i in range(20)
    }
    tapes = {state: f"t{idx // 2}" for idx, state in enumerate(values)}
    verdict = adaptive_headroom_verdict(values, tapes, observable_deltas=[.02] * 10, service_reduction=.10, resource_equivalent=True)
    assert not verdict["gates"]["ranking_diversity"]
    assert verdict["promote_to_rl"] is False


def test_ranking_reversal_observable_can_pass() -> None:
    values = {}
    tapes = {}
    for i in range(40):
        values[f"s{i}"] = {"A": .50 + (.03 if i % 2 == 0 else 0), "B": .50 + (.03 if i % 2 else 0)}
        tapes[f"s{i}"] = f"t{i // 4}"
    verdict = adaptive_headroom_verdict(
        values, tapes, observable_deltas=[.012] * 10,
        service_reduction=.06, resource_equivalent=True,
    )
    assert verdict["gates"]["ranking_diversity"]
    assert verdict["gates"]["oracle_headroom"]
    assert verdict["promote_to_rl"] is True


def test_contract_keeps_confirmatory_universes_locked() -> None:
    contract = json.loads((ROOT / "contracts/global_sensitivity_v1.json").read_text())
    assert "LOCKED" in contract["universes"]["confirmation"]
    assert "LOCKED" in contract["universes"]["virgin_rl"]
    assert "promote on sensitivity alone" in contract["forbidden"]


def test_categorical_design_has_full_main_effect_rank() -> None:
    design = d_optimal_categorical(
        {"queue": ["fifo", "spt", "mission"], "dispatch": ["clock", "ready"], "quality": ["rework", "discard"]},
        n_rows=7, seed=9,
    )
    assert len(design) == 7
    matrix = np.asarray([[1, row["queue"] == "spt", row["queue"] == "mission", row["dispatch"] == "ready", row["quality"] == "discard"] for row in design], dtype=float)
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]


def test_candidate_selection_excludes_influential_environment_factor() -> None:
    catalog = json.loads((ROOT / "contracts/decision_right_catalog_v1.json").read_text())["factors"]
    sensitivity = {
        "risk_frequency_scale": {"mu_star": 100, "sigma": 50},
        "op3_inventory_target": {"mu_star": 4, "sigma": 1},
        "op9_queue_rule": {"mu_star": 3, "sigma": 1},
        "op8_convoy_capacity": {"mu_star": 2, "sigma": 1},
    }
    selected = select_candidate_families(sensitivity, catalog, minimum=3, maximum=5)
    assert len(selected) == 3
    assert all(row["lead_factor"] != "risk_frequency_scale" for row in selected)
