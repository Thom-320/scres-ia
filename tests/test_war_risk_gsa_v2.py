from __future__ import annotations

import numpy as np
import pytest

from research.paper2_exhaustive_search.war_risk_gsa_v2 import (
    FactorSpace,
    PolicyMetrics,
    StochasticTimingResponse,
    TimingTapeEvaluation,
    analyze_salib_morris,
    audit_crn_noise,
    compute_h_timing_safe,
    fit_cross_validated_surrogate,
    salib_morris_design,
    surrogate_sobol_indices,
)
from research.paper2_exhaustive_search.war_risk_scenario_discovery import (
    validated_prim_discovery,
)


def _metrics(ret: float, *, worst: float = 0.8, lost: float = 0.0) -> dict[str, float]:
    return {
        "ret_excel": ret,
        "ret_excel_full_ledger": ret,
        "ration_ret_excel": ret,
        "ret_excel_cvar10": ret,
        "worst_node_or_product_fill": worst,
        "lost_orders": lost,
        "ret_excel_omitted_n": 0.0,
        "backorder_qty_final": 0.0,
        "backlog_age_max": 0.0,
        "shift_hours": 100.0,
        "surge_hours": 0.0,
        "buffer_target_unit_hours": 0.0,
        "op8_convoy_vehicle_hours": 100.0,
        "clipped_surge_quantity": 0.0,
        "risk_cap_hits": 0.0,
    }


def _response(config_id: str, mean: float, tape_ids: tuple[int, ...] = (1, 2, 3, 4)) -> StochasticTimingResponse:
    offsets = np.linspace(-0.0015, 0.0015, len(tape_ids))
    deltas = np.maximum(0.0, mean + offsets)
    return StochasticTimingResponse(
        config_id=config_id,
        tape_ids=tape_ids,
        event_tape_sha256s=tuple(f"{config_id}-event-{tape}" for tape in tape_ids),
        exogenous_base_stream_sha256s=tuple(f"base-{tape}" for tape in tape_ids),
        deltas=deltas,
        selected_policy_ids=("timing",) * len(tape_ids),
        mean=float(np.mean(deltas)),
        standard_error=float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))),
        favorable_tapes=int(np.sum(deltas > 0.0)),
    )


def test_salib_morris_has_standard_nonzero_steps_and_detects_effect() -> None:
    space = FactorSpace.for_mask("LOC_SURGE", "independent")
    design = salib_morris_design(
        space,
        candidate_trajectories=12,
        selected_trajectories=6,
        levels=8,
        seed=11,
    )
    trajectories = design.log2_points.reshape(6, len(space.names) + 1, -1)
    changed = np.sum(np.abs(np.diff(trajectories, axis=1)) > 1e-12, axis=2)
    assert np.all(changed == 1)
    response = design.log2_points[:, 0] + 2.0 * design.log2_points[:, 1]
    result = analyze_salib_morris(design, response, bootstrap_resamples=100, seed=12)
    assert result["n_evaluations"] == 6 * (len(space.names) + 1)
    assert result["mu_star"][1] > result["mu_star"][0] > 0.0


def test_h_timing_target_rejects_shed_to_win_and_preserves_crn() -> None:
    comparator = PolicyMetrics("static", _metrics(0.50), "event-hash", "base-hash")
    good = PolicyMetrics("timing-good", _metrics(0.60), "event-hash", "base-hash")
    shed = PolicyMetrics(
        "timing-shed",
        _metrics(0.95, worst=0.20, lost=2.0),
        "event-hash",
        "base-hash",
    )
    tapes = [
        TimingTapeEvaluation(
            tape_id=tape,
            comparator=comparator,
            restricted_candidates=(good, shed),
        )
        for tape in (3, 1, 2)
    ]
    result = compute_h_timing_safe("cell", tapes)
    assert result.tape_ids == (1, 2, 3)
    assert result.selected_policy_ids == ("timing-good",) * 3
    assert result.mean == pytest.approx(0.10)

    worse = PolicyMetrics("timing-worse", _metrics(0.40), "event-hash", "base-hash")
    retained = compute_h_timing_safe(
        "cell",
        [TimingTapeEvaluation(1, comparator, (worse,))],
    )
    assert retained.selected_policy_ids == ("static",)
    assert retained.mean == 0.0

    mismatched = PolicyMetrics("bad-crn", _metrics(0.70), "different-hash", "base-hash")
    with pytest.raises(ValueError, match="policy-dependent risk tape"):
        compute_h_timing_safe(
            "cell",
            [TimingTapeEvaluation(1, comparator, (mismatched,))],
        )


def test_crn_noise_audit_requires_same_tapes_and_reports_noise() -> None:
    responses = [_response("a", 0.01), _response("b", 0.02), _response("c", 0.03)]
    audit = audit_crn_noise(responses)
    assert audit.configuration_count == 3
    assert audit.tapes_per_configuration == 4
    assert 0.0 <= audit.monte_carlo_fraction <= 1.0

    bad = _response("bad", 0.02, tape_ids=(1, 2, 3, 5))
    with pytest.raises(ValueError, match="same ordered CRN"):
        audit_crn_noise([responses[0], bad])

    wrong_hash = StochasticTimingResponse(
        config_id="wrong-hash",
        tape_ids=responses[0].tape_ids,
        event_tape_sha256s=("wrong",) * 4,
        exogenous_base_stream_sha256s=("wrong-base",) * 4,
        deltas=responses[0].deltas,
        selected_policy_ids=responses[0].selected_policy_ids,
        mean=responses[0].mean,
        standard_error=responses[0].standard_error,
        favorable_tapes=responses[0].favorable_tapes,
    )
    with pytest.raises(ValueError, match="base CRN stream hashes differ"):
        audit_crn_noise([responses[0], wrong_hash])


def test_surrogate_gate_and_sobol_are_stratified_and_unclipped() -> None:
    rng = np.random.default_rng(21)
    space = FactorSpace.for_mask("LOC_SURGE", "independent")
    X = np.column_stack(
        [
            rng.uniform(lo, hi, 240)
            for lo, hi in space.log2_bounds
        ]
    )
    means = 0.01 + 0.004 * X[:, 0] + 0.003 * X[:, 1] * X[:, 2] + 0.002 * X[:, 3]
    responses = [_response(f"c{i}", float(value)) for i, value in enumerate(means)]
    fit = fit_cross_validated_surrogate(space, X, responses, seed=22)
    assert fit.gate_pass
    assert fit.cv_r2 >= 0.80
    result = surrogate_sobol_indices(
        fit,
        base_n=512,
        bootstrap_resamples=100,
        seed=23,
    )
    assert result["ST_raw"].shape == (4,)
    assert result["overlapping_higher_order_gap_ST_minus_S1"].shape == (4,)
    assert "not a unique" in result["claim_limit"]
    with pytest.raises(ValueError, match="dependent/coupled"):
        surrogate_sobol_indices(fit, base_n=128, dependent_inputs=True)


def test_surrogate_refuses_underpowered_design() -> None:
    space = FactorSpace.for_mask("LOC_SURGE", "independent")
    X = np.zeros((20, 4))
    responses = [_response(f"c{i}", 0.01) for i in range(20)]
    with pytest.raises(ValueError, match="at least 40"):
        fit_cross_validated_surrogate(space, X, responses)


def test_validated_prim_recovers_interaction_box_on_holdouts() -> None:
    rng = np.random.default_rng(31)
    X = rng.random((1_500, 3))
    positive = (X[:, 0] > 0.70) & (X[:, 1] > 0.60)
    result = validated_prim_discovery(
        X,
        positive,
        names=("r22", "r24", "irrelevant"),
        repeats=8,
        seed=32,
    )
    assert result.status == "VALIDATED_DEVELOPMENT_BOX_HYPOTHESIS"
    assert result.holdout_pass_fraction >= 0.70
    assert "r22" in result.stable_factors
    assert "r24" in result.stable_factors
    assert "irrelevant" not in result.stable_factors


def test_validated_prim_does_not_promote_sparse_or_noise_targets() -> None:
    rng = np.random.default_rng(41)
    X = rng.random((1_000, 3))
    sparse = np.zeros(1_000, dtype=bool)
    sparse[:5] = True
    sparse_result = validated_prim_discovery(X, sparse, names=("a", "b", "c"))
    assert sparse_result.status == "INSUFFICIENT_POSITIVE_OR_NEGATIVE_CONFIGURATIONS"

    noise = rng.random(1_000) < 0.10
    noise_result = validated_prim_discovery(
        X,
        noise,
        names=("a", "b", "c"),
        repeats=8,
        seed=42,
    )
    assert noise_result.status == "NO_STABLE_PRIM_BOX"
