from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from supply_chain.l_program_env import (
    BUFFER_LEVELS,
    OBSERVATION_FIELDS,
    CampaignTape,
    FixedNormalizerStats,
    GarridoLearningEnv,
    compute_system_recovery_metrics,
    fit_fixed_normalizer,
    materialize_campaign_tape,
)


def _tape(seed: int = 771001, *, weeks: int = 3) -> CampaignTape:
    return CampaignTape(
        campaign_id=f"test-{seed}",
        family="R1",
        risk_level="current",
        base_seed=seed,
        horizon_weeks=weeks,
        split="smoke",
    )


def test_campaign_tape_hash_and_r3_training_guard() -> None:
    tape = _tape()
    restored = CampaignTape.from_mapping(tape.payload(include_hash=True))
    assert restored == tape
    assert restored.digest() == tape.digest()

    with pytest.raises(ValueError, match="OOD-only"):
        CampaignTape(
            campaign_id="bad-r3",
            family="R3",
            risk_level="current",
            base_seed=1,
            horizon_weeks=3,
            split="training",
        )


def test_fixed_normalizer_is_immutable_and_field_locked() -> None:
    zeros = np.zeros(len(OBSERVATION_FIELDS))
    ones = np.ones(len(OBSERVATION_FIELDS))
    stats = fit_fixed_normalizer(
        [zeros, ones], calibration_sha256="abc", clip=5.0
    )
    before = stats.payload()
    transformed = stats.transform(ones.astype(np.float32))
    assert transformed.shape == (24,)
    assert stats.payload() == before

    with pytest.raises(ValueError, match="fields/mean/std"):
        FixedNormalizerStats(("x",), (), (1.0,))


def test_discrete_shift_has_one_week_lag_and_preserves_noncontrol_params() -> None:
    env = GarridoLearningEnv(max_steps=3, buffer_level=168)
    try:
        _obs, info = env.reset(
            seed=771001,
            options={
                "campaign_tape": _tape(),
                "buffer_level": 168,
                "initial_state_seed": 771001,
            },
        )
        assert env.action_space.n == 3
        assert len(OBSERVATION_FIELDS) == 24
        assert not any(
            forbidden in field
            for field in OBSERVATION_FIELDS
            for forbidden in ("forecast", "regime", "risk_id", "future")
        )
        frozen_targets = dict(info["buffer_targets"])
        frozen_params = {
            key: value
            for key, value in env.sim.params.items()
            if key not in {"assembly_shifts", "batch_size"}
        }

        _obs, _reward, _term, _trunc, first = env.step(2)  # request S3
        assert first["requested_shift"] == 3
        assert first["effective_shift"] == 1
        assert first["pending_shift"] == 3

        _obs, _reward, _term, _trunc, second = env.step(0)  # request S1
        assert second["effective_shift"] == 3
        assert second["pending_shift"] == 1
        assert second["buffer_targets"] == frozen_targets
        assert {
            key: value
            for key, value in env.sim.params.items()
            if key not in {"assembly_shifts", "batch_size"}
        } == frozen_params
    finally:
        env.close()


def test_reset_removes_physical_carryover_and_crn_is_deterministic() -> None:
    actions = [2, 0, 1]

    def rollout() -> tuple[np.ndarray, list[tuple[float, int, float]]]:
        env = GarridoLearningEnv(max_steps=3, buffer_level=336)
        try:
            obs0, _ = env.reset(
                seed=771002,
                options={
                    "campaign_tape": _tape(771002),
                    "buffer_level": 336,
                    "initial_state_seed": 771002,
                },
            )
            rows = []
            for action in actions:
                _obs, reward, term, trunc, info = env.step(action)
                rows.append(
                    (
                        float(reward),
                        int(info["effective_shift"]),
                        float(info["ret_excel"]),
                    )
                )
                if term or trunc:
                    break
            obs_again, _ = env.reset(
                seed=771002,
                options={
                    "campaign_tape": _tape(771002),
                    "buffer_level": 336,
                    "initial_state_seed": 771002,
                },
            )
            np.testing.assert_array_equal(obs0, obs_again)
            return obs0, rows
        finally:
            env.close()

    obs_a, rows_a = rollout()
    obs_b, rows_b = rollout()
    np.testing.assert_array_equal(obs_a, obs_b)
    assert rows_a == rows_b


def test_buffer_levels_are_exactly_the_six_preregistered_levels() -> None:
    assert BUFFER_LEVELS == (0, 168, 336, 504, 672, 1344)


def test_materialized_calendar_replays_identically_across_shift_policies() -> None:
    source = CampaignTape(
        campaign_id="test-materialized-crn",
        family="R2",
        risk_level="increased",
        base_seed=771090,
        horizon_weeks=12,
        split="calibration",
    )
    tape = materialize_campaign_tape(source)
    assert tape.risk_events

    def replay(
        policy_shift: int,
    ) -> tuple[
        list[tuple[str, float, float, tuple[int, ...]]],
        list[tuple[float, float, bool]],
    ]:
        env = GarridoLearningEnv(max_steps=12, buffer_level=168)
        try:
            env.reset(
                seed=tape.base_seed,
                options={
                    "campaign_tape": tape,
                    "buffer_level": 168,
                    "initial_state_seed": tape.base_seed,
                    "initial_shift": 1,
                },
            )
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(policy_shift - 1)
                done = terminated or truncated
            events = [
                (
                    str(event.risk_id),
                    float(event.start_time - env._treatment_start),
                    float(event.duration),
                    tuple(int(op) for op in event.affected_ops),
                )
                for event in env.sim.risk_events
            ]
            demand = [
                (
                    float(order.OPTj - env._treatment_start),
                    float(order.quantity),
                    bool(order.contingent),
                )
                for order in env.sim.orders
                if float(order.OPTj) >= env._treatment_start
            ]
            return events, demand
        finally:
            env.close()

    events_s1, demand_s1 = replay(1)
    events_s3, demand_s3 = replay(3)
    assert events_s1 == events_s3
    assert demand_s1 == demand_s3


def test_system_ttr_clusters_events_and_recovers_after_two_healthy_weeks() -> None:
    h = 168.0
    history = [
        {"time": 0.0, "fill_rate": 1.0, "backlog_qty": 100.0},
        {"time": h, "fill_rate": 1.0, "backlog_qty": 100.0},
        {"time": 2 * h, "fill_rate": 1.0, "backlog_qty": 100.0},
        {"time": 3 * h, "fill_rate": 1.0, "backlog_qty": 100.0},
        {"time": 4 * h, "fill_rate": 0.5, "backlog_qty": 500.0},
        {"time": 5 * h, "fill_rate": 0.96, "backlog_qty": 104.0},
        {"time": 6 * h, "fill_rate": 0.98, "backlog_qty": 100.0},
    ]
    events = [
        SimpleNamespace(start_time=3.5 * h, end_time=4.0 * h),
        # Less than one week from the previous event: same cluster.
        SimpleNamespace(start_time=4.5 * h, end_time=4.75 * h),
    ]
    metrics = compute_system_recovery_metrics(
        history, events, treatment_start=0.0
    )
    assert metrics["system_ttr_n_clusters"] == 1.0
    assert metrics["system_ttr_n_recovered"] == 1.0
    assert metrics["system_ttr_n_censored"] == 0.0
    assert metrics["system_ttr_mean"] == pytest.approx(2.5 * h)
