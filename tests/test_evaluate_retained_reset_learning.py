from __future__ import annotations

from scripts import evaluate_retained_reset_learning as evaluator
from supply_chain.config import (
    TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE,
    TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE,
    TRACK_A_TRAINING_RISK_OCCURRENCE_MODE,
    canonical_raw_material_flow_mode,
)


def test_retained_reset_parser_defaults_to_track_a_gate() -> None:
    args = evaluator.build_parser().parse_args([])

    assert args.downstream_q_source == TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE
    assert args.algo == "dqn"
    assert args.decision_cadence == "block"
    assert args.stochastic_pt is False
    assert args.pretrain_timesteps == 0
    assert args.online_timesteps_per_cycle == 0
    assert args.n_steps == 8
    assert args.n_epochs == 2
    assert evaluator.parse_ints(args.train_seeds)
    assert evaluator.parse_ints(args.eval_seeds)


def test_retained_reset_env_uses_repaired_track_a_defaults() -> None:
    args = evaluator.build_parser().parse_args(["--max-steps", "1"])
    env = evaluator.build_env(args)
    _obs, info = env.reset(seed=123)

    assert env.action_space.n == 18
    assert info["downstream_q_source"] == TRACK_A_TRAINING_DOWNSTREAM_Q_SOURCE
    assert (
        info["raw_material_flow_mode"]
        == canonical_raw_material_flow_mode(TRACK_A_TRAINING_RAW_MATERIAL_FLOW_MODE)
    )
    assert info["risk_occurrence_mode"] == TRACK_A_TRAINING_RISK_OCCURRENCE_MODE
    env.close()


def test_block_decision_env_holds_one_action_for_whole_block() -> None:
    args = evaluator.build_parser().parse_args(["--max-steps", "2"])
    env = evaluator.build_env(args)
    _obs, _info = env.reset(seed=123)
    _obs, _reward, terminated, truncated, info = env.step(0)

    assert info["decision_cadence"] == "block"
    assert info["held_action"] == 0
    assert info["block_steps"] == 2
    assert terminated or truncated
    env.close()


def test_retained_reset_env_can_enable_stochastic_pt_extension() -> None:
    args = evaluator.build_parser().parse_args(["--max-steps", "1", "--stochastic-pt"])
    env = evaluator.build_env(args)
    _obs, _info = env.reset(seed=123)

    assert env.unwrapped.stochastic_pt is True
    env.close()


def test_retained_reset_parser_accepts_ppo() -> None:
    args = evaluator.build_parser().parse_args(
        ["--algo", "ppo", "--n-steps", "4", "--batch-size", "4"]
    )

    assert args.algo == "ppo"
    assert evaluator.condition_name(args, "retained_online") == "retained_online_ppo"


def test_regime_disabled_by_default_is_stationary() -> None:
    args = evaluator.build_parser().parse_args([])
    assert evaluator.regime_enabled(args) is False
    assert evaluator.build_tape(args, 5, seed=1) is None
    # No regime => stationary current + neutral demand.
    kwargs = evaluator.env_kwargs(args, regime=None)
    assert kwargs["risk_level"] == args.risk_level
    assert kwargs["demand_mean_multiplier"] == 1.0


def test_regime_enabled_when_rho_given() -> None:
    args = evaluator.build_parser().parse_args(["--rho-disruption", "0.8"])
    assert evaluator.regime_enabled(args) is True
    tape = evaluator.build_tape(args, 5, seed=1)
    assert tape is not None and len(tape) == 5
    # Unset rho falls back to memoryless (the single-chain ablation).
    assert tape.rho_demand == 1 / 3


def test_env_kwargs_maps_regime_to_severity_and_demand() -> None:
    from supply_chain.scenario_tape import RegimePhase

    args = evaluator.build_parser().parse_args([])
    regime = RegimePhase(
        disruption_phase=1,
        demand_phase=2,
        disruption_level="severe",
        demand_multiplier=1.2,
    )
    kwargs = evaluator.env_kwargs(args, regime=regime)
    assert kwargs["risk_level"] == "severe"
    assert kwargs["demand_mean_multiplier"] == 1.2
