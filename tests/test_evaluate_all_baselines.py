from __future__ import annotations

from supply_chain.config import (
    BENCHMARK_OBSERVATION_VERSION,
    BENCHMARK_REWARD_MODE,
    CAPACITY_BY_SHIFTS,
    OPERATIONS,
)
from scripts.evaluate_all_baselines import build_parser, garrido_cf_policy


def test_evaluate_all_baselines_defaults_follow_frozen_contract() -> None:
    args = build_parser().parse_args([])
    assert args.reward_mode == BENCHMARK_REWARD_MODE
    assert args.observation_version == BENCHMARK_OBSERVATION_VERSION
    assert str(args.ret_unified_calibration).endswith(
        "supply_chain/data/ret_unified_v1_calibration.json"
    )


def test_garrido_cf_policy_uses_exact_des_bypass_values() -> None:
    action = garrido_cf_policy("s2")(None, {})
    assert isinstance(action, dict)
    assert action["assembly_shifts"] == 2
    assert action["op3_q"] == CAPACITY_BY_SHIFTS[2]["op3_q"]
    assert action["batch_size"] == CAPACITY_BY_SHIFTS[2]["op7_q"]
    assert action["op3_rop"] == OPERATIONS[3]["rop"]
    assert action["op9_q_min"] == OPERATIONS[9]["q"][0]
    assert action["op9_q_max"] == OPERATIONS[9]["q"][1]
    assert action["op9_rop"] == OPERATIONS[9]["rop"]
