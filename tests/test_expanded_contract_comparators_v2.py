from __future__ import annotations

from supply_chain.expanded_contract_controllers_v2 import (
    ALL_POSTURES,
    NODES,
    ProjectedDDMRPController,
    nearest_posture,
    posture_targets,
)
from scripts.run_expanded_contract_comparators_v2 import (
    make_replay_sim,
    materialize_tape,
    replay_prefix,
    splice_tapes,
    state_hash,
)
from scripts.fold_v2_arms_into_panel import PHYSICAL_GATE_KEYS


def test_static_domain_is_complete_and_node_independent() -> None:
    assert len(ALL_POSTURES) == 6**3 == 216
    assert len(set(ALL_POSTURES)) == 216
    assert (168, 0, 336) in ALL_POSTURES
    assert tuple(posture_targets((168, 0, 336))) == NODES


def test_projection_returns_common_domain() -> None:
    posture = nearest_posture(
        {"op3_rm": 20_000.0, "op5_rm": 0.0, "op9_rations": 40_000.0}
    )
    assert posture in ALL_POSTURES
    assert posture[1] == 0


def test_ddmrp_uses_window_and_projects_to_common_domain() -> None:
    tape = materialize_tape(1_499_001, 4 * 168.0, "R1r")
    sim = make_replay_sim(
        seed=int(tape["seed"]),
        horizon=4 * 168.0,
        family="R1r",
        tape=tape,
    )
    controller = ProjectedDDMRPController(window_days=28.0)
    targets = controller.act(sim, 0)
    posture = tuple(controller.last_diagnostic["posture"])
    assert posture in ALL_POSTURES
    assert targets == posture_targets(posture)
    for node in NODES:
        assert "on_order" in controller.last_diagnostic["nodes"][node]
        assert "qualified_spikes" in controller.last_diagnostic["nodes"][node]


def test_hybrid_replay_matches_realized_prefix_hash() -> None:
    horizon = 8 * 168.0
    actual = materialize_tape(1_499_101, horizon, "R1r")
    future = materialize_tape(1_509_101, horizon, "R1r")
    prefix = [(168, 0, 336)]

    actual_sim = replay_prefix(
        tape=actual,
        seed=int(actual["seed"]),
        horizon=horizon,
        family="R1r",
        prefix=prefix,
        epoch_hours=4 * 168.0,
    )
    hybrid = splice_tapes(actual, future, 4 * 168.0)
    hybrid_sim = replay_prefix(
        tape=hybrid,
        seed=int(actual["seed"]),
        horizon=horizon,
        family="R1r",
        prefix=prefix,
        epoch_hours=4 * 168.0,
    )
    assert state_hash(hybrid_sim) == state_hash(actual_sim)


def test_corrective_fold_gates_physics_not_historical_rpj_metric() -> None:
    assert PHYSICAL_GATE_KEYS == (
        "flow_fill_rate",
        "lost_orders",
        "delivered_rations",
        "unresolved",
        "strategic_injected",
        "terminal_stock",
    )
    assert "ret_excel" not in PHYSICAL_GATE_KEYS
