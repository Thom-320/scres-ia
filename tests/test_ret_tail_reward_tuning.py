from __future__ import annotations

from pathlib import Path

import pytest

import scripts.tune_ret_tail_reward as tuner


def test_generate_weight_grid_keeps_recovery_dominant_simplex() -> None:
    weights = tuner.generate_weight_grid(
        step=0.25,
        w_sc_min=0.25,
        w_rc_min=0.50,
        w_ce_min=0.25,
    )

    assert weights == [(0.25, 0.5, 0.25)]
    assert sum(weights[0]) == pytest.approx(1.0)


def test_parse_weight_triplets_normalizes_user_supplied_weights() -> None:
    assert tuner.parse_weight_triplets("2:6:2") == [
        (0.2, 0.6, 0.2),
    ]


def test_parse_exact_candidates_normalizes_weights_only() -> None:
    candidates = tuner.parse_exact_candidates("3:6:1:0.4:0.25:0")

    assert candidates == [
        tuner.Candidate(
            w_sc=0.3,
            w_rc=0.6,
            w_ce=0.1,
            cap_kappa=0.4,
            inv_kappa=0.25,
            boost=0.0,
        )
    ]


def test_select_candidates_spreads_across_grid() -> None:
    candidates = [
        tuner.Candidate(
            w_sc=0.1,
            w_rc=0.5,
            w_ce=0.4,
            cap_kappa=0.1,
            inv_kappa=0.25,
            boost=float(index),
        )
        for index in range(10)
    ]

    selected = tuner.select_candidates(candidates, 4)

    assert [candidate.boost for candidate in selected] == [0.0, 3.0, 6.0, 9.0]


def test_tuner_auditor_command_forwards_candidate_knobs(tmp_path: Path) -> None:
    args = tuner.build_parser().parse_args(
        [
            "--profiles",
            "increased",
            "--panel-cfis",
            "31",
            "--policy-set",
            "with_crossed",
            "--replications",
            "1",
        ]
    )
    candidate = tuner.Candidate(
        w_sc=0.2,
        w_rc=0.6,
        w_ce=0.2,
        cap_kappa=0.35,
        inv_kappa=0.75,
        boost=3.0,
    )

    command = tuner.build_auditor_command(
        args=args,
        candidate=candidate,
        out_root=tmp_path,
    )

    assert "--reward-mode" in command
    assert command[command.index("--reward-mode") + 1] == "ReT_tail_v1"
    assert command[command.index("--ret-tail-w-sc") + 1] == "0.2"
    assert command[command.index("--ret-tail-w-rc") + 1] == "0.6"
    assert command[command.index("--ret-tail-w-ce") + 1] == "0.2"
    assert command[command.index("--ret-tail-cap-kappa") + 1] == "0.35"
    assert command[command.index("--ret-tail-inv-kappa") + 1] == "0.75"
    assert command[command.index("--ret-tail-boost") + 1] == "3.0"
