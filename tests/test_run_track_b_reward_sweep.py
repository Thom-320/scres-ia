from __future__ import annotations

from pathlib import Path

import scripts.run_track_b_reward_sweep as reward_sweep


def test_build_label_is_stable_and_sanitized() -> None:
    label = reward_sweep.build_label(
        label_prefix="track_b_reward_sweep",
        algo="ppo",
        reward_mode="ReT_garrido2024_train",
        train_timesteps=500_000,
    )

    assert label == "track_b_reward_sweep_ppo_ret_garrido2024_train_500k"


def test_build_run_args_forwards_reward_mode_and_algo(tmp_path: Path) -> None:
    args = reward_sweep.build_parser().parse_args(
        [
            "--algo",
            "recurrent_ppo",
            "--train-timesteps",
            "100000",
            "--eval-episodes",
            "5",
            "--seeds",
            "11",
            "22",
        ]
    )

    run_args = reward_sweep.build_run_args(
        label="track_b_reward_sweep_recurrent_ppo_ret_seq_v1_100k",
        runs_root=tmp_path,
        args=args,
        reward_mode="ReT_seq_v1",
    )

    assert run_args.label == "track_b_reward_sweep_recurrent_ppo_ret_seq_v1_100k"
    assert run_args.algo == "recurrent_ppo"
    assert run_args.reward_mode == "ReT_seq_v1"
    assert run_args.train_timesteps == 100_000
    assert run_args.eval_episodes == 5
    assert list(run_args.seeds) == [11, 22]
