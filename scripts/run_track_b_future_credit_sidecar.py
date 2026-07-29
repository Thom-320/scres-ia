#!/usr/bin/env python3
"""Track B PPO+MLP trained with future-credit reward shaping (Arm 1).

Isolated test: standard v7 observation, standard PPO+MLP architecture (no
belief encoder, no memory features) -- ONLY the training reward changes, via
``scripts.track_b_future_credit_reward.TrackBFutureCreditRewardWrapper``.
Evaluation always reports the unmodified Garrido Excel ReT
(``order_ret_excel_mean``); the shaped reward only ever affects what PPO
optimizes against during training.

This is deliberately NOT combined with the belief-encoder retargeting arm
(``scripts/build_track_b_v10_belief_pretrain_dataset.py --target-risk R22`` +
``scripts/run_track_b_belief_encoder_sidecar.py``) so the two ideas can be
attributed independently before considering a combination.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_smoke import build_parser as smoke_build_parser, run_smoke
from scripts.run_track_b_observation_ablation import OBS_ABLATION_CONFIGS
from scripts.track_b_future_credit_reward import (
    SUPPORTED_MODES,
    TrackBBeliefConditionedFutureCreditRewardWrapper,
    TrackBFutureCreditRewardWrapper,
)

ALL_MODES = SUPPORTED_MODES + ("belief_conditioned_pbrs", "belief_conditioned_tail_pbrs")


def build_parser() -> argparse.ArgumentParser:
    parser = smoke_build_parser()
    parser.description = "Track B PPO+MLP with future-credit reward shaping (Arm 1, or the belief-conditioned combination of Arm 1 + Arm 2)."
    parser.add_argument("--credit-mode", choices=ALL_MODES, default="ReT_excel_terminal_shaped")
    parser.add_argument(
        "--obs-config",
        choices=list(OBS_ABLATION_CONFIGS.keys()),
        default=None,
        help="Optional observation ablation contract; e.g. v10_no_forecast keeps v10 memory but masks explicit forecasts.",
    )
    parser.add_argument("--pending-scale", type=float, default=1_000_000.0)
    parser.add_argument("--lost-scale", type=float, default=100.0)
    parser.add_argument("--pbrs-alpha", type=float, default=1.0)
    parser.add_argument("--pbrs-beta", type=float, default=0.5)
    parser.add_argument("--pbrs-eta", type=float, default=0.05)
    parser.add_argument("--pbrs-kappa", type=float, default=0.2)
    parser.add_argument("--pbrs-rho", type=float, default=0.05)
    parser.add_argument("--pbrs-exposure", type=float, default=0.0)
    parser.add_argument("--pbrs-backlog-age", type=float, default=0.0)
    parser.add_argument("--pbrs-tail", type=float, default=0.0)
    parser.add_argument("--pbrs-gamma", type=float, default=0.99)
    parser.add_argument("--belief-encoder-path", type=Path, default=None)
    parser.add_argument("--belief-head-path", type=Path, default=None)
    parser.add_argument("--belief-target-index", type=int, default=0)
    parser.add_argument("--belief-base-rate", type=float, default=0.0607)
    parser.add_argument(
        "--belief-mask-forecast",
        action="store_true",
        help="Zero explicit forecast channels before feeding observations to the belief head.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.obs_config is not None:
        obs_config = OBS_ABLATION_CONFIGS[str(args.obs_config)]
        args.observation_version = obs_config.observation_version
        args._observation_wrapper = obs_config.wrapper

    if args.credit_mode in ("belief_conditioned_pbrs", "belief_conditioned_tail_pbrs"):
        if args.belief_encoder_path is None or args.belief_head_path is None:
            raise SystemExit(f"{args.credit_mode} requires --belief-encoder-path and --belief-head-path")
        wrapper_factory = functools.partial(
            TrackBBeliefConditionedFutureCreditRewardWrapper,
            encoder_path=args.belief_encoder_path,
            head_path=args.belief_head_path,
            belief_target_index=args.belief_target_index,
            belief_base_rate=args.belief_base_rate,
            pending_scale=args.pending_scale,
            lost_scale=args.lost_scale,
            pbrs_alpha=args.pbrs_alpha,
            pbrs_beta=args.pbrs_beta,
            pbrs_kappa=args.pbrs_kappa,
            pbrs_rho=args.pbrs_rho,
            pbrs_exposure=args.pbrs_exposure if args.credit_mode == "belief_conditioned_tail_pbrs" else 0.0,
            pbrs_backlog_age=args.pbrs_backlog_age if args.credit_mode == "belief_conditioned_tail_pbrs" else 0.0,
            pbrs_tail=args.pbrs_tail if args.credit_mode == "belief_conditioned_tail_pbrs" else 0.0,
            pbrs_gamma=args.pbrs_gamma,
            mode_label=args.credit_mode,
            belief_mask_forecast=args.belief_mask_forecast,
        )
    else:
        wrapper_factory = functools.partial(
            TrackBFutureCreditRewardWrapper,
            mode=args.credit_mode,
            pending_scale=args.pending_scale,
            lost_scale=args.lost_scale,
            pbrs_alpha=args.pbrs_alpha,
            pbrs_beta=args.pbrs_beta,
            pbrs_eta=args.pbrs_eta,
            pbrs_gamma=args.pbrs_gamma,
        )
    args._ablation_wrapper = wrapper_factory  # type: ignore[attr-defined]
    args.invocation = "python scripts/run_track_b_future_credit_sidecar.py " + " ".join(sys.argv[1:])
    summary = run_smoke(args)
    print(f"Wrote Track B future-credit sidecar bundle to {summary['artifacts']['summary_json']}")
    for row in summary["policy_summary"]:
        if row["policy"] == "ppo":
            print(
                f"ppo (future-credit={args.credit_mode}): "
                f"order_ret_excel={float(row['order_ret_excel_mean']):.6f}, "
                f"cost={float(row['assembly_cost_index_mean']):.3f}"
            )


if __name__ == "__main__":
    main()
