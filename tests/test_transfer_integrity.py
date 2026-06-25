"""Harness-integrity gates for the corrected transfer protocol (audit track).

These validate that retention_transfer measures what it claims:
  * block-zero equality: at k=0 all arms are theta_0, so Delta R_0 == 0 exactly;
  * zero-update control: with no online training, every Delta R_k == 0.
If either fails, the harness is leaking signal and no MFSC result is trustworthy.
"""

from __future__ import annotations

import scripts.evaluate_retained_reset_learning as ev
from scripts.retention_transfer import run_seed


def _args(train_per_block: int):
    a = ev.build_parser().parse_args([])
    a.track = "a"; a.algo = "dqn"; a.decision_cadence = "weekly"
    a.reward_mode = "control_v1"; a.max_steps = 3; a.mask_preset = "none"
    a.pretrain_timesteps = 0; a.learning_starts = 10; a.buffer_size = 500
    a.rho_disruption = 0.85; a.rho_demand = None; a.regime_seed = 909
    a.surge_inertia = False; a.surge_budget_hours = float("inf"); a.surge_ramp_per_step = 1
    a.online_timesteps_per_cycle = train_per_block; a.retain_buffer = False
    return a


def test_block_zero_equality():
    a = _args(40)
    tape = ev.build_tape(a, 3, seed=909)
    r = run_seed(a, 1, tape, 3, 40)
    # retained == reset == frozen == theta_0 at block 0 -> identical cold eval.
    assert abs(r["mem"][0]) < 1e-9, r["mem"]
    assert abs(r["total"][0]) < 1e-9, r["total"]


def test_zero_update_control():
    a = _args(0)
    tape = ev.build_tape(a, 3, seed=909)
    r = run_seed(a, 1, tape, 3, 0)
    assert all(abs(x) < 1e-9 for x in r["mem"]), r["mem"]
    assert all(abs(x) < 1e-9 for x in r["total"]), r["total"]
