# Track A Selective Reruns Merge - 2026-06-24

This note records the integration boundary for the `codex/garrido-postfix-reruns`
review.

## Imported

- `sandbox/VERDICT.md`
- `sandbox/forensic_des.md`
- `sandbox/forensic_rl.md`
- `sandbox/probes/p0_warmup_det.py`
- `sandbox/probes/p_b1_risk_frequency.py`
- `sandbox/probes/p_table_6_10_replication.py`
- `sandbox/results/*.json`
- New dated Track A and reward-audit notes from `docs/`

These files are treated as evidence and historical audit artifacts from the
rerun branch.

## Not Imported

Core DES/Gym/training code from the rerun branch was not merged. In particular,
the rerun branch's global default migration for `warmup_trigger` and
`risk_occurrence_mode` was not applied here. Existing local Track A changes remain
the controlling implementation.

## Controlling Gate

Training remains blocked until the Track A reward-surface gate is run under both
downstream quantity lanes:

- thesis replication/training: `figure_6_2`
- robustness sensitivity: `table_6_20`

Only rewards whose rankings and best policies are stable across those two lanes
should be shortlisted for PPO or DQN smoke training.

## Local Checks

- `tests/test_thesis_faithful_lane.py`
- `tests/test_audit_thesis_reward_surface.py`
- `tests/test_thesis_decision_env.py`

The warm-up probe was also executed as a script smoke. It currently reproduces
the forensic finding that the engine's `op9_arrival` trigger occurs at 943 h,
104.2 h later than the thesis deterministic estimate of 838.8 h.
