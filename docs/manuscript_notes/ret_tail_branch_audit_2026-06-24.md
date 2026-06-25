# ReT_tail Branch and Lane Audit - 2026-06-24

## Scope

This audit answers the question: where is `ReT_tail`, what did the other branch
leave behind, and what can be safely reused in the current retained-vs-reset
Track A work?

No branch checkout or full merge was performed. The audit read:

- current branch: `main` at `2ad6a62`
- remote branch: `origin/codex/garrido-postfix-reruns` at `e901f86`
- detached review worktree: `/private/tmp/scres-ia-reruns-review`

## Branch / Worktree Map

Current visible branches after `git fetch --all --prune`:

```text
main
origin/main
origin/codex/garrido-postfix-reruns
```

Worktrees:

```text
/Users/thom/Projects/research/scres-ia      main @ 2ad6a62
/private/tmp/scres-ia-reruns-review         detached @ e901f86
```

The remote branch contains a large rerun/forensic lane, not just the tail reward.
The diff from `main` to `origin/codex/garrido-postfix-reruns` touches 101 files
and includes roughly 12.9k insertions. It should not be merged wholesale.

## Where `ReT_tail_v1` Lives

`ReT_tail_v1` is active on `origin/codex/garrido-postfix-reruns`, not on current
`main`.

Primary implementation:

```text
origin/codex/garrido-postfix-reruns:supply_chain/env_experimental_shifts.py
```

Relevant implementation points on that branch:

- `REWARD_MODE_OPTIONS` includes `ReT_tail_v1`.
- Defaults:
  - `RET_TAIL_W_SC = 0.30`
  - `RET_TAIL_W_RC = 0.60`
  - `RET_TAIL_W_CE = 0.10`
  - `RET_TAIL_CAP_KAPPA = 0.40`
  - `RET_TAIL_INV_KAPPA = 0.25`
  - `RET_TAIL_BOOST = 0.0`
  - `RET_TAIL_TRANSFORM = "identity"`
  - `RET_TAIL_GAMMA = 1.0`
  - `RET_TAIL_BETA = 2.0`
- The reward method is `_compute_ret_tail_v1`.
- The `step()` reward dispatcher has an explicit `elif
  self._canonical_reward_mode == "ReT_tail_v1"`.
- `out_info` always emits `ret_tail_step`, `ret_tail_base_step`,
  `ret_tail_components`, `ret_tail_service_continuity`,
  `ret_tail_recovery_containment`, `ret_tail_cost_efficiency`, and
  `ret_tail_stress`.

## Formula

The branch implementation defines:

```text
SC_t = 1 - new_backorder_qty / max(new_demanded, 1)
RC_t = 1 / (1 + pending_backorder_qty / D8_t)
CAP_EF_t = 1 - cap_kappa * (S_t - 1) / 2
INV_EF_t = 1 / (1 + inv_kappa * strategic_inventory / I1344_total)
CE_t = sqrt(CAP_EF_t * INV_EF_t)
R_base = SC_t^w_sc * RC_t^(w_rc * recovery_boost_t) * CE_t^w_ce
```

Then one optional monotone transform is applied:

```text
identity: R = R_base
power:    R = R_base^gamma
exp_norm: R = (exp(beta * R_base) - 1) / (exp(beta) - 1)
```

The crucial design choice is that `CE_t` is **not gated**. Inventory and extra
shifts always cost reward, including during disruptions. This directly addresses
the problem in `ReT_ladder_v1`, where efficiency is gated and can become too
weak under stress.

## Commits That Introduced the Tail Lane

Important commits on `origin/codex/garrido-postfix-reruns`:

```text
d4a7cdb Add ReT_tail_v1: tail/recovery-aligned reward with un-gated cost
f3d6fe8 Complete ReT tail reward audit contract
bc5e63a Add ReT tail reward tuning grid
ccefcf3 Tune ReT tail reward defaults
dccf46d Add ReT tail steepness ablations
3471d84 Add Track A preflight gate
6eb1eda Add Track A tail Kaggle screen
d0a628b Add Track A tail decision analyzer
09b55da Add Track A continuous_it_s screen kernel (sibling of tail-screen)
```

`d4a7cdb` is the small implementation commit. It changed:

```text
supply_chain/env_experimental_shifts.py
tests/test_ret_tail_v1.py
scripts/run_thesis_decision_ppo_smoke.py
scripts/build_tail_resilience_kernel.py
notebooks/scresia_tail_resilience.ipynb
```

This is the safest source for a selective port. Later commits add tuning,
preflight, Kaggle, and larger lane machinery.

## Documents / Audits Left Behind

The branch leaves several useful documents:

```text
docs/RET_TAIL_V1_TUNING_2026-06-17.md
docs/RET_TAIL_STEEPNESS_AUDIT_2026-06-17.md
docs/TRACK_A_PREFLIGHT_REVIEW_2026-06-17.md
docs/TRACK_A_EXHAUSTION_PLAN_2026-06-16.md
docs/TRACK_A_EXTENSION_IDEA_BANK_2026-06-17.md
docs/REWARD_DESIGN.md
```

Key claims from those docs:

- `ReT_tail_v1` was introduced because prior Track A rewards were too aligned
  with aggregate mean service and could miss tail/recovery performance.
- The selected defaults were the only candidate that passed both `increased`
  and `severe` full-panel static gates in the old audit:
  - `w_sc=0.30`
  - `w_rc=0.60`
  - `w_ce=0.10`
  - `cap_kappa=0.40`
  - `inv_kappa=0.25`
  - `tail_boost=0.0`
- The preferred steepness ablation was `power:1.25`, because it preserves static
  policy ordering and the `[0,1]` range.
- `exp_norm` passed the old gate but weakened correlations, so it remained
  secondary.
- The preflight gate required strict Track A variables only: common `I_{t,S}`
  plus shifts, no continuous buffer or Track B downstream actions.

## Scripts / Tests Left Behind

Useful code on the branch:

```text
tests/test_ret_tail_v1.py
tests/test_ret_tail_reward_tuning.py
tests/test_reward_surface_audit.py
scripts/tune_ret_tail_reward.py
scripts/reward_surface_audit.py
scripts/track_a_preflight_check.py
scripts/run_track_a_exhaustion_sweep.py
scripts/analyze_track_a_tail_screen.py
scripts/build_tail_resilience_kernel.py
```

Best tests to port selectively:

- registration: `ReT_tail_v1` in `REWARD_MODE_OPTIONS`;
- defaults equal the tuned constants;
- reward is finite and bounded;
- increasing `cap_kappa` / `inv_kappa` lowers reward for max-buffer + max-shift;
- `power` transform steepens reward without reordering;
- `exp_norm` is bounded and monotone.

## Compatibility Problems With Current Main

Do not merge the branch wholesale.

Main compatibility issues:

1. **Risk occurrence name drift.** The branch uses `thesis_periodic`; current
   code uses `thesis_window` and `legacy_renewal`. Current contract says
   `thesis_window` is paper-facing.
2. **Large lane drift.** The branch includes Track A tail, continuous `I_t,S`,
   cost-aware Garrido 2024, forensic probes, artifact quarantine, Kaggle kernels,
   and Track B edits.
3. **Current retained/reset lane is newer.** Current `scripts/evaluate_retained_reset_learning.py`,
   `scripts/pilot_learning_regime.py`, scenario tapes, `downstream_q_source`,
   and `PAPER_CONTRACT_2026-06-24.md` are not part of the older tail branch
   design.
4. **Different reward gate target.** The old tail audit optimized tail metrics
   such as `ret_p10_all`, `flow_fill_rate`, and `stockout_week_pct`. Current
   retained/reset work requires order-level Garrido ReT, service-loss area,
   backlog, cost/resource use, and Figure 6.2/Table 6.20 downstream-Q robustness.

## Current Status on Main

Current `main` plus local working tree contains tail documents. After this
audit, the tail idea has also been selectively ported as `ReT_tail_v2`.
`ReT_tail_v1` remains branch history; `ReT_tail_v2` is the current-contract
candidate.

Active current reward list:

```text
ReT_thesis
ReT_corrected
ReT_corrected_cost
ReT_unified_v1
ReT_seq_v1
ReT_cd
ReT_garrido2024_raw
ReT_garrido2024
ReT_garrido2024_train
ReT_ladder_v1
ReT_tail_v2
rt_v0
control_v1
control_v1_pbrs
ReT_cd_v1
ReT_cd_sigmoid
```

The selective port added:

```text
supply_chain/env_experimental_shifts.py
tests/test_ret_tail_v2.py
```

It also passed:

```text
pytest tests/test_ret_tail_v2.py tests/test_thesis_decision_env.py \
  tests/test_audit_thesis_reward_surface.py \
  tests/test_env.py::test_env_rt_v0_reward_mode_emits_components -q
```

and short reward-surface smokes for `ReT_tail_v2` under both
`downstream_q_source=figure_6_2` and `downstream_q_source=table_6_20`.

The full current/increased/severe rerun then showed:

- Figure 6.2: `ReT_tail_v2` ranked 11/16, gate `audit_only`, best policy
  `L1a_uniform_I336_S1`.
- Table 6.20: `ReT_tail_v2` ranked 8/16, gate `shortlist`, best policy
  `L1a_uniform_I504_S2`.
- Strict downstream-Q comparison: `ReT_tail_v2` did not pass because it was
  not shortlisted in both lanes and its best static policy changed.

So the port is technically valid, but the current parameterization is not the
paper-facing reward. It remains a tail/cost ablation or a candidate for
predeclared calibration.

## Recommendation

Keep the manual port; do not merge the branch.

Remaining safe path:

1. Re-run the current Track A reward-surface audit under:
   - `downstream_q_source=figure_6_2`
   - `downstream_q_source=table_6_20`
   - `risk_levels=current,increased,severe`
2. Compare `ReT_tail_v2` against `ReT_ladder_v1`, `ReT_cd_v1`, and `ReT_cd`.
3. Only if the current audit passes, include it in the retained-vs-reset
   reward bake-off.

The lowest-risk naming is `ReT_tail_v2`, because the surrounding fidelity
contract has changed (`thesis_periodic` -> `thesis_window`; retained/reset
learning tapes; Figure 6.2/Table 6.20 robustness). Calling it `v2` prevents us
from pretending the old June 17 audit automatically validates the current
June 24 contract.
