# Track A Reward Audit - 2026-06-24

## Scope

This note audits every active reward mode in `MFSCGymEnvShifts` plus the
historical `ReT_tail_v1` idea. It separates three questions that were getting
mixed together:

1. Is the reward implemented and runnable?
2. Does the no-training static surface point toward externally good policies?
3. Is it a defensible training objective for the retained-vs-reset learning
   claim?

The answer is not "use `ReT_cd` because it was robust." `ReT_cd` was the least
unstable candidate under the downstream-Q ambiguity, but it is not the strongest
training reward by signal quality.

## Evidence Used

- Active reward list from `supply_chain/env_experimental_shifts.py`:
  `REWARD_MODE_OPTIONS`.
- Full no-training Track A reward-surface audit:
  - Figure 6.2 lane:
    `outputs/benchmarks/thesis_reward_surface/track_a_full_figure_6_2_20260624`
  - Table 6.20 lane:
    `outputs/benchmarks/thesis_reward_surface/track_a_full_table_6_20_20260624`
  - Downstream-Q comparison:
    `outputs/benchmarks/downstream_q_sensitivity/track_a_full_downstream_q_20260624`
- Minimal implementation smoke run on 2026-06-24: all 15 active reward modes ran
  for both downstream-Q sources without non-finite rewards.
- Post-audit selective port on 2026-06-24: `ReT_tail_v2` was added as a current
  Track A candidate and passed unit tests plus a short reward-surface smoke
  under both Figure 6.2 and Table 6.20 downstream-Q sources.
- Full current/increased/severe reward-surface rerun with `ReT_tail_v2`:
  - Figure 6.2:
    `outputs/benchmarks/thesis_reward_surface/track_a_full_figure_6_2_ret_tail_v2_20260624`
  - Table 6.20:
    `outputs/benchmarks/thesis_reward_surface/track_a_full_table_6_20_ret_tail_v2_20260624`
  - Downstream-Q comparison:
    `outputs/benchmarks/downstream_q_sensitivity/track_a_full_ret_tail_v2_downstream_q_20260624`
- Historical `ReT_tail_v1` tuning notes:
  `docs/RET_TAIL_V1_TUNING_2026-06-17.md` and
  `docs/RET_TAIL_STEEPNESS_AUDIT_2026-06-17.md`.

## Current Active Rewards

At the start of the audit, the environment exposed 15 reward modes:

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
rt_v0
control_v1
control_v1_pbrs
ReT_cd_v1
ReT_cd_sigmoid
```

After the selective port, the current working tree also exposes
`ReT_tail_v2`. It is intentionally not called `ReT_tail_v1`, because the old
branch audit used an earlier `thesis_periodic` contract while the current paper
lane uses `thesis_window`, Figure 6.2/Table 6.20 downstream-Q controls, and the
retained-vs-reset learning harness.

## Static Reward-Surface Ranking

The table below reports the relevant no-training static audit result. Lower rank
is better. "Best stable?" means the best static policy selected by the reward
matched under Figure 6.2 and Table 6.20.

| reward | Figure rank | Table rank | Best stable? | Figure gate | Table gate | Audit read |
|---|---:|---:|---|---|---|---|
| `ReT_ladder_v1` | 1 | 2 | no | shortlist | shortlist | Best active Figure reward; high external alignment; downstream-Q best policy shifts from S3 to S2. |
| `ReT_cd_v1` | 2 | 3 | no | shortlist | shortlist | Very high external correlation; clean but lacks cost/inventory efficiency. |
| `ReT_cd_sigmoid` | 3 | 6 | no | shortlist | shortlist | Works empirically, but the implementation itself documents sigmoid compression as a bad scale choice. |
| `control_v1` | 4 | 4 | no | audit_only | shortlist | Good operational signal, but current weights choose low-cost/high-buffer static policies in Figure. |
| `control_v1_pbrs` | 5 | 5 | no | audit_only | shortlist | Same as `control_v1`; PBRS is shaping, not a better objective. |
| `rt_v0` | 6 | 8 | no | shortlist | shortlist | Legacy weighted sum; ugly scale and weak interpretability. |
| `ReT_cd` | 7 | 1 | yes | shortlist | shortlist | Least unstable across downstream-Q, but Figure correlations are mediocre and it strongly rewards S3 capacity. |
| `ReT_seq_v1` | 8 | 9 | no | audit_only | shortlist | Historical primary, but no inventory cost; prone to over-buffering/cheap S1. |
| `ReT_unified_v1` | 9 | 7 | no | audit_only | shortlist | Conceptually clean, but current calibration points to S1/S2 and remains too cost-gated. |
| `ReT_garrido2024_train` | 10 | 10 | no | shortlist | audit_only | Useful cost-aware ablation, not stable enough as primary. |
| `ReT_garrido2024_raw` | 11 | 11 | yes | audit_only | audit_only | Evaluation/audit family; poor Track A training signal. |
| `ReT_corrected` | 12 | 12 | no | audit_only | audit_only | Corrected diagnostic, still piecewise and S1-biased. |
| `ReT_corrected_cost` | 13 | 13 | no | audit_only | audit_only | Alias of `ReT_corrected`; not a distinct candidate. |
| `ReT_thesis` | 14 | 14 | no | negative_control | negative_control | Keep as negative control/reporting proxy only. |
| `ReT_garrido2024` | 15 | 15 | yes | audit_only | audit_only | Paper-facing audit index, not a training reward. |

## Reward-by-Reward Audit

### `ReT_thesis`

Verdict: reject as training reward; keep as negative control.

It is a step-level approximation of the thesis' order-level ReT logic, not the
true order-level thesis metric. It is discontinuous and selects low-cost S1
policies in both downstream-Q lanes. Use order-level `compute_ret_per_order` for
thesis evaluation instead.

### `ReT_corrected` / `ReT_corrected_cost`

Verdict: reject as training reward; keep as diagnostic.

The autotomy correction removes one local non-monotonicity, but the reward is
still a step-level piecewise proxy and remains S1-biased. `ReT_corrected_cost`
is only an alias, not a separate idea.

### `ReT_unified_v1`

Verdict: promising idea but not primary in current form.

It uses service, recovery containment, and gated cost efficiency. The issue is
that cost is suppressed during bad service/recovery states, so the static
surface still tends toward edge policies. It is better as a design ancestor for
a new reward than as the immediate frozen training reward.

### `ReT_seq_v1`

Verdict: do not use as primary.

It is smooth and thesis-motivated, but the only cost term is shift cost.
Inventory has no holding or resource penalty. In Track A, where inventory buffer
is one of the two decision variables, this creates a reward-hacking path:
increase buffers cheaply, then avoid shifts. Historical Track A notes already
flag this failure mode.

### `ReT_ladder_v1`

Verdict: best active reward if we must choose from current code today.

It has the best Figure 6.2 diagnostic score and remains second under Table
6.20. It includes service continuity, backlog recovery, and a strategic-buffer
plus shift efficiency term. This is the most thesis-aligned active dense reward.

Weakness: efficiency is gated and lightly weighted, so it may still under-price
capacity/inventory when the agent can exploit them. Its best static policy is
not stable across the downstream-Q ambiguity (`I336_S3` under Figure 6.2 versus
`I504_S2` under Table 6.20). That makes it a good primary candidate under the
Figure 6.2 paper-facing source, but not a fully robust conclusion by itself.

### `ReT_cd`

Verdict: useful robustness reward, not the best primary reward.

This is why it was selected earlier: it was the only shortlisted reward whose
best static policy matched under both downstream-Q sources (`I336_S3`). That is
a robustness argument, not a quality argument. Under Figure 6.2 it ranked only
7th and had weaker external correlations than `ReT_ladder_v1` and `ReT_cd_v1`.
It also gives positive reward to spare capacity and only a mild inverse-cost
penalty, so it naturally likes S3. The retained-vs-reset pilot under `ReT_cd`
showed only a tiny positive gap, not a compelling reward signal.

### `ReT_cd_v1`

Verdict: strong ablation candidate; not enough as the primary reward.

It is a clean continuous bridge: fill rate times availability. The audit
correlations are excellent. But it has no explicit capacity, inventory, or
switching/resource cost. It may train a service-maximizing controller, not a
resource-aware resilience policy.

### `ReT_cd_sigmoid`

Verdict: reject except as a documented scale ablation.

The implementation explains the problem: applying sigmoid to a log score whose
inputs are already in `(0, 1]` compresses the best possible reward to 0.5. It
passed the static shortlist, but the scale is conceptually wrong for training.

### `ReT_garrido2024_raw`

Verdict: reject as Track A training reward.

It is closer to the 2024 factory-resilience index, but its static correlations
with Track A external outcomes are weak or negative under Table 6.20. It is also
cumulative and macro-index-like, which makes it less useful as a dense stepwise
control signal.

### `ReT_garrido2024`

Verdict: keep as audit/evaluation index only.

The sigmoid 2024-style index is useful to report as an adjacent resilience
measure. It ranked last in the static Track A reward audit, so it should not be
the training reward.

### `ReT_garrido2024_train`

Verdict: ablation only.

The reduced cost coefficient was introduced to avoid full S1/S3 collapse. It
shortlisted under Figure 6.2 but failed the Table 6.20 gate. It is useful to
show that a Garrido-2024-inspired cost-aware training reward was considered, but
not strong enough to freeze.

### `rt_v0`

Verdict: legacy baseline only.

It combines recovery, inventory, service loss, and shifts, but its scale is ugly
and its interpretation is weak. Keep it only to understand historical runs.

### `control_v1`

Verdict: keep as operational baseline; do not use current weights as primary.

This is the cleanest engineering reward family: service loss plus shift cost
plus disruption penalty. But current `control_v1` lacks an explicit inventory
target/use penalty and selected an edge S1 policy under Figure 6.2. The idea is
good; the current contract is incomplete for Track A.

### `control_v1_pbrs`

Verdict: shaping ablation only.

PBRS can help learning speed, but it does not fix the underlying objective. It
inherits `control_v1`'s missing inventory/resource accounting.

### `proxy`

Verdict: deprecated.

The old `MFSCGymEnv` exposes `proxy` and `rt_v0`, but this is not the Track A
thesis-factorized retained-learning lane. Do not use it for paper evidence.

## `ReT_tail_v2` / Historical `ReT_tail_v1`

Verdict: best overall reward idea; now implemented as a candidate, but not yet
frozen for paper-facing training.

The old `ReT_tail_v1` design is closer to what we now need than `ReT_cd`:

```text
SC_t = 1 - new_backorder_qty / max(new_demanded, 1)
RC_t = 1 / (1 + pending_backorder_qty / D8_t)
CAP_EF_t = 1 - cap_kappa * (S_t - 1) / 2
INV_EF_t = 1 / (1 + inv_kappa * strategic_inventory / I1344_total)
CE_t = sqrt(CAP_EF_t * INV_EF_t)
R_t = SC_t^0.30 * RC_t^0.60 * CE_t^0.10
```

Why this is stronger:

- It makes recovery/backlog containment primary, which is closer to the
  retained-learning question than one-step fill rate alone.
- It includes both inventory and capacity efficiency.
- Its cost term is not hidden behind an overly weak gate.
- The prior full-panel static gate found an interior/low-resource policy rather
  than a raw S3 or max-inventory collapse.

Governance status: the reward has now been reintroduced as `ReT_tail_v2`, with
the old tuned defaults preserved as the initial candidate. A short smoke showed
it enters the current reward-surface pipeline and avoids an immediate S3
collapse under both downstream-Q sources. That smoke is not a selection result;
the full current/increased/severe audit is still required before it can replace
`ReT_cd` or `ReT_ladder_v1` in any training claim.

## Recommendation

Do not treat `ReT_cd` as the best reward. It is the strict downstream-Q
robustness reward: the only reward whose shortlist status and best static
policy remained stable across Figure 6.2 and Table 6.20 in the full rerun.
That makes it defensible for a conservative smoke, not automatically best for
learning.

Use this hierarchy:

1. **Best Figure 6.2 signal:** `ReT_ladder_v1`, with `ReT_cd_v1` close behind.
2. **Best downstream-Q robustness lane:** `ReT_cd`.
3. **Best service/availability ablation:** `ReT_cd_v1`.
4. **Best operational baseline family:** `control_v1`, but only after adding
   explicit inventory/resource accounting.
5. **Best tail/cost family to calibrate, not freeze yet:** `ReT_tail_v2`.

`ReT_tail_v2` did not become the winning reward. Under Figure 6.2 it ranked
11/16, was marked `audit_only`, and selected `I336_S1`; under Table 6.20 it
ranked 8/16, passed `shortlist`, and selected `I504_S2`. That is useful:
the formula prevents S3 collapse, but the current cost penalty is too strong
or too sensitive to the downstream-Q interpretation for paper-facing training.

## Proposed Reward Bake-Off Before Any Big Run

Run a small training-only reward bake-off before cloud-scale retained/reset:

```text
Candidates:
1. ReT_ladder_v1
2. ReT_cd_v1
3. ReT_cd
4. control_v1_pbrs
5. ReT_tail_v2, only as a calibrated tail/cost ablation

Evaluation:
- Figure 6.2 paper-facing source
- Table 6.20 robustness panel
- common training tapes only
- retained-vs-reset micro-smoke
- report order-level ReT, service-loss area, fill rate, backlog, shift-hours,
  inventory target/use, and cost/resource proxy
```

Acceptance rule:

- reject rewards that win only by choosing S1/S3 trivially;
- reject rewards that improve retained-vs-reset only by spending materially more
  surge or inventory;
- shortlist rewards where retained-reset improves order-level ReT and service
  loss in the same direction;
- only then freeze one reward for the powered run.

Immediate interpretation after the full rerun:

- conservative smoke: `ReT_cd`, because it is downstream-Q stable;
- science-first Figure 6.2 smoke: `ReT_ladder_v1` and `ReT_cd_v1`, because
  they have the strongest external alignment;
- tail/cost diagnostic: tune or ablate `ReT_tail_v2`, but do not promote it
  unchanged.
