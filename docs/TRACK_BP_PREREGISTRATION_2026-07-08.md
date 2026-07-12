# Track B-P Pre-registration: dynamic buffering under commitment physics (2026-07-08)

Status: **pre-registered before any Gate 0/1 result was read.** This document fixes the
design, the decision rules, and the stop conditions for the preventive re-opening, per the
C26 stop rule ("do not resume prevention work without a pre-registered environment change").

## Motivation (thesis-grounded)

Garrido (2017) §6.5.2 assumes a proactive-ONLY strategy: "the MFSC does not react post hoc
to the materialization of risks, but... adopts a proactive strategy (buffering strategy)."
His decision variable I_{t,S} (Table 6.16) is pre-positioned stock at Op3/Op5/Op9,
replenished every t∈{168,336,504,672,1344}h — quantity and cadence coupled ("strategic
inventory reserves"). Track A/B inverted that assumption: every `track_b_v1` dim is an
instantaneous, reversible multiplier, and under that physics the preventive ceiling is
exactly zero (clairvoyant PPO = ceiling = no gain; six exact nulls in the forced-prep
battery, `docs/TRACK_B_PREVENTION_HEADROOM_GENERALIZED_VERDICT_2026-07-08.md`).

Track B-P restores Garrido's lever but makes it dynamic: the policy chooses buffer targets
under a real replenishment lead time. This answers Garrido §8.6.1 (I×S synergy, open) with
a contract where anticipation is mechanically distinguishable from reaction.

## Environment change (the pre-registered difference vs. the closed battery)

- Contract `track_bp_v1` (11D): dims 1–8 = `track_b_v1` verbatim (instant levers KEPT —
  prevention must add value on top of the best reactive contract, not against a
  handicapped one); dims 9–11 = buffer target fractions for op3_rm/op5_rm/op9_rations,
  scaled by I_1344, applied via `sim.inventory_buffer_targets` with
  `inventory_replenishment_lead_time` ∈ {168h, 336h} (`supply_chain/track_bp_env.py`).
- Verified commitment physics (local probe, seed 7): targets raised at step t produce
  stock arrival at t + lead (one weekly step at 168h), not before.
- Verified lever liveness: op5_rm and op9_rations bind (top-ups land and persist);
  **op3_rm is a dead lever** under `kit_equivalent_order_up_to` flow (WDC working stock
  ~4M units ≫ any thesis-scale target). R12/R13-via-op3 channels are therefore expected
  weak for physics reasons, not contract reasons — recorded here BEFORE reading results.
- Observation: existing v10 (101 dims) already contains leak-free event clocks
  (R11/R13/R22/R23/R24), calendar phases (op1/op2 cycles), and windowed counts; the
  privileged 7 fields are masked via `v10_no_regime_forecast`
  (`scripts/run_track_b_observation_ablation.py`). NOTE: the canonical Track B policy
  already saw the R22/R23/R24 clocks and prevention was still null — signal existed,
  channel did not. The contract, not the observation, was the binding constraint.

## Gates and decision rules (fixed in advance)

### Pre-analysis calibration (recorded before any Gate 0/1 result was read)

A CRN-clean starvation probe (seed 42, R21 freq×8 impact×4, 104 weeks, identical
10-event calendar in both arms — `strict_exogenous_crn=True` is hardcoded in the env)
produced the **first non-identical pre-positioning outcome of the whole program**:
episode `ret_excel` 0.360998 (no buffers) vs 0.366523 (full buffers), **+1.53%
relative**, with min SB stock 0 vs 11,250 rations. Under unmodified `current`
intensity the same comparison is bit-identical: mean R21 outage (120h) never drains
the ~3-week working cover, so the lever cannot bind. Conclusion fixed in advance:
**Gate 0/1 tiers must run in a starvation regime** (impact multipliers that produce
multi-week outages), matching the thesis's own finding that buffer moderation (H2)
appears under increased risk, not current risk. This mirrors the closed battery's
Case C convention and is not a post-hoc choice.

### Gate 0 — forced-prep ceiling (eval-only)
`scripts/audit_prevention_headroom_sweep.py --env-factory track_bp --reference-policy
constant`, postures forced on the full 11D vector during the prep window (max_prep = full
buffers + max dispatch + S3; calm = zero buffers + min dispatch + S1), placebo anchors and
clean-physics tiers as in the closed battery. Tiers (anchor risk × lead time):

| Tier | enabled_risks | target | lead | rationale |
|---|---|---|---|---|
| G0-R21 | R21 (impact×3) | R21 | 168h | multi-op knockout, ~2-week outages; op5/op9 bridge |
| G0-R21-L2 | R21 (impact×3) | R21 | 336h | harder commitment |
| G0-R23 | R23 (impact×3) | R23 | 168h | forward-unit destruction; op9/forward stock |
| G0-R24 | R24 (freq×3) | R24 | 168h | priority demand surge; op9 absorbs |
| G0-R11 | R11 (freq×0.125, impact×8) | R11 | 168h | AL breakdown bridging; frequency reduced 8× so isolated anchors exist (natural R11 fires ~2×/week, which blankets every isolation window) — recorded before launch |
| G0-R22 | R22 (impact×3) | R22 | 168h | negative control (no buffer channel expected) |

Decision rule (per tier): headroom exists iff DiD(max_prep − calm | real − placebo) > 0
AND real positive-pair rate materially exceeds placebo rate (same promotion bar as the
closed battery: real ≥ 60% with placebo in the historical 3–19% band). All tiers null →
the prevention boundary generalizes to Garrido's own buffering physics; write the verdict
and STOP (no training).

### Gate 1 — static clock-policy oracle (CRN)
`scripts/run_track_bp_gate1_oracle.py`: never_prepared / always_prepared /
calendar_prepared (deterministic calendar timing), identical seeds, episode-level
`ret_excel` primary. Headroom = always−never or calendar−never > 0 with bootstrap CI95
excluding zero at ≥24 episodes. This bounds what a non-learning preventive policy already
achieves — the honest comparator for any learned policy.

### Gate 2 — training (ONLY if Gate 0 or Gate 1 positive)
PPO on `track_bp_v1` + `v10_no_regime_forecast`, 3-seed × 30k screen first; Real-KAN as
efficiency sidecar only. Conversion metric: fraction of the Gate-1 oracle headroom
captured. Track A warning stands: measurable headroom ≠ convertible headroom (regime
oracle was converted 0%).

## Interpretation guardrails

- Primary metric is `order_ret_excel_mean` / episode `ret_excel` (Excel convention),
  never `ret_thesis`.
- Buffer holding is intentionally unpriced in reward; if Gate 2 ever runs and wins, a
  holding-cost sensitivity (analogue of the dispatch λ_d) is mandatory before any claim.
- No claim of "anticipation" without the Gate-0 causal structure (real vs placebo, forced
  postures); no splice gates.
- The manuscript in `docs/manuscript_current/` is NOT touched by this lane.

## Falsifiers recorded in advance

1. If always_prepared ≡ never_prepared bit-identically at full horizon under frequent
   R21, the buffer containers do not actually shield deliveries — the lever would be
   cosmetic and the design dies at Gate 1 (check container→consumption plumbing before
   concluding anything about prevention).
2. If Gate 0 is positive only in tiers whose placebo rate is also elevated, treat as the
   known busy-environment contamination pattern, not headroom.
3. If Gate 2 trains and matches always_prepared but not calendar_prepared timing, the
   learned policy is doing static buffering, not anticipation — report as such.
