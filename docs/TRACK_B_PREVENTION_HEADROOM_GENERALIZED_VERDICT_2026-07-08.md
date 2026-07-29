# Prevention headroom generalizes to a boundary result — full risk-roster sweep (2026-07-08)

## Question

`docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md` closed prevention for **R22 only**
under `track_b_v1`. But R22 is mechanistically the one risk with no causal path from any action
lever (exogenous recovery, direct transport-link knockout). Other risks have an in-principle
mediable channel: R23 (CSSU destruction blocks Op12 dispatch, but theatre stock delivered *before*
the event still serves demand — recon-confirmed in `supply_chain.py:2166-2222`), R12/R13 (upstream
contract/supplier stalls, bridgeable by op3/op5/op9 buffer levers), R21 (multi-op knockout,
downstream-prepositioning-mediable in principle). A `surge_inertia` lever already exists on the
Track B CLI that adds activation lag + a finite budget to shift changes — explicitly designed to
reward pre-positioning capacity before a shock. None of these were ceiling-tested before. This
sweep closes that gap using the same `scripts/audit_prevention_headroom_sweep.py` from the R22
verdict, extended additively with `--surge-inertia/--surge-ramp-per-step/--surge-budget-hours`
pass-through.

## Protocol

3 seeds × 8 eval episodes, `max_steps=104`, forced posture (`calm=-1, medium=0, max_prep=+1`) in
the 4 weeks before each anchor, local Excel ReT on orders exposed to the event window, isolated
real anchors + matched placebo anchors (same design as the R22 verdict). Reference policy: the
Case C reactive PPO checkpoint in all tiers (off-distribution for single-risk rosters, a known
caveat — see below).

## Results

| Tier | `enabled_risks` | Real anchors | Real positive | Placebo positive | DiD (local ReT) |
|---|---|---:|---:|---:|---:|
| R23-only (clean physics) | R23 | 45 | 0/45 (0%) | 0/96 (0%) | **0.0 exact** |
| R23 in Case C (mixed risks) | R22,R23,R24 | 28 | 2/28 (7.1%) | 4/96 (4.2%) | +0.00022 |
| R12-only (clean physics) | R12 | 65 | 0/65 (0%) | 0/96 (0%) | ~0.0 |
| R12 with R13 background | R12,R13 | 66 | 0/66 (0%) | 2/96 (2.1%) | ~0.0 |
| R21-only (clean physics) | R21 | 24 | 0/24 (0%) | 0/96 (0%) | **0.0 exact** |
| R23-only + `surge_inertia` | R23 | 45 | 0/45 (0%) | 0/96 (0%) | **0.0 exact** |

Four of six tiers are **exact** zero (bit-identical local ReT across calm/medium/max_prep for every
anchor, matching the R22-only signature already verified not to be a plumbing bug — cost varies,
local ReT does not). The two non-exact tiers (R12-only, R12+R13) show real-anchor rates
indistinguishable from their own placebo rates (0% vs 0-2%) — no signal.

The one tier with a nonzero real-anchor rate — R23 evaluated inside the full Case C environment
(R22+R23+R24 all active) — sits at 7.1% (2/28), *below* its own placebo rate is not the concern
(4.2%), but both are squarely inside the 3-19% noise band every other test this investigation has
produced, and it is directly contradicted by the clean-physics R23-only tier's exact zero. Reading:
this is the same contamination signature diagnosed in the original gate autopsy — evaluating an
R23 anchor inside a busy multi-risk environment lets nearby R22/R24 reactive activity leak into the
window. It is not independent evidence of an R23-specific preventive channel.

**`surge_inertia` — the one lever explicitly designed to reward pre-positioning — also gives an
exact zero.** Adding activation lag and a finite budget to shift changes does not create measurable
preventive value for R23 either.

## Verdict

**The preventive-headroom boundary generalizes across the tested risk roster (R12, R13-background,
R21, R22, R23) and across both standard and surge-inertia dispatch dynamics, under the `track_b_v1`
action contract.** This is a stronger, more defensible paper claim than the R22-only result: it is
not one risk's idiosyncratic physics, it is a property of the action contract itself relative to
this risk family. Combined with the clairvoyant-PPO null (`docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md`)
and Codex's independent event-tape Gate v2 null (`docs/TRACK_B_PREVENTION_GATE_V2_IMPLEMENTATION_2026-07-07.md`),
three independent methodologies now agree.

**Caveat to carry into the writeup:** all tiers reused the Case C reactive PPO checkpoint as the
reference policy, which was never trained on these single-risk rosters — an off-distribution
concern. Given the result is an *exact* zero (not merely small) in 4/6 tiers, and the response is
driven by forced actions rather than the reference policy's own choices in the pre-window, this
caveat is unlikely to change the conclusion, but it should be stated explicitly rather than
implied: "under an off-distribution reactive reference policy" qualifies the claim technically even
though the DiD design (real vs. placebo, same policy) controls for it to first order.

## Corroborating historical evidence (found while auditing past runs, 2026-07-08)

Two earlier, previously-undigested runs independently support the same conclusion via different
methods:

- **`outputs/experiments/track_b_oracle_resilience_metrics_2026-07-04/`**: an oracle-boost design
  (force shift/op10/op12 up by a fixed amount for an 8-week lead before R22/R24 events) shows
  `ret_excel_mean` moving from 0.0059205 (baseline) to 0.0059263-0.0059416 across boost conditions
  — a ~0.1-0.3% relative change, i.e. functionally flat. A third independent method (fixed-boost
  oracle rather than forced-posture-window or clairvoyant-observation) reaching the same null.
- **`outputs/experiments/track_b_event_resilience_purchase_vs_ppo_2026-07-04/`** and
  `..._vs_heur_disruption_2026-07-04/` (`docs/TRACK_B_EVENT_RESILIENCE_PURCHASE_VERDICT_2026-07-04.md`):
  a *different*, non-causal, complementary metric — does elevated posture correlate with better
  local outcomes around real R22/R24 events, compared to a cheap heuristic? PPO+MLP vs
  `heur_disruption_aware`: **74.0% (R22) / 72.9% (R24) positive rate** on local service continuity,
  with material backorder avoidance (+41-43k) and backlog-AUC reduction (+11-14k) per event. This
  is NOT a prevention claim (it doesn't test pre-event causality) — it is strong quantitative
  support for the *adaptive recovery / exposure reduction* mechanism, and directly strengthens the
  manuscript's operational-significance narrative. Recommend folding into Phase 3's mechanism
  section as a "local resilience purchase" table alongside the branch-shift exposure evidence.

## Artifacts

- `outputs/experiments/track_b_headroom_r23_only_2026-07-08/`
- `outputs/experiments/track_b_headroom_r23_case_c_2026-07-08/`
- `outputs/experiments/track_b_headroom_r12_only_2026-07-08/`
- `outputs/experiments/track_b_headroom_r12_r13bg_2026-07-08/`
- `outputs/experiments/track_b_headroom_r21_only_2026-07-08/`
- `outputs/experiments/track_b_headroom_r23_surge_inertia_2026-07-08/`
- Script: `scripts/audit_prevention_headroom_sweep.py` (extended additively with surge-inertia
  pass-through)

## Next steps

Per the approved Q1 plan: Phase 2 (gate redesign) remains skipped — the boundary is confirmed, not
reopened, by this sweep. Proceed to Phase 3 (manuscript sprint): the prevention-boundary subsection
should cite this generalized result (not just R22), and the event-resilience-purchase numbers
should be folded into the mechanism/operational-significance narrative.
