# Program O — terminal-outcome certificate (PI synthesis)

**Date:** 2026-07-15
**Status:** `PENDING_STATE_RICH_H_OBS_FIT` — the decisive experiment is running; this
document consolidates the machine-verified state and freezes the exact criteria for
each terminal branch. It is finalized the moment the H_obs fit resolves.

This is the PI-level synthesis over Program O. It references, and does not duplicate,
the committed operational artifacts. It exists to make the terminal outcome — positive
Paper 2 environment **or** quantitative boundary certificate — a mechanical consequence
of one pending result rather than a judgement call.

---

## 1. Established, custody-verified quantitative ceiling (H_PI)

Program O exposes a **material, resource-matched, causally-controlled full-DES
perfect-information gap** under the disclosed two-class (P_C, P_H) non-fungibility
extension.

| Quantity | Value | Source |
|---|---|---|
| safe H_PI (holdout, least-favorable cell) | **0.15151** | validation custody verdict |
| simultaneous safe LCB95 | **0.11562** | 10,000 paired bootstrap, frontier reselected |
| exact fungible-null H_PI (causal control) | **0.0** | metric-identical under fungibility |
| direct full-DES parity | 25,177 episodes | transducer ↔ SimPy, `parity_pass` |
| resource match | Δgross_production = 0; all charged/reserved resources equal | conservation_pass |

- Development seeds 7400049–72 (safe H_PI 0.1506); holdout validation seeds 7400097–7400120.
- Scientific commit `6ad6f10`; custody verdict committed `98ce2ce`;
  `scientific_result_sha256 = f5f2da8d035f5d33b82dec62383c56c164a0babe72a6e5d603b93868e2306fb1`
  (independently re-sealed by the PI agent against the remote manifest).
- Evidence: `results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json`
  (`PASS_PROGRAM_O_FULL_DES_HPI_TRANSLATION_CUSTODY_VERIFIED`).
- `claim_boundary.full_des_h_pi_established: true`. This is the first candidate in 18
  screened families whose clairvoyant ceiling is material **and survives real Op9–Op12
  buffering** (route recourse collapsed to ≤0.005 at the same test) **and** has an exact
  causal null.

**The ceiling was certified at conserved throughput** (the oracle reallocates the C/H
split, not the total). Any H_obs claim must be measured under the same condition.

## 2. What is refuted (label-only H_obs)

The minimal label-only HMM family gave strong *development* signal in the primary cell
(`belief_extreme_v1` init 2: Δ=0.0625 vs the full frontier; diagnostic reselected-frontier
LCB95 0.0407; run `program-o-hobs-fit-v1-20260715`) but is **not** H_obs:

1. **Auto-calibration excluded.** Giving `belief_extreme` the true (ρ, share) per cell
   changed **0 of 192** trajectories. Mechanism: the action is `sign(belief_c − 0.5)` and
   `predicted_share_c − 0.5 = (2·share − 1)(belief_c − 0.5)` is share-magnitude-invariant;
   symmetric transitions preserve sign. The cross-cell equity failure is decisional-structure,
   not parametric.
2. **Throughput leak.** Part of its visible-ReT gain came from more actual transport
   (+~388 vehicle-hours, +~20k rations), not allocation at conserved throughput.

Both point to a state-rich classical controller judged against a conserved-throughput
comparator — the operational gate below.

## 3. The pending decisive experiment (state-rich H_obs)

- **Operational gate:** `contracts/program_o_state_rich_comparator_fit_v1.json` +
  `supply_chain/program_o_state_rich.py` (frozen `5663c53`).
- **Independent convergent design review (PI):**
  `contracts/program_o_state_rich_hobs_prelearner_v1.json` +
  `docs/PROGRAM_O_STATE_RICH_HOBS_PREREGISTRATION_2026-07-15.md` (`fd29654`, `5e73fa7`).
  Two independently authored designs converged — a positive validity signal.
- **Finite family:** 10 enumerated configs across base-stock, max-pressure/hysteresis,
  min-cost-flow, belief-MPC (H3/4), bounded belief-DP. No continuous search, no post-fit
  retuning.
- **Pre-fit blockers, all cleared and audited:**
  - State-by-state parity of the observation vs direct SimPy (canary = `max_backlog_age`
    via true OAT/OPT, half-open) — **verified passing** (`test_operational_replay_state_matches_direct_full_des_events`, 3 passed, PI-run).
  - Four state placebos with paired LCB95; two state-perturbation counterfactuals on the winner.
  - Conserved real-use resource frontier as a hard gate.
- **Execution:** frozen `8d58815`, custody harness `041dcef`; running on VPS
  `program-o-state-rich-fit-v1-20260715` (producer PID=PGID=SID=969467), only burned fit
  tapes `7420001–48`. Sealed validation `7420049–96` untouched.

## 4. Terminal branches (frozen criteria)

### Branch A — PASS (fit clears, then sealed validation clears)
Fit condition: matched-frontier mean Δ ≥ 0.015, ≥34 favorable tapes, all guardrails
including conserved real-use, beats every placebo, state-dependence certificate, **and a
clean connected component of ≥3 of 4 cells**. → open sealed validation `7420049–96` once;
10,000 bootstrap; primary + ≥3 connected cells with simultaneous LCB95(H_obs) ≥ 0.01 and
every guardrail LCB/UCB rule.

→ **Terminal outcome: H_obs > 0 via classical observable control at conserved throughput.**
This is the positive precondition for Paper 2 and the first positive instance of the whole
search. It is **not** neural Paper 2: the passing controller becomes a mandatory comparator;
a learner must then beat the elementwise max over {frontier, base-stock, max-pressure,
min-cost-flow, belief-MPC, belief-DP, label-only} on a **fresh virgin** seed block (not the
742-series) with trajectory audits. Paper 3 (retained value) follows only after that.

### Branch B — STOP (any of: no headroom / resource-frontier confound / state-independent)
→ **Terminal outcome: quantitative boundary certificate.** The clairvoyant ceiling
(§1) is real, survives buffering, and has an exact causal null, yet no finite observable
controller converts it robustly at conserved throughput. This is the sharpest "when not to
train" instance and extends the Program D–K null pattern (including the Program K
cost-efficiency-not-resilience finding): perfect-information value without deployable,
resource-honest conversion.

## 5. Exact open Garrido question (non-blocking, either branch)

`Q13` in `research/paper2_exhaustive_search/garrido_face_validation_questions.md` (`7834ee1`):
does the MFSC have ≥2 mutually non-substitutable ration classes sharing the
capacity-constrained Op5–Op7 bottleneck with an uncertain, persistent, advance-observable
mix; and the real (share, persistence, setup, cadence). **Validates** → the result is an
MFSC-representative claim; **collapses** (fully substitutable / dedicated capacity /
deterministic mix) → the two-class construct reduces to the exact fungible null and Program
O is retired as an MFSC claim (the boundary/abstraction finding stands regardless). It does
not gate the internal research, which proceeds on the disclosed researcher extension.

## 6. Monitoring

- VPS watcher `969465` (concurrent process, 5 s) — producer/session liveness.
- Local monitor `b22f9vwps` (PI) — new commits + local custody verdict.
- VPS monitor `bjrzwuzlw` (PI) — state-rich fit `result.json` / terminal manifest
  (cell progress + PASS/STOP/exit).

On terminal, the PI agent independently retrieves + checksums the fit artifacts, verifies
η at conserved throughput, the connected component, placebo failure, and the
state-dependence certificate, then finalizes this certificate to Branch A or B and commits.
