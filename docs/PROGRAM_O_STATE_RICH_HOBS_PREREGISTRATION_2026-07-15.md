# Program O — state-rich observable-headroom (H_obs) pre-registration

**Date:** 2026-07-15
**Contract:** `contracts/program_o_state_rich_hobs_prelearner_v1.json`
**Status:** DESIGN PRE-REGISTRATION. Awaits implementation + a tracked fit-only execution freeze. No sealed tape may open until then.
**Supersedes for H_obs:** `contracts/program_o_hobs_prelearner_v1.json` (label-only family).

Titles / abstract / hypotheses in English per the reporting convention; rationale below.

---

## 1. Why this gate exists (the refutation that motivated it)

Program O has a **custody-verified full-DES perfect-information ceiling** (`full_des_h_pi_established: True`; safe H_PI 0.15151, simultaneous safe LCB95 0.11562, exact fungible-null 0, parity + conservation PASS, holdout seeds 7420-adjacent). It has **not** established observable value.

The first observable attempt used a **label-only HMM family**. Its selected configuration (`belief_extreme_v1`, init 2) showed strong *development* signal in the primary cell (Δ=0.0625 vs the full open-loop frontier, diagnostic reselected-frontier LCB95 0.0407). It is **not** H_obs, for two independently sufficient reasons:

1. **Cross-cell robustness failure that is structural, not parametric.** A direct recomputation gave `belief_extreme_v1` the **true** (ρ, dominant_share) per cell — an oracle advantage no online estimator can beat — and it changed **0 of 192 trajectories**. The mechanism explains why: the policy decides by `sign(belief_c − 0.5)`, and

   > predicted_share_c − 0.5 = (2·share − 1)·(belief_c − 0.5),

   so the action is **invariant to the magnitude of `share`**, and the symmetric transition `(2ρ−1)·(b−0.5)` preserves sign. A self-calibrating estimator can therefore only match or add noise. **Parametric auto-calibration is excluded.** The equity failure (worst-product degradation in mismatched cells) needs a different *decision structure* — inventory, backlog, committed lots, and intermediate actions — not better parameters.

2. **Matched-throughput resource fairness.** The certified H_PI held `gross_production_quantity` delta = 0 and every charged/reserved resource equal: the clairvoyant value came from reallocating the C/H **split at fixed throughput**. The label-only policy instead obtained part of its visible-ReT gain by **moving more freight** (+~8 loaded departures, +~20k rations, +~388 realized vehicle-hours in the primary cell). That is a different, easier lever than the one H_PI certified. An H_obs claim measured against a lazier-transport static is not the same construct as H_PI, and η = H_obs/H_PI would be meaningless.

Both facts point to the same correct next gate: a **finite, pre-registered, non-anticipative state-rich classical controller family**, judged against a **throughput-matched frontier**.

## 2. What must be true for a legitimate positive

- Value comes from **allocation intelligence at matched throughput**, not from out-transporting the comparator (binding matched-throughput frontier; `STOP_RESOURCE_FRONTIER_CONFOUND` otherwise).
- Value is **robust**: a clean connected component of ≥3 of 4 cells, positive **and** guardrail-clean, first demonstrated **on fit** (precondition to opening validation) and then confirmed **out-of-sample** on sealed tapes.
- Value is **genuinely state-driven**, not a disguised calendar: state-perturbation counterfactuals must fire in the specified directions; beats modal, phase-only, and every information placebo; `STOP_STATE_INDEPENDENT_POLICY` otherwise.
- Value survives the **exact causal null** (full fungibility → H_obs = 0) and all conservation/parity checks.

## 3. Anti-overfitting discipline (the second added guardrail)

A state-rich family is far more expressive than 16 label rules, so it is the natural place to reintroduce the winner's curse. The contract forbids that by construction:

- **Finite, fully enumerated family: 18 configurations across 5 controllers**, each with ≤2 discrete hyperparameters on a frozen grid (base-stock cover; max-pressure weight × hysteresis; min-cost-flow holding ratio; integer-MPC horizon × service weight; belief-DP discretization). Internal solvers are exact/deterministic given the enumerated meta-parameters.
- **No continuous parameter search. No post-fit retuning.** One configuration is selected by a frozen rule on fit; that is the only degree of freedom spent.
- **Validation is opened once**, only for the single selected configuration, only if it already cleared ≥3 connected cells on fit.

## 4. Family, observation, comparators (summary; contract is authoritative)

- **Observation (non-anticipative, half-open barrier at `decision_start+168·t`):** regime belief (label-history filter), inventory position, backlog quantity and max age, committed pipeline lots, blocked production slots, week, remaining rights, prior actions. WIP counted once; no current/future request, seed, oracle, or true cell params. Row-hash per decision.
- **Controllers:** base-stock per product; max-pressure with hysteresis; exact min-cost flow; integer MPC (H=3–4); bounded belief-DP.
- **Binding comparator:** the maximum over the **throughput-admissible** static subset (calendars whose per-tape actual loaded departures, payload, and vehicle-hours each ≥ the policy's), reselected inside every bootstrap resample. The unconstrained 65,536-frontier advantage is reported as a diagnostic; the gap between them quantifies the throughput component.
- **Placebos (state-rich):** stale-state (t−2), cross-tape state (shift-17 wraparound, reused on validation), swapped-state (C↔H channels), no-state. The real-state controller must beat each with paired LCB95 > 0.

## 5. Gates

- **Fit (7420001–7420048, reused development tapes):** matched-throughput mean Δ ≥ 0.015, ≥34 favorable tapes, all mean guardrails including matched-throughput, beats every placebo, state-dependence certificate, **and a clean connected component of ≥3 cells**. Otherwise `STOP_NO_OBSERVABLE_HEADROOM` — validation is not opened.
- **Validation (7420049–7420096, sealed, opened once):** 10,000 paired bootstrap with the matched-throughput frontier reselected each resample; primary + ≥3 connected cells with simultaneous LCB95(H_obs) ≥ 0.01; every guardrail LCB/UCB rule; matched-throughput admissible on every scored tape; placebos fail; state-dependence, conservation, parity, and nulls pass.

## 6. What a PASS does and does not license

A PASS establishes **H_obs > 0 via classical observable control at matched throughput** — the precondition for Paper 2, and a positive instance the whole search has been seeking. It is explicitly **not** neural Paper 2:

- **Neural Paper 2 gate:** a learner must beat the elementwise maximum over {matched-throughput frontier, base-stock, max-pressure, min-cost-flow, integer-MPC, belief-DP, label-only interpretable} on a **fresh virgin** seed block (not the 742-series), with state-dependence and trajectory audits.
- **Paper 3 gate:** retained/persistent value vs reset/scratch/frozen, fresh campaigns, matched compute — only after neural Paper 2.

## 7. Honest terminal outcomes

- **PASS → validated classical H_obs**, then the neural Paper 2 gate above.
- **STOP_NO_OBSERVABLE_HEADROOM / STOP_RESOURCE_FRONTIER_CONFOUND / STOP_STATE_INDEPENDENT_POLICY** are all legitimate terminal results and would convert Program O into a **quantitative boundary certificate**: the perfect-information ceiling is real and survives buffering, but no observable controller converts it robustly at matched throughput — the sharpest possible "when not to train" instance, consistent with the Program D–K null pattern and the Program K cost-efficiency-not-resilience finding.

Garrido face-validation of the two-class construct (real ration non-fungibility, shares, BOM) runs **in parallel and non-blocking**: the thesis already discloses 21 real products compressed to one, which motivates the two-class extension independently; Garrido refines external calibration, he is not a gate.
