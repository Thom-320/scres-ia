# Paper 2 — Methods (draft v1, 2026-07-18)

*(Companion to `01_introduction_draft.md`. Every protocol element below is frozen in a
machine-readable contract committed before execution; file pointers are given inline so the
section can be verified — and later compiled — directly against the repository.)*

## 2.1 Base simulation and primary endpoint

The base model is a discrete-event simulation (SimPy) of the thirteen-operation military food
supply chain of Garrido-Ríos (2017), replicating the thesis's operations (Op1–Op13), risk
taxonomy, warm-up semantics, cap-60 backorder list with SPT/contingent-priority scheduling, and
demand process. Fidelity was established in two stages: behavioural reproduction of the thesis's
H2/H3 findings, and exact reproduction of the resilience formula on the author's original
workbooks — 0 mismatches over 47,546 formula cells given source snapshots
(`research/paper2_exhaustive_search/excel_metric_reaudit_20260713.json`).

The primary endpoint throughout is the thesis's own construct: canonical order-level resilience
`ReT = 1 − (Bt + Ut)/j` evaluated with request-snapshot semantics
(`ret_excel_request_snapshot_v2`; contract
`docs/RET_EXCEL_REQUEST_SNAPSHOT_V2_CONTRACT_2026-07-14.md`). Snapshot timing is supported by the
thesis's Annex B (a per-request barrier matrix carrying generation time and cumulative
backorders/losses). The one unresolved semantic — event ordering at identical timestamps — is
handled by a prespecified dual-semantics sensitivity analysis (both plausible orderings recomputed
on burned tapes); conclusions are claimed only under the primary convention, with survival of sign
and practical size reported. Tail behaviour (`CVaR10` of per-order ReT) is reported descriptively
with confidence intervals and is never a promotion gate; no deployment-safety claim is made.

## 2.2 Two-product extension (Program O physics)

The extension introduces two non-fungible ration classes, `P_C` and `P_H`, competing for the
shared Op5–Op7 assembly line (`supply_chain/program_o_full_des.py`, class
`ProgramOFullDESSimulation(MFSCSimulation)`). The weekly decision is the product mix
`k ∈ {0,1,2,3}`: how many of the next three production batches are allocated to `P_C`; committed
batches cannot be redirected (24 h activation latency). An episode spans eight weekly decisions
plus a clearance horizon. Demand labels follow a two-state Markov regime with persistence ρ and
dominant share s; the three evaluation cells are (ρ, s) ∈ {(.75,.90), (.90,.75), (.90,.90)}.
Downstream freight runs under fixed-clock physical semantics: scheduled missions depart loaded or
empty, so no policy can buy outcome with extra transport. A complete-substitution switch collapses
the two products into one and constitutes an exact mechanism null (measured H_PI = 0). An exact
transducer replicates the full DES for fixed calendars (parity certified over 25,177 episodes;
`contracts/program_o_exact_transducer_v1.json`), making the exhaustive comparator frontier
computable; every scientific verdict is additionally backed by direct full-DES replay of the
realized calendars.

## 2.3 Estimands: the four-level ladder

Let V(π) denote mean canonical ReT over an evaluation tape set.

- **Level 1 — physical opportunity:** `H_PI = V(π_PI) − V(σ*_OL)`, the per-tape clairvoyant best
  against the best fixed calendar, under a per-tape safe-selection oracle (resources and
  guardrails non-inferior).
- **Level 2 — observable conversion:** `H_obs = V(π_classical) − V(σ*_OL)` for the strongest
  frozen non-privileged classical controller.
- **Level 3 — learned adaptation:** `H_OL = V(π_RNN) − V(σ*_OL)`.
- **Level 4 — neural premium:** `Δ_N = V(π_RNN) − V(π*_classical)`.

Here `σ*_OL` is the best of all 4⁸ = 65,536 open-loop calendars **selected by mean across tapes,
never per-tape** (a per-tape maximum would be a clairvoyant comparator), and `π*_classical` is the
best of ten frozen classical controllers per cell (base-stock, max-pressure/hysteresis,
min-cost-flow variants, belief-MPC, belief-DP), selected the same way. A second, stricter reading
reports the single globally best classical controller across cells (universal bar), matching the
generality of the single learned policy.

## 2.4 Learner protocol

Ten RecurrentPPO policies (independent optimizer seeds 8101–8110) were trained for exactly
200,192 timesteps each (391 rollouts × 512 steps) on disjoint partitions of 250,240 stepped
training tapes (block 748100001–748350250, one fresh stochastic realization per episode, plus one
reset sentinel per seed), with the three cells alternating round-robin
(`contracts/program_o_ret_only_learner_v1.json`). The observation (21 features) contains only
non-privileged operational state — per-product on-hand inventory, locked (committed) pipeline,
backlog quantity, order count and maximum age, in-flight quantity, a frozen HMM belief summary
whose parameters are fixed across all cells (the true cell parameters are forbidden, so the
belief model is deliberately misspecified in two of the three cells), the previous action, and
temporal phase. Realized demand enters only through its operational traces (inventory draws and
backlog), never as an explicit demand window; the tape seed/hash, true cell parameters, latent
regime, and any future information are contractually forbidden and tested. Reward is the terminal canonical ReT only (no shaping). Training artifacts
(model SHA-256s, contract hash, git commit) are custody-manifested.

## 2.5 Evaluation protocol and integrity gates

Evaluation opens seed blocks in a one-shot ladder: burned development tapes for design, a
48-tape calibration block (7480001–7480048) opened once, and a virgin confirmation block that
opens only on a hash-bound authorization chain (calibration manifest → direct full-DES audit →
adjudication → independent sign-off); in the present study calibration did not authorize
confirmation and the virgin block remains permanently sealed. The evaluator
(`scripts/evaluate_program_o_ret_learner.py`, amendment v1.2) enforces fail-closed gates: (i)
action-trajectory feedback audit (≥8 distinct calendars per seed, modal fraction ≤ 0.50, ≥2
varying weeks); (ii) three information-replacement comparators executed and gated per learner
seed (modal, phase-only, frequency-matched); (iii) exact scheduled-resource equality across
learners, classical controllers and the full frontier (maximum absolute deviation must equal
zero); (iv) demand-ledger identities (generated = visible + omitted, exact; residuals ≤ 1e-8);
(v) SHA-256 custody manifests over all 144 raw evaluation matrices. An absent, empty, or
unexecuted component fails the gate — never defaults to pass (pinned by 23+ unit tests).

## 2.6 Statistical analysis

Primary inference uses paired per-tape contrasts with two-way bootstrap resampling over learner
seeds and tapes (10,000–20,000 resamples), comparator reselection inside every resample, and
studentized simultaneous max-t lower confidence bounds at familywise one-sided 95%. Promotion
thresholds were frozen before any seed opened: simultaneous LCB95 ≥ 0.01 for value estimands,
≥ 34/48 favorable tapes per cell, ≥ 8/10 positive learner seeds, guardrail non-inferiority at
−0.02, and exact resource equality. The prospective replication (Program Q,
`contracts/program_q_frozen_policy_replication_v1.json`) tests hypotheses generated by
calibration on entirely new tapes with the ten policies frozen by SHA: E1 replicated adaptation
(LCB95(H_OL) ≥ 0.01, all cells) and E2 the neural relation, graded as premium
(LCB95(Δ_N) ≥ +0.01), TOST equivalence (CI95(Δ_N) ⊂ [−0.01, +0.01]; equivalence is demonstrated
by two one-sided bounds, never inferred from non-significance), or a non-inferiority floor
(LCB95 ≥ −0.01 with ≥8/10 seeds above −0.01). The ±0.01 margin is the program's pre-existing
practical-effect threshold, frozen before the replication tapes exist. Sample size (256 tapes
per cell, CRN across cells; block 7490001–7490256) was selected by a frozen smallest-N rule
from a resampling power analysis of the calibration matrices that reproduced the full inference
pipeline — including comparator reselection inside every bootstrap resample, which raises the
simultaneous critical value to 3.20 (joint power 1.000 for E1, 0.876 for E2-equivalence at
N=256); an earlier approximation omitting reselection noise selected N=128 and is retained in
the record as a quarantined non-authoritative attempt. Replication integrity adds three frozen
Class-B non-inferiority guardrails against both comparator families at margin −0.02:
full-ledger ReT, quantity-weighted ReT, and worst-product fill; any guardrail failure forces
the compound STOP verdict regardless of endpoint outcomes. A mandatory computational benchmark
(per-decision latency median and p95, memory, parameters, planner cost, equal hardware)
quantifies amortization value separately from outcome value.

## 2.7 Governance and reproducibility

Every experiment is preregistered as a frozen, hash-addressed contract; sealed seeds open once
with no post-failure changes to thresholds, guardrails, comparators, or physics; terminal
verdicts (including failures) are committed and pushed with their raw artifacts in the same
session. Implementation and independent verification are split between two agents, and every
retraction or interpretation correction in the program's history is preserved as a dated,
committed document. The repository is public; scientific commits are pinned (learner terminal:
`821c8d8`) and evidence branches tagged. All randomness is seed-addressed; no wall-clock or
non-deterministic input enters any scientific path.
