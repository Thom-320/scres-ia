# C&IE manuscript — structure plan (2026-07-21, pre-writing)

**Not the manuscript. This is the section division + artifact mapping + table/figure plan.**
Target: Computers & Industrial Engineering (primary). This is the Program-Q / four-level-ladder
manuscript, reframed from "a decomposition that found a null" → **"a computational decision
framework for when learned adaptive control is worth deploying over structured control"** (the
frame C&IE's scope rewards: originality in problem/tool/approach/result, not a new algorithm).

## The reframe (the single most important structural decision)

Current drafts lead with the ladder as a *finding*. C&IE version leads with the ladder as a
**method for a decision**: *given a validated DES supply chain, how do you decide computationally
whether a learned controller adds resilience value over structured control?* The RL≈MPC tie stops
being a disappointing null and becomes the **demonstration of the framework's discriminating
power** — it separates "learning happened" (Level 3, +0.06–0.10 over all open-loop) from "learning
beat structure" (Level 4, ≈0), a distinction weak-baseline RL-in-DES papers cannot make.

Two things this session added that MUST enter the manuscript (the current drafts predate them):
1. **The robustness campaign** — the tie survives 5 independent adversarial tests (prospective
   replication, a different architecture, warm-start from three initializations, a reward-geometry
   factorial, and an off-policy learner). This is the paper's biggest strengthening; it converts
   "no premium" into "no premium, and here is why it is not an artifact of any fixable RL choice."
2. **The mechanism** — the decision is belief-insensitive (better inference changes the optimal
   action in <5% of states) and the static top is flat; structured control already sits at the
   observable frontier. This is the *causal explanation* a reviewer asks for.

## Garrido alignment (cite as motivation, do NOT inherit v0)

Frame the contribution as **delivering Garrido et al. (2024)'s proposed bridge**: an AI algorithm
that converts the open-loop DES into a closed-loop system (RL, one of the three methods Garrido
names). We deliver it and measure it rigorously. We explicitly do NOT claim the v0 draft's
unsupported promises (predictive accuracy, path-dependency, cumulative cross-campaign learning);
cross-campaign Supply-Chain-Learning is named as the future-work upgrade.

## Section division (target ~9–10k words main text)

**Title** (decision-framework framing; pick at writing):
- "Deciding when learned control adds resilience value: an exact-frontier falsification benchmark
  on a validated military food supply chain"
- alt: "When does adaptive control beat structured control? A four-level benchmark for
  learned resilience in a full discrete-event supply chain"

**Abstract (~200 words).** Computerized resilience models often expand an agent's action space
without testing whether the added adaptivity is operationally useful. We propose a four-level
falsification benchmark that localizes adaptive value — physical opportunity, observable
conversion, learned adaptation, structural premium — with an exact complete open-loop frontier as
the answer key and fail-closed evaluation. Applied to a thesis-grounded 13-operation military food
supply chain, a recurrent learner acquires genuine closed-loop adaptation (beats all 65,536
open-loop schedules by 0.06–0.10 canonical ReT) yet shows no premium over structured belief-based
control; the equivalence survives five independent adversarial tests. The deployable value is
amortization (a single forward pass vs online planning), not a resilience premium. Contribution: a
computational design-and-falsification framework for deciding when resilience adaptivity belongs in
online control and when structured control already sits at the frontier.

**1. Introduction (~1200 w).**
- Hook: Garrido's closed-loop gap — open-loop DES cannot learn from experience; AI proposed as the
  bridge. But RL-in-DES resilience papers benchmark against weak open-loop baselines, so they
  cannot distinguish *acquired adaptation* from *rediscovering a good fixed schedule*.
- The decision problem: when is a learned controller worth deploying over structured (OR) control?
- Contributions (bulleted, C&IE-shaped): (i) the four-level falsification ladder; (ii) the exact
  complete-frontier + fail-closed adversarial benchmark; (iii) application to a validated military
  DES; (iv) a five-test robustness campaign; (v) an amortization analysis. Delivering Garrido's
  closed-loop bridge; no v0 overclaims.
- Roadmap.

**2. Problem setting and environment (~1000 w).**
- The 13-operation military food supply chain (thesis-grounded reconstruction; fidelity: 0 of
  47,546 formula cells mismatched).
- The two-product non-fungible extension sharing the Op5–Op7 line; weekly product-mix decision over
  8 weeks → exact complete open-loop space 4⁸ = 65,536.
- Primary endpoint: canonical thesis ReT (`ret_excel_request_snapshot_v2`), thesis-faithful.
- Regime-switching demand cells (ρ, s).

**3. The four-level falsification framework (~1800 w — the methodological core).**
- 3.1 The four estimands as a decision ladder: H_PI (physical opportunity) → H_obs (observable
  classical conversion) → H_OL (learned adaptation vs complete frontier) → Δ_N (structural
  premium). Each falsifiable separately; the ladder localizes *where* value lives or dies.
- 3.2 The exact complete open-loop frontier (65,536, exact transducer, parity-certified) as the
  answer key — never read during search.
- 3.3 Comparators: best static by mean (never per-tape); belief-MPC and the classical family;
  reselection inside every bootstrap resample.
- 3.4 Fail-closed evaluation: exact resource equality, information placebos, trajectory-feedback
  audit, direct full-DES replay.
- 3.5 Statistical machinery: two-way bootstrap over seeds×tapes, studentized simultaneous max-t.

**4. Experimental design (~1200 w).**
- 4.1 Learner protocol (RecurrentPPO, seeds, disjoint training tapes, terminal ReT reward).
- 4.2 Prospective replication (Program Q): frozen-by-hash policies, N=256 power-selected, sealed
  one-shot block, graded outcomes.
- 4.3 The robustness campaign (the adversarial battery): different architecture (transformer);
  warm-start from scratch/static/MPC with checkpoints; reward-geometry factorial; off-policy
  (QR-DQN); static-search benchmark (CEM vs RL). Each targets one objection.
- 4.4 Governance: preregistered hash-addressed contracts, sealed seeds, custody manifests.

**5. Results (~2500 w).**
- 5.1 Physical opportunity is material and mechanism-specific (H_PI = 0.152; exact fungibility
  null = 0).
- 5.2 Observable classical control converts most of it (H_obs LCB95 +0.043/+0.059/+0.066; placebos
  beaten; worst-product tail reported honestly).
- 5.3 A recurrent policy learns the adaptation from scratch (H_OL +0.06–0.10, beats the complete
  frontier, 10/10 seeds; the **scratch learning curve** figure — converges to structured control
  from below, no drift). = Garrido's closed-loop bridge, delivered and measured.
- 5.4 No structural premium (Δ_N ≈ 0; TOST-equivalent).
- 5.5 **The equivalence is robust** — the five-test matrix: prospective replication, architecture,
  warm-start, reward, off-policy. Each kills a specific objection (not undertraining / init /
  reward / architecture / off-policy). This is the subsection that makes the null unimpeachable.
- 5.6 **Why** — the decision is belief-insensitive (<5% action change under better inference); the
  static top is flat (top-38 within 0.0025); structured control sits at the observable frontier.
- 5.7 Amortization — single forward pass vs online planning; cost/quality.

**6. Discussion (~1500 w).**
- 6.1 The decision-framework reading: when adaptive control adds value (large H_PI convertible
  residual beyond structure) vs when structured control suffices (flat/belief-insensitive) — a
  practitioner's go/no-go.
- 6.2 Delivering Garrido's bridge; the informative null vs weak-baseline "wins".
- 6.3 Methodological lessons (comparator discipline; fail-closed gates; sealed one-shot validation;
  machine-generated numbers).
- 6.4 Limitations (single-mechanism study; stationary environment; worst-product safety not
  certified; cumulative cross-campaign learning not addressed).
- 6.5 Future work: cumulative learning / Supply-Chain-Learning (Garrido's second ask) — the
  cross-campaign retention design, where structured MPC has no memory (the natural upgrade path
  to a higher-tier venue).

**7. Conclusion (~400 w).**

## Tables (all machine-generated from custodied artifacts; per-cell source SHA)

- **T1 — Master results ladder.** L1–L4 + Program Q rows (extend the existing `results_table.md`;
  already built by `build_results_table.py`). The paper's central quantitative object.
- **T2 — Robustness matrix (NEW).** Rows = the 5 adversarial tests; columns = objection targeted /
  Δ_N or equivalent / CI / verdict. Sources: Q confirmation, DMLPA notebook, warm-start probe,
  reward factorial, QR-DQN adjudication.
- **T3 — Amortization / compute.** Per-decision latency (median, p95), parameters, planner cost,
  break-even; RL vs belief-MPC at equal hardware.
- Supplement: full comparator family; all placebo contrasts; static-search benchmark (CEM vs
  optimizers); demand-ledger identities; custody hash manifest; compute manifest.

## Figures

- **F1 — Environment + mechanism.** The 13-op chain schematic + the two-product shared-line
  decision (adapt thesis Fig; SCUA-style).
- **F2 — The four-level ladder (conceptual).** The decision framework as a diagram: opportunity →
  conversion → adaptation → premium, with the go/no-go reading.
- **F3 — The scratch learning curve (headline figure).** RecurrentPPO from 0→60k: Δ vs best static
  (rises to +0.099) and Δ vs MPC (rises to ~0), monotone, no drift. "RL learns the bridge and
  converges exactly to structured control."
- **F4 — Equivalence + robustness forest plot.** Δ_N (and equivalent) across the 5 tests with CIs
  straddling 0 — the visual proof the null is robust.
- **F5 — Amortization.** Cost (online compute) vs quality (ReT): the RL point at MPC-quality but a
  fraction of the online cost.

## What is NEW vs the current 01–04 drafts (rewrite deltas)

1. Reframe title/abstract/intro from "ladder finding" → "decision framework." (§1)
2. New §3 elevates the framework as the *method*; the four estimands become the contribution, not
   just a result structure.
3. New §5.5 (robustness matrix) + §5.6 (mechanism) — the entire robustness campaign of this
   session, absent from the current drafts. Biggest addition.
4. New §5.7 / T3 amortization with the compute benchmark (contract-mandated, not yet executed —
   needs the latency/planner-cost measurement).
5. Discussion pivots to the practitioner decision-aid reading (6.1) and Garrido's second ask (6.5).
6. Cut/curtail the current draft's "Program Q slot" hedging (§3.5) now that Q is terminal.

## Open inputs needed before drafting prose

- The amortization/compute benchmark numbers (latency, params, planner cost) — not yet measured;
  needed for §5.7/T3. This is the one missing measurement (descriptive, no seeds).
- Final title choice.
- Author/venue confirmation (C&IE primary; single vs Garrido co-authorship).

## Naming note

This is the Q/ladder manuscript (repo "Paper 2"). If it becomes the primary submission, drop the
"Paper 2" label in the manuscript itself; keep it only in repo bookkeeping.
