# Program H — belief-audit DEVELOPMENT result (1060001) — 2026-07-12

Status: **Strong Case-A (information-limited) evidence on DEVELOPMENT; the certified close (exact
belief-MDP J*_obs / info-relaxation dual on the LOCKED test) is the remaining gated step.** No
1070001/1080001 tape opened. Built: `supply_chain/program_h_belief.py` (augmented semi-Markov filter,
independent per CSSU), `scripts/run_program_h_belief_audit.py`.

## Result (development 1060001, 200 tapes, ret_order)
- **Filter validated (informative):** filtered tempo log-loss **0.230** vs marginal-prior **1.002** —
  the augmented `(tempo, dwell_age)` belief recovers the latent tempo far better than chance.
- **Clairvoyant headroom present:** `H_PI = J_PI − J_ABAB = +0.0149 [0.0114, 0.0184]` (≈ Program G).
- **Every available-information policy LOSES to ABAB:**
  - belief cover (signal + inventory-revealed demand): **−0.0209 [−0.0273, −0.0145]**
  - belief cover (signal only): **−0.0209** (identical)
  - **QMDP-nowcast (knows CURRENT tempo exactly): −0.0209** (identical)
- `belief ≈ QMDP-nowcast` because the filter is near-certain about the CURRENT tempo. So even a policy
  with a *perfect nowcast* of both theatres' current regimes is materially worse than blind alternation.

## Reading (honest scope)
The clairvoyant value (+0.015) comes almost entirely from knowing the FUTURE transition timing, which is
inaccessible; knowing the present (nowcast) yields −0.021 vs ABAB. This is exactly the
information-limited pattern: the recoverable state (current tempo) is not enough, and the ret_order
optimum is near-alternation, so concentrating on the currently-surging theatre (the "adaptive" move)
hurts fairness/continuity regardless of how well it is nowcast.

**What this does NOT yet prove.** These are strong policies (cover-class + a nowcast diagnostic), not the
certified `J*_obs` over ALL observable policies. Development cannot rule out a non-cover observable policy
capturing part of the +0.015. The amendment's rigorous close remains:
1. exact belief-MDP backward induction for `J*_obs` (primary), or the Brown–Smith–Sun
   information-relaxation dual upper bound (fallback) — both on the LOCKED test 1070001;
2. if `UB(J*_obs) − J_ABAB < δ_min=0.01` → **Case A proven** (information-insufficiency); virgin 1080001
   stays sealed and the spatial contract is terminal.

## Consequence
Development strongly favours Case A. Per the non-rescue boundary, the manuscript writing proceeds in
parallel now; the locked-test certified bound is the final Program H artifact. Even a Case-A close is a
publishable formal result (the collection of nulls becomes an information-insufficiency explanation).
`results/program_h/dev/verdict.json`.
