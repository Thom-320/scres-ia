# Paper 2 claim ladder — frozen 2026-07-18

The decision ladder is now complete and measured end to end. Scientific commit for the learner
stage: `821c8d8` (later commits on that branch are non-promotable sandbox).

## Confirmed (each traceable to a custodied artifact)
1. **Physical opportunity** — full-DES safe H_PI = 0.15151 (LCB95 0.11562, exact fungible null).
2. **Observable classical control** — corrective validation: mean canonical-ReT PASS in all 3
   cells (LCB95 +0.043..+0.066; 27/27 placebos; equal resources). Joint CVaR10 tail gate failed
   (zero-margin = de-facto superiority; instrument audit retained numerically).
3. **Learned adaptive value over the COMPLETE open-loop frontier (calibration evidence)** —
   O-R: H_OL LCB95 +0.037/+0.043/+0.066, favorable 41-44/48, genuine feedback, placebos beaten,
   exact resources, 990 direct replays max err 5.55e-16. Terminal:
   `STOP_CALIBRATION_NOT_ELIGIBLE` under its compound gate — both preregistered estimands are
   reported separately, as frozen.
4. **Training integrity** — 10 seeds x 200,192 steps (391x512 rollouts), 250,240 stepped tapes
   + 10 sentinels, disjoint partitions, custody verified.

## Not confirmed
- **Neural premium over belief-MPC** (Delta_N ≈ −0.002, LCB95 −0.008..−0.014; 0-2/10 seeds).
- **Tail-safe deployment** (never claimed; CVaR secondary only).
- **Sealed independent replication** (Program Q, frozen 2026-07-18, N=128/cell, block 7490001+).
- **Retained-learning value** (Paper 3; blocked until Q passes E1 with premium or equivalence).

## Estimand rename (prospective only)
`H_learned → H_OL` (learner − best complete open-loop); `H_neural → Δ_N` (learner − best
classical). Historical artifacts are not rewritten.

## The sentence main must stop implying
"No learner has established adaptive value" is scientifically outdated: a learned policy
established adaptive value over every open-loop schedule at calibration level. What no learner
has established is a neural PREMIUM over structured belief control — and that separation is the
paper.
