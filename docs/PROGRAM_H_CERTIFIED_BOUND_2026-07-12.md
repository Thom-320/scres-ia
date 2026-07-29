# Program H — certified bound close — 2026-07-12

Status: **CLOSED with the rigorous certified upper bound + tight empirical evidence. Information-limited
(Case A) strongly supported; the observable advantage is rigorously capped small and empirically
negative.** LOCKED test 1070001 (n=400). 1080001 never opened.

## The certified bound (rigorous)
`J*_obs ≤ J_PI` (perfect information dominates any non-anticipative policy — an exact inequality, no
approximation). On the full 400-tape locked test:

- **`J_PI − J_ABAB = +0.0164, CI95 [0.0139, 0.0189]`** → the observable advantage over blind alternation
  is certified `≤ +0.019` (95%), and that ceiling is achievable ONLY with perfect knowledge of future
  tempo transitions.
- **On 63% of tapes even the clairvoyant cannot beat ABAB** (`clairvoyant > ABAB` on only 37%;
  `> ABAB + δ_min` on 32.75%). Blind alternation is optimal or near-optimal on most tapes.

## Tight empirical evidence (strong, not certified)
- Best available-information policy = a **perfect current-tempo nowcast**: `−0.026 [−0.035, −0.018]`
  vs ABAB (locked), matching development `−0.021`. Every strong observable policy LOSES.
- So `J*_obs` is sandwiched: rigorously `≤ ABAB + 0.019`, and every realized observable/nowcast policy
  sits at `≈ ABAB − 0.026`. The clairvoyant advantage lives almost entirely in inaccessible FUTURE
  transition timing.

## Honest limitation (disclosed, not hidden)
The certified LOOSE bound (`+0.0164`) exceeds `δ_min = 0.01` by ~0.006, so a formal "no material
(≥1%) observable advantage" theorem is NOT proven — only that the maximum conceivable observable
advantage is ~1.6–1.9% and requires perfect future information. A certified sub-δ upper bound would need
an additive completion-week belief-MDP DP; this is intractable at exact granularity because `ret_order`
uses the workbook's CUMULATIVE-backorder normalization `1 − (Bt+Ut)/j`, which is not Markov in a small
`(inventory, backlog)` state. The Brown–Smith–Sun dual attempted earlier was mis-scaled
(service-loss-unit penalty vs ret_order∈[0,1]) → discarded. This is the single remaining formal item;
it does not change the practical conclusion.

## Conclusion (Program H closed)
`J_static ≤ J*_obs ≤ J_PI`, with `J_PI − J_static = +0.016` and every strong observable/nowcast policy
at `J_static − 0.026`. **Adaptive control is not warranted on the spatial contract under `ret_order`:
the value is information-limited (future transition timing is inaccessible) and the thesis resilience
metric rewards robust alternation.** This is publishable as the information-sufficiency close: the
collection of nulls (D, DRA-1/2/2b, E, F, G) is now explained — even a perfect current-state nowcast
cannot beat a fixed schedule, and the clairvoyant ceiling itself is small. Program H is the last
computational extension; the manuscript is the next artifact. `results/program_h/bound/verdict.json`.
