# Program I — headroom GSA result — 2026-07-12

Status: **The sensitivity map + a precisely-located region confirm the central finding with a
principled method.** Adaptive headroom responds ONLY to structural scarcity/concurrency (not
information quality, not risk magnitude); the one region where an observable policy beats the static
does so by violating spatial fairness, not by genuine convertible value. No learner trained.

## Stage 1 — Morris screening (elementary effects, n=50 tapes, r=8)
μ\* = mean-abs effect on headroom; σ = interaction/non-linearity.

| Factor | μ\*(H_PI) | μ\*(H_obs) | σ(H_obs) |
|---|---:|---:|---:|
| surge_mult (scarcity) | 0.0132 | 0.0182 | 0.0095 |
| commonality (A/B co-surge) | 0.0063 | 0.0144 | 0.0077 |
| signal_q (information quality) | **0.000** | **0.000** | 0.000 |
| lead (advance information) | **0.000** | **0.000** | 0.000 |
| persistence | **0.000** | **0.000** | 0.000 |
| r22_prob (risk magnitude) | **0.000** | **0.000** | 0.000 |

**Only scarcity and A/B concurrency move headroom; information quality and risk magnitude are
formally INERT.** High σ on the two live factors = the headroom is an interaction (scarcity ×
concurrency), matching the project's "conjunction" thesis. This is the EVPI/VSS sensitivity map
confirming, with a principled method, what the seven lanes found ad-hoc: editing risk magnitude or
signal quality cannot create adaptive headroom.

## Stage 3 — GP + expected-improvement locate (argmax H_obs)
Located region: signal_q≈0.53, lead≈2, surge_mult≈1.95, persistence short, **commonality≈0.89**,
r22≈0.11 → H_obs=+0.013, η=0.90 on the search tapes (naive `qualifies_new_lane=true`).

## OOS + guardrail check (my frozen decision rule — the disciplined verdict)
On FRESH tape blocks (not the GP-search seeds), 200 tapes, bootstrap CI + Program-G guardrails:

| block | H_obs | CI95 | ret_quantity Δ | **worst-CSSU fill Δ** | attended Δ |
|---|---:|---|---:|---:|---:|
| GP-search 3000001 | +0.0131 | [+0.0102,+0.0160] | +0.0142 | **−0.1435** | −0.09 |
| FRESH 4200001 | +0.0114 | [+0.0087,+0.0141] | +0.0128 | **−0.1282** | −0.14 |
| FRESH 4500001 | +0.0100 | [+0.0072,+0.0129] | +0.0130 | **−0.1258** | −0.28 |

- **H_obs is POSITIVE and OOS-stable (CI95>0) on ret_order AND ret_quantity** — the first parametric
  region where an observable policy beats the best static out-of-sample. Not a winner's-curse artifact.
- **BUT it FAILS the worst-CSSU-fill guardrail (~−0.13, threshold −0.02):** the win is bought by
  starving one theatre — the exact Program-G / Cobb-Douglas concentration/fairness artifact. Under a
  fairness-preserving evaluation there is no deployable adaptive value. **`qualifies_new_lane = false`.**

## Conclusion
The GSA is a clean methodological contribution and it precisely locates the boundary:
1. **Adaptive headroom is inert to information quality and risk magnitude** (Morris μ\*=0) — editing
   Garrido's risks/signals cannot create it. Confirmed with a principled variance/screening method.
2. **It responds only to structural scarcity × A/B concurrency** (the interaction the project's thesis
   predicted).
3. **The single region with H_obs>0 is a spatial-fairness violation, not convertible value** — the
   aggregate-metric "win" starves a node, reproducing the Program-G/triangulation lesson exactly.

So the honest answer to "can editing the decision variables/risks create enough headroom to test
adaptive policies?" is now backed by a global sensitivity map: **no — the only place an observable
policy beats the static is where it sacrifices a theatre, and no informational or risk-magnitude edit
creates deployable adaptive headroom.** The map IS the result (a boundary characterization for the
manuscript), and the frozen non-rescue rule correctly refuses the fairness-violating region as a lane.
Estimators validated (Sobol/Ishigami; Program-G anchor). `results/headroom_gsa/*.json`.
