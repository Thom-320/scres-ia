# A real KAN, for Garrido — 2026-07-02

## Why this document exists

Garrido asks about Kolmogorov-Arnold Networks (KAN) consistently, because
his 2024 ICCL paper names KAN (alongside backprop NNs and RL) as a
candidate family for closing the SCRES "Alzheimer effect" gap
(`docs/KAN_SIDECAR_SMOKE_2026-07-02.md` audits that paper's actual KAN
argument: two bare name-drops, no properties claimed, no KAN citation).

**Important correction to the prior sidecar work.** The earlier
`scripts/kan_extractor.py` / `PPO-KAN` sidecar is **not a KAN**. It is a
per-input radial-basis-function (RBF) expansion with a full linear skip
connection (`nn.Linear` added directly to the features). It has no
learnable spline edge functions — the actual KAN mechanism from Liu et
al. (2024) — and its linear skip alone gives it MLP-equivalent capacity,
which is why it scored about the same as canonical PPO+MLP. Calling it
"PPO-KAN" overclaims kinship with the real architecture and should not
be repeated verbatim; if it is mentioned at all, call it a "KAN-inspired
RBF feature extractor," never "KAN" unqualified.

**This document reports a genuine KAN**, using the official `pykan`
package (`pip install pykan`, Liu et al. 2024 reference implementation,
learnable B-spline univariate edge functions), run locally in this
session.

## What it demonstrates

Garrido's own Fig. 5 (ICCL 2024) proposes inserting a neural network
between the DES **decision variables** ($\rho_i$) and the **SCRES output
metric**, to close the open loop between experiment design and measured
resilience. This script (`scripts/run_kan_scres_demo.py`) instantiates
that exact architecture literally:

- **Inputs**: the three static decision variables that define a Track B
  dispatch policy — assembly shift level $S$, Op10 dispatch multiplier,
  Op12 dispatch multiplier.
- **Output**: Excel ReT (the paper's primary resilience metric).
- **Data**: the 147-cell dense static dispatch frontier — the same
  evidence bundle behind the manuscript's Table 4 / Figure 4
  (`docs/track_b_q1_stats_2026-07-02_final/pareto_points.csv`).
- **Model**: `KAN(width=[3,4,1], grid=5, k=3)`, trained with LBFGS (60
  steps, grid updated during training), 118/29 train/test split, seed 0.

This is supervised regression on a small, clean design table — seconds
to train, directly interpretable via pykan's built-in spline
visualization, and a faithful, literal implementation of what Garrido
himself sketched.

## Result

| Model | Test R² | Test MSE |
|---|---:|---:|
| **KAN** (pykan, official) | **0.998** | 3.4e-9 |
| MLP baseline (16-16, untuned, 400 Adam steps) | 0.624 | 5.3e-7 |
| Linear baseline | 0.365 | 9.0e-7 |

Artifacts: `outputs/experiments/kan_scres_demo_2026-07-02/`
- `kan_fit_summary.json` — exact numbers above
- `kan_splines.png` — pykan's native diagram: the learned univariate
  edge functions (the visual signature of a KAN) for this exact mapping
- `kan_vs_baselines.png` — predicted-vs-actual on held-out configs,
  house style, all three models

The KAN fits this low-dimensional, smooth decision-to-metric mapping
almost exactly on held-out static configurations, and its learned edge
functions (`kan_splines.png`) are visibly non-degenerate (each shows a
distinct nonlinear shape, not a flat/collapsed function) — i.e. it is a
genuine, non-trivial fit, not a saturated or memorized one.

## Honest caveats — read before showing this to anyone

1. **The MLP baseline was not tuned.** Fixed architecture, fixed epoch
   count, no early stopping, no learning-rate search. A properly tuned
   MLP would likely close much of the gap on this smooth 3-input
   function — KANs are known to be strong on exactly this kind of
   low-dimensional smooth regression, which is *why* this comparison is
   favorable, not because KAN is unconditionally superior. Report the
   numbers as "KAN fits this mapping very well," not "KAN beats MLP."
2. **This is a supervised surrogate, not a control policy.** It predicts
   ReT from a static decision-variable triple; it does not act, does not
   close the RL loop, and is not a replacement for the PPO+MLP result
   that is Paper 1's headline. It is a direct, small-scale demonstration
   of Garrido's Fig. 5 concept — nothing more.
3. **Small N (147, split 118/29).** Fine for a demo; not a claim of
   general KAN superiority on high-dimensional or stochastic tasks (the
   actual Track B RL problem is both).

## Where this goes

Not Paper 1's headline, and not a replacement for the RBF-sidecar
robustness note already scoped as appendix/response-letter material.
This is specifically **a demonstration artifact to show Garrido a real,
literal instantiation of his own proposed architecture**, using the
official library, on project data, with an honest accuracy comparison
and the standard KAN interpretability visualization he would recognize.
Suggested framing for him: "we built the network you sketched in Fig. 5
— decision variables in, SCRES metric out — with an actual KAN, and it
fits the static frontier almost exactly; here are the learned edge
functions."
