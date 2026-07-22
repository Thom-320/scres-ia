# Horizon diagnostic (EXPLORATORY_NO_CLAIM, burned) — 2026-07-22

**Does NOT certify exhaustion and does NOT authorize M1 or Door 2.** A burned, p16, single-θ,
binary-carrier diagnostic. Partially redundant with Codex's existing
`results/q_r1/retained_mpc_calibration_v3` (which already showed H4=H6=0.7582191); I could not
independently verify that artifact (VPS/Codex-local only), but this probe agrees with it.

## Result (burned roots 7570801–7570808, 176 paired campaigns, stratified planner, p16, θ=(.90,.90))

| Paired contrast (retained arm, full-cohort) | mean | nonzero campaigns |
|---|---:|---:|
| retained_H4 − retained_H3 | +0.0000000 | 0 / 176 |
| retained_H6 − retained_H4 | +0.0000000 | 0 / 176 |

The retained controller makes **identical decisions at H3, H4, and H6** on every campaign. The reset
arm changed in 1/88 campaigns (+0.0007 mean). So at p16/this-cell/burned, the retained MPC is
horizon-saturated by H3.

## Honest scope (limitations, per the Codex review — accepted)
- **p16 only** — not p16/p64/p256 convergence; p16 is not ground truth. H8 burned did not pass
  convergence and cost ~88s (p16) / ~345s (p64) per first action.
- **single θ=(.90,.90)**, **binary regime carrier** only.
- roots are **burned**; planner actions/values not stored.
This is a descriptive diagnostic. It is a prior toward control-saturation in the stationary env; it
is NOT a saturation certificate and authorizes nothing.

## Implication for the corrected route
The horizon door (Door 1) shows no burned-diagnostic control headroom at these settings — consistent
with routing away from "a deeper-horizon learner wins" and toward the residual/risk-sensitivity
analysis of the corrected route, but only the proper full-cohort + policy-independent-CRN + power-audited
residual measurement (against the frozen comparator v2, AFTER the canonical Q-R1 adjudication) can
authorize anything.
