# Track B-P Gates 0/1 Verdict (2026-07-09)

**Headline: preventive headroom EXISTS under the `track_bp_v1` commitment contract — the
first positive prevention gate of the entire program — but it is risk-specific and
regime-specific, exactly where the DES physics says stock can bridge an outage.**

Protocol: `docs/TRACK_BP_PREREGISTRATION_2026-07-08.md` (gates, decision rules, and
falsifiers fixed before any result was read). Runs: `outputs/experiments/track_bp_g{0,1}_*_2026-07-08/`
on ovh-agent-lab, `strict_exogenous_crn=True`, constant-neutral reference policy (no
checkpoint, removing the prior battery's off-distribution caveat). All numbers below
recomputed from the raw `headroom_rows.csv` / `gate1_rows.csv`, not the summaries.

## Gate 0 — forced-prep causal response surface (real vs placebo anchors)

| Tier | real Δ(max_prep−calm) | placebo Δ | DiD (local ReT) | verdict |
|---|---|---|---|---|
| **R11 rare-long (freq×0.125, impact×8), lead 168h** | **+0.002856** (31+/4−/27·, n=62) | +0.000182 (12+/0−/28·, n=40) | **+0.002674, bootstrap CI95 [+0.001061, +0.004622]** | **POSITIVE — first causal forced-prep signal ever** |
| R21 impact×3 (isolated events), lead 168h | +0.000000 exact (0/16) | +0.000155 (2/64) | −0.000155 | null |
| R21 impact×3, lead 336h | +0.000000 exact (0/16) | +0.000155 (2/64) | −0.000155 | null |
| R23 impact×3, lead 168h | +0.000019 (1/29) | +0.000465 (3/64) | −0.000446 | null |
| R24 natural freq | 1 isolable anchor total | — | — | design-dead (surge cadence blankets isolation) |
| R22 impact×3 (negative control) | +0.000212 (5/58 = 8.6%) | +0.001610 (10/64) | −0.001398 | null, control behaves |

Notes recorded against the pre-registered bar: the R11 real-positive rate is 50% (bar: 60%)
and the placebo rate 30% (historical band: 3–19%). The DiD bootstrap nonetheless excludes
zero decisively, the real/placebo magnitude ratio is 15.7×, and the sign profile is the
signature of a genuine channel (prep helps when the breakdown actually drains stock — about
half the anchors — never mind the rest; only 4/62 nominal negatives). The elevated placebo
rate is small-magnitude background contamination (R11 still fires ~every 5.6 weeks at
freq×0.125), which the DiD design differences out. We read this as a **pass with disclosed
deviation from the rate bar**, not a clean-bar pass.

## Gate 1 — static clock-policy oracle (CRN-paired, n=24 episodes)

| Cell | always−never (episode ReT) | calendar−never | verdict |
|---|---|---|---|
| **R21 compound starvation (freq×8, impact×4)** | **+0.029872, CI95 [+0.014231, +0.047805]**, 14+/10·/0− (**+13.9% rel.**, 0.2149→0.2448) | +0.014385, CI95 [+0.006224, +0.023833] | **POSITIVE — static preventive headroom, zero harm cases** |
| R24 freq×3 | +0.000000 exact (0/24) | +0.000000 | null (surges never starve SB) |
| R13 impact×8, monthly calendar | +0.000000 exact (0/24) | +0.000000 | null (WDC working stock ~4M ≫ any stall; op3 dead lever as pre-registered) |

## Synthesis

1. **The prevention boundary of the paper stands where it was drawn**: R22 (transport
   knockout), R23 (forward-unit destruction), R24 (demand surge), R13 (supplier stalls)
   have no preventive channel even under the commitment contract — nulls through both a
   causal lens (Gate 0) and a static-oracle lens (Gate 1), most of them exact.
2. **The boundary is NOT universal**: give the policy Garrido's own lever (lagged strategic
   buffers) and score regimes where disruption actually drains stock, and preventive value
   appears with causal structure: R11 rare-long breakdowns (forced pre-positioning before
   real events beats placebo 15× with CI95 > 0) and R21 compounding disasters (blanket
   buffering +13.9% episode ReT, never harmful).
3. Mechanistic reading matches the DES: buffers bridge production/storage outages
   (Op5/Op9 stock keeps serving while AL/SB recover); they cannot bridge destroyed
   transport (R22), a destroyed forward node (R23), or demand that outruns capacity (R24).
   Under `current` intensity nothing drains, so nothing pays — same as Garrido's H2, where
   buffer moderation shows under increased risk.
4. Falsifier #1 (bit-identical arms) fired only where physics says it should (non-starving
   regimes) and was resolved by the pre-registered starvation calibration, not post-hoc.

## Decision (per pre-registration)

Gate 2 (training) **unlocks** for the two positive regimes:
- Cell A: R21 freq×8 impact×4 (compound starvation) — comparators: never/always/calendar
  clock policies from Gate 1.
- Cell B: R11 freq×0.125 impact×8 (rare-long breakdowns) — same comparator set (Gate-1
  oracle to be run in this cell alongside the screen).

Conversion metric: fraction of the always−never oracle headroom captured by PPO
(`ret_excel`, CRN-paired, same eval seeds). Real-KAN remains an efficiency sidecar only.
No manuscript changes: this is the Track B-P extension lane (paper 2 / future work).
