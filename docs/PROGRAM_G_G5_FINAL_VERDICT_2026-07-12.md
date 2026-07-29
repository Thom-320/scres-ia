# Program G — G5 final verdict (learner on the frozen region) — 2026-07-12

> **⚠ SUPERSEDED HEADLINE (2026-07-12): the win below is SERVICE-LOSS-PROXY ONLY.** The ret_excel
> confirmation (`docs/PROGRAM_G_RETEXCEL_CONFIRMATION_2026-07-12.md`) shows that under the project's
> PRIMARY metric ret_excel, NO observable policy beats blind alternation (cover −0.036, ret_excel-native
> tree −0.012), even though clairvoyant ret_excel headroom exists (+0.018). Program G is a boundary
> result under ret_excel; the adaptive win here is an artifact of the ration-mass proxy. Read the two
> docs together.


Status: **`G5_LEARNER_BEATS_STATIC_OOS_NO_NEURAL_INCREMENT` — the project's FIRST virgin-confirmed
observable adaptive-control win, delivered by an INTERPRETABLE depth-3 rule; PPO matches but does not
beat it.** Run under PI autonomy (`docs/PROGRAM_G_AUTONOMY_AUTHORIZATION_2026-07-12.json`). Service-loss
proxy, disclosed; stylized weekly V1.2 contract; convoy ontology Option A pending Garrido.

## Pipeline (G3→G5)
- **Region frozen** = the 12 surge-1.50 cells from the G2 connected component (uniform mixture).
- **G3**: depth-3 observable tree + logistic contextual bandit fit on the clairvoyant-action dataset
  from TRAIN tapes (990001+); best full-contract static (120 periodic calendars) frozen on TRAIN;
  cover heuristic frozen.
- **G4**: frozen policies evaluated on HOLDOUT (1000001+), no tuning.
- **G5**: MaskablePPO trained ONCE on TRAIN (60k steps, seed 9701, 2×64 tanh, no sweep); all policies
  evaluated on VIRGIN tapes (1010001+, opened only after freezing).

## Virgin-tape result (H = best full-contract static − policy service-loss; CI95 lower > 0 ⇒ win)

| Policy | H (rations) | CI95 | η vs oracle | beats static? |
|---|---:|---|---:|:---:|
| depth-3 tree | +560 | [408, 737] | 0.734 | **WIN** |
| cover heuristic | +560 | [408, 737] | 0.734 | **WIN** |
| contextual bandit | +560 | [408, 737] | 0.734 | **WIN** |
| MaskablePPO | +500 | [328, 698] | 0.655 | WIN |
| MPC/rollout | +354 | [190, 543] | 0.463 | WIN |

- **PPO − cover = [−60, −130, +8]** → CI95 straddles 0: **PPO adds NO neural increment over the
  interpretable heuristic** (slightly worse on the mean).
- Frozen best static calendar = ABAB (blind alternation).

## The interpretable decision aid (the depth-3 tree)
```
if cover_A <= 0.15 weeks:   dispatch the shared convoy to A
else:                       dispatch to B
```
It never uses HOLD and never uses the advance signal (the signal_A split is degenerate) — it just
serves the CSSU running low on days-of-cover. This is the whole adaptive policy.

## Conclusion (claim-ladder rung: "tree wins, PPO does not → interpretable adaptive control suffices")
Program G is the **first contract in the entire project where an observable, resource-matched,
closed-loop policy beats the best full-contract static out-of-sample AND on virgin tapes** — capturing
~73% of the clairvoyant spatial headroom. Crucially:
1. **The winner is an interpretable one-line rule**, not a neural network. PPO matches the win but adds
   nothing over the cover heuristic (consistent with Program E's PPO≈heuristic finding).
2. **The value is spatial-commitment observability, not advance information** (G1: signal marginal +9;
   G2: signal quality/lead do not discriminate; the winning tree ignores the signal). When a shared
   scarce resource must be committed to one of two theatres and current inventory reveals which is
   starving, watching cover suffices; forecasting is redundant.

This completes the diagnostic ladder positively:
`physical authority → clairvoyant headroom → OBSERVABLE conversion (here, unlike E/F) → interpretable
adaptive control; neural learning adds no increment`.

## For Garrido (results to approve on)
Program G's shared-transport spatial-commitment mechanism produces genuinely learnable, deployable
adaptive value — an interpretable "serve the lower-cover theatre" convoy rule beats every fixed
schedule on unseen scenarios, resource-matched, capturing 73% of the clairvoyant optimum. This is the
positive counterpart to the six boundary results and the E/F nulls. It is conditional on the disclosed
V1.2 contract (Option-A convoy, surge 1.50, service-loss proxy); if Garrido rejects an assumption the
result reclassifies, it is not defended.

## Limits
Stylized weekly proxy (not `ret_excel`, not the full 12-op DES); the win is at surge 1.50 (1.25 shows
no headroom); L(e−1)/retained learning NOT tested (no cross-campaign latent parameter here). Next, if
desired: port the mechanism into the full DES for a ret_excel confirmation, and a persistence-vs-signal
sensitivity for the paper.
