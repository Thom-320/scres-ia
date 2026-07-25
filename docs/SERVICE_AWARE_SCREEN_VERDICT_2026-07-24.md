# Service-aware retained screen — final verdict (2026-07-24)

**Verdict: `STOP_SERVICE_AWARE_NO_SAFE_CONVERSION`.** 0 of 9 preregistered configs pass.
Claim status `BURNED_DEVELOPMENT_NO_CLAIM` (burned roots 7570801–24 only; no learner anywhere).

- Contract: `contracts/q_r1_service_aware_retained_screen_v1.json`, frozen 2026-07-23 before
  the first shard ran; gates copied verbatim into the adjudicator.
- Instrument: `scripts/run_q_r1_service_aware_screen.py`,
  `scripts/adjudicate_q_r1_service_aware_screen.py`
- Result: `results/q_r1/service_aware_screen_v1/adjudication_final.json` (9/9 configs,
  48 CRN-paired arms each, no truncated shard).

## What the screen tested

The H3 frontier: the retained-belief mean ReT gain is confirmed (+0.066 prospective,
LCB95 +0.052) but is **not deployable-safe** — the sealed confirmatory stopped at
worst-product-fill LCB95 −0.0365 and the burned κ=0.75 stratum breaches at −0.054. The
screen asked whether any *planner-side* service mechanism can keep the mean gain inside the
safety bar. Three families, grid fixed in advance: V1 lower-tail feasibility floor
(α ∈ {0.10, 0.25} × floor ∈ {0.70, 0.80}), V2 belief-conditional floor (slope ∈ {0.4, 0.6}),
V3 penalty composite (λ ∈ {0.5, 1.0, 2.0} at floor 0.80). V0 is a documented no-op: the
conditional bank is already regime-consistent. V4 was rejected earlier for ranking instability.

## Results (clustered on history_root, 10k bootstrap, one-sided LCB95)

| config | κ0.90 ret mean / LCB | κ0.90 wf LCB | κ0.75 wf LCB | κ0.75 Δunresolved | gate |
|---|---|---|---|---|---|
| V2 bfs 0.40 | +0.0692 / +0.0542 | −0.0279 | −0.0580 | +1.04 | fail |
| V2 bfs 0.60 | +0.0692 / +0.0539 | −0.0276 | −0.0430 | +0.62 | fail |
| V1 fta 0.10 / 0.70 | +0.0718 / +0.0574 | −0.0516 | −0.0988 | +2.54 | fail |
| V1 fta 0.25 / 0.70 | +0.0664 / +0.0514 | −0.0380 | −0.0727 | +1.62 | fail |
| V1 fta 0.10 / 0.80 | +0.0690 / +0.0541 | −0.0552 | −0.0982 | +2.58 | fail |
| V1 fta 0.25 / 0.80 | +0.0664 / +0.0509 | −0.0391 | −0.0742 | +1.75 | fail |
| V3 λ 0.5 | +0.0690 / +0.0540 | −0.0496 | −0.0902 | +2.33 | fail |
| V3 λ 1.0 | +0.0718 / +0.0571 | −0.0492 | −0.0902 | +2.33 | fail |
| V3 λ 2.0 | +0.0690 / +0.0539 | −0.0506 | −0.0902 | +2.33 | fail |

Gate bars: ret mean ≥ +0.05 at κ0.90 (**met by all 9**); worst-product-fill LCB95 ≥ −0.02 in
**both** strata (**failed by all 9**); Δunresolved ≤ +2.0 mean and ≤ 12 max.

## Mechanism — why every family fails identically

`total_planner_fallbacks = 0` across all nine configs, and the V3 λ grid is **degenerate**:
λ = 0.5, 1.0 and 2.0 return byte-identical safety and unresolved statistics (−0.0902, +2.33).
Quadrupling the penalty changes nothing because the penalised term is never non-zero.

The cause is prediction-side, not constraint-side. The scenario bank's predicted
`worst_product_fill` (min over the two products, computed over 256 conditional paths) never
dips below the floor even on campaigns whose realized fill collapses to 0.66 — so a floor, a
belief-scaled floor, and a penalty on the same statistic are all inert by construction. The
gain is bought late-campaign (the early cohort never breaches) in the product the posterior
*favored*: over-commitment followed by a regime flip. No planner that reasons only over its
own optimistic forecast can see that coming.

This is the third independent confirmation of the same structural fact and it upgrades the
finding from "these variants failed" to "this mechanism class cannot work here".

## Consequences

1. **The honest boundary stands and is now certified**: the retained-belief effect converts in
   the **mean** and does not convert **jointly safely**. This is the same shape Program O
   closed with (mean observable conversion real, jointly safe conversion not established) —
   two independent programs reaching the identical frontier is a paper-grade finding, not a
   pair of failures.
2. **The remaining non-refuted route is realized-service feedback**: a controller that
   observes realized per-product fill and backlog age instead of only its forecast. That is a
   different observation contract and requires a new preregistration — it is NOT a rerun of
   this screen, and nothing here licenses it as a positive.
3. No sealed block is spent; no seed opened beyond the burned 7570801–24; the frozen c256
   contract, receipt and Pareto artifacts are byte-untouched.

## Process defects fixed during execution (disclosed)

The first VPS attempt lost 137 completed arms to an abort-not-continue 12-hour cap. Fixed in
`ec37e255` (per-arm jsonl checkpoint + resume, truncate-and-keep instead of raise), the cap
made opt-in in `f9afc8d3`, and a `None`-guard defect from that change fixed in `021db187`
after it crashed both fleets. No scientific artifact is affected: all 9 configs ran to
48/48 arms with `truncated_by_hard_cap: false`, and the checkpoint rows are the same
deterministic CRN-paired evaluations.
