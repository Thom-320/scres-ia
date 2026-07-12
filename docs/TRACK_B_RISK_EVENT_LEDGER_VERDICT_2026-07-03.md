# Track B risk-event ledger — first result — 2026-07-03

## What this is

`scripts/audit_track_b_risk_event_ledger.py` instruments `sim.risk_events` directly (real
`risk_id` + `start_time` from the DES) instead of the old regime-transition proxy anchor. Eval-only,
no retraining. Run: 5 seeds x 12 episodes x {ppo_mlp, real_kan}, `adaptive_benchmark_v2`, v7,
`track_b_v1`. Artifact: `outputs/experiments/track_b_risk_event_ledger_2026-07-03/`.

## Frequency sanity check (confirms the classification empirically)

| Risk | Category | n observed | events/year | vs. R1-baseline expectation |
|---|---|---:|---:|---:|
| R11 | frecuente | 15,647 | 65.4 | 103.7 (same order of magnitude) |
| R13 | frecuente | 9,149 | 38.2 | 14.4 |
| R24 | frecuente | 4,721 | 19.7 | 26.0 |
| R22 | intermedio | 819 | 3.4 | 4.3 |
| R12 | intermedio | 306 | 1.3 | 2.2 |
| R23 | intermedio | 335 | 1.4 | 2.2 |
| R21 | raro | 129 | 0.5 | 1.1 |
| R3 | raro | 16 | 0.07 | 0.11 |

105,576 total risk events logged. Frequent risks give 4,700-15,600 occurrences to average over;
rare risks give only 16-129 — confirms the frequent/intermediate/rare split is empirically real,
not just a theoretical distinction, and gives the frequent-risk event-study massively more
statistical power than the old single-anchor-per-episode method (~13,000-13,700 step observations
per relative-week bucket here, vs. tens in the earlier counterfactual sanity check).

## Event-aligned action study (frequent risks, t=-4..+8 weeks relative to each real risk onset)

PPO+MLP `action_intensity` (mean over ~13,000+ steps per bucket):

| Week | -4 | -3 | -2 | -1 | 0 (event) | +1 | +2 | +3 | +4 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| action_intensity | 0.2332 | 0.2333 | 0.2341 | 0.2337 | 0.2346 | 0.2348 | 0.2344 | 0.2354 | 0.2363 | 0.2350 |

**Reading**: there is a small, consistent uptick even before the event (0.2332 → 0.2337, weeks
-4 to -1), but the much larger rise happens after (0.2346 → 0.2363, event to +4), then it recedes.
**Predominantly reactive, with a small anticipatory component** — not purely reactive, not clean
prevention either.

**Important caveat**: R11 alone occurs roughly once per week on average (mean inter-arrival
~84.5h ≈ half a week). At that frequency, one event's "post" window overlaps heavily with the next
event's "pre" window — the small pre-event uptick could be genuine anticipation, or it could be
residual recovery from the *previous* nearby event bleeding into what we're calling "baseline."
This event-study is descriptive, not causal — it does not yet isolate value added (that requires
the `R_full - R_reset(w)` counterfactual, now anchored on these precise per-risk-id timestamps
instead of the old regime proxy, which is the natural next step).

Real-KAN shows a much flatter profile (~0.969 throughout, since it already operates near the
action ceiling most of the time per earlier findings) — its narrow operating range leaves little
room for this simple mean-intensity metric to show a reactive/preventive pattern either way.

## Verdict

This is real, much-higher-power descriptive evidence than the earlier regime-anchored study, and
it is consistent with (not a reversal of) the earlier finding that the causal counterfactual
audit could not establish clean prevention. It sharpens the picture: PPO+MLP's adaptive behavior
around frequent risks looks mostly reactive with a small, statistically fragile anticipatory
signal — worth testing causally next with per-risk-id anchors, but not yet evidence of prevention.

## Next step

Re-run the `R_full - R_reset(w)` counterfactual (already fixed to use per-policy calm baselines,
see `docs/TRACK_B_PREVENTION_COUNTERFACTUAL_VALIDATION_2026-07-03.md`) anchored on these precise
per-risk-id event timestamps instead of the regime-transition proxy, separately for frequent vs.
rare risks. Precise anchoring plus the much larger sample size for frequent risks should give the
counterfactual far more power than the 60-pair regime-anchored version that failed to validate.
