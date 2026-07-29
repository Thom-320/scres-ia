# Paper 2 — Bottleneck-Migration Response Control

Status: frozen before seeds 1100001+ or 1110001+ are opened. Programs E–H remain terminal.

## Thesis basis

Garrido-Rios defines PT, Q, ROP and risk exposure across Op3–Op12. Op3/5/9 were selected for
the thesis inventory experiment because they were judged critical storage points, not because
the remaining operations were ineligible as decisions. This study declares one extension: a
single response team that can support only one bottleneck family per week.

## Action and mechanism

At week t choose exactly one equal-cost posture:

- M: manufacturing recovery at Op5–Op7; realized R11 downtime ×0.20.
- T: LOC recovery at Op4/8/10/11/12; realized R22/R23 downtime ×0.20.
- R: mission response at Op13; up to 5,000 rations of the existing finite reserve may be issued
  per R24 event.

The choice requested at week t activates at t+1 and remains active for that week. Each posture
consumes exactly one team-week. Threat onsets, base durations and demand are identical across
policies. The action changes realized damage/response, never the exogenous tape.

The 80% recovery is a declared high-authority design cell, not a thesis estimate. It is used
once to test whether the expanded Op3–Op12 decision scope can expose adaptive value. No efficacy,
signal or risk sweep follows this result.

## Information and policies

The hidden context is equipment pressure, interdiction campaign or mission surge, with dwell
4–8 weeks. A balanced-accuracy 0.85 signal predicts next week's context. Policies see the three
scores, recent realized events, backlog, inventory, active/pending posture and time; they never
see the context label or future events.

Comparators: constant M, constant T, constant R. Candidate observable policy: choose the largest
next-week signal score. A depth-3 cross-fitted regret tree is secondary. No PPO participates in
environment selection.

## Data and gates

- calibration: 60 tapes, seeds 1100001–1100060, 24 weeks;
- locked confirmation: 120 tapes, seeds 1110001–1110120, 24 weeks;
- learner virgin: seeds 1120001+, unopened unless the observable gate passes.

Adaptive PASS requires on the locked set: ReT delta over calibration-frozen best constant ≥0.01
with CI95 lower bound >0; service-loss AUC reduction ≥5% with CI95 lower bound >0; lost orders
non-inferior; ≥70% favorable tapes; equal team-hours; mass conservation; and consumed threat and
demand hashes identical within tape. Failure is a valid adaptive-negative Paper 2 result and
blocks PPO. Passing authorizes a separately frozen learner comparison; it does not guarantee a
neural increment.
