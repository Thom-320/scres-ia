# Program G — pre-build charter amendment V1.1 (frozen 2026-07-12)

Status: **FROZEN. BUILD (G0, tapes 980001+) REMAINS UNAUTHORIZED PENDING GARRIDO SIGN-OFF.**

Authored by the verifier (Codex out; verifier owns this turn). This amendment supersedes the
*minimal envelope* implied by the Program G charter (`8d555b9`) after an external review
(ChatGPT-Pro, 2026-07-12) found a physical contradiction that would kill the mechanism before it
started. The charter's scientific idea and its guardrails (no RL in environment selection; E/F
terminal; branching, placebos, holdout, resource-matched comparators, ≥2 adjacent cells before
training) are UNCHANGED and endorsed. Only the physical/informational ontology is corrected.

The math was verified first-hand: 168h weekly epoch − 48h convoy cycle = **120h idle/week**; the
enumerable open-loop oracle is **393** sequences over 4 weeks and **1,569** over 8 weeks
(`1 + 2n + 4·C(n,2)`); `RATIONS_PER_BATCH = 5000` is a real thesis anchor (`supply_chain/config.py:44`).

## The eight binding amendments

1. **Emergency-reserve overlay, normal MFSC flow remains.** Program G does NOT replace downstream
   logistics with a one-truck abstraction. Normal flow serves aggregate demand `D_t`; the convoy
   moves ONLY the emergency 10,000-ration reserve between SB and A/B. Isolates prepositioning value
   from base-system value and correctly reuses the earlier reserve-alert study.
2. **Convoy decisions occur when the resource is available, nominally every 48 h — not weekly.** The
   weekly cadence left the convoy parked at SB for five days; "persistent convoy location" was false.
   With 48h decisions the persistent state is genuinely the convoy location/availability PLUS reserve
   depletion, deployed-ration location and A/B backlog. (Shift decisions, if S2 is ever enabled,
   stay weekly.)
3. **Reserve = 10,000 rations as two full 5,000-ration loads; no in-episode replenishment; inventory
   creation forbidden; 56-day horizon.** 5,000 is defensible as the thesis **batch anchor**
   (`RATIONS_PER_BATCH`), NOT as historical vehicle capacity. No replenishment in the primary study —
   route-aware replenishment with a frozen lead time is a later sensitivity, never a code-wished top-up.
4. **R22 pauses both outbound and return travel; primary v1 has NO destruction.** Bidirectional LOC
   closure; cargo and vehicle survive; the convoy stays unavailable until physical return. A
   destruction variant (replacement lead, physical loss) is a separate future contract, not mixed in.
5. **Spatial control is screened with S1 fixed; S2 is a secondary, liveness-gated extension.** In
   Program L, S1/S2/S3 gave identical ReT/service for all positive buffers — shift value is not
   assumed. S2 re-enters only after a preflight shows changing shift alters SB stock / rebuild
   capacity / service-loss / reachable states. Program G first isolates SPATIAL value, then tests
   production–transport coordination — not both in one six-action causal soup.
6. **The static bar includes strategic prepositioning and exact open-loop schedules**, not just the
   six constants. Comparing a sequential learner to "always A / always B / always HOLD" would repeat
   the Track B error (learner gets sequences, static gets a statue). Primary comparison:
   `π_observable − max(strategic posture, open-loop doctrine, constant doctrine)` inside the same
   resource envelope (convoy missions, unavailable hours, forward-reserve ration-days, reserve
   consumed). Simple observable doctrines are reported SEPARATELY (they are heuristics, not statics).
7. **Signal quality is sensitivity/FPR, not aggregate accuracy.** Scalar accuracy is prevalence-
   dependent — a "nothing happens" alarm scores well when surges are rare. Two tiers: moderate
   (sens 0.70, FPR 0.20), high (sens 0.85, FPR 0.10); lead 7 or 14 days; binary, weekly, per-CSSU.
   A **third placebo — wrong-CSSU (swap A/B)** — is ADDED: it is the most direct test that value comes
   from knowing WHERE, not merely from receiving an exciting bell.
8. **Primary envelope is the 16-cell connected-region design**, not the 24-cell grid. Persistence
   4–8 weeks already dwarfs the decision epoch; the important omitted axis is **A/B concurrency**
   (if both surge together the value of knowing the destination collapses; if never together it is
   artificially easy). Grid = 2 signal tiers × 2 leads × 2 surge weights × 2 commonality levels
   (localized 0.10, partially-common 0.35). Persistence 6–8 weeks becomes a later sensitivity on the
   passing connected component.

## Naming correction

`Base/T/TR/TS/TRS/TRSC` is a **nested mechanism ladder / staged component decomposition**, NOT a
factorial (it omits R-without-T, S-without-T, RS-without-T). Identified contrasts: `T−Base`, `TR−T`,
`TS−T`, `TRS−TR`, `TRS−TS`, `TRSC−TRS`. Report it as such; do not call it a factorial.

## Corrected experimental sequence (frozen)

- **G0** — physics preflight, no tapes 980001+: A+B conservation, single convoy, full-load, finite
  reserve, no replenishment, R22 outbound/return pause, label-swap symmetry, same-prefix-same-action
  identity, signal ledger, placebos, action masks.
- **G1** — one central cell, all six ladder arms → which component opens the channel.
- **G2** — only `TRS` and `TRSC` over the 16 cells (not all six arms in every corner — that is rigor
  by tonnage).
- **G3** — if a connected component of ≥2 cells passes: 60 fresh calibration tapes, tree + hysteresis,
  exact open-loop comparator, placebos, resource frontier; train/eval on the UNIFORM distribution over
  the eligible connected component, never the max-ReT cell.
- **G4** — open holdout `1000001+` only after freezing region, tree, thresholds, observations,
  comparator, analysis code.
- **G5** — only then rollout/MPC, contextual bandit, MaskablePPO. Persistent/reset/frozen stays out
  until a virgin-tape win.

## Terminal condition of this amendment

V1.1 is frozen. `build_gate.authorized = false`. G0 is authorized ONLY after Garrido signs
`docs/PROGRAM_G_DOMAIN_SIGNOFF_TEMPLATE_2026-07-12.md` and every parameter in
`docs/PROGRAM_G_INTERVENTION_LEDGER_2026-07-12.md` has a source/falsifier/sensitivity/claim-limit.
Program G does NOT block the already-supported manuscript. Freezing this envelope is ~two days of
work plus one Garrido meeting; if it becomes another week of plans-about-plans, the known disease is
recurring.
