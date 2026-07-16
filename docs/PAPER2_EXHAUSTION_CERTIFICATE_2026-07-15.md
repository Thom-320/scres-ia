# Paper 2 adaptive-headroom search — exhaustion certificate

**Date:** 2026-07-15 · **Science commit:** `adbfb8f`
**Machine-readable:** `results/paper2_search/paper2_exhaustion_certificate_2026-07-15.json`
**Status:** `PAPER2_SEARCH_BOUNDARY_CERTIFIED__NO_POSITIVE_INSTANCE__EXACT_REOPENERS_NAMED__OAT_SCOPE_DISCLOSED`

> ⚠️ **SCOPE (added on methodological review).** The thesis-native risk evidence is **one-factor-at-a-time**
> (Garrido Cf1–Cf20 + one-at-a-time R11–R24). Per Saltelli et al., **OAT cannot detect interactions** (factors are
> never varied concomitantly) and covers only a **'hypercross'** of negligible volume. **The negative is therefore
> established ALONG THE AXES; the simultaneous multi-risk (wartime) INTERIOR is unexplored by that design.** It is
> pre-registered for discovery + frozen virgin confirmation in
> `research/paper2_exhaustive_search/WAR_SCENARIO_INTERACTION_SENSITIVITY_PREREGISTRATION_2026-07-15.md`.
> **This certificate is provisional on that probe.**

**Primary endpoint throughout:** `ret_excel_request_snapshot_v2` via `supply_chain.episode_metrics.compute_episode_metrics`.
Metric substitution, Cobb–Douglas promotion, hand-rolled ReT, and temporal-index promotion are forbidden;
time-resolved indices are secondary sensitivities only.

---

## 1. Terminal finding

Across the **thesis-native envelope** — 4 decision surfaces, 3 independent programs, 2 physics generations,
and now **Garrido's own Cf1–Cf20 risk-escalation table** — and across **one engineered extension with ~7×
more clairvoyant headroom**, **no decision contract converts perfect-information headroom into deployable,
resource-honest, out-of-sample-stable observable value.**

**No positive instance exists. Adaptive/learned control is not warranted in this system under the stated gates.**
**Zero PPO seeds were wasted** — every gate killed its candidate before training.

The mechanism is consistent and mechanistically identified:
1. The DES is **structurally over-buffered** → constants are near-optimal.
2. The **optimal posture is invariant to the risk regime** (new, decisive — §2.6).
3. **Commitment lags defeat clairvoyance** — a non-privileged EWMA detector beats the true-state instant
   switcher ×2.3 (Track C), so even perfect information cannot be converted at the weekly epoch.
4. Where clairvoyant headroom does exist, it fails **observability**, **resource-honesty**, or **out-of-sample
   prospective consistency**.

## 2. Quantitative ceilings

### Thesis-native envelope — **NEGATIVE**

| Program | Ceiling | Observable outcome |
|---|---|---|
| Full-DES max H_PI (DRA2b_finite_convoy) | **0.0221** | H_obs ≈ 0 — Program E: **0/10 PPO seeds**, tree, heuristic, convex mixture all fail |
| Stylized VoI atlas (64 cells) | H_PI mean 0.0135 / max 0.0246 | H_obs mean **−0.0079**, 80% ≤ 0, **η mean −0.79** — ⚠️ **RETRACTED** (the 2 positive cells were **shed-to-win**; excluded from inference, directional only) |
| Track C campaign oracle (R22/R23/R24, Ch7-motivated) | switching headroom **6.5e-05** (oracle 2.8e-05), 24/24 tapes, bar 5% | **NULL** — constants near-optimal even under engineered non-stationarity with priced costs. **Hysteresis beats clairvoyance ×2.3.** |
| L(e−1) program | clairvoyant constant bound +1.6–3.6% | **STOP** — optimal weekly shift = **S1 in 100%** of branched states; zero within-tape switching value |
| Route recourse (Program L / R03) | full-DES H_PI **≤ 0.005** at every R22 regime (24h→720h) | **STOP** — buffer-intercept (stylized screen showed 0.15; full-DES collapsed ~30×) |
| **Garrido risk escalation (2026-07-15)** | **max H_profile_safe 6.93e-05 vs bar 0.01 → 144× below** | **0/63 cells**, `passing_doors=[]`, guardrails all clean |

#### 2.6 The decisive new result — risk escalation creates no headroom

Using **Garrido's own escalation methodology** (exact **Cf1–Cf20** + one-at-a-time **R11–R24** + impacts
1×/1.5×/2×; **R3 black-swan frozen and excluded**), **45 profiles × 18 constant postures × 6 seeds =
4,860 ten-year evaluations** on the canonical metric:

> **The optimal constant posture is INVARIANT across all 45 risk profiles, at every budget cap.**
> Escalating the thesis's own recurrent risks **does not change what you should do**.

⚠️ **Valid along the OAT axes only** — a reversal requiring e.g. R22 **and** R24 elevated *concomitantly* could
not have been detected by this design (see scope note above).

Risks genuinely degrade physical resilience (ReT 0.53 → 0.20 under escalation) but create **zero
regime-tailoring headroom** (along the tested axes) — and this is **not** a guardrail/shed rejection: all guardrails are clean;
the tailoring simply has no value. Custody: result `e4a3d4a0`, execution `01c1d9d` (child of frozen
`6794fe6`, contract byte-identical), independently recomputed. Scope: development (6 seeds) — a cheap door
that did not open; it requires no confirmation to stay closed, since the burden was to *demonstrate*
headroom and it is absent by two orders of magnitude.

### Extended envelope — Program O (two non-fungible products) — **CLOSED**

| Stage | Result |
|---|---|
| Full-DES **H_PI** | **0.15151** (simultaneous safe LCB95 **0.11562**), exact **fungible-null = 0.0**, 25,177-episode parity, **conserved throughput**. Custody `f5f2da8d` / verdict `98ce2ce`. **The only material ceiling in the entire search** — and the only one to survive real Op9–Op12 buffering. |
| H_obs — label-only HMM | **REFUTED**: oracle-true (ρ,share) changed **0/192** trajectories (action is share-magnitude-invariant) |
| H_obs — state-rich classical | `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND` (`d67ac97a`) |
| Dual-resource diagnostic | `DIAGNOSTIC_STABLE_SIGNAL_FIXED_CLOCK_ONLY` (`e48606e7`): under fixed-clock, belief-MPC beats **all** information placebos over 3 connected cells (incremental state value LCB95 **+0.025…+0.073**); pay-per-use fails. **Development only.** |
| **Fixed-clock-physical OOS validation** | **`STOP_PROGRAM_O_CLASSICAL_HOBS_VALIDATION`** (`09ec3f16`): sealed tapes opened once; **rho90_share90 favorable on 26/48 vs required 34** (mean +0.083 but **bimodal/unstable**). Closed with **no rescues**. |

*Bug disclosure:* the validation executor mixed 69 incompatible-scale estimands into one simultaneous
critical (23.65M), voiding the LCB/placebo/guardrail flags. **The verdict is independent of that bug** — the
26/48 favorable-tape count is a raw per-tape count computed pre-bootstrap.

## 3. Exact Garrido questions (the only reopeners)

Canonical batch: `research/paper2_exhaustive_search/garrido_face_validation_questions.md` (Q1–Q13).

| Question | Reopens if | Status |
|---|---|---|
| **Q11 / R09 — mission expiry** | Hard deadlines with permanent abandonment, **tighter** than 24–120h recovery, **AND** doctrinal triage authority | **OPEN — strongest thesis-native reopener** |
| **Q6/Q7 — integrated shared resource** | ONE named resource mutually-exclusively committed across plant / LOC / theatre; or Maintenance Battalion teams **fewer** than disabled sites (forced serialization) | **OPEN** |
| **Q13 — Program O construct** | ≥2 mutually non-substitutable ration classes share the Op5–Op7 bottleneck with uncertain, persistent, advance-observable mix | OPEN — but restores representativeness of the **ceiling only**; Program O's H_obs already failed OOS |
| **Q14 — freight economics** | Fixed-clock reserved vs pay-per-use fleet | OPEN — **not decisive**: the fixed-clock OOS validation already failed, so Q14 only *scopes* a retired development finding |
| **Q2 / R03 — route recourse** | ≥2 routes, one finite fleet, persistence + pre-dispatch warning | **BAR RAISED** — already has a full-DES negative; would additionally require finite downstream storage |

## 4. Residual untested thread (disclosed)

**Within-event intervention timing** (endogenous/impulse control of *when* to act, attacking Track C's
fixed-168h-cadence + commitment-lag root cause) is **un-licensed**: the frozen rule gated it behind the
risk-sensitivity door, which did not open (R2_frequency @ cap 0.5 = 5.0e-05 vs bar 0.01).

Honest assessment: its motivation is **substantially undercut** — the risk screen shows the optimal posture
is invariant across all 45 profiles, so better-timed *switching between postures* has nothing to exploit at
the regime level; and Track C already showed hysteresis beats clairvoyance under commitment lags. It is
**not strictly refuted** for within-event response timing, and would require a new preregistration with
independent motivation.

## 5. Claim boundary

| | |
|---|---|
| full-DES H_PI established | **true** — Program O only (0.152), a disclosed researcher extension |
| H_obs established | **false** (nowhere) |
| learner authorized | **false** |
| Paper 2 confirmed | **false** |
| Paper 3 authorized | **false** |
| positive instance found | **false** |

## 6. Disposition

The portfolio result is the **negative**: *when not to train*. It is unusually well-controlled — it holds
across four decision surfaces, three independent programs, two physics generations, Garrido's own
risk-escalation table, and an extension engineered with ~7× more clairvoyant headroom — and it is
**mechanistically explained**, not merely observed. The two exact domain questions that could reopen a
thesis-native family (Q11/R09, Q6/Q7) are stated falsifiably above and are the only legitimate path to a
positive without a new disclosed extension.
