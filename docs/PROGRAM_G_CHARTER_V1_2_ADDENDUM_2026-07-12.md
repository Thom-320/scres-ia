# Program G — charter V1.2 addendum (frozen 2026-07-12)

Status: **FROZEN. BUILD (G0, tapes 980001+) REMAINS UNAUTHORIZED PENDING GARRIDO SIGN-OFF.**
Supersedes V1.1 (`program_g_domain_envelope_v1_1`). Source: external dictamen #3
(`docs/external_assessments/program_g_dictamen_v3_2026-07-12.md`), critically evaluated — not adopted
wholesale. Codex out; verifier owns builder+verifier.

## Why V1.1 was wrong (verified, not asserted)

`5000 rations / 48 h = 2500/day`. Thesis daily demand is 2400–2600/day; shift capacities are
S1 = 2564/day (`config.py:70` `RATIONS_PER_SHIFT`), S2 = 5128, S3 = 7692; Op9 one-week level = 15,750
(`config.py:694`). Two consequences, both fatal to V1.1:

1. **Transport is the binding constraint.** A 2500/day convoy-equivalent cannot move S2's 5128/day.
   Dynamic shift control (S1/S2 in the action space) is a **dead dimension** — S2 fills SB with
   immovable stock without increasing delivery. This is exactly the "observe-move-log a dimension that
   doesn't change the endpoint" failure the repository already spent weeks discovering.
2. **The V1.1 emergency-reserve overlay was non-binding.** An overlay convoy on top of a *complete*
   normal flow would deliver ~2500 (normal) + ~2500 (convoy) = ~5000/day against 2500/day demand → the
   convoy rarely binds → no scarcity → the mechanism dies. The convoy must BE the scarce downstream
   transport, not an extra asset.

## The three BINDING changes (V1.2)

1. **Action space 6 → 3: `a_t ∈ {A, B, HOLD}`.** Dynamic shift control removed from the primary study;
   production fixed at S1. Exact oracle collapses from 6^8 = 1,679,616 to 3^4 = 81 (4wk) / 3^8 = 6561
   (8wk) sequences per state — enumerable, not "a religious ceremony to warm processors."
2. **The weekly action is a dispatch PRIORITY, not a single weekly departure.** Choosing `A` in week k
   means: every time the convoy returns to SB and can depart that week, its only authorized destination
   is A; if A lacks storage/load/open route the convoy WAITS — it does NOT auto-reorient to B. This
   preserves temporal commitment, the cost of a bad decision, multiple cycles/week, a small action
   space and manageable exact branching. A single weekly departure would leave the convoy idle ~5 days
   (the "deliberately absurd operation the algorithm then fixes" trick reviewers and children both spot).
3. **Static comparator = exhaustive periodic calendars, period 1–4 = 120.** `3+9+27+81 = 120`
   calendars, deduped, under CRN, convex-hulled on resources, frozen on calibration, evaluated on
   holdout — this is the best full-contract static. Comparing a learner to only "always-A/B/HOLD" would
   repeat the Track B statue-vs-sequence error.

## Other adopted corrections
- **10,000 = initial SB finished stock** (genealogy-tracked), not a separate abstract reserve; ordinary
  production keeps feeding SB; dispatch carries `q = min(5000, I_SB, free capacity at destination)`; no
  lateral A↔B transshipment, no return to SB, no inventory creation on retarget. (10k is a researcher
  param — thesis Op9 week level is 15,750; frozen as a later single sensitivity, not a selection axis.)
- **Surge multiplier 1.25 / 1.50, NOT 2.0 / 3.0.** A sustained 4–8 week 50% regime is already severe;
  a one-day R24 spike and a two-month regime are different objects with the same big number.
- **Primary-screen risks: R22 ON** (localized per route, current level); **R24-native OFF** (tempo
  replaces it), **R23 OFF** (destroying a CSSU makes the destination obvious/moot), **R11/R21/R3 OFF**.
  Reintroduce the rest only after a pass, never to select the environment.
- **Tempo**: semi-Markov independent per CSSU; dwell U{4,5,6} short / U{6,7,8} long; transitions
  low→routine, surge→routine, routine→{low 0.5, surge 0.5} → ~50/25/25 → co-surge ≈ 6.25%. Demand base
  U{2400,2600} 6d/wk, split by tempo weights, `D_A+D_B` identical ACROSS POLICIES on a tape (NOT forced
  equal to original thesis demand when regimes are active — that would recreate artificial conservation
  and delete the scarcity).
- **Signal = balanced accuracy** q∈{0.65,0.75,0.85} (sens = spec = q; precision reported, not input);
  target future local surge at t+L, L∈{1,2} weeks; **tempo/demand content only** (planned troop level,
  deployment calendar pressure), route_threat_score EXCLUDED from primary (mixing demand and R22 makes
  accuracy uninterpretable). Placebos: circular block-shuffle per CSSU + post-onset delayed, on rollout.
- **Arms**: TR distinction = physical convoy memory; **TRS is the primary physical arm; TRSC deferred**
  (no defensible holding/expiry coefficients — report a resource/Pareto vector, not fabricated money).
- **Grid 24 cells** = 2 persistence × 3 signal quality × 2 lead × 2 surge; adjacency = one level on one
  axis; valid region = connected component ≥2 passing cells; primary cell = first in the frozen
  least-to-most-favorable order (surge 1.25→1.50, signal 0.65→0.75→0.85, lead 1→2, persistence
  short→long) inside a valid component — the minimally sufficient condition, not the max-ReT theme park.
- **episode_weeks: 52.**

## Verifier's critical adjudication (where I did NOT simply adopt)
1. **Convoy ontology is a genuine open conflict between the two reviews.** Dictamen #2 preferred an
   overlay; dictamen #3 (and the throughput physics) favor the convoy AS the downstream transport
   (Option A). I adopt Option A as the frozen recommendation BECAUSE the overlay is non-binding by the
   2500/day math — but this is now the **decisive Garrido question** in the sign-off, not a unilateral
   call. If Garrido says the real system keeps a separate emergency asset, we revisit.
2. **Signal metric**: balanced accuracy (sens=spec=q) is cleaner for cross-cell comparison and adopted,
   but real advance signals rarely have sens=spec (specificity is usually higher for rare surges) →
   one asymmetric tier is kept as a documented sensitivity.
3. **Grid axis**: dictamen #2 made A/B commonality a primary axis (16 cells); dictamen #3 keeps 24 cells
   and handles commonality via A/B tempo INDEPENDENCE (emergent 6.25% co-surge). I adopt independence as
   primary (thesis-defensible, natural) and correlated-surge as a documented sensitivity — reconciling
   both reviews without a 6-arm × 24-cell tonnage explosion.

## Terminal condition
V1.2 frozen. `build_gate.authorized = false`. G0 is authorized ONLY after Garrido answers the sign-off
template — above all the **convoy-ontology question** (transport vs overlay) and the surge-magnitude and
dead-shift-dimension confirmations. Program G does NOT block the already-supported manuscript.
