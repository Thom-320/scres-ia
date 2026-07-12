# Program G external dictamen #3 (ChatGPT-Pro, 2026-07-12) — archived verbatim-substance

Provenance: user-relayed external review, 2026-07-12. Third in the Program F/G series
(#1 → Program F risk-mitigation; #2 → Program G spatial + my V1.1 amendment; #3 → this, the sharpest,
which corrects V1.1). Verdict: **continue Program G, but revise the minimal envelope to v1.x before
build.** Verifier adopted it as DOMAIN_ENVELOPE_V1.2 (`contracts/program_g_domain_envelope_v1_2.json`,
`docs/PROGRAM_G_CHARTER_V1_2_ADDENDUM_2026-07-12.md`).

## The decisive physics claim (VERIFIED first-hand)

`5000 rations / 48 h = 2500 rations/day`, which ≈ the thesis daily demand 2400–2600/day and Op10
daily flow. Shift capacities: S1 = 2564/day (`RATIONS_PER_SHIFT`, config.py:70), S2 = 5128, S3 = 7692.
Therefore with a 2500/day convoy-equivalent, **transport is the binding constraint and S2 does not
increase final delivery** — it fills SB with immovable stock. Dynamic shift control is a dead
dimension in the primary study. Op9 one-week level is 15,750 (config.py:694), so 10,000 is a
researcher design parameter, not a thesis level. Consequence: an "emergency overlay" convoy ON TOP of
a complete normal flow would deliver ~5000/day vs 2500/day demand → convoy rarely binds → no scarcity.
The overlay (my V1.1) was non-binding; the convoy must BE the scarce downstream transport.

## Decisions table (dictamen #3)

| Element | Decision |
|---|---|
| Program G | Continue |
| Verifier gates | Commit before any build |
| Single convoy | Keep, pending Garrido validation |
| Capacity 5000 | Keep as **aggregate convoy-equivalent**, not a historical vehicle fact |
| 24h out + 24h return | Keep as declared extension |
| Initial location | SB |
| Partial load | Allowed; consumes the full cycle |
| R22 | Pause outbound and return; no permanent vehicle destruction |
| Weekly action | A **weekly dispatch PRIORITY**, not a single weekly departure |
| Dynamic S1/S2 | **Remove from primary action space** |
| Primary production | S1 fixed |
| Initial reserve | 10,000 at SB (= 2×5,000 loads; researcher param, not thesis level) |
| Expiry | Not included without physical data |
| Holding | Measure as inventory-time, not invented money |
| Tempo | Semi-Markov independent A/B |
| Signal "precision" | Replace with balanced accuracy (sens=spec=q) |
| 24-cell grid | Keep |
| Static comparators | Expand to periodic calendars, not just three constants |
| PPO | Stays blocked |

## Key mechanism changes
- **Action space 6 → 3**: `a_t ∈ {A, B, HOLD}`, a weekly priority applied to EVERY departure
  opportunity that week (multiple 48h cycles); no auto-reorientation to the other CSSU; the convoy
  waits if the chosen destination lacks storage/load/open route. Enumerable oracle: 3^4=81 (4wk),
  3^8=6561 (8wk) — vs 6^8=1,679,616 for six actions ("a religious ceremony to warm processors").
- **10,000 is just initial SB finished stock** with genealogy; ordinary production keeps feeding SB;
  each dispatch carries `q = min(5000, I_SB, free capacity at destination)`; no lateral A↔B
  transshipment, no return to SB, no inventory creation on retargeting. No separate abstract reserve.
- **Risks in primary screen**: R22 ON (localized per route, current level); R24-native OFF (tempo
  replaces it); R23 OFF (destroying a CSSU makes the destination obvious/moot); R11/R21/R3 OFF.
  Reintroduce the rest only AFTER a pass, never to select the primary environment.
- **Tempo**: semi-Markov independent per CSSU; dwell U{4,5,6} (short) or U{6,7,8} (long); transitions
  low→routine, surge→routine, routine→{low 0.5, surge 0.5} → ~50% routine / 25% low / 25% surge →
  simultaneous surge ≈ 0.25² = 6.25% of weeks. Demand: base U{2400,2600} 6d/wk, split
  D_A = round(D_base/2 · m(Z_A)), D_B = D_ProgramG − D_A; multipliers low 0.75, routine 1.0, surge
  **1.25 / 1.50** (NOT 2.0/3.0 — a sustained 50% over 4–8 weeks is already severe; a 1-day R24 spike
  and a 2-month regime are different objects). Conservation: D_A+D_B identical ACROSS POLICIES on the
  same tape (NOT equal to the original thesis demand when regimes are active).
- **Signal**: balanced accuracy q∈{0.65,0.75,0.85} (sensitivity=specificity=q); predicts local surge
  at t+L, L∈{1,2} weeks; **tempo/demand only** (planned troop level, deployment calendar pressure) —
  NOT route_threat_score (mixing demand and R22 into one variable makes "accuracy" uninterpretable).
  Placebos: circular block-shuffle per CSSU (preserve positive rate + run lengths), and post-onset
  delayed (L+1). Evaluate on rollout, not accuracy.
- **Static comparators**: exhaustively enumerate all periodic policies of period 1–4 = 3+9+27+81 =
  **120**, dedup, CRN, convex hull, freeze on calibration, evaluate on holdout = the best full-contract
  static. Plus threshold, hysteresis, depth-3 tree, rollout/MPC, contextual bandit, PPO.
- **Arms**: TR distinction = physical convoy memory (resource does NOT carry availability across weeks
  when absent). **TRS is the primary physical arm; TRSC (cost) is deferred** — no defensible holding/
  expiry coefficients exist; the resource envelope already guards against buying the improvement.
- **Grid 24 cells** = 2 persistence × 3 signal quality × 2 lead × 2 surge. Adjacency: differ by one
  level on one axis. Deterministic selection order (least→most learning-favorable): surge 1.25 before
  1.50; signal 0.65 before 0.75 before 0.85; lead 1 before 2; persistence 4–6 before 6–8. Primary cell
  = first in that order inside a valid connected component (≥2 passing cells) — the minimally
  sufficient condition, not the max-ReT theme park.
- **episode_weeks: 52.**

## Verifier's critical adjudication (not blanket adoption)
1. The transport-binding argument is VERIFIED and corrects my V1.1 overlay → adopt convoy-as-downstream-
   transport. BUT dictamen #2 preferred an overlay; the two reviews conflict, so the convoy ontology is
   the **decisive Garrido question** (is the convoy the real downstream transport, or an extra asset?).
2. Signal balanced-accuracy (sens=spec=q) is cleaner for cross-cell comparison than my V1.1 sens/FPR;
   adopted, but real signals rarely have sens=spec (specificity usually higher for rare events) → keep
   one asymmetric tier as a documented sensitivity.
3. Grid: adopt dictamen #3's 24 cells (persistence×signal×lead×surge); reconcile dictamen #2's
   commonality axis by making A/B-tempo INDEPENDENCE the primary (emergent 6.25% co-surge) and
   correlated-surge a documented sensitivity, not a primary axis.
