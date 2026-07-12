# Program G — G1 result (central cell, pre-RL, under PI autonomy) — 2026-07-12

Status: **FIRST POSITIVE OBSERVABLE ADAPTIVE-CONTROL RESULT IN THE PROJECT — but the value is
spatial-commitment-under-persistence, NOT advance information.** Built and run under PI autonomy
(`docs/PROGRAM_G_AUTONOMY_AUTHORIZATION_2026-07-12.json`) to bring Garrido a result to approve on.
No PPO, no virgin tapes. Stylized weekly-resolution extension; service-loss (unmet rations) is a
disclosed Program-G proxy, not `ret_excel`.

## What was built
`supply_chain/program_g.py` (V1.2 physics): two CSSU (A/B) with separate inventory/backlog/demand,
one aggregate two-day convoy-equivalent (2500 rations/day = thesis flow) as the scarce shared
downstream transport, weekly dispatch priority `{A,B,HOLD}` (all cycles, no auto-reorient), unified SB
stock feeding on S1 production, semi-Markov tempo per CSSU, balanced-accuracy surge signal, R22 route
pause. G0 physics preflight `tests/test_program_g_physics.py` **8/8 green** (determinism, A/B
label-swap symmetry, exogenous demand, HOLD dominated, oracle lower bound, no inventory creation,
weekly-priority multiple departures, iid≠persistent tempo). Exact 3^4 oracle; calibration 990001+,
disjoint holdout 1000001+.

## Result (holdout, best static frozen on calibration; 200/200 tapes)

| Arm | tempo | signal | H_PI (static−oracle) | H_obs (static−observable) | η | convoy missions obs/static |
|---|---|---|---:|---:|---:|---|
| Base | iid | no | +767 | **+79** (CI95 crosses 0) | 0.10 | 12.0 / 12.0 |
| T | persistent | no | +766 | **+522** [350, 716] | 0.68 | 12.0 / 12.0 |
| TR | persistent+memory | no | +766 | +522 | 0.68 | 11.8 / 11.8 |
| TS | persistent | yes | +766 | +532 | 0.69 | 12.0 / 12.0 |
| TRS | persistent+memory | yes | +766 | +532 [357, 716] | 0.69 | 11.8 / 11.8 |

Component attribution (Δ in converted headroom H_obs):
- **Persistence (T − Base) = +443** — the dominant driver.
- **Advance signal (TS − T) = +9** — nearly redundant given observable inventory.
- **Convoy memory (TR − T) = +0** — negligible.

## Honest reading
1. **Spatial headroom is real and material**: a clairvoyant open-loop policy beats the best of 120
   frozen periodic calendars by +766 rations (~2.9% of oracle service-loss), CI95 lower > 0.
2. **It IS observably convertible** — the FIRST time in the project an observable closed-loop policy
   (cover-based: dispatch to the lower days-of-cover CSSU) materially and OUT-OF-SAMPLE beats the best
   full-contract static (H_obs +522, η≈0.68), **resource-matched** (11.8 vs 11.8 convoy missions — not
   bought). This is qualitatively unlike Program E/F, where η≈0.
3. **But the value is PERSISTENCE + contemporaneous spatial state, not the advance signal.** When tempo
   persists, current inventory/cover already reveals which CSSU is under stress; the advance alert adds
   ~9 rations. The wrong-CSSU placebo failing (+1973) shows a *misleading* signal hurts — not that a
   correct one helps materially over watching inventory. So `I(O_contemporaneous; a*) > 0` strongly;
   `I(advance_signal; a* | inventory) ≈ 0`.

## Implication
Program G converts clairvoyant headroom that DRA-2b/Program E could not — because the decisive state
(which CSSU is starving) is here both observable AND directly actionable (send the shared convoy
there). This passes the pre-RL eligibility gate at the central cell (H_obs CI95>0, η≥0.30,
resource-matched, value-of-observability confirmed). The open question for a learner (G5): can PPO/tree
capture the residual η 0.68→higher, or is a cover heuristic already the ceiling? And the sharpened
scientific claim: the headroom is spatial-commitment observability, and **advance information is nearly
redundant once contemporaneous inventory is observed** — a cleaner, more surprising result than "the
signal helped."

## G2 — 24-cell screen (added): PROMOTABLE CONNECTED REGION (pre-RL gate PASSES)

`scripts/run_program_g_screen.py`, grid 2 persistence × 3 signal × 2 lead × 2 surge, TRS arm,
cover-signal observable, best static frozen on calibration, holdout evaluation, adjacency = one level
on one axis. Result (`results/program_g/g2/verdict.json`): **12/24 cells pass, one connected component
of 12 → `G2_PROMOTABLE_CONNECTED_REGION`, promotable = True.**

- **The decisive axis is surge magnitude**: all 12 passing cells are surge **1.50** (H_obs_lo ≈ +508,
  η ≈ 0.76); all 12 failing cells are surge **1.25** (η ≈ 0). At 1.25 the A/B demand differential is
  too small to make the shared convoy a binding spatial commitment.
- **Signal quality (0.65/0.75/0.85) and lead (1/2) do NOT discriminate** within surge 1.50 — every
  combination passes equally, reconfirming G1: the advance signal is nearly redundant; the value is
  spatial commitment under sustained stress.
- **Minimally-sufficient cell** (frozen least-favorable order, surge 1.25 fails so 1.50, then signal
  0.65, lead 1, persistence short) = `Pshort_Q65_L1_S150` — NOT the max-outcome cell.

This is the **FIRST time the project's pre-RL eligibility gate has PASSED**: a promotable connected
region of ≥2 adjacent cells where an observable, resource-matched, OOS policy converts material
clairvoyant headroom. RL is now WARRANTED (G5) — though the interpretable cover heuristic already
converts ~0.68–0.76, so the open question is whether a learner adds incremental value over it.

## Discipline / limits (for Garrido and Reviewer #2)
- IN-SAMPLE at ONE central cell (high signal / lead 1 / surge 1.50 / short persistence). The 24-cell
  screen + ≥2 adjacent passing cells still gate any learner. Result is OOS across tapes but not across
  cells.
- Service-loss proxy, not `ret_excel`; stylized weekly model; convoy ontology = Option A (disclosed,
  pending Garrido confirmation).
- Every arm reported including the near-null Base and the near-redundant signal; no cell selected by
  max outcome. If Garrido rejects the convoy ontology or ranges, this reclassifies, it is not defended.
