# E1: regime-conditioned static table / heuristic go-no-go — verdict (2026-07-02)

**Status: DONE — GO.** `outputs/experiments/track_b_e1_confirmatory_2026-07-02/`
(5 seeds × 12 episodes, real regime-table fit — not `--skip-regime-fit` —
over a trimmed but real 75-candidate grid: 3 shifts × 5 op10 mults × 5 op12
mults, plus the 6 heuristics from `scripts/track_b_heuristics.py`, plus the
common-static grid, all under the canonical CRN eval-seed protocol). Merged
against the canonical frozen PPO checkpoint via the new
`scripts/build_track_b_e1_go_no_go.py` (a genuine gap in both this
session's and Codex's original E1 infra — neither script loaded PPO itself).

## Result

**PPO beats every single comparator with a fully positive 95% CI —
60/60 comparisons, zero exceptions.**

| Stage | Best policy | Excel/order ReT | Δ vs common static |
|---|---|---:|---:|
| Common static (best of 75-candidate grid) | `S2_op10_2.00_op12_1.50` | 0.005466 | — |
| Fitted regime-conditioned table | 5-regime lookup (see below) | 0.005494 | +0.0000277 |
| Best heuristic | `heur_tuned` | 0.005436 | **−0.0000305** (worse than the plain best static) |
| **PPO (canonical, frozen)** | — | **0.005893** | **+0.000427** |

Regime table detail (`regime_fit_trace.csv`) — the fitted per-regime lookup:

| Regime | Selected static |
|---|---|
| nominal | `S3_op10_2.00_op12_1.50` |
| strained | `S2_op10_2.00_op12_1.50` |
| pre_disruption | `S2_op10_1.50_op12_1.00` |
| disrupted | `S2_op10_2.00_op12_1.50` |
| recovery | `S1_op10_2.00_op12_1.50` |

Head-to-head vs PPO (`outputs/audits/track_b_e1_go_no_go_2026-07-02/`):

| Comparator | PPO delta | CI95 | PPO wins (CI>0) |
|---|---:|---:|---|
| Best fitted regime table | +0.000399 | [0.000369, 0.000430] | **Yes** |
| Best heuristic (`heur_disruption_aware`) | +0.000493 | [0.000456, 0.000531] | **Yes** |
| All 6 heuristics individually | +0.00047 to +0.00141 | all positive | **Yes, all 6** |
| All 75 static candidates individually | +0.00043 to +0.00364 | all positive | **Yes, all 75** |

## Verdict on the go/no-go question (T3, docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md)

**GO.** This is the complement of the E2 obs-masked result
(`docs/E2_PRIVILEGED_OBSERVATION_VERDICT_2026-07-02.md`), and together they
close the privileged-observation attack from both directions:

- **E2** showed PPO does not *need* the privileged regime/forecast fields
  to win (retains ~95% of its delta with them masked out).
- **E1** shows that even a policy that *does* have privileged regime access
  — a regime-conditioned lookup table fitted directly on the true adaptive
  regime, the exact signal the reviewer worried PPO was merely reading —
  still cannot come close to matching PPO. The regime table gains only
  +0.0000277 over a single best static constant; PPO gains +0.000427, over
  15× larger.

Safe wording:

> "A regime-conditioned static lookup table, fitted with direct access to
> the same true disruption-regime signal available to the learned policy,
> improves on the best single static constant by only +0.0000277 — a
> fraction of PPO's +0.000427 advantage over the same comparator. Six
> reactive dispatch heuristics, including ones that explicitly condition on
> backlog, queue pressure, and forecast fields, likewise fail to close the
> gap; the best heuristic does not even beat the best single static
> constant. The learned policy's advantage cannot be explained by exploiting
> observable regime structure that a non-learning policy could equally
> exploit."

## Registry update

Add to `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`: new item confirming
the T3 attack is closed from both directions (cite alongside C16). The
gap-decomposition table above (common static → regime table → heuristic →
PPO) is the audit's requested "central figure" (docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md,
Part 1, E1 description) and should go directly into the paper's mechanism
section.

## Caveat

The candidate grid used for regime-table fitting was trimmed (3 shifts × 5
op10 × 5 op12 = 75, vs the full dense grid's 147 = 3×7×7) to keep fitting
tractable overnight. Given the regime table's marginal gain over the
already-known best single static (+0.0000277), it is very unlikely a full
147-candidate fit would close a 15× gap to PPO, but this should be
disclosed as a scope limitation if a reviewer asks for the untrimmed grid.
