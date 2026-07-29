# Track B canonical seed expansion (seeds 6-10) — verdict (2026-07-02)

**Status: DONE.** `outputs/experiments/track_b_seed_expansion_2026-07-02/track_b_seed_expansion_6_10_claude/`,
run on the OVH VPS via `scripts/run_track_b_benchmark.py` with the exact
canonical protocol (`control_v1`, `adaptive_benchmark_v2`, h104,
`--learning-rate 0.0003 --n-steps 1024 --batch-size 256 --n-epochs 10
--train-timesteps 60000 --eval-episodes 12`), seeds 6-10.

Requested by four independent Reviewer-#2-style reviews as the single
best remaining compute investment ("n=5 training seeds is thin for a
Q1 learning-paper headline"). This doubles the training-seed count for
the canonical result.

## Result

`episode_metrics.csv` for this run uses the same 9-cell embedded static
comparator set (`s{1,2,3}_d{1.00,1.50,2.00}`) as the canonical seeds
1-5 run's own internal eval, not the external 147-cell dense grid used
for the manuscript headline. Two checks, both against `order_ret_excel`
(Excel ReT):

**(a) Against the fixed headline 147-cell best static (0.005466,
constant — static performance does not depend on the PPO training
seed), pooling seeds 1-10:**

| n seeds | mean Δ | CI95 | seeds positive |
|---:|---:|---|---|
| 10 | +0.000432 | [+0.000414, +0.000450] | 10/10 |

Per-seed PPO Excel ReT: seed 1: 0.005921, 2: 0.005906, 3: 0.005920,
4: 0.005862, 5: 0.005855, 6: 0.005880, 7: 0.005908, 8: 0.005891,
9: 0.005924, 10: 0.005911. All ten seeds land in a tight band
(0.005855-0.005924); the canonical 5-seed delta was +0.000426, the
10-seed delta is +0.000432 — consistent, not a regression.

**(b) Against the best static within the run's own embedded 9-cell set
(`s2_d1.50`, fixed across all 10 seeds for comparator consistency):**

| n seeds | mean Δ | CI95 | seeds positive |
|---:|---:|---|---|
| 10 | +0.000458 | [+0.000444, +0.000472] | 10/10 |

## Verdict

**Confirms the canonical result at doubled scale.** All ten
independently-trained seeds (five new, five original) produce a
positive Excel ReT delta versus the headline static comparator, with a
tight seed-clustered CI95 that excludes zero by a wide margin under
either comparator convention. The 10-seed mean (+0.000432) is within
noise of the 5-seed canonical mean (+0.000426).

## What this does and does not license

**Licensed:** upgrading the manuscript's seed count disclosure from "5
seeds, all positive" to "10 seeds, all positive" wherever seed-count is
cited as a robustness signal (abstract, §3.5 statistical inference,
limitations). This directly answers the most common Reviewer-#2 note
across all four external reviews.

**Superseded same-day:** this initial caveat was correct before the
CRN-paired static re-evaluation landed. The rigorous 10-seed bundle now
evaluates seeds 6-10 against the exact headline static comparator and is
licensed for the manuscript headline; see the update below.

## Registry update

Strengthens C1 and adds C21 in
`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` — cite this verdict doc
alongside `docs/track_b_q1_stats_2026-07-02_final_10seed/`.

## Update: superseded by rigorous CRN-paired evaluation (same day)

A more rigorous check landed in parallel:
`docs/track_b_q1_stats_2026-07-02_final_10seed/` evaluates seeds 6-10
against the exact canonical static comparator (`S2_op10_2.00_op12_1.50`)
at the full 147-cell-grid convention, CRN-paired at the episode level
(120 total pairs, 10 seeds x 12 episodes). This supersedes the two
approximate checks in this doc (sections (a) and (b) above, which used
the run-internal 9-cell embedded static set as a stand-in).

**Authoritative 10-seed result:** Excel ReT delta $+0.000438$, CI95
$[+0.000409, +0.000468]$ (episode-pooled); seed-clustered mean delta
$+0.000438$, CI95 $[+0.000421, +0.000458]$; order-level ReT delta
$+0.000426$, CI95 $[+0.000398, +0.000455]$. All 10 seed-level mean
deltas positive. This is now cited directly in the manuscript
(`sections/04_results.tex`, Section 4.3) and is consistent with both the
5-seed canonical result and the approximate checks above.
