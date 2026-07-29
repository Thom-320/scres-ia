# Track B Top-Tier Audit Plan and Status (2026-06-30)

## Claim Boundary

Primary claim candidate:

> Track A, using Garrido's original buffer/shift decision families, reaches a dense static frontier. Track B unlocks learning value by controlling the downstream delivery bottleneck, where PPO can learn an adaptive policy that improves resilience/service/cost against static policies.

Do not promote H4 retained learning as a main claim unless retained-vs-reset is positive with CI.

## Current Evidence

The existing Kaggle Track B v5 run is a 3-seed exploratory-confirmatory result:

- Source run: `outputs/experiments/track_b_gain_2026-06-29/kaggle_joint_confirm_50k_v5_output/track_b_joint_confirm_50k_3seed_h104`
- Audit bundle: `outputs/audits/track_b_top_tier_audit_2026-06-30`
- Workbook: `outputs/audits/track_b_top_tier_audit_2026-06-30/track_b_top_tier_audit.xlsx`

Key audit verdicts from that run:

| Verdict | Result |
|---|---|
| Raw ReT win | Pass: PPO ReT 0.005631 vs best static-by-ReT 0.005237 |
| ReT tail/CVaR win | Pass: PPO lower-tail ReT 0.005392 vs static 0.004961 |
| Resource-efficient win | Fail: PPO cost index 0.6759 vs static 0.6667 |
| ReT/cost Pareto | Pass in current audit: no static has both higher/equal ReT and lower/equal cost |

Important limitation: the v5 runner did not export a Garrido-style order ledger, so APj/RPj/DPj/CTj/ReTj are not auditable order-by-order for that run. This is now fixed in `scripts/run_track_b_smoke.py` with `--export-order-ledger`.

## Dense Frontier Update

The first dense Track B static frontier exists at:

- `outputs/experiments/track_b_dense_frontier_2026-07-01`

It evaluates 147 static cells (`3 shifts x 7 op10 multipliers x 7 op12 multipliers`) across 3 seeds. Its best static is:

- `S1_op10_0.50_op12_0.50`, ReT = `0.005550`, flow fill = `0.9962`, lost rate = `0.0000`.

This is higher than the old 9-static best-by-ReT (`0.005237`). The old PPO v5 still beats it on raw ReT (`0.005631 - 0.005550 = +0.000081`), but the margin is much smaller. Therefore the Track B claim is still alive, but it must be confirmed against the dense frontier under held-out CRN before using strong wording.

Interpretation update: do not frame Track B as simply "increase Op12 dispatch." The dense static frontier currently suggests that lower/moderate dispatch settings can score better on Excel ReT, likely by reducing downstream congestion/recovery penalties. The mechanism audit must identify what PPO is doing relative to this dense frontier.

The audit builder now accepts `--external-static-frontier` and writes a separate `raw_ret_win_vs_external_dense` verdict. Test artifact:

- `outputs/audits/track_b_top_tier_audit_with_dense_2026-06-30/track_b_top_tier_audit.xlsx`

## Implemented This Sprint

- Extended `scripts/run_track_b_smoke.py` to export:
  - episode-level CD components;
  - `compute_episode_metrics` order/service metrics;
  - optional Garrido-style `order_ledger.csv` with `j,Q,OPTj,OATj,LT,CTj,APj,RPj,DPj,R11..R3,sumBt,sumUt,lost,backorder,ReTj,case`.
- Added `scripts/build_track_b_top_tier_audit.py`.
- Added `scripts/build_track_b_top_tier_workbook.mjs`.
- Added a reusable watcher: `scripts/watch_kaggle_kernel.py`.
- Built the local Track B audit workbook for the old v5 run.

## Active Confirmatory Run

Kaggle kernel:

- Slug: `thomaschisica/scresia-track-b-top-tier-confirm`
- Profile: `track_b_top_tier_confirm_5seed_60k_h104`
- Seeds: 1,2,3,4,5
- Timesteps: 60k
- Eval episodes: 12
- Risk level: `adaptive_benchmark_v2`
- Reward: `control_v1`
- Ledger: enabled
- Audit on Kaggle: CSV/JSON only; workbook rebuilt locally after download.

Watcher:

- PID file: `/tmp/scres_track_b_top_tier_watcher.pid`
- Log: `outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_watch/watcher_abs.log`
- Output target: `outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output`

Operational note: the first two kernel versions failed immediately because the Kaggle dataset payload was not mounted and auxiliary kernel files were not available. Version 3 embeds the 691 KB payload directly in the script and is currently running.

## Confirmatory Result (5 seeds / 60k)

The v3 Kaggle confirmatory finished and downloaded to:

- `outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104`

The workbook with external dense frontier is:

- `outputs/audits/track_b_top_tier_confirm_5seed_with_dense_2026-06-30/track_b_top_tier_audit.xlsx`

Key verdicts:

| Verdict | Result |
|---|---|
| Raw ReT vs internal 9-static best-by-ReT | Pass: PPO `0.005666` vs `0.005214`, delta `+0.000452`; paired seed CI clears zero. |
| ReT lower-tail/CVaR vs internal static | Pass: PPO `0.005428` vs `0.004740`, delta `+0.000688`. |
| Balanced CD sigmoid mean | Pass: PPO `0.938526` vs `0.751707`; paired seed CI clears zero. |
| Flow fill | Pass: PPO `0.961320` vs `0.670677`; paired seed CI clears zero. |
| Service-loss AUC/order | Pass: PPO improves by about `1.03e6`; paired seed CI clears zero. |
| Cost/resource-efficient win | Fail/not confirmed: PPO cost index `0.682051` vs static `0.666667`. |
| Raw ReT vs external 147-cell dense frontier | Pass as a signal: PPO `0.005666` vs dense static `0.005550`, delta `+0.000116`. |

Caveat: the external dense frontier (`outputs/experiments/track_b_dense_frontier_2026-07-01`) used 3 seeds and is not yet paired seed-for-seed against the 5-seed PPO confirmatory. The signal survives, but a final paper claim should use a CRN dense-static eval matched to the 5-seed PPO eval seeds.

## Gates After Download

1. Rebuild workbook locally from the downloaded output:
   - `scripts/build_track_b_top_tier_audit.py --run-dir <downloaded_run_dir> --output-dir outputs/audits/<new_dir>`
2. Primary gate:
   - PPO order-level Excel ReT > best static-by-ReT.
3. Secondary gates:
   - ReT CVaR/lower-tail not worse, ideally better.
   - Service metrics improve: rolling fill, flow fill, backorder.
   - Cost reported separately; raw ReT and Pareto/resource verdicts are separate.
4. Mechanism checks:
   - PPO uses downstream dispatch variables, especially Op12/Op10, rather than simply maxing shifts.
   - Shift mix does not collapse to fixed S3.
   - Action traces correlate with backlog/risk/hazard, not only time.
5. Generalization:
   - Evaluate frozen PPO against dense static frontiers on current/increased/severe and R1/R2/R3/R24/mixed.

## Lessons From Track A Applied Here

- No claim against coarse frontiers.
- No in-sample checkpoint evidence.
- No single-metric win claim: report ReT, CVaR/tail, service, cost, CD, and order ledger metrics.
- Track A null is evidence for frontier-dependent learning, not a hidden failure.
