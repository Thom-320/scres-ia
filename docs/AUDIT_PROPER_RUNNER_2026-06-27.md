# Proper-Runner Re-evaluation — Honest Verdict (2026-06-27)

## What was done
The `compare_garrido_dynamic_vs_static.py` runner (the canonical, full-metrics
PPO-vs-18-statics comparison) was run on the war cell `phi4/psi1.5,
stochastic_pt=False, demand_multiplier=1.0` with the following
configuration:
- `--reward-mode ReT_garrido2024_raw` (default of the runner; CD-aligned)
- `--seeds 1,2,3,4,5`
- `--eval-episodes 5`
- `--train-timesteps 50000`
- `--risk-frequency-multiplier 4.0`, `--risk-impact-multiplier 1.5`
- Output: `outputs/calibration/war_cell_phi4_psi1p5_proper/smoke/`
  (the `smoke` is the runner's default `--label`, not a smoke-mode run).

The runner produces per-episode PPO metrics (real ledger values:
`cd_sigmoid_mean`, `mean_ret_excel_formula`, `unattended_orders_terminal`,
`flow_fill_rate`, `service_loss_*`, `extra_shift_hours_total`,
`strategic_buffer_target_units_mean`, branch shares, CTj/APj/RPj/DPj
quantiles, CD components) and a `best_static_by_metric.csv` with the
best static per regime per metric across all 18 statics × 5 seeds.

## Honest verdict (the comparison the user asked for)
**PPO mean vs best_static_per_regime, per metric, per regime — 22 pairs
with real signal (others are NaN/0 because of the known delay=54
metric degeneration):**

```
regime   metric                          PPO     best_static  (policy)         Δ       verdict
current  cd_sigmoid_mean                 0.6258  0.6347       (S1_I168)         -0.0089  LOSE
current  mean_ret_excel_formula          0.0022  0.0026       (S3_I336)         -0.0004  LOSE
current  unattended_orders_terminal      34.92   13.48        (S3_I336)        -21.44    LOSE
current  extra_shift_hours_total         1942.08 0.00         (S1_I0)          -1942.08  LOSE
current  strategic_buffer_target_units   81501.2 0.00         (S1_I0)         -81501.2   LOSE
current  flow_fill_rate                  0.7858  0.8425       (S3_I336)         -0.0567  LOSE
current  service_loss_cvar95             1.1084  1.0663       (S2_I1344)        -0.0421  LOSE
current  service_loss_mean               0.6490  0.5877       (S3_I336)         -0.0613  LOSE
increased cd_sigmoid_mean                0.5626  0.5737       (S3_I1344)        -0.0112  LOSE
increased mean_ret_excel_formula         0.0003  0.0003       (S3_I336)         -0.0001  ~
increased unattended_orders_terminal     217.56  201.08       (S3_I336)        -16.48    LOSE
increased extra_shift_hours_total        10644.5 0.00         (S1_I0)         -10644.5    LOSE
increased strategic_buffer_target_units   116782  0.00         (S1_I0)        -116782     LOSE
increased service_loss_cvar95            1.2048  0.00         (S1_I0)           -1.2048  LOSE
increased service_loss_mean              0.9259  0.00         (S1_I0)           -0.9259  LOSE
severe   cd_sigmoid_mean                 0.5117  0.5065       (S3_I168)        +0.0052  WIN
severe   mean_ret_excel_formula          0.0001  0.0001       (S2_I336)        -0.0000  ~
severe   unattended_orders_terminal     253.56  242.24       (S2_I1344)       -11.32    LOSE
severe   extra_shift_hours_total        15153.6 0.00         (S1_I0)        -15153.6    LOSE
severe   strategic_buffer_target_units   58587.9 0.00         (S1_I0)        -58587.9    LOSE
severe   service_loss_cvar95            1.1456  0.00         (S1_I0)           -1.1456  LOSE
severe   service_loss_mean              0.9888  0.00         (S1_I0)           -0.9888  LOSE

Totals: 1 WIN, 2 ~, 19 LOSSES  (across 22 regime-metric pairs with real signal)
```

## What this means (the user's criteria)
Per the audit plan:
> "Si gana CD pero pierde flow/service/Excel, se etiqueta como metric gaming."

**This is metric gaming.** PPO wins CD only in **severe** (the regime where the best static is weakest: `S3_I168` at 0.5065, the heavy corner), and even that win is +0.005 — while it **loses on every other real metric in severe** AND loses CD in `current` and `increased` against the real best per regime. Across all 22 regime-metric pairs with real signal: **1 WIN, 2 ~, 19 LOSSES**.

The imitation gate fails: PPO does **not** meet the best_static_per_regime on the same metric in 2 of 3 regimes for CD, and loses on every service metric in every regime.

## Data-quality caveat
Several `best_static` values in `increased` and `severe` for
`service_loss_*` and resource metrics are 0 because `original_S1_I0` has
`flow_fill_rate=NaN` under the runner's config (the known delay=54
metric degeneration; `S1_I0` has no upstream buffer replenishment, so
`flow_fill` is NaN and `service_loss=0` falls out). `original_S1_I0` is
the **free** baseline (0 shift-hours, 0 buffer, 0 service-loss), not a
strong policy — so PPO "losing" to it on resource metrics means PPO is
expensive, not that `S1_I0` is a contender. The verdict on the *real*
service metrics (`unattended_orders_terminal` in all 3 regimes, and
`flow_fill_rate` in `current` where the best static is `S3_I336@0.8425`)
stands regardless: **PPO loses on every real service signal that is not
degenerate.**

## Conclusion
- **No defendible win** for PPO on the war cell at 50k timesteps with the
  proper runner and real metrics.
- The earlier 4-case grid "PASS" at 10k timesteps was a calibration
  artifact (PPO beating a fixed weak robust `S1_I1344` on CD only; real
  service and resources not captured). The proper runner with 5 seeds,
  50k ts, and all 18 statics + best_static_per_regime per metric shows
  the honest result: 1 WIN / 2 ~ / 19 LOSSES.
- The honest paper position: Track A under the discrete `[buffer, shifts]`
  action space and the current DES is **null** for any PPO-vs-static
  dominance claim. The war cell is the highest-headroom env available
  (frontier gate confirmed eligible), and even there PPO cannot match the
  best static per regime on the real service metrics.

## Next step (not executed; for the user's decision)
- The frontier gate for the 3 learning-extension calibration cells
  (A/B/C, rho/surge/lead perturbations) all returned **ineligible** at
  26-week horizon (all collapsed to `S1_I1344` corner; max_gap=0.0000).
  See `outputs/audits/frontier_gate/calib_*.json` and
  `tests/test_learning_frontier_exists.py`.
- Combined with this proper-runner verdict, the honest next move is to
  declare the Track-A + learning-extension lane a calibrated null and
  write it up as such, OR pivot to Track B (downstream dispatch) which
  was the only lane with a confirmed regime-diverse, service-real
  frontier (F12/F16 in the prior findings registry).
