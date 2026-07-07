# Track A Buffer-Friction Robustness Verdict (2026-07-03)

## Verdict

The old `per_op_buffer` Track A contract remains negative under the baseline
replicate and under more realistic replenishment friction. Arm B is mildly
interesting because the learned policy is Pareto non-dominated, but it does not
beat the held-out best static on Excel ReT and should not be promoted as a Track
A revival.

These arms are defensive diagnostics for the old top-up contract. They do not
replace the cleaner conservation-respecting Track A v2 result.

## Artifact

`outputs/experiments/track_a_repair_leadtime_robustness_2026-07-02/`

## Arms

| Arm | Setting | Best held-out static Excel | PPO Excel | Delta | Pareto |
|---|---|---:|---:|---:|---|
| A | baseline replicate, `lead_time=0`, `holding_cost=0` | `0.155254` | `0.154993` | `-0.000262` | dominated |
| B | `lead_time=168`, `holding_cost=0` | `0.156288` | `0.155042` | `-0.001246` | non-dominated |
| C | `lead_time=168`, `holding_cost=0.05` | `0.156288` | `0.154690` | `-0.001599` | dominated |

## Interpretation

- Arm A reproduces the old Track A null.
- Arm B shows a resource/tail trade-off that is not dominated by the evaluated
  frontier, but it still loses the primary Excel ReT comparison.
- Arm C loses on Excel ReT and is dominated.

The conservative manuscript-safe conclusion is that adding replenishment
friction does not turn the old Track A buffer contract into a primary ReT win.
The stronger and cleaner result remains the conservation-respecting Track A v2
audit: static/oracle headroom exists, but PPO did not convert it.

## Manuscript-Safe Wording

Use only if needed:

> Additional buffer-friction checks on the retired Track A buffer contract did
> not produce a primary Excel-ReT win; a one-period replenishment-delay arm was
> Pareto non-dominated but remained below the best held-out static comparator on
> the pre-specified resilience metric.

Avoid:

> Track A improves under realistic buffer replenishment.
