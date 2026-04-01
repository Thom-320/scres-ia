# Garrido Exact Baselines

This note defines the baseline family used to compare the thesis-inspired
static policies against the learned RL policy in the current DES/RL repo.

## Why these baselines were added

The original benchmark already exposed `static_s1`, `static_s2`, and
`static_s3`. Those policies were useful as coarse shift-count references,
but they were not exact reproductions of the thesis capacity configurations.

The reason is that the RL action mapping uses normalized inventory-control
signals that are transformed inside the environment as:

`multiplier = 1.25 + 0.75 * signal`

With that mapping, the neutral action `[0, 0, 0, 0, 0]` does not reproduce
the thesis base configuration. It reproduces a 1.25x-scaled inventory lane
combined with `assembly_shifts = 2`.

For a serious "Garrido thesis model vs learned dynamic model" comparison,
the benchmark now includes exact static baselines that bypass the RL action
mapping and inject the DES control parameters directly.

## Baseline families

### Existing proxy baselines

- `static_s1`
- `static_s2`
- `static_s3`

These remain useful as repo-internal historical comparators.

### Exact thesis-style baselines

- `garrido_cf_s1`
- `garrido_cf_s2`
- `garrido_cf_s3`

These use direct DES actions and pull their parameters from the thesis-backed
capacity/config tables in [config.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py).

## Parameter source

The exact baselines use:

- `CAPACITY_BY_SHIFTS[S]["op3_q"]`
- `CAPACITY_BY_SHIFTS[S]["op7_q"]`
- `OPERATIONS[3]["rop"]`
- `OPERATIONS[9]["q"]`
- `OPERATIONS[9]["rop"]`

That gives a static configuration with:

- exact `assembly_shifts = S`
- exact thesis-style `op3_q`
- exact downstream dispatch/reorder controls
- exact batch size tied to the capacity table

## Interpretation

Use the exact Garrido baselines when the paper question is:

"Does the learned dynamic policy outperform the best static thesis-style
configuration under the same DES and risk regime?"

Use the historical `static_s*` baselines when the question is narrower and
repo-internal:

"How does PPO compare against the old shift-count proxies used in earlier
reward experiments?"

## Recommended reporting

For serious paper-facing comparisons, report at least:

- PPO
- `garrido_cf_s2`
- `garrido_cf_s3`
- best Garrido static baseline
- best historical static baseline

And evaluate all of them with one external resilience index, preferably
`ReT_garrido2024` when the comparison is framed around Garrido.
