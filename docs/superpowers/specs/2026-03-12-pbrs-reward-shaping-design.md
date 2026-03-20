# PBRS Reward Shaping Design Spec

## Problem

The `control_v1` reward provides sparse, delayed feedback on service quality.
In late-episode steps (>200 of 260), cumulative metrics barely move, so the
agent receives vanishingly small gradients precisely when severe disruptions
hit hardest — the paper's headline finding. PBRS injects a denser learning
signal while preserving the optimal policy (Ng et al. 1999).

## Scope

- New reward mode `control_v1_pbrs` in `env_experimental_shifts.py`
- CLI integration in `benchmark_control_reward.py`
- Two variants: **cumulative** (main) and **step-level** (ablation)
- Tests covering correctness, edge cases, and integration

## Design

### Theoretical Foundation

PBRS adds a shaping bonus to the base reward:

```
F(s, s') = γ·Φ(s') - Φ(s)
reward_shaped = reward_base + F(s, s')
```

where Φ(s) is a potential function over states and γ is the discount factor.
Ng et al. (1999) proved that any Φ preserves the optimal policy.

### Variant 1: Cumulative Target-Deficit (Main)

**Potential function:**

```
Φ(s) = -α × max(0, τ - FR_cumulative(s)) / τ
```

- `FR_cumulative` = `obs[6]` (cumulative fill rate, [0,1])
- `τ` = fill-rate target (default 0.95)
- `α` = scaling hyperparameter (default 1.0)
- Division by `τ` normalizes the deficit to [0, 1]

**Properties:**
- Φ = 0 when FR ≥ τ (shaping turns off above target)
- Φ = -α when FR = 0 (maximum deficit)
- Monotonically increasing in FR below τ
- Legitimate state-dependent potential → Ng et al. guarantee holds

**Limitation (documented):** Cumulative FR changes by ~1/N per step at step N,
so the shaping signal attenuates in late episodes. This is a known trade-off
for theoretical cleanliness — addressed by the step-level ablation.

### Variant 2: Step-Level Backorder (Ablation)

**Potential function:**

```
Φ(s) = -α × prev_step_backorder_qty_norm(s)
```

- Uses `obs[16]` from v2 observation (`prev_step_backorder_qty_norm`)
- This IS a state variable in the v2 observation, so Ng et al. holds
- Requires `observation_version="v2"`
- No deficit-gap formulation (not a fill rate signal)

**Properties:**
- Responds to *this week's* service failures, not historical average
- Higher temporal responsiveness under disruptions
- Noisier signal than cumulative variant

**Paper framing:** "The step-level PBRS variant leverages the enriched
observation (v2) to shape rewards based on the most recent service outcome
rather than cumulative history."

### Edge Cases

**Zero-demand steps:** When `new_demanded == 0`, cumulative fill rate may not
update meaningfully. For the cumulative variant this is handled naturally
(FR doesn't change → F ≈ γΦ - Φ ≈ (γ-1)Φ, which is small). No special case
needed since Φ is computed from obs[6] which the env already handles.

**Initialization:** `self._prev_phi` is set in `reset()` to
`_compute_phi(initial_fill_rate)`. After warmup, initial FR is typically 1.0,
so Φ(s₀) = -α × max(0, 0.95-1.0)/0.95 = 0. For step-level variant,
`prev_step_backorder_qty_norm` is 0.0 at reset, so Φ(s₀) = 0.

### Hyperparameters

| Param | CLI flag | Default | Grid (tuning) |
|-------|----------|---------|---------------|
| α | `--pbrs-alpha` | 1.0 | {0.1, 0.5, 1.0, 2.0} |
| τ | `--pbrs-tau` | 0.95 | fixed |
| γ | `--pbrs-gamma` | 0.99 | must match SB3 gamma |
| variant | `--pbrs-variant` | cumulative | {cumulative, step_level} |

α tuning is per-variant: cumulative {0.1, 0.5, 1.0, 2.0},
step-level {0.05, 0.1, 0.5, 1.0} (noisier signal → smaller scale).

## Implementation

### File: `supply_chain/env_experimental_shifts.py`

1. Add `"control_v1_pbrs"` to valid reward_mode values
2. Add constructor params: `pbrs_alpha`, `pbrs_tau`, `pbrs_gamma`, `pbrs_variant`
3. Add state: `self._prev_phi: float` (initialized in reset)
4. Add method `_compute_phi(fill_rate_or_backorder: float) -> float`
5. In `step()`: compute Φ(s'), F, add to base control_v1 reward
6. In `step()` info dict: add `pbrs_phi`, `pbrs_shaping_bonus`,
   `pbrs_base_reward`, `pbrs_variant`
7. Validation: step-level variant requires observation_version="v2"

### File: `scripts/benchmark_control_reward.py`

1. Add CLI flags: `--pbrs-alpha`, `--pbrs-tau`, `--pbrs-gamma`, `--pbrs-variant`
2. Pass through `build_env_kwargs()` when reward_mode is `control_v1_pbrs`
3. No changes to evaluation logic (reward mode handled by env)

### File: `tests/test_control_reward_benchmark.py`

Unit tests:
- `test_pbrs_phi_zero_above_target` — FR=0.98 > τ=0.95 → Φ=0
- `test_pbrs_phi_deficit_below_target` — FR=0.80 → Φ=-α×(0.95-0.80)/0.95
- `test_pbrs_shaping_bonus_positive_on_improvement` — FR improves → F > 0
- `test_pbrs_shaping_bonus_negative_on_degradation` — FR degrades → F < 0
- `test_pbrs_step_level_requires_v2` — ValueError if v1 + step_level
- `test_pbrs_prev_phi_initialized_in_reset` — after reset, _prev_phi = Φ(s₀)
- `test_pbrs_step_level_env_runs` — 3 steps with v2 + step_level, no crash

Integration:
- `test_benchmark_smoke_pbrs` — full benchmark with control_v1_pbrs

### Ablation Table (Paper)

| Condition | α | Variant | Description |
|-----------|---|---------|-------------|
| No shaping | — | — | `control_v1` baseline |
| PBRS-cumulative | 1.0 | cumulative | Main result |
| PBRS-step | 1.0 | step_level | Temporal responsiveness ablation |
| PBRS-α sweep | 0.1/0.5/2.0 | cumulative | Sensitivity analysis |

### Paper Wording

Main method:
> "We use potential-based reward shaping with a target-deficit service
> potential derived from cumulative fill-rate shortfall, preserving the
> canonical PBRS formulation (Ng et al. 1999) while injecting
> quality-of-service signal into step-level learning."

Ablation:
> "We additionally test a step-level PBRS variant using the enriched
> observation (v2) to assess temporal responsiveness under late-episode
> disruptions."

## Verification

1. Unit tests pass: `pytest tests/test_control_reward_benchmark.py -v -k pbrs`
2. Smoke benchmark: `python scripts/benchmark_control_reward.py --reward-mode control_v1_pbrs --seeds 1 --train-timesteps 32 --eval-episodes 1 --step-size-hours 24 --max-steps 4 --w-bo 1.0 --w-cost 0.02 --w-disr 0.0 --risk-level increased --stochastic-pt --output-dir outputs/benchmarks/pbrs_smoke --skip-artifact-export`
3. Full test suite: `pytest tests/ -v`
4. Quality: `black . && ruff check . --fix`
