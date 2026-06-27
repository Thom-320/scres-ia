# Cobb-Douglas Frontier Unification (2026-06-27)

## Verdict

The three Cobb-Douglas investigations do not materially disagree on the data. They differ mainly in:

- the headroom definition being reported,
- whether current-regime saturation is allowed,
- whether max-inventory static optima are allowed,
- and whether the metric is true Garrido-2024 C-D or a C-D-adjacent continuous ReT screen.

The recommended primary lane is:

```text
primary_clean_same_bar:
  phi: 2.0
  psi: 1.0
  stochastic_pt: false
  demand_multiplier: 1.0
  reward/eval family: Garrido-2024 C-D same-bar
```

This lane is robust, interior, regime-dependent, and avoids the max-buffer corner.

The recommended exploratory war-stress lane is:

```text
exploratory_war_raw_headroom:
  phi: 3.0
  psi: 1.0
  stochastic_pt: true
  demand_multiplier: 1.0
  reward/eval family: Garrido-2024 C-D same-bar
```

This lane maximizes robust static spread among checked war candidates, but its severe-regime static optimum uses `S3_I1344`. It is therefore useful as a stress extension, not as the clean Pareto/efficiency headline.

## Evidence

### 1. Independent convergence on `phi2/psi1`

`scripts/cd_static_frontier.py` and `scripts/screen_cd_same_bar_frontier.py` converge on the same substantive cell:

```text
phi=2.0, psi=1.0, deterministic PT, demand multiplier=1.0
```

The robust Track-A C-D screen reports:

| Regime | Best Static | C-D Sigmoid | Excel ReT | Flow Fill |
| --- | --- | ---: | ---: | ---: |
| current | `static_S1_I168` | 0.7298 | 0.0050 | 0.9988 |
| increased | `static_S1_I504` | 0.6480 | 0.0020 | 0.8365 |
| severe | `static_S3_I336` | 0.5951 | 0.0007 | 0.5701 |

This is the clean decision frontier: the best static action changes across regimes, inventory remains interior, and the optimum is not simply `S3_I1344`.

The conservative oracle-vs-robust fixed-policy gap from `cd_static_frontier.py` is about `0.00816`. The broader best-vs-worst static spread from `screen_cd_same_bar_frontier.py` is about `0.168`. These are both valid numbers, but they answer different questions:

- `0.00816` is the realistic dynamic-policy headroom versus the best fixed static policy.
- `0.168` is the action-surface spread including poor policies no one would deploy.

For paper claims, use the conservative gap when discussing expected dynamic-vs-static gains.

### 2. Why `audit_track_a_cd_gate` diverged

`audit_track_a_cd_gate.py` used a stricter off-saturation filter and a C-D-adjacent continuous ReT metric rather than the same Garrido-2024 C-D eval signal. It rejected cells where the current regime was too easy even if the optimum moved under stress.

That is a gate-definition difference, not a contradiction in the static surface.

### 3. War-risk sweep outcome

The war-risk sweep confirms that stress is the real headroom lever. Cost dials did not meaningfully change the C-D frontier after recalibration; risk stress did.

Single-seed war sweeps showed larger apparent gaps, but robust 5-seed checks pushed the best static policy to max-inventory corners:

| Cell | Eligible Cleanly? | Best Static Sequence | Static Spread | Wrong-Regime Penalty | Issue |
| --- | --- | --- | ---: | ---: | --- |
| `phi3/psi1/spt` | no | `S1_I168 -> S2_I336 -> S3_I1344` | 0.2142 | 0.0504 | severe uses max inventory |
| `phi3/psi1.25/spt` | no | `S1_I168 -> S1_I1344 -> S2_I168` | 0.2012 | 0.0218 | increased uses max inventory |
| `phi4/psi1/det` | no | `S1_I168 -> S2_I1344 -> S3_I168` | 0.1904 | 0.0324 | increased uses max inventory |

The 10-seed war panels currently available reinforce the same warning:

- `phi4/psi1.5` has `S3_I1344` as the increased-regime top policy.
- `phi3/psi2` has `S2_I1344` as the increased-regime top policy.
- `phi4/psi2` is less cornered but the top sequence collapses partly back to `S1_I168`, lowering dynamic headroom.

So war-risk stress is useful, but the cleanest paper lane should not freeze on the raw peak if the claim is Pareto/resource efficiency.

## Decision

Use two lanes:

1. **Primary paper lane:** `phi2/psi1/deterministic`.
   - Goal: C-D same-bar dynamic policy versus efficient static frontier.
   - Claim type: clean, regime-dependent, interior, defensible.
   - Evaluation primary: `cd_sigmoid_mean`.
   - Training primary: `ReT_garrido2024_raw` or full-cost C-D raw.
   - Excel ReT: secondary continuity metric.

2. **Exploratory war-stress lane:** `phi3/psi1/stochastic_pt`.
   - Goal: maximize raw C-D action-surface spread under military stress.
   - Claim type: stress extension only.
   - Caveat: static optimum reaches `S3_I1344` under severe, so this is weaker for Pareto/resource-efficiency claims.

Do not use `phi4/psi1.5` as the primary freeze without another robust summary that proves it remains non-corner and efficient. The current artifacts show it creates headroom, but also pushes the increased-regime static optimum to `I1344`.

## Next Step

Run C-D reward training first on `primary_clean_same_bar`:

```text
env: phi2/psi1/deterministic
reward: ReT_garrido2024_raw or full-cost C-D raw
primary eval: cd_sigmoid_mean
baselines: all 18 static policies, especially S1_I168, S1_I504, S3_I336
secondary metrics: Excel ReT, flow fill, service-loss CVaR, resource composite
```

Only after that should the war-stress lane be trained as an extension.
