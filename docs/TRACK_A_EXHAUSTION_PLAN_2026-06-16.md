# Track A Exhaustion Plan - 2026-06-16

## Purpose

Before moving the main positive-RL search to Track B, exhaust Track A using only
the Garrido-facing decision variables:

- common inventory buffer target `I_t,S`;
- short-term manufacturing capacity `S`.

The strict thesis contract remains:

```text
action_space_mode=thesis_factorized
MultiDiscrete([6, 3])
I_t,S in {0, I168, I336, I504, I672, I1344}
S in {S1, S2, S3}
```

The nearest Track A extension is:

```text
action_space_mode=continuous_it_s
Box([0, -1], [1, 1])
action[0] = continuous common I_t,S fraction of I1344
action[1] = shift signal mapped to S1/S2/S3
```

`continuous_it_s` is not Track B. It de-discretizes Garrido's buffer-size
variable but does not expose per-node ROP control, downstream dispatch, or other
new operational levers.

## Fixed Fidelity Modes

All Track A exhaustion runs should keep the post-fix environment:

- `risk_occurrence_mode=thesis_periodic`
- `raw_material_flow_mode=kit_equivalent_order_up_to`
- `raw_material_order_up_to_multiplier=2.0`
- `inventory_period_mode=thesis_strict`

The only exceptions are explicitly labeled stress-extension probes.

## What Is Still Worth Testing

### 1. Reward steepness

The current evidence says PPO does not beat the best static grid under the
default Track A reward. The remaining reward question is whether the learning
signal is too flat, not whether the external result metric should change.

Screen the following reward families:

| profile | reward mode | intent |
|---|---|---|
| `ret_ladder` | `ReT_ladder_v1` | thesis-decision ladder reward |
| `ret_ladder_steep` | `ReT_ladder_v1` | steeper service/recovery gate |
| `ret_seq` | `ReT_seq_v1` | sequential thesis proxy |
| `control_steep` | `control_v1` | stronger backorder penalty |
| `control_pbrs_steep` | `control_v1_pbrs` | stronger backorder penalty plus PBRS |
| `ret_cd_sigmoid` | `ReT_cd_sigmoid` | smoother nonlinear resilience signal |

Do not claim success from reward-total alone. A candidate survives only if PPO
improves external metrics against the same-run best static grid:

- `fill_rate_order_level_mean`;
- `order_level_ret_mean`;
- optionally `ReT_garrido2024` where the runner reports it.

### 2. Risk stress

Use risk stress as a predeclared axis, not as a hand-picked positive case:

```text
risk_level in {increased, severe, severe_extended, severe_training}
```

`severe` and above are stress extensions, not thesis-nominal claims. If PPO wins
only at one isolated point, report it as a probe and do not promote it. If the
PPO advantage grows coherently with stress, it becomes a defensible stress
response result.

### 3. Stochastic processing time

`stochastic_pt` is off unless `--stochastic-pt` is passed. Test both deterministic
and stochastic PT explicitly:

| profile | flags |
|---|---|
| `det_pt` | no `--stochastic-pt` |
| `stoch_pt_hist` | `--stochastic-pt --stochastic-pt-spread 1.0` |
| `stoch_pt_mean` | `--stochastic-pt --stochastic-pt-spread 1.0 --stochastic-pt-mean-preserving` |
| `stoch_pt_mean_hi` | `--stochastic-pt --stochastic-pt-spread 2.0 --stochastic-pt-mean-preserving` |

Prefer the mean-preserving profiles for causal claims because they vary PT
variance without changing expected PT.

### 4. Continuous Garrido buffer

If strict `[6,3]` still shows no positive signal, test `continuous_it_s` before
Track B. It keeps Garrido's two variables but removes the DOE discretization of
buffer size. Any positive result still needs the fair comparison chain:

```text
Garrido static discrete -> best static continuous I_t,S -> PPO continuous I_t,S
```

Until the best static continuous policy is measured, a PPO win over the discrete
grid could be only a de-discretization win, not an adaptivity win.

## Screening Runner

Use:

```bash
python scripts/run_track_a_exhaustion_sweep.py \
  --action-space-modes thesis_factorized continuous_it_s \
  --reward-profiles ret_ladder ret_ladder_steep control_steep ret_seq \
  --risk-levels severe_extended \
  --pt-profiles det_pt stoch_pt_mean \
  --train-timesteps 30000 \
  --eval-episodes 3 \
  --max-steps 80
```

For a tighter first screen:

```bash
python scripts/run_track_a_exhaustion_sweep.py \
  --action-space-modes thesis_factorized \
  --reward-profiles ret_ladder_steep control_steep control_pbrs_steep \
  --risk-levels severe_extended severe_training \
  --pt-profiles det_pt stoch_pt_mean stoch_pt_mean_hi \
  --train-timesteps 30000 \
  --eval-episodes 3 \
  --max-steps 80
```

Escalate to Kaggle only for profiles that survive local screening on external
metrics. A confirmatory run should use multiple PPO seeds, longer training, more
evaluation episodes, and the same static-grid baseline in every cell.

## Decision Rule

Track A remains open only if a candidate shows a coherent external-metric gain:

- PPO beats the best static grid on `fill` or `ReT` by more than run noise;
- the result repeats across seeds or a monotonic stress/PT axis;
- the winning condition is predeclared in the full matrix, not cherry-picked.

If none of the reward/risk/PT/continuous-buffer cells pass this gate, the honest
Track A conclusion is robust parsimonia: Garrido's simple static buffers are
hard to beat under the original decision contract.

## First Local Screen

Completed locally with `30k` PPO timesteps, `3` evaluation episodes, `80` weekly
steps, strict `[6,3]`, `stoch_pt_mean`, and the post-fix environment.

Artifact:
`outputs/benchmarks/track_a_exhaustion_screen/track_a_screen_20260616T045608Z/sweep_summary.csv`

| reward profile | risk level | PPO fill | best static | static fill | delta fill | PPO ReT | static ReT | delta ReT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `ret_ladder_steep` | `severe_extended` | 0.5305 | `static_grid_I504_S2` | 0.6071 | -0.0766 | 0.1370 | 0.1582 | -0.0212 |
| `ret_ladder_steep` | `severe_training` | 0.3569 | `static_grid_I504_S3` | 0.4257 | -0.0688 | 0.0992 | 0.0956 | +0.0036 |
| `control_pbrs_steep` | `severe_extended` | 0.5684 | `static_grid_I504_S2` | 0.6071 | -0.0386 | 0.1334 | 0.1582 | -0.0247 |
| `control_pbrs_steep` | `severe_training` | 0.2652 | `static_grid_I504_S3` | 0.4257 | -0.1605 | 0.0708 | 0.0956 | -0.0249 |

Interpretation: this first screen does not support the hypothesis that steeper
Track A rewards plus stochastic PT and higher stress are enough for PPO to beat
the best static `[6,3]` grid on external metrics. `control_pbrs_steep` improves
the `severe_extended` fill gap relative to `ret_ladder_steep`, but still loses.
The only positive cell is a tiny ReT gain under `ret_ladder_steep` +
`severe_training`, paired with a large fill loss, so it is not promotable.

## Continuous I_t,S Confirmation Gate

After the strict `[6,3]` screen, we tested the nearest Track A extension:
`continuous_it_s`. This keeps Garrido-Rios' two decision variables but
de-discretizes the common `I_t,S` buffer level.

Runner:
`scripts/run_track_a_continuous_it_s_confirm.py`

Configuration:

- `action_space_mode=continuous_it_s`
- `risk_level=severe`
- `risk_occurrence_mode=thesis_periodic`
- `raw_material_flow_mode=kit_equivalent_order_up_to`
- `reward_profile=ret_ladder_steep`
- `pt_profile=stoch_pt_hist`
- `train_timesteps=40000`
- seeds: `4242,4243,4244`
- static continuous grid: `buffer_fraction in {0.0, 0.1, ..., 1.0}` and
  `S in {1,2,3}`

Artifact:
`outputs/benchmarks/track_a_continuous_it_s_confirm/continuous_it_s_severe_ret_ladder_steep_histpt_40k_3seed/seed_summary.csv`

| seed | PPO fill | best static continuous | static fill | delta fill | PPO ReT | static ReT | delta ReT |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4242 | 0.7944 | `static_continuous_b0.800_S3` | 0.8396 | -0.0452 | 0.2536 | 0.2652 | -0.0116 |
| 4243 | 0.8055 | `static_continuous_b0.200_S2` | 0.8396 | -0.0341 | 0.2541 | 0.2759 | -0.0218 |
| 4244 | 0.7439 | `static_continuous_b0.700_S3` | 0.8310 | -0.0871 | 0.2367 | 0.2585 | -0.0219 |

Overall:

- positive fill seeds: `0/3`
- positive ReT seeds: `0/3`
- mean delta fill: `-0.0554`
- mean delta ReT: `-0.0184`

Interpretation: the directional single-seed result that suggested PPO
continuous `I_t,S` could beat a static continuous policy is not confirmed when
the static baseline is a broader continuous grid. The likely failure mode was a
weak static comparator, not a real adaptive advantage. Track A is therefore
still parsimonious under this gate.
