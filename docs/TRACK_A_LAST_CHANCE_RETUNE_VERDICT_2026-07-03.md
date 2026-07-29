# Track A Last-Chance Retune Verdict — 2026-07-03

## Verdict

**Track A does not revive.** The last-chance critic-pretrained BC+PPO run fails the
pre-registered promotion threshold.

The run is useful evidence because it tests a specific failure diagnosis: the earlier Track A v2
conservation run may have failed because BC initialized the actor while the PPO critic remained
uncalibrated. Critic pretraining did not fix the result. PPO still loses to the held-out best
static comparator in all five seeds.

## Artifact

- Remote source: `ovh-agent-lab:~/scres-ia/outputs/experiments/track_a_v2_conservation_ppo_critic_warmstart_2026-07-03/`
- Local copy: `outputs/experiments/track_a_v2_conservation_ppo_critic_warmstart_2026-07-03/`
- Preregistration: `docs/TRACK_A_LAST_CHANCE_PREREGISTRATION_2026-07-03.md`

This was the LC1a variant launched before the preregistration was finalized:

- `--critic-pretrain-epochs 50`
- `--teacher oracle_if_better`
- `--seeds 1,2,3,4,5`
- `--timesteps 40000`
- `--max-steps 52`
- `reward_mode="ReT_excel_delta"` inherited from the Track A v2 conservation environment

## Preregistered Promotion Rule

Track A could be promoted only if both conditions held:

1. PPO beats the held-out best static on mean Garrido/Excel ReT.
2. At least 4/5 seed deltas are positive.

## Result

| Quantity | Value |
|---|---:|
| Held-out best static Excel ReT | 0.174422 |
| Critic-pretrained PPO mean Excel ReT | 0.167295 |
| Mean delta vs held-out static | -0.007127 |
| Positive seed deltas | 0/5 |
| Promotion threshold met? | No |

Seed-level deltas from `seed_health.csv`:

| Seed | PPO Excel ReT | Delta vs held-out static | Selected step |
|---:|---:|---:|---:|
| 1 | 0.163627 | -0.010795 | 15000 |
| 2 | 0.169781 | -0.004641 | 10000 |
| 3 | 0.165149 | -0.009272 | 10000 |
| 4 | 0.168581 | -0.005841 | 15000 |
| 5 | 0.169337 | -0.005085 | 10000 |

Independent recomputation from `seed_health.csv` matches `summary.json` exactly:

- mean PPO Excel ReT: `0.16729512893420936`
- mean seed delta: `-0.00712659137057377`
- positive seed count: `0/5`

## Interpretation

Track A v2 remains scientifically interesting, but not as a positive PPO result:

- The conservation-respecting 5D gate shows real static-oracle headroom.
- BC learns the oracle teacher action mapping.
- PPO does not convert that headroom, even after critic pretraining.

This strengthens the Paper 1 boundary result: **action-space headroom is not sufficient for PPO to
improve SCRES**. Track B remains the manuscript spine. Track A should be reported as a clean null
or boundary case, not as revived evidence.

## Next-Step Boundary

Do not relaunch Track A blindly. A future Track A attempt would need a separately preregistered
training-design change, most plausibly around delayed-credit handling (`gamma`, GAE horizon, or an
auxiliary shaping reward), while preserving Garrido/Excel ReT as the primary evaluation endpoint.
