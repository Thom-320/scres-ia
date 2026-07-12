# Track A Last-Chance Retune Preregistration — 2026-07-03

## Purpose

Track A v2 now has a conservation-respecting action contract and a real static-oracle opening:
the best per-regime static table beats the best single held-out static by about `+0.0097` Excel
ReT in the 5D gate/confirmation artifacts. The previous PPO run did not convert that headroom:
PPO lost to the held-out best static in all five seeds.

This preregistration defines the final Track A rescue attempt before closing Track A as an
unconverted-headroom boundary result for Paper 1.

## Bias Guardrail

It would be convenient for the project if Track A revived. That convenience is a bias risk. The
claim threshold is therefore fixed before the retune is launched:

- The primary metric is Garrido/Excel ReT (`ret_excel` / `order_ret_excel` convention), not reward
  total, fill rate, CVaR, or native order-level ReT.
- The comparator is the held-out best static from the same Track A v2 conservation gate.
- Track A is promoted only if PPO beats that held-out best static on mean Excel ReT and has at
  least 4/5 positive seed deltas.
- If the retuned run loses or produces a mixed result below this threshold, Track A remains a
  negative boundary: static-oracle headroom exists, but PPO does not reliably convert it.

## Diagnosis From The Failed Run

The failed run (`outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/`) showed:

- Held-out best static Excel ReT: `0.174422`.
- PPO mean Excel ReT: `0.168105`.
- Mean delta: `-0.006316`.
- Positive seed deltas: `0/5`.
- Behavior cloning did learn the teacher mapping: BC loss fell from about `0.432` to `0.023`.
- Checkpoint selection was early in 4/5 seeds (`5k` to `30k` out of `40k`), suggesting PPO
  fine-tuning may damage a cloned actor before the value function is calibrated.

## Registered Retune

### Arm LC1 — Critic-Pretrained BC+PPO

Use the existing Track A v2 PPO runner with value-function pretraining enabled:

```bash
.venv/bin/python scripts/run_track_a_v2_conservation_ppo.py \
  --gate-dir outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03 \
  --output outputs/experiments/track_a_v2_conservation_ppo_critic_pretrain_5seed_40k_2026-07-03 \
  --seeds 1,2,3,4,5 \
  --n-envs 4 \
  --timesteps 40000 \
  --checkpoint-interval 5000 \
  --bc-epochs 150 \
  --critic-pretrain-epochs 100 \
  --max-steps 52 \
  --selection-seed0 8000 \
  --eval-seed0 9000 \
  --learning-rate 1e-4 \
  --clip-range 0.1 \
  --target-kl 0.02 \
  --ent-coef 0.0 \
  --teacher oracle_if_better
```

Rationale: BC initializes the actor; critic pretraining initializes the value network on teacher
rollout returns so that early PPO advantages are less destructive.

### Actual launch currently running

Before this note was finalized, Claude launched the same intervention on the VPS as:

```bash
python3 -u scripts/run_track_a_v2_conservation_ppo.py \
  --gate-dir outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03 \
  --output outputs/experiments/track_a_v2_conservation_ppo_critic_warmstart_2026-07-03 \
  --seeds 1,2,3,4,5 \
  --n-envs 4 \
  --timesteps 40000 \
  --checkpoint-interval 5000 \
  --bc-epochs 150 \
  --bc-batch-size 128 \
  --max-steps 52 \
  --critic-pretrain-epochs 50 \
  --teacher oracle_if_better \
  --learning-rate 0.0001 \
  --ent-coef 0.0 \
  --clip-range 0.1 \
  --target-kl 0.02
```

Treat this as **LC1a**. It is close enough to the registered intervention to be diagnostic, but
the epoch count and output label differ from the preferred LC1 command above. If LC1a passes the
promotion threshold, it should be preserved and followed by an explicit confirmation rather than
silently substituted for a pre-registered final claim.

### Optional diagnostic only — BC-only held-out evaluation

If LC1 fails but the logs suggest the value function is not the issue, evaluate the BC actor before
PPO fine-tuning as a diagnostic. This is not a new promotion path unless separately documented
before running a full confirmation.

## Registered Outputs

The run must produce:

- `summary.json`
- `seed_health.csv`
- `checkpoint_metrics.csv`
- `static_frontier_heldout.csv`
- `static_frontier_selection.csv`
- `teacher.json`
- `report.md`

The verdict document will be:

- `docs/TRACK_A_LAST_CHANCE_RETUNE_VERDICT_2026-07-03.md`

## Manuscript Rule

Paper 1's Track B spine does not change unless this run passes the preregistered threshold. If it
passes, Track A becomes a qualified positive extension requiring a fresh manuscript integration
and likely one additional confirmation. If it fails, Track A is closed as the clean negative
boundary: the action family contains static-oracle headroom, but PPO does not reliably convert it.
