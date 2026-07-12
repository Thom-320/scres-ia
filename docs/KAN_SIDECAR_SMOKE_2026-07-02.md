# KAN Sidecar Smoke and 5-Seed Sidecar -- 2026-07-02

## Purpose

Garrido et al. (2024) name Kolmogorov-Arnold Networks (KANs), classical
backpropagation neural networks, and reinforcement learning as candidate AI
families for addressing the DES-SCRES "Alzheimer effect." The main Paper 1
claim intentionally uses a conservative PPO+MLP learner so that the structural
result is about bottleneck-aligned action authority, not architectural novelty.

This smoke test checks whether a KAN-style PPO sidecar is technically viable
under the same Track B operational-control contract.

## Implementation

- New feature extractor: `scripts/kan_extractor.py`
- New runner: `scripts/run_track_b_kan_sidecar.py`
- Architecture: PPO with a lightweight KAN-style univariate RBF basis feature
  extractor and linear policy/value heads.
- Canonical contract preserved:
  - `reward_mode=control_v1`
  - `observation_version=v7`
  - `risk_level=adaptive_benchmark_v2`
  - `action_contract=track_b_v1`
  - `max_steps=104`

This is not the official `pykan` package. It is a small dependency-free
Kolmogorov-Arnold-style additive basis layer suitable for smoke testing.

## Smoke Run

Artifact:

`outputs/experiments/track_b_kan_sidecar_2026-07-02/smoke_seed1_10k_h104/`

Command:

```bash
.venv/bin/python scripts/run_track_b_kan_sidecar.py \
  --output-dir outputs/experiments/track_b_kan_sidecar_2026-07-02/smoke_seed1_10k_h104 \
  --seeds 1 \
  --train-timesteps 10000 \
  --eval-episodes 2 \
  --max-steps 104 \
  --n-steps 1024 \
  --batch-size 256 \
  --kan-features-dim 64 \
  --kan-centers 9
```

## Result

The smoke run completed successfully and wrote the normal Track B bundle:

- `episode_metrics.csv`
- `seed_metrics.csv`
- `policy_summary.csv`
- `comparison_table.csv`
- `summary.json`
- `summary.md`

Primary smoke result:

| Policy | Order-level ReT | Shift-utilization cost | Op10 mean | Op12 mean |
|---|---:|---:|---:|---:|
| PPO-KAN | 0.005565 | 0.668 | 1.228 | 1.329 |
| Best static in smoke (`s3_d1.00`) | 0.005123 | 1.000 | 1.000 | 1.000 |
| Best heuristic in smoke (`heur_disruption_aware`) | 0.005325 | 0.458 | 1.391 | 1.391 |

Smoke delta versus best static:

`+0.000442` order-level ReT, `raw_ret_win=True`.

## Claim Boundary

This is a technical viability and early-signal result only. It is not
manuscript-facing evidence because it uses one seed, 10k training steps, and two
evaluation episodes. It should not alter Paper 1's headline.

## Recommended Next Step

The one-seed canonical sidecar and a five-seed sidecar have now been run locally.
The remaining escalation is optional: a 10-seed KAN sidecar can be used as
response-letter ammunition, but it is not needed to lock Paper 1.

## Canonical One-Seed Sidecar

Artifact:

`outputs/experiments/track_b_kan_sidecar_2026-07-02/seed1_60k_h104/`

Command:

```bash
.venv/bin/python scripts/run_track_b_kan_sidecar.py \
  --output-dir outputs/experiments/track_b_kan_sidecar_2026-07-02/seed1_60k_h104 \
  --seeds 1 \
  --train-timesteps 60000 \
  --eval-episodes 12 \
  --max-steps 104 \
  --n-steps 1024 \
  --batch-size 256 \
  --kan-features-dim 64 \
  --kan-centers 9
```

Result:

| Policy | Order-level ReT | Shift-utilization cost |
|---|---:|---:|
| PPO-KAN | 0.005697 | 0.757 |
| Best static in run (`s2_d1.50`) | 0.005225 | 0.667 |

Delta versus best static: `+0.000471`, `raw_ret_win=True`.

## Five-Seed Sidecar

Artifact:

`outputs/experiments/track_b_kan_sidecar_2026-07-02/confirm_5seed_60k_h104/`

Command:

```bash
.venv/bin/python scripts/run_track_b_kan_sidecar.py \
  --output-dir outputs/experiments/track_b_kan_sidecar_2026-07-02/confirm_5seed_60k_h104 \
  --seeds 1 2 3 4 5 \
  --train-timesteps 60000 \
  --eval-episodes 12 \
  --max-steps 104 \
  --n-steps 1024 \
  --batch-size 256 \
  --kan-features-dim 64 \
  --kan-centers 9
```

Primary result:

| Policy | Order-level ReT | Shift-utilization cost | Op10 mean | Op12 mean |
|---|---:|---:|---:|---:|
| PPO-KAN | 0.005705 | 0.818 | 1.769 | 1.895 |
| Best static in run (`s2_d1.50`) | 0.005214 | 0.667 | 1.500 | 1.500 |
| Best heuristic in run (`heur_tuned`) | 0.005221 | 0.592 | 1.404 | 1.404 |

Seed-level PPO-KAN minus best-static deltas:

| Seed | Delta |
|---:|---:|
| 1 | +0.000463 |
| 2 | +0.000479 |
| 3 | +0.000512 |
| 4 | +0.000511 |
| 5 | +0.000487 |

Mean delta: `+0.000490`; normal-approx seed-level CI95:
`[+0.000472, +0.000509]`; sign count: `5/5`.

## Interpretation

The KAN-style sidecar preserves the Track B structural conclusion: when the
action surface reaches the downstream bottleneck, a learned neural policy can
beat the dense downstream-dispatch static family under the canonical Track B
contract. This supports, rather than replaces, the current Paper 1 framing.

Do not rewrite Paper 1 around KAN. The strongest manuscript-safe sentence is:

> A five-seed KAN-style PPO sidecar preserved the positive Track B result,
> suggesting that the finding is not specific to the default MLP feature
> extractor; because this was an architectural sidecar rather than a
> pre-registered gate, we leave full KAN comparison to future work.

Use this in an appendix or response letter only if it helps with Garrido's
architectural-novelty concern. The headline should remain the benchmark
diagnosis, not KAN.
