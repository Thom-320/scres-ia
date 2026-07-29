# Track B Real-KAN batch-size sweep verdict - 2026-07-06

## Status

Completed on the VPS and fetched locally.

Artifacts:

- `outputs/experiments/track_b_realkan_batch_size_sweep_32_3seed_30k_2026-07-06/`
- `outputs/experiments/track_b_realkan_batch_size_sweep_64_3seed_30k_2026-07-06/`
- `outputs/experiments/track_b_realkan_batch_size_sweep_128_3seed_30k_2026-07-06/`
- `outputs/experiments/track_b_realkan_batch_size_sweep_256_3seed_30k_2026-07-06/`
- `outputs/experiments/track_b_realkan_batch_size_sweep_512_3seed_30k_2026-07-06/`

Protocol verified from summaries:

- Architecture: Real-KAN sidecar (`ppo_real_kan`)
- Risk: `adaptive_benchmark_v2`
- Observation: `v7` full
- Reward: `control_v1`
- Seeds: `1,2,3`
- Train timesteps: `30000`
- Eval episodes: `8`
- Horizon: `104`
- `n_steps=1024`
- `learning_rate=3e-4`
- `gamma=0.99`
- `gae_lambda=0.95`
- Batch sizes: `32,64,128,256,512`

Important scope note: this is **adaptive_benchmark_v2 / v7-full**, not the
reviewer-safe `v7_no_forecast` protocol. Use it to tune Real-KAN training, not
as a no-forecast final claim.

## Results

| Rank | Batch size | ReT Excel | CVaR05 | Best static | Delta vs static | Relative delta | Cost |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 512 | 0.005934737 | 0.002559888 | `s3_d1.50` | +0.000520 | +9.61% | 0.902 |
| 2 | 128 | 0.005926983 | 0.002525568 | `s3_d1.50` | +0.000512 | +9.46% | 1.000 |
| 3 | 64 | 0.005920892 | 0.002487260 | `s3_d1.50` | +0.000506 | +9.35% | 0.906 |
| 4 | 256 | 0.005920544 | 0.002489971 | `s3_d1.50` | +0.000506 | +9.35% | 0.906 |
| 5 | 32 | 0.005916057 | 0.002425837 | `s3_d1.50` | +0.000502 | +9.26% | 0.708 |

## Reading

The sweep does **not** support `256` as uniquely best for Real-KAN. Under this
short adaptive-v2/v7-full protocol, `512` is best on both mean Excel ReT and
CVaR05, while keeping cost close to the 64/256 cells.

The differences are small in absolute terms:

- `512` beats `256` by about `0.0000142` ReT Excel.
- `512` beats `128` by about `0.0000078` ReT Excel.
- All batch sizes beat the same best static by roughly +9.3% to +9.6%.

The practical tradeoff is:

- `512`: best ReT/CVaR in this sweep, good candidate for Real-KAN adaptive-v2
  confirmation.
- `128`: close ReT/CVaR but collapses to cost 1.0, less attractive.
- `64` and `256`: essentially tied.
- `32`: cheapest, but lowest ReT/CVaR.

## Recommendation

For future Real-KAN adaptive-v2/v7-full work, use `batch_size=512` as the
candidate to confirm, not `256`.

For no-forecast reviewer-facing work, do **not** automatically switch protocols
based only on this sweep. The final A/B h104 no-forecast confirmation already
used Real-KAN `batch_size=256`; if Real-KAN is promoted beyond sidecar status,
the next clean step would be a focused no-forecast Real-KAN confirmation at
`batch_size=512` under the same A/B or selected Case C protocol.

Bottom line:

`batch_size=512` is now the best Real-KAN tuning candidate, but the paper spine
remains PPO+MLP no-forecast unless a same-protocol Real-KAN confirmatory run
beats it.
