# Kaggle H4 Δmemory Runtime Audit (2026-06-29)

## Question

Why did the H4 retained-vs-reset continuous run fail/cancel on Kaggle when other long Kaggle runs completed?

## Finding

The H4 `retention_transfer.py --track continuous` workload is materially heavier than the previous successful continuous/Pareto kernels.

Successful kernels such as `scresia-preventive-pareto-final`, `scresia-preventive-pareto-dense-crn`, and `scresia-continuous-its-confirm` ran one direct PPO experiment with vectorized training (`n_envs=4`) and wrote a final `summary.json`/`decision.json` after roughly 1-2 hours.

The H4 runner is a different protocol:

- For each block it evaluates three cold arms: `frozen`, `retained`, `reset`.
- For each block it trains two learners: `retained` and `reset`.
- For continuous PPO, those updates are on-policy and non-vectorized in this helper path.
- The script only wrote `transfer.json` after all seeds finished.

In the Kaggle logs, `scresia-continuous-dmemory-probe-v2-cpu` reached:

```text
[transfer] seed 8201 ...
[transfer] seed 8202 ...
```

at about `29291` seconds, meaning seed `8201` alone took about 8.1 hours. The kernel was later cancelled before seed `8202` finished, so no `transfer.json` or `decision.json` was written. This is a runtime/write-late failure, not a scientific null.

## Corrective Action

1. Do not run multi-seed H4 as one Kaggle job until seed-level partial writes exist.
2. Run one seed per Kaggle kernel or one seed per subprocess so completed seeds produce artifacts.
3. Keep CPU forced (`CUDA_VISIBLE_DEVICES=""`, `enable_gpu=false`) because the workload is SimPy/CPU-bound and Kaggle GPU images can trigger PyTorch/CUDA compatibility noise.
4. Use `python -u` and a watcher that downloads outputs on completion/cancel.

## Current Relaunches

- `scresia-continuous-dmemory-probe-v3-micro`: micro directional probe, running.
- `scresia-continuous-dmemory-seed-confirm-v1`: one-seed fuller confirm profile, prepared to run `seed=8201`, `n_blocks=4`, `max_steps=12`, `train_per_block=1000`, `n_steps=128`, `n_epochs=3`.

## Claim Boundary

No H4 Δmemory result exists yet. The prior cancelled jobs are not evidence for or against retained learning.
