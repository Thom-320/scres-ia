# E2: obs-masked PPO retrain — verdict (2026-07-02)

**Status: DONE.** Completed on Kaggle
(`thomaschisica/scresia-track-b-e2-obs-masked-confirm`, kernel v2,
`KernelWorkerStatus.COMPLETE`), fetched to
`outputs/experiments/track_b_e2_obs_masked_confirm_2026-07-02/fetched_kaggle_complete/`.
5 seeds × 60k timesteps, exact canonical hyperparameters (control_v1,
adaptive_benchmark_v2, h104, lr=3e-4, n_steps=1024, batch=256, n_epochs=10),
observation `v7_no_regime_forecast` — the true regime one-hot (5 dims) and
true-transition-matrix forecast fields (2 dims) masked to zero, everything
else in obs v7 unchanged. **Independently reproduced on the OVH VPS**
(separate infrastructure, separate Python/library versions, same code and
config) — `outputs/experiments/track_b_e2_obs_masked_confirm_vps_2026-07-02/`.

## Result

| | PPO, masked (Kaggle) | PPO, masked (VPS) | PPO (canonical, full v7) | Best static |
|---|---:|---:|---:|---:|
| order_level_ret_mean | **0.005605** | **0.005613** | 0.005666 | 0.005210 (masked-arm best: `s3_d2.00`) / 0.005251 (canonical) |
| Δ vs best static | **+0.000395** | **+0.000403** | +0.000415 | — |
| Raw ReT win | **True** | **True** | True | — |
| Rolling fill 4w | 1.000 | — | — | 0.859 (best static) |

Kaggle and VPS agree to within 0.000008 (seed-to-seed noise scale) despite
running on completely independent infrastructure — strong evidence the
result is a genuine, reproducible property of the policy, not an
environment artifact.

**The win survives losing the privileged regime/forecast observation
fields, retaining ~95% of the canonical delta** (+0.000395 vs +0.000415).
PPO still beats every static/heuristic comparator in this arm (`raw ReT
win: True`) and achieves perfect rolling fill (1.000) despite not observing
the true disruption regime or the true-transition-matrix forecasts.

## Verdict on the privileged-observation attack (T3, docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md)

**Answered, favorably — this closes the single highest-priority open
threat from the audit.** The concern was that PPO's win might reduce to
exploiting a five-constant regime-lookup signal available only because the
simulator hands the agent its own ground-truth latent state. With that
signal (and the model-derived forecasts) zeroed out, PPO still wins by
almost the same margin. The agent is learning genuine closed-loop control
from observable state (backlog, queue pressure, rolling fill, downstream
multiplier history), not primarily reading privileged internals.

Safe wording:

> "To rule out the possibility that the learned policy's advantage reduces
> to exploiting simulator-privileged observation fields — a one-hot
> encoding of the true disruption regime and forecasts derived from the
> true transition matrix — we retrained PPO with those seven fields masked
> to zero. The policy retains 95% of the canonical Excel/order-level ReT
> advantage over the best static comparator (+0.000395 vs +0.000415) and
> continues to beat every evaluated static and heuristic policy, indicating
> the gain is driven by adaptive response to observable operational state
> rather than access to privileged simulator internals."

This does **not** replace the E1 regime-table/heuristic comparison
(`scripts/build_track_b_e1_go_no_go.py`, pending E1's completion) — that
test answers a related but distinct question (does a *zero-learning*
comparator that also has access to the privileged fields match PPO?). Both
together give the strongest possible defense: PPO doesn't need the
privileged fields (this result), and a non-learning policy that DOES have
them still can't match PPO (E1, pending).

## Registry update

Add to `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` (new item, T3
resolution): "PPO's Track B advantage does not depend on privileged
regime/forecast observation fields — **Supported**, obs-masked retrain
retains ~95% of canonical delta (docs/E2_PRIVILEGED_OBSERVATION_VERDICT_2026-07-02.md)."
