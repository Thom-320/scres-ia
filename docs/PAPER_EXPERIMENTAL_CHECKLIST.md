# Paper Experimental Checklist

Source of truth for the remaining code and experimental work before the paper is submission-ready.
Updated: 2026-03-30.

This checklist now freezes the **Track A** paper backbone:

1. thesis-faithful `shift_control`,
2. `control_v1` as the primary training reward,
3. `v4` as the paper-facing observation contract,
4. `168h` cadence,
5. one final `RecurrentPPO 500k x 5` run before manuscript lock.

---

## A. What Is Already Done

### Infrastructure

- [x] DES model rebuilt and validated against thesis reference targets
- [x] Gymnasium environments available (`base`, `shift_control`)
- [x] Observation contracts available (`v1`, `v2`, `v3`, `v4`)
- [x] Exact Garrido static baselines implemented (`garrido_cf_s1/s2/s3`)
- [x] Historical static baselines implemented (`static_s1/s2/s3`)
- [x] Heuristic baselines implemented (`heuristic_hysteresis`, `heuristic_disruption`, `heuristic_tuned`)
- [x] Random baseline implemented
- [x] PPO training pipeline available
- [x] RecurrentPPO wired into benchmark harness
- [x] Frame-stacking supported
- [x] PBRS comparator supported (`control_v1_pbrs`)
- [x] Cross-scenario evaluation supported (`current`, `increased`, `severe`)
- [x] Trajectory export pipeline available
- [x] DKANA dataset builder available
- [x] DKANA handoff guide aligned to the frozen online/offline contract
- [x] Bootstrap CI95 already appears in aggregated policy summaries

### Historical reference runs

- [x] Historical `ReT_seq_v1` long run completed:
  [final_ret_seq_v1_500k](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/benchmarks/final_ret_seq_v1_500k)
- [x] Historical result is usable as a comparator/baseline
- [ ] Historical result is **not** the final paper evidence, because it uses `ReT_seq_v1 + v1`, not the frozen paper contract

---

## B. Current Paper Contract Status

### Frozen Track A contract

- [x] Online contract frozen as `reward_mode="control_v1"` + `observation_version="v4"`
- [x] Offline export contract frozen around the same observation contract
- [x] `ReT_garrido2024` remains available as an audit/bridge metric
- [x] `ReT_unified_v1` remains available only as an exploratory/audit lane

### Deferred from the paper backbone

- [x] `ReT_unified_v1` is implemented and usable
- [x] `ReT_unified_v1` is **not** the frozen Track A paper contract
- [x] `v5 + 48h` remains exploratory because random is too close and the cadence/reward story is not frozen enough for the main paper claim

---

## C. Critical Path

These items block paper submission.

### C1. Verify the frozen Track A lane still runs cleanly

- [ ] Smoke `RecurrentPPO + control_v1 + v4 + 168h`
- [ ] Confirm the benchmark bundle is written cleanly
- [ ] Confirm cross-eval still supports `current`, `increased`, `severe`

### C2. Final production experiment

- [ ] Run exactly one final production benchmark:
  - `algo=recurrent_ppo`
  - `reward_mode=control_v1`
  - `observation_version=v4`
  - `step_size_hours=168`
  - `risk_level=increased`
  - `eval_risk_levels=current increased severe`
  - `seeds=11 22 33 44 55`
  - `train_timesteps=500000`
- [ ] Write full benchmark bundle with `comparison_table.csv`, `policy_summary.csv`, `proof_trajectories.csv`, `summary.json`

### C3. Formal statistical comparison

- [ ] Add p-value table
- [ ] Add at least one formal pairwise test:
  - Welch's t-test or Mann-Whitney U
- [ ] Add effect size:
  - Cohen's d (or equivalent justified effect size)
- [ ] Report tests for learned policy vs:
  - `static_s2`
  - best Garrido baseline
  - best heuristic baseline
  - `random`

---

## D. Important but Not Blocking

### D1. Algorithm comparison for the manuscript

- [ ] Freeze Section 4.3 around:
  - PPO + MLP
  - PPO + frame-stack
  - RecurrentPPO
  - optional PBRS comparator if already usable and non-distracting

### D2. Exploratory lanes (not paper backbone)

- [ ] Keep `ReT_unified_v1` as exploratory/audit evidence only
- [ ] Keep `v5 + 48h` out of the core paper claim

### D3. Severe cross-evaluation story

- [ ] Verify whether PPO shows advantage, parity, or degradation under `severe`
- [ ] Freeze the cross-scenario table used in the paper

---

## E. Nice-to-Have Differentiators

- [ ] SAC ablation
- [ ] Extended stress scenarios beyond the standard three
- [ ] Recovery-time analysis after disruptions
- [ ] Formal reproducibility appendix with exact commands + pinned environment + commit hash

---

## F. Corrected Status Against the Earlier Plan

### Still true

- [x] The repo already has the necessary infrastructure for a publishable paper
- [x] The repo already has enough baselines
- [x] The DKANA contributor can already work against a frozen env/data contract
- [x] The missing backbone is now mostly experimental evidence, not plumbing

### Corrected

- [x] The paper no longer depends on proving `ReT_unified_v1` as the primary train reward
- [x] The next step is not another reward redesign
- [x] `final_ret_seq_v1_500k` remains a historical comparator, not the main paper lane
- [x] The one remaining decisive experiment is `RecurrentPPO + control_v1 + v4 + 168h`

---

## G. Stop / Go Rules

### STOP

Stop changing the paper backbone if:

- [ ] the final `RecurrentPPO 500k x 5` run has completed

At that point:

- no more reward redesign,
- no more observation redesign,
- no new Track B code before submission.

---

## H. Submission-Readiness Checklist

The paper is experimentally done only when all items below are checked:

- [x] Final Track A contract frozen
- [x] Final weights frozen (`w_bo=4.0`, `w_cost=0.02`, `w_disr=0.0`)
- [ ] Final RecurrentPPO production benchmark completed
- [ ] Cross-scenario evaluation completed
- [ ] Statistical test table generated
- [ ] Proof-of-learning artifacts generated
- [ ] DKANA-ready handoff bundle generated
- [ ] Historical comparator bundle retained

---

## I. Current Best Interpretation

As of this update:

- the repo is **close** to paper-ready;
- the main blocker is **not** infrastructure;
- the main blocker is completing one final long-run memory test under the frozen Track A contract.

After that run, the paper should move to manuscript production, not further backbone redesign.
