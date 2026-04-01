# Paper Experimental Checklist

Source of truth for the remaining code and experimental work before the paper is submission-ready.
Updated: 2026-03-31.

This checklist now freezes the **Track B** paper backbone:

1. `track_b_adaptive_control`,
2. `ReT_seq_v1` with `κ=0.20`,
3. `v7 + track_b_v1`,
4. `168h` cadence,
5. `500k x 5` auditable benchmark completed.

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

### Frozen Track B contract

- [x] Online contract frozen as `reward_mode="ReT_seq_v1"` + `observation_version="v7"` + `action_contract="track_b_v1"`
- [x] Track B launcher is auditable
- [x] `ReT_garrido2024` remains available as an audit/bridge metric
- [x] `ReT_unified_v1` remains available only as an exploratory follow-up lane

### Deferred from the paper backbone

- [x] `ReT_unified_v1` is implemented and usable
- [x] `ReT_unified_v1` is **not** the frozen main paper contract
- [x] Track A remains part of the paper, but not as the primary positive result

---

## C. Critical Path

These items block paper submission.

### C1. Freeze evidence

- [x] Track A comparator bundles audited
- [x] Track B smoke completed
- [x] Track B `500k x 5` completed
- [ ] Generate final figure/table bundle for the manuscript

### C2. Formal statistical comparison

- [ ] Add p-value table
- [ ] Add pairwise tests for Track B PPO vs:
  - `s2_d1.00`
  - `s3_d2.00`
- [ ] Add effect size:
  - Cohen's d or non-parametric alternative if justified
- [ ] Report Track A negative controls separately from Track B positive result

---

## D. Important but Not Blocking

### D1. Manuscript framing for the manuscript

- [ ] Freeze Section 4.3 around:
  - Track A failure across PPO / RecurrentPPO
  - Track B success with PPO
  - minimal MDP repair explanation

### D2. Exploratory lanes (not paper backbone)

- [ ] Keep `ReT_unified_v1` in Track B as exploratory/audit evidence only
- [ ] Keep `v5 + 48h` and other Track A variants out of the core paper claim

### D3. Track A as negative evidence

- [x] Verify Track A PPO does not beat strong static baselines
- [x] Verify RecurrentPPO does not rescue Track A
- [ ] Freeze the Track A negative-results table used in the paper

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

- [x] The paper no longer depends on proving `control_v1` or `RecurrentPPO` as the winning lane
- [x] The next step is not another reward redesign
- [x] `final_ret_seq_v1_500k` remains a historical comparator, not the main paper lane
- [x] The decisive positive result is now `Track B + ReT_seq_v1 κ=0.20`

---

## G. Stop / Go Rules

### STOP

Stop changing the paper backbone now that:

- [x] the final Track B `500k x 5` run has completed

At this point:

- no more reward redesign,
- no more control-contract redesign for the main paper,
- no new primary experiment unless it is strictly a statistical replicate.

---

## H. Submission-Readiness Checklist

The paper is experimentally done only when all items below are checked:

- [x] Final Track A contract frozen
- [x] Final Track B contract frozen
- [x] Final weights frozen (`w_bo=4.0`, `w_cost=0.02`, `w_disr=0.0`)
- [x] Final RecurrentPPO production benchmark completed
- [ ] Statistical comparison table generated
- [ ] Statistical test table generated
- [ ] Track A vs Track B manuscript tables generated
- [ ] DKANA-ready handoff bundle generated
- [ ] Historical comparator bundle retained

---

## I. Current Best Interpretation

As of this update:

- the repo is experimentally **past** the main blocker;
- the main blocker is now manuscript packaging, not infrastructure;
- the core paper story is Track A negative result plus Track B positive result.

The next move should be manuscript production, not more backbone redesign.
