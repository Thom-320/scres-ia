# Paper Experimental Checklist

Status tracker for all code/experimental work needed before paper submission.
Updated: 2026-03-30.

---

## INFRASTRUCTURE (all DONE)

- [x] DES model validated (-4.43% vs thesis)
- [x] Gymnasium env (MFSCGymEnvShifts, 5-dim action, 15-24 dim obs)
- [x] ReT_unified_v1 reward function implemented
- [x] Garrido exact baselines (garrido_cf_s1/s2/s3)
- [x] Static baselines (static_s1/s2/s3)
- [x] Heuristic baselines (hysteresis, disruption-aware, tuned)
- [x] Random baseline
- [x] Baseline evaluation script (evaluate_all_baselines.py)
- [x] PPO training pipeline (train_agent.py + SB3)
- [x] RecurrentPPO wired in benchmark
- [x] Frame-stacking (VecFrameStack)
- [x] PBRS (control_v1_pbrs)
- [x] Cross-eval flag (--eval-risk-levels current increased severe)
- [x] Step trajectory logging (proof_trajectories.csv)
- [x] Proof-of-learning artifact generator
- [x] DKANA handoff guide (DKANA_CONTRIBUTOR_HANDOFF.md)
- [x] Export pipeline (export_trajectories + build_dkana_dataset)
- [x] CI95 bootstrap in policy_summary.csv

---

## CRITICAL (blocks paper submission)

- [ ] **C1. ReT_unified_v1 calibration finalized**
  - Codex found and fixed a bug where JSON always overrode CLI params
  - Extended grid running (60 combos, kappa up to 1.0)
  - Need: valid candidate where best static has correct ranking
  - Note: S2 does NOT have to be optimal. Any ranking is fine as long as PPO can beat the best static.

- [ ] **C2. Production PPO run with ReT_unified_v1 (500k x 10 seeds)**
  - BLOCKED by C1 (need finalized calibration first)
  - Scenarios: increased + cross-eval severe
  - Est: 4-8 hours compute

- [ ] **C3. Statistical comparison table with formal tests**
  - Mann-Whitney U or Welch t-test for PPO vs each baseline
  - Cohen's d effect sizes
  - Bootstrap CI95
  - Need: new script or extend analyze_paper_benchmark_trio.py
  - Est: 2-4 hours code

- [ ] **C4. Verify PPO beats best static on reward AND fill_rate**
  - BLOCKED by C2
  - Acceptance criteria:
    - PPO >= best_static on ReT_unified_v1
    - PPO >= best_static on fill_rate
    - No shift > 80% (no collapse)

---

## IMPORTANT (strengthens paper significantly)

- [ ] **I1. Section 4.3 algorithm comparison under unified reward**
  - PPO-v1, PPO-v2, PPO+frame-stack-4, RecurrentPPO
  - All under ReT_unified_v1
  - BLOCKED by C1
  - Est: 8-12 hours compute

- [ ] **I2. Sensitivity analysis on reward hyperparameters**
  - Calibration grid with 10+ episodes per combo
  - Partially done by Codex's extended grid
  - Est: 2-4 hours compute

- [ ] **I3. Severe stress cross-evaluation**
  - Train on increased, evaluate on severe
  - "Headline finding": RL advantage under unseen stress
  - Included in C2 production run

- [ ] **I4. Garrido2024 precedence fix**
  - Apply same parameter-override fix to ReT_garrido2024_* family
  - Preventive patch, same pattern as unified bug

---

## NICE-TO-HAVE (Q1 differentiators)

- [ ] N1. PBRS comparison (PPO+PBRS vs PPO)
- [ ] N2. SAC comparison (alternative RL algorithm)
- [ ] N3. 7 stress scenarios (demand spike, supplier failure, etc.)
- [ ] N4. Sensitivity to disruption parameters
- [ ] N5. Recovery time analysis
- [ ] N6. Formal reproducibility appendix (frozen requirements.txt, commit hash, run-to-figure)

---

## DEPENDENCY GRAPH

```
C1 (calibration) ──→ C2 (500k run) ──→ C4 (verify PPO beats static)
                                    ──→ I3 (severe cross-eval)
                 ──→ I1 (Sec 4.3 comparison)
                 ──→ I2 (sensitivity)

C3 (stats script) ── independent, can start now

I4 (garrido fix) ── independent, can start now
```

---

## CURRENT BLOCKERS

1. **Codex's extended calibration grid is running** (PID 86041, ~60 combos)
   - Once done: either we have a valid calibration or we need to redesign the cost term
2. **No completed 500k run with unified reward exists**
3. **Statistical test script doesn't exist yet**

---

*This file is the source of truth for experimental completion. Update checkboxes as items are completed.*
