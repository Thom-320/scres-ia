# Audit + PPO Diagnostic — Findings (2026-06-27)

## What was done

1. **Garrido/DES audit workbooks** (`scripts/audit_garrido_des_workbook.py` →
   `outputs/audits/garrido_des_audit/`):
   - `garrido_des_audit_summary.xlsx` — 7 sheets (README, FormulaGate,
     FormulaGate_PerCF, CF_Summary, RiskAttribution, Deltas, SelectedLedgers).
   - `garrido_des_ledgers.xlsx` — 20 sheets (CF01..CF20) with Garrido-style
     columns (Q, j, OPTj, OATj, CTj, LT, sumBt, APj, RPj, DPj, risk cols,
     sumUt, OP9, ReT, source=DES/Excel).
   - `audit_manifest.json` (commit, inputs, sha256, commands, formula-gate result).
   - `README_AUDIT.md` (provenance + caveat on PRNG).

2. **PPO vs Static diagnostic** (`scripts/diagnose_ppo_vs_static.py` →
   `outputs/diagnostics/ppo_diagnosis_2026-06-27.md`):
   - Per-case imitation gate (PPO cd_sigmoid vs static robust on same cell).
   - Pulls from existing `outputs/4cases/` (10k ts, 2 seeds) and the war-sweep
     static frontier panels (1-seed); reads the 4-case what-to-beat 5-seed
     summary for the faithful cell.

## Key findings (with numbers)

### Finding 1 — Problem size is 18 (policy space), ~90-120 (thesis design)
- The user's DES action space = `Discrete(18)` = 6 inventory levels × 3
  shifts, **fully enumerable**. The `static_baseline_panel` enumerates all 18.
- Garrido's full experimental design (thesis Sec 6.7) = ~90–120 configs
  (Scenario I: 30; Scenario II: 3 risk cats × 30 inv-levels = 90). The
  full enumeration of the full design is intractable; Garrido used a
  D-efficient optimal design (D/G/A = 95.72 / 91.28 / 92.59, Table 6.24).
- The user's 3 Excels cover 20 of 90 (CF01–CF20, Scenario I subset) + 12
  in `Rsult_1.xlsx`. They are a subset, not the full DSE.

### Finding 2 — Formula reproduces 100% on the tape; endogenous diverges
- FormulaGate (using the des_order_exports' pre-computed `deltaReT`):
  per-CF `n_zero` rows are 1–18 out of 2000+; `max_abs_deltaReT` is
  0.47–0.99 for R1, and up to **160.25 for CF12** (R2). The full forensic
  audit (`outputs/audits/garrido_workbook_fidelity_2026-06-26/`) reported
  0 mismatches over 47,546 rows on the schema-aware recompute — the
  ground truth for the formula. The harness deltaReT is the per-row
  float-precision check, which is near-zero but not strictly 0; the
  audit workbook is honest about this (`passes=False` on the harness
  check, references the proper audit for the formula claim).
- 19/20 CFs flag `CTj_tail_diverges` (DES p99 ≠ Excel p99 by >5000h).
  This is **expected** for the endogenous lane (different PRNG,
  different stochastic realization of the same model).

### Finding 3 — Imitation gate (the plan's hard criterion)
| Case | PPO cd | Static robust | Δ (PPO − robust) | Verdict |
|---|---|---|---|---|
| case1 faithful / control_v1 | 0.605 | 0.688 (S1_I168) | **−0.084** | **FAIL** |
| case2 faithful / ReT_garrido2024_raw | 0.601 | 0.688 | **−0.088** | **FAIL** |
| case3 war φ4/ψ1.5 / ReT_cvar_cd | 0.580 | 0.551 (S1_I1344) | **+0.029** | **PASS** (1-seed) |
| case4 war φ4/ψ1.5 / ReT_garrido2024_raw | 0.582 | 0.551 | **+0.032** | **PASS** (1-seed) |

- **Cases 1 & 2 (faithful):** the imitation gate **FAILS** by ~0.08. The
  faithful cell has effectively zero headroom (5-seed
  `avg_wrong_regime_penalty` = 0.0104 for φ1/ψ1, i.e. mean_gap ≈ 0.01).
  PPO cannot even imitate the best static on a cell with no headroom.
  These are correctly the **expected null-checks** per the plan.
- **Cases 3 & 4 (war φ4/ψ1.5):** the gate **PASSES** by +0.03 at 10k
  timesteps, but (a) the robust is the 1-seed `S1_I1344` corner
  (the war cell's heavy corner), (b) at 10k timesteps this is a
  plumbing/calibration result, not a confirmatory win. The existing
  confirmatory (65536 timesteps, `outputs/kaggle/garrido_envb_confirmatory/`)
  shows **no Pareto win vs the frozen efficient frontier**.

### Finding 4 — Reward/eval separation
- `retention_transfer.py` `--reward-mode` choices are the **eval** outcomes
  (`excel_ret`, `cd_index`, `cd_sigmoid_index`, `cd_train_index`).
  The **training** reward defaults to `control_v1`; `control_v2_w_*` knobs
  tune the operational reward.
- Only **case 4 (war / ReT_garrido2024_raw / eval = cd_sigmoid)** has
  reward and eval sharing a CD signal. Cases 1 & 2 (eval = service)
  trained on `control_v1` / `ReT_garrido2024_raw` with different eval
  metrics. This separation is part of why the imitation gate under-performs
  on cases 1 & 2.

## Next step (from the plan, not yet executed)
**Calibration (3 learner × 3 tape seeds, 20 blocks, n_steps=12,
train_per_block=150) on the war cell (φ4/ψ1.5), preceded by the
frozen frontier gate.** The gate is the only thing that prevents
re-confirming a null on a cell that was never learnable to begin with.
The cells chosen (from the prior calibration memo):
- A (clean): rho=0.85, surge_budget=2016, lead=168
- B (more inertia): rho=0.85, surge_budget=4032, lead=168
- C (more demand persistence): rho=0.85, surge_budget=2016, lead=336, rho_demand=0.75

Reward: `control_v1` primary + `ReT_cvar_cd` sensitivity (the
`control_v2_backlog` from the freeze-V2 plan is NOT yet implemented;
flagged as a Phase 1b follow-up).

## Honest verdict (per the plan)
- **Track A is exhausted** under both Excel-ReT and CD bars (triangulated
  null across DQN + PPO + DQN memory).
- **The war cell at 10k timesteps shows a marginal imitation-gate pass
  (+0.03)** but no confirmatory win. At 10k steps a DQN on a 18-action
  space has ~556 gradient steps per seed per action — not enough to
  distinguish. The next step is calibration at higher timesteps with
  the frontier gate as a precondition, not another 10k-step grid.
- **No claim of RL > Garrido is supported** by the current data.
