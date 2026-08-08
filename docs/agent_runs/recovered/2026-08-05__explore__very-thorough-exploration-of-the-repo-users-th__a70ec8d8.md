# Explore — Very thorough exploration of the repo <HOME>/Projects/research/scres-ia.

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a70ec8d81e30dd9bf`
- **session**: `091a3905-ebf9-481b-beaa-fc646c3902d5`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-05T18:01:19.742Z`
- **effort**: `xhigh` · **version**: `2.1.219`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/091a3905-ebf9-481b-beaa-fc646c3902d5/subagents/agent-a70ec8d81e30dd9bf.jsonl`
- **turns**: 167 · **assistant messages**: 2

## Task given

```text
Very thorough exploration of the repo <HOME>/Projects/research/scres-ia.

I need a complete map of the "Alzheimer effect / memory-carrying neuron / meta-learner across simulation RUNS" lane. This is the lane where a Fig-5-style neuron learns ACROSS successive DES runs (configuration x vs configuration x-1), as opposed to within-episode RL control.

Find and report:

1. All code implementing the memory neuron / meta-learner / across-run learner. Look for files matching patterns like *meta_learner*, *alzheimer*, *fig5*, *neuron*, *des288*, *garrido_q2*. Report exact file paths, the key functions/classes, and what the learner actually is (perceptron? surrogate? bandit?).

2. All contracts (contracts/*.json) and preregistrations (docs/PREREGISTRO*, docs/*META*, docs/*ALZHEIMER*) that govern this lane. Quote their status fields, primary estimands, falsifiers and any authorization gates.

3. All results directories for this lane, especially: results/garrido_meta_learner_h3power_local, results/garrido_meta_learner_h3power_vps, results/garrido_meta_learner_smoke_v2, results/garrido_meta_learner_thesis90_v2, results/garrido_q2_des288_v1. For each, report claim_status, the headline numbers (runs-to-optimum, advantage, CIs), and whether falsifiers passed.

4. The measured headline "memory-carrying Fig.5 neuron reaches the optimum in 7.24 runs vs 13.54 reset and 12.42 for the thesis's OFAT; advantage grows +0.00 -> +10.00 across six contexts" — find where exactly this is recorded (doc + result json), and report the full context: what the six contexts are, what the comparators are, what the seeds/replications are, and any stated limitations.

5. What comparators for SEARCH over the configuration grid already exist in code (OFAT, random, best-first, branch-and-bound, Bayesian optimization, grid). And what does NOT exist yet.

6. The 288-configuration grid (or 90-config Garrido reproduction): where is it defined, what are the axes/levels, and is there a cached/precomputed evaluation of the full grid (which would make meta-search cheap)?

Report file paths and exact quoted numbers. Do not summarize vaguely — I need specifics I can cite in a plan.
```

## Final answer

I have a complete map. Here it is.

## 1. Code: the memory neuron / meta-learner / across-run learner

There are **four distinct implementations** of the across-run learner in this repo. All of them are **online linear/logistic learners, not RL, not bandits, not surrogate-model BO**.

### 1a. `<HOME>/Projects/research/scres-ia/scripts/run_meta_learner_over_configs_v1.py` (500 lines) — the canonical 288-config DES runner

- `class Fig5Neuron` (line 122): `self.rho = np.zeros(dim)`, `predict` = `1/(1+exp(-clip(rho@x, -30, 30)))`, `update` = `self.rho += self.lr * (y - self.predict(x)) * x`, `lr=0.35`. **A single logistic (perceptron-style) unit with plain logistic-loss gradient steps.** Dim = 5 (4 factor coordinates normalised to [0,1] + bias).
- `def features(config)` (line 101): **decision coordinates only** — `[FACTORS[n].index(config[n])/(len-1) for n in FACTOR_NAMES] + [1.0]`. This is the post-leak-fix version; the drivers are update-signal only.
- `def evaluate(config, context, seed, horizon)` (line 77): runs a real `MFSCSimulation`, returns `(ret_excel_risk_conditional, 4-driver vector)`.
- `def search(strategy, seed, rng)` (line 183): four arms — `ofat`, `random`, `neuron_reset`, `neuron_memory`. Greedy argmax over unseen configs after a 3-run cold start (`if len(seen) < 3: idx = int(rng.choice(unseen))`). **No exploration bonus, no UCB, no Thompson — pure greedy exploitation of the linear predictor.**
- OFAT is generated **lazily from the incumbent** (line 199-217) precisely so `f2` (one coordinate changed per proposal) can pass.
- Falsifiers `f1`–`f6` built at lines 379–438; `f5` is the shadow-surface driver-permutation test (lines 258–279).
- `--contract` is **required, no default** (line 148), with an in-code comment that the default "is how the H3-prime slices got sealed against the wrong document".

### 1b. `<HOME>/Projects/research/scres-ia/scripts/run_garrido_q2_des288_v1.py` (804 lines) — the preregistered Q2 DES-288 runner

- `class VectorLinearLearner` (line 205): `self.rho = np.zeros((len(FACTOR_NAMES)+1, len(SERVICE_FIRST_V2_COMPONENTS)))` — a **5×4 linear coefficient matrix**, LMS update `self.rho += lr * np.outer(x, target - prediction)`, `lr=0.25`. Predicts the **four endpoint components separately** and ranks by **lexicographic tuple comparison** (no scalarisation).
- Five arms: `{"ofat", "random", "no_update", "retained", "reset"}`.
- Endpoint: `service_first_resilience_v2` = `(worst_claimant_fill, flow_fill_rate, -backorder_qty_final, ret_excel_visible_clipped_0_1)`.
- Nine falsifiers `f1`–`f9` (lines 537–621).
- Metric is `runs_to_oracle` (exact match to the oracle key), not `runs_to_within_1pct`.

### 1c. `<HOME>/Projects/research/scres-ia/scripts/run_meta_learner_thesis90_v1.py` (318 lines) — thesis-native 90-cell **replay**

- Same `class Fig5Neuron` (line 66). `features(row)` (line 55) = `[buffer_hours/1344, (shifts-1)/2, family one-hot(3), scenario one-hot(3), pattern flags(4), bias]` → dim 13.
- **Does not run the DES.** `load_surface()` reads `results/garrido_drivers_per_configuration/result.json` and refuses if `claim_status != "DEVELOPMENT_DRIVER_TABLE"`.
- Arms: `thesis_order` (deliberately *not* called OFAT), `random`, `neuron_memory`, `neuron_reset`. Contexts are the nine published blocks `H1a…H3c`.

### 1d. `<HOME>/Projects/research/scres-ia/scripts/run_garrido_wrap_closed_loop.py` (501 lines) — the WRAP closed-loop smoke

- `class BetweenRunLearner` (line 150): retained state is the **observation list**, not weights. Refits from scratch each selection with `kind ∈ {"linear", "backprop", "kan"}` (`sklearn.LinearRegression`, `fit_mlp`, `fit_kan`). **This is the only lane member that can swap in an MLP/KAN surrogate.** Default is `linear` because the Q1 gate says `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL`.
- Has an in-process DES cache (line 453: `cache: dict[tuple[int, float | None], dict]`), reported as `simulation_cache_entries`. **Not persisted to disk.**

### Adjacent / supporting
- `<HOME>/Projects/research/scres-ia/scripts/build_garrido_fig5_surrogate.py` (459) — Fig-5-as-drawn (drivers→ReT) vs learnable version; backprop/KAN/linear/constant.
- `<HOME>/Projects/research/scres-ia/scripts/run_continuous_space_fig5.py` (214) — continuous-space version of the Fig-5 fit question.
- `<HOME>/Projects/research/scres-ia/scripts/run_h1_h3_v2.py:191` — `search_cost_variance(strategy)`, the H3′ estimand, computed **off the sealed meta-learner artifact with no new simulation**.
- `<HOME>/Projects/research/scres-ia/scripts/adjudicate_h3_merge_v1.py`, `<HOME>/Projects/research/scres-ia/scripts/merge_garrido_h3_power_v1.py`.
- Tests: `<HOME>/Projects/research/scres-ia/tests/test_garrido_q2_des288_runner.py` (136 lines), `tests/test_garrido_wrap_custody.py`, `tests/test_seed_custody_module.py:116`.

## 2. Contracts and preregistrations

**There is no `contracts/*.json` for this lane.** I grepped all 68 files in `contracts/` for `meta.?learner|alzheimer|memory.?carry|des288|fig.?5 neuron` — zero hits. The contracts here are **markdown preregistrations**, sealed by SHA-256 into the artifacts.

| contract | sha256 | status field |
|---|---|---|
| `docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md` | `a24b164d095fe2237710b0b86acfe4aabc10666091a2451fc5586d4df71a512a` | *"Escrito y commiteado ANTES de correr."* — no explicit status line |
| `docs/PREREGISTRO_META_APRENDIZ_V2_2026-08-01.md` | `91614d392a727af4ea7c5f995869e53ed7ce3593e6c9b53949092055a600293b` | *"**Estado:** escrito antes de los reruns v2. Este documento sustituye la lectura operativa del runner v1"* |
| `docs/PREREGISTRO_GARRIDO_Q2_DES288_V1_2026-08-01.md` | `69c8f56be27f05675d28c4ca15b14068158ddccd7a12e6195ed56b9d5f6d33ce` | **`READY_NOT_STARTED_H3_BLOCKING`** |
| `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md` | `576d02b5de7609eb2188f22fcaaffbb5b99a8206ba86c02d903cac7d7fc999d3` | *"Escrito y commiteado ANTES de correr. Instruido por el PI"* |
| `docs/PREREGISTRO_H1_H3_V2_2026-08-01.md` | — | reuses the sealed artifact, *"**sin simulación nueva**"* |

**Primary estimands, quoted:**
- v1 prereg: *"¿cuántas corridas de simulación necesita cada estrategia para encontrar la mejor configuración en un contexto de riesgo **nuevo**, y **cuánto de esa diferencia se debe a recordar**?"* — `(3) contra (4) es el efecto Alzheimer medido`.
- DES-288 prereg: *"El estimando primario es la diferencia pareada **`retained − reset` en eficiencia de búsqueda**, definida como `runs_to_oracle(reset) − runs_to_oracle(retained)`. … La unidad de inferencia es la réplica/semilla, que agrupa sus seis contextos de riesgo; el intervalo se obtiene por bootstrap de esos bloques, no tratando cada contexto como independiente."*
- H3′ prereg: `LCB95 > 0` on the paired-by-replicate difference in the **variance of search cost across the six contexts**. Powered to `n = 120` (`n ≈ 90` required; *"Se fija **`n = 120`**"*).

**Falsifiers (v1, six):** `f1_the_surface_has_a_real_optimum`, `f2_ofat_is_really_one_factor_at_a_time`, `f3_memory_is_the_only_difference`, `f4_random_search_is_uninformed`, `f5_no_context_leakage` (later renamed `f5_the_search_cannot_read_an_unrun_configuration`), `f6_seeds_are_virgin` (later `f6_seed_custody`).

**Falsifiers (DES-288, nine):** `f1_surface_has_real_variation`, `f2_ofat_moves_one_factor`, `f3_memory_reset_share_contract`, `f4_zero_budget_is_identical`, `f5_random_does_not_read_outcomes_before_draw`, `f6_drivers_are_post_episode_only`, `f7_endpoint_key_recomputes_independently`, `f8_service_mass_and_claimant_boundary`, `f9_confirmation_seeds_are_virgin`.

**Falsifiers (H3′ merge, four):** `f_merge_seeds_are_disjoint`, `f_merge_contexts_and_budget_match`, `f_merge_source_is_identical`, `f_merge_reaches_the_contracted_n`.

**Authorization gates — these are hard blockers:**
- DES-288 prereg §Reglas: *"Ningún resultado de este contrato autoriza MLP/PPO: esa decisión sigue bloqueada por el gate de headroom E1 y por el contrato neural separado."*
- v2 prereg §Regla de promoción: *"No autoriza MLP, PPO ni PPO recurrente. … ese gate sigue en HOLD."*
- `research/seed_custody_registry.json`: `"status": "BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED"`, `"scientific_execution_authorized": false`, `"new_seed_opening": false`.
- `docs/GARRIDO_WRAP_CURRENT_STATE_2026-08-01.md`: global state `HOLD_WRAP_BEHAVIORAL_FIDELITY` / `DEVELOPMENT_ONLY`; E1 at `HOLD_E1_PLACEBO_NOT_OPENED`, `training_authorized=false`.

## 3. Results directories, claim_status, headline numbers

All the 288-surface runs share `budget=24`, `n_configurations=288`, `metric="ret_excel_risk_conditional"`, contexts `["R1r","R2r","R1r+R2r","R1r|esc","R2r|esc","R1r+R2r|esc"]`.

| artifact | claim_status | reps / seeds | memory | reset | ofat | random | Alzheimer (reset−mem) | falsifiers |
|---|---|---|---|---|---|---|---|---|
| `results/garrido_meta_learner/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | 12, `5300001–12` | **7.236111** | **13.541667** | **12.416667** | **19.541667** | `+6.3056 [+5.1806, +7.4861]` (key `alzheimer_effect_runs_saved_by_memory`) | all 6 pass — **but `f5_no_context_leakage` was hardcoded `passed: True`** |
| `results/garrido_meta_learner_v2/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | 12, `5300001–12` | 6.986111 | 14.888889 | 12.416667 | 19.541667 | `+7.9028 [+6.8750, +8.9306]` | all 6 pass (real `f5`) |
| `results/garrido_meta_learner_v2_vps_crosscheck/result.json` | same | 12, same | identical to v2 to the last decimal | | | | identical | all 6 pass |
| `results/garrido_meta_learner_smoke_v2/result.json` | **`NEURON_DOES_NOT_BEAT_THE_NULL`** | 1 rep, budget **4**, seed `5910001` | 3.3333 | 3.3333 | 4.3333 | 3.3333 | absent | all 6 pass (degenerate smoke) |
| `results/garrido_meta_learner_h3power_local/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | **90**, `6000001–6000090` | 7.511111 | 14.781481 | 12.550000 | 19.698148 | not sealed in this file | all 6 pass; sealed against the **wrong contract** (`PREREGISTRO_META_APRENDIZ`, `a24b164d…`) |
| `results/garrido_meta_learner_h3power_vps/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | **30**, `6000091–6000120` | 7.694444 | 15.305556 | 12.616667 | 19.383333 | not sealed | same wrong-contract defect |
| `results/garrido_meta_learner_h3power_h3_contract_local_v2/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE`, `audit_status="BEHAVIORAL_REPRODUCIBILITY_FOR_H3_ESTIMAND"`, `replay_of="garrido_h3_local"` | 90 | 7.511111 | 14.781481 | 12.550000 | 19.698148 | **`+7.2704 [+6.7519, +7.7760]`** | 5 pass, `f6_seed_custody` = NOT_APPLICABLE |
| `results/garrido_meta_learner_h3power_h3_contract_vps_v2/result.json` | same, `replay_of="garrido_h3_vps"` | 30 | 7.694444 | 15.305556 | 12.616667 | 19.383333 | **`+7.6111 [+6.6110, +8.6556]`** | 5 pass, `f6` NOT_APPLICABLE |
| `results/garrido_meta_learner_h3power_vps_local_replay/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE`, schema `garrido_h3_source_audit_v1` | 30 | identical to vps | | | | — | 5 pass, `f6` NA |
| `results/garrido_q2_des288_reconciled_v2/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE`, `audit_status="RUNTIME_F3_F4_RECONCILIATION_NOT_A_NEW_CONFIRMATION"`, `replay_of="garrido_q2_des288"` | 12, `5300001–12` | 6.986111 | 14.888889 | 12.416667 | 19.541667 | **`+7.9028 [+6.8750, +8.9306]`** | 5 pass, `f6` NA |
| `results/garrido_meta_learner_thesis90_v2/result.json` | **`SURFACE_REPLAY_MEMORY_EFFECT`** (ledger label `SURFACE_REPLAY_ONLY`) | 256 **replays**, budget 10, 90 configs, 9 contexts | 1.930122 | 2.091146 | `thesis_order` 2.777778 | 2.538628 | `+0.16102 [+0.14280, +0.17970]`, `"inference": "algorithmic_replays_not_DES_replicates"` | all 6 pass |
| `results/garrido_h3_merge_adjudication/result.json` | **`H3_PRIME_SUSTAINED_AT_N120`** | 120 merged | variance 44.264167 | 53.578611 | 60.484444 | 67.503333 | H3′ mem−reset `+9.3144 [+2.3491, +16.3474]`; mem−OFAT `+16.2203 [+9.6107, +22.7405]` | all 4 merge falsifiers pass |

**`results/garrido_q2_des288_v1/` — the preregistered confirmation never ran.** The directory contains only three smoke files:
- `smoke_result_after_fix.json`: `HALTED_FALSIFIER_FAILED`, `n_configurations=17`, `f1` and `f8` FAIL.
- `smoke_fullsurface_result.json`: `HALTED_FALSIFIER_FAILED`, 288 evals, 1 context, `f1` and `f8` FAIL.
- `smoke_fullsurface_result_v2.json`: **`Q2_EFFECT_NOT_ESTABLISHED`**, all 9 falsifiers PASS, but `contexts=["R1r"]`, `repeats=1`, `budget=4`, and every comparison is `{mean:0.0, lcb95:0.0, ucb95:0.0, n_groups:1}`. `decision = {"des_288_complete": false, "mlp_ppo_authorized": false, "primary_lcb95_positive": false, "service_guardrail_passed": true}`.

**A concrete custody trap for your plan:** the DES-288 prereg reserves `7_100_001 … 7_100_012` as the virgin confirmation block. That block is **not registered** in `research/seed_custody_registry.json`, and seed `7100001` has already been consumed by all three smoke artifacts (each of which nonetheless sealed `"virgin_seed_block": true`).

**Git custody:** `results/garrido_meta_learner_h3power_local/`, `.../h3power_vps/`, `.../smoke_v2/`, `results/garrido_q2_des288_v1/` are all **untracked** (`git status` shows `??`). `results/garrido_h3_merge_adjudication/result.json` and `results/garrido_q2_des288_reconciled_v2/result.json` are tracked.

## 4. The "7.24 / 13.54 / 12.42 / +0.00 → +10.00" headline — exact provenance and status

**Recorded in exactly two places:**

1. **Doc:** `<HOME>/Projects/research/scres-ia/docs/RESULTADO_META_APRENDIZ_2026-07-31.md`, lines 15–18 and 35:
   - `| **neurona con memoria** (\`ρ\` cruza contextos) | **7,24** | **0,000090** |`
   - `| OFAT — **el diseño de su propia tesis** | 12,42 | 0,000297 |`
   - `| neurona **reiniciada** en cada contexto | 13,54 | 0,000404 |`
   - `| búsqueda aleatoria — el nulo | 19,54 | 0,000821 |`
   - `| ventaja de la memoria | **+0,00** | +2,67 | +8,08 | +8,75 | +8,33 | **+10,00** |`
2. **Result json:** `<HOME>/Projects/research/scres-ia/results/garrido_meta_learner/result.json`, `self_sha256 = "230a0074a10f12ee…"`, `runs_to_within_1pct = {"neuron_memory": 7.236111111111112, "neuron_reset": 13.541666666666666, "ofat": 12.416666666666666, "random": 19.541666666666668}`.

**The six contexts** (`CONTEXTS` dict, `run_meta_learner_over_configs_v1.py:63`) are traversed **in fixed order** as "successive experiences":

| # | key | risks enabled | frequency multiplier |
|---|---|---|---|
| 1 | `R1r` | R11,R12,R13,R14 | none |
| 2 | `R2r` | R21,R22,R23,R24 | none |
| 3 | `R1r+R2r` | all eight | none |
| 4 | `R1r\|esc` | R1r | ×3.0 on each |
| 5 | `R2r\|esc` | R2r | ×3.0 on each |
| 6 | `R1r+R2r\|esc` | all eight | ×3.0 on each |

**Comparators:** `ofat` (thesis one-factor-at-a-time from `DEFAULT = {buffer_hours:0.0, shifts:1, op9_rop:24.0, op12_rop:24.0}`), `random` (uninformed null), `neuron_reset` (ablation of memory), `neuron_memory`.

**Seeds / replications:** 12 replicates, seeds `5_300_001 … 5_300_012`. CRN: one seed per (context, repeat), **the same surface for every strategy**. RNG stream per replicate is `np.random.default_rng(90_000 + r)`, matched across arms. Budget 24 runs per context. Total surface = 288 × 6 × 12 = **20,736 DES episodes** (the doc's *"20.736 episodios"*), horizon 52 weeks.

I recomputed the per-context advantage from `per_context` directly — it reproduces the doc exactly:

```
ctx             R1r     R2r  R1r+R2r  R1r|esc  R2r|esc  R1r+R2r|esc
neuron_memory  9.9167 10.6667  4.4167   4.1667   9.8333    4.4167
neuron_reset   9.9167 13.3333 12.5000  12.9167  18.1667   14.4167
ADV(reset−mem)   0.00    2.67    8.08     8.75     8.33     10.00
```

**Stated limitations — this is the load-bearing part:**

- **The numbers are RETIRED.** `<HOME>/Projects/research/scres-ia/docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md` documents that `run_meta_learner_over_configs_v1.py` ranked unrun candidates with `preds = [neuron.predict(features(CONFIGS[i], table[i][1])) for i in unseen]` where `table[i][1]` is the **driver vector of an already-simulated episode** — i.e. reading the answer. And `"f5_no_context_leakage": {"passed": True, ...}` was **hardcoded**: *"Un falsador que no puede fallar no es un falsador."*
  - `memoria vs OFAT = +5,18` → **RETIRADO**. `memoria vs aleatorio = +12,31` → **RETIRADO**. The `+6,31` Alzheimer number → *"el estimando sobrevive … pero **el número se retira**"*. The H2 curve → *"forma retirada como medida; el cero del primer contexto sigue siendo una comprobación estructural válida"*.
- `docs/GARRIDO_WRAP_CURRENT_STATE_2026-08-01.md`: *"Los contrastes del meta-aprendiz v1 (`+6.31`, `+5.18`, `+12.31`) y la curva H2 antigua **no son evidencia**."*
- Claim ledger `docs/GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md`: row `H2` → artifact "meta-aprendiz v1 retirado", status **`RETIRED_LEAKAGE`**, prohibited: *"usar la curva antigua"*. Row `H4` → **`PENDING_CONFIRMATION`**, prohibited: *"usar el `+6.31` antiguo"*.
- `docs/GARRIDO_WRAP_RECONCILIATION_2026-08-01.md`: `results/garrido_meta_learner/result.json` → **"retired for contrasts"**, *"provenance only; its search used the leaked driver ranking"*.

**The post-fix replacement curve** (from `results/garrido_meta_learner_v2/result.json`, 12 reps, same seeds, no leak) is:

```
ctx             R1r    R2r  R1r+R2r  R1r|esc  R2r|esc  R1r+R2r|esc
ADV(reset−mem)  0.00   6.08     9.00    10.33    10.92       11.08
```
and at n=90 (`h3power_h3_contract_local_v2`): `0.00, 6.28, 7.21, 8.87, 10.67, 10.60`. The structural zero in context 1 survives everywhere.

**Additional scope limits from `docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md`:**
- §4: *"Escribí que el efecto Alzheimer «cubre H4 (Path Dependency)». **Sobreafirmado.**"* → defensible wording is *"**Apoyo estrecho a H4**"*.
- §5: *"H2 se reporta como **análisis descriptivo exploratorio**"* — no preregistered estimand exists for it.
- §3: H3′ (search-cost variance) ≠ draft H3 (performance variance). *"En el manuscrito debe escribirse **H3′ sostenida**, nunca «H3 probada»."*
- `docs/RESULTADO_WRAP288_RECONCILIACION_F3_F4_2026-08-02.md` §5: *"El efecto Alzheimer sigue sin entrar al manuscrito por esta vía. … el citable con potencia es el de H3′ a `n = 120`."*
- Cross-cutting caveat from `docs/RESULTADO_H1_H3_2026-08-01.md` §1: memory and OFAT **deploy the identical configuration** (`buffer 1344 h · turnos 2 · op9_rop 12 · op12_rop 12`); the advantage is *"en **cuánto tarda en encontrar** el óptimo, no en **qué** encuentra"*, because `H_regime = 0` — the optimum does not move.

## 5. Search comparators over the configuration grid: what exists, what doesn't

**Exists, in the lane:**

| comparator | where | notes |
|---|---|---|
| OFAT (thesis design) | `run_meta_learner_over_configs_v1.py:204-217`, `run_garrido_q2_des288_v1.py:~295-318` | lazy from incumbent; `f2` enforces ≤1 coordinate change |
| `thesis_order` | `run_meta_learner_thesis90_v1.py` (`idx = step % len(cf_order)`) | deliberately *not* rebranded OFAT |
| Random search (uninformed null) | all four runners | `f4`/`f5` shadow-value test proves it draws before reading |
| Greedy linear/logistic surrogate, memory | `Fig5Neuron` / `VectorLinearLearner` | argmax over unseen, 3-run cold start |
| Greedy linear surrogate, reset | same code, `rho` zeroed per context | the ablation |
| `no_update` | `run_garrido_q2_des288_v1.py` only | learner exists but never observes; selects `min(available)` |
| Refit-from-scratch surrogate with MLP / KAN heads | `run_garrido_wrap_closed_loop.py:150` (`kind ∈ linear/backprop/kan`) | gated off by Q1 `NO_GO`; only ever run as `linear` |
| Full-enumeration oracle | implicit — the whole surface is evaluated, `best = max(values)` | the regret reference |

**Does NOT exist anywhere in the repo for this grid:**
- **Bayesian optimisation / GP-EI over the configuration grid.** A GP-EI implementation exists — `supply_chain/gsa.py:68` `gp_locate` ("GaussianProcess + expected-improvement active learning to find argmax of f", `sklearn.gaussian_process.GaussianProcessRegressor`, Matern) — but it is used by the GSA/war-stress lane, **never wired into any meta-learner runner**. This is the single cheapest missing comparator.
- **Branch-and-bound.** The only hits are negative assertions: `scripts/build_war_stress_exact_reduction_certificate.py:63` `"branch_and_bound_used": False`, verified at `scripts/verify_war_stress_gsa_execution_preflight.py:84` and `tests/test_war_stress_executor.py:53`.
- **Best-first / A\*** — no implementation.
- **Simulated annealing, hill-climbing, genetic algorithms, Nelder-Mead, CMA-ES** — no implementation (`CMA-ES` appears only as a suggestion string in `scripts/run_per_op_explore.py:256`).
- **Bandits** (UCB1, Thompson sampling, ε-greedy over configs) — none. The neuron is pure greedy.
- **Full-grid / exhaustive sweep as a *comparator arm*** — the grid is enumerated to build the surface, but no arm is scored as "grid search under a budget".
- **Any exploration bonus or acquisition function** at all.

## 6. The grids: definition, axes, levels, and cached evaluations

### The 288-config extension
Defined identically in two places — `scripts/run_meta_learner_over_configs_v1.py:52-61` and `scripts/run_garrido_q2_des288_v1.py:44-55`:

```python
FACTORS = {
    "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),   # 6 — Garrido's six
    "shifts": (1, 2, 3),                                          # 3 — Garrido's three
    "op9_rop": (12.0, 24.0, 36.0, 48.0),                          # 4
    "op12_rop": (12.0, 24.0, 36.0, 48.0),                         # 4
}
DEFAULT = {"buffer_hours": 0.0, "shifts": 1, "op9_rop": 24.0, "op12_rop": 24.0}
```
6×3×4×4 = **288**, full Cartesian product via `itertools.product`. The prereg justifies the last two: *"más las dos que la campaña de sensibilidad identificó como las únicas con autoridad"*.

**There is NO cached/precomputed evaluation of the 288 grid.** Every runner rebuilds it from scratch:
```python
for ctx in CONTEXTS:
    for seed in seeds:
        surface[(ctx, seed)] = [evaluate(cfg, ctx, seed, horizon) for cfg in CONFIGS]
```
I searched for surface caches (`find … -iname "*surface*"`, and grepped the runners for `np.save|pickle|joblib|cache_path`) — nothing. The artifacts persist only `surface_sha256`, `visited_sequence`, `regret_curve`, `best`, `chosen_config`, `chosen_value`. You can recover the *running max* along a visited path, not per-config values. **Meta-search over this grid is therefore NOT cheap today.**

Cost, quoted from `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md`: *"288 configuraciones × 6 contextos = **1.728 episodios por réplica**, ~126 s en el M1 Pro y ~504 s en el VPS"*. Measured wall-clocks: 12 reps = 1515.9 s local / 4491.7 s VPS; 90 reps local and 30 reps VPS ≈ 3.4 h in parallel. **A surface cache is the highest-leverage engineering change in this lane.**

### The 90-config Garrido reproduction
- Design as data: `<HOME>/Projects/research/scres-ia/supply_chain/garrido_thesis_design.py` — `build_design() -> dict[int, Configuration]`, transcribed from WRAP_Theses_Garrido_Rios_2017.pdf Ch. 6, Tables 6.11–6.23. `FACTOR_CODING` (line 49) gives the `-`/`+` risk levels for R11–R14, R21–R24, R3. Structure: Scenario I = Cf1–Cf30 (risk frequency only, buffers 0, S=1), Scenario II = Cf31–Cf60 (buffers × risk, S=1), Scenario III = Cf61–Cf90 (shifts × risk, buffers 0). Blocks map to `H1a…H3c`, 10 configs each. Seeds repeat with stride 30 (*"the seed used for ReT(Cf7) is the same for ReT(Cf37) and ReT(Cf67)"*).
- **This grid IS fully cached.** `<HOME>/Projects/research/scres-ia/results/garrido_drivers_per_configuration/result.json` (`self_sha256 = 491694175a3975a70d3f6a9d7af90f3cc5b97849a5d348786bd44c9c3455a392`, `claim_status = "DEVELOPMENT_DRIVER_TABLE"`), 90 rows, each with `ret_excel`, the four driver shares/means/contributions, `rho: {buffer_hours, shifts}`, `family`, `pattern`, `seed`, `seed_is_thesis_seed`. Flat CSV at `results/garrido_drivers_per_configuration/drivers.csv` (91 lines incl. header, 24 columns).
- Upstream source: `results/garrido_reproduction/reproduction.json` (`DEVELOPMENT_REPRODUCTION_NO_CLAIM`, 90 rows, 20 validatable against `Raw_data1+Re.xlsx` / `Raw_data2+Re.xlsx`).
- **Known gap sealed in the table:** `"known_gap": "Re(APj) is identically zero here: the shipped fulfilment constant is 54 h against LT = 48, so the autotomy branch is structurally unreachable. Close it with the freight-wave arm before fitting Fig. 5"`.
- **Why this cache is scientifically limited:** one seed per Cf, so `run_meta_learner_thesis90_v1.py` repetitions are *"replays algorítmicos sobre una misma tabla, no réplicas físicas independientes"*, and the endpoint state is `HOLD_METRIC_PROVISIONAL`. Result label is `SURFACE_REPLAY_ONLY`.
- Also derived from it: `results/garrido_fig5_surrogate/result.json` and `results/garrido_wrap_q1/result.json` (`DEVELOPMENT_FIG5_SURROGATE`), which establish `task_A_figure_as_drawn.status = "IDENTITY_NOT_A_LEARNING_TASK"` (R²=1.0, `max_abs_identity_error = 3.22e-15`) and the learnable Task B: KAN 0.9913 / backprop 0.9863 / linear 0.9697 on regression; on the activation question KAN 0.7711 / backprop 0.7178 / **linear 0.7111**, verdict `B2_activation: {backprop: false, kan: false}`.

### Missing artifacts referenced as "next"
- `results/garrido_q2_des288_v1/result.json` — the preregistered confirmation. **Does not exist.**
- `results/garrido_meta_learner_v2_corrected/result.json` — listed as "pending" in `docs/GARRIDO_WRAP_RECONCILIATION_2026-08-01.md`. **Does not exist.**
