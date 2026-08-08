# Explore — In the repo /Users/thom/Projects/research/scres-ia, investigate three things. Read-only, r

- **agent_id**: `a6c7c59bd617d2ffc`
- **session**: `1fab0490-70ae-4d2e-a373-23cec6b3ba5f`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T02:11:09.872Z`
- **effort**: `medium` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/1fab0490-70ae-4d2e-a373-23cec6b3ba5f/subagents/agent-a6c7c59bd617d2ffc.jsonl`
- **turns**: 69 · **assistant messages**: 4

## Task given

```text
In the repo /Users/thom/Projects/research/scres-ia, investigate three things. Read-only, report exact numbers.

1. WITHIN-FAMILY PAIRED CONTRASTS for RQ1. Open `results/search_ladder_v5/result.json`. It has `mean_auc_regret` (a dict of arm -> AUC) and `per_arm` (each arm has an `auc` list, one entry per seed/context replicate). Report:
   (a) the full `mean_auc_regret` dict;
   (b) the `memory_arms` list;
   (c) for each of the six stateful arms, its memoryless twin's name (e.g. ucb1_transfer/ucb1, neuron_memory/neuron_reset, ofat_transfer/ofat, gp_ei_transfer/gp_ei, thompson_transfer/thompson, lookahead_kg_transfer/lookahead_kg) and the LENGTH of each arm's `auc` array;
   (d) whether the per_arm auc arrays are PAIRED — i.e. same length across arms, and whether there is any per-replicate identifier (seed, context) stored alongside, so a paired contrast could be computed. Report the exact keys inside `per_arm[<arm>]`.
   Also report whether any EXISTING artifact already contains within-family retention contrasts with confidence intervals — search `results/` for files containing "vs_neuron_memory", "memory_vs_reset", "retention", or "delta_memory". Report paths and the contrast values if found.

2. GARRIDO'S UNADDRESSED ASKS. Search the repo for any existing work on (a) randomised R2 risks — look for scripts/results/docs matching `*r2_random*`, `*risk_random*`, `randomized_r2`, or docs mentioning "R2 aleatoriz"; and (b) AIC / Ramsey RESET / linearity diagnostics — search for `ramsey`, `RESET`, `aic`, `akaike`, `linearity` across scripts/, docs/, results/. Report what exists, if anything, with paths and status.

3. KAN LATENT ENDPOINT. Open `results/dmlpa_kan_latent/result.json`. Report: `claim_status`, `scope`, the exact endpoint/metric name it uses, its units and orientation (is higher better or lower better?), the absolute mean values of BOTH arms (not just the difference), the number of seeds, and the falsifier names with pass/fail. The audits complain that "kan - mlp = -0.86225" is uninterpretable without these.

Report each numbered item separately with exact values. Do not modify anything.
```

## Final answer

Findings below. Nothing was modified.

---

# 1. Within-family paired contrasts (RQ1)

Source: `/Users/thom/Projects/research/scres-ia/results/search_ladder_v5/result.json`
Generator: `/Users/thom/Projects/research/scres-ia/scripts/run_search_comparator_ladder_v5.py`

### (a) `mean_auc_regret` (metric `auc_regret_norm`, **lower is better**; budget 24)

| arm | mean AUC regret |
|---|---|
| oracle | 0.0 |
| ucb1_transfer | 0.04502300698358852 |
| neuron_memory | 0.05203274760040456 |
| ofat_transfer | 0.06274306709376977 |
| lookahead_kg_transfer | 0.08018189890560519 |
| gp_ei_transfer | 0.08390483487553962 |
| thompson_transfer | 0.08907820238728746 |
| ucb1 | 0.09655146536034616 |
| ofat | 0.10024197702481123 |
| gp_ei | 0.10661346477746912 |
| thompson | 0.10893184647329195 |
| lhs_local | 0.10948882277743688 |
| neuron_reset | 0.11273605766343564 |
| lookahead_kg | 0.11479477721224257 |
| random | 0.1397945613431999 |
| annealing | 0.1742042148688457 |

### (b) `memory_arms`
`["gp_ei_transfer", "ucb1_transfer", "ofat_transfer", "neuron_memory", "lookahead_kg_transfer", "thompson_transfer"]`

### (c) The six families — all `auc` arrays are length **12**

| stateful arm | len(auc) | memoryless twin | len(auc) |
|---|---|---|---|
| ucb1_transfer | 12 | ucb1 | 12 |
| neuron_memory | 12 | neuron_reset | 12 |
| ofat_transfer | 12 | ofat | 12 |
| gp_ei_transfer | 12 | gp_ei | 12 |
| thompson_transfer | 12 | thompson | 12 |
| lookahead_kg_transfer | 12 | lookahead_kg | 12 |

(All 16 arms including `oracle`, `random`, `lhs_local`, `annealing` are also length 12.)

### (d) Are they paired? **Yes — positionally, by seed.**

Exact keys inside `per_arm[<arm>]`: **`["auc", "budget_used", "final"]`** — and nothing else. There is **no** per-entry seed or context field.

The pairing is nonetheless well-defined and recoverable from the generator (`run_search_comparator_ladder_v5.py` lines 182–219):

- Outer loop `for r, seed in enumerate(seeds)`; inner loop `for ctx in contexts`.
- `per_arm[name]["auc"].append(float(np.mean(aucs)))` is appended **once per seed**, after averaging over the six contexts.
- So `auc[i]` ↔ `seeds[i]`, with top-level `seeds = [5300001 … 5300012]` (n=12) and `contexts = ["R1r", "R2r", "R1r+R2r", "R1r|esc", "R2r|esc", "R1r+R2r|esc"]` (n=6) already collapsed by the mean.
- `final` is likewise length 12 (seed-indexed). `budget_used` is length **72** = 12 seeds × 6 contexts, seed-major/context-minor — this is the only place context resolution survives.

Consequence: **within-family paired contrasts are computable exactly as-is** (element-wise on the 12 seed slots), but **only at seed resolution** — a context-level paired contrast is not recoverable from this artifact, because context was averaged before storage.

The file's own precedent confirms this is the intended operation: `vs_neuron_memory` is built at line 233–234 as `boot(per_arm[n]["auc"] - per_arm["neuron_memory"]["auc"])`, i.e. an element-wise paired difference with a seed bootstrap, n=12.

### Existing artifacts with retention contrasts + CIs

**Same file, top-level `vs_neuron_memory`** (paired, n=12, bootstrap over seeds; positive = that arm is worse than neuron_memory). Only one of the fourteen entries is a within-family contrast — `neuron_reset`:

| arm vs neuron_memory | mean | lcb95 | ucb95 | n |
|---|---|---|---|---|
| **neuron_reset** | **0.06070331006303109** | **0.04590626893881853** | **0.07997188775380663** | 12 |
| ucb1_transfer | −0.007009740616816028 | −0.024441318883473704 | 0.014076439977282206 | 12 |
| ofat_transfer | 0.01071031949336522 | 3.564844184833131e-05 | 0.02171366105332719 | 12 |
| lookahead_kg_transfer | 0.028149151305200626 | 0.012845676382162115 | 0.04191154996295932 | 12 |
| gp_ei_transfer | 0.03187208727513507 | 0.019229943011605136 | 0.042145755333430956 | 12 |
| thompson_transfer | 0.037045454786882905 | 0.02038444229207778 | 0.05185218017696919 | 12 |
| ucb1 | 0.04451871775994159 | 0.03518653818669935 | 0.05447320053251466 | 12 |
| ofat | 0.04820922942440667 | 0.03292031155918523 | 0.0634593535389054 | 12 |
| gp_ei | 0.05458071717706455 | 0.035851158494365966 | 0.07257890214069028 | 12 |
| thompson | 0.05689909887288736 | 0.04068412119715085 | 0.07242185714216451 | 12 |
| lhs_local | 0.05745607517703233 | 0.03514962730858726 | 0.08190136753893594 | 12 |
| lookahead_kg | 0.06276202961183801 | 0.043346493918471785 | 0.08081807702258073 | 12 |
| random | 0.08776181374279535 | 0.07033476374620021 | 0.10375751133298124 | 12 |
| annealing | 0.12217146726844114 | 0.09547940053705108 | 0.14881565146767145 | 12 |

**So: the only within-family retention contrast that exists anywhere as a sealed artifact is `neuron_memory` vs `neuron_reset`.** The other five families (ucb1, ofat, gp_ei, thompson, lookahead_kg) have **no** stored within-family contrast — they are only compared to `neuron_memory`, which is a cross-family baseline.

That same neuron number is the canonical claim, in `/Users/thom/Projects/research/scres-ia/docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md` (line ~50): `AUC +0,06070 [+0,04556, +0,08020]` — mean identical, CI differs only by bootstrap draw. It also appears in `results/search_ladder_v4/result.json` (mean 0.06070331006303109, [0.04573646060596155, 0.07986029555425553]) and `results/search_ladder_v3/result.json` ([0.04590124512602244, 0.07980176463078424]).

Other `memory_vs_reset` hits, all on the **ReT lane, not the AUC lane** (different endpoint, do not mix):

- `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_v2/result.json` → `H3.contrasts.memory_vs_reset` = mean **7.858333333333331**, [−14.717013888888888, 28.305624999999985] (claim_status `H1_SUPPORTED__H3_NOT_SUPPORTED`)
- `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_v2_1/result.json` → mean **7.858333333333331**, [−14.842291666666663, 28.80916666666665]
- `/Users/thom/Projects/research/scres-ia/results/garrido_h3_merge_adjudication/result.json` → `h3_prime.memory_vs_reset` = mean **9.314444444444444**, [2.3491250000000012, 16.34740972222222], n=120 (claim_status `H3_PRIME_SUSTAINED_AT_N120`)
- `results/garrido_normaliser_audit{,_v2,_v3}/result.json` contain further `memory_vs_reset` blocks under normaliser variants (4 each).
- `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md:223` defines `Delta_memory = ReT_retained − ReT_reset` as an H4 protocol estimand — **specified only, never executed**.

**Derived by me from the paired arrays (NOT an existing artifact — recompute in-repo before citing).** Percentile bootstrap, 20 000 resamples, `random.seed(0)`, sign convention memoryless − stateful (positive = memory helps):

| family | mean Δ | CI95 | seed wins |
|---|---|---|---|
| neuron_memory vs neuron_reset | +0.060703 | [+0.045963, +0.079875] | 12/12 |
| ucb1_transfer vs ucb1 | +0.051528 | [+0.035897, +0.065986] | 11/12 |
| ofat_transfer vs ofat | +0.037499 | [+0.029202, +0.046377] | 12/12 |
| lookahead_kg_transfer vs lookahead_kg | +0.034613 | [+0.026180, +0.043350] | 12/12 |
| gp_ei_transfer vs gp_ei | +0.022709 | [+0.012726, +0.034006] | 12/12 |
| thompson_transfer vs thompson | +0.019854 | [+0.010019, +0.029307] | 10/12 |

All six exclude zero. Scope caveat from the artifact: `scope = "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION"`, `claim_status = "THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH"`, and falsifier `f7_no_fresh_seeds` records these 12 seeds as a **deliberate re-execution** of registry block `garrido_q2_des288` (not virgin).

---

# 2. Garrido's unaddressed asks

### (a) Randomised R2 risks — **nothing exists.**

Zero hits across the entire repo:
- No files matching `*r2_random*`, `*risk_random*`, `*randomized_r2*`, `*r2_alea*`, `*aleator*`.
- No `randomi[sz]ed_r2`, `randomized R2`, `R2 aleatoriz`, `riesgos aleatoriz`, or `R2 randomi` in any `.py`, `.md`, or `.json`.

The only R2-adjacent material is the existing **fixed/deterministic** R2 treatment: `docs/PREREGISTRO_PROPAGACION_RIESGOS_2026-07-30.md` (R2r intensity 1.4, 2169 events), `docs/RESULTADO_MEZCLA_RIESGOS_2026-07-31.md` (R2r = 0.324806), `docs/R2_AUDIT_DECOMPOSITION_2026-06-29.md` (endogenous R2 ReT 0.483 vs Excel target 0.202, 2.39×), and the `_risk_R21/R22/R23/R24` processes in `legacy/supply_chain_v3.py:614-667` and `legacy/supply_chain_v4.py:602-655`. None of these randomise R2.

### (b) AIC / Ramsey RESET / linearity — **nothing exists.**

`ramsey` and `akaike` appear in **exactly one file in the whole repository**, and only as a mention of the ask itself, not as work:

`/Users/thom/Projects/research/scres-ia/docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_3.md:115`

> `| bloqueantes de escritura | ninguno: ni KAN-latente, ni AIC/Ramsey, ni R2 aleatorio, ni el motor estacional |`

i.e. the claim freeze explicitly declares AIC/Ramsey, randomised R2, KAN-latent and the seasonal engine to be **non-blockers for writing** — which is precisely the mechanism by which they remain unaddressed. There is no `reset_test`, no AIC/BIC computation, no `statsmodels` RESET call anywhere.

The nearest existing linearity work is a **different construct** — a curvature share, not a specification test:
- `/Users/thom/Projects/research/scres-ia/scripts/run_g1_buffer_price_cobb_douglas.py:348` — `"nonlinearity_one_minus_linear_r2": {"cobb_douglas": curv_cd, "ret_excel": curv_ret}` (this is 1 − R², not an F-test)
- Results in `results/headroom/g1_buffer_price/result.json:335` and `results/headroom/g1_ablation_ci_zero/result.json:341`
- `scripts/run_g2_autotomy_threshold.py:6` and `scripts/run_buffer_profile_prediction_premium.py:203` discuss non-linearity narratively
- `docs/PROGRAM_I_HEADROOM_*_2026-07-12.md` — Morris elementary effects σ as an interaction/non-linearity proxy

**Status: both asks are entirely unstarted.** No script, no result directory, no preregistration.

---

# 3. KAN latent endpoint

Source: `/Users/thom/Projects/research/scres-ia/results/dmlpa_kan_latent/result.json`
Generator: `/Users/thom/Projects/research/scres-ia/scripts/run_dmlpa_kan_latent_v1.py`
Preregistration: `/Users/thom/Projects/research/scres-ia/docs/PREREGISTRO_DMLPA_KAN_LATENT_2026-08-07.md`

- **`claim_status`**: `KAN_LATENT_HURTS`
- **`scope`**: `DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN`
- **Endpoint / metric name**: `ret_mean_track_b_v1` (field `endpoint`). Mean Garrido order-level ReT (resilience) over 24 eval episodes per seed, on the Track B `track_b_v1` 8D action contract.
- **Units and orientation**: dimensionless resilience index (Garrido ReT, built from the Re^max=1.0 / Re=0.5 / Re^min=0.0 levels in `supply_chain/config.py:861-863`), summed/aggregated over the episode — hence the ~98 scale. **Higher is better.** Confirmed by `docs/environment_spec.md:99` ("H1: The retained online learner produces **higher** held-out order-level Garrido ReT…"). Note this is the *opposite* orientation to the `auc_regret_norm` in item 1, where lower is better. Caveat: neither the result file nor the preregistration states a unit string explicitly — orientation is inferred from the spec and is consistent with `claim_status = KAN_LATENT_HURTS` given a negative `kan_minus_mlp`.

### Absolute arm means (the number the audits say is missing)

| arm | mean ReT | params | hidden_dim | use_kan |
|---|---|---|---|---|
| **dmlpa_mlp** | **98.44314398434753** | 200052 | 152 | false |
| **dmlpa_kan** | **97.5808931153368** | 199082 | 10 | true |

So `−0.86225` is a **−0.876 % relative** shift off a base of ~98.4 — and it is smaller than the within-seed evaluation noise, which runs `ret_sd_within` ≈ **0.762 to 1.060** across the ten rows.

`kan_minus_mlp`: mean **−0.8622508690107338**, lcb95 **−1.605043533470689**, ucb95 **−0.11945820455077864**, n **5**.

### Seeds

**5 seeds**: 9491, 9492, 9493, 9494, 9495 (`seed_block`: start 9491, end 9495, n 5, derived_from `seeds`). Paired by seed; bootstrap over 5 seeds — which is what makes the CI reach almost to zero.

Per-seed rows (ReT mean, within-seed SD):

| seed | dmlpa_mlp | sd | dmlpa_kan | sd | kan − mlp |
|---|---|---|---|---|---|
| 9491 | 98.91837163513163 | 0.7766228365348222 | 98.23892572453418 | 0.8310205871882346 | −0.67944591 |
| 9492 | 97.47920713248368 | 1.059935921704918 | 97.38433545680061 | 0.9142330838401106 | −0.09487168 |
| 9493 | 97.97826418309762 | 0.800749898441005 | 98.11421330270247 | 0.7622763281978748 | **+0.13594912** |
| 9494 | 99.13093262043155 | 0.8097438889870702 | 97.32553990549793 | 0.8761586462977329 | −1.80539271 |
| 9495 | 98.7089443505932 | 1.0351837419393308 | 96.84145118714882 | 0.883917058269791 | −1.86749316 |

**4 of 5 seeds negative, 1 positive.** The point estimate is carried largely by seeds 9494 and 9495.

### Falsifiers — `all_passed: true` (5 of 5)

| falsifier | passed | key evidence |
|---|---|---|
| `f1_parameters_are_matched_within_our_tolerance` | **true** | kan 199 082 params (error 0.00459), mlp 200 052 (error 0.00026), tolerance 0.1 |
| `f2_only_the_latent_rw_differs` | **true** | held fixed: features_dim 84, ff_mult 4, nhead 12, num_layers 2. **`not_covered`: "normalisation order differs from David's version"** |
| `f3_the_two_arms_are_behaviourally_distinct` | **true** | fingerprints differ: kan `[-0.369356, -1.769662, -1.072951, 0.535967, -1.437132, -2.501684]` vs mlp `[-0.241437, -2.219524, -1.059008, 0.76903, -0.90835, -1.902737]` |
| `f4_the_harness_reproduces` | **true** | delta 0.0; replicas [89.23527051364351, 89.23527051364351] |
| `f5_no_new_seeds` | **true** | seeds [9491…9495] |

### Interpretation load-bearing context also in the file

- `result_before_training`: at equal parameter budget the KAN affords `hidden_dim=10` against the MLP's `152` — so this contrasts a 15×-narrower network, and the effect is capacity-confounded in that specific sense even though *parameter count* is matched.
- `not_covered` (top level): "David's version also differs in normalisation order: `pre_norm(latent+pos)` here, `pre_norm(latent)+pos` there. Only the latent_rw is tested."
- Design: total_steps 100 000, n_envs 8, eval_episodes 24, features_dim 84, num_layers 2, nhead 12, ff_mult 4. Elapsed 8850.87 s.
- Contract sha256 `1c23273417a5b0d69dcd00fd35a83a21965ce61778785c598b5597eaa822f235`; self sha256 `f79ca248bb983bd4eb77e05f119eafc837b72d53d291c00c97ee8d4b30699ae9`; reference `results/determinism_repair_control/result.json`.

**Bottom line for the audits:** the honest reportable form is *"mean ReT 97.58 (KAN) vs 98.44 (MLP), higher-is-better; paired difference −0.86 [−1.61, −0.12] over 5 seeds, i.e. −0.88 % relative, an effect smaller than the ~0.8–1.1 within-seed evaluation SD, with 4/5 seeds negative"* — and the scope tag forbids confirmation (`NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN`).
