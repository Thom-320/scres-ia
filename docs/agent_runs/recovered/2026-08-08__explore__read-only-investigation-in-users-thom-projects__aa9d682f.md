# Explore — Read-only investigation in /Users/thom/Projects/research/scres-ia. Four external audits sa

- **agent_id**: `aa9d682fc32c35d0f`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T02:09:19.249Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-aa9d682fc32c35d0f.jsonl`
- **turns**: 57 · **assistant messages**: 2

## Task given

```text
Read-only investigation in /Users/thom/Projects/research/scres-ia. Four external audits say Paper 2's RQ1 must NOT rest on "the top six of fifteen methods are exactly the six that retain state" (a ranking observation) but on PAIRED WITHIN-FAMILY contrasts: for each method family f, Delta_f = AUC(f_reset) - AUC(f_retained), with intervals. I need to know whether the existing sealed artifacts can support that without any new computation.

1. `results/search_ladder_v5/result.json`. Report its FULL structure: what per-arm data exists. Specifically: are there per-seed / per-context AUC values for each arm (something like per_arm[arm]["auc"] being an array over seeds x contexts), or only aggregate means? Quote the actual keys and array shapes. If per-seed values exist, the paired contrasts are computable; say so explicitly.

2. Which arms pair up as retained/reset twins? The audits assert these six families: neuron (neuron_memory vs neuron_reset), ucb1 (ucb1_transfer vs ucb1), ofat (ofat_transfer vs ofat), lookahead_kg (lookahead_kg_transfer vs lookahead_kg), gp_ei (gp_ei_transfer vs gp_ei), thompson (thompson_transfer vs thompson). Verify each twin actually exists in the artifact and report the mean AUC for each of the 12 arms. Compute the six raw mean differences.

3. Does the artifact already contain any pairwise contrast block (e.g. `vs_neuron_memory`, `contrasts`, bootstrap CIs)? List every contrast key present and what it compares. Report whether a retained-vs-reset contrast is ALREADY sealed for any family.

4. Check `results/garrido_normaliser_audit_v3/result.json` — it has memory_vs_reset under prefix (+0.06070 [+0.04556, +0.08020]). Is that the SAME quantity as the neuron family's paired contrast in search_ladder_v5? Same seeds, same endpoint, same normaliser? If yes, one of the six is already sealed with a CI.

5. RQ2 separation: confirm the three experiments are distinct contracts by reporting for each — `results/surrogate_architecture_bakeoff/`, `results/dmlpa_kan_latent/`, `results/garrido_fig5_surrogate/` — the endpoint/primary_metric, the seed block, the scope string, and what is actually being compared. I need to state plainly that dmlpa_kan_latent is a DIFFERENT contract (PPO intra-episode / track_b) from the outer-loop surrogate bakeoff. Report `results/dmlpa_kan_latent/result.json` fully: claim_status, endpoint name, units, orientation (is higher better?), both arm means, per-seed spread, n seeds, falsifiers.

Give concrete numbers and key paths; do not summarise vaguely.
```

## Final answer

All four questions answered from the sealed artifacts. Here are the findings.

---

## 1. `results/search_ladder_v5/result.json` — per-seed AUC exists

**Verdict: the paired within-family contrasts ARE computable from the sealed artifact. No new simulation is needed.**

Top-level keys (30):
`arms, budget, calibration_provenance, claim_status, contexts, contract_path, contract_sha256, created_at, elapsed_seconds, falsifiers, mean_auc_regret, memory_arms, module_manifest, new_arms, per_arm, predecessor, preregistration, primary_metric, ranking_best_first, reference_path, reference_sha256, replay_of, run_role, schema_version, scope, seeds, self_sha256, static_arms, vs_neuron_memory, what_mpc_means_here`

Design block:
- `primary_metric = "auc_regret_norm"` — **lower is better** (it is normalised regret)
- `budget = 24`
- `seeds` = `[5300001 … 5300012]`, n = 12
- `contexts` = `["R1r","R2r","R1r+R2r","R1r|esc","R2r|esc","R1r+R2r|esc"]`, n = 6
- `arms` = 16 (15 + `oracle`); `memory_arms` = 6, `static_arms` = 7, `new_arms` = 4
- `scope = "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION"`, `run_role = "CACHE_ANALYSIS"`, `replay_of = "garrido_q2_des288"`
- `claim_status = "THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH"`

**`per_arm` shape (the load-bearing answer):** for every one of the 16 arms,

```
per_arm[arm]["auc"]         -> list, len 12   (one float PER SEED)
per_arm[arm]["final"]       -> list, len 12   (one float PER SEED)
per_arm[arm]["budget_used"] -> list, len 72   (12 seeds x 6 contexts)
```

So it is **per-seed, not seed x context**. The generating line in `/Users/thom/Projects/research/scres-ia/scripts/run_search_comparator_ladder_v5.py:218` is:

```python
per_arm[name]["auc"].append(float(np.mean(aucs)))   # aucs = one entry per context
```

i.e. the 6 contexts are averaged *inside* each seed, then appended. `budget_used` is the only array kept at the 72-cell granularity. `mean_auc_regret[arm]` is exactly `np.mean(per_arm[arm]["auc"])` — I verified all 16 to 1e-16.

**Consequence:** seeds are shared across all arms (common random numbers — `f6_v4_arms_reproduce` reports `max_drift = 0.0` against `search_ladder_v4`), so `per_arm[f_reset]["auc"][i] - per_arm[f_retained]["auc"][i]` is a legitimate paired difference over n=12 independent replication units. **The contrast is a pure re-read of sealed numbers; nothing is re-simulated.** Paired *within-context* contrasts are NOT available (contexts are already collapsed) — if the audits demand context-level pairing, that would require a re-run.

---

## 2. The six twins all exist; six raw paired deltas

All 12 arms confirmed present in `arms`, `per_arm`, and `mean_auc_regret`. Orientation: `Delta_f = AUC(reset) - AUC(retained)`, **positive = retention helps** (lower regret is better).

| family | reset arm | mean AUC | retained arm | mean AUC | **Delta_f** | seeds where retained wins |
|---|---|---|---|---|---|---|
| neuron | `neuron_reset` | 0.11273606 | `neuron_memory` | 0.05203275 | **+0.06070** | 12/12 |
| ucb1 | `ucb1` | 0.09655147 | `ucb1_transfer` | 0.04502301 | **+0.05153** | 11/12 |
| ofat | `ofat` | 0.10024198 | `ofat_transfer` | 0.06274307 | **+0.03750** | 12/12 |
| lookahead_kg | `lookahead_kg` | 0.11479478 | `lookahead_kg_transfer` | 0.08018190 | **+0.03461** | 12/12 |
| gp_ei | `gp_ei` | 0.10661346 | `gp_ei_transfer` | 0.08390483 | **+0.02271** | 12/12 |
| thompson | `thompson` | 0.10893185 | `thompson_transfer` | 0.08907820 | **+0.01985** | 10/12 |

**All six are positive, in the same direction.** For reference, reconstructing the artifact's own bootstrap recipe (`boot()` at `scripts/run_search_comparator_ladder_v5.py:228-231`, percentile bootstrap over the 12 paired differences) gives:

| family | Delta_f | 95% percentile bootstrap |
|---|---|---|
| neuron | +0.06070 | [+0.04587, +0.07990] |
| ucb1 | +0.05153 | [+0.03583, +0.06592] |
| ofat | +0.03750 | [+0.02943, +0.04642] |
| lookahead_kg | +0.03461 | [+0.02609, +0.04335] |
| gp_ei | +0.02271 | [+0.01294, +0.03371] |
| thompson | +0.01985 | [+0.01002, +0.02937] |

(Paired-t CIs agree: neuron [+0.04046, +0.08095], thompson [+0.00851, +0.03120]. Every interval excludes zero. Note my `neuron` bootstrap [+0.04587, +0.07990] reproduces the *sealed* `vs_neuron_memory.neuron_reset` interval [+0.04591, +0.07997] to 5e-5 — the residual is bootstrap RNG only. That is a strong check that the recipe is the artifact's own.)

Note the ranking claim the audits object to is also literally in the file: `ranking_best_first[0:6]` = `["ucb1_transfer","neuron_memory","ofat_transfer","lookahead_kg_transfer","gp_ei_transfer","thompson_transfer"]` = exactly `memory_arms`. That is the observation the audits say cannot carry RQ1.

---

## 3. Contrast blocks already sealed in `search_ladder_v5`

There is **exactly one** contrast block: **`vs_neuron_memory`**, with 14 entries (all arms except `oracle` and `neuron_memory` itself). Each entry has `{mean, lcb95, ucb95, n:12}`. Line 233-235 of the script: `arm_auc - neuron_memory_auc`, per seed, then percentile bootstrap.

Full key list and what each compares (all are `arm − neuron_memory`, positive = arm worse than the neuron):

| key | mean | lcb95 | ucb95 |
|---|---|---|---|
| `annealing` | +0.12217 | +0.09548 | +0.14882 |
| `random` | +0.08776 | +0.07033 | +0.10376 |
| `lookahead_kg` | +0.06276 | +0.04335 | +0.08082 |
| **`neuron_reset`** | **+0.06070** | **+0.04591** | **+0.07997** |
| `lhs_local` | +0.05746 | +0.03515 | +0.08190 |
| `thompson` | +0.05690 | +0.04068 | +0.07242 |
| `gp_ei` | +0.05458 | +0.03585 | +0.07258 |
| `ofat` | +0.04821 | +0.03292 | +0.06346 |
| `ucb1` | +0.04452 | +0.03519 | +0.05447 |
| `thompson_transfer` | +0.03705 | +0.02038 | +0.05185 |
| `gp_ei_transfer` | +0.03187 | +0.01923 | +0.04215 |
| `lookahead_kg_transfer` | +0.02815 | +0.01285 | +0.04191 |
| `ofat_transfer` | +0.01071 | +0.0000356 | +0.02171 |
| `ucb1_transfer` | −0.00701 | −0.02444 | +0.01408 |

**Retained-vs-reset already sealed with a CI: exactly ONE family — `neuron`** (`vs_neuron_memory.neuron_reset`). Because both arms in that family are the pivot and its twin, that entry *is* Delta_neuron.

The other five families are **not** sealed as contrasts. Their *means* are recoverable by subtraction inside the block (e.g. `vs_neuron_memory.ucb1.mean − vs_neuron_memory.ucb1_transfer.mean` = 0.04451872 − (−0.00700974) = **0.05152846**, matching the direct paired mean to 1e-16), but their **CIs are not** — differencing two bootstrap intervals is invalid. The CIs must be recomputed from `per_arm[...]["auc"]`, which is arithmetic on sealed per-seed numbers, not new simulation.

There is no top-level `contrasts` key in this artifact.

---

## 4. Is `garrido_normaliser_audit_v3`'s `memory_vs_reset` the same quantity? **Yes — identical, for the `prefix` normaliser.**

Path: `results/garrido_normaliser_audit_v3/result.json` → `summary.prefix.contrasts.auc_regret_norm.memory_vs_reset`

```
{ "mean": 0.06070331006303109, "lcb95": 0.04556081327092027,
  "ucb95": 0.08019782091887094, "n": 12 }
```

versus `search_ladder_v5.vs_neuron_memory.neuron_reset`:

```
{ "mean": 0.06070331006303109, "lcb95": 0.04590626893881853,
  "ucb95": 0.07997188775380663, "n": 12 }
```

**The point estimates are bit-identical** (`0.06070331006303109` in both files). The intervals differ only in the 4th decimal — different bootstrap RNG streams, same estimator.

Matching provenance:
- **Same seeds:** both `[5300001 … 5300012]`, both `replay_of = "garrido_q2_des288"`, both declare the same registry conflict against `results/garrido_meta_learner_v2/result.json`
- **Same contexts:** both list the identical 6, `["R1r","R2r","R1r+R2r","R1r|esc","R2r|esc","R1r+R2r|esc"]`
- **Same endpoint:** both `primary_metric = "auc_regret_norm"`, both `budget = 24`
- **Same normaliser:** `audit_v3.summary.prefix.means_auc_regret_norm` = `{neuron_memory: 0.05203274760040456, neuron_reset: 0.11273605766343564, ofat: 0.10024197702481125}` — these match `search_ladder_v5.mean_auc_regret` exactly (ofat matches to 2e-17). **`search_ladder_v5` runs the `prefix` normaliser.**
- Same `calibration_provenance` block (schema `calibration_provenance_v1`, `fulfilment_delay_hours = 54.0`, `lead_time_promise_hours = 48.0`)
- Same `supply_chain/arm_runner.py` sha256 `35135c53…` and `supply_chain/seed_custody.py` sha256 `3d46668e…` in both `module_manifest.modules`

**Caveat you must state:** the `oracle` normaliser gives a *different* number — `summary.oracle.contrasts.auc_regret_norm.memory_vs_reset.mean = 0.09014693` [0.07419, 0.10803], with `neuron_reset` mean 0.14131158. So `+0.06070` is the **`prefix`-normaliser** figure specifically. `audit_v3.claim_status = "ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER"`, `scope = "DEVELOPMENT_REPLAY_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER"`. `audit_v3` also seals `memory_vs_ofat` and `memory_vs_random` for both normalisers and for a second endpoint `runs_to_within_1pct` (prefix `memory_vs_reset` = 5.833 runs [4.444, 7.306]) — but note `primary_rationale` in that file explicitly deprecates `runs_to_within_1pct` as censored (censoring rates: `neuron_memory` 0.056, `neuron_reset` 0.153, `ofat` 0.222, `random` 0.611).

**So: one of the six is already sealed with a CI, twice, in two independent artifacts.**

---

## 5. RQ2 separation — three genuinely distinct contracts

### A. `results/surrogate_architecture_bakeoff/result.json` — outer-loop surrogate search
- `schema_version = "surrogate_architecture_bakeoff_v1"`, `claim_status = "KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP"`
- **Endpoint:** `primary_metric = "auc_regret_norm"` (no `endpoint` key). **Lower is better.**
- **Seeds:** `[5300001 … 5300012]`, n=12, `replay_of = "garrido_q2_des288"` — the *same burned tape block* as `search_ladder_v5`
- `contexts` = the same 6; `budget = 24`
- `scope = "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION"`, `run_role = "CACHE_ANALYSIS"`
- **Compares:** 7 surrogate models used as the *acquisition model of an outer-loop configuration search*: `["neuron_5p","kan","mlp_matched","gp_matern","gbt","spline_poly","random"]`
- `mean_auc_regret`: `neuron_5p` 0.05203275, `mlp_matched` 0.08852563, `kan` 0.09889469, `spline_poly` 0.09753787, `gbt` 0.10832339, `gp_matern` 0.11379410, `random` 0.13979456
- Sealed contrast `kan_minus_matched_mlp = {mean: +0.01036905, lcb95: +0.00301822, ucb95: +0.01892608, p_two_sided: 0.0012}` — KAN *worse*
- Parameter-matched at `{kan: 532, mlp_matched: 529}` params
- `per_arm[arm]["auc"]` is again len-12 per-seed (same shape as the ladder)
- Contains `scope_note`: it explicitly disclaims the MPC lane — *"This does NOT live in the MPC lane… nothing is deployed: a configuration is chosen, the oracle is exact."*
- `neuron_5p` mean 0.05203274760040456 is **byte-identical** to `search_ladder_v5.mean_auc_regret.neuron_memory` — same tape, same arm

### B. `results/dmlpa_kan_latent/result.json` — PPO intra-episode / track_b (DIFFERENT CONTRACT)
Full report as requested:
- `schema_version = "dmlpa_kan_latent_v1"`, `claim_status = "KAN_LATENT_HURTS"`
- **`endpoint = "ret_mean_track_b_v1"`** — set at `scripts/run_dmlpa_kan_latent_v1.py:198`
- **Units:** ReT, the sequential operational resilience index (`ReT_seq_v1`, Cobb-Douglas `SC^0.60 × BC^0.25 × AE^0.15`, extending Garrido Eq. 5.5), reported on a ~0–100 scale; arm means sit near 98
- **Orientation: HIGHER IS BETTER.** Confirmed by sign consistency: `kan_minus_mlp.mean = −0.86225` and `claim_status = "KAN_LATENT_HURTS"`
- **Arm means:** `dmlpa_mlp = 98.44314398434753`, `dmlpa_kan = 97.5808931153368`
- **Sealed contrast** `kan_minus_mlp = {mean: −0.86225087, lcb95: −1.60504353, ucb95: −0.11945820, n: 5}` — interval excludes zero
- **Seeds / seed block:** `seeds = [9491, 9492, 9493, 9494, 9495]`; `seed_block = {derived_from: "seeds", start: 9491, end: 9495, n: 5}`. **n = 5**, and these are a *completely different* seed family from the 5300001–5300012 burned tape used by A.
- **Per-seed spread** (from `rows`):

| seed | `dmlpa_mlp` | `dmlpa_kan` | kan − mlp |
|---|---|---|---|
| 9491 | 98.91837 | 98.23893 | −0.67945 |
| 9492 | 97.47921 | 97.38434 | −0.09487 |
| 9493 | 97.97826 | 98.11421 | **+0.13595** |
| 9494 | 99.13093 | 97.32554 | −1.80539 |
| 9495 | 98.70894 | 96.84145 | −1.86749 |

  SD of the paired difference = 0.93791; 4/5 seeds negative, 1 sign flip at seed 9493. Between-seed SD: mlp 0.69189 (range 97.479–99.131), kan 0.58482 (range 96.841–98.239). Each row also carries `ret_sd_within` (0.76–1.06), the within-seed spread across the 24 eval episodes.
- **Scope:** `"DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN"` — weaker custody than A's burned-tape replay
- **Design:** `{total_steps: 100000, n_envs: 8, eval_episodes: 24, features_dim: 84, nhead: 12, num_layers: 2, ff_mult: 4}` — this is PPO training, 100k steps, not a 24-evaluation configuration search
- **Falsifiers** (`all_passed: true`), 5 of them:
  - `f1_parameters_are_matched_within_our_tolerance` — `dmlpa_kan` 199,082 params (hidden_dim 10, `use_kan: true`, error 0.00459) vs `dmlpa_mlp` 200,052 params (hidden_dim 152, `use_kan: false`, error 0.00026), tolerance 0.1
  - `f2_only_the_latent_rw_differs` — held fixed `{features_dim, ff_mult, nhead, num_layers}`; `not_covered: "normalisation order differs from David's version"`
  - `f3_the_two_arms_are_behaviourally_distinct` — 6-element output fingerprints differ
  - `f4_the_harness_reproduces` — `delta: 0.0`, replicas `[89.23527051364351, 89.23527051364351]`
  - `f5_no_new_seeds` — 9491–9495 already open
- Other honesty fields: `result_before_training` (*"at equal budget the KAN affords hidden_dim=10 against the MLP's 152"*), `not_covered`, `whose_defect`, `what_it_broke` (David's run collapsed both labels onto one 225,410-param object)

### C. `results/garrido_fig5_surrogate/result.json` — static regression replication of Garrido Fig. 5
- `schema_version = "garrido_fig5_surrogate_v1"`, `claim_status = "DEVELOPMENT_FIG5_SURROGATE"`
- **No `endpoint`, no `primary_metric`, no `scope`, no `seeds` list, no `contexts`.** It has a **single** `seed = 20260731`.
- **Endpoint is per-task:** task B1 uses out-of-sample **R²**, task B2 uses **classification accuracy**; both higher-is-better, both 5-fold with grouped folds (`n_groups: 30`, `n_rows: 90`, `n_pairs: 87`, `n_configurations: 90`)
- **Compares:** `{linear, backprop, kan, constant}` on a *fixed static driver table* (`results/garrido_drivers_per_configuration/result.json`), decision rule `"a network counts as learning something only if it beats the linear/logistic baseline by more than one between-fold SD of that baseline"`
- `task_B1_regression.mean` = `{linear: 0.96975 (sd 0.01651), backprop: 0.98632 (sd 0.00816), kan: 0.99133 (sd 0.00737), constant: 0.0}` → `verdict.B1_regression = {backprop: true, kan: true}`
- `task_B2_activation_question.mean` = `{linear: 0.71111 (sd 0.08240), backprop: 0.71778 (sd 0.13531), kan: 0.77111 (sd 0.09250), constant: 0.33333}` → `verdict.B2_activation = {backprop: false, kan: false}`
- `task_A_figure_as_drawn.status = "IDENTITY_NOT_A_LEARNING_TASK"`, R² = 1.0, `max_abs_identity_error = 3.2e-15` — the figure as Garrido draws it has nothing to learn
- **There is no simulator in this artifact at all** — no PPO, no search, no seed block, no custody replay

### The plain statement you asked for

The three are provably distinct contracts, and the artifacts themselves say so:

- **A** and **B** share the word "KAN" and nothing else. A is a **surrogate inside an outer-loop configuration search** — endpoint `auc_regret_norm` (lower better), 12 burned-tape seeds 5300001–5300012, budget 24 evaluations, 6 contexts, `run_role: CACHE_ANALYSIS`, and it carries a `scope_note` that explicitly refuses the control lane. **B** is a **PPO policy network's latent representation under the `track_b_v1` action contract** — endpoint `ret_mean_track_b_v1` (higher better), 5 open development seeds 9491–9495, 100,000 training steps, 24 eval episodes, `scope: DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE`. **Different endpoint, different orientation, different seed family, different n, different unit of analysis, different scope string.** `dmlpa_kan_latent` is intra-episode control; `surrogate_architecture_bakeoff` is outer-loop simulation-optimisation over configurations — the very distinction `search_ladder_v5.what_mpc_means_here` spells out.
- **C** is neither: it is a static supervised-learning replication on a 90-row driver table with one seed and no simulator, whose headline finding is that the figure as drawn is an algebraic identity.

They happen to agree in direction (KAN does not help in A or B; KAN does not beat linear on the question that matters in C), which is worth saying — but they are three separate evidentiary claims, not three measurements of one.

---

## Bottom line for RQ1

The paired within-family framing the four audits demand is **fully supported by sealed data with zero new computation**: `per_arm[arm]["auc"]` gives 12 paired per-seed values for all 12 arms under common random numbers, all six twins exist, all six deltas are positive, and one of the six (`neuron`) is already sealed with a bootstrap CI in two independent artifacts at the identical point estimate `+0.06070331006303109`. The five remaining CIs require re-running the artifact's own `boot()` over already-sealed arrays — arithmetic, not simulation. The one thing you *cannot* do without a re-run is pair within context, since the 6 contexts are averaged away at `run_search_comparator_ladder_v5.py:218`.
