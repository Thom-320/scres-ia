# Plan — You are designing an implementation plan for a research repo at <HOME>/Projects/resea

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a06e128f3aec59b78`
- **session**: `091a3905-ebf9-481b-beaa-fc646c3902d5`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-05T18:18:56.492Z`
- **effort**: `xhigh` · **version**: `2.1.219`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/091a3905-ebf9-481b-beaa-fc646c3902d5/subagents/agent-a06e128f3aec59b78.jsonl`
- **turns**: 135 · **assistant messages**: 2

## Task given

```text
You are designing an implementation plan for a research repo at <HOME>/Projects/research/scres-ia. Be adversarial and concrete. Read the files I cite before concluding.

# The goal
Produce a submittable paper for *Computers & Industrial Engineering* that answers the two questions in Garrido, Pongutá & Adarme (ICCL 2024, LNCS 15168, pp. 80-94):
Q1: what category of AI algorithms best mimics the supply-chain-learning (SCL) attribute?
Q2: how can that family be integrated into the internal structure of a DES model for SCRES assessment?
The PI wants RL (hybrid or pure) to beat every other way of finding policies, if that is achievable honestly.

# My core hypothesis about where the problem is — pressure-test it
Garrido's Fig. 2 loop is drawn over SUCCESSIVE SIMULATION RUNS, not over timesteps within an episode. Nodes (3) data gathering and (8) verification & validation are the two ends of an open loop across ITERATIONS of the DES. His example activation function is "Is the SCRES measure at configuration x higher than at configuration (x-1)?" — configuration, i.e. an experimental-design point, not a timestep. His conclusions name three candidates: backpropagation NN, Kolmogorov-Arnold NN, and "simulation-optimization approach as a form of reinforcement learning".

So my claim is: this project has spent a year on the INNER loop (within-episode adaptive control: PPO/RecurrentPPO vs MPC vs structured rules) and measured it saturated, while Garrido asked about the OUTER loop (learning across runs to choose the next configuration). And the only place the project has ever measured a learner beating everything is exactly the outer loop.

Evidence for saturation of the inner loop (verify these):
- results/headroom/g3_obs_conversion_v2/result.json -> STRUCTURED_CONTROL_SUFFICES_G3_OBS (a two-branch `if` exhausts the observable value)
- results/headroom/g2_autotomy_threshold/result.json -> THRESHOLD_RULE_SUFFICES
- results/headroom/buffer_prediction_premium/result.json -> NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE (MLP R2 0.5548 WORSE than linear 0.6826; KAN 0.7163)
- results/headroom/cd_surface_prediction_premium/result.json -> both networks below the classical baseline, spline wins
- papers/paper2/results_table.json -> Delta_N negative in all three cells
- docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md -> no positive instance

Evidence for the outer loop being where the value is (verify these):
- results/garrido_meta_learner_v2/result.json -> memory 6.99 runs to optimum vs reset 14.89 vs OFAT 12.42 vs random 19.54; advantage +7.90 [6.88, 8.93]
- results/garrido_h3_merge_adjudication/result.json -> H3_PRIME_SUSTAINED_AT_N120, memory vs reset variance-of-search-cost +9.3144 [+2.3491, +16.3474], memory vs OFAT +16.2203 [+9.6107, +22.7405]
- docs/RESULTADO_META_APRENDIZ_2026-07-31.md and docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md (the v1 numbers 7.24/12.42/13.54 are RETIRED for driver leakage plus a hardcoded f5)

# The known holes you must design around
1. The outer-loop learner today is `class Fig5Neuron` in scripts/run_meta_learner_over_configs_v1.py:122 — a single logistic unit with GREEDY argmax, no exploration. Its only comparators are OFAT, random, and itself-with-memory-reset. A C&IE referee will immediately ask for Bayesian optimisation. A GP-EI implementation already exists at supply_chain/gsa.py:68 (`gp_locate`) but is NOT wired into any meta-learner runner. Missing entirely: bandits (UCB/Thompson), best-first/branch-and-bound, simulated annealing, CMA-ES.
2. There is NO cached evaluation of the 288-configuration surface. Every runner rebuilds it: 288 configs x 6 contexts x 12 seeds = 20,736 DES episodes, measured ~46 s/config, 1728 episodes per replicate (~126 s on M1 Pro, ~504 s on the VPS). Caching the surface would make every search strategy nearly free to compare and would allow training an RL search policy.
3. The 288 grid is defined in scripts/run_meta_learner_over_configs_v1.py:52-61 as buffer_hours(6) x shi
```

## Final answer

I read every artifact you cited plus the ones you did not know existed. Here is what survived verification, what did not, and the plan.

## 1. Verification: your evidence, checked

**Confirmed as stated:**

| claim | verified value | source |
|---|---|---|
| inner loop saturated | `STRUCTURED_CONTROL_SUFFICES_G3_OBS`, `THRESHOLD_RULE_SUFFICES`, MLP R² 0.5548 < linear 0.6826 < KAN 0.7163, all `passes_sesoi_and_ci: false` | the four `results/headroom/*/result.json` |
| outer loop numbers | memory 6.986 / reset 14.889 / OFAT 12.417 / random 19.542; Alzheimer +7.903 [6.875, 8.931] | `<HOME>/Projects/research/scres-ia/results/garrido_meta_learner_v2/result.json` |
| H3′ at n=120 | memory−reset +9.3144 [2.3491, 16.3474]; memory−OFAT +16.2203 [9.6107, 22.7405] | `<HOME>/Projects/research/scres-ia/results/garrido_h3_merge_adjudication/result.json` |

**Wrong, and it matters:**

- **The surface is not expensive.** Measured across four sealed artifacts: **72–91 ms per DES episode** (`garrido_meta_learner_v2`: 20,736 episodes / 1,492.8 s = 72.0 ms; `h3power_local_v2`: 155,520 / 11,357.1 s = 73.0 ms; `h3power_vps_v2`: 86.0 ms; `q2_des288` smoke: 91.0 ms). Your "~46 s/config" is off by ~600×. The full 288×6×12 surface is **25 minutes**, and the n=120 surface is **4.3 hours**. The cache is still the right call, but not because the surface is expensive — because it makes *strategy comparison and RL training* free, and because it is the only way to re-measure the headline under a corrected learner (see §2.1).

- **Hole #3 is half wrong.** Upstream buffers at WDC and AL are **already decision variables**. `MFSCSimulation.__init__` takes `initial_buffers` with keys `op3_rm` → `raw_material_wdc`, `op5_rm` → `raw_material_al`, `op9_rations` → `rations_sb` (`<HOME>/Projects/research/scres-ia/supply_chain/supply_chain.py:1467-1484`, `_deliver_buffer_top_up`). Both meta-learner runners already pass them, pinned at `0.0`. And `<HOME>/Projects/research/scres-ia/results/garrido_continuous_space/result.json` already swept 384 points over `{op3_rm, op5_rm, op9_rations, period_hours}`. What is *not* wired is finite **capacity** at wdc/al/sb — a different lever from the buffer level.

- **You are planning against the wrong runner.** `scripts/run_garrido_q2_des288_v1.py` (804 lines, 2026-08-01) already supersedes `run_meta_learner_over_configs_v1.py`: five arms (`ofat/random/no_update/retained/reset`), the service-first v2 lexicographic endpoint, `surface_sha256`, learner `state_digest()`, nine falsifiers including a zero-budget identity and an endpoint-recomputation check. It has only ever been run as a smoke (`results/garrido_q2_des288_v1/smoke_*.json`, 288 configs × **1 context × 1 seed**, `Q2_EFFECT_NOT_ESTABLISHED`). This is the platform. Do not extend the v1 runner.

## 2. Three defects that will kill this paper at review unless fixed first

### 2.1 The headline learner reads the unrun surface (a third leak of the family already retired twice)

`<HOME>/Projects/research/scres-ia/scripts/run_meta_learner_over_configs_v1.py:189-190`:

```python
values = [v for v, _ in table]           # ALL 288 values, including unrun configs
best, lo, span = max(values), *scaled(values)
```
and line 234:
```python
neuron.update(features(CONFIGS[idx]), (value - lo) / span)
```

The learner's target normaliser is min/max over the **entire surface**, i.e. over configurations that have not been run. This is load-bearing, not cosmetic: `ret_excel_risk_conditional` is ~0.009 in the R1r contexts, so without the oracle rescaling every gradient step would be `(0.009 − σ(0)) = −0.49` and `rho` would collapse uniformly negative. The neuron only works *because* it is handed the surface's range.

Two independent reasons this is fatal as written:

1. `<HOME>/Projects/research/scres-ia/docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md` set the precedent explicitly: memory-vs-reset survived a leak because both arms shared it, but **memory-vs-OFAT and memory-vs-random were RETIRED** because "OFAT no recibe esa información". By the repo's own rule, `memory_vs_ofat` = +5.431 and `memory_vs_random` = +12.556 in `garrido_meta_learner_v2` are exposed to the identical objection.
2. The successor runner **already fixed it**. `scripts/run_garrido_q2_des288_v1.py:85-87` defines `TARGET_SCALES` with the comment *"Fixed a priori scales. They are not estimated from the observed surface and are not retained as adaptive normalizers."* The fix exists; the headline artifact predates it.

All H3′ slices inherit the defect — I checked the manifests: `garrido_meta_learner_h3power_h3_contract_local_v2` and `_vps_v2` both carry `entry_script: scripts/run_meta_learner_over_configs_v1.py`, sha `41f73ce8addf`.

### 2.2 The primary estimand is censored, at very different rates per arm

`runs_to_within_1pct` imputes `budget + 1 = 25` when the target is never reached. Actual censoring in `garrido_meta_learner_v2`, over 72 (context × replicate) cells:

| arm | censored at 25 | median |
|---|---|---|
| neuron_memory | 5 / 72 | 4.0 |
| neuron_reset | 11 / 72 | 15.0 |
| ofat | 16 / 72 | 8.0 |
| random | **44 / 72** | **25.0** |

Random's mean of 19.54 is a censored mean whose median sits exactly on the censoring point. The direction is conservative for your two headline contrasts (both understate memory's advantage), which is defensible — but it must be *disclosed*, and a C&IE referee will ask for an uncensored primary. The repo already stores what is needed: `final_regret` and the full `regret_curve` per cell. Switch the primary to **normalized simple regret at budget B** or **AUC of the regret curve**, and demote `runs_to_oracle` to a secondary with an explicit survival-style treatment.

### 2.3 The optimum is a fixed box corner, and it is the same corner in every context

From `chosen_config` over 72 cells of the memory arm:

- `buffer_hours = 1344` (the **maximum** level) in **70/72**
- `op9_rop = 12` (the **minimum** level) in **70/72**
- `op12_rop = 12` (the **minimum** level) in **68/72**
- only `shifts` moves (2: 47, 3: 20, 1: 5)

And the per-context optima: `best` in R1r and R1r|esc is **exactly 0.009259259 = 1/108 with sd = 0.000000 across all 12 seeds**.

This is the hardest thing in your dossier. The 288-configuration surface is **monotone in three of four factors with the argmax at a vertex, and the vertex does not move across the six contexts**. Consequences:

- A logistic unit scoring a *linear* function of normalized coordinates has exactly the right inductive bias for a monotone surface: its argmax is always a vertex. The neuron is not learning a landscape; it is learning a sign vector.
- "Memory beats reset" therefore reduces to **"the answer is the same in every context, so remembering it is free."** That is the *same* invariance result as the inner loop (`docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md` §2.6: *"The optimal constant posture is INVARIANT across all 45 risk profiles"*). Your outer-loop positive and your inner-loop null may be one finding wearing two hats.
- On a monotone, separable, corner-optimum surface, **OFAT is near-optimal by construction and GP-EI will crush everything**. Wiring in `gp_locate` on *this* surface is the fastest way to lose the paper.

**Verdict on your framing.** The diagnosis (Garrido's loop is over runs, not timesteps) is right and well-supported — `CLAUDE.md`, the runner docstrings, and `<HOME>/Projects/research/scres-ia/docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md` §7 already say so. The *mechanism* you propose is wrong. Search cost does not grow because dimension grows; **OFAT fails when the surface is non-separable**. On a separable surface OFAT costs Σᵢ Lᵢ — *linear* in dimension — and finds the global optimum. The 288 grid is close to separable, which is why OFAT is only 12.4 vs memory's 7.0 rather than catastrophically worse.

Your argument survives only if the extended grid is **non-separable**. And there is direct evidence it is. Refitting `garrido_continuous_space` rows with pattern and shift fixed effects:

| family | R² pattern+shift | +ρ linear | +ρ pairwise interactions | incremental interaction |
|---|---|---|---|---|
| R1r | 0.1488 | 0.3412 | 0.5229 | **+0.1817** |
| R2r | 0.6903 | 0.7016 | 0.7158 | +0.0142 |
| R3 | 0.0148 | 0.0354 | 0.1971 | **+0.1617** |

and the argmax is **interior in all three families** and **in a different place in each** (R1r: op3≈65.9k, op5≈1.3k, op9≈70.4k, period≈537; R3: op3≈25.4k, op5≈5.4k, op9≈117.6k, period≈989 — against bounds [0, 122880]/[0, 126000]/[0, 1344]).

So: **the upstream buffers are where the interaction and the moving interior optimum live; the current 288 grid is where they do not.** That is the repair to your theory, and it is testable *before* you spend a seed. Caveat: each of those 384 rows carries a distinct seed (4,100,001+, a block **absent from the registry**), so there is no CRN and the R² gains are in-sample with 6 extra parameters on n≈128. Treat it as a lead, not a result.

## 3. Sequenced plan

### Phase 0 — Free work on burned tapes (no seeds, no authorization, ~1 day)

Everything here is explicitly permitted by `contracts/garrido_expanded_des_e_star_v1.json` → `seed_and_execution_policy.allowed_before_receipt`: *"already burned data audits"*, *"flags-off harness construction"*, *"synthetic tests"*, *"falsifier mutation tests"*.

**P0.1 — Build the surface cache** (§(a) below), populated by a **declared replay** of blocks `garrido_q2_des288` (5,300,001–012) and `garrido_h3_local`/`garrido_h3_vps` (6,000,001–120). `custody_falsifier(..., replay_of="garrido_q2_des288")` in `<HOME>/Projects/research/scres-ia/supply_chain/seed_custody.py:176` already returns `not_applicable=True` for exactly this. Cost: 25 min + 4.3 h, one time.

**P0.2 — Re-measure the headline under an honest normaliser.** Replay the search on the cached surface with a **prefix normaliser** (running min/max over configurations *already run*, with a declared cold-start rule) instead of `scaled(values)`. Report both. Three outcomes, all publishable:
- effect survives → the leak was cosmetic, and you can now say so *with a measurement*;
- effect shrinks but LCB95 > 0 → report the corrected number, retire the old one (the repo has done this twice already);
- effect vanishes → you have saved the paper, and Phase 1 becomes the paper.

**P0.3 — Switch the primary to an uncensored estimand** and recompute from stored `regret_curve`s. No DES needed.

**P0.4 — Fix three defects in `run_garrido_q2_des288_v1.py`** before it is ever run at scale:
- `_falsifiers` `f9` uses the hand-maintained `PRIOR_SEEDS` tuple (lines 70-82). That is precisely the pattern `supply_chain/seed_custody.py` exists to abolish — its module docstring names it. Replace with `custody_falsifier`.
- `search()` OFAT tail: when `factor_index >= len(FACTOR_NAMES)`, the guard `and "idx" not in locals()` is False because `idx` survives from the previous loop iteration (`del idx` only fires at end of context). OFAT therefore re-runs its **last proposal**, not the incumbent. Low numerical impact, but it is in the comparator the headline contrast is against, and it is the kind of thing a referee who asks for code will find.
- `falsifiers["all_passed"] = all(check["passed"] ...)` has no `not_applicable` handling; the v1 runner's version does, and the reason is documented ("eight falsifiers pass" when one could not fail).

### (a) The surface cache

**Shape.** One artifact per `(grid_id, context, seed)`, not one giant file — this makes partial builds resumable and lets the VPS and local machine contribute disjoint slices, which is how the n=120 H3′ was actually produced.

```
results/surface_cache/<grid_id>/<context>/<seed>/result.json
results/surface_cache/<grid_id>/manifest.json
```

**Row schema.** Reuse `evaluate()`'s return in `run_garrido_q2_des288_v1.py:117-202` verbatim — it already stores `config`, `context`, `seed`, `service_key`, `claimant_fills`, `demanded_by_claimant`, `delivered_by_claimant`, `drivers`, and a 12-field `panel`. Do **not** store a scalar. Storing the panel is what makes `f7_endpoint_key_recomputes_independently` possible off the cache, and it means a future endpoint change (v2 → v3) does not invalidate the cache.

**Sealing.** Write via `seal_and_write` from `<HOME>/Projects/research/scres-ia/supply_chain/arm_runner.py:170`, which stamps `contract_path`, `contract_sha256`, `reference_sha256`, `calibration_provenance` and `self_sha256`. Each cache slice additionally carries `module_manifest(script=__file__)` from `seed_custody.py:180` — `module_manifest_v2`, which hashes `supply_chain/supply_chain.py`, `config.py`, `episode_metrics.py`, `arm_runner.py`, `cssu_allocation.py`, `service_first_metric.py`, `seed_custody.py`. The H3′ VPS slice sat in `HOLD_SOURCE_AUDIT` for exactly the absence of this.

**Validation on read.** A consumer must refuse a cache whose:
1. `module_manifest.modules` differs from the current tree → **physics drift**, hard stop, no silent recompute;
2. `calibration_provenance` differs from `calibration_stamp()` → **metric drift** (this is the `fulfilment_delay_hours` / `lead_time_promise_hours` comparability rule already recorded in every artifact);
3. `physical_protocol` differs from `THESIS_FAITHFUL_PROTOCOL`;
4. `horizon_hours`, `grid_id`, `factor levels` differ from the request;
5. `self_sha256` does not reverify.

**The failure mode you must design for is silent physics drift.** A stale cache is *not* mainly a risk of wrong numbers — it is a risk of **irreproducible** numbers, because a cached surface makes it possible to publish a search comparison against physics that no longer exists in the tree. Countermeasure, and it is cheap: a **spot-recompute falsifier**. On every consuming run, re-execute the DES for `k = 8` randomly chosen `(config, context, seed)` cells and require bit-identical `service_key` and `panel`. `why_it_can_fail`: *any* change to the simulator, the metric, the CRN wiring or the interpreter that alters an episode makes it fail. Cost: 8 × 80 ms. This is the check that cannot be satisfied by the code under test, because the cache was written by a *previous* version of that code.

**Reuse `canonical_payload_sha256`** (`arm_runner.py:207`) for cross-version comparison: it excludes `VOLATILE_KEYS` (`created_at`, `elapsed_seconds`, `self_sha256`, `calibration_provenance`, `contract_path`, `reference_path`, `module_manifest`, `audit_status`, `replay_of`) so two runs of different code that produce identical science hash the same. That is exactly the tool for "did the physics move?" as opposed to "did the file move?".

**Do not** use `_surface_digest` from `run_garrido_q2_des288_v1.py:399` as the cache key — it `json.dumps(..., default=str)` a nested float structure and is repr-dependent. Use `canonical_payload_sha256` per slice, and a Merkle-style manifest hashing the sorted slice digests.

### (b) Grid extension

**Phase 1 (recommended, zero DES change).** Extend `FACTORS` in `run_garrido_q2_des288_v1.py:44-49`:

```python
"op3_rm": (0.0, L1, L2, L3),   # WDC raw material, already wired
"op5_rm": (0.0, L1, L2, L3),   # AL raw material, already wired
```

and pass them through `evaluate()`'s `initial_buffers` dict (currently hardcoded to `0.0` at lines 126-130). Levels should be taken from the range already exercised in `garrido_continuous_space`: `[0, 122880]` for both, quantized to 4 levels.

Grid size: `6 × 3 × 4 × 4 × 4 × 4 = 4,608` configurations. Surface cost at 6 contexts × 12 seeds = **331,776 episodes ≈ 6.6–8.3 h**; at 30 seeds ≈ 17–21 h. Use 3 levels each (2,592 configs) if you want an overnight build.

**The null-arm identity test is stronger here than the CSSU one, and free.** The sub-grid `op3_rm = op5_rm = 0.0` is *definitionally* the current 288 surface, because both runners already pass those keys at `0.0`. So:

```
test_extended_grid_null_arm_reproduces_the_frozen_288_surface
```
asserts that for a frozen set of `(config, context, seed)` cells, `scientific_payload_sha256(canonical_scientific_payload(sim, panel))` on the extended runner equals a **golden constant frozen from the sealed 288 cache**. Follow `<HOME>/Projects/research/scres-ia/tests/test_cssu_capacity_bridge.py` exactly — its two-test structure is the pattern:
- `test_the_flag_is_inert_when_the_caps_cannot_bind` (shipped == extension-off), which alone proves only that two paths agree;
- `test_the_null_is_anchored_outside_the_code_path_it_guards` against `GOLDEN_SHIPPED_PAYLOAD_SHA256 = "f3fe61b1e2b1f4a63ff30beb296d4c6bf54be029fed0ab7877b69e24201af385"`, which is the one that can fail on a shared-code defect.

Add the third test from that file — `test_a_spilling_mutant_in_the_production_path_is_caught` — adapted: monkeypatch `_deliver_buffer_top_up` to ignore its `key` argument and assert the tight arm stops differing. Without a mutation test you have not shown the extension reaches the production path.

**Phase 2 (only if Phase 1 does not produce interaction).** Wire `NodeCapacityLedger` to `wdc`/`al`/`sb`. `CAPACITY_NODES` in `<HOME>/Projects/research/scres-ia/supply_chain/node_capacity.py:37` already names all five; only `cssu_a`/`cssu_b` reach the DES (`supply_chain.py:511-515` construction, `supply_chain.py:5085-5086` the single `admit()` call site). You need `admit()` at the three `put` sites: `raw_material_wdc`, `raw_material_al`, `rations_sb` in `_deliver_buffer_top_up` and in the `t=0` injection at `supply_chain.py:1066-1080`. Use `budgeted_ledger(total, shares)` (node_capacity.py:154), **not** independent caps — the module docstring states the measured reason: *"Capping each node independently only makes the chain worse; nothing has to be traded off, so there is no decision and no headroom"*, and Program O measured H_PI = 0.1515 under non-fungibility versus **exactly 0** when fungible. This is the single most likely source of genuine outer-loop interaction, because a shared budget makes the three buffer levels *compete*.

### (c) The comparator ladder, in referee order

A C&IE referee reading "our learner beats one-factor-at-a-time and random" will stop at the first sentence. The ladder, cheapest first, all of them free once the cache exists:

| # | comparator | reuse / write | notes |
|---|---|---|---|
| 0 | oracle ordering (descending true value) | write, ~10 lines | the ceiling; regret 0 at step 1 |
| 1 | random without replacement | exists (`search(strategy="random")`) | the null |
| 2 | OFAT / thesis order | exists, **fix the `idx` staleness bug** | Garrido's own design |
| 3 | Latin-hypercube + greedy local refinement | write | the standard simulation-optimization baseline |
| 4 | **GP-EI (Bayesian optimization)** | **reuse `gp_locate`**, `<HOME>/Projects/research/scres-ia/supply_chain/gsa.py:68` | the one the referee will name |
| 5 | UCB1 and Thompson sampling over factor-level arms | write, ~40 lines | cheap, and the honest "bandit" answer |
| 6 | simulated annealing on the grid graph | write, ~30 lines | discrete-space standard |
| 7 | CMA-ES on the continuous relaxation, snapped | `scipy`/`cma` | only if you go continuous |
| 8 | best-first / branch-and-bound with a monotone bound | write | strong on a monotone surface — include it *because* it is strong |
| 9 | the RL search policy | write (§d) | |

Three concrete problems with reusing `gp_locate` that you must fix, or the comparison is a straw man in your favour and a referee will say so:

1. **`n_init=16` of a budget of 24.** The signature (`gp_locate(f, bounds, n_init=16, n_iter=24, seed=0)`) spends two-thirds of the budget on LHS before its first EI step. Budget-match it: expose `n_init` and set it to 4–6, `n_iter = B − n_init`.
2. **No history returned.** The docstring says *"Returns best (x, y) + history"* but the return dict is `{"x_best", "y_best", "n_eval"}`. A regret curve needs the visit sequence. Extend the return; do not reimplement.
3. **Continuous bounds vs a discrete grid.** `gp_locate` proposes from `lo + rng.random((2048, k)) * (hi - lo)`. You must snap to the nearest grid point and handle duplicates (either re-propose the best unvisited candidate by EI, or count the duplicate against the budget). Declare which; both are defensible, only one is honest if chosen after seeing results.

`gp_locate` is already validated: `tests/test_program_i_gsa.py` checks it against Ishigami, and `results/headroom_gsa/verdict_gp.json` shows a live 40-evaluation run. It is production-ready.

### (d) RL over configurations

**Episode = a career.** One episode is the ordered traversal of `C` contexts, each with a budget of `B` evaluations, on one cached seed's surfaces. Total `C × B` steps. This makes memory an ordinary intra-episode hidden state and lets you use plain PPO — **do not reach for RecurrentPPO**; lane 11 of `docs/PROMISING_LANES_REGISTRY.md` records `RecurrentPPO (memory) × v4 × control_v1 | TRIED 500k×5 | LOST`, and `contracts/garrido_expanded_des_e_star_v1.json` gates it behind an "observability aliasing gate" anyway.

**State** `s_k` — everything must be derivable from configurations *already run*:
- `k/B`, `c/C` (budget and context position);
- for each factor and each level: running mean and count of the endpoint at that level among visited configs, plus a `visited` mask (this is a sufficient statistic for a separable surface and is where the policy can learn that the surface is *not* separable);
- the incumbent's normalized coordinates and its four endpoint components;
- **the retained vector** carried across the context boundary (the analogue of `rho`) — this is the object under test.

Forbidden in the state, and each gets a falsifier: any unrun configuration's outcome; any surface-wide statistic (min, max, mean, argmax); the risk-set label as a raw identifier (use only an *observable* context descriptor computed from episodes already run in that context, or nothing at all).

**Action.** Primary: the Fig. 5 parameterization — the policy emits a weight vector `ρ_θ` over design features, scores all unvisited configurations, and samples from `softmax(ρ_θ · φ(x) / τ)`. This is Garrido's neuron with a *learned and stochastic* activation, it fixes the greedy-argmax hole you identified, and it transfers across grids of different cardinality (essential for the 288 → 4,608 generalization test). Ablation: a factorized OFAT-shaped action (choose factor, choose level, relative to incumbent), `|A| = 17` on the current grid — this makes the policy directly comparable to OFAT.

**Reward.** Per-step improvement in normalized simple regret, `r_k = (best_k − best_{k−1}) / span_train`, where `span_train` is a **fixed a priori constant** derived from the *training* surfaces only — never from the evaluation surface. The E* contract permits a development-only scalar (`cvar.may_be_used_in_development_reward: true`) while forbidding an unregistered weighted sum as an *estimand* (`scalarization.unregistered_weighted_sum_forbidden: true`, `lexicographic_service_selection_not_an_estimand: true`). So: **train on the scalar, report on the lexicographic key**, and say so in the preregistration. Do not use `ret_excel` as the reward — `docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md` measured that it rewards abandoning a claimant, and `service_first_key_v2` (worst-claimant fill first) exists precisely because of it.

**Transfer, trained and tested.** Three splits, all preregistered:
- **seed split** — train on `S_train`, evaluate on disjoint `S_test`; guards against memorizing a realization;
- **context split** — train on 4 contexts, evaluate on 2 held out; this is the transfer claim;
- **grid split** — train on the 288 sub-grid, evaluate on the 4,608 extended grid. This is the direct test of your dimension argument, and BO structurally cannot do it (a GP prior does not transfer across a change of design space). If RL beats BO anywhere, it is here.

**Null and placebo arms** (the paper lives or dies on these):

1. **Untrained policy** — same architecture, random init, same budget.
2. **Shuffled-surface policy** — trained on surfaces whose values are permuted across configurations (marginals preserved, structure destroyed). If it matches the real policy, nothing about the surface was learned.
3. **Marginal-replay placebo** — a policy that reproduces the trained policy's *marginal visit distribution* but ignores the state. **This is the decisive one**, and §2.3 is why: on a corner-optimum surface a fixed static ordering ("go to the corner first") is near-optimal, so a policy that memorized the corner would beat every comparator *with zero adaptivity*. If the real policy does not beat its own marginal replay, you have measured a lookup table, not learning.
4. **Permuted context descriptor** — if transfer survives a randomly permuted context label, the transfer is not context-specific.
5. **Retained-state-reset arm** — the Alzheimer contrast, carried over from `retained` vs `reset`.

Placebos 1–4 have no analogue in the current runner and their absence is the biggest single gap between what you have and what C&IE will accept. `CLAUDE.md` already mandates *"Uninformed placebo in every headroom measurement"*.

### (e) Falsifiers per stage

Following the repo's rule — a falsifier states *why it can fail*, and a check whose expected value is computed by the code under test proves nothing.

**Cache:**
- `f_cache_reproduces_the_simulator`: 8 random cells recomputed from the live DES must match bit-for-bit. *Can fail because any change to simulator, metric, CRN or interpreter changes an episode.* This is the anti-drift check and it is not self-referential — the cache was written by a prior version.
- `f_cache_manifest_matches_tree`: `module_manifest` and `calibration_provenance` equal the current stamp. *Can fail on any edit to the seven declared modules.*
- `f_cache_mutation_is_detected`: corrupt one stored `service_key` by 1 ULP and require the consumer to reject. *Can fail if the validator only checks metadata.*

**Grid extension:**
- `f_null_arm_is_the_frozen_288`: golden `scientific_payload_sha256` anchors. *Can fail if the extension changed physics when off.*
- `f_extension_is_live`: at least one non-zero upstream level changes the sealed science AND `binding_fraction() > 0` (Phase 2). *Can fail if the new variables are decoration — and if they are, the whole grid extension is padding and the paper should say so.*
- `f_extension_is_non_separable`: fit an additive model and a model with pairwise interactions on the **cached** extended surface; require the interaction gain to exceed a preregistered threshold with a positive LCB95 across seeds. *Can fail — and on the current 288 grid it very likely would (§2.3). This is the go/no-go gate for the whole outer-loop thesis and it must be run before the RL work, not after.*
- `f_optimum_moves_across_contexts`: the argmax config must differ across contexts by more than seed noise. *Can fail — and on the current grid it does fail: `buffer_hours=1344, op9_rop=12, op12_rop=12` in ~70/72 cells. If it fails on the extended grid too, "transfer" is the transfer of a constant and must be described as such.*

**Comparators:**
- `f_budgets_are_matched`: every arm consumes exactly `B` DES evaluations per context, counted from the cache-access log, not asserted. *Can fail if `gp_locate`'s `n_init` is not counted.*
- `f_no_arm_reads_an_unrun_cell`: keep the existing shadow-surface tests (`f5`/`f6` in `run_garrido_q2_des288_v1.py:490-511`) and extend them to every new arm.
- `f_normaliser_is_not_an_oracle`: replay each arm with the surface's values affinely rescaled per context by a random positive factor; any arm whose visit sequence changes has read a surface-wide statistic. *Can fail — and it would have caught §2.1.* This is the highest-value new falsifier in the plan.

**RL:**
- `f_policy_beats_its_own_marginal_replay` (placebo 3). *Can fail, and it is the one most likely to.*
- `f_policy_beats_shuffled_surface_training` (placebo 2).
- `f_evaluation_surface_never_entered_training`: assert disjointness of `(grid_id, context, seed)` between train and eval cache slices, by hash. *Can fail on an off-by-one in the split.*
- `f_transfer_is_context_specific` (placebo 4).

### (f) Power and seeds — what the PI actually signs

Current authority: `research/seed_custody_registry.json` is `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, `scientific_execution_authorized: false`, `new_seed_opening: false`. The precedent for lifting it is recorded in the registry itself — block `g3_obs_v2_powered` carries:

```json
"authorization": {
  "granted_by": "PI, in session 2026-08-02",
  "note": "authority_ladder_v1 is status DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY: it neither
           prohibits nor authorizes. The authority is the PI, and the PI authorized."
}
```

and `docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md` §1 states the governance resolution. So the mechanism is: **per-block authorization recorded in the block, scoped to one contract.**

**What needs no signature:** all of Phase 0. Replays of 5,300,001–012 and 6,000,001–120 via `--replay-of`, cache construction, the normaliser re-measurement, the estimand change, the separability test on `garrido_continuous_space`, all synthetic and mutation tests, all comparator implementations, and RL *implementation*.

**What needs a signature, and it is three separate things — do not bundle them:**

1. **A development block for the extended surface.** `n = 12` replicates × 6 contexts × 4,608 configs. Recommend 8,100,001–8,100,012 (12 seeds), labelled `USED_DEVELOPMENT_NOT_VIRGIN` from the outset. This block is where the separability falsifier and all comparator tuning live. Development, explicitly not confirmation (`rules.development_is_not_confirmation: true`).
2. **A confirmation block, sized by a power calculation done on block 1.** The H3′ precedent is the template: 16 seeds gave MDE 0.0256 against SESOI 0.010, the PI authorized 140, and the powered rerun landed MDE 0.0092. Do the same here — do not guess `n` now. Reserve 8,200,001–8,200,xxx as `RESERVED_NOT_OPENED` and fill `xxx` after the power calculation.
3. **Neural training authorization.** `contracts/garrido_expanded_des_e_star_v1.json` → `seed_and_execution_policy.forbidden_before_receipt` lists *"PPO/MLP/KAN training"* and *"neural architecture selection"*, gated on the Submission-A receipt. An RL **search** policy over configurations is arguably outside E*'s scope (E* governs an intra-episode controller over the expanded DES kernel), but that reading is exactly the kind of self-serving scope call this repo has burned itself on — see `results/headroom/g3_obs_conversion_v2/contract_scope_adjudication.json`, `SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE`. **Get it in writing as a scope declaration, not as an inference.**

The PI signs a single document naming: the two seed ranges and their statuses; the primary estimand (uncensored regret, not `runs_to_oracle`); the SESOI; the reading rule (LCB95 > 0 → sustained); the frozen endpoint (`service_first_resilience_v2`); the comparator list *fixed in advance* — a comparator added after seeing results is post-hoc selection, which `cd_surface_prediction_premium/result.json` already discloses as `known_defect_baseline_selection`; and the explicit E* scope declaration for the RL search policy.

Also register block **4,100,001–4,100,384** (used by `garrido_continuous_space`) — it is currently absent from the registry, and I would not want a referee's reproducibility audit to find that before you do.

### (g) Manuscript

**Use neither manuscript as-is; use the Garrido–WRAP/v0 lane, rebuilt in the Elsevier LaTeX skeleton.**

The `docs/manuscript_current/submission/elsevier/` tree compiles, is `elsarticle`-ready, and has 17 publication-grade figures — but its title (*"When Is Reinforcement Learning Warranted in a Supply Chain DES?"*) and abstract are a **negative-result eligibility-framework paper about the inner loop**, with PPO −0.000018049 as the headline. That is the Program Q/O lane. Its *claims* are wrong for this paper; its *infrastructure* is exactly right. Take `main.tex`, `references.bib`, the `elsarticle` setup and `scripts/build_manuscript_figures.py`'s style block (Okabe-Ito, STIX serif, `save()` writing paired PDF + 300-dpi PNG to `docs/manuscript_current/submission/elsevier/figures`); replace `sections/`.

Section map, following `docs/CIE_MANUSCRIPT_REWRITE_MAP_GARRIDO_WRAP_2026-08-01.md` (which is the authoritative outline and is currently `HOLD_RESULTS_BEFORE_MANUSCRIPT_CLAIMS`):

| § | content | evidence | new figure |
|---|---|---|---|
| 1 | Garrido's Fig. 2 open loop; the loop is over runs, not timesteps | ICCL 2024 | reuse `fig17_control_loop` restyled: nodes ③→⑧ across *iterations* |
| 2 | related work: simulation-optimization, BO for DES, meta-learning across design points | — | — |
| 3 | the WRAP substrate; thesis-faithful vs extended DES boundary | `garrido_wrap_source_audit` | reuse `fig2_mfsc_topology`, `fig12_des_validation` |
| 4.1 | **Q1**: the AI family. Four independent contracts where structured control captured the value networks did not | the four `results/headroom/*` | new: forest plot of the four SESOI contrasts with CIs |
| 4.2 | **Q2 interface**: DES episode → observable outcome → retained state → next configuration | `run_garrido_q2_des288_v1.py` | new: the interface diagram, with the state/action boundary drawn |
| 4.3 | **Q2 evidence**: the comparator ladder on the extended grid | new runs | new: regret curves, all 10 arms, mean ± bootstrap band |
| 4.4 | separability / interaction of the design surface — *why* search is hard here and was not on the 288 grid | new | new: interaction-gain panel by context |
| 4.5 | transfer: seed / context / grid splits, with all five placebos | new | new: placebo ladder bar chart |
| 5 | limitations: censoring, the corrected normaliser, corner optimum on the original grid, the retired numbers | the `_2026-07-31` correction docs | new: claim-status table |

**Permitted** (verbatim from the ledgers): *"The present WRAP panel does not demonstrate a neural premium."* · *"H3′ sustained, n=120, LCB95 +2.3491, as a reduction in the variance of **search cost**"* (never "H3 proved") · *"Narrow support for H4: between-campaign memory measurably reduces the cost of finding a good configuration"* (never path-dependency of resilience) · *"The 90-configuration result validates replay/search logic, not independent DES behavior."*

**Prohibited until gates close:** *"Garrido's Figure 5 has been replicated as a valid planner"* · *"The retained learner improves Q2 in the corrected DES"* (this one is `PENDING_CONFIRMATION` in the claim ledger — H4) · *"H1 is a time-to-recovery result"* (H1 is `NO_EVALUABLE`, censored) · *"KAN/MLP/PPO is superior"* or any neural-premium claim · anything from G3-obs's positive number, which is `SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE` · the retired v1 meta-learner figures 7.24/12.42/13.54 and +6.31.

Note that `docs/GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md` currently marks `results/garrido_meta_learner_v2/result.json` as `ARTIFACT_PRESENT_CANONICAL_CUSTODY_PENDING` with the permission *"audit contract, hash, seeds, falsifiers and ledger before any figure"*. §2.1 is that audit, and it comes back with a finding.

`deliverables/v0_neuralNet-scres_DES_and_oracle_metric.docx` (§4.2 and §4.3 empty) should be regenerated from the LaTeX at the end via the existing `deliverables/*.py` port scripts, not authored in parallel — two sources of truth is how the two incompatible manuscripts happened.

### (h) Risk register

| # | risk | probability | detection | fallback |
|---|---|---|---|---|
| R1 | **The Alzheimer effect does not survive the honest normaliser** (§2.1) | medium | P0.2, day 1, zero cost | The paper becomes "a leak-audit of an outer-loop learner, and the corrected effect". Still publishable — the leak taxonomy across three runners is a real methodological contribution — but the PI must be told on day 2, not month 2. |
| R2 | **The extended grid is still separable** (§(e) `f_extension_is_non_separable`) | medium-high | after the ~7 h extended cache build | Then OFAT is near-optimal by theory, no search method can beat it by much, and the honest paper is *"when is a learner warranted between DES runs? — a separability criterion"*, with the criterion as the contribution and the negative as the result. This mirrors the existing Program Q paper's structure and is the safest fallback. |
| R3 | **BO beats RL** | **high** | comparator ladder, before any RL training | This is the likeliest single outcome and you should plan for it as the *primary* alternative, not the fallback. The paper then answers Q1 as *"the family that mimics SCL across runs is surrogate-based simulation-optimization, of which Garrido's Fig. 5 neuron is a degenerate linear special case"* — which **is** a direct, defensible answer to Garrido's Q1, names one of his own three candidates ("simulation-optimization approach as a form of reinforcement learning"), and is far more likely to survive review than "our RL won". RL then appears as the *transfer* arm: BO wins within a context, RL wins across contexts or across grids, and the paper's contribution is the boundary between them. |
| R4 | Censoring / estimand objection | high if unfixed | P0.3 | Uncensored primary; report `runs_to_oracle` as secondary with censoring rates disclosed per arm. |
| R5 | Seed authorization not granted, or granted late | medium | now | Phase 0 is fully unblocked and is ~40% of the work. Sequence it first regardless. |
| R6 | E* scope objection to RL training | medium | now | Get the written scope declaration before writing a line of RL code. |
| R7 | Physics drifts mid-project and invalidates the cache | low-medium | `f_cache_reproduces_the_simulator` | Rebuild is 7–20 h, not a catastrophe — but only if the drift is *detected*. Without the spot-recompute falsifier this is the failure that silently destroys the paper. |
| R8 | The transfer result is trivial because the optimum does not move | **high on the current grid, verified** (§2.3) | `f_optimum_moves_across_contexts` | If it also fails on the extended grid, describe the finding honestly: *"the optimal configuration is invariant to the risk regime; the value of between-run memory is the value of not re-deriving a constant"*. That is a real and citable finding — it is the outer-loop twin of the exhaustion certificate — but it is not "RL beat everything". |

**The one thing I would not do:** run the RL work before `f_extension_is_non_separable` and `f_optimum_moves_across_contexts` return positive on the cached extended surface. Both are cheap, both are computable from the cache, both are go/no-go, and both currently fail on the grid you have.

---

### Critical Files for Implementation

- `<HOME>/Projects/research/scres-ia/scripts/run_garrido_q2_des288_v1.py` — the platform to extend (surface build in `main()`, `evaluate()`, `search()`, `_falsifiers()`, the `PRIOR_SEEDS`/OFAT-`idx` defects)
- `<HOME>/Projects/research/scres-ia/supply_chain/arm_runner.py` — `seal_and_write`, `canonical_payload_sha256`, `VOLATILE_KEYS`, `run_falsifiers`: the cache's sealing and cross-version comparison primitives
- `<HOME>/Projects/research/scres-ia/supply_chain/seed_custody.py` — `custody_falsifier`, `check_seeds`, `module_manifest`: replay declaration and the anti-drift manifest
- `<HOME>/Projects/research/scres-ia/supply_chain/supply_chain.py` — `initial_buffers` handling at lines 1063-1080 and `_deliver_buffer_top_up` at 1467-1490 (Phase 1 grid extension); `_cssu_capacity_ledger` at 511-515 and the `admit()` call at 5085 (Phase 2 capacity wiring)
- `<HOME>/Projects/research/scres-ia/supply_chain/gsa.py` — `gp_locate`, the BO comparator to reuse (needs `n_init` exposure, history return, and grid snapping)
- `<HOME>/Projects/research/scres-ia/tests/test_cssu_capacity_bridge.py` — the golden-anchor + mutation-test pattern the null-arm identity test must follow
