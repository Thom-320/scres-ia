# Explore — In <HOME>/Projects/research/scres-ia, explore VERY THOROUGHLY the reusable experiment

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a9fc46ef4dee40981`
- **session**: `091a3905-ebf9-481b-beaa-fc646c3902d5`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-01T21:26:39.228Z`
- **effort**: `low` · **version**: `2.1.219`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/091a3905-ebf9-481b-beaa-fc646c3902d5/subagents/agent-a9fc46ef4dee40981.jsonl`
- **turns**: 79 · **assistant messages**: 3

## Task given

```text
In <HOME>/Projects/research/scres-ia, explore VERY THOROUGHLY the reusable experimental infrastructure I would need to build a new headroom experiment ("G3: three asymmetric claimants"). I want to REUSE, not rewrite.

Find and report with file:line:
1. `arm_runner.py` — its public interface, especially `seal_and_write`, how arms/cells are specified, how episodes are run, how metrics panels come back. What is the canonical way to run a sweep through the pipeline?
2. The virgin-seed scanner: `seeds_used_by_sealed_artifacts()` in scripts/run_h1_h3_v2.py — signature and how other runners import/copy it. What seed blocks are already burned? (grep results/**/result.json for "seeds")
3. The oracle/headroom measurement helpers: how is `H_regime` and `H_PI` computed in existing runners? Which script is the best template for a headroom gate with an uninformed placebo? (look at results/metric_audit/contention_service_first_v2/, docs/RESULTADO_CONTENCION_HEADROOM_2026-07-31.md, and any run_*headroom* script)
4. The classical-comparator ladder already implemented: where are linear_additive, linear_interactions, spline, CART, threshold_rule, backprop, KAN implemented? (scripts/run_cd_surface_prediction_premium.py and scripts/run_g2_autotomy_threshold.py). Are they in a shared module or duplicated per script? Is there a belief-MPC / DP / tabular-policy comparator anywhere in the repo?
5. supply_chain/cobb_douglas_resilience.py — CobbDouglasRecorder, derive_exponents, score_comparison_set signatures.
6. Contracts directory: contracts/*.json — the schema of a frozen contract (look at program_q_frozen_policy_replication_v1.json). What fields does a contract carry?

Report concisely with paths and signatures. Do not write any files.
```

## Final answer

## 1. `supply_chain/arm_runner.py` (194 lines) — the shared arm-comparison instrument

Public interface (all module-level functions; no classes):

| line | signature |
|---|---|
| `arm_runner.py:54` | `scored_orders(sim) -> list` — the *one* population (excludes `metrics_excluded`, `OPTj >= sim.warmup_time`) |
| `arm_runner.py:63` | `episode_moments(sim) -> dict[str,float]` — 6 fidelity moments, thesis-year basis (`THESIS_YEAR_HOURS = 8064.0`, `arm_runner.py:51`) |
| `arm_runner.py:118` | `aggregate(rows: Sequence[Mapping[str,float]], reference: Mapping[str,MomentReference]) -> dict` — mean/SE/`d_k` for one arm×family cell |
| `arm_runner.py:135` | `verdict(cells, epsilons=EPSILON_BAND) -> dict` — non-dominated set + epsilon stability (never a winner, never `sum(d_k)`) |
| `arm_runner.py:151` | `build_reference(blob, family) -> dict[str, MomentReference]` |
| `arm_runner.py:158` | `seal_and_write(payload: dict, path: Path, *, contract: Path, reference: Path, stamp_extra: Mapping|None = None) -> str` |
| `arm_runner.py:182` | `run_falsifiers(checks: Mapping[str, Callable[[], tuple[bool, Any]]]) -> dict` — stores `{passed, evidence}` per check plus `all_passed` |

`seal_and_write` (the only thing most new runners import) adds: `created_at`, `contract_path` + `contract_sha256`, `reference_path` + `reference_sha256` (read from the reference file's own `self_sha256`), `calibration_provenance` (from `supply_chain.provenance.calibration_stamp`), then computes `self_sha256` over `json.dumps(payload, indent=1, sort_keys=True, default=str)` **without** the `self_sha256` key, and writes with the same serialization. Reproducing that digest externally is a standard falsifier (`run_h1_h3_v2.py:157-160`).

**Important:** arm_runner does *not* own arms, cells, sweeps, or episode running. There is no `run_sweep`. The canonical pattern each runner reimplements in its own `main()`:

1. module constants: `ARMS` / `FAMILIES` / `ESCALATIONS` / grid tuples, `SEED_BASE`, `LEADING`/`TARGET` metric name;
2. a local `episode(...) -> dict[str,float]` that builds `MFSCSimulation(...)`, `sim.run()`, `compute_episode_metrics(sim)` and returns a flat float dict — see `run_contention_service_first_v2.py:80-102` (best short example) and `run_g2_autotomy_threshold.py:~95-117`;
3. nested loops filling `rows[(cell, arm)] = [episode(..., s) for s in seeds]` (with a `cache` keyed on config when arms can pick identical configs, `run_h1_h3_v2.py:165-177`);
4. numpy cube + bootstrap over seeds/cells;
5. `falsifiers = {...}` dict of `{passed, evidence.why_it_can_fail, ...}`;
6. verdict string, Spanish console print, `seal_and_write(payload, args.output, contract=Path("docs/PREREGISTRO_*.md"), reference=Path(...))`, `return 0 if falsifiers["all_passed"] else 1`.

Metrics panels come from `supply_chain.episode_metrics.compute_episode_metrics(sim)` (a flat dict); claimant-level fill from `supply_chain.service_first_metric.claimant_fills(sim)`.

## 2. Virgin-seed scanner

`scripts/run_h1_h3_v2.py:65`:
```python
def seeds_used_by_sealed_artifacts(root: Path = Path("results"),
                                   exclude: Path | None = None) -> set[int]
```
Walks `results/**/result.json`, collecting any list under keys `{"seeds","crn_seeds","seed_block"}`; lists are truncated to first 50 elements when recursing; `exclude` skips the artifact the run is about to overwrite (resolved-path compare).

It is **copy-pasted verbatim into 6 runners, not shared**: `run_h1_h3_v2.py:65`, `run_cd_surface_prediction_premium.py:71`, `run_contention_service_first_v2.py:52`, `run_g1_buffer_price_cobb_douglas.py:54`, `run_buffer_profile_prediction_premium.py:60`, `run_g2_autotomy_threshold.py:69`. Call site is always `prior_seeds = seeds_used_by_sealed_artifacts(exclude=args.output)` feeding falsifier `f5`/`f7` `seeds_are_virgin`. **This is the single most obvious thing to hoist into `arm_runner` for G3.**

### Burned seed blocks (scan of `results/**/result.json`, 406 distinct seeds)

| block | consumer |
|---|---|
| 4_800_001–05 | `results/sensitivity/perrisk_crn_v1` |
| 5_100_001–12 | backlog_confirmation_v1 |
| 5_200_001–16 | contention_headroom_v1 / v1_1 / v1_2 / v1_3 |
| 5_300_001–12 | garrido_meta_learner (+v2, +vps) |
| 5_500_001–08 | metric_audit/abandonment_v1 |
| 5_600_001–08 | headroom/cobb_douglas_v1 |
| 5_700_001–10 | manuscript/h1_h3_v1 |
| 5_800_001–08 | sensitivity/expedite_headroom_v2 (+5_800_101–102 smokes); also `manuscript/h1_h3_v2` 5_800_001–05 — the known collision |
| 5_910_001 | meta_learner_smoke_v2 |
| 6_000_091–120 | meta_learner_h3power_vps |
| 6_200_001–05 | manuscript/h1_h3_v2_1 |
| 6_300_001–06 | metric_audit/service_first_v2 |
| 6_400_001–10 | metric_audit/contention_service_first_v2 |
| 6_500_001–03 | des_change_differential, vps_vs_local_differential |
| 6_600_001–06 | cssu_liveness_scope |
| 6_700_001–06 | g1_buffer_price, g1_ablation_ci_zero |
| 6_800_001–10 | headroom/buffer_prediction_premium |
| 6_900_001–08 | headroom/cd_surface_prediction_premium |
| 7_000_001–08 | headroom/g2_autotomy_threshold |
| 7_400_025–7_450_006 | program_o family (222 seeds: 7_400_*, 7_410_*, 7_420_*, 7_430_*, 7_450_*) |
| 7_600_001–10 | sensitivity/bottleneck_asymmetry_v1 |

Additionally declared-but-unscanned reservations: `6_710_001–6_710_120` / `6_720_001–6_720_300` (`run_k3_strong_mpc.py:18-19`, hardcoded ranges, **not** in a `seeds` list so the scanner misses them), `7_490_001–7_490_256` (program_q confirmation, reserved unopened, `contracts/program_q_frozen_policy_replication_v1.json`), `7_510_001–12` (program_s native), `949_1xx_xxx` sandbox blocks.

**Free for G3:** `7_100_001`+ is claimed by `run_garrido_q2_des288_v1.py:67` (not sealed yet), `7_200_001+` flagged as opened-unledgered (`verify_paper2_exhaustion.py:1123`). The clean choice is a fresh decade, e.g. **`7_700_001+`** or `7_800_001+` — nothing at 7_7/7_8/7_9 anywhere.

## 3. Headroom / oracle measurement

Two distinct estimands, both implemented inline per script (no shared helper):

**`H_regime` = mean_r[max_a f(a,r)] − max_a[mean_r f(a,r)]** — value of knowing the regime.
- Definition in the docstring: `scripts/run_mixed_risk_downstream_headroom.py:13`; bootstrap helper at `:82`, returns `{"H_regime", "lcb95", "ucb95"}` (`:103`).
- Cleanest implementation: `scripts/run_contention_service_first_v2.py:137-153` — builds a `(regimes, actions, seeds)` cube, `stat(idx)` = `sub.max(axis=1).mean() - sub.mean(axis=0).max()`, 5000 bootstrap resamples of seeds with `np.random.default_rng(20260801)`.
- Also `run_second_order_and_risk_search.py:144-172`, `run_headroom_gate.py` (buffer-grid version, `scripts/run_headroom_gate.py:1-18` docstring gives the full recipe: per-intensity best constant vs. best single constant, bootstrap LB > 0).

**`H_PI` (perfect information) and `H_obs`**: `scripts/run_program_g_screen.py:53-57` —
```python
H_PI, H_obs = static - oracle, static - obs
eta = H_obs.sum() / max(H_PI.sum(), 1e-9)
passes = bool(obs_lo > 0 and eta >= 0.30 and m_obs <= m_stat + 0.2 and boot_lo(H_PI) > 0)
```
i.e. `H_PI` = static baseline − clairvoyant oracle, `H_obs` = static − best *observable* policy, `eta` = realizable fraction. This is the closest thing to a reusable headroom-gate formula.

**Best template for a headroom gate with an uninformed placebo:**
- Gate spec: `docs/PREREGISTRO_NEURAL_HEADROOM_ENV_V1_2026-08-01.md` — 8 simultaneous conditions (`H_obs >= 0.01`, `LCB95(H_obs) > 0`, **placebo does not capture the gain**, argmax moves, action physically live, best classical control does not reach the oracle, no gain from abandonment, disjoint calibration/eval seeds) and the model ladder `constante → umbral → regla finita → lineal/logística → árbol/tabular → MLP → PPO → PPO recurrente`. Its E1 environment is *literally* "CSSU split, two claimants, non-fungible" and its sealed result is `H_regime = 0` on the leading component — G3 is the direct successor.
- Runner template: `scripts/run_contention_service_first_v2.py` (falsifiers f1–f6, incl. positive-control + injected-defect pair for the lexicographic ranking at `:158-185`).
- Placebo/adjudication: `scripts/adjudicate_neural_headroom_gate_v1.py:21-46` (`placebo_required: True`, reads `source["placebo"]["status"] == "PASS"`); a worked real-vs-placebo-anchor headroom is `scripts/audit_prevention_headroom_sweep.py:28, 248-287` (`headroom = (max_prep − calm | real anchors) − (max_prep − calm | placebo anchors)`).
- Prior result to build on: `docs/RESULTADO_CONTENCION_HEADROOM_2026-07-31.md` §4 names **exactly your experiment**: the two differences from Program O are (1) contention downstream of the bottleneck, (2) Program O's claimants were **asymmetric** whereas ours are symmetric by construction via 50/50 hashing (`stable_cssu_destination`). §4b documents the U-shaped surface, moving argmax, and `H_regime = 1.5e-4` (65× below the 0.01 bar).

## 4. The classical-comparator ladder — duplicated, not shared

Two partly-overlapping ladders, each defined inline in its own script:

`scripts/run_cd_surface_prediction_premium.py` — `models = ("constant","linear_additive","linear_interactions","spline_buffer","tree","train_cell_mean_comparator","backprop","kan")` (`:193`); `r2` `:126`, `ols` `:131`, `spline_features` (piecewise-linear, knots 336/672/1008) `:~215`, `tree_predict` (hand-rolled CART, depth 4, min-leaf 4) `:~225`, `classical = ("linear_additive","linear_interactions","spline_buffer","tree")` `:268`, `PRIMARY_BASELINE = "linear_interactions"` `:282`.

`scripts/run_g2_autotomy_threshold.py` — `models = ("constant","linear_additive","linear_interactions","threshold_rule","train_cell_mean_comparator","backprop","kan")` (`:207`); `r2` `:129`, `ols` `:135`, **`threshold_rule(x_tr, y_tr, x_te)` `:142`** (single best `if` over one feature/one cut, min 5 per side), `PRIMARY_BASELINE = "linear_interactions"` `:62`.

`r2` and `ols` are byte-identical duplicates across the two files. The **only** genuinely shared pieces are the neural ones, imported from a script (not a package):
```python
from build_garrido_fig5_surrogate import fit_kan, fit_mlp, grouped_folds
```
at `run_cd_surface_prediction_premium.py:48`, `run_g2_autotomy_threshold.py:41`, `run_buffer_profile_prediction_premium.py:40`; `run_continuous_space_fig5.py:33` imports `evaluate, grouped_folds`. Signatures used: `fit_mlp(x_tr, y_tr, x_te, seed=..., classify=False) -> (pred, info)`, same for `fit_kan` (wrapped in `try/except` → NaN, KAN may be unavailable). `grouped_folds(g, n_folds=...)` gives seed-grouped CV folds. `T_CRIT` dict for paired t on folds.

Leakage guard worth copying: `target_from_training(train_idx)` in `run_cd_surface_prediction_premium.py:~178` freezes Cobb-Douglas exponents and the `kappa_dot` normaliser on **training rows only** — because both are set-relative, computing the target over all rows leaks the test fold into its own label.

**Belief-MPC / DP / tabular comparator:** no belief-state MPC, no value iteration, no POMDP solver anywhere. The closest is `scripts/run_k3_strong_mpc.py` — a *calibrated open-loop/feedback grid search*, not real MPC: `periodic_static` (best of `periodic_calendars(4)`), `budgeted_sS` (`sS_policy(s,S)` over a 0–2 / 0.5–3 grid), `inventory_paced` and `strong_mpc` (both `paced_policy(alpha,beta,gamma)` — `strong_mpc` just searches a 3-parameter grid where `inventory_paced` fixes `alpha=0`). Policies live in `supply_chain/replenish_ret.py` (`paced_policy`, `sS_policy`, `periodic_calendars`, `rollout_policy`, `rollout_actions`, `BUDGET_D0`, `WEEKS`) with tapes from `supply_chain/replenish.py` (`materialize_tape`, `central_cell`). Calibration seeds 6_710_001–120, test 6_720_001–300, disjoint. `train_cell_mean_comparator` (in both ladders) is the tabular/oracle-cell comparator. Other classical families: `scripts/run_expanded_contract_comparators_v2.py` + `contracts/expanded_contract_comparators_v2.json`, and `program_o_state_rich_comparator_fit_v1` (10 frozen state-rich configurations).

## 5. `supply_chain/cobb_douglas_resilience.py`

```python
validate_costs(costs: Mapping[str,float]) -> dict[str,float]                       # :86
kappa_from_components(...)                                                          # :100
derive_exponents(maxima: Mapping[str,float]) -> dict[str,float]                     # :146
conditioning(maxima: Mapping[str,float]) -> dict[str, dict[str, float|bool]]        # :166
assert_terms_bounded(exponents, ...)                                                # :186
resilience_index(components: Mapping[str,float], exponents) -> dict                 # :209  -> {"R_cobb_douglas", ...}
kappa_dot(kappa_by_policy: Mapping[str,float]) -> dict[str,float]                   # :226

@dataclass
class CobbDouglasRecorder:                                                          # :241
    period_hours: float = 24.0
    costs: Mapping[str,float] = field(default_factory=lambda: dict(UNIT_COSTS))
    def reset(self) -> None                                                         # :275
    def sample(self, sim) -> dict[str,float]                                        # :306
    def aggregate(self) -> dict[str,float]                                          # :345

score_comparison_set(aggregates: Mapping[str, Mapping[str,float]],
                     exponents: Mapping[str,float]) -> dict[str, dict[str,float]]   # :377
maxima_over(aggregates: Iterable[Mapping[str,float]],
            kappa_dots: Sequence[float]) -> dict[str,float]                         # :402
```
- `derive_exponents`: Garrido's rule `exponent = 0.20/ln(x_max)` over `VARIABLES = zeta, epsilon, phi, tau, kappa_dot`; **raises** if any `x_max <= 1.0`. `SIGNS` at `:140`: zeta/phi `+1`, epsilon/tau/kappa_dot `−1`.
- `CobbDouglasRecorder` is external to the DES (reads public attrs, writes nothing). Usage: construct with `period_hours=STEP, costs=dict(UNIT_COSTS)`, then loop `sim.step(step_hours=STEP); recorder.sample(sim)`; `aggregate()` returns `zeta/epsilon/phi/tau/kappa` (raw) plus `mean_*` diagnostics and `T_periods`. Example: `run_cd_surface_prediction_premium.py:~110-125`.
- `score_comparison_set` is **set-relative** — every policy affects every other through `kappa_dot`, so the comparison set must be declared before evaluation. `aggregate()` alone cannot produce an R.

## 6. Contract schema — `contracts/program_q_frozen_policy_replication_v1.json`

Free-form JSON (no JSON-Schema validator in repo), but a stable idiom. Top-level fields:

- **identity**: `contract_id`, `schema_version` (mirrors the id), `date_created`, `status` (e.g. `FROZEN_POWER_PASS_N_256_PENDING_SEED_AUTHORIZATION`), `question` (one sentence).
- **immutability**: `historical_verdicts_immutable` (prior verdicts + scientific commit SHAs), `unchanged_contract` (physics, `cells`, actions, observation, reward, `forbidden_information`).
- **candidate/arms**: `primary_candidate` with `learner_seeds`, `checkpoints` (SHA-256-frozen file), `retraining: forbidden`, `replacement_by_external_challenger: forbidden`.
- **seed custody**: `architecture_sidecar_firewall` (sandbox seed ranges + `SANDBOX_DEVELOPMENT_ONLY_NOT_PROMOTABLE`), `confirmation` = `{reserved_block: [lo, hi], opened: false, N, same_tapes_all_cells_crn, opening_requires: [...], preopening_gate_state: {...PASS/PENDING}}`.
- **power**: `power` = `{script, candidate_N, minimum_joint_power, failure verdict, authoritative_result: {run_id, source_commit, selected_N, joint_power, result path + result_sha256, cache_sha256, independent_audit + sha}}`.
- **comparators**: `comparators` = `{open_loop, classical, replacement_controls, reselection_inside_bootstrap}`.
- **inference**: `inference` = `{resampling_units, method (studentized simultaneous max-t), all_three_cells_simultaneous, H_OL / Delta_N definitions, adaptation_rule, neural_premium_rule, equivalence_rule, favorable_tapes_fraction_minimum, learner_seeds_H_OL_positive_minimum}`.
- **gates & outcomes**: `class_b_integrity_gates` (per-metric margins), `secondary_nonblocking`, `terminal_verdicts` (enumerated verdict strings), `paper3_authorized`, `prospective_amendments_before_scientific_seed_opening`.

Lighter contracts (e.g. `contracts/program_g_structured_spatial_headroom_v1.json`, `contracts/program_i_headroom_sensitivity_v1.json`, `contracts/cssu_allocation_v1.json`) keep the same spine minus the power/custody machinery. Note that some runners pass a **markdown preregistro** as the `contract=` argument to `seal_and_write` (`run_contention_service_first_v2.py:~285` uses `docs/PREREGISTRO_METRICA_SERVICE_FIRST_V2_2026-08-01.md`) — the seal only needs a hashable file.

## Blocker you'll hit for "three asymmetric claimants"

The DES hardcodes **two** claimants: `supply_chain/supply_chain.py:923-924`
```python
self.cssu_delivered = {"A": 0.0, "B": 0.0}
self.cssu_demanded  = {"A": 0.0, "B": 0.0}
```
with `cssu_topology_mode` restricted to `{"aggregate","split_v1"}` (`:418-421`), a scalar `cssu_allocation_a: float = 0.50` validated by `validate_allocation_a` (`:422`), `"A"/"B"` membership checks at `:3568`, and `stable_cssu_destination(*, simulation_seed: int, order_id: int) -> str` (`supply_chain/cssu_allocation.py:42`) returning a 50/50 hash. `service_first_metric.claimant_fills` and `service_first_key_v2` are already N-claimant-generic (`min(fills.values())`), and `f3_claimant_partition_exists` in the v2 runner asserts `claimants == {2}` — that falsifier becomes `== {3}`. So G3 needs a `split_v2` topology (3 buckets, a 2-simplex allocation, asymmetric demand weights in the destination hash) — that's the one piece of genuinely new physics; everything else above is reusable as-is.
