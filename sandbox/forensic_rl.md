# Forensic RL Audit — `MFSCGymEnvShifts` and Reward Layer

**Date:** 2026-06-19
**Scope:** Reward hacking, observation leaks, normalization bugs, action-decoder correctness, priming/eval-window integrity, and static-vs-learned comparison fairness.
**Method:** Read-only forensic pass over `supply_chain/env_experimental_shifts.py` (2781 LOC), `supply_chain/ret_thesis.py` (74 LOC), `supply_chain/external_env_interface.py` (930 LOC), `supply_chain/supply_chain.py` (1894 LOC), `supply_chain/config.py`, plus the benchmark/train scripts that drive the headline PPO numbers.
**Stance:** Skeptical. Every "audit" doc in `docs/` is treated as a claim, not as truth.

---

## TL;DR (verdict tally)

| Verdict | Count |
|---|---:|
| `BUG` | 0 |
| `REWARD_HACKING_RISK` | 3 |
| `RIGGED_COMPARISON` | 2 |
| `AMBIGUITY` | 5 |
| `EXTENSION` | 4 |
| `MATCH` | 6 |

**Bottom line:** No outright bug corrupts the DES during RL training. The env API is Gymnasium-compliant; obs and reward are deterministic functions of (state, action, RNG). **The integrity threats are at the comparison layer, not the dynamics layer**: the headline PPO benchmark runs on a 5-year window with priming on and Op3/Op9 inventory multipliers centered at 1.25×, while Garrido's thesis numbers come from a 20-year window with priming off and 5 discrete inventory levels. The repo *can* produce an apples-to-apples comparison via `benchmark_control_reward.py` with the `garrido_cf_sN` columns — but the README headline `train_agent.py` command does not compute that comparison at all.

---

## A) Observation leaks

### Finding A1: Base observation v1 is clean (current state only)
- **Claim:** "Observation values are normalized continuous features emitted by the shift-control environment." (`external_env_interface.py:333`)
- **Code does:** `supply_chain.py:748-793` `get_observation()` returns 15 dims: 6 inventory levels (current `.level`), 2 rates (current fill/backorder), 4 binary down-flags (current `_is_down`), and 3 calendar/queue scalars (current `env.now`, current pending batch, current contingent demand). All current-state. No future info.
- **Verdict:** `MATCH`
- **Impact:** None. v1 (the README headline obs version) is leak-free.
- **Evidence:** `get_observation()` does not read pending risk-event schedule, RNG state, or future demand draws.

### Finding A2: Observation v6 contains forward-looking forecasts (Track-B only)
- **Claim:** `external_env_interface.py:339` "v6 extends v5 with Track-B adaptive benchmark features: regime state, imperfect disruption forecasts, maintenance debt, backlog age, and theatre cover days."
- **Code does:** `supply_chain.py:849-878` `get_observation_v6_extra()` exposes `adaptive_risk_forecast_48h_norm` and `adaptive_risk_forecast_168h_norm`, computed in `_update_adaptive_forecasts()` (`supply_chain.py:1015-1032`) as `expected_intensity + self.rng.normal(0, noise_std)`. The expected intensity is computed deterministically from the observable regime + transition matrix; the noise is drawn from `self.rng` — **the same RNG that drives actual risk events**.
- **Verdict:** `REWARD_HACKING_RISK`
- **Impact:** Two issues. (1) The forecast is a noisy but informative model-based prediction — by itself legitimate (it is what an operator would actually have). (2) The noise draw consumes RNG state, which creates non-trivial stochastic coupling between the forecast signal and the very events it forecasts. A policy with enough capacity could in principle learn to extract RNG-state info from the forecast noise sequence and partially anticipate disruptions — exactly the "anticipation oracle" pattern that invalidates resilience claims. Mitigating factor: **the headline benchmark uses obs v1 or v4, never v6**, and v6 is gated to `risk_level in {adaptive_benchmark_v1, v2}`. So this risk only materializes for Track-B research runs, not the paper-facing PPO lane.
- **Evidence:** `supply_chain.py:1024` `forecast_48h = next_intensity + float(self.rng.normal(0.0, noise_std))`. The same `self.rng` is used by the risk-event schedule.

### Finding A3: v2/v3/v4/v5/v7 augmentations are honest
- **Claim:** Each obs version is described in `external_env_interface.py:333-348` as adding past-step, cumulative-past, current, or calendar features.
- **Code does:** `_compose_observation()` (`env_experimental_shifts.py:981-1021`) appends `self._prev_step_*` (lagged one step, set at end of previous `step()`), `_normalized_cumulative_features()` (running stats since warmup), `get_observation_v4_extra()` (current shift + op1/op2 flags), `get_observation_v5_extra()` (deterministic calendar phases of `env.now`), `get_observation_v7_extra()` (rolling 4-week service). All are past-or-present, no oracle.
- **Verdict:** `MATCH`
- **Impact:** None for obs v1–v5, v7.

### Finding A4: Observation `Box(low=0, high=20)` under-bounds inventory in extreme states
- **Claim:** `env_experimental_shifts.py:663-665` declares `observation_space = Box(low=0.0, high=20.0, ...)` with the comment that "inventory dims rarely exceed ~10 in practice."
- **Code does:** The Op3 raw-material buffer can legitimately hold ≥1.5M units under sustained R3 black-swan recovery (ζ_max ≈ 3.17M per `data/ret_garrido2024_calibration.json:7`). Divided by 1e6, that is obs value ≈ 3.17 — under the 20 bound, OK. But under 2× inventory multipliers + R21 overlapping at Op3, transiently the normalized value can drift above the comment's "~10" and the space is correctly bounded at 20, so the Gymnasium contract is technically satisfied. The real risk is **VecNormalize running-stats pollution**: a single 1-in-10ⁿ episode with a buffer spike will skew the running mean used to normalize the obs for PPO.
- **Verdict:** `AMBIGUITY`
- **Impact:** Minor. PPO with `clip_obs=10.0` (train_agent.py:374) bounds the damage; benchmark_control_reward.py runs without VecNormalize so is unaffected.

---

## B) Reward design

### Finding B1: Reward is path-dependent via `cumulative_demanded_post_warmup` (legitimate)
- **Claim:** `ReT_seq_v1` docstring (`env_experimental_shifts.py:1294-1305`) says BC_t uses "pending backorder stock relative to cumulative demand ... intentionally measures overall recovery health across the episode."
- **Code does:** `_compute_ret_seq_v1` (`env_experimental_shifts.py:1352-1389`) computes `bc_t = 1 - min(1, pending_bo_qty / cumulative_demanded_post_warmup)`. `cumulative_demanded_post_warmup` is a running accumulator initialized in `reset()` (`env_experimental_shifts.py:2136`) and incremented by `sim.total_demanded`. Same logic for `ReT_unified_v1`, `ReT_ladder_v1`, `ReT_tail_v1`, `ReT_garrido2024`. The path-dependence is over **past** cumulative state, not future. It is the standard way to turn a flow (orders) into a recovery-ratio proxy when step-level RP_j isn't observable.
- **Verdict:** `MATCH`
- **Impact:** None. This is honest path-dependence, not a leak.

### Finding B2: `ReT_seq_v1` (the FROZEN benchmark reward) has NO inventory holding cost → over-buffering reward hack
- **Claim:** README:158 "Training reward: `ReT_seq_v1`". `env_experimental_shifts.py:1307-1313` defines AE_t = `1 - κ(S-1)/2` as the only cost term, with κ=0.20.
- **Code does:** `_compute_ret_seq_v1` rewards SC_t (fill rate) and BC_t (low pending backorders), and penalizes ONLY shifts via AE_t. The agent controls Op3/Op9 inventory via dims 0-3 of the action vector, and these multipliers cost **nothing** in the reward. So the unconstrained optimum is: hold the maximum allowed buffer (a1=a2=+1 → 2.0× Op3 and Op9 inventory) AND pick S=1 (cheapest shift). Large buffers + S=1 is exactly the policy that maximizes SC_t and BC_t at minimum shift cost. The DES partially constrains this via upstream throughput, but the reward function itself provides no gradient against over-buffering.
- **Verdict:** `REWARD_HACKING_RISK`
- **Impact:** Substantial for the paper's claim. The very document that introduced `ReT_tail_v1` (`docs/RET_TAIL_V1_TUNING_2026-06-17.md:6-9`) states verbatim: *"ReT_tail_v1 replaces the Track A training reward that was too aligned with mean service performance."* And `env_experimental_shifts.py:139-143` admits the prior reward caused PPO to *"learn the worst-tail policy"*. So the repo authors already know ReT_seq_v1 has this failure mode. Yet README:158 still lists `ReT_seq_v1` as the default. **Any "PPO beats static" claim made under ReT_seq_v1 is suspect of being a buffer-maximization artifact, not a resilience improvement.**
- **Evidence:** `env_experimental_shifts.py:1368-1369`: `ae_t = max(EPS, 1.0 - self.ret_seq_kappa * (shifts - 1) / 2.0)` — no inventory term. Compare to `_compute_ret_tail_v1:1522-1532` which DOES include `cap_ef_t * inv_ef_t` under an un-gated `ce_t` term.

### Finding B3: `ReT_garrido2024` κ̇ normalization ≠ paper's `7κ/Σκ`
- **Claim:** `docs/RET_GARRIDO2024_IMPLEMENTATION.md:42-44` claims κ̇ is "Average operational cost normalized by a Monte-Carlo reference cost."
- **Code does:** `env_experimental_shifts.py:1848` `kappa_dot = max(eps, average_cost / max(self.ret_g24_kappa_ref, eps))`. The paper (Garrido 2024 Eq. 3) defines κ̇ = `7κ(S_ij)/Σκ(S_ij)` — a per-substrategy cost relative to the SUM of all 7 substrategies' costs (bounded, mean≈1). The repo replaces this with a Monte-Carlo reference cost `kappa_ref = 1.50M` (from `data/ret_garrido2024_calibration.json:5`). This is a **different quantity**, and since `n_kappa=0.249` is the largest exponent (line 17 of the calibration file), mis-scaling κ̇ distorts the index the most.
- **Verdict:** `EXTENSION`
- **Impact:** Documented as an adaptation (`docs/RET_GARRIDO2024_AUDIT_2026-06-18.md:69-73`, finding F3). The audit explicitly says: "Since n=0.25 is the LARGEST exponent, κ̇ mis-scaling distorts the index most." The recent re-calibration (audit doc:16-29, "RESOLVED 2026-06-18") reports Δ < 0.002 on the four non-cost exponents, so the practical impact is small. But the paper-facing index is **not** the paper's exact index — it is a related quantity. Must be disclosed, not silently equated.

### Finding B4: Step-level `ReT_thesis` is NOT the thesis's order-level ReT
- **Claim:** `env_experimental_shifts.py:53-63` calls `_compute_ret_thesis_components` a "step-level approximation" of Garrido-Rios 2017 Eq. 5.5.
- **Code does:** The thesis Eq. 5.5 (`supply_chain/ret_thesis.py:11-39`, `compute_ret_per_order`) selects one Re-sub-indicator PER ORDER based on whether that order's CTj falls inside APj, RPj, or (DPj-RPj). The env's `_compute_ret_thesis_components` (`env_experimental_shifts.py:1027-1098`) instead classifies a whole 168-hour STEP into one of four cases using step-aggregate fill_rate and step-aggregate disruption_fraction. These are different mathematical objects: the thesis quantity is per-order and uses order-level cycle-time vs lead-time; the env quantity is per-step and uses backorder-quantity vs demanded-quantity.
- **Verdict:** `AMBIGUITY`
- **Impact:** The env docstring is honest that this is "NOT suitable as training objective" and that the thesis-exact version is only computed at episode end via `sim.compute_order_level_ret()` (`supply_chain.py:1880-1894`). But if any figure in the paper plots the step-level `ret_thesis_step` and labels it "thesis ReT," that label is wrong. Use only `compute_order_level_ret()` for thesis comparison.

### Finding B5: All reward modes are bounded in [0, 1] except `control_v1` (and `rt_v0`)
- **Claim:** The Cobb-Douglas rewards are documented as "proper weighted geometric mean in (0, 1]" (`env_experimental_shifts.py:1972-1973`).
- **Code does:** `ReT_seq_v1`, `ReT_unified_v1`, `ReT_ladder_v1`, `ReT_tail_v1`, `ReT_cd_v1`, `ReT_garrido2024_*` all multiply sub-indicators in (0, 1] with weights summing to 1.0, so the product is in (0, 1]. `control_v1` (`env_experimental_shifts.py:1138-1163`) is `-(w_bo·service_loss + w_cost·shift_cost + w_disr·disruption_frac)` — unbounded negative. With README defaults w_bo=4.0, w_cost=0.02, an episode of 260 steps with average fill_rate 0.8 gives reward ≈ -200, comparable across PPO and static. Different reward families should NOT be cross-compared by their raw numbers.
- **Verdict:** `MATCH`
- **Impact:** Reward-magnitude confusion across families is a presentation hazard, not a correctness bug.

### Finding B6: PBRS variant is potential-based (Ng et al. 1999), policy-invariant
- **Claim:** `env_experimental_shifts.py:2040-2062` says PBRS uses Φ(s) = α·fill_rate − β·backorder_rate.
- **Code does:** The shaping bonus is `pbrs_shaping_bonus = γ·Φ(s') - Φ(s)` (`env_experimental_shifts.py:2452`). This is the canonical potential-based shaping form that preserves the optimal policy (Ng, Harada, Russell 1999). No leak, no policy distortion.
- **Verdict:** `MATCH`

---

## C) Action decoder

### Finding C1: Inventory multiplier for dims 0-3 is centered at 1.25×, NOT 1.0×
- **Claim:** `THESIS_FIDELITY_AUDIT.md:110` and `env_experimental_shifts.py:45` state `m = 1.25 + 0.75·a → [0.5, 2.0]`.
- **Code does:** `env_experimental_shifts.py:2332` `multipliers_q = 1.25 + 0.75 * clipped[:4]`. So `a=0` (the neutral / zero-action output) gives `m=1.25`, which is **25% above the thesis-faithful Op3/Op9 default**. The agent must output `a = -1/3 ≈ -0.333` to recover the thesis 1.0× value.
- **Verdict:** `AMBIGUITY`
- **Impact:** Two consequences. (1) `static_s1/s2/s3` baselines (`benchmark_control_reward.py:47-49` use `[0,0,0,0,0,±1]`) inherit this bias: they run with **Op3 inventory = 1.25 × 15,500 = 19,375**, Op9 dispatch range = `[3000, 3250]`, Op3 ROP = 210h (not 168h), Op9 ROP = 30h (not 24h). They are NOT thesis-faithful S1/S2/S3. (2) PPO trained with reward centered around zero-action (which is the typical Gaussian policy initialization) starts life at 1.25× inventory — biased toward over-buffering from step 0.

### Finding C2: Op5 multiplier IS centered at 1.0× — asymmetric with Op3/Op9
- **Claim:** `env_experimental_shifts.py:46-47`, `THESIS_FIDELITY_AUDIT.md:111`: dim 4 (Op5) uses `m = 1.0 + 0.5·a → [0.5, 1.5]`.
- **Code does:** `env_experimental_shifts.py:2335` `op5_multiplier = 1.0 + 0.5 * float(clipped[4])`. Centered at 1.0, range [0.5, 1.5].
- **Verdict:** `AMBIGUITY`
- **Impact:** The action decoder treats Op3/Op9 (range [0.5, 2.0], center 1.25) differently from Op5 (range [0.5, 1.5], center 1.0). This is undocumented in THESIS_FIDELITY_AUDIT.md (which presents both rows as if symmetric). It also means the policy network sees an asymmetrically-shaped action space: dim 4 has half the dynamic range of dims 0-3. Not a bug, but it should be disclosed.

### Finding C3: Shift decoder uses wide S=2 band (`[-0.33, +0.33]`)
- **Claim:** `env_experimental_shifts.py:48-50` and `external_env_interface.py:328-331` describe the three bands.
- **Code does:** `env_experimental_shifts.py:2338-2344`: `signal < -0.33 → S=1`, `-0.33 ≤ signal < 0.33 → S=2`, `signal ≥ 0.33 → S=3`. The S=2 band spans 0.66 of the action range, vs 0.33 each for S=1 and S=3.
- **Verdict:** `MATCH`
- **Impact:** The asymmetry is reasonable (S=2 is the Garrido middle case and gets the largest basin). PPO can still reach S=1 or S=3 deterministically. No bug.

### Finding C4: Dict-action bypass exists for "Garrido-faithful" static baselines
- **Claim:** `external_env_interface.py:411` "Use direct DES action dictionaries for static Garrido baselines when comparing S1/S2/S3 without multiplier artifacts."
- **Code does:** `env_experimental_shifts.py:2288-2298` accepts a dict action that bypasses the multiplier decoder entirely; `benchmark_control_reward.py:50-76` `garrido_cf_s1/s2/s3` use this path with direct `CAPACITY_BY_SHIFTS[S]["op3_q"]` etc.
- **Verdict:** `EXTENSION`
- **Impact:** **Two structurally different "static S=1/2/3" baselines coexist** in the same script: `static_sN` (goes through 1.25× multiplier) and `garrido_cf_sN` (direct DES, thesis-faithful). They are NOT numerically equivalent. The paper must specify which one it is comparing PPO against. The `static_sN` baselines should not be labeled "thesis-style."

---

## D) Priming / eval window

### Finding D1: Env default `priming_enabled=True`; PPO benchmark uses default; thesis-faithful runner does not
- **Claim:** `DIVERGENCE_FIX_PLAN.md:9,16-25` admits: "the env RL (`MFSCGymEnvShifts`) hace priming extra post-warmup ... `MFSCSimulation.run()` (la simulación DES pura) **NO hace priming**."
- **Code does:** `env_experimental_shifts.py:355` constructor default is `priming_enabled=True`. `train_agent.py:335-355` (the README headline runner) does NOT override it, so PPO trains and evaluates with priming ON. `external_env_interface.py:702` `make_thesis_aligned_training_env` sets `priming_enabled=False`. So Garrido-faithful runs use priming OFF. **PPO and Garrido-faithful statics start their evaluation episodes from different DES states.**
- **Verdict:** `EXTENSION`
- **Impact:** Documented in DIVERGENCE_FIX_PLAN.md as "intencional." But it is a confound for the headline claim: PPO's `reset()` runs `_prime_after_warmup()` (`env_experimental_shifts.py:959-979`) which advances the DES up to `max_priming_hours=2016` (12 weeks) under `priming_shifts=2` until fill_rate crosses `operational_fill_rate_thresholds[risk_level]`. This (i) shifts the post-warmup start time forward by up to 12 weeks, (ii) pre-clears the startup backlog transient that thesis-faithful runs include in their measurement window, (iii) biases the initial state distribution toward S=2-reachable states regardless of the agent's first action. A PPO policy that "wins" partly because it starts from a primed state is not proving the same thing as a Garrido-faithful policy evaluated from the raw warm-up state.

### Finding D2: Priming eats into the episode when `max_steps` is fixed
- **Claim:** `DIVERGENCE_FIX_PLAN.md:127` estimates "un offset de ~336 h (2 semanas)".
- **Code does:** `env_experimental_shifts.py:633-636` `max_steps = (SIMULATION_HORIZON - warmup_hours) / step_size`. But when the benchmark passes `max_steps=260` explicitly (`benchmark_control_reward.py:795`), `reset()` first runs warmup (838.8h) + priming (up to 2016h), THEN counts 260 steps × 168h. So the actual evaluation window can start at env.now ≈ 838.8+2016 = 2854.8h and run to env.now ≈ 2854.8 + 260×168 = 163,262.8h, which exceeds `SIMULATION_HORIZON=161,280`. The DES clamps to horizon (`supply_chain.py:553` `target = min(self.env.now + dt, self.horizon)`), so the last few steps get shorter. Episode length is silently truncated.
- **Verdict:** `AMBIGUITY`
- **Impact:** Episode-length accounting under priming is sloppy. PPO and static baselines within `benchmark_control_reward.py` both see the same sloppiness (same env_kwargs → symmetric), so the comparison stays fair. But the **absolute** numbers (e.g., total reward over 260 steps) are not what the config claims they are.

### Finding D3: RL episode is 5.4 years (260 × 168h), not the thesis 20-year horizon
- **Claim:** README:158 "Main paper scenarios: `increased + stochastic_pt`, `severe + stochastic_pt`." `config.py:788` `BENCHMARK_REFERENCE_MAX_STEPS = 260`. `benchmark_control_reward.py:525` calls this "the historical 260x168h physical horizon." `config.py:789-791` `BENCHMARK_EPISODE_HORIZON_HOURS = 168 × 260 = 43,680h`.
- **Code does:** `benchmark_control_reward.py:721-730` `resolve_episode_max_steps` returns 260 when step_size=168. `train_agent.py:307-316` does the same. So both headline runners evaluate on a 43,680h window. The thesis (`config.py:40` `SIMULATION_HORIZON = 161,280`) and THESIS_FIDELITY_AUDIT.md:87 evaluate on a 161,280h window. The Black-Swan R3 distribution is `U(1, 161280)` (`config.py` RISKS_CURRENT→R3) — a 20-year return period. **In a 43,680h window, the probability of a Black-Swan event is ≈ 27%.** Most PPO eval episodes never see one.
- **Verdict:** `RIGGED_COMPARISON`
- **Impact:** **This is the single biggest threat to the paper's claim.** Garrido's thesis resilience metric is dominated by how the chain handles R3 black swans (the worst-case event the model is designed to be resilient against). The RL benchmark cuts the horizon to 27% of the thesis and thereby excludes ~73% of black-swan exposure. A PPO policy that "wins" on a 5-year no-black-swan window is not proving resilience to the events Garrido's metric is designed to measure. The README does not disclose this horizon gap; `THESIS_FIDELITY_AUDIT.md:87` claims the horizon is identical to the thesis when in fact the RL benchmark does not use it.
- **Evidence:** `config.py:788-791` vs `config.py:40`; `benchmark_control_reward.py:730` returns 260.

### Finding D4: `step()` advances exactly `step_size` hours (fixed, not event-driven)
- **Claim:** AGENTS.md:64-66 "Strict Rule: The assembly line operations (Op5-Op7) operate on hourly granularity."
- **Code does:** `supply_chain.py:551-572` `dt = step_hours or self._step_size; target = min(self.env.now + dt, self.horizon); self.env.run(until=target)`. So each env step is exactly 168h at the env layer (with the small end-of-episode clamp from D2). Event-driven dynamics happen INSIDE the step via SimPy; the agent only sees the post-step state. Same `step_size` used at train and eval.
- **Verdict:** `MATCH`
- **Impact:** None. Train/eval step semantics are identical.

### Finding D5: Termination is via `truncated` (step count) not `terminated` (event)
- **Claim:** `env_experimental_shifts.py:2500` sets `truncated = self.current_step >= self.max_steps`.
- **Code does:** The env returns `terminated=False` essentially always (sim.step returns `done=True` only at horizon; with max_steps=260 and priming, `truncated` fires first). The episode is ended by `truncated` (step count) not by any domain event.
- **Verdict:** `MATCH`
- **Impact:** The episode horizon is not "artificially short to make PPO look good" in the sense of cutting off a bad tail — it is artificially short **uniformly** at 260 steps. But see D3: 260 steps is also too short to expose R3.

---

## E) Static-vs-learned comparison fairness

### Finding E1: Within `benchmark_control_reward.py`, all policies share env_kwargs and seeds (CRN preserved)
- **Claim:** Implicit: the benchmark is structurally fair across policies.
- **Code does:** `evaluate_policy` (`benchmark_control_reward.py:1244-1346`) builds `env_kwargs = build_env_kwargs(args, weight_combo)` once per (args, weight_combo) and uses it for **every** policy — `static_s1`, `static_s2`, `static_s3`, `garrido_cf_s1/s2/s3`, all heuristics, and PPO. `eval_seed = seed + EVAL_EPISODE_SEED_OFFSET + episode_idx` (`:1263`) is the same seed sequence for every policy. So Common Random Numbers ARE preserved across policies at the same episode index.
- **Verdict:** `MATCH`
- **Impact:** The structural fairness is OK. Whatever PPO-vs-static comparison comes out of this runner is on the same env config and same RNG trajectories.

### Finding E2: Two non-equivalent "static S=1/2/3" baselines coexist
- **Claim:** `benchmark_control_reward.py:42-44` defines `STATIC_POLICY_ORDER = ("static_s1", "static_s2", "static_s3")` and `GARRIDO_POLICY_ORDER = ("garrido_cf_s1", "garrido_cf_s2", "garrido_cf_s3")`.
- **Code does:** `FIXED_POLICY_ACTIONS["static_s1"] = np.array([0,0,0,0,0,-1])` (`:47`) → goes through multiplier decoder → Op3 inv = 19,375, Op9 dispatch = [3000, 3250], Op3 ROP = 210h, Op9 ROP = 30h. `FIXED_POLICY_ACTIONS["garrido_cf_s1"] = {dict}` (`:50-58`) → bypasses decoder → Op3 inv = 15,500, Op9 dispatch = [2400, 2600], Op3 ROP = 168h, Op9 ROP = 24h, batch_size = 5000. **These are different policies** despite the shared "S=1" label.
- **Verdict:** `RIGGED_COMPARISON`
- **Impact:** Depends entirely on which baseline is headline. If the paper reports "PPO beats static_s1," it is comparing against an over-buffered non-thesis baseline (which is actually FAVORABLE to the static — bigger buffers). If it reports "PPO beats garrido_cf_s1," the comparison is thesis-faithful on action mapping but still confounded by D1 (priming) and D3 (5-year horizon). **The benchmark reports BOTH columns** (`benchmark_control_reward.py:1906-1913` emits `static_s2_reward_mean` AND `garrido_cf_s2_reward_mean`), so the data is there — but the paper text must specify which is the headline comparator, and a reader who sees "static S2" can be misled.

### Finding E3: README headline runner (`train_agent.py`) does NOT compute any PPO-vs-static comparison
- **Claim:** README:107-117 "Frozen benchmark backbone" gives a `train_agent.py` command and says it produces artifacts under `outputs/`.
- **Code does:** `train_agent.py` only trains PPO and reports PPO's eval reward (functions `random_baseline` and `evaluate_trained`, lines 381-442). It never instantiates `static_s1/s2/s3` or `garrido_cf_s1/s2/s3`. There is no code path in `train_agent.py` that produces a "PPO vs thesis static" number.
- **Verdict:** `RIGGED_COMPARISON`
- **Impact:** The README's headline 500k-timestep command **cannot**, by itself, support any "PPO beats static" claim. To make that claim you must invoke `scripts/benchmark_control_reward.py` instead, which uses a different runner, different seed offset (80,000 vs 20,000 — `benchmark_control_reward.py:306` vs `train_agent.py:417`), no VecNormalize wrapper (vs train_agent.py:374 VecNormalize-on), and a different obs default (BENCHMARK_OBSERVATION_VERSION=v4 vs README's v1). **The PPO model trained by the README command is not directly re-usable for the comparison** without rerunning through the benchmark runner. This is a structural gap in the reproducibility story.

### Finding E4: PPO has a strictly larger action space than Garrido's thesis
- **Claim:** `THESIS_FIDELITY_AUDIT.md:99-106`, `DIVERGENCE_FIX_PLAN.md:94-122`: Track A exposes 6 continuous controls (Op3_q, Op9_q, Op3_rop, Op9_rop, Op5_q, shift); Garrido exposes 3 discrete (5 inv-levels × 3 shifts, but ROP is NOT a Garrido decision variable).
- **Code does:** `env_experimental_shifts.py:680-685` action_space is `Box(-1, 1, shape=(6,))`. The thesis-faithful discrete decision contract `THESIS_DECISION_ACTION_FIELDS` (`external_env_interface.py:100-104`) is 18D one-hot over (5 inv periods × 3 nodes) + (S1, S2, S3). The PPO action space includes **Op3 ROP and Op9 ROP** which the thesis explicitly does not manipulate. PPO can therefore execute policies that are **not realizable** under any Garrido configuration.
- **Verdict:** `EXTENSION`
- **Impact:** Documented as a deliberate extension. But the paper claim must be precise: "PPO with a 6D extension action space beats the best 3D thesis discrete action" is true to the code; "PPO beats thesis-style policies" without that qualifier is misleading. The repo already has the machinery for the apples-to-apples version: `THESIS_FACTORIZED_ACTION_FIELDS` (`external_env_interface.py:108-111`) is 2D (common I_{t,S} level + S), which IS the strict Garrido decision surface. That variant is not the headline.

### Finding E5: VecNormalize convention differs across runners (no cross-runner number portability)
- **Claim:** README implies `train_agent.py` and `benchmark_control_reward.py` are interchangeable benchmark paths.
- **Code does:** `train_agent.py:374` wraps the env in `VecNormalize(vec, norm_obs=True, norm_reward=(mode=="proxy"), clip_obs=10.0)`. At eval (`train_agent.py:421`) it freezes `vec_norm.training=False` and calls `vec_norm.normalize_obs(obs_raw)` before `model.predict` — internally consistent. `benchmark_control_reward.py:1110` uses bare `DummyVecEnv([make_monitored_training_env(...)])` with NO VecNormalize, and eval (`:1344`) feeds raw obs directly to `model.predict` — also internally consistent. So within each runner the obs distribution the policy sees is consistent across train and eval. **But a PPO model trained under train_agent.py expects normalized obs; loading it into benchmark_control_reward.py without re-applying the saved VecNormalize stats would feed it raw obs and silently break it.**
- **Verdict:** `MATCH` (within-runner) / `AMBIGUITY` (cross-runner)
- **Impact:** No bug if you stay inside one runner. Hazard if you train in one and evaluate in the other.

### Finding E6: Reward weights `w_bo=4.0, w_cost=0.02, w_disr=0.0` are inert for non-`control_v1` rewards
- **Claim:** README:116 passes `--w-bo 4.0 --w-cost 0.02 --w-disr 0.0` alongside `--reward-mode ReT_seq_v1`.
- **Code does:** `_compute_ret_seq_v1` (`env_experimental_shifts.py:1352-1389`) does not read `self.w_bo`, `self.w_cost`, or `self.w_disr` — those are only used by `_compute_control_v1_components` (`:1138-1163`). Passing them with `reward_mode=ReT_seq_v1` is silently ignored.
- **Verdict:** `AMBIGUITY`
- **Impact:** The README command's `w_bo/w_cost/w_disr` flags are documentation noise under the headline reward. Not a bug (the env correctly ignores them), but reproducibility confusion: a reader changing those flags expects to affect training, and they don't.

---

## Cross-cutting: what each named reward actually computes

| Reward mode | What it computes | Training? | Audit? | Notes |
|---|---|---|---|---|
| `ReT_thesis` | Step-level piecewise Eq. 5.5 approximation, then `−δ(S−1)` shift cost | No (collapses to S1) | Yes | `_compute_ret_thesis_components:1027` |
| `ReT_corrected` | Same as above but autotomy scored with recovery formula | Historical | Yes | `:1100` |
| `ReT_unified_v1` | `FR^0.60 · RC^0.25 · CE^(0.15·gate)` with sigmoid gate on service+recovery | Research | Yes | `:1169` |
| **`ReT_seq_v1`** | `SC^0.60 · BC^0.25 · AE^0.15` — **no inventory cost** (see B2) | **README default** | Yes | `:1259` |
| `ReT_ladder_v1` | `SC^0.65 · RC^0.30 · EF^(0.05·gate)` — gated cost | Research | Yes | `:1405` |
| `ReT_tail_v1` | `SC^0.30 · RC^(0.60·boost) · CE^0.10` — un-gated cost (fixes B2) | Recommended | Yes | `:1490` |
| `ReT_cd` | 4-variable C-D: `FR^a · IB^b · SC_cap^c · IC^d` | Research | Yes | `:1598` |
| `ReT_garrido2024_raw` | 5-variable C-D raw product (Eq. 3), κ̇ = avg_cost/κ_ref | Candidate | Yes | `:1762` |
| `ReT_garrido2024` | Sigmoid of the 5-variable log-score (Eq. 6) — paper-facing audit index | No | **Yes** | `:1762` |
| `ReT_garrido2024_train` | Same with κ̇ term scaled by `kappa_train_frac=0.20` to avoid S1/S3 collapse | Candidate | Yes | `:1860-1866` |
| `ReT_cd_v1` | `FR^0.70 · AT^0.30` (raw C-D, no sigmoid — sigmoid ≤0.5 bias) | Bridge | Yes | `:1924` |
| `ReT_cd_sigmoid` | Same with sigmoid wrapper; documented as NOT RECOMMENDED | No (bias) | Yes | `:2004` |
| `control_v1` | `−(w_bo·service_loss + w_cost·shift_cost + w_disr·disruption_frac)` — linear | Legacy | No | `:1138` |
| `control_v1_pbrs` | control_v1 + Ng et al. PBRS shaping bonus | Extension | No | `:2440-2454` |
| `rt_v0` | Legacy weighted sum (recovery + holding + service + shift) | Legacy | No | `:1120` |

`ret_thesis.py::compute_order_level_ret` (the 74-LOC file) implements the **true thesis order-level ReT** (Eq. 5.5 per order). It is called only at episode end via `sim.compute_order_level_ret()` (`supply_chain.py:1880-1894`) and surfaced by `get_episode_terminal_metrics()` (`external_env_interface.py:225-269`). It is **never used as an RL step reward**. Any paper figure plotting per-step ReT using the env's `_compute_ret_thesis_components` is plotting a different quantity than the thesis.

---

## Specific answers to the brief's questions

**Q1. Action decoder bounds and shift mapping.** Dims 0-3: `m = 1.25 + 0.75·a`, range [0.5, 2.0] (`env_experimental_shifts.py:2332`). Dim 4 (Op5): `m = 1.0 + 0.5·a`, range [0.5, 1.5] (`:2335`). Dim 5: piecewise `< -0.33 → S=1`, `[-0.33, 0.33) → S=2`, `≥ 0.33 → S=3` (`:2338-2344`). The decoder materially changes DES state: `action_dict` is applied to `sim.params` (`supply_chain.py:531-549`) including batch_size coupling. **Verified not a no-op.**

**Q2. Step semantics.** `step()` advances exactly `step_size` = 168h via `env.run(until=target)` (`supply_chain.py:551-572`). Fixed, not event-driven at the env layer. Same at train and eval.

**Q3. Termination.** `truncated = current_step ≥ max_steps` (`env_experimental_shifts.py:2500`). With `max_steps=260`, episode ends after 260 steps regardless of state. `terminated` essentially never fires (would require reaching horizon).

**Q4. Priming extra.** `_prime_after_warmup` exists at `env_experimental_shifts.py:959-979`, gated by `priming_enabled` (default True, `:355`, `:456`, `:2119`). It advances up to `max_priming_hours=2016` (12 weeks) under `priming_shifts=2` until fill_rate ≥ `operational_fill_rate_thresholds[risk_level]` (current: 0.55, increased: 0.40, severe: 0.15). `MFSCSimulation.run()` (`supply_chain.py:507-511`) does NOT prime — it just runs `env.run(until=horizon)` and marks warmup on first Op9 Q=5000 arrival. **Confirmed: priming is in the RL env, not in the pure DES.** Quantified offset: up to 2016h (12 weeks) earlier than Garrido's measurement window start.

**Q5. PPO apples-to-apples?** See E1–E4. The structural machinery exists (`garrido_cf_sN` baselines bypass the multiplier, CRN seeds are aligned within `benchmark_control_reward.py`), but the README headline command does not invoke it, and even when invoked the comparison is confounded by D1 (priming ON for PPO, OFF for thesis-faithful runner) and D3 (5-year RL horizon vs 20-year thesis horizon).

---

## Top risks to paper validity (ranked)

1. **D3 — Horizon mismatch.** `RIGGED_COMPARISON`. RL benchmark is 5.4 years; thesis is 20 years. Black-swan R3 (return period 20yr) is excluded from ~73% of PPO eval episodes. `config.py:788-791` vs `config.py:40`. **Fix:** run the headline PPO benchmark with `max_steps = (161280 - warmup_hours)/168 ≈ 956`, not 260.

2. **B2 — `ReT_seq_v1` has no inventory holding cost.** `REWARD_HACKING_RISK`. The frozen benchmark reward lets PPO over-buffer for free; the repo's own `RET_TAIL_V1_TUNING_2026-06-17.md` admits this collapses the policy. `env_experimental_shifts.py:1368-1369`. **Fix:** switch the headline reward to `ReT_tail_v1` (which adds un-gated inventory + shift cost).

3. **E2/E3 — Static-baseline ambiguity + headline runner doesn't compute the comparison.** `RIGGED_COMPARISON`. The README's `train_agent.py` command does not evaluate against any Garrido-faithful baseline. Within `benchmark_control_reward.py`, `static_sN` (multiplier-mapped, 1.25× inventory) and `garrido_cf_sN` (thesis-faithful, direct DES) are different policies under the same "S=N" label. `benchmark_control_reward.py:47-76`. **Fix:** headline the `garrido_cf_sN` columns; never label `static_sN` as "thesis-style."

---

## Verdict on the brief's final question

> Is the claim "PPO outperforms thesis-style static policies" computable from what's in the code on an apples-to-apples basis?

**Conditionally yes, but not from the README's headline command, and not without three disclosed deviations.**

- **Computable yes**, via `scripts/benchmark_control_reward.py`, using the `garrido_cf_s1/s2/s3` columns (which bypass the multiplier decoder and use direct DES parameters). CRN seeds, env_kwargs, priming state, and max_steps are aligned across all policies in that runner (E1).
- **Not from `train_agent.py`** (the README's headline 500k command). That runner trains PPO in isolation; it does not compare against any static baseline (E3).
- **Three confounds remain even in the fair runner:**
  1. The 5-year horizon (D3) vs thesis 20-year.
  2. The priming-on initial state (D1) vs thesis warmup-only.
  3. The 6D extension action space (E4) including ROPs that Garrido does not manipulate.

A defensible paper claim reads: *"Under our 5-year RL benchmark window with priming enabled and a 6D continuous extension of Garrido's discrete decision variables, PPO trained on ReT_tail_v1 outperforms the thesis-faithful S1/S2/S3 baselines on the per-step ReT_tail audit metric."* Anything broader than that overstates what the code supports.
