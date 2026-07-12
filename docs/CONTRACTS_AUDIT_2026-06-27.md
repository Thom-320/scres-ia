# Full contracts audit (2026-06-27): observation versions × action contracts

Complete enumeration of every observation version and action contract, what each exposes/controls,
and which combinations have actually been trained — to locate the real bottleneck for a dynamic win.

## Observation contracts (`OBSERVATION_VERSION_OPTIONS`)

| ver | n | anticipation block? | notes |
|---|---|---|---|
| v1 | 15 | NONE | reactive only (inventories, fill, backorder, current down-flags) |
| v2 | 18 | NONE | reactive |
| v3 | 20 | NONE | reactive |
| v4 | 24 | NONE | reactive — **used for ALL Track-A CD/Excel runs so far** |
| v5 | 30 | NONE | reactive (+ cycle phase, workweek/workday sin/cos) |
| **v6** | 40 | **YES** | first anticipation obs: `regime_{nominal,strained,pre_disruption,disrupted,recovery}` one-hot + `risk_forecast_48h_norm` + `risk_forecast_168h_norm` + backlog_age, maintenance_debt, theatre_cover_days |
| **v7** | 46 | **YES** | v6 + **downstream** (`op10_down`,`op12_down`,`op10/op12_queue_pressure`) + `rolling_fill_rate_4w`, `rolling_backorder_rate_4w` |

**Key:** only **v6 and v7** let the agent SEE risk coming (forecast + regime). v1–v5 are purely
reactive → an agent on them *cannot* anticipate, only react after a shock is already visible.
For a **clean Track-A** anticipation test, **v6 is preferable to v7** — v7's op10/op12 fields are
downstream (Track-B) signals that are irrelevant noise when the action is only buffer×shift.

## Action contracts

| contract | space | controls | strategic-buffer LEVEL? | learn initial preposition? |
|---|---|---|---|---|
| `track_a_v1` | Box(6) | op3_q, op9_q, op3_rop, op9_rop **multipliers** (×[0.5,2.0]); op5_q (×[0.5,1.5]); shift (tri-level) | **NO** — only dispatch-Q / reorder-point multipliers | n/a |
| `track_b_v1` | Box(8) | track_a_v1 + op10_q, op12_q downstream multipliers | NO | n/a |
| `track_a_thesis_factorized_v1` | MultiDiscrete([6,3]) | **buffer level (6 presets) × shift (3)** | **YES (per step, discrete)** | YES (DKANA/factorized wrapper supports `learn_initial_decision`) |
| `track_a_discrete18_v1` | Discrete(18) | flattened [6×3] buffer×shift | **YES (per step, discrete)** | **NO — the Discrete(18) wrapper discards `learn_initial_decision`** |
| `track_a_continuous_its_v1` | Box([0,-1],[1,1]) | **continuous common buffer fraction × shift signal** | **YES (per step, continuous buffer)** | fixed `init_frac`; learned initial decision not yet supported |

**Three structural facts this reveals:**
1. **The continuous strategic-buffer target has now been recovered locally** as
   `track_a_continuous_its_v1` (`supply_chain/continuous_its_env.py`), ported from
   `origin/codex/garrido-postfix-reruns`. It keeps Garrido-Rios' two decisions but de-discretizes
   the common `I_t,S` buffer fraction.
2. **The CD/Excel war-PPO ran on `track_a_v1`** → it never had the strategic-buffer lever (only
   dispatch multipliers) → that comparison was action-space-mismatched (see CD_LANE_AUDIT_NOTE).
3. **Learned initial prepositioning** exists only in the factorized/DKANA wrapper; the **Discrete(18)
   wrapper we use discards it** → we *fix* the initial buffer (`--ppo-initial-static-policy`) instead
   of *learning* it (Codex's prepositioning audit).

## What has actually been trained (combination matrix)

| lane | action | obs | anticipation available? | result |
|---|---|---|---|---|
| retention DQN | Discrete(18) | v5/v7 but **masked** (`direct_disruption_blind` removes forecast+regime) | NO (masked out) | memory null |
| CD/Excel PPO (war) | Discrete(18) | **v4** | NO (reactive) | collapses to constant; ties static |
| CD/Excel PPO (war) | track_a_v1 | v4 | NO | no buffer lever; loses |
| **v7 forecast test (today)** | Discrete(18) | **v7 (forecast VISIBLE)** | YES | **still collapses to near-constant** (S1_I1344×305 / S2_I168×302) |

**So:** across the project, the only runs with a real forecast either *masked it out* or — once we
finally exposed it today (v7) — **still collapsed to a near-constant.**

## Conclusion: the bottleneck is the ACTION contract, not the observation
Giving Pepe the forecast (v6/v7) did **not** make him adaptive on Discrete(18). The per-step
buffer×shift action cannot exploit foresight in a way that beats a good constant — a strategic
buffer is "insurance" that is best held *constant*, not *timed*, and the discrete level-setting
can't ramp ahead smoothly. This converges with every prior null: **in Track A [buffer×shift] the
optimal policy is a constant; RL finds it but cannot beat it via dynamics, with or without a forecast.**

### The two untested levers that could still produce a dynamic win
1. **Learned initial prepositioning** (factorized/DKANA wrapper, NOT discrete18) — Codex's two-phase
   runner. But note: at t=0 there is no state to condition on, so "learning" it ≈ picking the best
   constant initial → likely still ≈ best static.
2. **A richer action contract** — either a **continuous strategic-buffer target** (net-new; the
   user's "continuous inventory" idea) or **downstream control** (`track_b_v1`, op10/op12) where the
   agent reaches the actual bottleneck. Track B is where foresight (v6/v7) could plausibly pay off,
   because downstream dispatch *can* be timed against a forecast.

**Recommended next test:** if we want a genuine *anticipatory* win, pair **v6 (clean forecast) with
`track_b_v1` (downstream control)** — the one combination where the agent both *sees* the shock and
has a *timeable* lever to act on it. Track A buffer×shift is, by this audit, a constant-optimal
control problem.
