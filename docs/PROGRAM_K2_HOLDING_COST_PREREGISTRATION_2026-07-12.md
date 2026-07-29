# Program K2 — replenishment under a VALIDATED holding/obsolescence cost (prereg, frozen BEFORE any result)

Supersedes the contested Program K (2-week perishability, incompatible with the thesis product). This
prereg is frozen **before** building the corrected environment or seeing any result, to satisfy the
discipline the K critique correctly demanded.

## What the K critique got right (conceded, and now designed against)

1. **Physical:** thesis combat rations are non-perishable ~3 years at ≤27 °C. K's 2-week shelf life is
   *less* realistic than a long shelf life and manufactured most of the headroom. → K2 uses the **real
   ~3-year shelf life (effectively infinite on an 8–52-week horizon)** as the default, and makes hoarding
   costly through a **holding/obsolescence cost**, not spoilage.
2. **Comparator too weak.** Beating a *fixed schedule* is trivial in inventory control — (s,S) does it by
   textbook. The real "is RL warranted" question is whether a learner beats the **best classical adaptive
   policy**. → K2's primary comparator is the **best-tuned (s,S)/order-up-to policy**, not the static
   schedule. RL must beat *that* on virgin seeds to count.
3. **Process:** no frozen charter, holdout inspected during exploration, unvalidated params, same-week
   order arrival, look-ahead signal. → K2 freezes this prereg first, uses **fresh virgin seed blocks**,
   a **lead time L≥1**, a signal that is a noisy forecast **not** used to reach any conclusion (ablated),
   and reports the full **service-loss vs holding decomposition**.
4. **Cost-service tradeoff, not resilience.** K's gain was waste↓ at service↑. → K2 reports service loss
   as a **first-class, separate** outcome; a policy that only trades service for cost is labelled a
   cost-efficiency result, not a resilience result.

## Environment (to build: `supply_chain/replenish.py`)

- Age-tracked inventory with **real shelf life** (swept: {∞≈3yr, 26wk, 8wk, 4wk} — thesis value first).
- **Lead time L ∈ {1, 2} weeks** (orders do NOT serve same-week demand).
- Finite storage capacity (swept), persistent calm/surge demand regime, noisy 1-wk demand signal
  (ablated: with vs without).
- **Objective J = p·service_loss + h·holding_units** (holding = mean on-hand per week). p = shortage
  penalty, h = holding/obsolescence cost. Report p/h swept over a range; the DEFAULT p/h must be a
  **Garrido/literature-defensible ratio**, not chosen to make RL win.
- Actions: discrete order levels (multiples of D0). CRN: demand/signal fixed per tape.

## Comparators (all on the SAME sealed virgin tapes)

1. Best static periodic order schedule (weak baseline, for context).
2. **Best-tuned classical adaptive (s,S) / order-up-to-S policy** (grid-searched on calibration) —
   **the primary comparator**.
3. Clairvoyant open-loop oracle (H_PI ceiling).
4. Learner: PPO (MlpPolicy 2×64), ≥6 seeds, obs = [inv-by-age, signal, pipeline, week]; no privileged
   future/state access.

## Frozen seed universes (distinct from K's 63/64xxxx)

- Calibration (train + tune S,s + best static): 6600001+.
- Test (design iteration): 6700001+.
- **Virgin (evaluation only, opened once, after gates on test):** 6800001+ — sealed.

## Decision rule (binding)

- **RL is warranted** ⇔ on virgin seeds, PPO beats the **best classical (s,S)** with H_obs LCB95 > 0
  **and** does not worsen service loss (the resilience outcome) — i.e. a genuine joint gain, not a
  cost-for-service swap.
- **Falsification (required):** in a null regime (real 3-yr shelf life, h→0) H_PI must be ≈0 and PPO must
  tie both comparators. If PPO "wins" in the null, the pipeline is broken — investigate, do not report.
- **Expected honest outcome, stated in advance:** with the real shelf life and only a holding cost, a
  well-tuned (s,S) is likely optimal, so PPO will most plausibly **tie (s,S) → RL NOT warranted (a
  tuned classical policy suffices)** — which is the project's central thesis, now on the *correct*
  physics. A PPO win over (s,S) would require a genuine structural complication (regime-dependent
  lead time, capacity coupling, multi-echelon) — only then is Paper 2 real. No metric/ratio is tuned to
  force either outcome; the result is reported whichever way it lands.

## Garrido questions (concrete, falsifiable — the thesis already answers the naive one)

The thesis ALREADY states rations are non-perishable ~3 yr at ≤27 °C, so we do **not** ask "is a
2-week shelf life realistic" (it is not). Instead the operational numbers that actually determine
whether adaptive replenishment has value:
1. **Residual shelf life per echelon** (CDC → WDC → CSSU): how much life remains after upstream
   storage, and is there any echelon (fresh inputs upstream of Op5?) where it is genuinely short.
2. **Rotation / discard rules** (FIFO enforced? write-off-at-expiry cadence?).
3. **Physical storage capacity per echelon** (the real cap that limits hoarding — likely the true
   binding constraint, not spoilage).
4. **Holding/obsolescence cost per ration-week vs the shortage penalty per unmet ration** (the p/h
   ratio) and **the real replenishment lead time**.

K2's outcome is conditional on these and is FALSIFIED if the realistic numbers (long shelf life,
modest holding cost) kill the headroom. If capacity — not perishability — is the real binding
constraint, K2 should be reframed as a **capacity/lead-time** problem, which is the physically honest
version of "hoarding is not free."

## Comparators must be STRONG (K2 first-screen finding, 2026-07-12)

The first K2 screen (correct physics: shelf 156 wk, h=0.4, lead=1, TEST 6700001+) exposed that with a
**coarse 4-level order grid** neither the best static NOR my grid (s,S) approaches the oracle
(H_PI≈+15.7k ~37%; (s,S) even slightly *worse* than best static, and it wins holding only by losing
service). That large clairvoyant gap is largely a **discretization/weak-comparator artifact**, exactly
the error the critique warns about — a PPO "win" over a weak (s,S) would be meaningless. Therefore,
BEFORE any learner run, K2 must add:
- a **finer/continuous order quantity** action space (or a DP/base-stock analytical optimum), so the
  static and (s,S) comparators are genuinely strong;
- **dynamic programming** value-iteration optimum (or a tight base-stock formula) as the true
  observable optimum J*_obs, and an **MPC / receding-horizon** policy using the demand signal;
- the learner is "warranted" ONLY if it beats the STRONGEST of {tuned (s,S), DP/base-stock, MPC} on
  virgin seeds AND does not worsen service loss.

## Artifact discipline (K critique point: the null run was ephemeral)

Every K2 run (null + central) MUST write a versioned JSON with: cell params, seed ranges + tape
hashes, per-seed results, CIs, and the **service-loss / holding decomposition** — no result is cited
from an ephemeral shell command. `results/k2/*.json` is the single source of truth.

## Status

Prereg FROZEN. Env `supply_chain/replenish.py` built (holding cost + lead time + real shelf life +
classical (s,S)); G0 tests `tests/test_replenish_physics.py`. First screen done (NOT virgin) — it
shows the comparator/action-grid must be strengthened (DP/finer actions/MPC) before a fair learner
test. **No learner trained and no virgin seeds opened yet.** Program K stays reclassified EXPLORATORY.
This does not touch the closed lanes D–J or Paper 1 ("when NOT to train" stands regardless).
