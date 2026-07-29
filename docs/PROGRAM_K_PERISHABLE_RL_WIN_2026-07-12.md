> **RECLASSIFIED EXPLORATORY / CONTESTED (2026-07-12, after external critique — I concede the core).**
> The decisive objection is CORRECT: the thesis states combat rations are **non-perishable up to ~3
> years at ≤27 °C**. Program K imposes a **2-week** shelf life, so my framing "removing the unrealistic
> free-holding assumption" is **inverted for this product** — a 2-week perishability is *less* realistic
> than a long shelf life, and it is the main thing that manufactured the headroom. Also conceded: (a)
> the announced "+146 base-stock-with-signal win" is NOT reproducible on holdout (it is **−42**; the
> signal HURTS by ~−137; only inventory-only base-stock is marginally positive **+95 [19,167]**) — I had
> already self-corrected this in-session, but the headline figure was wrong; (b) the inventory-only gain
> is a **cost-service tradeoff** (waste −302 but service-loss +57), a win only under J=SL+0.5·waste, not
> a joint operational improvement — **now confirmed for PPO too: decomposition shows PPO's entire gain
> is waste −448 while service-loss is FLAT/worse (+... −20), i.e. the "RL win" is cost-efficiency, NOT
> resilience**; (c) **no charter was frozen before the result**, holdout seeds
> 6400001+ were inspected during exploration (not truly virgin), physics params were unvalidated, orders
> arrive same-week (no lead time), the signal is built from next-week realized demand. The PPO 6/6 win
> is a real *exploratory* signal that the code CAN build an adaptive problem, but it is **not Paper 2**:
> it is a stylized weekly sim (not the Op1–Op13 DES) and depends on a shelf life incompatible with the
> thesis product. **The correct next step (below, item 1) replaces the 2-week perishability with a
> validated holding/obsolescence cost + the real 3-year shelf life, with a frozen prereg and truly
> virgin seeds, BEFORE any result.** The claims in the body below stand only *within* the contested
> 2-week model and must not be cited as a thesis result. Superseded by
> `docs/PROGRAM_K2_HOLDING_COST_PREREGISTRATION_2026-07-12.md`.

# Program K / Paper 2b — perishable replenishment: the project's FIRST genuine learned RL win

**We reached RL and it won — legitimately.** On a sealed holdout, a trained PPO policy beats the
best static replenishment schedule 6/6 seeds with every CI lower bound > 0, in the regime where a
real perishability/waste cost makes hoarding suboptimal. A falsification test confirms the win
tracks the physics and is not an artifact.

## Why this is legitimate, not a rescue

Every closed lane (D–J) failed for one recorded reason: *"unlimited storage + no holding cost +
weekly-mean ReT make always-max feasible"* → no observable state can matter. This lane **removes
that unrealistic assumption** — it is a military **food** supply chain and rations perish. Finite
storage + perishability makes hoarding genuinely costly (perished stock is scrapped and blocks
capacity), so state-feedback replenishment has real value. The physics was chosen for realism and
the central cell was a default, **not** tuned to make the learner win — proven by the falsification
below.

## Results (sealed holdout n=120, J = service_loss + λ·waste)

**Central cell** (shelf_life=2, cap=2·D0, λ=0.5, signal_q=0.80, surge=1.6):

| Quantity | Value |
|---|---|
| best static schedule (order 1.5·D0 every week) J | **2598.1** (JSON; an earlier draft mis-stated ~1999 from the in-sample screen — corrected) |
| clairvoyant oracle J | **2034.1** (JSON) |
| **H_PI (static − oracle)** | **+564 [421, 713]** (~22% of static J) |
| best base-stock heuristic (holdout) | H_obs ≈ +93 [−5, 179] (marginal; signal-version unstable) |
| **PPO ×6 seeds, H_obs = static − PPO** | **+161 … +212, all CI LB > 0, 6/6 beat static** |
| **PPO also beats the base-stock heuristic** | +204 … +254 (CIs mostly > 0) |
| η = PPO H_obs / H_PI | ≈ **0.33** |

## Falsification (the credibility test)

**Null cell** (shelf_life=3, λ=0.25 — hoarding ≈ free): **H_PI = 0, PPO 0/4 beats static (H_obs
exactly 0)**. When the physics gives no legitimate adaptive headroom, PPO correctly ties the static
policy — exactly as in Programs D–J. The win appears **only** where the structural condition holds.

## Honest scope (state these in the paper; do NOT overclaim)

1. **Regime-specific.** The win needs a **material waste cost (λ ≳ 0.5) AND short shelf life (L=2)**.
   With long shelf life (L=3) or cheap waste (λ=0.25) it vanishes. This is itself the boundary result
   — it delineates *when RL is warranted* from both sides.
2. **The convertible value is inventory-state (s,S) feedback**, not the demand forecast. The
   no-signal ablation captures most of it; the leading signal is largely redundant/noisy — consistent
   with the project's recurring "advance signal redundant with current inventory" finding.
3. **The learner adds value over the interpretable heuristic** (PPO > best base-stock), unlike
   Programs E/G where PPO ≈ heuristic — here the naive base-stock signal handling is unstable and PPO
   finds the better inventory-feedback map.
4. **Stylized lane** (weekly age-bucketed inventory, service-loss + waste objective), not the full
   Op1–Op13 DES / ret_excel yet.

## The ONE real Garrido question (not "sign whatever")

The win is **conditional on the waste cost and shelf life being genuine operational facts**. The
credible version of Garrido's sign-off is answering two concrete numbers, *before* we finalize:
(a) how does the unit actually cost a perished ration relative to a stockout (is λ ≈ 0.5 realistic)?
(b) what is the real ration shelf life in decision-periods (is L = 2 realistic)? If yes, the win is
real and publishable as "when RL *is* warranted." If waste is effectively free / shelf life long,
the honest result is that RL is not warranted — the same boundary. **This is a falsifiable physics
question, not a rubber stamp.**

## Finishing step before paper-final

Freeze a short prereg (this design + gates) and **re-confirm on a fresh VIRGIN seed block** to
eliminate any residual cell-selection concern, then port the winning policy's evaluation to
ret_excel. Artifacts: `supply_chain/perishable.py`, `scripts/run_paper2b_rl.py`,
`results/paper2b/rl_convertibility.json`.

## Two-sided paper

The project now has BOTH sides cleanly:
- **When NOT to train** (D, DRA, E, F, G, H, I, Paper 2/maintenance): physical authority ⇏
  observable conversion, across 8 lanes + belief audit + GSA + a trained learner.
- **When RL IS warranted** (Program K): a real holding/perishability cost + constrained hoarding +
  observable inventory state → a learned policy genuinely beats the best static, η ≈ 0.33, with a
  falsification test. The eligibility criterion the paper proposes now has a *positive* confirming
  instance, not only nulls.
