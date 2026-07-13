# Program K2 — strong-comparator screen verdict (EVPI-dominated, no convertible headroom)

The user chose "strengthen comparators first." Done — and it resolves the K2 question **without any
learner and without opening virgin seeds**.

## What changed vs the coarse first screen

Replaced the coarse 4-level order grid with **continuous** order quantities and added the provably
strong observable comparators: best static constant, best **base-stock S\*** (fine 1-D search), and an
**MPC** policy that uses the demand signal. Continuous clairvoyant greedy gives the EVPI floor.

## Results (correct physics: shelf 156 wk, holding h=0.4, lead=1; TEST 6700001+, n=120)

| Policy | J | note |
|---|---:|---|
| best static constant (q*≈1.45·D0) | ~41946 | orders ≈ mean demand |
| MPC (uses signal) | ~41946 | **static − MPC = +202 [−1536, 1964] → does NOT reliably beat static** |
| base-stock S* | ~48008 | **worse than the constant** (see below) |
| clairvoyant greedy (EVPI floor) | ~22168 | perfect future demand — not observable |
| **best observable − clairvoyant** | **+15820 [14754, 16896]** | **the whole gap is EVPI, non-convertible** |

## Why base-stock loses to a constant (a finding, not a bug)

The S-curve is clean (min near S≈2.5). A constant order ≈ mean demand beats reactive base-stock
because the **calm/surge demand variability (1.0 vs 1.6) is low relative to the holding cost (h=0.4)**:
when variability is small and holding is expensive, buffering is wasteful and "order the mean every
period" (a static rule) is near-optimal. Reactive replenishment adds holding without a matching service
gain. This is the project's central mechanism again — low convertible adaptive value.

## Verdict

```text
K2_STRONG_COMPARATOR_SCREEN_EVPI_DOMINATED_NO_CONVERTIBLE_HEADROOM
best observable policies (static const, MPC) cluster together (~42000)
gap to clairvoyant (~20000, ~47%) = EVPI (perfect future demand), NOT convertible
signal (MPC) does not reliably beat static; base-stock does not beat static
=> RL not warranted here, consistent with Programs D-J. No learner trained; virgin 6800001+ sealed.
```

Artifact: `results/k2/strong_comparators.json` (params, seeds, tape sha, per-policy CI, service/holding
decomposition). No parameters were tuned toward a learner win.

## Honest boundary — where a legitimate K2 win could still live (physics, not tuning)

A convertible adaptive win would require a regime with **genuinely high, observably-forecastable demand
variability relative to holding cost**, or a **binding capacity + long lead time** that a static order
cannot straddle — and those must come from Garrido's real numbers (residual shelf life per echelon,
storage capacity, lead time, holding/shortage ratio), not from turning the surge/holding knobs until
PPO wins. Under the *default correct physics*, K2 is another confirmation of the central finding, not
Paper 2. The manuscript's "when NOT to train" spine (D–J + K2) is strengthened; the positive instance
does not yet exist.
