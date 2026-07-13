# Paper 2 — RL convertibility verdict (the definitive learner test)

**We reached RL.** On the pre-committed second family (adaptive maintenance of Op5–Op7 with a
persistent observable degradation state and a single shared crew), we ran the definitive learner
test on the cell with the most physical authority. Result: the central finding holds — a competent
learner cannot convert real clairvoyant headroom into an observable advantage.

## Numbers (holdout n=120, sealed seeds 5400001+, service loss)

| Quantity | Value | Reading |
|---|---|---|
| best static periodic calendar | `PM7 PM6 PM7 PM5 PM6 PM5 PM7 PM6`, mean SL 21023.5 | protects downstream Op7 |
| clairvoyant oracle | 20604.3 | exact min over 3^8 sequences |
| **H_PI (static − oracle)** | **+419.1, CI95 [275.3, 578.1]** | **real physical authority (~2%)** |
| H_obs worst-condition heuristic | −500.1 [−702, −295] | loses to static |
| H_obs forecast heuristic | −529.6 [−724, −326] | predictive signal does **not** help |
| **PPO ×6 seeds, H_obs = static − PPO** | −4.4 to −105.3; **every CI straddles 0** | **ties static, 0/6 beat it** |
| η = captured / H_PI | ≈ 0 | no conversion |

## Why this is the cleanest confirmation yet

1. **This lane is not headroom-starved.** Unlike Programs E and I (H_PI ≈ 0 or ~1e-5), here the
   clairvoyant oracle genuinely beats the best static by ~2%, CI excluding zero. So the failure to
   convert is **not** "there was nothing to capture."
2. **The learner is competent, not weak.** Naive observable heuristics lose ~2.4% (they chase the
   highest-wear station Op5). PPO instead **discovers the good static policy** — matching the
   optimized calendar to within noise. It beat the heuristics by ~500 units; it just could not
   exceed the best fixed calendar.
3. **The residual headroom is structurally non-observable.** In a serial line, throughput value
   lives in the **downstream position** (protect Op7 — a *static* fact the best calendar already
   encodes), while the remaining 2% requires **clairvoyantly** timing which week's realized threat
   to pre-empt. A fair sensor + noisy 1-week forecast cannot supply that; the oracle can only
   because it sees the whole tape. Physical authority ≠ adaptive value — again, and most sharply.

## Gate outcome

```text
STOP_PAPER2_NO_OBSERVABLE_CONVERSION
H_PI_real = true            # +419 [275,578], ~2% of service loss
observable_heuristics_convert = false   # both ~ -500
ppo_beats_static = false    # 0/6 seeds, all CI straddle 0, eta ~ 0
clairvoyant_ceiling_below_practical_gate = true   # ~2% < 5%
virgin_tapes_opened = false
paper3_retained_learning = not_authorized   # requires a Paper 2 win that did not occur
```

Paper 3 (retained learning) is **not** opened: it was conditional on a Paper 2 adaptive win on
virgin tapes, which did not occur. No metric switch, risk inflation, or seed selection is used to
rescue this — the honest result is the deliverable.

## What Paper 2 contributes to the manuscript

The strongest structural case for RL in this DES — endogenous, observable, persistent, controllable
degradation with intertemporal opportunity cost — still yields η ≈ 0 under a legitimately trained
PPO. This is the hardest test of "When is RL *not* warranted in a supply-chain DES?" and it passes:
even a competent learner's best move is to reproduce a well-chosen static calendar. The clairvoyant
gap is a value-of-perfect-information artifact, not a value-of-adaptivity one.

## Open Garrido / ChatGPT-Pro doubt (flag to PI)

Degradation efficacy, `TAU`, and wear ranges are chosen physics calibrated to the thesis R11 window
but not Garrido-signed. Given the null (negative observable conversion, sub-5% ceiling), this is a
robustness note rather than a blocker — but if the manuscript foregrounds Paper 2, a Garrido sign-off
on the degradation ranges is the remaining validity gate.
