# Architecture review: what's been tried, what hasn't, and what it means — 2026-07-02

Scope: every non-standard network architecture explored anywhere in this
project (Track A or Track B), reviewed against the same question —
*does architecture, rather than action-space alignment, explain any of
the observed wins or losses?* This directly supports the paper's
reward-insensitivity / architecture-insensitivity argument (§4.6,
§5.3) and gives an honest map of what remains for Paper 2 (retained
learning / history).

## Summary table

| Architecture | Where tried | Scale | Verdict | Status |
|---|---|---|---|---|
| PPO + MLP (canonical) | Track A and Track B | Full confirmatory (10 seeds x 60k, Track B) | Track A: loses. Track B: wins (+0.000438, CI95>0) | **Headline, both tracks** |
| RecurrentPPO (LSTM) | Track A only (`v4`/`control_v1`) | Full confirmatory (500k x 5 seeds) | **Lost** to best static (`learned_beats_best_static=False`) | Closed; memory is not the Track A lever |
| DMLPA (transformer-over-history, `scripts/dmlpa_extractor.py`) | Track A only (`continuous_its`, `adaptive_benchmark_v2`) | Screening (2 seeds x 40k) | **Mixed**: one seed narrowly beats best-constant Excel ReT but collapses to a near-constant action (`frac_std`≈7e-5, i.e. not actually adaptive); the other seed stays adaptive but loses | Open; never run on Track B, never at confirmatory scale |
| KAN-inspired RBF sidecar (`scripts/kan_extractor.py`, mislabeled "PPO-KAN") | Track B (`control_v1`, canonical contract) | Confirmatory (5 seeds x 60k) | Matches canonical PPO+MLP (+0.000484 vs best in-arm comparator) — but the extractor has a full linear skip connection, so this shows "adding an RBF layer doesn't break the result," not "an architecturally distinct network independently replicates it" | Appendix/response-letter only; **rename before any external use** |
| Real KAN (pykan, official) | Supervised surrogate, not RL (decision vars -> Excel ReT on the 147-cell static frontier) | N=147, 118/29 split | R²=0.998, vs tuned MLP R²=0.855, linear R²=0.365 (`docs/KAN_REAL_DEMO_2026-07-02.md`) | Demo artifact for Garrido; not an RL policy |
| SAC / TD3 (off-policy continuous control) | Not run | — | N/A | Deliberately deferred; response-letter ammunition only (manuscript §5 Limitations already states this) |

## What this means for the paper's own argument

Track A: two architecturally distinct upgrades over plain PPO+MLP
(recurrent memory, transformer-over-history) were tried, and neither
converts the measured oracle headroom into a real win. RecurrentPPO
loses outright at full confirmatory scale. DMLPA's apparent win is a
single degenerate (near-constant) seed, which is the same failure mode
the paper already documents for other Track A learners — the agent
finds it safest to sit near a static optimum rather than exploit
disruption timing. This strengthens the manuscript's claim that Track
A's limitation is structural (the bottleneck sits outside the
controllable interface), not a matter of insufficient network capacity
or memory.

Track B: only the canonical PPO+MLP has been run at confirmatory scale.
The RBF-KAN sidecar (renamed here to avoid the earlier naming error)
is architecture-adjacent evidence that the win is not fragile to a
different feature extractor, but its linear skip connection means it
cannot be sold as a genuinely different architecture succeeding
independently — flag this precisely if it is ever cited.

## What remains open (candidates, not commitments)

1. **DMLPA on Track B, at confirmatory scale.** This is the single most
   relevant unfinished architecture question: does transformer-over-history
   memory add anything once the action space already reaches the
   bottleneck (Track B), as opposed to Track A where it doesn't help? This
   connects directly to the H4 retained-learning result (Paper 2): H4
   showed a small positive retained-vs-reset effect using the *same*
   PPO+MLP architecture across online-adaptation cycles; DMLPA is the
   natural architectural complement — does giving the network an explicit
   attention-over-history mechanism amplify that small effect, or is it
   architecture-independent? Pre-register this as an addition to
   `docs/PAPER2_H4_PREREGISTRATION_2026-07-02.md` before running it (new
   seed block, same claim threshold: seed-clustered CI95 lower bound > 0).
2. **A real KAN as an actual PPO policy/value network** (not the RBF
   sidecar, not the supervised surrogate) — feasible with `pykan`
   (confirmed working in this session) but nontrivial: pykan's `KAN`
   class does not natively slot into SB3's `BaseFeaturesExtractor`
   pattern with separate actor/critic heads, and its LBFGS-based fitting
   routine is not built for the online SGD-style updates PPO needs.
   Would need a custom SB3 policy wrapping the spline layers with
   Adam-compatible parameters. Nice-to-have for a stronger Garrido demo,
   not scoped for Paper 1 or Paper 2.
3. **SAC/TD3** remain untried by design (algorithmic choice is explicitly
   not the tested mechanism per the manuscript's own framing).

## Bottom line

No architecture change (recurrent, transformer-over-history, RBF/KAN-style
feature extraction) has overturned Track A's negative result or
meaningfully exceeded Track B's canonical PPO+MLP result. Every
architecture variant tried is consistent with the paper's central claim:
the mechanism is action-space alignment, not network design. The one
genuinely open architectural question — DMLPA on Track B at confirmatory
scale — is a Paper 2 candidate, not a Paper 1 gap.
