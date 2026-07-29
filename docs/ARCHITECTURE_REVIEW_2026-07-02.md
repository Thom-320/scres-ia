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
| RecurrentPPO (LSTM) | Track A (`v4`/`control_v1`); **now Track B too** (`--algo recurrent_ppo` in `run_track_b_smoke.py`) | Track A: full confirmatory (500k x 5 seeds). Track B: 3 seeds x 30k screen | Track A: **lost** (`learned_beats_best_static=False`). Track B: beats best static 3/3 seeds (`order_ret_excel_mean=0.005857` vs `0.005451`), but **loses to canonical PPO+MLP** on a paired same-seed comparison (mean `-0.0000588`, 3/3 seeds negative, though at half PPO+MLP's training budget) | Track A closed. Track B: preliminary negative vs canonical — motivated by the 2026-07-02 Garrido/David meeting debate on whether history/memory alone helps; suggests it doesn't (unlike DMLPA/real-KAN, which use attention or splines, not just recurrence) |
| DMLPA (transformer-over-history, `scripts/dmlpa_extractor.py`) | Track A and Track B | Track A screening (2 seeds x 40k); Track B sidecar (5 seeds x 60k, h104); same-run fair bakeoff on v9/history (5 seeds x 60k) | Track A: **mixed/degenerate**. Track B sidecar: **beats static** on the sidecar primary metric (`order_level_ret_mean=0.005505` vs best static `0.005214`, 4/5 seeds) and on Garrido/Excel ReT (`order_ret_excel=0.005721` vs `0.005428`), but **does not exceed canonical PPO+MLP**. Fair bakeoff: `ppo_mlp=0.005920` > `ppo_dmlpa_positional=0.005871` > `ppo_mlp_history=0.005832` on Excel ReT | Closed for Paper 1 headline; robustness sidecar |
| KAN-inspired RBF sidecar (`scripts/kan_extractor.py`) | Track B (`control_v1`, canonical contract) | Confirmatory with linear skip (5 seeds x 60k); no-skip smoke (1 seed x 10k) | Linear-skip sidecar matches the positive Track B pattern but cannot prove KAN is necessary. No-skip smoke is also positive (`order_level_ret_mean=0.005580` vs best static `0.005123`), but it is only 1 seed/10k and not confirmatory. | Appendix/response-letter only; do not headline |
| Real KAN (pykan, official) | Supervised surrogate (147-cell static frontier); **Track B online RL under both 8D-completo and post-CDC-only**, confirmed (`scripts/real_kan_extractor.py`, `run_track_b_real_kan_sidecar.py`) | Surrogate: N=147, 118/29 split. RL: 8D completo 10 seeds x 60k; post-CDC-only 5 seeds x 60k (both 12 eval episodes) | Surrogate: R²=0.998 vs tuned MLP 0.855. RL 8D: 10/10 seeds positive vs canonical PPO+MLP, mean `+0.000041`, CI `[+0.000022,+0.000059]`, cost ~0.97-0.99. RL post-CDC: 4/5 seeds positive vs PPO+MLP post-CDC, mean `+0.000057` (slightly larger gap), cost at the ceiling `1.0` | **Confirmed strong architecture sidecar under both decision-surface variants** — clears the ReT bar cleanly in both, but the cost gap widens under post-CDC, so it still does not replace the manuscript spine |
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

Track B: canonical PPO+MLP remains the strongest architecture. The
DMLPA sidecar (`outputs/experiments/track_b_dmlpa_sidecar_2026-07-03/full8d_5seed_60k_h104/`)
confirms that David's transformer-over-history extractor can exploit
the corrected full-8D Track B contract and beat static policies. On the
sidecar primary metric (`order_level_ret_mean`), DMLPA scores 0.005505 vs
best static 0.005214, with 4/5 seeds positive. On Garrido/Excel ReT, DMLPA
scores 0.005721 vs best static 0.005428. The same-metric comparison to the
current canonical PPO+MLP headline should therefore use the Excel row:
0.005721 vs about 0.005898, a gap of about -0.000177. Do not compare the
sidecar primary `order_level_ret_mean=0.005505` directly against the canonical
`order_ret_excel=0.005898` headline as if they were the same metric. The
RBF-KAN sidecars are architecture-adjacent evidence that the win is not fragile
to a different feature extractor. The no-skip version is promising in a tiny
smoke, but not confirmatory; the linear-skip version cannot be sold as a
genuinely different architecture succeeding independently.

The official pykan Real-KAN result is now confirmed at 10 seeds, matched
exactly to the canonical PPO+MLP protocol: 10/10 seeds positive, mean paired
delta +0.000041, 95% CI [+0.000022, +0.000059] (clean, not borderline). Its
higher shift-utilization cost (~0.97-0.99 vs canonical's ~0.68) means it
should be treated as a confirmed strong architecture sidecar, not an
automatic replacement for the paper's PPO+MLP spine.

## What remains open (candidates, not commitments)

1. **DONE (2026-07-03): same-run history fairness check.** The completed
   Track B DMLPA sidecar used the canonical v7 contract, while David's
   notebook compares history-aware policies on v9 plus previous-action
   history. The matched three-arm bakeoff (`ppo_mlp`, `ppo_mlp_history`,
   `ppo_dmlpa_positional`) has now run under identical seeds, evaluation
   episodes, and Excel/order-level ReT reporting. Result: plain `ppo_mlp`
   remains best on Garrido/Excel ReT (0.005920), followed by
   `ppo_dmlpa_positional` (0.005871) and `ppo_mlp_history` (0.005832). This
   closes the collaborator-facing fairness check: history/attention is useful
   enough to beat static, but does not promote over MLP for Paper 1.
2. **DONE (2026-07-03): a real KAN as an actual PPO policy/value network,
   confirmed at 10 seeds.** Built `scripts/real_kan_extractor.py`
   (`RealKANFeaturesExtractor`), wrapping the official `kan.KAN` class as an
   SB3 `BaseFeaturesExtractor`. Key finding: pykan's `KAN.forward` works fine
   with standard autograd + Adam (no LBFGS needed) — the earlier assumption
   that it would need a from-scratch policy wrapper was wrong. The real
   obstacle was pykan's `save_act`/`symbolic_enabled` interpretability
   bookkeeping, which made a single forward pass ~160x slower than needed for
   online RL; disabling both makes a normal-scale PPO loop feasible. See
   `docs/REAL_KAN_SIDECAR_PREREGISTRATION_2026-07-03.md`,
   `docs/REAL_KAN_SIDECAR_CONFIRMATORY_VERDICT_2026-07-03.md`, and
   `docs/REAL_KAN_10SEED_EXTENSION_PREREGISTRATION_2026-07-03.md`. The
   10-seed/60k result is clean (CI does not touch zero) and this is now the
   strongest architecture sidecar found this session, but the ~45% higher
   shift-utilization cost means it stays a sidecar, not a spine replacement.
3. **SAC/TD3** remain untried by design (algorithmic choice is explicitly
   not the tested mechanism per the manuscript's own framing).

## Bottom line

No architecture change has overturned Track A's negative result. For Track B,
DMLPA is closed for the Paper 1 headline: it is a useful robustness result, not
a replacement for PPO+MLP. Real-KAN is now a **confirmed** positive sidecar
(10/10 seeds, clean CI) and should be discussed as the direct, rigorous
response to Garrido's KAN concern — the strongest architecture-alternative
evidence of the whole bakeoff. The manuscript spine should still remain
action-space alignment rather than architecture novelty: Real-KAN's ReT gain
is real and clean, but it costs ~45% more shift/dispatch utilization than
canonical PPO+MLP, so it is not superior on the operational tradeoff, only on
the single ReT metric.
