# Track B-P Gate 2 Screen Verdict (2026-07-09)

**Headline: the preventive channel is real, learnable, and large. The contract ablation
isolates a pure preventive increment of +0.053 episode ReT (PPO_11D − PPO_8D, paired,
every seed CI95 > 0, 70/72 episodes positive) — 178% of the static blanket-buffer oracle,
achieved with only 10–21% buffer holding. Learning extracts MORE from Garrido's buffer
lever than any static posture can, because it times and sizes it.**

## The decomposition (Cell A, R21 compound starvation)

| Policy stack | episode ReT | increment |
|---|---|---|
| neutral clock (never_prepared) | 0.2149 | — |
| + static blanket buffers (always_prepared) | 0.2448 | +0.0299 (preventive, static) |
| + adaptive learning, 8D `track_b_v1` PPO | 0.2690–0.2914 | +0.0658 over neutral (adaptive) |
| + preventive lever learned, 11D `track_bp_v1` PPO | 0.3318–0.3354 | **+0.0533 on top of 8D (preventive, dynamic)** |

Preventive increment per training seed (paired per eval seed, bootstrap CI95):
seed 1 +0.053124 [+0.035168, +0.073308] (23/24 pos); seed 2 +0.043953
[+0.031139, +0.057754] (24/24); seed 3 +0.062748 [+0.043017, +0.083653] (23/24).
Pooled +0.053275. Of the total 11D-vs-neutral gain (+0.120), roughly 55% is the known
adaptive channel and 45% is the new preventive channel.

Runs: `track_bp_g2_cellA_8d_ablation_2026-07-09` (8D arm) vs
`track_bp_g2_cellA_r21starv_2026-07-09` (11D arm), identical cell/seeds/eval.

Protocol: Gate 2 of `docs/TRACK_BP_PREREGISTRATION_2026-07-08.md`, unlocked by
`docs/TRACK_BP_GATES_0_1_VERDICT_2026-07-09.md`. Runs:
`outputs/experiments/track_bp_g2_cell{A,B}_*_2026-07-09/` (3 seeds × 30k, canonical PPO
hparams, obs `v10_no_regime_forecast`, contract `track_bp_v1`, CRN eval on 24 episodes,
clock-policy oracle re-run on the identical eval seeds). Numbers recomputed from
`gate2_rows.csv`, not the summaries.

## Cell A — R21 compound starvation (freq×8, impact×4)

| Arm | episode ReT | vs never | holding |
|---|---|---|---|
| never_prepared | 0.2149 | — | 0.00 |
| calendar_prepared | 0.2293 | +0.0144 | 0.19 |
| always_prepared | 0.2448 | +0.0299 | 1.00 |
| **PPO seed 1** | **0.3347** | **+0.1199 CI95 [+0.1029, +0.1353]** | 0.167 |
| **PPO seed 2** | **0.3354** | **+0.1205 CI95 [+0.1057, +0.1344]** | 0.098 |
| **PPO seed 3** | **0.3318** | **+0.1169 CI95 [+0.0939, +0.1361]** | 0.208 |

- Conversion of the pre-registered oracle (always−never): **3.9–4.0×** across seeds.
- PPO beats the blanket-buffer posture itself by +0.087 to +0.091 (+36–37% relative)
  while holding only 10–21% of max buffers — the anticipation-check signature of timed,
  selective buffering rather than learned static buffering (falsifier #3 does NOT fire).
- All three seeds tightly clustered; every CI95 excludes zero by a wide margin.

## Cell B — R11 rare-long (freq×0.125, impact×8)

| Arm | episode ReT | vs never | holding |
|---|---|---|---|
| never_prepared | 0.6608 | — | 0.00 |
| always_prepared | 0.6608 | **+0.0000 exact** | 1.00 |
| PPO seeds 1–3 | 0.6632–0.6651 | +0.0024 to +0.0043 (2/3 CI95 > 0) | 0.05–0.16 |

- The static buffer channel is **episode-level null** in this cell: blanket buffers change
  nothing (always ≡ never exactly), even though Gate 0 found positive LOCAL causal DiD.
- Re-attribution needed for the Gate-0 R11 signal: the forced max_prep posture raises ALL
  11 dims (dispatch + shift + buffers). With buffers episode-null here, the local Gate-0
  gain likely flows through dispatch/shift preparation, or exists only at local-window
  scale. Recorded as an open attribution question — do not cite the R11 Gate-0 signal as
  a buffer-channel result until a buffers-only forced posture is run.
- PPO's small gain here is presumably adaptive (dims 1–8).

## The attribution caveat (why conversion > 1 is not the final answer)

The clock-policy oracle holds dims 1–8 at neutral; PPO optimizes all 11 jointly. The
+0.12 therefore bundles the known adaptive channel (dispatch/shift/ROP) with the new
preventive channel (lagged buffers). The decisive decomposition is the **contract
ablation**: PPO on plain `track_b_v1` (8D, no buffer dims), same cell, same seeds, same
eval — the preventive increment is PPO_11D − PPO_8D.

Interpretation matrix fixed in advance (before the ablation was read):
- PPO_11D ≈ PPO_8D → the buffers add nothing on top of adaptation; prevention remains
  practically subsumed even where the static channel exists.
- PPO_11D − PPO_8D > 0 (CI95 excluding 0) → genuine preventive increment converted by
  learning; size it against the +0.0299 static oracle.

**Outcome: the second branch, decisively** (headline section above): +0.0533 pooled,
all seeds CI95 > 0, 178% of the static oracle.

## Holding-cost sensitivity (mandatory check — done, favorable)

Adjusted score = episode ReT − λ_h × mean buffer-holding fraction, paired per episode
across the 72 (seed × eval-seed) pairs, pooled bootstrap CI95:

| λ_h | 11D − 8D adjusted increment | CI95 | 11D − always adjusted |
|---|---|---|---|
| 0.00 | +0.0533 | [+0.0432, +0.0637] | +0.089 |
| 0.05 | +0.0454 | [+0.0352, +0.0561] | +0.131 |
| 0.10 | +0.0375 | [+0.0275, +0.0483] | +0.173 |
| 0.20 | +0.0217 | [+0.0113, +0.0328] | +0.258 |
| 0.34 (crossover) | ≈0 | [−0.0108, +0.0116] | +0.373 |
| 0.50 | −0.0256 | [−0.0374, −0.0134] | +0.510 |

The preventive increment survives holding charges up to λ_h = 0.2 with CI95 excluding
zero (crossover λ_h* ≈ 0.34 = 0.0533/0.158 mean holding). Against the blanket posture the
dominance *grows* with λ_h, since `always_prepared` pays holding 1.0. The selective-timing
property is therefore not just cosmetic: it is what makes the channel robust to pricing.

## Guardrails

- Screen scale only (3 seeds × 30k): no confirmatory claims; 5-seed × 60k confirm needed
  before any paper-facing number.
- Buffer holding is unpriced in reward — a holding-cost sensitivity is mandatory before
  claiming operational value (PPO's low holding makes this likely favorable, but measure).
- These cells are engineered stress regimes (freq/impact multipliers), not Garrido-native
  intensities; frame any claim as regime-conditional.
- Not part of the current manuscript (paper-2 / extension lane).
