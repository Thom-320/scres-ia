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

## Regime-breadth frontier (R21 freq × impact grid, Gate-1 oracle, n=24 each)

always−never episode ReT (positive count /24; * = bootstrap CI95 excludes zero):

| freq \ impact | ×1 | ×2 | ×3 | ×4 |
|---|---|---|---|---|
| ×1 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| ×2 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| ×4 | 0 (0) | 0 (1) | +0.0023 (3) | **+0.0068 (6)*** |
| ×8 | 0 (0) | +0.0018 (3) | **+0.0174 (8)*** | **+0.0299 (14)*** |

The static preventive channel requires BOTH frequent AND long outages — compounding
starvation. It is exactly null across the whole natural-to-moderate region (freq ≤2 or
impact ≤2), turns on at freq×4/impact×4, and grows monotonically. At Garrido-native
intensity, prevention remains worthless — fully consistent with the paper's boundary and
with the thesis's own H2 (buffer moderation under increased risk). Runs:
`track_bp_breadth_r21_f{1,2,4,8}_i{1,2,3,4}_2026-07-09`.

## R11 re-attribution: RETRACTED as prevention evidence

Buffers-only forced posture: exact zero (real 0/62, placebo 0/40, DiD +0.000000). The
original tier's response surface shows calm < medium ≈ max_prep — the delta was the calm
arm hurting (de-preparation harm), not preparation helping. Details in the Gates 0/1
verdict update. Cell B's episode-null is thereby explained; the preventive claim now rests
solely, and cleanly, on the R21-compounding cell + contract ablation.

## Confirmatory scale: 5 seeds × 60k (both contracts)

Runs `track_bp_confirm_cellA_{11d,8d}_5seed_60k_2026-07-09`. Preventive increment
(PPO_11D − PPO_8D, paired per eval episode, same training seed):

| seed | 11D ReT (holding) | 8D ReT | increment | bootstrap CI95 | pos/24 |
|---|---|---|---|---|---|
| 1 | 0.3397 (0.191) | 0.3114 | +0.0284 | [+0.0169, +0.0408] | 24/24 |
| 2 | 0.3393 (0.134) | 0.3106 | +0.0287 | [+0.0162, +0.0423] | 21/24 |
| 3 | 0.3409 (0.218) | 0.2959 | +0.0450 | [+0.0306, +0.0599] | 24/24 |
| 4 | 0.3407 (0.270) | 0.3223 | +0.0183 | [+0.0113, +0.0264] | 24/24 |
| 5 | 0.3402 (0.217) | 0.3182 | +0.0221 | [+0.0142, +0.0307] | 24/24 |

**Pooled +0.028488, seed-clustered t-CI95 [+0.015813, +0.041163], 5/5 seeds positive
(117/120 episodes), mean holding 0.206.**

Honest scale note (the C23 lesson applied in reverse): the increment shrinks from the
screen's +0.053 to +0.028 because the 8D adaptive baseline improves more with training
(0.269–0.291 → 0.296–0.322) than the 11D arm (0.332–0.335 → 0.339–0.341) — mature
adaptation reactively closes part of the gap. Unlike C23, the effect CONFIRMS rather than
evaporates: every seed's CI95 excludes zero and the pooled increment equals **95% of the
full static blanket oracle** (+0.0299), delivered on top of a stronger adaptive baseline
at one-fifth the holding. The 11D total advantage over the neutral clock at confirmatory
scale is +0.125 (0.340 vs 0.215).

## Real-KAN sidecar (official pykan extractor, 3 seeds × 60k, Cell A)

Run `track_bp_kan_cellA_11d_3seed_60k_2026-07-09` (`--features-extractor real_kan`),
same eval seeds as the PPO confirm:

| Arm | episode ReT | holding |
|---|---|---|
| PPO_11D (MLP, 5 seeds) | 0.3402 | 0.206 |
| **KAN_11D (pykan, 3 seeds)** | **0.3403** (0.3394/0.3410/0.3406, extremely tight) | **0.171** |
| PPO_8D (adaptive-only) | 0.3117 | — |
| always_prepared | 0.2448 | 1.000 |
| never_prepared | 0.2149 | 0.000 |

KAN captures **100.1% of PPO's gain** over the neutral clock and delivers the same
preventive increment over the adaptive-only arm (+0.0286 vs PPO's +0.0285) at LOWER
holding (0.171 vs 0.206). In the preventive domain the Real-KAN is not a partial-capture
sidecar (its historical 76–82% pattern on the adaptive lanes) — it fully matches PPO.
Reading: the preventive channel is **architecture-robust**; the mechanism is the
contract/physics, not the network family — consistent with the program's standing
architecture finding (clairvoyant/efficiency ladder: architecture shapes cost profiles,
not access to channels). Minor caveat: the 8D baseline is MLP-PPO; a KAN_8D arm would
make the architecture-matched decomposition fully symmetric (not run — the cross-
architecture tie at 11D with three tightly-clustered seeds already serves the sidecar
role).

## Guardrails

- Screen scale only (3 seeds × 30k): no confirmatory claims; 5-seed × 60k confirm needed
  before any paper-facing number.
- Buffer holding is unpriced in reward — a holding-cost sensitivity is mandatory before
  claiming operational value (PPO's low holding makes this likely favorable, but measure).
- These cells are engineered stress regimes (freq/impact multipliers), not Garrido-native
  intensities; frame any claim as regime-conditional.
- Not part of the current manuscript (paper-2 / extension lane).
