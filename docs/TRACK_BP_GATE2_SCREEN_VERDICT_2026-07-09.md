# Track B-P Gate 2 Screen Verdict (2026-07-09)

**Headline: PPO on the preventive contract does not merely convert the static preventive
oracle — it quadruples it, and it does so with selective (10–20%) buffer holding rather
than blanket buffering. Attribution between preventive and adaptive channels requires the
8D-vs-11D contract ablation (launched; pending).**

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
eval — the preventive increment is PPO_11D − PPO_8D. Launched as
`track_bp_g2_cellA_8d_ablation_2026-07-09` (`--contract track_b`).

Interpretation matrix fixed in advance:
- PPO_11D ≈ PPO_8D → the buffers add nothing on top of adaptation; prevention remains
  practically subsumed even where the static channel exists.
- PPO_11D − PPO_8D > 0 (CI95 excluding 0) → genuine preventive increment converted by
  learning; size it against the +0.0299 static oracle.

## Guardrails

- Screen scale only (3 seeds × 30k): no confirmatory claims; 5-seed × 60k confirm needed
  before any paper-facing number.
- Buffer holding is unpriced in reward — a holding-cost sensitivity is mandatory before
  claiming operational value (PPO's low holding makes this likely favorable, but measure).
- These cells are engineered stress regimes (freq/impact multipliers), not Garrido-native
  intensities; frame any claim as regime-conditional.
- Not part of the current manuscript (paper-2 / extension lane).
