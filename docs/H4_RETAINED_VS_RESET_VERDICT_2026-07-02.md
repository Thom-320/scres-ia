# H4: Track B retained-vs-reset confirmatory run — verdict (2026-07-02)

**Status: DONE.** `outputs/benchmarks/retention_track_b/h4_track_b_confirmatory_2026-07-02/retention_track_b.json`,
run on the OVH VPS (`scripts/retention_track_b.py`), 10 seeds
(8101-8110), 8 online-adaptation cycles, 3,000 online timesteps per
cycle, `control_v1` reward, `adaptive_benchmark_v2` risk, two observation
arms (`obs_full` = full v7, `obs_hidden` = the same seven
regime/forecast fields masked as in E2).

## Result

Both arms clear the pre-registered bar (registry C12: claim only if the
retained-minus-reset CI95 lower bound is strictly positive).

| Arm | Comparison | n | mean | SEM | CI95 |
|---|---|---:|---:|---:|---|
| `obs_full` | retained − reset | 10 | +0.0000310 | 0.0000109 | [+0.0000095, +0.0000525] |
| `obs_full` | retained − frozen | 10 | +0.0000379 | 0.0000092 | [+0.0000199, +0.0000559] |
| **`obs_hidden`** | **retained − reset** | 10 | **+0.0000493** | 0.0000166 | **[+0.0000167, +0.0000819]** |
| `obs_hidden` | retained − frozen | 10 | +0.0000484 | 0.0000152 | [+0.0000186, +0.0000781] |

Per-seed retained-minus-reset deltas (`obs_hidden`, the preferred arm per
C12): `+0.0000016, +0.0000680, +0.0000893, +0.0000756, +0.0000825,
-0.0000203, +0.0000197, +0.0001510, +0.0000051, +0.0000202`. 9 of 10
seeds positive; the one negative seed (`8106`, -0.0000203) is small
relative to the positive seeds and does not flip the seed-clustered CI.

## Verdict on H4 / the L_{t-1} question

**Positive, at confirmatory scale, in both observation arms.** A policy
that continues online adaptation across sequential disruption cycles
retains a small but statistically detectable Excel-ReT advantage over a
policy that resets between cycles, and this advantage survives masking
the same privileged regime/forecast fields removed in E2 — if anything
it is *larger* in the masked arm (+0.0000493 vs +0.0000310), which rules
out the retained advantage being privileged-observation leakage across
cycles.

Scale discipline: this is a **third-decimal-order-of-magnitude effect**
relative to the Track B headline gain (+0.0000493 vs the canonical
+0.000426, roughly an order of magnitude smaller) and a **two-orders**
smaller effect than the Track A oracle headroom. It is a real, positive,
seed-clustered-significant result — not noise — but it is not comparable
in size to the paper's primary claim and must not be presented as such.

## What this does and does not license

**Licensed:** a single sentence in Discussion/Future Work reporting that
a confirmatory-scale online-retention probe finds a small positive
retained-vs-reset effect, disclosing the two-arms/masking control and
the effect's size relative to the headline result.

**Not licensed:**
- Reframing the paper's central claim around retained learning. The
  paper's contribution remains action-space alignment (Track A vs Track
  B); H4 is a minor, disclosed extension, not a pivot.
- Calling this "organizational learning" or claiming path-dependent
  accumulation across campaigns in the Levitt & March sense — the probe
  measures within-checkpoint online adaptation over 8 cycles of 3k
  timesteps, not cross-campaign institutional memory.
- Dropping the existing "retained cross-campaign learning is not
  claimed, remains future work" sentence in the Introduction — replace
  it with an accurate summary of what was actually measured, not silence
  it.

Safe wording for the manuscript (future-work paragraph, not a new
section):

> "A confirmatory-scale probe (10 seeds, 8 sequential online-adaptation
> cycles) finds a small positive retained-versus-reset advantage in
> Excel ReT (+0.0000493, 95% CI [+0.0000167, +0.0000819], masked
> observation arm), roughly an order of magnitude smaller than the
> Track B headline gain and robust to masking the same privileged
> regime/forecast fields removed in Section 4.5. This is preliminary
> evidence for a retained-adaptation channel distinct from the
> action-space alignment mechanism studied here; it is not a claim of
> cross-campaign organizational learning and is left for dedicated
> future work."

## Registry update

C12 (`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`) — **upgrade from
"Deferred until CI95>0" to "Supported at small effect size, both
arms."** Cite this verdict doc.
