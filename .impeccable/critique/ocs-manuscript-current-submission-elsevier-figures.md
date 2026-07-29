# Impeccable critique — manuscript figure set (2026-07-02)

Target: `docs/manuscript_current/submission/elsevier/figures/` (7 figures).
Register: brand (PRODUCT.md). Method: two isolated assessments — A (design
review vs brand.md + DESIGN.md, no detectors) and B (deterministic:
grayscale luminance, WCAG contrast, font-size floor at print scale,
palette/fontsize consistency grep, resolution/format).

## Findings → resolution (all applied via scripts/build_manuscript_figures.py)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | fig4 off-family: default matplotlib, embedded title, PPO as red star (red reserved for boundary), PNG-only 160dpi meta / 298 effective dpi, not pipeline-generated | Rebuilt inside the pipeline from `docs/track_b_q1_stats_2026-07-02_final/pareto_points.csv`: STIX serif, no title, PPO = recovery-green star w/ dark edge, nondominated statics as open circles, log-x, vector PDF + 300dpi PNG; `.tex` ref made extensionless |
| 2 | High | Zero uncertainty encoding across the set despite seed-clustered inference story | fig3: 95% seed-clustered CI whisker on PPO point (delta CI [0.000389,0.000463] translated). fig6: 95% CIs per arm computed from per-seed paired deltas in `outputs/experiments/track_b_ablation_8d_final_2026-07-01/*/seed_metrics.csv` — joint [0.321,0.412], downstream [0.385,0.473], shift [0.344,0.409] (×10⁻³); seed-level means reproduce manuscript deltas exactly. Captions define the whiskers |
| 3 | High | fig6 value label collided with y-axis; embedded conclusion title | Labels above whiskers, xlim padding, title removed → conclusion moved to caption |
| 4 | Fail | fig2 font floor: risk codes ~4.1pt, node body ~4.8pt at print | All fig2 text raised (min now 7.6pt figure-size ≈ 5.1pt print); risk codes darkened 0.45→0.35 |
| 5 | Fail | fig3 annotation #E69F00 on white = 2.25:1 contrast | Text recolored to 0.25 dark grey; orange kept only for the arrow (linkage) |
| 6 | Med | fig2 orange/vermilion conflation (dispatch valves vs bottleneck callout) | Callout boxed (white fill, vermilion border) + heavier arrow — shape now separates the two roles |
| 7 | Med | #2e7d32 off-palette (duplicated #009E73 role) | Replaced with Okabe-Ito GREEN |
| 8 | Med | fig5 PiYG off-palette; black-on-dark cell annotation | Custom diverging ramp vermilion→white→recovery-green; white text on |v|>0.0005 cells; boundary label enlarged 6.8→7.4 |
| 9 | Med | fig1 decorative non-semantic pastels; wasted vertical space; color-only track cue | Neutral fills, tinted terminals only (vermilion=boundary outcome, green=recovery outcome); canvas tightened; Track B arrows heavier (grayscale cue) |
| 10 | Med | fig3 arrow struck value label; floating reference label; 6-digit axis | Curved arrow to dot underside; label rotated alongside dashed line mid-plot; axis rescaled ×10⁻³ (fig4/fig6 too) |
| 11 | Med | fig7 unanchored side note; caveat prose bolded inside terminal node; caution-yellow child fills | Leader line added; node split into bold identity + small grey caveat line; children neutral w/ FR child green-edged |
| 12 | Low | fig2 Demand label / Op12→Op13 arrow collision; band label / Op8→Op9 arrow collision | Arrow rerouted to right edge; band labels given white halo above arrows |

## Accepted residuals (disclosed, not fixed)
- fig5 grayscale sign near zero (Δ3.8/255): rescued by printed signed values,
  bold-if-positive, and the dashed vermilion boundary box.
- fig3 SKY/ORANGE grays collapse (Δ4.2): category identity carried by
  y-axis position + per-point labels, not hue.
- Several 3:1–4.5:1 contrast pairs on small bold annotations (band labels,
  vermilion callouts): at figure-annotation role, with redundant text.
- Cross-figure pastel drift between fig1/fig2 largely mooted (fig1 now neutral).

Recompiled `main.tex` with tectonic: clean; remaining overfull boxes < 14pt.
