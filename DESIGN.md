---
name: "SCRES-IA Q1 Manuscript"
description: "A restrained, evidence-first design system for a Garrido-grounded SCRES manuscript and figure set."
colors:
  evidence-blue: "#0072B2"
  service-sky: "#56B4E9"
  recovery-green: "#009E73"
  caution-orange: "#E69F00"
  boundary-vermilion: "#D55E00"
  audit-purple: "#CC79A7"
  neutral-ink: "#1F2933"
  neutral-rule: "#7F7F7F"
  neutral-paper: "#FFFFFF"
typography:
  manuscript-body:
    fontFamily: "Latin Modern Roman, Computer Modern, Times New Roman, serif"
    fontSize: "12pt"
    fontWeight: 400
    lineHeight: 1.35
  figure-title:
    fontFamily: "STIXGeneral, DejaVu Serif, serif"
    fontSize: "17pt"
    fontWeight: 700
    lineHeight: 1.1
  figure-body:
    fontFamily: "STIXGeneral, DejaVu Serif, serif"
    fontSize: "8pt"
    fontWeight: 400
    lineHeight: 1.15
  code-label:
    fontFamily: "Latin Modern Mono, Courier New, monospace"
    fontSize: "10pt"
    fontWeight: 400
    lineHeight: 1.2
rounded:
  none: "0px"
  figure-node: "6px"
  callout: "8px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "36px"
components:
  figure-node:
    backgroundColor: "{colors.neutral-paper}"
    textColor: "{colors.neutral-ink}"
    rounded: "{rounded.figure-node}"
    padding: "8px 12px"
  evidence-table:
    backgroundColor: "{colors.neutral-paper}"
    textColor: "{colors.neutral-ink}"
    rounded: "{rounded.none}"
    padding: "6px 8px"
  boundary-callout:
    backgroundColor: "{colors.neutral-paper}"
    textColor: "{colors.boundary-vermilion}"
    rounded: "{rounded.callout}"
    padding: "8px 12px"
---

# Design System: SCRES-IA Q1 Manuscript

## Overview

**Creative North Star: "The Forensic Benchmark Brief"**

The manuscript should read like a precise engineering brief, not a promotional
AI paper. Its visual system exists to make the evidence chain hard to miss:
Garrido-Rios thesis lineage, final Q1 evidence bundle, comparator hierarchy,
uncertainty, and claim boundary. The design voice is restrained, but not bland;
it uses color and diagrams only where they clarify the scientific argument.

This is a publication design system, not a product UI system. LaTeX remains the
source of truth, and every figure must also be suitable for a Word Online or
Overleaf port. Figures should be self-contained, with labels strong enough to
survive grayscale printing and enough whitespace to remain readable inside a
journal column or reviewer PDF.

**Key Characteristics:**
- Evidence-first hierarchy: result, comparator, uncertainty, boundary.
- Thesis-grounded visuals: topology and ReT lineage are redrawn, not replaced
  by decorative diagrams.
- Colorblind-safe semantic colors from the Okabe-Ito family.
- No motion, no decorative shadows, no AI/SaaS visual tropes.
- Figure code is reproducible from `scripts/build_manuscript_figures.py`.

## Colors

The palette is functional and semantic: blue marks baseline/control structure,
green marks validated recovery gain, orange/vermilion marks warning and
boundary cases, purple marks audit or metric-lineage structure, and neutral ink
carries the prose.

### Primary
- **Evidence Blue**: used for Track A, DES structure, and stable benchmark
  framing.
- **Recovery Green**: used for positive PPO or adaptive-recovery evidence.

### Secondary
- **Service Sky**: used for service/fill/accessory positive signal when a
  second non-green positive color is needed.
- **Audit Purple**: used sparingly for metric lineage and audit structure.

### Tertiary
- **Caution Orange**: used for comparator or warning states that are not failure.
- **Boundary Vermilion**: used for severe-regime boundary cases, bottleneck
  annotations, and disallowed overclaim warnings.

### Neutral
- **Neutral Ink**: primary manuscript and figure text.
- **Neutral Rule**: dividers, arrows, low-emphasis labels, and non-claim
  structure.
- **Neutral Paper**: white page and figure background.

### Named Rules

**The Semantic Color Rule.** Colors must encode meaning, not decoration. If a
color does not correspond to Track A, PPO gain, comparator, audit, or boundary,
remove it.

**The Boundary Is Visible Rule.** Severe/h52 and other claim boundaries must be
marked in vermilion or with redundant labels. Do not hide boundary cases in
footnotes or pale gray.

## Typography

**Display Font:** STIXGeneral or DejaVu Serif for generated figure titles.
**Body Font:** Latin Modern Roman / Computer Modern through LaTeX.
**Label/Mono Font:** Latin Modern Mono for code-style configuration labels.

**Character:** Scholarly, technical, and calm. The typography should feel closer
to a precise methods appendix than to a pitch deck. Monospace is reserved for
literal configuration names, metric identifiers, and artifact paths.

### Hierarchy
- **Figure Title** (700, 17pt, tight line height): one explanatory sentence at
  the top of generated figures.
- **Section Heading** (LaTeX defaults): journal-appropriate hierarchy; avoid
  oversized display type in the manuscript body.
- **Body** (12pt, 1.35 line height): long-form scholarly prose with clear
  paragraphing.
- **Figure Body** (7-9pt): labels, node text, and annotations; must remain
  legible after PDF scaling.
- **Code Label** (10pt mono): metric names, action contracts, and artifact
  paths only.

### Named Rules

**The No Hero Type Rule.** This is a journal article, not a landing page. Do not
use hero-scale display typography outside generated figure titles.

**The Metric Name Rule.** Exact metric names belong in monospace; prose claims
belong in roman text. Do not blur the two.

## Elevation

The manuscript is flat by default. Depth is conveyed through table rules,
tonal fills, labeled bands, and controlled line weight, not shadows. Generated
figures may use light filled nodes and thin borders; they should never use
drop-shadow card stacks or decorative glass effects.

### Named Rules

**The Print-First Rule.** If a figure depends on shadow, blur, or transparency
to communicate structure, redesign it with labels, rules, and tonal fills.

## Components

### Manuscript Figures
- **Shape:** labeled nodes with modest corners (6px equivalent), thin borders,
  and redundant text labels.
- **Color:** semantic Okabe-Ito colors only.
- **States:** no interactive states; Word/Overleaf exports must preserve the
  same information.
- **Use:** topology, ReT lineage, gap decomposition, Pareto view, heatmap, and
  ablation figures.

### Evidence Tables
- **Shape:** `booktabs` tables with no vertical rules unless absolutely needed.
- **Color:** no background color in manuscript tables by default.
- **Density:** compact enough for reviewers to scan, but not so dense that CI
  and comparator family disappear.
- **Use:** primary metric panel, evidence map, observation classes, and
  cross-regime matrix.

### Boundary Callouts
- **Shape:** prose callouts in text or figure labels, not decorative boxes by
  default.
- **Color:** vermilion in generated figures, explicit boundary wording in prose.
- **Use:** severe/h52 boundary, no-cost-win caveat, non-anticipation caveat,
  and fidelity-gate scope.

### Artifact References
- **Style:** exact paths in monospace in docs; concise citations in manuscript.
- **Use:** reviewer-facing docs and status reports. In the manuscript, artifact
  paths should move to supplement unless necessary for reproducibility.

## Do's and Don'ts

### Do:
- **Do** make every visual claim trace to `docs/track_b_q1_stats_2026-07-02_final/`
  or a named E1-E6 artifact.
- **Do** keep Track A boundary evidence and Track B positive evidence visually
  distinct.
- **Do** show severe-regime mixed behavior as a real boundary case.
- **Do** regenerate figures through `scripts/build_manuscript_figures.py` after
  changing figure data, labels, or style.
- **Do** keep LaTeX includes portable for Overleaf by omitting hard-coded local
  paths.

### Don't:
- **Don't** use perfect-fill, zero-backorder, 57% cost-win, universal RL-law,
  anticipation, retained-learning, or closed-generalization wording unless a
  new evidence bundle explicitly promotes it.
- **Don't** use SaaS/pitch-deck visual language: hero metrics, purple/blue AI
  gradients, nested cards, glassmorphism, or glowing dark-mode panels.
- **Don't** make Track B look thesis-faithful in decision-variable scope; call
  it a DES-preserving operational extension.
- **Don't** encode findings by color alone. Use labels, signs, table values, or
  direct annotations.
- **Don't** introduce figure aesthetics that cannot survive Word Online,
  Overleaf, and grayscale reviewer PDFs.
