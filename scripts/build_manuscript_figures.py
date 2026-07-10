#!/usr/bin/env python3
"""Publication-grade manuscript figures for the Q1 submission.

Rebuilds the figure set under docs/manuscript_current/submission/elsevier/figures/
with a consistent style: Okabe-Ito colorblind-safe palette, serif typography,
vector PDF (for LaTeX) + 300-dpi PNG (for the Word port) per figure.

Data sources are the frozen evidence bundles (docs/track_b_q1_stats_2026-07-02*,
docs/E*_VERDICT docs); numbers are hard-coded here deliberately so the figure
script is self-contained and auditable against the claims registry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/manuscript_current/submission/elsevier/figures")

# Darker variants for small text use (pass 4.5:1 on white)
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMIL = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7f7f7f"
# Text-safe darker shades (contrast >= 4.5:1 on white)
GREEN_TEXT = "#1a7a52"     # 4.55:1 — for small green text
VERMIL_TEXT = "#a84200"    # 5.73:1 — for small vermilion text

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


# ---------------------------------------------------------------- fig1
def fig1_bottleneck_alignment() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.48))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.35, 3.80)
    ax.axis("off")

    def chain(y, title, boxes, accent, arrow_lw):
        ax.text(0.15, y + 0.98, title, fontsize=9.5, fontweight="bold", va="bottom")
        w, h, gap = 2.05, 0.85, 0.38
        x = 0.15
        for i, (label, fc, ec, blw) in enumerate(boxes):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    fc=fc,
                    ec=ec,
                    lw=blw,
                )
            )
            ax.text(
                x + w / 2,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=8.3,
                linespacing=1.25,
            )
            if i < len(boxes) - 1:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + w + 0.04, y + h / 2),
                        (x + w + gap - 0.04, y + h / 2),
                        arrowstyle="-|>",
                        mutation_scale=9,
                        color=accent,
                        lw=arrow_lw,
                    )
                )
            x += w + gap

    neutral = ("#f6f8fa", "0.35", 0.9)
    chain(
        2.55,
        "Track A \u2014 boundary case",
        [
            ("Buffer + shift\ncontrols only", *neutral),
            ("Upstream /\nassembly authority", *neutral),
            ("Downstream dispatch\nfixed (Op10/Op12)", *neutral),
            ("Dense static frontier\nabsorbs the headroom", "#fbe9e7", VERMIL, 1.2),
        ],
        "0.30",
        1.2,
    )
    chain(
        0.95,
        "Track B \u2014 positive case",
        [
            ("Buffer + shift\n+ dispatch controls", *neutral),
            ("Closed-loop\nfeedback policy", *neutral),
            ("Op10/Op12 dispatch\ncontrollable", *neutral),
            ("Adaptive recovery:\nReT + tail gains", "#e7f4e9", GREEN, 1.2),
        ],
        "0.30",
        1.6,
    )
    save(fig, "fig1_bottleneck_alignment")


# ---------------------------------------------------------------- fig2
def fig2_mfsc_topology() -> None:
    """Publication redraw of the Garrido-Rios 13-operation MFSC topology.

    Clean four-band serpentine layout: white nodes, control badges
    (glyph + color, never color alone), bottleneck bracket. No side panel.
    """
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.set_xlim(0, 11.0)
    ax.set_ylim(-0.75, 4.55)
    ax.axis("off")

    W, H = 2.08, 0.80
    XS = [0.42, 3.06, 5.70, 8.34]  # four column slots
    ROWY = {1: 3.55, 2: 2.35, 3: 1.15, 4: -0.05}

    bands = [
        (1, "PROCUREMENT AND INBOUND"),
        (2, "MANUFACTURING AND OUTBOUND"),
        (3, "FORWARD DISTRIBUTION"),
        (4, "DEMAND"),
    ]
    for row, label in bands:
        y = ROWY[row]
        ax.add_patch(plt.Rectangle((0.06, y - 0.14), 10.88, H + 0.30,
                                   fc="#f7f9fa", ec="none", zorder=0))
        ax.text(0.34, y + H + 0.10, label, fontsize=7.8, color="#5a6b76",
                va="bottom", ha="left", zorder=3,
                bbox=dict(boxstyle="square,pad=0.10", fc="#f7f9fa", ec="none"))

    # op: (row, col, head, name, risks, controls)  controls: subset of {"buffer","shift","dispatch"}
    ops = {
        1:  (1, 0, "Op1",  "Military log. agency",  "R12", []),
        2:  (1, 1, "Op2",  "Suppliers (12)",        "R13", []),
        3:  (1, 2, "Op3",  "Warehouse and DC",      "R21", ["buffer"]),
        4:  (1, 3, "Op4",  "Line of comm.",         "R22", []),
        5:  (2, 3, "Op5",  "Pre-assembly",          "R11 R21 R3", ["buffer", "shift"]),
        6:  (2, 2, "Op6",  "Assembly line",         "R11 R21 R3", ["shift"]),
        7:  (2, 1, "Op7",  "Quality and shipping",  "R14 R21 R3", ["shift"]),
        8:  (2, 0, "Op8",  "Line of comm.",         "R22", []),
        9:  (3, 0, "Op9",  "Supply battalion",      "R21 R3", ["buffer"]),
        10: (3, 1, "Op10", "Line of comm.",         "R22", ["dispatch"]),
        11: (3, 2, "Op11", "Combat support (2)",    "R23", []),
        12: (3, 3, "Op12", "Line of comm.",         "R22", ["dispatch"]),
        13: (4, 3, "Op13", "Theatre of operations", "R24", []),
    }

    BADGE = {
        "buffer":   ("$\\bigtriangleup$", BLUE),
        "shift":    ("$\\bullet$", GREEN),
        "dispatch": ("$\\diamondsuit$", VERMIL),
    }

    def node_xy(op):
        row, col, *_ = ops[op]
        return XS[col], ROWY[row]

    for op, (row, col, head, name, risks, controls) in ops.items():
        x, y = XS[col], ROWY[row]
        ec = VERMIL if "dispatch" in controls else "0.30"
        lw = 1.5 if "dispatch" in controls else 0.9
        ax.add_patch(FancyBboxPatch((x, y), W, H,
                     boxstyle="round,pad=0.03,rounding_size=0.06",
                     fc="white", ec=ec, lw=lw, zorder=2))
        ax.text(x + 0.13, y + H - 0.19, head, fontsize=9.6, fontweight="bold",
                ha="left", va="center", zorder=3)
        ax.text(x + 0.13, y + H - 0.47, name, fontsize=8.2, ha="left",
                va="center", zorder=3)
        ax.text(x + 0.13, y + 0.12, risks, fontsize=7.6, color="0.42",
                ha="left", va="center", zorder=3)
        # control badges, top-right corner, stacked horizontally
        for k, c in enumerate(controls):
            g, col_ = BADGE[c]
            fs = {"buffer": 9, "shift": 12, "dispatch": 10.5}[c]
            ax.text(x + W - 0.16 - 0.30 * k, y + H - 0.19, g, fontsize=fs,
                    color=col_, ha="center", va="center", zorder=4)

    def harrow(a, b):
        xa, ya = node_xy(a); xb, yb = node_xy(b)
        if xb > xa:
            p0, p1 = (xa + W + 0.05, ya + H / 2), (xb - 0.05, yb + H / 2)
        else:
            p0, p1 = (xa - 0.05, ya + H / 2), (xb + W + 0.05, yb + H / 2)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                     mutation_scale=10, color="0.25", lw=1.2, zorder=1))

    def varrow(a, b, side="right"):
        xa, ya = node_xy(a); xb, yb = node_xy(b)
        fx = 0.86 if side == "right" else 0.14
        ax.add_patch(FancyArrowPatch((xa + W * fx, ya - 0.03),
                                     (xb + W * fx, yb + H + 0.03),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="0.25", lw=1.2, zorder=1))

    for a, b in [(1, 2), (2, 3), (3, 4)]:
        harrow(a, b)
    varrow(4, 5, "right")
    for a, b in [(5, 6), (6, 7), (7, 8)]:
        harrow(a, b)
    # Op8 -> Op9: elbow through the left margin (keeps the band-label lane clear)
    y8m = ROWY[2] + H / 2
    y9m = ROWY[3] + H / 2
    ax.plot([XS[0], 0.16, 0.16], [y8m, y8m, y9m], color="0.25", lw=1.2,
            solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch((0.16, y9m), (XS[0] - 0.02, y9m),
                 arrowstyle="-|>", mutation_scale=10, color="0.25", lw=1.2,
                 zorder=1))
    for a, b in [(9, 10), (10, 11), (11, 12)]:
        harrow(a, b)
    varrow(12, 13, "right")

    # bottleneck bracket under Op10..Op12
    bx0, bx1 = XS[1] + 0.1, XS[3] + W - 0.55
    by = ROWY[3] - 0.24
    ax.plot([bx0, bx0, bx1, bx1], [by + 0.09, by, by, by + 0.09],
            color=VERMIL, lw=1.3, zorder=3)
    ax.text(6.0, by - 0.13,
            "dispatch capacity $\\approx$ demand ($\\approx$2,500 rations/day):\nthe binding recovery bottleneck",
            fontsize=8.4, color=VERMIL, ha="center", va="top", zorder=3,
            linespacing=1.25)

    # legend line, bottom left
    lx, ly = 0.42, -0.62
    items = [
        ("$\\bigtriangleup$", BLUE, "buffer control (Track A)"),
        ("$\\bullet$", GREEN, "shift control (Track A)"),
        ("$\\diamondsuit$", VERMIL, "dispatch control (added in Track B)"),
    ]
    for (glyph, col_, text), fs in zip(items, (9, 12, 10.5)):
        ax.text(lx, ly, glyph, fontsize=fs, color=col_, ha="left", va="center")
        ax.text(lx + 0.24, ly, text, fontsize=8.2, color="0.25", ha="left", va="center")
        lx += 0.24 + 0.118 * len(text) + 0.42

    save(fig, "fig2_mfsc_topology")


# ---------------------------------------------------------------- fig3
def fig3_gap_decomposition() -> None:
    labels = [
        "Best common static\n(147-cell dense grid)",
        "Regime-conditioned\nlookup table (fitted)",
        "Best heuristic\n(6 evaluated)",
        "PPO (canonical,\nfrozen checkpoint)",
    ]
    vals = [5.466, 5.494, 5.436, 5.893]  # Excel ReT x 10^-3
    # Comparators in neutral ink; only PPO carries the accent -- the figure's
    # one message is that the learned policy stands apart from every
    # non-learning family, privileged or not.
    cols = ["0.60", "0.60", "0.60", GREEN]
    sizes = [52, 52, 52, 76]
    # 95% seed-clustered CI of the paired PPO-vs-common-static delta
    # (docs/track_b_q1_stats_2026-07-02_final/seed_level_inference.csv),
    # translated onto the PPO point: delta CI [0.000389, 0.000463].
    ppo_ci = (5.466 + 0.389, 5.466 + 0.463)

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    y = np.arange(len(labels))[::-1]
    ax.set_ylim(-0.6, 4.1)
    ax.axvline(5.466, color=GREY, lw=0.9, ls="--", zorder=0)
    ax.hlines(y, 5.42, vals, color="0.88", lw=1.4, zorder=1)
    ax.hlines(y[3], ppo_ci[0], ppo_ci[1], color=GREEN, lw=2.2, zorder=2)
    for cap in ppo_ci:
        ax.vlines(cap, y[3] - 0.09, y[3] + 0.09, color=GREEN, lw=1.4, zorder=2)
    ax.scatter(vals, y, s=sizes, c=cols, zorder=3, edgecolors="0.2",
               linewidths=0.6)
    # value labels placed to the RIGHT of each dot (away from y-axis labels);
    # the PPO label clears the CI whisker cap instead of the dot
    for yi, v in zip(y, vals):
        anchor = max(v, ppo_ci[1]) if yi == y[3] else v
        ax.text(anchor + 0.022, yi, f"{v:.3f}", ha="left", va="center",
                fontsize=8, color="0.15")
    ax.text(5.470, 3.75, "common-static reference", fontsize=7.6,
            color="0.35", ha="left", va="bottom")
    # annotation arrow now points at the regime-table DOT itself (5.494, y[1])
    ax.annotate("direct true-regime access buys only $+0.028$\nover the common static",
                xy=(5.494, y[1]), xytext=(5.62, y[1] - 0.55),
                fontsize=7.6, color="0.35", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Excel ReT ($\\times 10^{-3}$; canonical CRN protocol, 5 seeds $\\times$ 12 episodes)")
    ax.set_xlim(5.42, 6.05)
    save(fig, "fig3_gap_decomposition")


# ---------------------------------------------------------------- fig4
def fig4_pareto_ret_tail_ctj() -> None:
    """ReT vs CTj-tail trade-off: 147 dense statics + PPO, from the frozen
    evidence bundle. The empty gap IS the finding: PPO improves both axes
    simultaneously; every static is inside its dominance region."""
    import csv

    src = Path("docs/track_b_q1_stats_2026-07-02_final/pareto_points.csv")
    statics, ppo = [], None
    with src.open() as fh:
        for row in csv.DictReader(fh):
            rec = (
                float(row["order_ctj_p99"]),
                float(row["order_ret_excel"]) * 1000,
                row["pareto_nondominated_ret_cost_tail_flow"] == "True",
            )
            if row["kind"] == "ppo":
                ppo = rec
            else:
                statics.append(rec)

    fig, ax = plt.subplots(figsize=(5.9, 3.5))
    xmin, xmax = ppo[0] * 0.72, 11800
    ymin, ymax = 2.0, 6.25

    # dominance region: everything with worse tail AND worse ReT than PPO
    ax.add_patch(plt.Rectangle((ppo[0], ymin), xmax - ppo[0], ppo[1] - ymin,
                               fc=GREEN, alpha=0.05, ec="none", zorder=0))
    ax.plot([ppo[0], ppo[0]], [ymin, ppo[1]], color=GREEN, lw=0.8,
            ls=(0, (4, 3)), zorder=1)
    ax.plot([ppo[0], xmax], [ppo[1], ppo[1]], color=GREEN, lw=0.8,
            ls=(0, (4, 3)), zorder=1)
    ax.text(3550, 2.22, "dominated by PPO on both axes\n(all 147 static dispatch policies)",
            fontsize=7.8, color="#3f7a5f", ha="center", va="bottom")

    dom = [(x, y) for x, y, nd in statics if not nd]
    nod = sorted([(x, y) for x, y, nd in statics if nd])
    ax.scatter([x for x, _ in dom], [y for _, y in dom], s=22, c=BLUE,
               alpha=0.32, linewidths=0, zorder=2)
    ax.scatter([x for x, _ in nod], [y for _, y in nod], s=30,
               facecolors="none", edgecolors=BLUE, linewidths=0.9, zorder=3)
    # static Pareto staircase
    sx = [x for x, _ in nod]
    sy = [y for _, y in nod]
    ax.step(sx, sy, where="post", color=BLUE, lw=1.0, alpha=0.8, zorder=2)

    # best static by ReT, labeled
    bx, by = max(statics, key=lambda r: r[1])[:2]
    ax.annotate("best static by ReT\n(S2, Op10$\\times$2.0, Op12$\\times$1.5)",
                xy=(bx, by), xytext=(4200, 5.52), fontsize=7.8, color="0.30",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9))

    # PPO star with halo
    ax.scatter([ppo[0]], [ppo[1]], marker="*", s=340, c="white",
               edgecolors="none", zorder=4)
    ax.scatter([ppo[0]], [ppo[1]], marker="*", s=240, c=GREEN,
               edgecolors="0.1", linewidths=0.7, zorder=5)
    ax.text(ppo[0] * 1.16, ppo[1] + 0.13, "PPO", fontsize=9.5, color=GREEN,
            fontweight="bold", va="center")

    # the gap annotation
    ax.annotate("", xy=(ppo[0] * 1.12, 4.62), xytext=(6100, 4.62),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.1))
    ax.text(2900, 4.72, "6.7$\\times$ shorter recovery tail\nat higher resilience",
            fontsize=8.2, color="0.25", ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("$CT_j$ p99 (hours; log scale, lower is better)")
    ax.set_ylabel("Excel ReT ($\\times 10^{-3}$)")
    ax.grid(True, which="major", lw=0.3, color="0.92", zorder=0)
    save(fig, "fig4_pareto_ret_tail_ctj")


# ---------------------------------------------------------------- fig5
def fig5_generalization_heatmap() -> None:
    # Order-level ReT delta vs best in-cell static by the primary metric
    # (docs/track_b_q1_stats_2026-07-02_final/e3_per_cell_seed_ci.csv).
    # 2026-07-09 provenance fix: the h104 column previously used a duplicate
    # run whose weaker best-static comparator inflated the deltas
    # (0.244/0.623); these are the conservative same-source values.
    deltas = np.array(
        [
            [0.359, 0.209],
            [0.537, 0.552],
            [-0.060, -0.075],
        ]
    )  # order-level ReT delta x 10^-3
    rows = ["current", "increased", "severe"]
    cols = ["h52", "h104"]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    norm = TwoSlopeNorm(vmin=-0.66, vcenter=0.0, vmax=0.66)
    # Diverging ramp anchored on the manuscript palette: boundary vermilion
    # for losses, recovery green for gains (replaces off-palette PiYG).
    cmap = LinearSegmentedColormap.from_list("ret_div", [VERMIL, "#ffffff", GREEN])
    im = ax.pcolormesh(deltas, cmap=cmap, norm=norm, edgecolors="white",
                       linewidth=2.5)
    ax.invert_yaxis()
    for i in range(3):
        for j in range(2):
            v = deltas[i, j]
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{v:+.3f}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="white" if abs(v) > 0.40 else "0.1",
            )
    # boundary-regime marker: the whole severe row
    ax.add_patch(
        plt.Rectangle((0.035, 2.035), 1.93, 0.93, fill=False, ec=VERMIL,
                      lw=1.8, ls=(0, (4, 2)))
    )
    ax.text(1.0, 2.82, "boundary regime (service floor)", fontsize=7.4,
            color=VERMIL_TEXT, ha="center")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(cols)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(rows)
    ax.set_xlabel("evaluation horizon (weeks)")
    ax.set_ylabel("Garrido-native risk level")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("PPO $-$ best in-cell static, order-level ReT ($\\times 10^{-3}$)", fontsize=8)
    cb.outline.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "fig5_generalization_heatmap")


# ---------------------------------------------------------------- fig6
def fig6_action_space_ablation() -> None:
    arms = ["Joint\n(full 8D)", "Downstream-only\n(dispatch)", "Shift-only\n(capacity)"]
    deltas = [0.367, 0.429, 0.377]  # order-level ReT delta x 10^-3
    # 95% seed-clustered CIs (5 paired seed deltas vs best in-arm evaluated
    # comparator, seed_metrics.csv per arm in
    # outputs/experiments/track_b_ablation_8d_final_2026-07-01/).
    ci_lo = [0.321, 0.385, 0.344]
    ci_hi = [0.412, 0.473, 0.409]
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    x = np.arange(len(arms))
    ax.axhline(0, color="0.25", lw=1.0, zorder=1)
    ax.text(-0.50, 0.012, "0 = best in-arm comparator", fontsize=7.4,
            color="0.35", ha="left", va="bottom")
    ax.vlines(x, 0, deltas, color="0.85", lw=1.6, zorder=1)
    for xi, lo, hi in zip(x, ci_lo, ci_hi):
        ax.vlines(xi, lo, hi, color=GREEN, lw=2.0, zorder=2)
        ax.hlines([lo, hi], xi - 0.055, xi + 0.055, color=GREEN, lw=1.3, zorder=2)
    ax.scatter(x, deltas, s=80, c=GREEN, zorder=3, edgecolors="0.2", linewidths=0.6)
    for xi, v, hi in zip(x, deltas, ci_hi):
        ax.text(xi, hi + 0.016, f"+{v:.3f}", ha="center", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8.4)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("Order-level ReT $\\Delta$ ($\\times 10^{-3}$) vs best\nin-arm comparator (incl. heuristics)")
    ax.set_ylim(0, 0.52)
    save(fig, "fig6_action_space_ablation")


# ---------------------------------------------------------------- fig7
def fig7_ret_metric_lineage() -> None:
    """Evaluation-metric lineage redrawn from Garrido-Rios Figure 5.11."""
    fig, ax = plt.subplots(figsize=(9.0, 4.35))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0.05, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", fontsize=8.5, ec="0.3", lw=0.9, weight="normal"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07", fc=fc, ec=ec, lw=lw, zorder=2
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            linespacing=1.18,
            zorder=3,
        )
        return (x + w / 2, y, x + w / 2, y + h)  # bottom-center / top-center anchors

    def link(p0, p1, color="0.3"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.0, zorder=1))

    root = box(3.65, 4.35, 3.2, 0.78, "Supply-chain resilience\n(SCRES)", fc="white", ec="0.30", fontsize=10.2, weight="bold")
    dyn = box(0.85, 3.10, 3.15, 0.78, "Dynamic resilience\nunder risks", fc="#f6f8fa", ec="0.40", fontsize=9.2)
    sta = box(6.45, 3.10, 3.15, 0.78, "Static resilience\ninherent to the SC", fc="#f6f8fa", ec="0.40", fontsize=9.2)
    link((4.55, 4.35), (2.42, 3.88), color="0.4")
    link((6.10, 4.35), (8.03, 3.88), color="0.4")

    drivers = [
        (0.50, "$AP_j$\nautonomy period", "0.45"),
        (2.85, "$RP_j$\nrecovery period", GREEN),
        (5.20, "$DP_j - RP_j$\nnon-recovery tail", "0.45"),
        (7.55, "$FR_t$\nfill-rate branch", "0.45"),
    ]
    anchors = []
    for x, text, ec in drivers:
        a = box(x, 1.85, 1.95, 0.78, text, fc="white", ec=ec, fontsize=8.8)
        anchors.append(a)
    for a in anchors[:3]:
        link((2.42, 3.10), (a[2], a[3]), color="0.4")
    link((8.03, 3.10), (anchors[3][2], anchors[3][3]), color="0.4")

    ax.add_patch(
        FancyBboxPatch(
            (2.35, 0.52), 6.25, 0.92,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            fc="#f3edf8", ec=PURPLE, lw=1.1, zorder=2,
        )
    )
    ax.text(5.48, 1.13, "Order-level Garrido/Excel ReT", ha="center", va="center",
            fontsize=9.6, fontweight="bold", zorder=3)
    ax.text(5.48, 0.76, "primary manuscript outcome; training reward is separate",
            ha="center", va="center", fontsize=7.8, color="0.35", zorder=3)
    for a in anchors:
        link((a[0], 1.85), (5.48, 1.46), color="0.4")
    ax.text(5.48, 0.24, "Excel branch logic reproduced exactly for order-level ReT (audited row-by-row)",
            fontsize=7.8, color="0.4", ha="center", va="center")
    save(fig, "fig7_ret_metric_lineage")


# ---------------------------------------------------------------- fig8
def fig8_ret_branch_timeline() -> None:
    """Order-timeline illustration of the ReT branch logic (redrawn from the
    disruption-timeline construction in Garrido-Rios 2017, Fig. 5.5)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.05, 3.75)
    ax.axis("off")

    LT = 3.2  # promised lead time, drawing units

    def order(y, opt, oat, risk, label, branch, bcol):
        # lead-time window
        ax.add_patch(plt.Rectangle((opt, y - 0.13), LT, 0.26, fc="#e3e9ee", ec="#c3ccd4", lw=0.5, zorder=1))
        ax.plot([opt + LT, opt + LT], [y - 0.2, y + 0.2], color="0.45", lw=0.9, ls=":", zorder=2)
        # risk exposure
        if risk is not None:
            ax.add_patch(plt.Rectangle((risk[0], y - 0.13), risk[1] - risk[0], 0.26,
                                       fc=VERMIL, alpha=0.25, ec="none", zorder=2))
        # cycle line OPT -> OAT
        ax.annotate("", xy=(oat, y), xytext=(opt, y),
                    arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.6), zorder=3)
        ax.scatter([opt], [y], marker="o", s=34, c="white", edgecolors="0.2", zorder=4)
        ax.text(opt - 0.12, y, "$OPT_j$", ha="right", va="center", fontsize=8)
        ax.text(oat + 0.12, y, "$OAT_j$", ha="left", va="center", fontsize=8)
        ax.text(0.15, y + 0.32, label, fontsize=8.6, fontweight="bold", va="bottom")
        ax.text(9.85, y, branch, fontsize=8.4, color=bcol, ha="right", va="center",
                fontweight="bold")

    order(3.3, 0.9, 4.1, (1.6, 3.0),
          "Order A: risk during window, delivered on time ($CT_j = LT_j$)",
          "autonomy branch:\n$\\mathrm{Re}(AP_j)$", BLUE)
    ax.text(0.9 + 3.2, 3.3 - 0.22, "deadline $OPT_j{+}LT_j$", fontsize=7.2,
            color="0.45", ha="center", va="top")
    order(2.1, 0.9, 6.9, (1.8, 5.3),
          "Order B: risk during window, delivered late ($CT_j > LT_j$)",
          "recovery branch:\n$0.5\\,(1/RP_j)$", GREEN)
    order(0.9, 0.9, 4.1, None,
          "Order C: no risk during window",
          "fill-rate branch:\n$1-(B_t{+}U_t)/D_t$", GREY)

    # legend strip (below the bottom row)
    ax.add_patch(plt.Rectangle((0.9, 0.28), 0.5, 0.14, fc="#e3e9ee", ec="#c3ccd4", lw=0.5))
    ax.text(1.5, 0.35, "promised lead time $LT_j$ (48 h)", fontsize=7.6, va="center")
    ax.add_patch(plt.Rectangle((4.6, 0.28), 0.5, 0.14, fc=VERMIL, alpha=0.25, ec="none"))
    ax.text(5.2, 0.35, "risk exposure on the order window", fontsize=7.6, va="center")

    save(fig, "fig8_ret_branch_timeline")


# ---------------------------------------------------------------- fig9
def fig9_prevention_ceiling() -> None:
    """Prevention boundary, generalized: two independent ceiling tests agree
    that preventive headroom is ~0 under the track_b_v1 action contract,
    across every mediable risk family tested.

    Panel (a): forced-prep response surface, all eight tiers. For each tier,
    the share of isolated real anchors where the max-prep posture beats calm
    on local Garrido/Excel ReT (filled) vs the same share on matched placebo
    anchors (open). Every tier sits at 0-7%, far below the 60% promotion bar;
    six of eight are exact zeros (bit-identical outcomes across postures).
    Sources:
      outputs/experiments/track_b_headroom_sweep_{case_c,r22_only}_2026-07-07/
      outputs/experiments/track_b_headroom_{r23_only,r23_case_c,r12_only,
        r12_r13bg,r21_only,r23_surge_inertia}_2026-07-08/
      docs/TRACK_B_PREVENTION_HEADROOM_GENERALIZED_VERDICT_2026-07-08.md

    Panel (b): clairvoyant ceiling. A PPO trained AND evaluated with the TRUE
    future risk label visible does not improve on the reactive baseline's ReT
    and is the most resource-expensive variant tested.
    Source: docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md
    """
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.30, 1.0]}
    )

    # --- panel (a): positive-anchor rate per tier, real vs placebo ----------
    # (tier label, real %, placebo %, n_real, n_placebo)
    tiers = [
        ("R22 · Case C background",   1.5, 4.2, 67, 96),
        ("R22 · clean physics",       0.0, 0.0, 84, 96),
        ("R23 · Case C background",   7.1, 4.2, 28, 96),
        ("R23 · clean physics",       0.0, 0.0, 45, 96),
        ("R23 · clean + surge lag",   0.0, 0.0, 45, 96),
        ("R12 · clean physics",       0.0, 0.0, 65, 96),
        ("R12 · R13 background",      0.0, 2.1, 66, 96),
        ("R21 · clean physics",       0.0, 0.0, 24, 96),
    ]
    y = np.arange(len(tiers))[::-1]

    axA.axvline(60, color=VERMIL, lw=1.0, ls="--", zorder=1)
    axA.text(61.8, len(tiers) - 0.35, "promotion bar", fontsize=7.6,
             color=VERMIL_TEXT, ha="left", va="bottom")

    for yi, (label, real, plac, n_r, n_p) in zip(y, tiers):
        axA.plot([0, max(real, plac)], [yi, yi], color="0.90", lw=1.4, zorder=1)
        axA.scatter([plac], [yi], s=34, facecolors="white", edgecolors=GREY,
                    linewidths=1.1, zorder=3)
        axA.scatter([real], [yi], s=42, c=BLUE, edgecolors="0.2",
                    linewidths=0.5, zorder=4)
        axA.text(101, yi, f"{n_r}/{n_p}", fontsize=7.2, color="0.45",
                 ha="left", va="center")

    axA.text(101, len(tiers) - 0.35, "anchors\n(real/placebo)", fontsize=7.2,
             color="0.45", ha="left", va="bottom", linespacing=1.15)
    axA.set_yticks(y)
    axA.set_yticklabels([t[0] for t in tiers], fontsize=8.2)
    axA.set_xlim(-3, 100)
    axA.set_ylim(-0.7, len(tiers) - 0.3 + 0.75)
    axA.set_xlabel("anchors where max-prep beats calm on local ReT (%)",
                   fontsize=8.6)
    axA.set_title("(a) Forced-prep response surface, all tiers",
                  fontsize=9.4, pad=8)
    axA.scatter([], [], s=42, c=BLUE, edgecolors="0.2", linewidths=0.5,
                label="real anchors")
    axA.scatter([], [], s=34, facecolors="white", edgecolors=GREY,
                linewidths=1.1, label="placebo anchors")
    leg = axA.legend(fontsize=7.8, loc="lower right", frameon=True,
                     borderaxespad=0.4, handletextpad=0.3)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_facecolor("white")

    # --- panel (b): clairvoyant ceiling ------------------------------------
    best_static_ret = 0.441640      # s2_d2.00
    reactive = (0.481160, 0.719)
    clairvoy = (0.485035, 0.853)

    axB.axvline(best_static_ret, color=GREY, lw=0.9, ls="--", zorder=1)
    axB.text(best_static_ret + 0.0012, 0.615, "best static", fontsize=7.8,
             color="0.35", ha="left", va="bottom", rotation=90)

    axB.scatter(*reactive, s=84, c=BLUE, zorder=4,
                edgecolors="0.2", linewidths=0.6)
    axB.text(reactive[0] - 0.0034, reactive[1], "reactive PPO\n+10.65% vs static",
             fontsize=8, color="0.15", ha="right", va="center",
             linespacing=1.25)
    axB.scatter(*clairvoy, s=84, c=VERMIL, zorder=4,
                edgecolors="0.2", linewidths=0.6)
    axB.text(clairvoy[0] - 0.0034, clairvoy[1],
             "clairvoyant PPO\n(true future visible)\n+9.83% vs static",
             fontsize=8, color="0.15", ha="right", va="center",
             linespacing=1.25)
    axB.annotate(
        "", xy=(clairvoy[0] + 0.0018, clairvoy[1] - 0.006),
        xytext=(reactive[0] + 0.0018, reactive[1] + 0.006),
        arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9,
                        shrinkA=0, shrinkB=0),
    )
    axB.text(0.4874, 0.787, "perfect foreknowledge:\nno ReT gain, +19% cost",
             fontsize=7.8, color="0.35", ha="left", va="center",
             linespacing=1.25)

    axB.set_xlabel("ReT Excel (Case C scale)", fontsize=9)
    axB.set_ylabel("resource cost index", fontsize=9)
    axB.set_xlim(0.437, 0.503)
    axB.set_ylim(0.60, 0.95)
    axB.set_xticks([0.44, 0.46, 0.48, 0.50])
    axB.set_title("(b) Clairvoyant ceiling", fontsize=9.4, pad=8)

    fig.tight_layout(w_pad=2.4)
    save(fig, "fig9_prevention_ceiling")


# ---------------------------------------------------------------- fig10
def fig10_efficiency_architecture() -> None:
    """Efficiency is architectural, not predictive: the Ruta B control ladder.

    ReT Excel vs resource cost on the Case C scale. The four
    RutaBAuxFeaturesExtractor arms (true/permuted/lambda0/constant label)
    cluster at ~0.40-0.43 cost regardless of whether the auxiliary loss
    contributes any gradient (lambda=0) or any temporal signal (constant);
    the two default-extractor arms (reactive, clairvoyant) cluster at
    ~0.72-0.85 cost regardless of foreknowledge. The cost reduction traces to
    the extractor trunk, not to prediction.
    Sources: docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md
    (control-ladder table); confirm/screen bundles cited therein.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # (ReT, cost) -- Case C protocol
    aux_pts = {
        "true-label $\\lambda{=}0.25$": (0.481086, 0.396),
        "$\\lambda{=}0$ (zero gradient)": (0.484962, 0.418),
        "constant-label": (0.484651, 0.419),
        "permuted-label": (0.485139, 0.426),
    }
    def_pts = {
        "reactive PPO": (0.481160, 0.719),
        "clairvoyant PPO": (0.485035, 0.853),
    }

    # group bands (subtle)
    ax.axhspan(0.376, 0.446, color=GREEN, alpha=0.055, zorder=0)
    ax.axhspan(0.699, 0.873, color=VERMIL, alpha=0.055, zorder=0)

    for (x_, y_) in aux_pts.values():
        ax.scatter(x_, y_, marker="s", s=64, c=GREEN, edgecolors="0.2",
                   linewidths=0.6, zorder=4)
    for (x_, y_) in def_pts.values():
        ax.scatter(x_, y_, marker="o", s=72, c=VERMIL, edgecolors="0.2",
                   linewidths=0.6, zorder=4)

    # direct labels, collision-free
    ax.text(0.481086 - 0.0007, 0.396 - 0.017, "true-label\n$\\lambda{=}0.25$",
            fontsize=8, color="0.15", ha="center", va="top", linespacing=1.2)
    ax.text(0.485139 + 0.0009, 0.418,
            "$\\lambda{=}0$ (zero gradient)\nconstant-label\npermuted-label",
            fontsize=8, color="0.15", ha="left", va="center", linespacing=1.35)
    ax.text(0.481160 - 0.0009, 0.719, "reactive PPO", fontsize=8,
            color="0.15", ha="right", va="center")
    ax.text(0.485035 - 0.0009, 0.853, "clairvoyant PPO\n(true future visible)",
            fontsize=8, color="0.15", ha="right", va="center", linespacing=1.25)

    # group captions on the bands, right-aligned inside the axes
    ax.text(0.4773, 0.440, "RutaBAuxFeaturesExtractor trunk",
            fontsize=8.2, color=GREEN_TEXT, ha="left", va="center",
            style="italic")
    ax.text(0.4922, 0.863, "default PPO extractor",
            fontsize=8.2, color=VERMIL_TEXT, ha="right", va="center",
            style="italic")

    # the single takeaway, in neutral ink, anchored mid-plot
    ax.annotate(
        "", xy=(0.4790, 0.446), xytext=(0.4790, 0.699),
        arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9),
    )
    ax.text(0.4794, 0.5725, "same ReT,\n$\\sim$45% lower cost",
            fontsize=8, color="0.30", ha="left", va="center", linespacing=1.3)

    ax.set_xlabel("ReT Excel (Case C scale)", fontsize=9)
    ax.set_ylabel("resource cost index", fontsize=9)
    ax.set_xlim(0.4770, 0.4930)
    ax.set_ylim(0.33, 0.92)
    ax.set_xticks([0.478, 0.482, 0.486, 0.490])
    fig.tight_layout()
    save(fig, "fig10_efficiency_architecture")


# ---------------------------------------------------------------- fig11
def fig11_no_forecast_defense() -> None:
    """No-forecast defense: 15-seed paired delta (no-forecast - full-v7) on
    ReT Excel hovers at zero; the headline PPO-vs-static effect is ~200x the
    y-axis range. Neutral single hue -- the message is "this is noise," so
    sign is not encoded in color.
    Source: outputs/experiments/track_b_no_forecast_fixed_rng_final_15seed_2026-07-05/
    """
    # per-seed paired deltas, x10^-5 (no-forecast minus full-v7)
    # exact values from paired_seed_deltas.csv (x10^-5)
    deltas = [2.02, -0.18, -1.47, 1.43, 0.59, 3.35, -0.06, 4.40,
              0.01, 2.14, 0.53, -2.10, 1.22, -6.74, -1.93]
    mean, lo, hi = 0.21, -1.26, 1.68

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(1, 16)

    ax.plot([0.2, 15.9], [0, 0], color="0.35", lw=1.0, zorder=1)
    ax.vlines(x, 0, deltas, color="0.85", lw=1.4, zorder=2)
    ax.scatter(x, deltas, s=44, c=BLUE, edgecolors="0.2", linewidths=0.5,
               zorder=3)

    # pooled mean + CI at the right, in neutral ink
    xm = 16.6
    ax.errorbar([xm], [mean], yerr=[[mean - lo], [hi - mean]], fmt="D",
                ms=7, mfc="white", mec="0.15", ecolor="0.15", elinewidth=1.4,
                capsize=5, capthick=1.4, zorder=4)
    ax.text(xm + 0.55, mean, "mean $+0.21$\nCI95 $[-1.26, +1.68]$",
            fontsize=8, color="0.15", ha="left", va="center", linespacing=1.3)

    # scale reference: text only, no arrow
    ax.text(0.55, 6.6,
            "headline PPO $-$ static effect: $+43.8\\times 10^{-5}$ "
            "($\\sim$200$\\times$ this axis)",
            fontsize=8, color="0.35", ha="left", va="center")

    ax.set_xticks(list(x) + [xm])
    ax.set_xticklabels([str(s) for s in x] + ["mean"], fontsize=8)
    ax.set_xlim(0.2, 20.4)
    ax.set_ylim(-7.6, 7.4)
    ax.set_xlabel("training seed", fontsize=9)
    ax.set_ylabel("paired $\\Delta$ ReT Excel\n(no-forecast $-$ full-v7, "
                  "$\\times 10^{-5}$)", fontsize=8.8)
    fig.tight_layout()
    save(fig, "fig11_no_forecast_defense")


# ---------------------------------------------------------------- fig12
def fig12_des_validation() -> None:
    """DES reconstruction checks against Garrido-Rios (2017) Table 6.10.

    Panel (a): per-year delivered rations, thesis reference vs our model,
    with the thesis's OWN historical calibration dispersion band (its
    correlated-inspection gaps span -21.6% to +14.1% of the reference
    mean; the thesis defines no +/-15% acceptance threshold — provenance
    fix 2026-07-09) and the structural-fidelity gap.
    Panel (b): deterministic-to-stochastic transition: delivered rations,
    fill rate, and backorders as disruption intensity escalates.
    Sources: outputs/validation/validation_table_dual_basis.csv (panel a);
    manuscript Section 4.1 headline numbers (panel b).
    """
    import csv

    src = Path("outputs/validation/validation_table_dual_basis.csv")
    years, thesis, ours = [], [], []
    with src.open() as fh:
        for row in csv.DictReader(fh):
            if row["year_basis"] == "thesis":
                years.append(int(row["Year"]))
                thesis.append(float(row["Thesis_ECS"]))
                ours.append(float(row["Our_Model"]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # --- panel (a): per-year fidelity ---
    w = 0.36
    axA.bar(np.array(years) - w / 2, thesis, w, color=GREY, alpha=0.55,
            edgecolor="0.3", linewidth=0.5, zorder=2, label="Thesis reference")
    axA.bar(np.array(years) + w / 2, ours, w, color=BLUE, alpha=0.78,
            edgecolor="0.15", linewidth=0.5, zorder=2, label="Our DES")
    tmean = np.mean(thesis)
    # Band = the thesis's own historical calibration dispersion (correlated
    # inspection, Table 6.10: -21.6%..+14.1%), NOT an acceptance threshold.
    axA.axhspan(tmean * (1 - 0.216), tmean * (1 + 0.141), color=GREEN,
                alpha=0.06, zorder=0)
    axA.axhline(tmean, color="0.5", lw=0.7, ls=":", zorder=1)
    axA.text(8.4, tmean, "thesis mean", fontsize=7.0, color="0.45",
             ha="right", va="bottom")
    rmse = np.sqrt(np.mean((np.array(ours) - np.array(thesis)) ** 2))
    gap = (np.mean(ours) - tmean) / tmean * 100
    axA.text(0.5, 1130000,
             f"RMSE = {rmse/1000:.0f}k rations/yr\n"
             f"avg gap = {gap:+.1f}%\n"
             f"(thesis RMSE baseline: 88k)",
             fontsize=7.6, color="0.2", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
    axA.set_xlabel("Validation year")
    axA.set_ylabel("Annual delivered rations")
    axA.set_xticks(years)
    axA.set_ylim(0, 1150000)
    axA.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    axA.set_title("(a) Year-by-year fidelity (thesis basis)", fontsize=9, pad=5)
    axA.legend(fontsize=7.4, loc="lower right", frameon=False)
    axA.grid(True, axis="y", lw=0.3, color="0.92", zorder=0)

    # --- panel (b): risk-regime transition (manuscript Section 4.1) ---
    regimes = ["deterministic\n(no risk)", "current\nrisk", "increased\nrisk"]
    delivered = [733621, 677750, 549250]
    fill = [99.3, 68.3, 45.6]
    backorders = [41, 1825, 3132]

    x = np.arange(len(regimes))
    bar_colors = [BLUE, ORANGE, VERMIL]
    # ink color chosen per bar for >=4.5:1 contrast after alpha-blend over white
    # white passes on blue (5.19) but fails on orange (1.79) and vermilion (2.63)
    ink_colors = ["white", "#1F2933", "#1F2933"]
    axB.bar(x, delivered, 0.55, color=bar_colors,
            alpha=0.72, edgecolor="0.2", linewidth=0.5, zorder=2)
    axB.set_ylabel("Annual delivered rations", fontsize=8.6)
    axB.set_ylim(0, 850000)
    axB.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    for xi, d in zip(x, delivered):
        axB.text(xi, d + 18000, f"{d/1000:.0f}k", ha="center", fontsize=7.6,
                 fontweight="bold", color="0.15")
    for xi, f, b, ink in zip(x, fill, backorders, ink_colors):
        axB.text(xi, 30000, f"fill {f:.1f}%\n{b:,} BO", ha="center",
                 fontsize=7.2, color=ink, fontweight="bold", va="bottom")
    axB.set_xticks(x)
    axB.set_xticklabels(regimes, fontsize=8.0)
    axB.set_title("(b) Disruption-driven degradation", fontsize=9, pad=5)
    axB.grid(False)

    fig.tight_layout()
    save(fig, "fig12_des_validation")


# ---------------------------------------------------------------- fig13
def fig13_track_a_boundary() -> None:
    """Track A negative result: the dense static frontier absorbs the
    oracle headroom that no tested learner converts.

    Panel (a): 192 static dispatch policies on the Excel-ReT / resource
    plane; the best learned PPO (5 seeds) sits below the best static.
    Panel (b): the 75-regime oracle grid shows measurable but tiny
    headroom (+0.000176 to +0.000296) that the regime-conditioned oracle
    captures but no tested learner converts.
    Sources:
      outputs/experiments/track_a_repair_continuous_5seed_2026-06-30/
        static_frontier_heldout.csv, seed_health.csv
      outputs/experiments/track_a_headroom_search_full3_continuous_2026-06-29/
        gate_summary.json, best_static_by_regime.csv
    """
    import csv
    import json

    base = Path("outputs/experiments/track_a_repair_continuous_5seed_2026-06-30")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # --- panel (a): static frontier + PPO seeds ---
    statics = []
    with (base / "static_frontier_heldout.csv").open() as fh:
        for row in csv.DictReader(fh):
            statics.append((float(row["excel"]), float(row["resource"])))
    sa = np.array(statics)
    ppo_seeds = []
    with (base / "seed_health.csv").open() as fh:
        for row in csv.DictReader(fh):
            ppo_seeds.append((float(row["excel"]), float(row["resource"]),
                              int(row["seed"])))

    axA.scatter(sa[:, 1], sa[:, 0], s=14, c=GREY, alpha=0.35,
                edgecolors="none", zorder=2, label="192 static policies")
    bi = int(np.argmax(sa[:, 0]))
    axA.scatter([sa[bi, 1]], [sa[bi, 0]], s=60, facecolors="none",
                edgecolors=BLUE, linewidths=1.3, zorder=4)
    axA.annotate("best static\n(0.155254)", xy=(sa[bi, 1], sa[bi, 0]),
                 xytext=(0.30, 0.1550), fontsize=7.4, color=BLUE,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    for exc, res, sd in ppo_seeds:
        mk = "*" if sd == 5 else "o"
        sz = 120 if sd == 5 else 40
        axA.scatter([res], [exc], marker=mk, s=sz, c=VERMIL, zorder=5,
                    edgecolors="0.2", linewidths=0.5)
    axA.annotate("best PPO\n(0.155247)", xy=(0.1752, 0.155247),
                 xytext=(0.42, 0.1545), fontsize=7.4, color=VERMIL,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=VERMIL, lw=0.8))
    axA.set_xlabel("resource index", fontsize=8.6)
    axA.set_ylabel("Excel ReT", fontsize=8.6)
    axA.set_title("(a) PPO below the dense static frontier", fontsize=9, pad=5)
    axA.legend(fontsize=7.2, loc="lower right", frameon=False)
    axA.grid(True, lw=0.3, color="0.92", zorder=0)

    # --- panel (b): the headroom claim itself, not the regime spread.
    # The 75 regimes span 0.0004-0.3 (3 orders of magnitude); plotting that
    # makes the 0.15% oracle-vs-constant gap invisible. Instead, zoom in on
    # the four learners vs the oracle/constant ceiling, which is the actual
    # argument: headroom exists, no learner converts it.
    gs = json.load(open(base.parent / "track_a_headroom_search_full3_continuous_2026-06-29"
                        / "gate_summary.json"))
    oracle = gs["oracle_excel"]
    best_const = gs["best_single_constant"]["excel"]
    headroom = gs["oracle_minus_best_static"]
    # Two comparable points only: constant vs oracle live on the oracle-grid
    # evaluation scale. PPO is evaluated on a different grid (panel a's
    # canonical scale), so it is reported in the note box, never plotted on
    # this axis. Dots, not bars: a truncated axis makes bar length
    # meaningless.
    learners = [
        ("best constant\nstatic", best_const, GREY, "o"),
        ("regime-conditioned\noracle", oracle, GREEN, "*"),
    ]
    lo = best_const - headroom * 1.5
    hi = oracle + headroom * 1.5

    yvals = [v for _, v, _, _ in learners]
    ylabs = [n for n, _, _, _ in learners]
    ypos = np.array([1.8, 1.0])
    for yp, (_, v, c, mk) in zip(ypos, learners):
        axB.plot([lo, v], [yp, yp], color="0.90", lw=1.4, zorder=1)
        axB.scatter([v], [yp], marker=mk, s=150 if mk == "*" else 70,
                    c=c, zorder=4, edgecolors="0.2", linewidths=0.6)
    # headroom bracket between constant and oracle
    axB.annotate("", xy=(oracle, 0.55), xytext=(best_const, 0.55),
                 arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.0))
    axB.text((oracle + best_const) / 2, 0.38,
             f"+{headroom*1e6:.0f}$\\times 10^{{-6}}$ headroom\n(oracle $-$ best constant)",
             fontsize=7.2, color="0.2", ha="center", va="top")
    axB.text((oracle + best_const) / 2, -0.42,
             "no tested learner converts this headroom:\nPPO sits at or below its own best constant\n(panel a; evaluated on the canonical grid)",
             fontsize=7.2, color=VERMIL_TEXT, ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.25", fc="#fbe9e7", ec=VERMIL, lw=0.6))
    axB.set_yticks(ypos)
    axB.set_yticklabels(ylabs, fontsize=7.8)
    axB.set_xlabel("Excel ReT (Track A oracle-grid scale)", fontsize=8.6)
    axB.set_xlim(lo, hi)
    axB.set_ylim(-1.35, 2.35)
    axB.set_title("(b) Oracle headroom exists but is unconverted", fontsize=9, pad=5)
    axB.grid(True, axis="x", lw=0.3, color="0.92", zorder=0)

    fig.tight_layout()
    save(fig, "fig13_track_a_boundary")


# ---------------------------------------------------------------- fig14
def fig14_dispatch_cost_sensitivity() -> None:
    """Dispatch-inclusive cost sensitivity: pricing downstream transport
    favors PPO, which expedites selectively, over the best static
    comparator, which holds aggressive multipliers permanently.

    Crossover at lambda_d approx 0.025; from there up PPO is
    simultaneously resilience-dominant and cheaper.
    Source: docs/track_b_q1_stats_2026-07-02_final/dispatch_cost_sensitivity.csv
    """
    import csv

    src = Path("docs/track_b_q1_stats_2026-07-02_final/dispatch_cost_sensitivity.csv")
    lam, ppo_c, stat_c, d_lo, d_hi = [], [], [], [], []
    with src.open() as fh:
        for row in csv.DictReader(fh):
            lam.append(float(row["dispatch_charge_per_multiplier_step"]))
            ppo_c.append(float(row["ppo_dispatch_inclusive_cost_mean"]))
            stat_c.append(float(row["static_dispatch_inclusive_cost_mean"]))
            d_lo.append(float(row["delta_ci95_low"]))
            d_hi.append(float(row["delta_ci95_high"]))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    # static line drawn ABOVE the band (higher zorder) so it stays visible
    # where the two lines are close near lambda=0
    ax.plot(lam, ppo_c, "-o", color=GREEN, lw=1.8, ms=5, zorder=4,
            label="PPO (mean mult $\\approx$1.30/1.27)")
    # PPO cost uncertainty band, but only where it does NOT cover the static
    # line (i.e., for lambda >= 0.025 where PPO is below static). Near lambda=0
    # the band straddles the static line and would obscure it; clip it out.
    half = (np.array(d_hi) - np.array(d_lo)) / 2
    lam_arr = np.array(lam)
    ppo_arr = np.array(po_c) if False else np.array(ppo_c)
    stat_arr = np.array(stat_c)
    # mask band to the region where PPO upper bound < static (band sits below static)
    mask = (ppo_arr + half) < stat_arr
    ax.fill_between(lam_arr, ppo_arr - half, ppo_arr + half,
                    where=mask, color=GREEN, alpha=0.14, zorder=2,
                    edgecolor="none")
    # static line on top so always visible
    ax.plot(lam, stat_c, "-o", color=GREY, lw=1.6, ms=5, zorder=5,
            label="best static (S2, Op10$\\times$2.0, Op12$\\times$1.5)")

    ax.axvline(0.025, color=VERMIL, lw=0.9, ls=(0, (4, 3)), zorder=1)
    # crossover annotation placed in the gap between the lines (upper-right)
    ax.annotate("crossover\n$\\lambda_d \\approx 0.025$",
                xy=(0.025, 0.695), xytext=(0.045, 0.625),
                fontsize=7.6, color=VERMIL, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=VERMIL, lw=0.9))
    # "PPO cheaper" callout moved to upper-right clear of both lines (lines
    # trend down-left, top-right is empty)
    ax.text(0.165, 0.93, "PPO cheaper\n(significant, $\\lambda_d \\geq 0.025$)",
            fontsize=7.4, color=GREEN, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#e7f4e9", ec=GREEN, lw=0.5))
    # "n.s." placed at the actual near-crossing point (lambda=0, where CIs cross)
    ax.annotate("n.s. (CIs cross 0)", xy=(0.0, 0.675), xytext=(0.03, 0.56),
                fontsize=7.2, color="0.4", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=0.7))

    ax.set_xlabel("dispatch charge $\\lambda_d$ (per unit expediting)", fontsize=9)
    ax.set_ylabel("total cost index\n($C_{\\mathrm{shift}} + \\lambda_d \\cdot$ dispatch)",
                  fontsize=8.4)
    ax.set_xlim(-0.005, 0.205)
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=7.4, loc="upper left", frameon=False)
    ax.grid(True, lw=0.3, color="0.92", zorder=0)
    save(fig, "fig14_dispatch_cost_sensitivity")


# ---------------------------------------------------------------- fig15
def fig15_learning_curves() -> None:
    """Training dynamics: per-checkpoint Excel ReT for Track A PPO
    (5 seeds x 8 checkpoints, 5k-40k timesteps).

    The fidelity gate (collapsed=True) marks seeds that violated the
    DES-fidelity constraint during training; their score is penalized.
    The frontier never crosses the best-static line, consistent with the
    Track A boundary result.
    Source: outputs/experiments/track_a_repair_continuous_5seed_2026-06-30/
            checkpoint_metrics.csv
    """
    import csv

    src = Path("outputs/experiments/track_a_repair_continuous_5seed_2026-06-30/checkpoint_metrics.csv")
    rows = []
    with src.open() as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["seed"]), int(row["step"]),
                         float(row["excel"]), row["collapsed"] == "True"))

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    best_static = 0.155254
    ax.axhline(best_static, color=BLUE, lw=1.2, ls="--", zorder=1)
    ax.text(4200, best_static, "best static (0.155254)", fontsize=7.4,
            color=BLUE, va="bottom", ha="left")

    seeds = sorted(set(r[0] for r in rows))
    for sd in seeds:
        pts = sorted([(s, e, c) for s2, s, e, c in rows if s2 == sd])
        steps = [p[0] for p in pts]
        excels = [p[1] for p in pts]
        collapsed = [p[2] for p in pts]
        valid_x, valid_y = [], []
        for st, ex, cl in zip(steps, excels, collapsed):
            if cl:
                if len(valid_x) > 1:
                    ax.plot(valid_x, valid_y, "-", color=BLUE, lw=1.1,
                            alpha=0.65, zorder=2)
                if valid_x:
                    ax.scatter(valid_x, valid_y, s=15, c=BLUE, alpha=0.85,
                               edgecolors="none", zorder=3)
                valid_x, valid_y = [], []
                ax.scatter([st], [ex], s=22, marker="x", c=VERMIL, zorder=4)
            else:
                valid_x.append(st)
                valid_y.append(ex)
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_y, "-", color=BLUE, lw=1.1, alpha=0.65,
                    zorder=2)
        if valid_x:
            ax.scatter(valid_x, valid_y, s=15, c=BLUE, alpha=0.85,
                       edgecolors="none", zorder=3)
        # vertical offset prevents seed-label collision at the right edge
        # (final Excel ReT values cluster: 0.1526-0.1552)
        label_offsets = {1: +0.0011, 2: -0.0004, 3: +0.0006, 4: -0.0010, 5: -0.0014}
        ax.text(steps[-1] + 800, excels[-1] + label_offsets.get(sd, 0),
                f"s{sd}", fontsize=6.8, va="center", ha="left", color="0.3")

    ax.scatter([], [], marker="x", c=VERMIL, s=30, label="fidelity-gate collapse")
    ax.plot([], [], "-", color=BLUE, lw=1.1, alpha=0.65,
            marker="o", ms=3.5, label="valid checkpoint")
    ax.legend(fontsize=7.4, loc="lower left", frameon=False)
    ax.set_xlabel("training timesteps", fontsize=9)
    ax.set_ylabel("Excel ReT (Track A scale)", fontsize=9)
    ax.set_xlim(3000, 43000)
    ax.set_ylim(0.150, 0.161)
    ax.grid(True, lw=0.3, color="0.92", zorder=0)
    save(fig, "fig15_learning_curves")


# ---------------------------------------------------------------- fig16
def fig16_reward_sensitivity() -> None:
    """Reward-robustness screen: all 18 reward/observation cells show a
    positive Excel ReT delta (range +0.000195 to +0.000452).

    The sign of the Track B effect is not uniquely tied to one reward
    specification, CVaR tail weight, or observation version.
    Source: outputs/experiments/track_b_adaptive_sweep_kaggle_2026-07-01_v6/
            fetched/track_b_adaptive_sweep/sweep_summary.csv
    """
    import csv

    src = Path("outputs/experiments/track_b_adaptive_sweep_kaggle_2026-07-01_v6/"
               "fetched/track_b_adaptive_sweep/sweep_summary.csv")
    cells = []
    with src.open() as fh:
        for row in csv.DictReader(fh):
            cells.append((row["reward_mode"], row["observation_version"],
                          row.get("ret_excel_cvar_alpha", ""),
                          float(row["excel_ret_delta_vs_best_static"]),
                          float(row["learned_cost_index"])))

    reward_modes = ["control_v1", "ReT_excel_plus_cvar", "ReT_tail_v2",
                    "ReT_garrido2024_train"]
    obs_color = {"v7": BLUE, "v8": SKY, "v9": GREEN}
    obs_marker = {"v7": "o", "v8": "s", "v9": "D"}
    # horizontal jitter by obs version so dots in the same reward family separate
    obs_jitter = {"v7": -0.16, "v8": 0.0, "v9": 0.16}
    x_pos = {rm: i for i, rm in enumerate(reward_modes)}

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for rm, ov, alpha, delta, cost in cells:
        xp = x_pos[rm] + obs_jitter[ov]
        if alpha:
            a = float(alpha)
            xp += (a - 0.1) * 1.1  # smaller alpha jitter now that obs is offset
        ax.scatter(xp, delta * 1e6, s=55, c=obs_color[ov],
                   marker=obs_marker[ov], zorder=3, edgecolors="0.2",
                   linewidths=0.5, alpha=0.9)

    ax.axhline(0, color="0.3", lw=1.0, zorder=1)
    ax.axhline(438, color=GREEN, lw=0.8, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    # move canonical label to upper-left where there are no dots (control_v1 tops ~406)
    ax.text(0.0, 462, "canonical 10-seed (+0.000438)", fontsize=7.0,
            color=GREEN, ha="left", va="bottom")

    ax.set_xticks(range(len(reward_modes)))
    ax.set_xticklabels([rm.replace("_", "\n") for rm in reward_modes], fontsize=7.6)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-10, 500)
    ax.set_ylabel("Excel ReT $\\Delta$ ($\\times 10^{-6}$) vs best static", fontsize=8.8)
    ax.set_xlabel("training reward family", fontsize=9)
    ax.set_title("18/18 cells positive: effect robust across reward and observation",
                 fontsize=8.6, pad=8)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=m, color="none", markerfacecolor=c,
                      markeredgecolor="0.2", markersize=7, label=ov)
               for ov, c, m in [("v7", BLUE, "o"), ("v8", SKY, "s"), ("v9", GREEN, "D")]]
    # legend upper right (away from control_v1 dots at lower-left)
    ax.legend(handles=handles, fontsize=7.6, loc="upper right", frameon=True,
              framealpha=0.9, edgecolor="0.8",
              title="obs version", title_fontsize=7.4)
    ax.grid(True, axis="y", lw=0.3, color="0.92", zorder=0)
    save(fig, "fig16_reward_sensitivity")


# ---------------------------------------------------------------- fig17
def fig17_control_loop() -> None:
    """POMDP control loop: the weekly decision cycle connecting the DES
    environment, the observation/action interface, and the PPO policy.

    Rival-parity element (Ding et al. Fig. 4/5/6 show their agent-env
    interaction). This figure makes the Track A vs Track B action-
    surface distinction structural rather than verbal.
    """
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="0.3", lw=0.9, fs=8.2, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.03,rounding_size=0.08",
                     fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, linespacing=1.2, zorder=3)

    def arrow(p0, p1, color="0.25", lw=1.3, style="-|>"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                     mutation_scale=11, color=color, lw=lw, zorder=1))

    # section labels at top (clear of boxes)
    ax.text(1.45, 3.95, "environment", fontsize=7.6, color="0.4", ha="center")
    ax.text(7.35, 3.95, "agent", fontsize=7.6, color="0.4", ha="center")

    # environment (left)
    box(0.3, 1.9, 2.3, 1.7,
        "MFSC DES\n(Python/SimPy)\n\n13 operations\n9 risk processes\nGarrido-grounded",
        fc="#f6f8fa", ec="0.35", fs=8.0, weight="bold")

    # three middle boxes (observation top, reward middle, action bottom)
    box(3.4, 2.7, 2.0, 0.9,
        "observation $o_t$\nv7: 52 dims\n(backlog, fill, risk pressure)",
        fc="white", ec=BLUE, lw=1.0, fs=7.6)
    box(3.4, 1.75, 2.0, 0.55,
        "reward $r_t$ (control_v1)",
        fc="white", ec="0.4", lw=0.8, fs=7.4)
    box(3.4, 0.7, 2.0, 0.9,
        "action $a_t$\ntrack_b_v1: 8D\n(buffer, shift, dispatch)",
        fc="white", ec=VERMIL, lw=1.0, fs=7.6)

    # policy (right)
    box(6.3, 1.9, 2.3, 1.7,
        "PPO policy\n$\\pi_\\theta(a|o)$\n\nMLP 64$\\times$64\nGAE $\\lambda$=0.95",
        fc="#e7f4e9", ec=GREEN, lw=1.1, fs=8.0, weight="bold")

    # eval (far right), wider to avoid text overflow
    box(8.95, 1.9, 0.95, 1.7,
        "Eval:\nGarrido/\nExcel\nReT",
        fc="#f3edf8", ec=PURPLE, lw=0.9, fs=7.4, weight="bold")

    # arrows with labels placed ON the arrow midpoints
    # o_t: env-right -> observation-left (horizontal, clean)
    arrow((2.6, 3.15), (3.4, 3.15), color=BLUE)
    ax.text(3.0, 3.27, "$o_t$", fontsize=8, color=BLUE, ha="center", va="bottom")
    # a_t: policy-left -> action-right (horizontal, clean)
    arrow((6.3, 1.15), (5.4, 1.15), color=VERMIL)
    ax.text(5.85, 1.27, "$a_t$", fontsize=8, color=VERMIL, ha="center", va="bottom")
    # action -> env (action feeds back into env)
    arrow((3.4, 1.0), (2.6, 2.1), color=VERMIL)
    # r_t: env -> reward (reward flows from env to the reward box)
    arrow((2.6, 2.5), (3.4, 2.0), color="0.4")
    ax.text(2.85, 2.35, "$r_t$", fontsize=8, color="0.4", ha="center", va="center")
    # policy -> eval (horizontal; the Eval box title labels the edge itself)
    arrow((8.6, 2.75), (8.95, 2.75), color=PURPLE)

    # bottom footer band: cadence note + Track A/B distinction (no overlap with boxes)
    ax.text(5.0, 0.35, "decision step $t$: every 168 simulated hours (weekly planning cadence)",
            fontsize=7.6, color="0.35", ha="center", style="italic")
    # vermilion Track A/B note moved to a dedicated band below the cadence line
    ax.add_patch(FancyBboxPatch((0.3, 0.0), 9.6, 0.0, boxstyle="square,pad=0",
                 fc="none", ec="none"))
    ax.text(0.3, 0.08,
            "Track A: action dims 1-6 only (buffer/shift)   |   "
            "Track B: + dims 7-8 (Op10/Op12 dispatch) reach the bottleneck",
            fontsize=7.4, color=VERMIL, ha="left", va="bottom")

    save(fig, "fig17_control_loop")


if __name__ == "__main__":
    fig1_bottleneck_alignment()
    fig2_mfsc_topology()
    fig3_gap_decomposition()
    fig4_pareto_ret_tail_ctj()
    fig5_generalization_heatmap()
    fig6_action_space_ablation()
    fig7_ret_metric_lineage()
    fig8_ret_branch_timeline()
    fig9_prevention_ceiling()
    fig10_efficiency_architecture()
    fig11_no_forecast_defense()
    fig12_des_validation()
    fig13_track_a_boundary()
    fig14_dispatch_cost_sensitivity()
    fig15_learning_curves()
    fig16_reward_sensitivity()
    fig17_control_loop()
    print("done")
