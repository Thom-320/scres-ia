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

# Okabe-Ito palette
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMIL = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7f7f7f"

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
        2.47,
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
        0.55,
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
    cols = [GREY, ORANGE, SKY, GREEN]
    # 95% seed-clustered CI of the paired PPO-vs-common-static delta
    # (docs/track_b_q1_stats_2026-07-02_final/seed_level_inference.csv),
    # translated onto the PPO point: delta CI [0.000389, 0.000463].
    ppo_ci = (5.466 + 0.389, 5.466 + 0.463)

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(labels))[::-1]
    ax.set_ylim(-0.5, 3.9)
    ax.hlines(y, 5.42, vals, color="0.85", lw=1.4, zorder=1)
    ax.scatter(vals, y, s=64, c=cols, zorder=3, edgecolors="0.2", linewidths=0.6)
    ax.hlines(y[3], ppo_ci[0], ppo_ci[1], color=GREEN, lw=2.2, zorder=2)
    for cap in ppo_ci:
        ax.vlines(cap, y[3] - 0.09, y[3] + 0.09, color=GREEN, lw=1.4, zorder=2)
    for yi, v in zip(y, vals):
        off = 0.028 if v != vals[3] else 0.048
        ax.text(v + off, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(5.466, color=GREY, lw=0.9, ls="--", zorder=0)
    ax.text(5.466, 3.62, "common-static reference", fontsize=7.6,
            color="0.35", ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Excel ReT ($\\times 10^{-3}$; canonical CRN protocol, 5 seeds $\\times$ 12 episodes)")
    ax.set_xlim(5.42, 6.01)
    ax.annotate(
        "regime table gains only +0.028\nwith direct true-regime access",
        xy=(5.503, y[1] - 0.26),
        xytext=(5.60, y[1] - 0.26),
        fontsize=7.6,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9),
        va="center", ha="left",
    )
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
    deltas = np.array(
        [
            [0.359, 0.244],
            [0.537, 0.623],
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
    im = ax.imshow(deltas, cmap=cmap, norm=norm, aspect="auto")
    for i in range(3):
        for j in range(2):
            v = deltas[i, j]
            ax.text(
                j,
                i,
                f"{v:+.3f}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="white" if abs(v) > 0.40 else "0.1",
            )
    # boundary-regime marker: the whole severe row
    ax.add_patch(
        plt.Rectangle((-0.5 + 0.03, 1.5 + 0.03), 1.94, 0.94, fill=False, ec=VERMIL, lw=1.8, ls=(0, (4, 2)))
    )
    ax.text(0.5, 2.34, "boundary regime (service floor)", fontsize=7.4, color=VERMIL, ha="center")
    ax.set_xticks(range(2))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(3))
    ax.set_yticklabels(rows)
    ax.set_xlabel("evaluation horizon (weeks)")
    ax.set_ylabel("Garrido-native risk level")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("PPO $-$ best in-cell static, order-level ReT ($\\times 10^{-3}$)", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.4")
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
    ax.text(2.52, 0.012, "best in-arm comparator", fontsize=7.4, color="0.35",
            ha="right", va="bottom")
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


if __name__ == "__main__":
    fig1_bottleneck_alignment()
    fig2_mfsc_topology()
    fig3_gap_decomposition()
    fig4_pareto_ret_tail_ctj()
    fig5_generalization_heatmap()
    fig6_action_space_ablation()
    fig7_ret_metric_lineage()
    fig8_ret_branch_timeline()
    print("done")
