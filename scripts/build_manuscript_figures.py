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
from matplotlib.colors import TwoSlopeNorm
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
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def chain(y, title, boxes, accent):
        ax.text(0.15, y + 1.06, title, fontsize=9.5, fontweight="bold", va="bottom")
        w, h, gap = 2.05, 0.85, 0.38
        x = 0.15
        for i, (label, fc) in enumerate(boxes):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    fc=fc,
                    ec="0.25",
                    lw=0.9,
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
                        mutation_scale=11,
                        color=accent,
                        lw=1.4,
                    )
                )
            x += w + gap

    chain(
        4.0,
        "Track A (boundary case): action surface misses the binding bottleneck",
        [
            ("Buffer + shift\ncontrols only", "#eaf1fb"),
            ("Upstream /\nassembly authority", "#eef7f0"),
            ("Downstream dispatch\nfixed (Op10/Op12)", "#fdf1df"),
            ("Dense static frontier\nabsorbs the headroom", "#fbe9e7"),
        ],
        GREY,
    )
    chain(
        1.1,
        "Track B (positive case): bottleneck authority converts into recovery",
        [
            ("Buffer + shift\n+ dispatch controls", "#eaf1fb"),
            ("Closed-loop\nfeedback policy", "#eef7f0"),
            ("Op10/Op12 dispatch\ncontrollable", "#e3f2f5"),
            ("Adaptive recovery:\nReT + tail gains", "#e7f4e9"),
        ],
        GREEN,
    )
    save(fig, "fig1_bottleneck_alignment")


# ---------------------------------------------------------------- fig2
def fig2_mfsc_topology() -> None:
    """Publication redraw of the Garrido-Rios 13-operation MFSC topology."""
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.set_xlim(0, 13.9)
    ax.set_ylim(0.0, 7.6)
    ax.axis("off")

    def band(x, y, w, h, label):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                fc="#f6f8fa",
                ec="#d5dee6",
                lw=0.8,
                zorder=0,
            )
        )
        ax.text(x + 0.1, y + h + 0.06, label, fontsize=8.5, color="#6b8795", fontweight="bold", zorder=1)

    band(0.1, 5.95, 2.2, 1.35, "Procurement")
    band(2.7, 5.95, 7.55, 1.35, "Inbound logistics")
    band(2.7, 3.70, 7.55, 1.45, "Manufacturing and outbound")
    band(0.1, 1.45, 10.15, 1.45, "Forward distribution")
    band(7.85, 0.02, 2.4, 1.22, "Demand")

    W, H = 1.9, 0.84
    ops = {
        1: (0.22, 6.18, "Op1", "Military logistics\nagency", "normal", "R12"),
        2: (2.88, 6.18, "Op2", "Suppliers (12)", "normal", "R13"),
        3: (5.08, 6.18, "Op3", "Warehouse and\ndistribution centre", "buffer", "R21"),
        4: (7.36, 6.18, "Op4", "Line of\ncommunication", "normal", "R22"),
        5: (7.36, 4.00, "Op5", "Assembly\npre-assembly", "shift_buffer", "R11 R21 R3"),
        6: (5.08, 4.00, "Op6", "Assembly\nline", "shift", "R11 R21 R3"),
        7: (2.88, 4.00, "Op7", "Assembly\nquality/shipping", "shift", "R14 R21 R3"),
        8: (0.22, 4.00, "Op8", "Line of\ncommunication", "normal", "R22"),
        9: (0.22, 1.70, "Op9", "Supply\nbattalion", "buffer", "R21 R3"),
        10: (2.88, 1.70, "Op10", "Line of\ncommunication", "dispatch", "R22"),
        11: (5.08, 1.70, "Op11", "Combat service\nsupport units (2)", "normal", "R23"),
        12: (7.36, 1.70, "Op12", "Line of\ncommunication", "dispatch", "R22"),
        13: (8.05, 0.22, "Op13", "Theatre of\noperations", "normal", "R24"),
    }

    def node(op):
        x, y, head, body, kind, risks = ops[op]
        if kind == "dispatch":
            fc, ec, lw = "#fff3df", VERMIL, 1.4
        elif kind in ("shift", "shift_buffer", "buffer"):
            fc, ec, lw = "#eaf5ea", BLUE, 1.1
        else:
            fc, ec, lw = "#e7f2fb", BLUE, 1.1
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.04,rounding_size=0.06", fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + W / 2, y + H * 0.80, head, ha="center", va="center", fontsize=8.6, fontweight="bold", zorder=3)
        ax.text(x + W / 2, y + H * 0.42, body, ha="center", va="center", fontsize=7.2, linespacing=1.1, zorder=3)
        ax.text(x + W / 2, y + H * 0.10, risks, ha="center", va="center", fontsize=6.2, color="0.45", zorder=3)
        # Track A control markers above the node
        marks = []
        if kind in ("buffer", "shift_buffer"):
            marks.append(("buffer $I_{LS}$", BLUE))
        if kind in ("shift", "shift_buffer"):
            marks.append(("shifts $S$", "#2e7d32"))
        if kind == "dispatch":
            marks.append(("dispatch $\\times$ (Track B)", VERMIL))
        for k, (mtext, mcol) in enumerate(marks):
            ax.text(x + W / 2, y + H + 0.07 + 0.21 * k, mtext, ha="center", va="bottom",
                    fontsize=6.6, color=mcol, fontweight="bold", zorder=3)

    for op in range(1, 14):
        node(op)

    def center(op):
        x, y, *_ = ops[op]
        return x + W / 2, y + H / 2

    def right(op):
        x, y, *_ = ops[op]
        return x + W, y + H / 2

    def left(op):
        x, y, *_ = ops[op]
        return x, y + H / 2

    def top(op):
        x, y, *_ = ops[op]
        return x + W / 2, y + H

    def bottom(op):
        x, y, *_ = ops[op]
        return x + W / 2, y

    def arrow(p0, p1, color="#263238", lw=1.2):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, zorder=1))

    for a, b in [(1, 2), (2, 3), (3, 4)]:
        arrow(right(a), left(b))
    x4, y4, *_ = ops[4]; x5, y5, *_ = ops[5]
    arrow((x4 + W * 0.88, y4), (x5 + W * 0.88, y5 + H))
    for a, b in [(5, 6), (6, 7), (7, 8)]:
        arrow(left(a), right(b))
    x8, y8, *_ = ops[8]; x9, y9, *_ = ops[9]
    arrow((x8 + W * 0.88, y8), (x9 + W * 0.88, y9 + H))
    for a, b in [(9, 10), (10, 11), (11, 12)]:
        arrow(right(a), left(b))
    arrow(bottom(12), top(13))

    # Bottleneck annotation (the argumentative point of the figure)
    ax.annotate(
        "downstream dispatch $\\approx$ 2,400\u20132,600 rations/day $\\approx$ demand\n$\\Rightarrow$ binding recovery bottleneck",
        xy=(3.83, 1.66),
        xytext=(4.5, 0.42),
        fontsize=7.6,
        color=VERMIL,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=VERMIL, lw=1.0),
        zorder=4,
    )

    # Side explanatory panel.
    ax.add_patch(
        FancyBboxPatch(
            (10.6, 2.95),
            2.8,
            3.95,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            fc="white",
            ec="#9aacb6",
            lw=1.0,
        )
    )
    ax.text(12.0, 6.47, "Decision surface", fontsize=10.5, fontweight="bold", ha="center")
    ax.text(10.95, 5.95, "Track A", fontsize=9.2, color=BLUE, fontweight="bold")
    ax.text(10.95, 5.52, "buffers and shifts\n(upstream + AL)", fontsize=8.2, linespacing=1.15)
    ax.text(10.95, 4.66, "Track B", fontsize=9.2, color=VERMIL, fontweight="bold")
    ax.text(10.95, 4.22, "adds Op10/Op12\ndispatch valves", fontsize=8.2, linespacing=1.15)
    ax.text(
        10.95,
        3.42,
        "Closed-loop policy\nacts on downstream\nrecovery bottlenecks",
        fontsize=7.9,
        color="0.25",
        linespacing=1.12,
    )

    save(fig, "fig2_mfsc_topology")


# ---------------------------------------------------------------- fig3
def fig3_gap_decomposition() -> None:
    labels = [
        "Best common static\n(147-cell dense grid)",
        "Regime-conditioned\nlookup table (fitted)",
        "Best heuristic\n(6 evaluated)",
        "PPO (canonical,\nfrozen checkpoint)",
    ]
    vals = [0.005466, 0.005494, 0.005436, 0.005893]
    cols = [GREY, ORANGE, SKY, GREEN]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, 0.00542, vals, color="0.85", lw=1.4, zorder=1)
    ax.scatter(vals, y, s=64, c=cols, zorder=3, edgecolors="0.2", linewidths=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.0000135, yi, f"{v:.6f}", va="center", fontsize=8)
    ax.axvline(0.005466, color=GREY, lw=0.9, ls="--", zorder=0)
    ax.text(0.005466, len(labels) - 0.42, " common-static reference", fontsize=7.2, color=GREY, ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Excel ReT (canonical CRN protocol, 5 seeds $\\times$ 12 episodes)")
    ax.set_xlim(0.00542, 0.00601)
    ax.annotate(
        "regime table gains only +0.0000277\nwith direct true-regime access",
        xy=(0.005494, y[1]),
        xytext=(0.00562, y[1] + 0.05),
        fontsize=7.4,
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9),
        va="center",
    )
    save(fig, "fig3_gap_decomposition")


# ---------------------------------------------------------------- fig5
def fig5_generalization_heatmap() -> None:
    deltas = np.array(
        [
            [0.000483, 0.000209],
            [0.000742, 0.000565],
            [-0.000060, 0.000009],
        ]
    )
    rows = ["current", "increased", "severe"]
    cols = ["h52", "h104"]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    norm = TwoSlopeNorm(vmin=-0.0008, vcenter=0.0, vmax=0.0008)
    im = ax.imshow(deltas, cmap="PiYG", norm=norm, aspect="auto")
    for i in range(3):
        for j in range(2):
            v = deltas[i, j]
            ax.text(
                j,
                i,
                f"{v:+.6f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if v > 0 else "normal",
                color="0.1",
            )
    # boundary-case marker
    ax.add_patch(
        plt.Rectangle((-0.5 + 0.03, 1.5 + 0.03), 0.94, 0.94, fill=False, ec=VERMIL, lw=1.8, ls=(0, (4, 2)))
    )
    ax.text(-0.02, 2.36, "boundary case\n(service-floor regime)", fontsize=6.8, color=VERMIL, ha="center")
    ax.set_xticks(range(2))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(3))
    ax.set_yticklabels(rows)
    ax.set_xlabel("evaluation horizon (weeks)")
    ax.set_ylabel("Garrido-native risk level")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("PPO $-$ best comparator (order-level ReT)", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.4")
    save(fig, "fig5_generalization_heatmap")


# ---------------------------------------------------------------- fig6
def fig6_action_space_ablation() -> None:
    arms = ["Joint\n(full 8D)", "Downstream-only\n(dispatch)", "Shift-only\n(capacity)"]
    deltas = [0.000367, 0.000429, 0.000377]
    cols = [BLUE, GREEN, SKY]

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    x = np.arange(len(arms))
    ax.vlines(x, 0, deltas, color="0.85", lw=1.6, zorder=1)
    ax.scatter(x, deltas, s=80, c=cols, zorder=3, edgecolors="0.2", linewidths=0.6)
    for xi, v in zip(x, deltas):
        ax.text(xi, v + 0.0000135, f"+{v:.6f}", ha="center", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8.4)
    ax.set_ylabel("Order-level ReT $\\Delta$ vs best\nin-arm comparator (incl. heuristics)")
    ax.set_ylim(0, 0.00050)
    ax.set_title("Gain concentrates in downstream dispatch access,\nnot in action-space size", fontsize=9)
    save(fig, "fig6_action_space_ablation")


# ---------------------------------------------------------------- fig7
def fig7_ret_metric_lineage() -> None:
    """Evaluation-metric lineage redrawn from Garrido-Rios Figure 5.11."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.6)
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

    root = box(3.65, 5.75, 3.2, 0.9, "Supply-chain resilience\n(SCRES)", fc="#e7f2fb", fontsize=10.5, weight="bold")
    dyn = box(0.85, 4.22, 3.15, 0.9, "Dynamic resilience\nunder risks", fc="#fff3df", fontsize=9.2)
    sta = box(6.45, 4.22, 3.15, 0.9, "Static resilience\ninherent to the SC", fc="#eaf5ea", fontsize=9.2)
    link((4.55, 5.75), (2.42, 5.12))
    link((6.10, 5.75), (8.03, 5.12))

    drivers = [
        (0.50, "$AP_j$\nautonomy period", "#fff8df"),
        (2.85, "$RP_j$\nrecovery period", "#fff8df"),
        (5.20, "$DP_j - RP_j$\nnon-recovery tail", "#fff8df"),
        (7.55, "$FR_t$\nfill-rate branch", "#eaf5ea"),
    ]
    anchors = []
    for x, text, fc in drivers:
        a = box(x, 2.42, 1.95, 0.88, text, fc=fc, fontsize=8.8)
        anchors.append(a)
    for a in anchors[:3]:
        link((2.42, 4.22), (a[2], a[3]))
    link((8.03, 4.22), (anchors[3][2], anchors[3][3]))

    fbox = box(
        2.35,
        0.70,
        6.25,
        1.02,
        "Order-level Garrido/Excel ReT\nprimary manuscript outcome; training reward is separate",
        fc="#f3edf8",
        fontsize=9.2,
        ec=PURPLE,
        lw=1.1,
        weight="bold",
    )
    for a in anchors:
        link((a[0], 2.42), (5.48, 1.72), color=GREY)
    ax.text(8.85, 1.17, "Excel branch logic\nreproduced for\norder-level ReT", fontsize=7.6, color="0.4", va="center")
    save(fig, "fig7_ret_metric_lineage")


if __name__ == "__main__":
    fig1_bottleneck_alignment()
    fig2_mfsc_topology()
    fig3_gap_decomposition()
    fig5_generalization_heatmap()
    fig6_action_space_ablation()
    fig7_ret_metric_lineage()
    print("done")
