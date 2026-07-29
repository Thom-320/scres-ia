#!/usr/bin/env python3
"""Publication figures for the C&IE manuscript (DES model + Program Q).

Style: yEd-like orthogonal diagrams; palette validated with the dataviz six-checks
(#2B6CB0 blue=learned, #D9642A orange=structured, #2F9E77 green=static/frontier,
#9C5BB8 spare; red reserved for risk/status). Ink #1A1A1A, muted #5A6570.
Outputs: PDF (vector) + PNG 300 dpi, double-column 7.48 in unless noted.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK, MUTED, BORDER = "#1A1A1A", "#5A6570", "#31415a"
FILL, FILL2, SURF = "#F2F5F8", "#E7EDF3", "#FFFFFF"
BLUE, ORANGE, GREEN, PURPLE = "#2B6CB0", "#D9642A", "#2F9E77", "#9C5BB8"
RISK = "#B3261E"
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 8.0, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})
DC = 7.28  # double-column = 185 mm (Elsevier)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=SURF)
    plt.close(fig)
    print("ok", name)


def box(ax, x, y, w, h, title, lines, fc=FILL, ec=BORDER, tfs=7.6, lfs=6.6,
        title_color=INK, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    n = len(lines)
    ty = y + h - 0.16 if n else y + h / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=tfs,
            weight="bold", color=title_color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.34 - i * 0.145, ln, ha="center",
                va="center", fontsize=lfs, color=MUTED)


def oarrow(ax, p1, p2, color=BORDER, lw=1.1, style="-|>", ms=9):
    """Orthogonal (elbow) arrow: horizontal then vertical (or straight)."""
    (x1, y1), (x2, y2) = p1, p2
    if abs(y1 - y2) < 1e-9 or abs(x1 - x2) < 1e-9:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, lw=lw,
                                     color=color, mutation_scale=ms,
                                     shrinkA=0, shrinkB=0))
    else:
        xm = x2
        ax.plot([x1, xm], [y1, y1], color=color, lw=lw, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch((xm, y1), p2, arrowstyle=style, lw=lw,
                                     color=color, mutation_scale=ms,
                                     shrinkA=0, shrinkB=0))


def risk_chip(ax, x, y, label):
    ax.add_patch(FancyBboxPatch((x, y), 0.42, 0.15, boxstyle="round,pad=0.008",
                                fc="#FDECEA", ec=RISK, lw=0.7))
    ax.text(x + 0.21, y + 0.075, label, ha="center", va="center",
            fontsize=6.0, color=RISK, weight="bold")


# ---------------------------------------------------------------- Figure 1
def fig1_mfsc_flow():
    fig, ax = plt.subplots(figsize=(DC, 4.9))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 7.6); ax.axis("off")
    W, H = 2.55, 0.92
    ops = {
        1: ("Op1 · MLA contracting", ["PT 672 h · Q 12 contracts", "ROP 4,032 h (biannual)"], ["R12"]),
        2: ("Op2 · Suppliers ship rm", ["PT 24 h · Q 190,000/rm", "ROP 672 h (monthly)"], ["R13"]),
        3: ("Op3 · WDC receive/store", ["PT 24 h · Q 15,500/rm", "ROP 168 h · I(t,1)=0"], ["R21"]),
        4: ("Op4 · LOC to assembly", ["PT 24 h · weekly kit"], ["R22"]),
        5: ("Op5 · Pre-assembly (AL)", ["λ 320.5 rations/h", "2,564 / 8-h shift"], ["R11", "R21"]),
        6: ("Op6 · Assembly (AL)", ["PT 1/λ · balanced line"], ["R11", "R21"]),
        7: ("Op7 · QC + packaging", ["boxes of 10 · Q 5,000", "ROP 48 h"], ["R14", "R21"]),
        8: ("Op8 · LOC to SB", ["PT 24 h · 5,000-unit lots"], ["R22"]),
        9: ("Op9 · SB receive/store", ["PT 24 h · Q 2,400–2,600", "ROP 24 h · I(t,1)=0"], ["R21"]),
        10: ("Op10 · LOC to CSSU", ["PT 24 h · daily order"], ["R22"]),
        11: ("Op11 · CSSU issue", ["PT 0 h · 2 CSSUs"], ["R23"]),
        12: ("Op12 · LOC to theatre", ["PT 24 h · daily order"], ["R22"]),
        13: ("Op13 · Theatre demand", ["U(2,400–2,600)/day × 6 d/wk", "contingent orders"], ["R24"]),
    }
    rows = [(6.25, [1, 2, 3, 4]), (4.35, [8, 7, 6, 5]), (2.45, [9, 10, 11, 12])]
    pos = {}
    for y, idxs in rows:
        for c, i in enumerate(idxs):
            pos[i] = (0.35 + c * 2.95, y)
    pos[13] = (0.35 + 3 * 2.95, 0.55)
    for i, (t, lines, risks) in ops.items():
        x, y = pos[i]
        fc = FILL2 if "LOC" in t else FILL
        box(ax, x, y, W, H, t, lines, fc=fc)
        for k, r in enumerate(risks):
            risk_chip(ax, x + W - 0.46 - k * 0.47, y - 0.075, r)
    chain = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for a, b in zip(chain[:-1], chain[1:]):
        xa, ya = pos[a]; xb, yb = pos[b]
        if abs(ya - yb) < 1e-9:
            p1 = (xa + W, ya + H / 2) if xb > xa else (xa, ya + H / 2)
            p2 = (xb, yb + H / 2) if xb > xa else (xb + W, yb + H / 2)
        else:
            p1 = (xa + W / 2, ya); p2 = (xb + W / 2, yb + H)
        oarrow(ax, p1, p2)
    # echelon brackets
    ax.plot([0.2, 0.2], [4.25, 7.3], color=GREEN, lw=2.2)
    ax.text(0.10, 5.8, "Assemble-to-stock (Op1–Op9)", rotation=90,
            va="center", ha="right", fontsize=7, color=GREEN, weight="bold")
    ax.plot([0.2, 0.2], [0.45, 3.5], color=BLUE, lw=2.2)
    ax.text(0.10, 2.0, "Assemble-to-order (Op9–Op13)", rotation=90,
            va="center", ha="right", fontsize=7, color=BLUE, weight="bold")
    ax.text(12.1, 0.18,
            "Red chips: Garrido-native risks affecting each operation "
            "(R1r operational · R2r LOC/disaster · R24 demand surge)",
            ha="right", fontsize=6.2, color=MUTED, style="italic")
    save(fig, "fig1_mfsc_flow")


# ---------------------------------------------------------------- Figure 2
def fig2_framework():
    fig, ax = plt.subplots(figsize=(DC, 4.35))
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 6.7); ax.axis("off")
    tfs = 7.0
    # left column: environment stack (titles sized to fit the 3.5-wide box)
    box(ax, 0.3, 4.75, 3.5, 1.35, "Garrido MFSC DES (Baseline 0)",
        ["13 operations · fixed policies", "static scenario design",
         "operational ReT construct"], fc=FILL, tfs=tfs)
    box(ax, 0.3, 2.65, 3.5, 1.5, "Program Q extension (P_C / P_H)",
        ["non-fungible · identical physics", "risks OFF",
         "latent 2-state regime (ρ, s)"], fc=FILL, tfs=tfs)
    box(ax, 0.3, 0.55, 3.5, 1.5, "Weekly decision interface",
        ["8 decisions × 168 h", "action in {0,1,2,3} = P_C count",
         "in next 3 batches · 24-h lock"], fc=FILL, tfs=tfs)
    oarrow(ax, (2.05, 4.75), (2.05, 4.15))
    oarrow(ax, (2.05, 2.65), (2.05, 2.05))
    # observation
    box(ax, 4.45, 2.6, 2.6, 1.6, "Observation (21-dim)",
        ["inventory · pipeline", "backlog · in-flight",
         "fixed HMM belief", "prev. action · phase"], fc=FILL2, tfs=tfs)
    oarrow(ax, (3.8, 3.4), (4.45, 3.4))
    # three controllers
    box(ax, 7.7, 4.7, 3.2, 1.35, "Exact open-loop frontier",
        ["all 4⁸ = 65,536 calendars", "no feedback"], fc=SURF, ec=GREEN,
        title_color=GREEN, lw=1.4, tfs=tfs)
    box(ax, 7.7, 2.72, 3.2, 1.35, "Structured feedback family",
        ["belief-driven planner", "same deployable history"], fc=SURF,
        ec=ORANGE, title_color=ORANGE, lw=1.4, tfs=tfs)
    box(ax, 7.7, 0.75, 3.2, 1.35, "RecurrentPPO (MLP-LSTM)",
        ["64-unit LSTM · 2×64 heads", "10 seeds × 200,192 steps"],
        fc=SURF, ec=BLUE, title_color=BLUE, lw=1.4, tfs=tfs)
    for yc in (5.37, 3.40, 1.42):
        oarrow(ax, (7.05, 3.4), (7.7, yc))
    # evaluation
    box(ax, 11.3, 2.5, 2.0, 1.85, "Common\nevaluation",
        ["terminal ReT", "+ service ledger", "+ resource audit"],
        fc=FILL, tfs=tfs)
    for yc in (5.37, 3.40, 1.42):
        oarrow(ax, (10.9, yc), (11.35, 3.4))
    ax.text(0.3, 6.45, "Environment", fontsize=8, weight="bold", color=MUTED)
    ax.text(7.7, 6.45, "Controllers under identical resources", fontsize=8,
            weight="bold", color=MUTED)
    ax.text(6.7, 0.12,
            "Every arm receives the same production rights (24 batch slots = "
            "120,000 units), freight entitlement, demand tape, and score time.",
            ha="center", fontsize=6.4, color=MUTED, style="italic")
    save(fig, "fig2_framework")


# ---------------------------------------------------------------- Figure 3
def fig3_ladder():
    fig, ax = plt.subplots(figsize=(DC, 3.05))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 4.7); ax.axis("off")
    steps = [
        ("L0", "Garrido reference", "physical anchor;\nnot a matched comparator", FILL, BORDER, INK),
        ("L1", "Exact static frontier", "65,536 calendars;\nfeedback-value test", SURF, GREEN, GREEN),
        ("L2", "Structured feedback", "non-neural adaptive;\nneural-premium test", SURF, ORANGE, ORANGE),
        ("L3", "RecurrentPPO", "executed learned\ncontroller", SURF, BLUE, BLUE),
        ("L4", "DMLPA / KAN sidecars", "matched rerun required\nbefore any claim", FILL2, MUTED, MUTED),
    ]
    W, H = 2.4, 1.15
    for i, (lv, t, sub, fc, ec, tc) in enumerate(steps):
        x = 0.3 + i * 2.55; y = 0.6 + i * 0.62
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + 0.16, y + H - 0.22, lv, fontsize=8.5, weight="bold", color=tc)
        ax.text(x + W / 2, y + H - 0.42, t, ha="center", fontsize=7.2,
                weight="bold", color=tc)
        ax.text(x + W / 2, y + 0.30, sub, ha="center", fontsize=6.2, color=MUTED)
        if i:
            oarrow(ax, (x - 0.40, y - 0.62 + H / 2 + 0.31), (x, y + H / 2),
                   color=MUTED, lw=1.0)
    ax.text(0.3, 4.35, "Comparator ladder — each level answers one question; "
            "conclusions may not skip levels.", fontsize=7.6, weight="bold",
            color=INK)
    save(fig, "fig3_ladder")


# ---------------------------------------------------------------- Figure 4
def fig4_results():
    cells = ["ρ=.75, s=.90", "ρ=.90, s=.75", "ρ=.90, s=.90"]
    rl_frontier = [0.07952, 0.07255, 0.11724]
    lcb = [0.06608, 0.06233, 0.10614]
    rl_struct = [-0.00159, -0.00072, -0.00041]
    fig, axes = plt.subplots(1, 2, figsize=(DC, 2.7))
    x = np.arange(3)
    a = axes[0]
    err = np.array(rl_frontier) - np.array(lcb)
    a.bar(x, rl_frontier, 0.5, color=GREEN, zorder=3)
    a.errorbar(x, rl_frontier, yerr=[err, err * 0], fmt="none", ecolor=INK,
               elinewidth=1.0, capsize=3, zorder=4)
    for i, v in enumerate(rl_frontier):
        a.text(i, v + 0.006, f"+{v:.3f}", ha="center", fontsize=7, color=INK)
        a.text(i, lcb[i] - 0.010, f"LCB {lcb[i]:.3f}", ha="center",
               fontsize=6.0, color=MUTED)
    a.set_ylim(0, 0.14)
    a.set_title("(a) Learned feedback vs exact static frontier",
                fontsize=8, color=INK)
    a.set_ylabel("Delta ReT (RL − best static)", fontsize=7.5)
    b = axes[1]
    b.axhspan(-0.01, 0.01, color="#EDF1F5", zorder=1)
    b.axhline(0, color=MUTED, lw=0.7, zorder=2)
    b.bar(x, rl_struct, 0.5, color=ORANGE, zorder=3)
    for i, v in enumerate(rl_struct):
        b.text(i, v - 0.0022, f"{v:+.4f}", ha="center", fontsize=7, color=INK)
    b.text(2.42, 0.0104, "practical-equivalence margin ±0.01",
           ha="right", fontsize=6.2, color=MUTED, style="italic")
    b.set_ylim(-0.015, 0.015)
    b.set_title("(b) Learned vs structured feedback", fontsize=8, color=INK)
    b.set_ylabel("Delta ReT (RL − structured)", fontsize=7.5)
    for a_ in axes:
        a_.set_xticks(x); a_.set_xticklabels(cells, fontsize=7.5)
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(axis="y", color="#E4E9EE", lw=0.6, zorder=0)
        a_.tick_params(labelsize=7)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig4_results")


# ---------------------------------------------------------------- Figure 5
def fig5_ret_tree():
    fig, ax = plt.subplots(figsize=(DC, 3.25))
    ax.set_xlim(0, 13.6); ax.set_ylim(-0.35, 4.9); ax.axis("off")
    box(ax, 0.35, 3.45, 2.7, 0.95, "Order j scored at its",
        ["request-time snapshot"], fc=FILL, tfs=7.4)
    box(ax, 4.0, 3.45, 2.2, 0.95, "Risk active on j?", [], fc=FILL2, tfs=7.6)
    oarrow(ax, (3.05, 3.925), (4.0, 3.925))
    box(ax, 7.35, 3.95, 1.9, 0.8, "AP_j > 0 ?", [], fc=FILL2, tfs=7.6)
    box(ax, 7.35, 2.6, 1.9, 0.8, "RP_j > 0 ?", [], fc=FILL2, tfs=7.6)
    oarrow(ax, (6.2, 4.1), (7.35, 4.35)); ax.text(6.72, 4.5, "yes", fontsize=6.6, color=MUTED)
    oarrow(ax, (8.3, 3.95), (8.3, 3.40), color=MUTED); ax.text(8.42, 3.66, "no", fontsize=6.6, color=MUTED)
    EW = 3.65
    eqs = [
        (9.7, 4.10, "Re_j = AP_j / LT", GREEN, "recovery already under way"),
        (9.7, 2.75, "Re_j = 0.5 / RP_j", ORANGE, "recovery pending"),
        (9.7, 1.40, "Re_j = 0", RISK, "no recovery"),
        (9.7, 0.20, "Re_j = 1 − (B_tj + U_tj) / j", BLUE, "no-risk branch (Program Q)"),
    ]
    for xq, yq, eq, c, note in eqs:
        ax.add_patch(FancyBboxPatch((xq, yq), EW, 0.74,
                                    boxstyle="round,pad=0.012", fc=SURF, ec=c, lw=1.3))
        ax.text(xq + 0.22, yq + 0.47, eq, ha="left", fontsize=8.2, color=INK, family="Helvetica")
        ax.text(xq + 0.22, yq + 0.17, note, ha="left", fontsize=6.4, color=MUTED, style="italic")
    oarrow(ax, (9.25, 4.35), (9.7, 4.47)); ax.text(9.4, 4.56, "yes", fontsize=6.6, color=MUTED)
    oarrow(ax, (9.25, 3.0), (9.7, 3.12)); ax.text(9.4, 3.22, "yes", fontsize=6.6, color=MUTED)
    oarrow(ax, (8.3, 2.6), (8.3, 1.77), color=MUTED)
    oarrow(ax, (8.3, 1.77), (9.7, 1.77)); ax.text(8.42, 2.15, "no", fontsize=6.6, color=MUTED)
    ax.plot([5.1, 5.1], [3.45, 0.57], color=MUTED, lw=1.0)
    oarrow(ax, (5.1, 0.57), (9.7, 0.57)); ax.text(5.25, 1.95, "no risk", fontsize=6.6, color=MUTED)
    note = ("B_tj backorders · U_tj unattended orders (both at request time) · "
            "AP autotomy period · LT promised lead time (48 h) · RP recovery period.\n"
            "Formula reproduced on 47,546 source-workbook rows with zero mismatches "
            "(maximum absolute error 0).")
    ax.text(0.35, -0.28, note, fontsize=6.3, color=MUTED, va="bottom")
    save(fig, "fig5_ret_tree")


# ---------------------------------------------------------------- Figure 6
def fig6_timeline():
    fig, ax = plt.subplots(figsize=(DC, 2.5))
    ax.set_xlim(-0.4, 12.6); ax.set_ylim(0, 3.6); ax.axis("off")
    y = 1.55
    ax.add_patch(Rectangle((0, y), 1.7, 0.55, fc=FILL2, ec=BORDER, lw=0.9))
    ax.text(0.85, y + 0.275, "warm-up", ha="center", fontsize=7, color=INK)
    ax.text(0.85, y - 0.28, "product-balanced:\n1 real P_C + 1 real P_H lot at Op9",
            ha="center", fontsize=6.2, color=MUTED)
    for w in range(8):
        x = 1.7 + w * 1.05
        ax.add_patch(Rectangle((x, y), 1.05, 0.55, fc=FILL if w % 2 else SURF,
                               ec=BORDER, lw=0.9))
        ax.text(x + 0.525, y + 0.275, f"wk {w+1}", ha="center", fontsize=6.8, color=INK)
        ax.plot([x], [y + 0.72], marker="v", color=BLUE, ms=5)
    ax.text(1.7 + 4.2, y + 0.98, "8 weekly decisions (168 h each) — action locks after 24 h",
            ha="center", fontsize=6.8, color=BLUE, weight="bold")
    ax.add_patch(Rectangle((10.1, y), 1.9, 0.55, fc="#F7F4EC", ec=BORDER, lw=0.9))
    ax.text(11.05, y + 0.275, "clearance 1,344 h", ha="center", fontsize=6.8, color=INK)
    ax.plot([12.0], [y + 0.275], marker="o", color=INK, ms=5)
    ax.text(12.08, y + 0.275, "common\nscore time", fontsize=6.2, va="center", color=INK)
    # inside week detail (widened so the legend clears the markers)
    ax.add_patch(Rectangle((1.7, 0.12), 6.9, 0.78, fc=SURF, ec=MUTED, lw=0.8, ls=(0, (3, 2))))
    ax.text(1.86, 0.75, "inside one week", fontsize=6.4, color=MUTED, weight="bold")
    for off, lab in ((24, "b1"), (72, "b2"), (120, "b3")):
        xx = 1.95 + 3.0 * off / 168
        ax.plot([xx], [0.50], marker="s", color=BLUE, ms=4.5)
        ax.text(xx, 0.27, f"{lab} · {off} h", ha="center", fontsize=6.0, color=BLUE)
    for off in (30, 54, 78, 102, 126, 150):
        xx = 1.95 + 3.0 * off / 168
        ax.plot([xx], [0.66], marker="D", color=ORANGE, ms=3.6)
    ax.plot([5.6], [0.66], marker="D", color=ORANGE, ms=3.6)
    ax.text(5.8, 0.66, "6 demand orders (30–150 h)", fontsize=6.2,
            color=ORANGE, ha="left", va="center")
    ax.plot([5.6], [0.44], marker="s", color=BLUE, ms=4.5)
    ax.text(5.8, 0.44, "3 batch completions (5,000 u)", fontsize=6.2,
            color=BLUE, ha="left", va="center")
    ax.plot([1.7, 1.7], [0.90, y], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.plot([8.6, 2.75], [0.90, y], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.text(6.1, 3.25, "Episode structure — identical for every policy; "
            "no production or demand rights after week 8.",
            ha="center", fontsize=7.4, weight="bold", color=INK)
    save(fig, "fig6_timeline")


if __name__ == "__main__":
    fig1_mfsc_flow(); fig2_framework(); fig3_ladder()
    fig4_results(); fig5_ret_tree(); fig6_timeline()
