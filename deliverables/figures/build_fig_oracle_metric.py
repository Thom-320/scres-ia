#!/usr/bin/env python3
"""Figure M5 — clairvoyant headroom and an unmatched-rights training pilot.

Panel (a): where each controller sits between the best static policy and the exact
clairvoyant ceiling, on the same 48 already-run campaigns.
Panel (b): capture ratio against training time for a methodological pilot whose neural
arms do not have the cross-campaign retained prior available to the retained MPC.

Style and palette inherited from build_figures.py (dataviz six-checks validated):
blue = learned, orange = structured feedback, green = static/frontier, ink/muted text.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/oracle_capture_v1"
INK, MUTED, BORDER = "#1A1A1A", "#5A6570", "#31415a"
SURF, GRID = "#FFFFFF", "#DCE3EA"
BLUE, ORANGE, GREEN, PURPLE = "#2B6CB0", "#D9642A", "#2F9E77", "#9C5BB8"
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 8.0, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})
DC = 7.28  # 185 mm

metric = json.loads((RES / "oracle_capture_metric.json").read_text())
pol = metric["policies"]
ceiling = metric["oracle"]["ceiling_mean"]
static_mean = metric["bars"]["best_static_open_loop"]["mean_label"]


def label_of(name):
    return pol[name]["absolute"]["mean_label"]


# ---------------------------------------------------------------- panel (a)
rows = [
    ("Clairvoyant ceiling\n(exact, 4^8 enumeration)", ceiling, GREEN, "oracle"),
    ("Retained belief-MPC\n(structured feedback)", label_of("frozen_c256_mpc_retained"),
     ORANGE, "policy"),
    ("Belief-reset MPC\n(feedback, no retention)", label_of("frozen_c256_mpc_reset"),
     ORANGE, "policy"),
    ("Best static calendar\n(distribution-aware)", static_mean, GREEN, "bar"),
    ("Constant allocation 1:2\n(discretionary anchor)", label_of("constant_action_1"),
     MUTED, "anchor"),
    ("Constant allocation 2:1\n(discretionary anchor)", label_of("constant_action_2"),
     MUTED, "anchor"),
]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(DC, 3.05),
                               gridspec_kw={"width_ratios": [1.06, 1.0]})

y = np.arange(len(rows))[::-1]
for (name, value, color, kind), yy in zip(rows, y):
    hatch = "///" if kind == "oracle" else None
    axa.barh(yy, value, height=0.56, color=color, alpha=0.30 if kind == "oracle" else 0.92,
             edgecolor=color, linewidth=0.9, hatch=hatch, zorder=3)
    axa.text(value + 0.004, yy, f"{value:.3f}", va="center", ha="left",
             fontsize=6.8, color=INK, zorder=4)
axa.axvline(static_mean, color=GREEN, lw=0.9, ls=(0, (4, 2)), zorder=2)
axa.axvline(ceiling, color=GREEN, lw=0.9, ls=(0, (1, 1.6)), zorder=2)
axa.set_yticks(y)
axa.set_yticklabels([r[0] for r in rows], fontsize=6.6, color=INK)
axa.set_xlim(0.575, ceiling + 0.048)
axa.set_xlabel("Mean terminal resilience ReT (48 campaigns)", fontsize=7.2, color=INK)
axa.set_title("(a) Distance to the exact clairvoyant ceiling", fontsize=7.8,
              weight="bold", color=INK, loc="left", pad=5)
axa.xaxis.grid(True, color=GRID, lw=0.4, zorder=0)
axa.set_axisbelow(True)
for side in ("top", "right", "left"):
    axa.spines[side].set_visible(False)
axa.tick_params(axis="both", labelsize=6.6, length=2, colors=MUTED)
axa.annotate("", xy=(static_mean, 0.62), xytext=(ceiling, 0.62),
             arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", lw=0.8, color=GREEN))
axa.text((static_mean + ceiling) / 2, 0.30, "learnable headroom", ha="center",
         va="center", fontsize=6.2, color=GREEN)

# ---------------------------------------------------------------- panel (b)
curves = {"ppo_mlp": ("PPO + MLP", BLUE), "recurrent_ppo": ("RecurrentPPO + LSTM", PURPLE)}
plotted = 0
for arch, (nice, color) in curves.items():
    path = RES / f"learning_curve_{arch}.json"
    if not path.exists():
        continue
    payload = json.loads(path.read_text())
    steps = [p["timesteps"] for p in payload["curves"][0]["points"]]
    mat = np.array([[p["pooled_ratio"] for p in c["points"]] for c in payload["curves"]])
    mean = mat.mean(axis=0)
    axb.fill_between(steps, mat.min(axis=0), mat.max(axis=0), color=color, alpha=0.16,
                     lw=0, zorder=2)
    axb.plot(steps, mean, color=color, lw=1.6, zorder=3,
             label=f"{nice} ({mat.shape[0]} seeds)")
    plotted += 1

mpc = pol["frozen_c256_mpc_retained"]["best_static_open_loop"]["pooled"]["pooled_ratio"]
axb.axhline(1.0, color=GREEN, lw=0.9, ls=(0, (1, 1.6)), zorder=2)
axb.axhline(0.0, color=GREEN, lw=1.0, ls=(0, (4, 2)), zorder=2)
axb.axhline(mpc, color=ORANGE, lw=1.2, zorder=2)
axb.text(0.985, 1.0, "clairvoyant ceiling", transform=axb.get_yaxis_transform(),
         ha="right", va="bottom", fontsize=6.2, color=GREEN)
axb.text(0.985, mpc, "retained belief-MPC (+0.74)", transform=axb.get_yaxis_transform(),
         ha="right", va="bottom", fontsize=6.2, color=ORANGE)
axb.text(0.985, 0.0, "best static policy", transform=axb.get_yaxis_transform(),
         ha="right", va="bottom", fontsize=6.2, color=GREEN)
lo = min(-1.6, min(mat.min() for mat in [np.array([[p["pooled_ratio"] for p in c["points"]]
     for c in json.loads((RES / f"learning_curve_{a}.json").read_text())["curves"]])
     for a in curves if (RES / f"learning_curve_{a}.json").exists()] or [np.array([0.0])]))
axb.set_ylim(lo - 0.18, 1.15)
axb.set_xlabel("Training experience (environment timesteps)", fontsize=7.2, color=INK)
axb.set_ylabel("Fraction of clairvoyant headroom captured", fontsize=7.2, color=INK)
axb.set_title("(b) Training-progress pilot (unmatched retention rights)",
              fontsize=7.8, weight="bold",
              color=INK, loc="left", pad=5)
axb.grid(True, color=GRID, lw=0.4, zorder=0)
axb.set_axisbelow(True)
for side in ("top", "right"):
    axb.spines[side].set_visible(False)
axb.tick_params(axis="both", labelsize=6.6, length=2, colors=MUTED)
if plotted:
    axb.legend(loc="lower right", fontsize=6.4, frameon=True, framealpha=0.96,
               edgecolor=GRID, borderpad=0.4).set_zorder(6)

fig.tight_layout(w_pad=1.6)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig7_oracle_metric.{ext}", dpi=300, bbox_inches="tight",
                facecolor=SURF)
plt.close(fig)
print(f"ok fig7_oracle_metric (curves plotted: {plotted})")
print(f"   ceiling {ceiling:.4f} | static {static_mean:.4f} | retained MPC capture {mpc:.4f}")
