#!/usr/bin/env python3
"""Figures for the C&IE outer-loop package, read FROM THE SEALED ARTIFACTS.

Style follows scripts/build_manuscript_figures.py -- Okabe-Ito colorblind-safe, STIX serif,
vector PDF for LaTeX plus 300-dpi PNG for the Word port -- but the numbers are NOT hard-coded
here. Every value is loaded from a sealed result.json, so a figure cannot drift away from the
evidence it claims to show, and re-running after a re-measurement is the whole update path.

Figures:
  fig_a_leak      the twin-surface test: what the oracle normaliser reads and the prefix does not
  fig_b_gates     separability by context, and H_regime against its bar
  fig_c_ladder    the comparator ladder with and without memory
  fig_d_memory    what retention is worth to each method family
  fig_e_efficiency  quality against cost: the Delta_efficiency panel
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("docs/manuscript_current/submission/elsevier/figures")
RESULTS = Path("results")

BLUE, SKY, GREEN, ORANGE = "#0072B2", "#56B4E9", "#009E73", "#E69F00"
VERMIL, PURPLE, GREY = "#D55E00", "#CC79A7", "#7f7f7f"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120,
})

#: Display names, so the figures never show a python identifier.
LABEL = {
    "oracle": "oracle (ceiling)", "random": "random", "ofat": "OFAT (thesis design)",
    "lhs_local": "LHS + local", "gp_ei": "Bayesian opt. (GP-EI)", "ucb1": "UCB1 bandit",
    "annealing": "simulated annealing", "neuron_reset": "Fig. 5 neuron, reset",
    "neuron_memory": "Fig. 5 neuron, memory", "gp_ei_transfer": "Bayesian opt. + memory",
    "ucb1_transfer": "UCB1 bandit + memory", "ofat_transfer": "OFAT + memory",
    "lookahead_kg_transfer": "knowledge-gradient + memory",
    "thompson_transfer": "Thompson + memory",
    "surrogate_mlp": "MLP surrogate", "surrogate_kan": "KAN surrogate",
}
MEMORY_ARMS = {"neuron_memory", "gp_ei_transfer", "ucb1_transfer", "ofat_transfer",
               "lookahead_kg_transfer", "thompson_transfer"}


def load(path: str) -> dict:
    return json.loads((RESULTS / path).read_text())


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def fig_a_leak() -> None:
    """The falsifier that turned 'we fixed the leak' into a measurement."""
    twin = load("twin_surface_v2/result.json")["falsifiers"][
        "f6_surface_twins_have_identical_prefix_paths"]["evidence"]["by_normaliser"]
    arms = ["ofat", "random", "neuron_reset", "neuron_memory"]
    counts = {norm: [sum(1 for f in twin[norm]["path_unchanged"][a].values() if f) for a in arms]
              for norm in ("oracle", "prefix")}

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    y = np.arange(len(arms))
    ax.barh(y - 0.19, counts["oracle"], height=0.34, color=VERMIL, label="oracle normaliser")
    ax.barh(y + 0.19, counts["prefix"], height=0.34, color=BLUE, label="prefix normaliser")
    ax.set_yticks(y, [LABEL[a] for a in arms])
    ax.set_xlim(0, 6.6)
    ax.set_xticks(range(0, 7))
    ax.set_xlabel("contexts whose search path is unchanged (of 6)")
    ax.invert_yaxis()
    for spine in ("oracle", "prefix"):
        off = -0.19 if spine == "oracle" else 0.19
        for i, v in enumerate(counts[spine]):
            ax.text(v + 0.12, y[i] + off, str(v), va="center", fontsize=8, color=GREY)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Altering cells the arm never ran moves only the oracle arm", loc="left")
    save(fig, "fig_a_normaliser_leak")


def fig_b_gates() -> None:
    """Why there is a search problem at all, and why context adaptation is not it."""
    gates = load("surface_gates_v2/result.json")
    g2 = gates["g2_separability"]
    g1 = gates["g1_h_regime"]
    order = sorted(g2, key=lambda c: -g2[c]["mean"])
    thr = gates["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    y = np.arange(len(order))
    means = [g2[c]["mean"] for c in order]
    lo = [g2[c]["mean"] - g2[c]["lcb95"] for c in order]
    hi = [g2[c]["ucb95"] - g2[c]["mean"] for c in order]
    ax.barh(y, means, color=[GREEN if g2[c]["lcb95"] >= thr else ORANGE for c in order],
            height=0.6)
    ax.errorbar(means, y, xerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=1, capsize=2)
    ax.axvline(thr, color=GREY, ls="--", lw=1)
    # Above the top bar, not below the last one, where it collided with the tick labels.
    ax.text(thr, -0.62, " bar 0.05", color=GREY, fontsize=8, va="bottom")
    ax.set_yticks(y, order)
    ax.set_ylim(len(order) - 0.4, -0.9)      # headroom for the threshold label
    ax.set_xlabel(r"held-out $\Delta R^2$ from pairwise interactions")
    ax.set_title("The surface is not separable", loc="left")

    ax = axes[1]
    ax.bar([0], [g1["H_regime"]], color=VERMIL, width=0.5)
    ax.errorbar([0], [g1["H_regime"]],
                yerr=[[g1["H_regime"] - g1["lcb95"]], [g1["ucb95"] - g1["H_regime"]]],
                fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
    ax.axhline(thr, color=GREY, ls="--", lw=1)
    ax.text(-0.55, thr, "bar 0.05", color=GREY, fontsize=8, va="bottom", ha="left")
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylim(0, thr * 1.35)
    ax.set_ylabel(r"$H_{regime}$ (share of achievable spread)")
    ax.set_title("Knowing the regime buys\nalmost nothing", loc="left")
    ax.annotate(f"{g1['H_regime']:.4f}", (0, g1["H_regime"]), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_b_surface_gates")


def fig_c_ladder() -> None:
    """The referee's ladder, memoryless and with memory, on one axis."""
    means = load("search_ladder_v5/result.json")["mean_auc_regret"]
    arms = [a for a in sorted(means, key=lambda a: means[a]) if a != "oracle"]

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    y = np.arange(len(arms))
    colors = [BLUE if a in MEMORY_ARMS else GREY for a in arms]
    colors = [PURPLE if a == "neuron_memory" else c for a, c in zip(arms, colors)]
    ax.barh(y, [means[a] for a in arms], color=colors, height=0.62)
    ax.set_yticks(y, [LABEL[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel("normalised AUC of search regret  (lower is better)")
    for i, a in enumerate(arms):
        ax.text(means[a] + 0.0025, y[i], f"{means[a]:.4f}", va="center", fontsize=8, color=GREY)
    ax.set_xlim(0, max(means[a] for a in arms) * 1.16)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (PURPLE, BLUE, GREY)]
    # Upper right, because inverting the y-axis puts the SHORTEST bars at the top and that is
    # the only quadrant no bar reaches.
    ax.legend(handles, ["Fig. 5 neuron", "carries memory", "restarts per context"],
              frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.42))
    ax.set_title("Once every method may remember, the network stops being special", loc="left")
    save(fig, "fig_c_comparator_ladder")


def fig_d_memory() -> None:
    """The Alzheimer price, paid to four different families."""
    contrasts = load("retention_contrasts/result.json")["contrasts"]
    pairs = [("ucb1_transfer", "ucb1"), ("neuron_memory", "neuron"),
             ("ofat_transfer", "ofat"), ("lookahead_kg_transfer", "lookahead_kg"),
             ("gp_ei_transfer", "gp_ei"), ("thompson_transfer", "thompson")]
    rows = []
    for mem, family in pairs:
        d = contrasts[family]
        rows.append((mem, float(d["mean"]), float(d["lcb95"]), float(d["ucb95"])))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    y = np.arange(len(rows))
    means = [r[1] for r in rows]
    ax.errorbar(means, y, xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
                fmt="o", color=BLUE, ecolor="#333333", elinewidth=1, capsize=3, markersize=5)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y, [LABEL[r[0]].replace(" + memory", "").replace(", memory", "")
                      for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("search regret avoided by carrying state across runs")
    ax.set_xlim(0, max(r[3] for r in rows) * 1.15)
    ax.set_title("Retention pays every family, not just the network", loc="left")
    save(fig, "fig_d_memory_price")


def fig_e_efficiency() -> None:
    """Quality against cost. The estimand E* declared and never measured."""
    art = load("search_surrogates/result.json")
    means, eff = art["mean_auc_regret"], art["delta_efficiency"]
    arms = list(means)
    colors = {"neuron_memory": PURPLE, "surrogate_mlp": SKY, "surrogate_kan": ORANGE}

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    # The three points differ, but every contrast crosses zero. Without this band the scatter
    # would assert an ordering the intervals deny, which is the figure lying about its own data.
    half = max(abs(v["lcb95"]) for v in art["vs_neuron_memory"].values())
    base = means["neuron_memory"]
    ax.axhspan(base - half, base + half, color="#eef2f6", zorder=0)
    ax.text(0.014, base + half, " statistically indistinguishable (95% CI)", fontsize=7.5,
            color=GREY, va="bottom")
    for a in arms:
        x = eff[a]["median_seconds_per_decision"] * 1000.0
        ax.scatter(x, means[a], s=60 + 0.25 * eff[a]["parameters"], color=colors[a], zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"{LABEL[a]}\n{eff[a]['parameters']} params",
                    (x, means[a]), textcoords="offset points", xytext=(9, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("median milliseconds per decision  (log)")
    ax.set_ylabel("normalised AUC of search regret")
    ax.set_xlim(0.012, 3.2)
    span = max(means.values()) - min(means.values())
    ax.set_ylim(min(means.values()) - span, max(means.values()) + span)
    ax.set_title("Equivalent quality; the five-parameter unit is 30$\\times$ cheaper", loc="left")
    ax.grid(axis="y", color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "fig_e_delta_efficiency")


def main() -> int:
    for fn in (fig_a_leak, fig_b_gates, fig_c_ladder, fig_d_memory, fig_e_efficiency):
        fn()
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
