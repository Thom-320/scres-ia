#!/usr/bin/env python3
"""Paper 1 -- the number layer and the three figures, from one extraction of the sealed artifacts.

WHY ONE SCRIPT. The figures and the prose must not be able to disagree. Here the numbers are
extracted ONCE from `result.json` files, sealed into `numbers.json` with a receipt, and the three
figures are drawn from that same dict. A figure cannot drift from the sentence beside it because
neither one holds a number of its own.

WHAT IT REFUSES TO DO. Nothing is hard-coded, and every source must appear in the citable table
(`docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_2.md`). The freeze's rule is that a number
without a row does not circulate; this script enforces it by reading the amendment receipt and
halting if a source it is about to plot is not admitted.

THE THREE FINDINGS, AND WHY THEY ARE ONE PAPER. Each is a metric failing at its frontier with what
was never observed:

    M1  ret_excel discards it   -- the abandoned order leaves the scored population
    M2  restricted TTR never closes it -- the recovery cluster does not end inside the horizon
    M3  the oracle normaliser uses it without the right -- it divides by a range including cells
        that were never run

Style follows scripts/build_manuscript_figures.py: Okabe-Ito, STIX serif, vector PDF for LaTeX and
300-dpi PNG for the Word port.

Contract: docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md
Read-only over artifacts. No seeds are opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import module_manifest  # noqa: E402

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
AMENDMENT = Path("results/claim_freeze_amendment_2/result.json")

BLUE, SKY, GREEN, ORANGE = "#0072B2", "#56B4E9", "#009E73", "#E69F00"
VERMIL, PURPLE, GREY = "#D55E00", "#CC79A7", "#7f7f7f"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120,
})

#: source -> the finding it carries. Every one must be admitted to the citable table.
SOURCES = {
    "M1": "results/metric_audit/abandonment_v1/result.json",
    "M2_halted": "results/manuscript/h1_h3_v1/result.json",
    "M2_repaired": "results/manuscript/h1_h3_originales_v3/result.json",
    "M3": "results/twin_surface_v2/result.json",
}

#: The column run_metric_abandonment_audit_v1.py:138 adjudicates on, copied from that line and not
#: guessed. The other three travel with it so no reader can suspect this one was chosen.
ADJUDICATED_RET = "ret_excel_risk_conditional"
RET_VARIANTS = ["ret_excel_risk_conditional", "ret_excel_visible_clipped_0_1",
                "ret_excel_full_ledger", "ret_thesis"]
RET_LABEL = {"ret_excel_risk_conditional": "ReT (risk-conditional)",
             "ret_excel_visible_clipped_0_1": "ReT (visible, clipped)",
             "ret_excel_full_ledger": "ReT (full ledger)",
             "ret_thesis": "ReT (thesis form)"}

ARM_LABEL = {"hybrid": "hybrid", "reset": "reset", "static": "static"}
METHOD_LABEL = {"neuron_memory": "neuron (memory)", "neuron_reset": "neuron (reset)",
                "ofat": "OFAT", "random": "random"}


def load_admitted() -> set[str]:
    """The paths the claim-freeze amendment admits. Halts the build if the receipt is missing."""
    if not AMENDMENT.exists():
        raise SystemExit(f"halt: {AMENDMENT} not found -- run build_claim_freeze_amendment_v1.py")
    return {r["path"] for r in json.loads(AMENDMENT.read_text())["rows"]}


def extract() -> dict:
    """One pass over the sealed artifacts. Everything the paper says lives in the dict it returns."""
    admitted = load_admitted()
    art = {}
    for key, path in SOURCES.items():
        if path not in admitted:
            raise SystemExit(f"halt: {path} is not in the citable table; it cannot be plotted")
        art[key] = json.loads(Path(path).read_text())

    m1 = art["M1"]
    m1_out = {"shares": m1["shares"], "regimes": m1["regimes"], "step_hours": m1.get("step_hours"),
              "adjudicated_ret_column": ADJUDICATED_RET, "ret_variants": RET_VARIANTS,
              "by_regime": {}}
    for regime in m1["regimes"]:
        rep = m1["report"][regime]
        bs = rep["by_share"]
        share_of = lambda k: [bs[k][str(s)] for s in m1["shares"]]  # noqa: E731
        row = {
            "fill": share_of("flow_fill_rate"),
            "cobb_douglas": share_of("R_cobb_douglas"),
            "lost_orders": share_of("lost_orders"),
            "omitted_n": share_of("ret_excel_omitted_n"),
            "visible_n": share_of("ret_excel_visible_n"),
            "ret_by_variant": {v: share_of(v) for v in RET_VARIANTS},
            "best_share_by_ret": rep["best_share_by_ret"],
            "best_share_by_service": rep["best_share_by_service"],
            "best_share_by_cobb_douglas": rep["best_share_by_cobb_douglas"],
            "ret_agrees_with_service": rep["ret_agrees_with_service"],
            "cobb_douglas_agrees_with_service": rep["cobb_douglas_agrees_with_service"],
        }
        # The punchline, computed and not typed: what each metric's own optimum actually delivers.
        i_ret = m1["shares"].index(rep["best_share_by_ret"])
        i_svc = m1["shares"].index(rep["best_share_by_service"])
        row["fill_at_ret_optimum"] = row["fill"][i_ret]
        row["fill_at_service_optimum"] = row["fill"][i_svc]
        row["lost_orders_at_ret_optimum"] = row["lost_orders"][i_ret]
        row["lost_orders_at_service_optimum"] = row["lost_orders"][i_svc]
        row["omitted_at_ret_optimum"] = row["omitted_n"][i_ret]
        row["omitted_at_service_optimum"] = row["omitted_n"][i_svc]
        # The invariant that closes off cherry-picking: no ReT variant can be defended by
        # choosing a different one, because EVERY variant bottoms out where service peaks.
        row["ret_argmin_by_variant"] = {
            v: m1["shares"][int(np.argmin(row["ret_by_variant"][v]))] for v in RET_VARIANTS}
        row["ret_argmax_by_variant"] = {
            v: m1["shares"][int(np.argmax(row["ret_by_variant"][v]))] for v in RET_VARIANTS}
        row["variant_is_best_at_the_service_optimum"] = {
            v: s == rep["best_share_by_service"] for v, s in row["ret_argmax_by_variant"].items()}
        row["variant_is_worst_at_the_service_optimum"] = {
            v: s == rep["best_share_by_service"] for v, s in row["ret_argmin_by_variant"].items()}
        # Relative spread, because a variant that ranks nothing cannot be said to rank this
        # wrongly either: the exception below is a variant that barely moves at all.
        row["ret_relative_spread_by_variant"] = {
            v: (max(y) - min(y)) / max(y) if max(y) > 0 else 0.0
            for v, y in ((v, row["ret_by_variant"][v]) for v in RET_VARIANTS)}
        m1_out["by_regime"][regime] = row
    cells = [(regime, v) for regime in m1["regimes"] for v in RET_VARIANTS]
    hits = [(regime, v) for regime, v in cells
            if m1_out["by_regime"][regime]["variant_is_worst_at_the_service_optimum"][v]]
    m1_out["best_at_service_optimum"] = {
        "n_cells": len(cells),
        "n_hits": sum(1 for regime, v in cells
                      if m1_out["by_regime"][regime]["variant_is_best_at_the_service_optimum"][v]),
        "note": "zero hits is the claim: no ReT variant, in either regime, is MAXIMISED where "
                "service is maximised"}
    m1_out["worst_at_service_optimum"] = {
        "n_cells": len(cells), "n_hits": len(hits),
        "exceptions": [{"regime": regime, "variant": v,
                        "argmin_share": m1_out["by_regime"][regime]["ret_argmin_by_variant"][v],
                        "relative_spread":
                            m1_out["by_regime"][regime]["ret_relative_spread_by_variant"][v]}
                       for regime, v in cells if (regime, v) not in hits]}

    halted = art["M2_halted"]["falsifiers"]["f3_ttr_censoring_leaves_an_estimand_at_all"]["evidence"]
    rep2 = art["M2_repaired"]
    m2_out = {
        "halted_status": art["M2_halted"]["claim_status"],
        "censored_fraction_by_arm": halted["censored_fraction_by_arm"],
        "max_censoring_allowed": halted["max_censoring_allowed"],
        "repaired_status": rep2["claim_status"],
        "per_arm": rep2["falsifiers"]["f2_the_recovery_endpoint_has_range"]["evidence"]["per_arm"],
        "absorbed_fraction": rep2["falsifiers"]["f2_the_recovery_endpoint_has_range"]["evidence"]["absorbed_fraction"],
        "censored_at_tau_fraction": rep2["falsifiers"]["f2_the_recovery_endpoint_has_range"]["evidence"]["censored_at_tau_fraction"],
        "contrasts": rep2["contrasts"],
        # tau is the RESTRICTION, 1344 h; h1_horizon_hours (6048) is the episode. Not the same
        # field, and labelling the episode as tau is the defect this line exists to avoid.
        "tau_hours": rep2.get("tau_hours"),
        "h1_horizon_hours": rep2.get("h1_horizon_hours"),
        "differing_share": rep2["falsifiers"]["f1_the_arms_deploy_different_configurations"]["evidence"]["differing_share"],
    }

    bn = art["M3"]["falsifiers"]["f6_surface_twins_have_identical_prefix_paths"]["evidence"]["by_normaliser"]
    m3_out = {"status": art["M3"]["claim_status"], "by_normaliser": {}}
    for norm, d in bn.items():
        pu = d["path_unchanged"]
        methods = sorted(pu)
        contexts = sorted(pu[methods[0]])
        m3_out["by_normaliser"][norm] = {
            "methods": methods, "contexts": contexts,
            "path_unchanged": pu,
            "all_paths_unchanged": d["all_paths_unchanged"],
            "protected": {c: d["changed_cells"][c]["protected"] for c in contexts},
            "n_changed_by_method": {m: sum(1 for c in contexts if not pu[m][c]) for m in methods},
        }

    return {"M1": m1_out, "M2": m2_out, "M3": m3_out,
            "sources": {k: {"path": p, "self_sha256": art[k].get("self_sha256"),
                            "claim_status": art[k].get("claim_status")}
                        for k, p in SOURCES.items()}}


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_abandonment(n: dict, out: Path) -> str:
    """M1. Top row: the metrics, each scaled inside its own range so the comparison is of SHAPE --
    one scale per frame, never two. All four ReT variants are drawn, because the finding is not
    about one of them: every variant bottoms out exactly where service peaks. Bottom row: the
    mechanism, in orders."""
    d, shares = n["M1"], n["M1"]["shares"]
    regimes = d["regimes"]
    fig, axes = plt.subplots(2, len(regimes), figsize=(7.2, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    axes = np.atleast_2d(axes)

    def scaled(y):
        y = np.asarray(y, float)
        span = y.max() - y.min()
        return (y - y.min()) / span if span > 0 else y * 0.0

    for col, regime in enumerate(regimes):
        r = d["by_regime"][regime]
        ax = axes[0, col]
        for v in d["ret_variants"]:
            if v == d["adjudicated_ret_column"]:
                continue
            ax.plot(shares, scaled(r["ret_by_variant"][v]), color=VERMIL, lw=1.0, alpha=0.42,
                    zorder=2)
        ax.plot([], [], color=VERMIL, lw=1.0, alpha=0.42, label="three other ReT variants")
        ax.plot(shares, scaled(r["ret_by_variant"][d["adjudicated_ret_column"]]), color=VERMIL,
                marker="o", ms=4.5, lw=2.0, zorder=4,
                label=RET_LABEL[d["adjudicated_ret_column"]])
        ax.plot(shares, scaled(r["cobb_douglas"]), color=BLUE, marker="s", ms=4.5, lw=2.0,
                zorder=4, label="Cobb-Douglas index")
        ax.plot(shares, scaled(r["fill"]), color=GREEN, marker="^", ms=4.5, lw=2.0, zorder=4,
                label="fill rate (service)")
        ax.axvline(r["best_share_by_service"], color=GREEN, ls=":", lw=1.3, zorder=1)
        ax.set_title(f"regime {regime}")
        ax.set_ylim(-0.06, 1.30)
        ax.set_yticks([0, 0.5, 1.0])
        ax.annotate(f"{r['fill_at_service_optimum']*100:.1f}% filled, "
                    f"{r['lost_orders_at_service_optimum']:.0f} lost",
                    xy=(r["best_share_by_service"], 1.06), color=GREEN, fontsize=7.4,
                    ha="center", va="bottom")
        ax.grid(axis="y", color=GREY, alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)

        ax = axes[1, col]
        w = 0.035
        ax.bar(np.asarray(shares) - w / 2, r["lost_orders"], w, color=VERMIL, zorder=3,
               label="orders lost outright")
        ax.bar(np.asarray(shares) + w / 2, r["omitted_n"], w, color=ORANGE, zorder=3,
               label="orders omitted from the ReT computation")
        ax.axvline(r["best_share_by_service"], color=GREEN, ls=":", lw=1.3, zorder=1)
        ax.set_xlabel("share allocated to product A")
        ax.grid(axis="y", color=GREY, alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("value, scaled inside\neach metric's own range")
    axes[1, 0].set_ylabel("orders per episode")
    b, w = n["M1"]["best_at_service_optimum"], n["M1"]["worst_at_service_optimum"]
    axes[0, 0].legend(frameon=False, fontsize=7.2, loc="lower center", ncols=4,
                      bbox_to_anchor=(1.06, 1.13), columnspacing=1.4, handletextpad=0.5)
    axes[1, 0].legend(frameon=False, fontsize=7.2, loc="lower center", ncols=2,
                      bbox_to_anchor=(1.06, 1.02), columnspacing=1.4, handletextpad=0.5)
    fig.suptitle(f"No ReT variant peaks where service peaks ({b['n_hits']}/{b['n_cells']} cells); "
                 f"{w['n_hits']}/{w['n_cells']} bottom out exactly there",
                 fontsize=10, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "fig1_abandonment")
    return "fig1_abandonment"


def fig2_recovery(n: dict, out: Path) -> str:
    """M2. Left: the endpoint with no estimand. Middle: the same endpoint after the repair.
    Right: the contrast the repair made measurable, with its 95% interval."""
    d = n["M2"]
    arms = list(d["per_arm"])
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(7.2, 2.9),
                                        gridspec_kw={"width_ratios": [1, 1.15, 1.15]})

    x = np.arange(len(arms))
    ax0.bar(x, [d["censored_fraction_by_arm"][a] for a in arms], width=0.6,
            color=VERMIL, zorder=3)
    ax0.axhline(d["max_censoring_allowed"], color=GREY, ls="--", lw=1.2,
                label=f"limit {d['max_censoring_allowed']}")
    ax0.set_xticks(x, [ARM_LABEL[a] for a in arms])
    ax0.set_ylim(0, 1.12)
    ax0.set_ylabel("censored fraction")
    ax0.set_title("(a) system TTR:\nno estimand", fontsize=9)
    ax0.legend(frameon=False, fontsize=7)
    for xi, a in zip(x, arms):
        ax0.text(xi, d["censored_fraction_by_arm"][a] + 0.03,
                 f"{d['censored_fraction_by_arm'][a]:.3f}", ha="center", fontsize=7.5)

    w = 0.36
    absorbed = [d["per_arm"][a]["absorbed"] / d["per_arm"][a]["n"] for a in arms]
    censored = [d["per_arm"][a]["censored"] / d["per_arm"][a]["n"] for a in arms]
    ax1.bar(x - w / 2, absorbed, w, color=BLUE, label="absorbed", zorder=3)
    ax1.bar(x + w / 2, censored, w, color=ORANGE, label=r"censored at $\tau$", zorder=3)
    ax1.set_xticks(x, [ARM_LABEL[a] for a in arms])
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("fraction of cells")
    tau = f"$\\tau$ = {d['tau_hours']:g} h" if d.get("tau_hours") else r"restricted $\tau$"
    ax1.set_title(f"(b) restricted TTR:\nthe endpoint separates ({tau})", fontsize=9)
    ax1.legend(frameon=False, fontsize=7)

    labels = ["hybrid\nvs static", "hybrid\nvs reset"]
    keys = ["H1_hybrid_vs_static", "H1_hybrid_vs_reset"]
    means = [d["contrasts"][k]["mean"] for k in keys]
    lo = [d["contrasts"][k]["mean"] - d["contrasts"][k]["lcb95"] for k in keys]
    hi = [d["contrasts"][k]["ucb95"] - d["contrasts"][k]["mean"] for k in keys]
    y = np.arange(len(keys))
    ax2.errorbar(means, y, xerr=[lo, hi], fmt="o", color=BLUE, ms=6, capsize=4, lw=1.8, zorder=3)
    ax2.axvline(0.0, color=GREY, lw=1.0)
    ax2.set_yticks(y, labels)
    ax2.set_ylim(-0.6, len(keys) - 0.4)
    ax2.set_xlabel("recovery advantage (hours)")
    ax2.set_title("(c) H1, once there is\nsomething to compare", fontsize=9)
    for m, yi in zip(means, y):
        ax2.text(m, yi + 0.22, f"+{m:.1f} h", ha="center", fontsize=7.5, color=BLUE)
    for ax in (ax0, ax1, ax2):
        ax.grid(axis="both", color=GREY, alpha=0.15, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "fig2_recovery_endpoint")
    return "fig2_recovery_endpoint"


def fig3_leak(n: dict, out: Path) -> str:
    """M3. The twin-surface test. A filled cell means the search path CHANGED when only cells the
    method never visited were altered -- which is only possible if the normaliser read them."""
    d = n["M3"]["by_normaliser"]
    norms = ["oracle", "prefix"]
    ctxs = d["oracle"]["contexts"]
    methods = d["oracle"]["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, norm in zip(axes, norms):
        pu = d[norm]["path_unchanged"]
        for j, m in enumerate(methods):
            for i, c in enumerate(ctxs):
                changed = not pu[m][c]
                ax.add_patch(plt.Rectangle((i - 0.42, j - 0.42), 0.84, 0.84,
                                           facecolor=VERMIL if changed else "white",
                                           edgecolor=VERMIL if changed else GREY,
                                           lw=1.2, zorder=2))
        n_changed = sum(d[norm]["n_changed_by_method"].values())
        ax.set_xlim(-0.6, len(ctxs) - 0.4)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_xticks(range(len(ctxs)), ctxs, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(methods)), [METHOD_LABEL.get(m, m) for m in methods], fontsize=8)
        ax.set_title(f"{norm} normaliser — {n_changed} of "
                     f"{len(ctxs) * len(methods)} paths move", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Altering only never-visited cells moves the search — under the oracle "
                 "normaliser only", fontsize=10, y=1.06)
    fig.text(0.5, -0.13, "filled = the search path changed. OFAT and random do not read the "
             "normalised signal, so they cannot move: they are the internal control.",
             ha="center", fontsize=7.6, color=GREY)
    fig.tight_layout()
    save(fig, out, "fig3_normaliser_leak")
    return "fig3_normaliser_leak"


def emit_macros(n: dict) -> str:
    """LaTeX macros, so no sentence in the manuscript can hold a number of its own. A figure and a
    claim that both expand the same macro cannot disagree."""
    m1, m2, m3 = n["M1"], n["M2"], n["M3"]
    r2, r12 = m1["by_regime"]["R2r"], m1["by_regime"]["R1r+R2r"]
    b, w = m1["best_at_service_optimum"], m1["worst_at_service_optimum"]
    exc = w["exceptions"][0] if w["exceptions"] else None
    ora, pre = m3["by_normaliser"]["oracle"], m3["by_normaliser"]["prefix"]
    n_cells_m3 = len(ora["contexts"]) * len(ora["methods"])
    d = {
        # M1 -- the abandonment test
        "MOneRetPeaksAtService": f"{b['n_hits']}", "MOneCells": f"{b['n_cells']}",
        "MOneRetBottomsAtService": f"{w['n_hits']}",
        "MOneExceptionVariant": (RET_LABEL[exc["variant"]] if exc else "none"),
        "MOneExceptionRegime": (exc["regime"] if exc else "none"),
        "MOneFillAtRetOptRTwo": f"{r2['fill_at_ret_optimum']*100:.1f}",
        "MOneFillAtSvcOptRTwo": f"{r2['fill_at_service_optimum']*100:.1f}",
        "MOneLostAtRetOptRTwo": f"{r2['lost_orders_at_ret_optimum']:.1f}",
        "MOneLostAtSvcOptRTwo": f"{r2['lost_orders_at_service_optimum']:.0f}",
        "MOneOmittedAtRetOptRTwo": f"{r2['omitted_at_ret_optimum']:.1f}",
        "MOneOmittedAtSvcOptRTwo": f"{r2['omitted_at_service_optimum']:.1f}",
        "MOneFillAtRetOptBoth": f"{r12['fill_at_ret_optimum']*100:.1f}",
        "MOneFillAtSvcOptBoth": f"{r12['fill_at_service_optimum']*100:.1f}",
        "MOneLostAtRetOptBoth": f"{r12['lost_orders_at_ret_optimum']:.1f}",
        "MOneRetOptShare": f"{r2['best_share_by_ret']}",
        "MOneSvcOptShare": f"{r2['best_share_by_service']}",
        "MOneCDOptShare": f"{r2['best_share_by_cobb_douglas']}",
        # M2 -- the endpoint that never closes
        "MTwoCensoredHybrid": f"{m2['censored_fraction_by_arm']['hybrid']:.3f}",
        "MTwoCensoredReset": f"{m2['censored_fraction_by_arm']['reset']:.3f}",
        "MTwoCensoredStatic": f"{m2['censored_fraction_by_arm']['static']:.3f}",
        "MTwoCensorLimit": f"{m2['max_censoring_allowed']}",
        "MTwoTau": f"{m2['tau_hours']:.0f}", "MTwoHorizon": f"{m2['h1_horizon_hours']:.0f}",
        "MTwoAbsorbedFrac": f"{m2['absorbed_fraction']:.4f}",
        "MTwoCensoredFrac": f"{m2['censored_at_tau_fraction']:.4f}",
        "MTwoAbsorbedHybrid": f"{m2['per_arm']['hybrid']['absorbed']}",
        "MTwoAbsorbedStatic": f"{m2['per_arm']['static']['absorbed']}",
        "MTwoCellsPerArm": f"{m2['per_arm']['hybrid']['n']}",
        "MTwoHOneMean": f"{m2['contrasts']['H1_hybrid_vs_static']['mean']:.1f}",
        "MTwoHOneLCB": f"{m2['contrasts']['H1_hybrid_vs_static']['lcb95']:.1f}",
        "MTwoHOneUCB": f"{m2['contrasts']['H1_hybrid_vs_static']['ucb95']:.1f}",
        "MTwoDifferingShare": f"{m2['differing_share']*100:.1f}",
        # M3 -- the normaliser that reads what was never run
        "MThreeOracleMoved": f"{sum(ora['n_changed_by_method'].values())}",
        "MThreePrefixMoved": f"{sum(pre['n_changed_by_method'].values())}",
        "MThreeCells": f"{n_cells_m3}", "MThreeContexts": f"{len(ora['contexts'])}",
        "MThreeMethods": f"{len(ora['methods'])}",
    }
    head = ("% GENERATED by scripts/build_paper1_evidence_v1.py from the sealed artifacts.\n"
            "% Do not edit. A number typed here instead of expanded is a number that can drift.\n")
    return head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in d.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--paper-root", type=Path,
                    default=Path("papers/paper1_unobserved_frontier"))
    args = ap.parse_args()

    n = extract()
    figdir = args.paper_root / "figures"
    stems = [fig1_abandonment(n, figdir), fig2_recovery(n, figdir), fig3_leak(n, figdir)]

    payload = {
        "schema_version": "paper1_evidence_v1",
        "claim_status": "PAPER1_NUMBERS_AND_FIGURES_EXTRACTED",
        "scope": "EXTRACTION_AND_RENDERING_ONLY_NO_SCIENTIFIC_CLAIM_NO_SEEDS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": "none_extraction_only",
        "module_manifest": module_manifest(MODULES, script=__file__),
        "preregistration": str(args.contract),
        "figures": [str(figdir / f"{s}.pdf") for s in stems],
        "macro_file": str(args.paper_root / "numbers.tex"),
        "numbers": n,
    }
    numbers_path = args.paper_root / "numbers.json"
    args.paper_root.mkdir(parents=True, exist_ok=True)
    numbers_path.write_text(json.dumps(n, indent=2, sort_keys=True))
    tex_path = args.paper_root / "numbers.tex"
    tex_path.write_text(emit_macros(n))
    payload_extra = str(tex_path)
    digest = seal_and_write(payload, args.paper_root / "evidence_receipt.json",
                            contract=args.contract, reference=AMENDMENT)

    m1 = n["M1"]["by_regime"]
    for regime, r in m1.items():
        print(f"  M1 {regime:10s} ReT optimum {r['best_share_by_ret']} -> "
              f"{r['fill_at_ret_optimum']*100:.1f}% filled, "
              f"{r['lost_orders_at_ret_optimum']:.1f} lost · service optimum "
              f"{r['best_share_by_service']} -> {r['fill_at_service_optimum']*100:.1f}% filled, "
              f"{r['lost_orders_at_service_optimum']:.1f} lost · CD agrees: "
              f"{r['cobb_douglas_agrees_with_service']}")
    b = n["M1"]["best_at_service_optimum"]
    print(f"  M1 ReT is BEST at the service optimum in {b['n_hits']}/{b['n_cells']} "
          f"regime x variant cells")
    w = n["M1"]["worst_at_service_optimum"]
    print(f"  M1 ReT is WORST exactly at the service optimum in {w['n_hits']}/{w['n_cells']} "
          f"regime x variant cells")
    for e in w["exceptions"]:
        print(f"     exception: {e['variant']} under {e['regime']} -- argmin {e['argmin_share']}, "
              f"relative spread only {e['relative_spread']*100:.1f}%")
    print(f"  M2 censored 1.000 -> absorbed {n['M2']['absorbed_fraction']:.4f}, "
          f"censored {n['M2']['censored_at_tau_fraction']:.4f}")
    for norm, d in n["M3"]["by_normaliser"].items():
        print(f"  M3 {norm:7s} paths moved: {d['n_changed_by_method']}")
    print(f"  -> {len(stems)} figures in {figdir} · {numbers_path} · seal {digest[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
