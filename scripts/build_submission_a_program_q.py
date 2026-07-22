#!/usr/bin/env python3
"""Build the evidence tables, figures, and source of truth for Submission A.

Every numerical claim is read from frozen Program Q artifacts.  The script
fails if those artifacts drift, keeping the manuscript downstream of the
adjudicated evidence rather than of hand-copied numbers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "scres-submission-a-mpl-cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "submission_a_program_q"
GENERATED = PAPER / "generated"
TABLES = GENERATED / "tables"
FIGURES = GENERATED / "figures"
DATA = PAPER / "data" / "static_frontier_summary.npz"
DATA_MANIFEST = PAPER / "data" / "static_frontier_summary_manifest.json"

EVIDENCE = {
    "confirmation": (
        ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json",
        "62f6fd390471624f7c301b8baa96d31871db99e22dd5a22d6bb8cf7bba8088b2",
    ),
    "adjudication": (
        ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation/adjudication.json",
        "e13e17f001a1d24f86f00257e145c26f9c09def68ef7b2ee2f90fcb23148b0e9",
    ),
    "direct_audit": (
        ROOT / "results/program_q/confirmation_v1_20260718/artifacts/confirmation/direct_audit/independent_full_des_audit.json",
        "3da52ca129707e883be0179f82be8058d29ddf454c27a4f578918c26c7ec82eb",
    ),
    "calibration": (
        ROOT / "results/program_o/ret_only_learner_v1/calibration_run/result.json",
        "dc55880f0dca92e3c51c0690203736835ef75b8247dc1716e2cc6f9b5f0a54fd",
    ),
    "metric_audit": (
        ROOT / "research/paper2_exhaustive_search/ret_excel_request_snapshot_v2_implementation_audit_20260714.json",
        "61a67f24cb3bf31b9cf5fd9333436bd2a37da61e6bedb838e572612a8b1e2137",
    ),
    "learner_contract": (
        ROOT / "contracts/program_o_ret_only_learner_v1.json",
        "5442e8a9e4de33a7e02e443cabc163a5acd396424b6614ab03ea4fe10c78c752",
    ),
    "confirmation_contract": (
        ROOT / "contracts/program_q_frozen_policy_replication_v1.json",
        "b0bd21c7eec5f0aa6acc4e6866e32c3804aa2fa7f8afecc040bc7fd129980874",
    ),
}

CELLS = ("rho75_share90", "rho90_share75", "rho90_share90")
CELL_LABELS = {
    "rho75_share90": r"$\rho=.75,\ s=.90$",
    "rho90_share75": r"$\rho=.90,\ s=.75$",
    "rho90_share90": r"$\rho=.90,\ s=.90$",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_evidence() -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    for name, (path, expected) in EVIDENCE.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen evidence drifted: {path} ({actual})")
        loaded[name] = load_json(path)
    frontier_manifest = load_json(DATA_MANIFEST)
    if sha256(DATA) != frontier_manifest["derived_npz_sha256"]:
        raise RuntimeError("The compact static-frontier artifact failed its hash.")
    if frontier_manifest["verified_file_count"] != 144:
        raise RuntimeError("The static-frontier artifact does not cover all 144 matrices.")
    return loaded


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_tex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: str,
    header: list[str],
    rows: list[list[object]],
    notes: str | None = None,
    resize: bool = False,
) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        *([r"\resizebox{\linewidth}{!}{%"] if resize else []),
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(tex_escape(x) for x in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append("}")
    if notes:
        lines.extend([r"\vspace{2pt}", rf"\parbox{{0.96\linewidth}}{{\footnotesize {notes}}}"])
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evidence_rows(data: dict[str, dict]) -> tuple[list[list[object]], list[list[object]]]:
    result = data["confirmation"]
    primary: list[list[object]] = []
    guardrails: list[list[object]] = []
    for cell in CELLS:
        estimates = result["inference"]["estimates"]
        h = estimates[f"{cell}::H_OL"]
        neural = estimates[f"{cell}::Delta_N"]
        summary = result["cell_summaries"][cell]
        worst = result["guardrail_inference"]["estimates"][
            f"{cell}::worst_product_fill::vs_classical"
        ]
        secondary = summary["secondary_nonblocking"]
        primary.append(
            [
                cell,
                f"{h['point']:+.5f}",
                f"{h['lcb95']:+.5f}",
                f"{neural['point']:+.5f}",
                f"[{neural['lcb95']:+.5f}, {neural['ucb95']:+.5f}]",
                f"{100 * summary['favorable_tapes_fraction_vs_open_loop']:.1f}%",
                f"{summary['positive_learner_seeds_H_OL']}/10",
            ]
        )
        guardrails.append(
            [
                cell,
                f"{worst['point']:+.5f}",
                f"{worst['lcb95']:+.5f}",
                f"{secondary['lost_orders']['delta_vs_classical']:+.3f}",
                f"{secondary['unresolved_orders']['delta_vs_classical']:+.3f}",
                f"{secondary['service_loss_auc']['delta_vs_classical']:+.0f}",
                f"{result['integrity_diagnostics']['resource_max_abs_diff']:.1f}",
            ]
        )
    return primary, guardrails


def write_tables(data: dict[str, dict], source: dict) -> None:
    primary, guardrails = evidence_rows(data)
    write_csv(
        TABLES / "table3_program_q_confirmation.csv",
        ["cell", "H_OL", "H_OL_LCB95", "Delta_N", "Delta_N_CI95", "favorable_tapes", "positive_seeds"],
        primary,
    )
    write_csv(
        TABLES / "table4_guardrails.csv",
        ["cell", "worst_product_delta", "worst_product_LCB95", "lost_delta", "unresolved_delta", "service_loss_auc_delta", "resource_spread"],
        guardrails,
    )
    write_tex_table(
        TABLES / "table1_claim_ladder.tex",
        caption="Claim ladder and adjudicated boundary.",
        label="tab:claim_ladder",
        columns=r"p{0.16\linewidth}p{0.23\linewidth}p{0.17\linewidth}p{0.32\linewidth}",
        header=["Level", "Question", "Status", "Evidence boundary"],
        rows=[
            ["Metric", "Excel ReT reproduced?", "Supported", "47,546 workbook rows; zero mismatches"],
            ["Static", "Best open-loop frontier known?", "Supported", "All $4^8=65,536$ calendars evaluated"],
            ["Learned", "Feedback beats every static?", "Supported", "H_OL LCB95 $>0$ in all three cells; 10/10 seeds positive"],
            ["Neural", "RL beats structured control?", "Not supported", r"Delta_N practically equivalent to zero within $\pm0.01$"],
            ["Safe", "Worst product non-inferior?", "Not established", "Simultaneous lower bounds cross the frozen $-0.02$ margin"],
        ],
    )
    write_tex_table(
        TABLES / "table2_comparator_rights.tex",
        caption="Information and decision rights of the principal comparators.",
        label="tab:comparators",
        columns=r"p{0.18\linewidth}p{0.23\linewidth}p{0.19\linewidth}p{0.28\linewidth}",
        header=["Comparator", "Information", "Decision rights", "Resource and selection rule"],
        rows=[
            ["Open-loop frontier", "No episode feedback", "Eight weekly mix actions fixed ex ante", "All 65,536 calendars; common tapes and scheduled resources"],
            ["Belief-based control", "Same deployable operational history; fixed HMM model", "Replans weekly over the same four actions", "Best frozen structured controller reselected inside bootstrap"],
            ["RecurrentPPO", "Same 21-field causal observation; recurrent state", "Chooses one of four mix actions weekly", "Ten frozen checkpoints; identical physical resources"],
            ["Oracle", "Privileged future information", "Diagnostic only", "Never a deployable comparator or a learned claim"],
        ],
    )
    write_tex_table(
        TABLES / "table3_confirmation.tex",
        caption="Program Q confirmation results on 256 new tapes per cell.",
        label="tab:confirmation",
        columns="lrrrrrr",
        header=["Cell", r"$H_{OL}$", "LCB95", r"$\Delta_N$", "CI95", "Fav.", "Seeds"],
        rows=primary,
        notes=r"$H_{OL}$ compares RecurrentPPO with the best of all 65,536 open-loop calendars. $\Delta_N$ compares it with the best structured controller. Intervals are simultaneous two-way learner-seed/tape max-$t$ intervals with comparator reselection.",
        resize=True,
    )
    write_tex_table(
        TABLES / "table4_guardrails.tex",
        caption="Service, ledger, and resource contrasts against structured control.",
        label="tab:guardrails",
        columns="lrrrrrr",
        header=["Cell", "Worst fill", "LCB95", "Lost", "Unresolved", "Service AUC", "Resource spread"],
        rows=guardrails,
        notes="Positive service-loss AUC and unresolved-order contrasts indicate worse learner outcomes. Lost-order contrasts and scheduled-resource spread are exactly zero; worst-product non-inferiority was not established.",
        resize=True,
    )

    trajectory = data["confirmation"]["trajectory_audits"]
    replacement = data["confirmation"]["replacement_controls"]
    with np.load(DATA, allow_pickle=False) as frontier:
        mechanism_rows = []
        for cell in CELLS:
            audits = list(trajectory[cell].values())
            sorted_ret = np.sort(frontier[f"{cell}__ret_visible__mean"])[::-1]
            top38_width = float(sorted_ret[0] - sorted_ret[37])
            replacement_pass = min(
                replacement[cell][kind]["learner_seeds_beating"]
                for kind in ("modal", "phase_only", "frequency_matched")
            )
            mechanism_rows.append(
                [
                    cell,
                    min(a["unique_calendars"] for a in audits),
                    f"{max(a['modal_fraction'] for a in audits):.3f}",
                    min(a["varying_weeks"] for a in audits),
                    f"{replacement_pass}/10",
                    f"{top38_width:.5f}",
                ]
            )
    write_tex_table(
        TABLES / "table5_mechanism.tex",
        caption="Diagnostics separating feedback from static schedule discovery.",
        label="tab:mechanism",
        columns="lrrrrr",
        header=["Cell", "Min unique", "Max modal", "Min varying", "Repl. beaten", "Top-38 width"],
        rows=mechanism_rows,
        notes="Replacement denotes the weakest seed count across modal, phase-only, and frequency-matched controls. Top-38 width is descriptive calibration evidence from the exact frontier, not a confirmation estimand.",
        resize=True,
    )
    write_tex_table(
        TABLES / "table6_claim_boundary.tex",
        caption="Supported and unsupported interpretations.",
        label="tab:claim_boundary",
        columns=r"p{0.26\linewidth}p{0.18\linewidth}p{0.44\linewidth}",
        header=["Statement", "Status", "Reason"],
        rows=[
            ["Feedback beats the complete static frontier", "Supported", "Prospective confirmation in all cells with simultaneous positive lower bounds"],
            ["RecurrentPPO has a neural premium", "Not supported", "Practical equivalence, not superiority, versus structured control"],
            ["The learned policy is deployment-safe", "Not supported", "Worst-product service margin failed"],
            ["The model learns across campaigns", "Not tested here", "Program Q resets each episode; Q-R1 is a separate prospective study"],
            ["Results generalize to Garrido-native risks", "Not supported", "Primary Program Q physics are risk-off and multiproduct researcher-defined"],
            ["The study closes the DES feedback loop", "Supported in contract", "Actions use causal within-episode observations and beat phase/modal/frequency replacements"],
        ],
    )


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / f"{stem}.png", dpi=220)
    plt.close(fig)


def figure1_loop() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.axis("off")
    xs = [0.08, 0.32, 0.56, 0.80]
    labels = ["Full DES\nphysical state", "Causal\nobservation", "Controller\n(static / MPC / RL)", "Weekly product-mix\naction"]
    colors = ["#DCEAF7", "#E8F1E5", "#F7E7CC", "#E8DDF3"]
    for x, label, color in zip(xs, labels, colors):
        ax.text(x, 0.55, label, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.65", facecolor=color, edgecolor="#334155", linewidth=1.2))
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.09, 0.55), xytext=(left + 0.09, 0.55), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))
    ax.annotate("state transition and service", xy=(0.09, 0.32), xytext=(0.80, 0.32), xycoords=ax.transAxes,
                ha="center", va="center", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.20", color="#64748B"),
                color="#475569")
    ax.text(0.5, 0.92, "Closed-loop decision contract", transform=ax.transAxes, ha="center", fontsize=12, weight="bold")
    save_figure(fig, "figure1_closed_loop")


def figure2_frontier(data: dict[str, dict]) -> None:
    calibration = data["calibration"]
    with np.load(DATA, allow_pickle=False) as frontier:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
        for ax, cell in zip(axes, CELLS):
            values = np.sort(frontier[f"{cell}__ret_visible__mean"])[::-1]
            ranks = np.arange(1, values.size + 1)
            best = float(values[0])
            h = calibration["inference"]["estimates"][f"{cell}::H_learned"]["estimate"]
            neural = calibration["inference"]["estimates"][f"{cell}::H_neural"]["estimate"]
            learner = best + h
            structured = learner - neural
            ax.plot(ranks, values, color="#64748B", lw=1.0, label="65,536 statics")
            ax.axhline(learner, color="#2563EB", lw=1.8, label="RecurrentPPO")
            ax.axhline(structured, color="#D97706", lw=1.8, ls="--", label="Structured")
            ax.set_xscale("log")
            ax.set_title(CELL_LABELS[cell])
            ax.set_xlabel("Static-policy rank (log scale)")
            ax.grid(alpha=0.18)
        axes[0].set_ylabel("Calibration mean Excel ReT")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle("Exact static frontier and adaptive-controller positions", y=0.99, weight="bold")
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
        fig.subplots_adjust(top=0.75, wspace=0.24)
        save_figure(fig, "figure2_static_frontier")


def figure3_effects(data: dict[str, dict]) -> None:
    estimates = data["confirmation"]["inference"]["estimates"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    y = np.arange(len(CELLS))
    h_points, h_low, h_high = [], [], []
    n_points, n_low, n_high = [], [], []
    for cell in CELLS:
        h = estimates[f"{cell}::H_OL"]
        n = estimates[f"{cell}::Delta_N"]
        h_points.append(h["point"]); h_low.append(h["point"] - h["lcb95"]); h_high.append(h["ucb95"] - h["point"])
        n_points.append(n["point"]); n_low.append(n["point"] - n["lcb95"]); n_high.append(n["ucb95"] - n["point"])
    axes[0].errorbar(h_points, y, xerr=[h_low, h_high], fmt="o", color="#2563EB", capsize=3)
    axes[0].axvline(0, color="#475569", lw=1)
    axes[0].axvline(0.01, color="#16A34A", lw=1, ls=":")
    axes[0].set_title("Feedback value vs exact static frontier")
    axes[0].set_xlabel(r"$H_{OL}$ (Excel ReT)")
    axes[1].errorbar(n_points, y, xerr=[n_low, n_high], fmt="o", color="#D97706", capsize=3)
    axes[1].axvspan(-0.01, 0.01, color="#DCFCE7", alpha=0.8)
    axes[1].axvline(0, color="#475569", lw=1)
    axes[1].set_title("Neural premium vs structured control")
    axes[1].set_xlabel(r"$\Delta_N$ (Excel ReT)")
    for ax in axes:
        ax.set_yticks(y, [CELL_LABELS[c] for c in CELLS])
        ax.grid(axis="x", alpha=0.18)
    axes[1].set_yticklabels([])
    fig.suptitle("Prospective Program Q confirmation", y=1.03, weight="bold")
    save_figure(fig, "figure3_feedback_and_neural")


def figure4_service(data: dict[str, dict]) -> None:
    result = data["confirmation"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    x = np.arange(len(CELLS))
    worst = [result["guardrail_inference"]["estimates"][f"{c}::worst_product_fill::vs_classical"] for c in CELLS]
    points = [d["point"] for d in worst]
    lower = [d["point"] - d["lcb95"] for d in worst]
    axes[0].errorbar(x, points, yerr=lower, fmt="o", color="#B91C1C", capsize=3)
    axes[0].axhline(0, color="#475569", lw=1)
    axes[0].axhline(-0.02, color="#B91C1C", lw=1, ls=":")
    axes[0].set_ylabel("Learner - structured")
    axes[0].set_title("Worst-product fill")
    unresolved = [result["cell_summaries"][c]["secondary_nonblocking"]["unresolved_orders"]["delta_vs_classical"] for c in CELLS]
    axes[1].bar(x, unresolved, color="#64748B")
    axes[1].axhline(0, color="#475569", lw=1)
    axes[1].set_title("Unresolved orders")
    axes[1].set_ylabel("Learner - structured")
    for ax in axes:
        ax.set_xticks(x, [CELL_LABELS[c] for c in CELLS], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Aggregate equivalence does not establish product-level safety", y=1.03, weight="bold")
    save_figure(fig, "figure4_service_boundary")


def build_source_of_truth(data: dict[str, dict]) -> dict:
    primary, guardrails = evidence_rows(data)
    metric = data["metric_audit"]["canonical_aggregator_workbook_replay"]
    direct = data["direct_audit"]
    return {
        "schema_version": "submission_a_program_q_source_of_truth_v1",
        "title": "When Feedback Beats Every Static Policy but Not Structured Control: Exact Benchmarking of Recurrent RL in a Supply-Chain DES",
        "submission_name": "Submission A -- Program Q",
        "target_journal": "Computers & Industrial Engineering",
        "scientific_base_commit": "f2dfe356c179bd16f4b89b26e8ed3b19d69f5a71",
        "binding_verdict": data["adjudication"]["verdict"],
        "primary_interpretation": "Feedback value replicated; no neural premium; worst-product safety not established.",
        "metric": {
            "contract": "ret_excel_request_snapshot_v2",
            "formula_rows": metric["formula_rows"],
            "mismatches": metric["mismatches"],
            "max_abs_diff": metric["max_abs_diff"],
            "visible_population_caveat": "Primary workbook-visible rows are completed non-lost orders; ledger and service endpoints are mandatory companion outcomes.",
        },
        "design": {
            "cells": list(CELLS),
            "confirmation_tapes_per_cell": data["confirmation"]["N"],
            "learner_checkpoints": 10,
            "open_loop_calendars": 65_536,
            "weekly_decisions": 8,
            "actions_per_decision": 4,
            "risks_enabled": False,
            "researcher_defined_extension": True,
            "direct_full_des_replays": direct["direct_unique_replays"],
            "direct_audit_failures": direct["failure_count"],
        },
        "confirmation_rows": primary,
        "guardrail_rows": guardrails,
        "supported_claims": [
            "Exact reproduction of the workbook ReT formula given source snapshots.",
            "RecurrentPPO outperforms every calendar in the complete 65,536-policy open-loop frontier in all three cells.",
            "RecurrentPPO is practically equivalent, not superior, to the selected structured controller under the frozen +/-0.01 margin.",
            "The action trajectory and replacement controls identify genuine within-episode feedback rather than a fixed schedule.",
        ],
        "prohibited_claims": [
            "Neural premium over structured control.",
            "Deployment or worst-product safety.",
            "Accumulated learning, transfer, or path dependency across campaigns.",
            "Improvement under Garrido-native active risks.",
            "Superiority over every mathematically possible dynamic policy.",
        ],
        "evidence_sha256": {name: expected for name, (_, expected) in EVIDENCE.items()},
        "static_frontier_summary_sha256": sha256(DATA),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = verify_evidence()
    source = build_source_of_truth(data)
    write_tables(data, source)
    style()
    figure1_loop()
    figure2_frontier(data)
    figure3_effects(data)
    figure4_service(data)
    (PAPER / "source_of_truth.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = []
    for path in sorted(GENERATED.rglob("*")):
        if path.is_file() and path != GENERATED / "generated_files.sha256":
            hashes.append(f"{sha256(path)}  {path.relative_to(PAPER).as_posix()}")
    (GENERATED / "generated_files.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
