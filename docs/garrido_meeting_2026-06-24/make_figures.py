#!/usr/bin/env python3
"""Generate figures for the Garrido meeting document (2026-06-24)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BENCH = ROOT / "outputs" / "benchmarks"
BLUE, RED, GREEN, GRAY = "#2c6fbb", "#c0392b", "#27ae60", "#7f8c8d"


def load(p):
    return json.loads((BENCH / p).read_text(encoding="utf-8"))


# --- Fig 1: G1 risk-frequency fidelity (thesis_window vs legacy vs target) ---
def fig_risk_fidelity():
    risks = ["R11", "R21", "R22", "R23", "R24", "R3"]
    target = [48, 0.5, 2, 1, 12, 0.05]
    window = [48.0, 0.5, 2.0, 1.0, 12.0, 0.05]
    legacy = [92.9, 0.85, 3.7, 1.9, 24.4, 0.05]
    x = np.arange(len(risks)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w, target, w, label="Tesis (Tabla 6.11)", color=GRAY)
    ax.bar(x, window, w, label="thesis_window (nuestro)", color=GREEN)
    ax.bar(x + w, legacy, w, label="legacy_renewal (bug)", color=RED)
    ax.set_yscale("log"); ax.set_ylabel("eventos / año (log)")
    ax.set_xticks(x); ax.set_xticklabels(risks)
    ax.set_title("G1: reproducción de frecuencias de riesgo (Tabla 6.11)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig1_risk_fidelity.png", dpi=150)
    plt.close(fig)


# --- Fig 2: headroom — best ReT per shift per regime (optimal shift flips) ---
def fig_headroom():
    t = load("headroom/argmax_stability.json")
    regimes = ["current", "increased", "severe"]
    byshift = {r: {} for r in regimes}
    for r in regimes:
        for row in t[r]["rows"]:
            s = row["policy"].split("_S")[1]
            byshift[r][s] = max(byshift[r].get(s, -9), row["ret"])
    x = np.arange(len(regimes)); w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (s, col) in enumerate(zip(["1", "2", "3"], [BLUE, GREEN, RED])):
        ax.bar(x + (i - 1) * w, [byshift[r][s] for r in regimes], w, label=f"S{s}", color=col)
    ax.set_ylabel("mejor ReT (orden)"); ax.set_xticks(x); ax.set_xticklabels(regimes)
    ax.set_title("Headroom: el turno óptimo cambia con el régimen (S3→S1)")
    ax.legend(title="turno", fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "fig2_headroom_shift.png", dpi=150); plt.close(fig)


def _rr(p):
    d = load(p)["results"][0]["retained_minus_reset_clustered"]
    return d["mean"], d["sem"]


# --- Fig 3: observability ablation (Track A) ---
def fig_observability():
    full_m, full_s = _rr("retained_reset_learning/pilots/obs_full_v1/pilot.json")
    hid_m, hid_s = _rr("retained_reset_learning/pilots/obs_hidden_v1/pilot.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    labels = ["régimen\nvisible", "régimen\noculto"]
    means = [full_m, hid_m]; sems = [full_s, hid_s]
    ax.bar(labels, means, yerr=[1.96 * s for s in sems], capsize=6,
           color=[GRAY, BLUE])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("retained − reset (ReT)")
    ax.set_title("Track A: la retención emerge al ocultar el régimen")
    for i, m in enumerate(means):
        ax.text(i, m + (0.001 if m >= 0 else -0.003), f"{m:+.4f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_observability.png", dpi=150); plt.close(fig)


# --- Fig 4: Track B retention probe (retained-reset vs retained-frozen) ---
def fig_track_b():
    d = load("retention_track_b/probe_v1/retention_track_b.json")["results"]
    conds = ["obs_full", "obs_hidden"]
    rr = [d[c]["retained_minus_reset"]["mean"] for c in conds]
    rr_e = [1.96 * d[c]["retained_minus_reset"]["sem"] for c in conds]
    rf = [d[c]["retained_minus_frozen"]["mean"] for c in conds]
    x = np.arange(len(conds)); w = 0.35
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - w / 2, rr, w, yerr=rr_e, capsize=5, color=BLUE, label="retained − reset (retención)")
    ax.bar(x + w / 2, rf, w, color=GREEN, label="retained − frozen (control adaptativo)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["régimen visible", "régimen oculto"])
    ax.set_ylabel("Δ ReT"); ax.set_title("Track B (probe, 3 semillas): control sí, retención no")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "fig4_track_b.png", dpi=150); plt.close(fig)


# --- Fig 5: synthesis across experiments ---
def fig_synthesis():
    labels = ["A·visible\n24bloq", "A·oculto", "B·visible", "B·oculto"]
    rr = [
        load("retained_reset_learning/pilots/diag_blocks_v1/pilot.json")["results"][0]["retained_minus_reset_clustered"]["mean"],
        _rr("retained_reset_learning/pilots/obs_hidden_v1/pilot.json")[0],
        load("retention_track_b/probe_v1/retention_track_b.json")["results"]["obs_full"]["retained_minus_reset"]["mean"],
        load("retention_track_b/probe_v1/retention_track_b.json")["results"]["obs_hidden"]["retained_minus_reset"]["mean"],
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(labels, rr, color=[BLUE if v >= 0 else RED for v in rr])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("retained − reset (ReT)")
    ax.set_title("Síntesis: retained − reset ≈ 0 en todos los experimentos")
    for i, v in enumerate(rr):
        ax.text(i, v + (0.0005 if v >= 0 else -0.0015), f"{v:+.4f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig5_synthesis.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_risk_fidelity(); fig_headroom(); fig_observability(); fig_track_b(); fig_synthesis()
    print("figures written to", FIG)
