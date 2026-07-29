#!/usr/bin/env python3
"""Figures for docs/action_space_design_note (the action-contract explainer for Garrido).

House style matches scripts/build_manuscript_figures.py: Okabe-Ito palette,
STIX serif, vector PDF + 300dpi PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/action_space_design_note/figures")

BLUE = "#0072B2"
GREEN = "#009E73"
VERMIL = "#D55E00"
GREY = "#7f7f7f"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
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


def fig_reduction_chain() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 3.6))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    def stage(x, w, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, 0.85), w, 2.55,
                     boxstyle="round,pad=0.04,rounding_size=0.10",
                     fc=fc, ec=ec, lw=1.1, zorder=2))
        ax.text(x + w / 2, 3.68, title, fontsize=9.6, fontweight="bold",
                ha="center", va="bottom")
        ax.text(x + w / 2, 2.12, body, fontsize=8.1, ha="center", va="center",
                linespacing=1.4)

    def arrow(x0, x1, label):
        ax.add_patch(FancyArrowPatch((x0, 2.12), (x1, 2.12), arrowstyle="-|>",
                     mutation_scale=13, color="0.25", lw=1.3, zorder=1))
        ax.text((x0 + x1) / 2, 3.68, label, fontsize=7.6, color="0.35",
                ha="center", va="bottom", style="italic", linespacing=1.3)

    w = 3.0
    gap = 1.15
    xs = [0.15 + i * (w + gap) for i in range(4)]

    stage(xs[0], w, "Garrido-Rios (2017)",
          "3 ubicaciones $\\times$ 5 niveles\ndiscretos de reposición\n"
          "($I_{168}\\ldots I_{1344}$, horas)\n$+$ 3 turnos discretos\n"
          "un factor a la vez",
          "#f6f8fa", "0.35")
    stage(xs[1], w, "continuous\\_its (2D)",
          "una fracción continua\ncomún de buffer en\n"
          "Op3/Op5/Op9\n$+$ 1 señal continua de turno\n"
          "(lane exploratorio inicial)",
          "#eaf1fb", BLUE)
    stage(xs[2], w, "per\\_op\\_buffer (4D)",
          "3 fracciones continuas\nindependientes de buffer\n"
          "(Op3, Op5, Op9)\n$+$ 1 señal continua de turno\n"
          "(contrato real de Track A)",
          "#eaf5ea", GREEN)
    stage(xs[3], w, "Track B (8D)",
          "misma familia upstream\n(forma multiplicativa)\n"
          "$+$ despacho Op10\n$+$ despacho Op12\n"
          "(la extensión ganadora)",
          "#fdf1df", VERMIL)

    arrow(xs[0] + w + 0.08, xs[1] - 0.08, "de-discretizar")
    arrow(xs[1] + w + 0.08, xs[2] - 0.08, "quitar restricción\nde fracción común")
    arrow(xs[2] + w + 0.08, xs[3] - 0.08, "añadir autoridad\nde despacho")

    ax.text(8.3, 0.35,
            "Cada etapa conserva los propios puntos de decisión de Garrido (niveles de buffer, turno); "
            "solo Track B añade una palanca genuinamente nueva (despacho).",
            fontsize=8.2, color="0.3", ha="center", va="bottom")

    save(fig, "fig_reduction_chain")


def fig_dof_comparison() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    rows = [
        ("Garrido-Rios (tesis)", "15 configuraciones discretas de buffer (3 ubic. $\\times$ 5 niveles) "
         "+ 3 turnos discretos;\nvariados un factor a la vez (nunca conjuntamente)", GREY),
        ("continuous\\_its", "1 fracción continua de buffer (compartida) + 1 señal continua de turno;\n"
         "de-discretizado, pero obliga a las 3 ubicaciones a moverse juntas", BLUE),
        ("per\\_op\\_buffer (Track A)", "3 fracciones continuas independientes de buffer + 1 señal "
         "continua de turno;\ngranularidad de Garrido preservada, grilla DOE eliminada", GREEN),
    ]
    y = 2.55
    for label, desc, col in rows:
        ax.add_patch(plt.Rectangle((0.1, y - 0.62), 0.12, 0.62, fc=col, ec="none"))
        ax.text(0.35, y - 0.08, label, fontsize=9.3, fontweight="bold", va="top")
        ax.text(0.35, y - 0.34, desc, fontsize=8.0, va="top", linespacing=1.35, color="0.25")
        y -= 0.95

    save(fig, "fig_dof_comparison")


if __name__ == "__main__":
    fig_reduction_chain()
    fig_dof_comparison()
    print("done")
