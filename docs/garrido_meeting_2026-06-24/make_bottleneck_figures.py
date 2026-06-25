#!/usr/bin/env python3
"""Figuras del cuello de botella en la MFSC (para el reporte a Garrido)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
GREEN, RED, GRAY, DARK, BLUE = "#27ae60", "#c0392b", "#bdc3c7", "#2c3e50", "#2c6fbb"
acc = json.loads((HERE / "sb_accumulation.json").read_text())

CEILING = 2500          # despacho downstream Op9-12 (Q=2400-2600, ROP diario)
S = {"S1": 2564, "S2": 5128, "S3": 7692}   # 320.5 raciones/h x {8,16,24} h/día


# --- Fig B1: producción vs entrega (S2=S3 en entrega; exceso atrapado) ---
def fig_prod_vs_deliver():
    days = 8064 / 24
    labels = ["S1", "S2", "S3"]
    prod = [acc[s]["produced"] / days for s in ("1", "2", "3")]
    deliv = [acc[s]["delivered"] / days for s in ("1", "2", "3")]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, prod, w, label="producido (Op5–7)", color=GREEN)
    ax.bar(x + w / 2, deliv, w, label="entregado (vía Op9–12)", color=BLUE)
    ax.axhline(CEILING, color=RED, ls="--", lw=1.5, label="techo de despacho Op9–12 (~2500/día)")
    for i, (p, d) in enumerate(zip(prod, deliv)):
        if p - d > 100:
            ax.annotate("", xy=(i + w/2, d), xytext=(i + w/2, p),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
            ax.text(i + w/2 + 0.05, (p + d) / 2, "atrapado", color=RED, fontsize=8, va="center")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("raciones / día")
    ax.set_title("S2 y S3 ENTREGAN lo mismo: el exceso de S3 queda atrapado")
    ax.legend(fontsize=8, loc="upper left"); fig.tight_layout()
    fig.savefig(FIG / "figb1_prod_vs_deliver.png", dpi=150); plt.close(fig)


# --- Fig B2: inventario atrapado creciendo en Op9 (Supply Battalion) ---
def fig_accumulation():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for s, col, lab in (("2", BLUE, "S2"), ("3", RED, "S3")):
        tr = acc[s]["sb_traj"]
        t = [p[0] / 7 for p in tr]; lv = [p[1] / 1000 for p in tr]
        ax.plot(t, lv, color=col, lw=2, label=f"{lab} (final {acc[s]['sb_final']/1000:.0f}k)")
    ax.set_xlabel("semanas"); ax.set_ylabel("inventario atrapado en Op9 (miles de raciones)")
    ax.set_title("Inventario que se acumula en el Supply Battalion (Op9) y no se despacha")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(FIG / "figb2_accumulation.png", dpi=150); plt.close(fig)


# --- Fig B3: la cadena de 13 operaciones con el cuello de botella ---
def fig_chain():
    fig, ax = plt.subplots(figsize=(9.2, 5.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ops = {
        1: ("Op1\nAgencia\nLogística", 0.5, 5.4, GRAY),
        2: ("Op2\nProveedores\n(12)", 3.0, 5.4, GRAY),
        3: ("Op3\nWDC", 5.5, 5.4, GRAY),
        4: ("Op4\nLOC", 8.0, 5.4, GRAY),
        5: ("Op5\nPre-ensamble", 8.0, 3.6, GREEN),
        6: ("Op6\nEnsamble", 5.5, 3.6, GREEN),
        7: ("Op7\nCalidad/Envío", 3.0, 3.6, GREEN),
        8: ("Op8\nLOC", 0.5, 3.6, GRAY),
        9: ("Op9\nSupply\nBattalion", 0.5, 1.6, RED),
        10: ("Op10\nLOC", 3.0, 1.6, RED),
        11: ("Op11\nCSSU (2)", 5.5, 1.6, RED),
        12: ("Op12\nLOC", 8.0, 1.6, RED),
        13: ("Op13\nTeatro\n(demanda)", 8.0, 0.1, DARK),
    }
    W, H = 1.7, 1.1
    anc = {}
    for k, (txt, x, y, col) in ops.items():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, ec="black", lw=1))
        fc = "white" if col in (RED, DARK) else "black"
        ax.text(x + W/2, y + H/2, txt, ha="center", va="center", fontsize=7.5, color=fc)
        anc[k] = {"cb": (x + W/2, y), "ct": (x + W/2, y + H),
                  "l": (x, y + H/2), "r": (x + W, y + H/2)}
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                     color="black", lw=1.1))
    # row1 L->R
    arrow(anc[1]["r"], anc[2]["l"]); arrow(anc[2]["r"], anc[3]["l"]); arrow(anc[3]["r"], anc[4]["l"])
    arrow(anc[4]["cb"], anc[5]["ct"])  # down Op4->Op5
    # row2 R->L
    arrow(anc[5]["l"], anc[6]["r"]); arrow(anc[6]["l"], anc[7]["r"]); arrow(anc[7]["l"], anc[8]["r"])
    arrow(anc[8]["cb"], anc[9]["ct"])  # down Op8->Op9
    # row3 L->R
    arrow(anc[9]["r"], anc[10]["l"]); arrow(anc[10]["r"], anc[11]["l"]); arrow(anc[11]["r"], anc[12]["l"])
    arrow(anc[12]["cb"], anc[13]["ct"])  # down Op12->Op13
    # leyendas
    ax.text(5, 6.7, "Cadena MFSC de 13 operaciones (Tesis, Fig. 6.4)", ha="center", fontsize=11, weight="bold")
    # lever annotations
    ax.text(5.7, 4.85, "↑ PALANCA DEL AGENTE: turnos S1/S2/S3 (capacidad variable)",
            ha="center", fontsize=8.5, color=GREEN, weight="bold")
    ax.text(4.3, 2.95, "✖ CUELLO DE BOTELLA: despacho FIJO ~2500 rac/día (ROP diario, Q=2400–2600)",
            ha="center", fontsize=8.5, color=RED, weight="bold")
    # inventory lever markers
    for k in (3, 5, 9):
        x0 = ops[k][1]; y0 = ops[k][2]
        ax.text(x0 + W/2, y0 + H + 0.08, "◆ inventario", ha="center", fontsize=6.5, color=BLUE)
    ax.text(0.5, 0.2, "Verde = el agente controla aquí (upstream).  Rojo = el límite está aquí (downstream).",
            fontsize=8, color=DARK)
    fig.tight_layout(); fig.savefig(FIG / "figb3_chain.png", dpi=150); plt.close(fig)


# --- Fig B4: por qué no se traduce en ReT (fill rate acotado) ---
def fig_ret_link():
    fig, ax = plt.subplots(figsize=(7.4, 2.6)); ax.axis("off")
    ax.text(0.02, 0.82, r"ReT incluye el fill rate (Tesis, Eq. 5.4 / Fig. 6.5):", fontsize=11)
    ax.text(0.06, 0.5, r"$Re(FR_t) = 1 - \dfrac{B_t + U_t}{D_t}$", fontsize=18)
    ax.text(0.5, 0.6, "$B_t,U_t$ (backorders, no atendidos) dependen de lo\nENTREGADO, no de lo producido.",
            fontsize=10, color=DARK)
    ax.text(0.5, 0.2, "Como la entrega está topada por Op9–12 (~2500/día),\n"
                      "producir más (S3) no baja $B_t$ → no mejora ReT.\n"
                      "→ S2 ya es casi óptima; el RL no tiene margen en Track A.",
            fontsize=10, color=RED)
    fig.tight_layout(); fig.savefig(FIG / "figb4_ret_link.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_prod_vs_deliver(); fig_accumulation(); fig_chain(); fig_ret_link()
    print("bottleneck figures ->", FIG)
