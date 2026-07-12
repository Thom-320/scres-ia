"""Figures for the Garrido Track B alternatives document (2026-07-03) -- short version, resilience-only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "/Users/thom/Projects/research/scres-ia/docs/garrido_track_b_alternatives_2026-07-03/figures"

# ---------------------------------------------------------------------------
# Figure 1: 13-operation topology, decision-variable authority under Track B 8D
# ---------------------------------------------------------------------------
ops = [
    (1, "Agencia\nLogística"), (2, "Proveedores"), (3, "CDC"), (4, "LOC\nWDC→AL"),
    (5, "Ensamble\nPre"), (6, "Ensamble"), (7, "QC &\nEnvío"), (8, "LOC\nAL→SB"),
    (9, "Batallón\nSupply"), (10, "LOC\nSB→CSSU"), (11, "CSSU"),
    (12, "LOC\nCSSU→Teatro"), (13, "Teatro de\nOperaciones"),
]
decision_ops = {3: "thesis", 5: "thesis", 9: "thesis", 10: "new", 12: "new"}

fig, ax = plt.subplots(figsize=(14, 3.6))
n = len(ops)
xs = np.linspace(0.6, n - 0.4, n)
y = 1.0
box_w, box_h = 0.72, 0.8
color_none, color_thesis, color_new = "#e6e6e6", "#8fd3a0", "#4a90d9"

for (op_id, name), x in zip(ops, xs):
    color = color_thesis if decision_ops.get(op_id) == "thesis" else color_new if decision_ops.get(op_id) == "new" else color_none
    box = FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.2, edgecolor="#333333", facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, f"Op{op_id}\n{name}", ha="center", va="center", fontsize=7.3, zorder=4)

for i in range(n - 1):
    ax.add_patch(FancyArrowPatch((xs[i] + box_w / 2, y), (xs[i + 1] - box_w / 2, y),
                                  arrowstyle="-|>", mutation_scale=10, color="#555555", zorder=2))

legend_handles = [
    mpatches.Patch(facecolor=color_thesis, edgecolor="#333333", label="Variable heredada o alineada con tesis"),
    mpatches.Patch(facecolor=color_new, edgecolor="#333333", label="Extensión Track B aguas abajo"),
    mpatches.Patch(facecolor=color_none, edgecolor="#333333", label="Sin variable de decisión"),
]
ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8.5, frameon=False)
ax.set_xlim(0, n + 0.4)
ax.set_ylim(0.3, 1.7)
ax.axis("off")
ax.set_title("Contrato Track B 8D completo: CDC incluido, Op1/Op2 excluidos", fontsize=10.5, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_topology.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: resilience-only comparison, 8D complete vs best static
# ---------------------------------------------------------------------------
labels = ["Mejor\nestática", "PPO+MLP\n8D completo", "Real-KAN\n8D completo"]
improvement = [0.0, 8.0, 9.4]
ci_halfwidth = [0.0, 0.55, 0.3]
colors = ["#a6a6a6", "#8fd3a0", "#4a90d9"]

fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(labels, improvement, yerr=ci_halfwidth, capsize=5, color=colors, edgecolor="#333333")
ax.set_ylabel("% de mejora en resiliencia (ReT Excel)\nvs. nuestra mejor política estática")
ax.set_title("Resiliencia: contrato Track B 8D completo vs. grilla estática", fontsize=11)
ax.set_ylim(0.0, 10.5)
for b, v, ci in zip(bars, improvement, ci_halfwidth):
    label = "0%" if v == 0.0 else f"+{v:.1f}%"
    ax.text(b.get_x() + b.get_width() / 2, v + ci + 0.35, label, ha="center", fontsize=9.5, fontweight="bold")
ax.tick_params(axis="x", labelsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_results.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_results_preview.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: buffer replenishment mechanism, old (flawed) vs new (conservation)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 5.2))

def draw_chain(ax, steps, colors_list, title):
    n = len(steps)
    box_w = 1.55
    gap = 0.35
    xs = [0.9 + i * (box_w + gap) for i in range(n)]
    for i, (txt, x, c) in enumerate(zip(steps, xs, colors_list)):
        box = FancyBboxPatch((x - box_w / 2, 0.3), box_w, 0.9, boxstyle="round,pad=0.03,rounding_size=0.08",
                              linewidth=1.2, edgecolor="#333333", facecolor=c, zorder=3)
        ax.add_patch(box)
        ax.text(x, 0.75, txt, ha="center", va="center", fontsize=8.0, wrap=True, zorder=4)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + box_w / 2, 0.75), (xs[i + 1] - box_w / 2, 0.75),
                                          arrowstyle="-|>", mutation_scale=12, color="#555555", zorder=2))
    ax.set_xlim(0, xs[-1] + box_w / 2 + 0.3)
    ax.set_ylim(0, 1.3)
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")

old_steps = ["Acción: target\nde buffer", "container.put(shortfall)\n(SimPy)", "Inventario aparece\nde inmediato",
             "NO se descuenta\nde ningún origen", "NO respeta\ncapacidad/lead time"]
old_colors = ["#dbe8f7"] * 2 + ["#f4b7b7"] * 3
draw_chain(axes[0], old_steps, old_colors, "Mecanismo ANTIGUO (el problema que señaló Garrido)")

new_steps = ["Acción continua:\ntarget / multiplicador", "Evento DES:\nreposición programada",
             "Restricción física:\nqty = min(shortfall,\ndisponible, capacidad)", "Lead time:\nllega tras transporte",
             "Costo de mantener\n/mover se contabiliza"]
new_colors = ["#dbe8f7", "#dbe8f7", "#c9e6c9", "#c9e6c9", "#c9e6c9"]
draw_chain(axes[1], new_steps, new_colors, "Mecanismo CORREGIDO (conserva el flujo físico real)")

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_buffer_mechanism.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_buffer_mechanism_preview.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("Figures written to", OUT)
