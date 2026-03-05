#!/usr/bin/env python3
"""
visualize.py — Phase 2 diagnostic plots + Step API validation.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from supply_chain.supply_chain import MFSCSimulation
from supply_chain.config import HOURS_PER_YEAR_GREGORIAN

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 1. Run stochastic simulation (2 years for clearer visualization)
# =====================================================================
HORIZON_2Y = HOURS_PER_YEAR_GREGORIAN * 2  # 2 years (gregorian basis)

sim = MFSCSimulation(
    shifts=1,
    risks_enabled=True,
    risk_level="current",
    seed=42,
    horizon=HORIZON_2Y,
    year_basis="gregorian",
).run()

# Extract time series
t_sb = [x[0] for x in sim.daily_inventory_sb]
inv_sb = [x[1] for x in sim.daily_inventory_sb]
t_th = [x[0] for x in sim.daily_inventory_theatre]
inv_th = [x[1] for x in sim.daily_inventory_theatre]
t_prod = [x[0] for x in sim.daily_production]
prod = [x[1] for x in sim.daily_production]
t_dem = [x[0] for x in sim.daily_demand]
dem = [x[1] for x in sim.daily_demand]

# Convert hours to days for readability
t_sb_d = [t / 24 for t in t_sb]
t_th_d = [t / 24 for t in t_th]
t_prod_d = [t / 24 for t in t_prod]
t_dem_d = [t / 24 for t in t_dem]

# =====================================================================
# FIGURE 1: Inventory + Risk Events
# =====================================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle(
    "MFSC Stochastic Simulation — 2 Years (S=1, Current Risks)",
    fontsize=14,
    fontweight="bold",
)

# Panel 1: Theatre inventory
ax1 = axes[0]
ax1.plot(t_th_d, inv_th, color="#2196F3", linewidth=0.7, alpha=0.8)
ax1.fill_between(t_th_d, 0, inv_th, alpha=0.15, color="#2196F3")
ax1.set_ylabel("Rations")
ax1.set_title("Theatre Inventory (Op13)")
ax1.axhline(y=2500, color="red", linestyle="--", alpha=0.5, label="Daily demand avg")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2: Supply Battalion inventory
ax2 = axes[1]
ax2.plot(t_sb_d, inv_sb, color="#4CAF50", linewidth=0.7, alpha=0.8)
ax2.fill_between(t_sb_d, 0, inv_sb, alpha=0.15, color="#4CAF50")
ax2.set_ylabel("Rations")
ax2.set_title("Supply Battalion Inventory (Op9)")
ax2.grid(True, alpha=0.3)

# Panel 3: Daily production
ax3 = axes[2]
ax3.bar(t_prod_d, prod, width=0.8, color="#FF9800", alpha=0.7)
ax3.set_ylabel("Rations/Day")
ax3.set_title("Daily Production (Assembly Line, Hourly Resolution)")
ax3.axhline(y=2564, color="gray", linestyle="--", alpha=0.5, label="Max S=1 (2,564)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Risk events timeline
ax4 = axes[3]
risk_colors = {
    "R11": "#E91E63",
    "R12": "#9C27B0",
    "R13": "#673AB7",
    "R14": "#BDBDBD",
    "R21": "#F44336",
    "R22": "#FF5722",
    "R23": "#FF9800",
    "R24": "#FFC107",
    "R3": "#000000",
}
y_positions = {
    rid: i
    for i, rid in enumerate(
        ["R11", "R12", "R13", "R14", "R21", "R22", "R23", "R24", "R3"]
    )
}

for event in sim.risk_events:
    y = y_positions.get(event.risk_id, 0)
    start_d = event.start_time / 24
    dur_d = max(0.5, event.duration / 24)  # Min width for visibility
    color = risk_colors.get(event.risk_id, "gray")
    ax4.barh(y, dur_d, left=start_d, height=0.6, color=color, alpha=0.6)

ax4.set_yticks(list(y_positions.values()))
ax4.set_yticklabels(list(y_positions.keys()), fontsize=9)
ax4.set_xlabel("Day")
ax4.set_title("Risk Event Timeline")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "mfsc_phase2_diagnostic.png"), dpi=150, bbox_inches="tight"
)
print("Figure 1 saved: mfsc_phase2_diagnostic.png")

# =====================================================================
# FIGURE 2: Hourly vs Daily granularity comparison (R11 impact)
# =====================================================================
fig2, (ax_h, ax_d) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig2.suptitle(
    "R11 Impact: Hourly vs Daily Granularity (30-day window)",
    fontsize=13,
    fontweight="bold",
)

# Zoom into first 30 days
mask_prod = [i for i, t in enumerate(t_prod_d) if t <= 30]
zoom_t = [t_prod_d[i] for i in mask_prod]
zoom_p = [prod[i] for i in mask_prod]

# R11 events in first 30 days
r11_events = [
    e for e in sim.risk_events if e.risk_id == "R11" and e.start_time <= 30 * 24
]

ax_h.bar(zoom_t, zoom_p, width=0.8, color="#2196F3", alpha=0.7, label="Hourly model")
ax_h.axhline(y=2564, color="gray", linestyle="--", alpha=0.5)
ax_h.set_ylabel("Rations/Day")
ax_h.set_title("Hourly Assembly (captures partial-day losses)")

# Mark R11 events
for e in r11_events:
    ax_h.axvspan(e.start_time / 24, e.end_time / 24, alpha=0.3, color="red")
ax_h.legend(fontsize=8)
ax_h.grid(True, alpha=0.3)

# Simulated "daily check" version for comparison
# If checked daily: production is 2564 if assembly NOT down at check time, else 0
daily_check_prod = []
for i in mask_prod:
    t_check = t_prod[i]  # Daily check time
    # Was assembly down at this exact moment?
    down_at_check = False
    for e in sim.risk_events:
        if e.risk_id in ("R11", "R21", "R3") and e.start_time <= t_check <= e.end_time:
            if any(op in e.affected_ops for op in [5, 6, 7]):
                down_at_check = True
                break
    daily_check_prod.append(0 if down_at_check else 2564)

ax_d.bar(
    zoom_t,
    daily_check_prod,
    width=0.8,
    color="#FF5722",
    alpha=0.7,
    label="Daily check model",
)
ax_d.axhline(y=2564, color="gray", linestyle="--", alpha=0.5)
ax_d.set_ylabel("Rations/Day")
ax_d.set_xlabel("Day")
ax_d.set_title("Daily Check (binary: full day lost or full day produced)")
for e in r11_events:
    ax_d.axvspan(e.start_time / 24, e.end_time / 24, alpha=0.3, color="red")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "mfsc_hourly_vs_daily.png"), dpi=150, bbox_inches="tight"
)
print("Figure 2 saved: mfsc_hourly_vs_daily.png")

# =====================================================================
# 3. STEP API VALIDATION
# =====================================================================
print("\n" + "=" * 60)
print("  STEP API TEST")
print("=" * 60)

step_sim = MFSCSimulation(
    shifts=1,
    risks_enabled=True,
    seed=99,
    horizon=HOURS_PER_YEAR_GREGORIAN,
    year_basis="gregorian",
)
obs_history = []
reward_history = []

for i in range(365):  # 365 daily steps = 1 year
    obs, reward, done, info = step_sim.step(step_hours=24)
    obs_history.append(obs)
    reward_history.append(reward)
    if done:
        print(f"  Episode ended at step {i+1}")
        break

obs_arr = np.array(obs_history)
print(f"  Steps completed: {len(obs_history)}")
print(f"  Final time: {step_sim.env.now:,.0f} hrs")
print(f"  Total reward: {sum(reward_history):,.0f}")
print(f"  Observation shape: {obs_arr.shape}")
print("  Observation ranges:")
labels = [
    "rm_wdc",
    "rm_al",
    "rat_al",
    "rat_sb",
    "rat_cssu",
    "rat_th",
    "fill_rate",
    "bo_rate",
    "al_down",
    "loc_down",
    "op9_down",
    "op11_down",
    "time_frac",
    "pend_batch",
    "cont_dem",
]
for j, label in enumerate(labels):
    print(f"    {label:>12}: [{obs_arr[:,j].min():.3f}, {obs_arr[:,j].max():.3f}]")

# Test action application
step_sim2 = MFSCSimulation(
    shifts=1,
    risks_enabled=True,
    seed=99,
    horizon=168 * 4,
    year_basis="gregorian",
)
# Run 1 week with default params
obs1, _, _, _ = step_sim2.step(step_hours=168)
print(f"\n  Before action: op9_rop = {step_sim2.params['op9_rop']}")
# RL agent changes Op9 dispatch frequency from 24h to 12h
obs2, _, _, _ = step_sim2.step(action={"op9_rop": 12}, step_hours=168)
print(f"  After action:  op9_rop = {step_sim2.params['op9_rop']}")
print("  ✅ Step API working — params mutable at runtime")
print("=" * 60)
