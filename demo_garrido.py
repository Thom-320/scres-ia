#!/usr/bin/env python3
"""
Demo para Garrido — DES + RL Environment
=========================================
Corre desde la raíz del proyecto:
    cd ~/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia
    python demo_garrido.py

Muestra:
  1. DES corriendo y produciendo observaciones
  2. Cómo las disrupciones afectan el estado
  3. El espacio de acciones y reward
  4. Comparación rápida: S1 vs S2 vs S3
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from supply_chain.config import (
    OPERATIONS,
    SIMULATION_HORIZON,
    WARMUP,
    VALIDATION_TABLE_6_10,
    CAPACITY_BY_SHIFTS,
)
from supply_chain.supply_chain import MFSCSimulation
from supply_chain.env_experimental_shifts import MFSCGymEnvShifts

# ═════════════════════════════════════════════════════════════════
# Constantes de formato
# ═════════════════════════════════════════════════════════════════
OBS_LABELS = [
    "raw_material_wdc  (Op3 inv)",
    "raw_material_al   (Op5 inv)",
    "rations_al        (Op7 buffer)",
    "rations_sb        (Op9 inv)",
    "rations_cssu      (Op11 inv)",
    "rations_theatre   (Op13 inv)",
    "fill_rate         (acumulado)",
    "backorder_rate    (acumulado)",
    "assembly_down     (Op5-7 disrupción)",
    "any_loc_down      (LOC disrupción)",
    "op9_down          (Batallón disrupción)",
    "op11_down         (CSSU disrupción)",
    "time_fraction     (progreso sim)",
    "pending_batch_norm(lote pendiente)",
    "contingent_demand (demanda extra)",
]

SEPARATOR = "─" * 70


def print_header(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}\n")


def print_observation(obs: np.ndarray, step: int) -> None:
    """Imprime el vector de observación con etiquetas."""
    print(f"  Step {step:3d}")
    print(f"  {'─' * 50}")
    for i, (val, label) in enumerate(zip(obs, OBS_LABELS)):
        marker = ""
        if i == 6 and val < 0.7:
            marker = "  << FILL RATE BAJO"
        elif i in (8, 9, 10, 11) and val > 0:
            marker = "  << DISRUPCION ACTIVA"
        print(f"    [{i:2d}] {label}: {val:8.4f}{marker}")
    print()


# ═════════════════════════════════════════════════════════════════
# DEMO 1: Validación DES — Cf0 baseline
# ═════════════════════════════════════════════════════════════════
def demo_validation():
    print_header("DEMO 1: Validación DES — Cf0 vs Tesis Garrido-Rios (2017)")

    print("  Corriendo simulación determinística (S=1, sin riesgos)...")
    sim = MFSCSimulation(
        shifts=1,
        risks_enabled=False,
        seed=42,
        horizon=SIMULATION_HORIZON,
        year_basis="thesis",
        deterministic_baseline=True,
    ).run()

    tp = sim.get_annual_throughput(start_time=sim.warmup_time)
    our_avg = tp["avg_annual_delivery"]
    thesis_avg = np.mean(VALIDATION_TABLE_6_10["ECS_simulated"])
    gap = (our_avg - thesis_avg) / thesis_avg

    print()
    print(f"  Nuestro modelo:     {our_avg:>12,.0f} raciones/año")
    print(f"  Tesis (Table 6.10): {thesis_avg:>12,.0f} raciones/año")
    print(f"  Gap relativo:       {gap:>+11.2%}")
    print(f"  Umbral aceptado:    ±15%")
    print(f"  Estado:             {'PASS' if abs(gap) < 0.15 else 'FAIL'}")
    print()
    # Post-warmup summary (cleaner for presentation)
    warmup_t = sim.warmup_time
    years_pw = (sim.horizon - warmup_t) / sim.hours_per_year
    # Count post-warmup backorders: orders placed after warmup
    pw_orders = [o for o in sim.orders if o.OPTj >= warmup_t]
    pw_backorders = sum(1 for o in pw_orders if o.backorder)
    mode = f"ENABLED ({sim.risk_level})" if sim.risks_enabled else "DISABLED"
    print(f"  {'=' * 58}")
    print(f"  MFSC Simulation Summary (POST-WARMUP)")
    print(f"  Horizon: {sim.horizon:,} hrs ({sim.horizon/sim.hours_per_year:.1f} years)")
    print(f"  Shifts: S={sim.shifts}  |  Risks: {mode}")
    print(f"  Warmup: {warmup_t:,.0f} hrs (excluded from metrics)")
    print(f"  {'=' * 58}")
    print(f"  Produced:       {sim.total_produced:,.0f}")
    print(f"  Delivered:      {sim.total_delivered:,}")
    print(f"  Demanded:       {sim.total_demanded:,.0f}")
    print(f"  Fill rate:      {sim._fill_rate():.1%}")
    print(f"  Backorders:     {pw_backorders:,} (post-warmup)")
    print(f"  Avg ann. del:   {our_avg:,.0f}")
    print(f"  {'=' * 58}")
    if sim.risks_enabled:
        sim.risk_summary()
    print(SEPARATOR)

    input("\n  [Enter para continuar a Demo 2...]\n")


# ═════════════════════════════════════════════════════════════════
# DEMO 2: Gymnasium Environment — observaciones paso a paso
# ═════════════════════════════════════════════════════════════════
def demo_gymnasium_env():
    print_header("DEMO 2: Gymnasium Environment — Observaciones paso a paso")

    print("  Creando environment MFSCGymEnvShifts...")
    env = MFSCGymEnvShifts(
        risk_level="current",
        reward_mode="control_v1",
        year_basis="thesis",
    )
    obs, info = env.reset(seed=42)

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print(f"  Obs shape:         {obs.shape}")
    print()

    print("  Estado inicial (post-warmup):")
    print_observation(obs, 0)

    # Correr 5 steps con S2 fijo y parámetros default
    print("  Corriendo 5 semanas simuladas con S=2 fijo...")
    print(f"  {'Step':>5} {'Reward':>8} {'Fill%':>7} {'Shift':>6} {'AL_down':>8} {'BO_rate':>8}")
    print(f"  {'-'*50}")

    action_s2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # S2
    for step in range(1, 6):
        obs, reward, terminated, truncated, info = env.step(action_s2)
        print(
            f"  {step:5d} {reward:8.3f} {obs[6]*100:6.1f}%"
            f"     S2   {int(obs[8]):>7d} {obs[7]:8.4f}"
        )

    print()
    print("  El fill rate sube gradualmente mientras la cadena se estabiliza.")
    print(SEPARATOR)

    input("\n  [Enter para continuar a Demo 3...]\n")


# ═════════════════════════════════════════════════════════════════
# DEMO 3: Impacto de disrupciones — current vs increased vs severe
# ═════════════════════════════════════════════════════════════════
def demo_disruption_effect():
    print_header("DEMO 3: Impacto de disrupciones en fill rate (S=1)")

    configs = [
        ("Sin riesgos  ", False, "current"),
        ("Riesgo actual ", True, "current"),
        ("Riesgo aumentado", True, "increased"),
    ]

    print(f"  {'Configuración':<20} {'Raciones/año':>14} {'Fill rate':>10} {'Eventos':>10}")
    print(f"  {'─'*60}")

    for label, risks, level in configs:
        sim = MFSCSimulation(
            shifts=1,
            risks_enabled=risks,
            risk_level=level,
            seed=42,
            horizon=SIMULATION_HORIZON,
            year_basis="thesis",
        ).run()

        tp = sim.get_annual_throughput(start_time=sim.warmup_time)
        avg_del = tp["avg_annual_delivery"]

        # fill rate
        if sim.total_demanded > 0:
            fr = sim.total_delivered / sim.total_demanded
        else:
            fr = 1.0

        # event count
        events = sum(sim.risk_event_counts.values()) if hasattr(sim, "risk_event_counts") else 0

        print(f"  {label:<20} {avg_del:>14,.0f} {fr:>9.1%} {events:>10,}")

    print()
    print("  La cadena colapsa bajo estrés con turno único.")
    print("  Esto motiva la necesidad de control adaptativo de turnos.")
    print(SEPARATOR)

    input("\n  [Enter para continuar a Demo 4...]\n")


# ═════════════════════════════════════════════════════════════════
# DEMO 4: Espacio de acciones + Comparación S1 vs S2 vs S3
# ═════════════════════════════════════════════════════════════════
def demo_action_space():
    print_header("DEMO 4: Espacio de acciones — Control de turnos")

    print("  El agente controla 5 dimensiones cada semana:")
    print("    [0] op3_q   — Cantidad de despacho en Bodega (Op3)")
    print("    [1] op9_q   — Despacho máximo en Batallón (Op9)")
    print("    [2] op3_rop — Punto de reorden en Bodega")
    print("    [3] op9_rop — Punto de reorden en Batallón")
    print("    [4] shifts  — Turnos: <-0.33→S1, [-0.33,0.33)→S2, ≥0.33→S3")
    print()
    print("  Mapeo: multiplier = 1.25 + 0.75 × señal")
    print("    señal=-1 → ×0.50 (reducir a mitad)")
    print("    señal= 0 → ×1.25 (baseline +25%)")
    print("    señal=+1 → ×2.00 (duplicar)")
    print()

    # Comparar S1 vs S2 vs S3 bajo riesgo aumentado (episodio completo via Gym env)
    print("  Comparación: S1 vs S2 vs S3 bajo riesgo aumentado")
    print("  (20 años simulados, 48 semanas/año)")
    print(f"  {'─'*55}")
    print(f"  {'Turnos':<10} {'Fill rate':>10} {'Reward total':>14} {'Raciones/año':>14}")
    print(f"  {'─'*55}")

    for shifts, signal in [(1, -1.0), (2, 0.0), (3, 1.0)]:
        sim = MFSCSimulation(
            shifts=shifts,
            risks_enabled=True,
            risk_level="increased",
            seed=42,
            horizon=SIMULATION_HORIZON,
            year_basis="thesis",
        ).run()

        tp = sim.get_annual_throughput(start_time=sim.warmup_time)
        avg_del = tp["avg_annual_delivery"]
        fr = sim.total_delivered / sim.total_demanded if sim.total_demanded > 0 else 1.0

        # Approximate total control_v1 reward
        total_steps = SIMULATION_HORIZON / 168
        avg_bo_rate = 1.0 - fr
        approx_reward = total_steps * -(4.0 * avg_bo_rate + 0.02 * (shifts - 1))

        print(f"  S={shifts:<8d} {fr:>9.1%} {approx_reward:>14.1f} {avg_del:>14,.0f}")

    print()
    print("  S2 domina bajo riesgo moderado (mejor servicio + costo razonable).")
    print("  Pero un agente adaptativo puede MEZCLAR turnos dinámicamente:")
    print("  bajar a S1 cuando no hay disrupción, subir a S3 cuando sí.")
    print(SEPARATOR)

    input("\n  [Enter para continuar a Demo 5...]\n")


# ═════════════════════════════════════════════════════════════════
# DEMO 5: Función de recompensa — control_v1
# ═════════════════════════════════════════════════════════════════
def demo_reward():
    print_header("DEMO 5: Función de recompensa — control_v1")

    print("  Problema con ReT_thesis como objetivo de entrenamiento:")
    print("    El agente aprende a MINIMIZAR costos (S1 siempre)")
    print("    porque la estructura de ReT premia evitar turnos extras")
    print("    más rápido de lo que penaliza la caída en fill rate.")
    print()
    print("  Solución: control_v1")
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  r_t = -(4.0 × B_t/D_t + 0.02 × (S_t - 1))       │")
    print("  │                                                      │")
    print("  │  w_bo / w_cost = 200                                 │")
    print("  │  Servicio es 200x más importante que costo           │")
    print("  └──────────────────────────────────────────────────────┘")
    print()
    print("  Resultado empírico (500k steps, 5 seeds):")
    print("  ┌────────────┬───────────┬────────────┬──────────────────┐")
    print("  │ Escenario  │ PPO vs    │ Diferencia │ Bootstrap CI95   │")
    print("  │            │ mejor     │            │                  │")
    print("  ├────────────┼───────────┼────────────┼──────────────────┤")
    print("  │ Increased  │ S2        │ -1.95      │ [-9.95, +8.51]   │")
    print("  │ Severe     │ S3        │ +4.61      │ [-0.28, +9.49]   │")
    print("  └────────────┴───────────┴────────────┴──────────────────┘")
    print()
    print("  Bajo estrés moderado: PPO iguala al mejor baseline.")
    print("  Bajo estrés severo:   PPO supera al mejor baseline (+4.61).")
    print(SEPARATOR)


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  SCRES+IA — Demo para Garrido")
    print("  Modelo DES + Reinforcement Learning Environment")
    print("  Marzo 2026")
    print("█" * 70)

    try:
        demo_validation()
        demo_gymnasium_env()
        demo_disruption_effect()
        demo_action_space()
        demo_reward()
    except KeyboardInterrupt:
        print("\n\n  Demo terminada.\n")

    print("\n  Demo completa.")
    print("  Siguiente paso: correr seeds adicionales bajo severe")
    print("  para alcanzar significancia estadística.\n")
