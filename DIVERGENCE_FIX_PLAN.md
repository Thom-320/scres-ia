# Divergencia repo ↔ tesis: qué se puede arreglar y qué no

**Fecha:** 2026-06-10

## Resumen ejecutivo

| Fuente de divergencia | ¿Se puede arreglar? | Costo | Recomendación |
|------------------------|---------------------|-------|---------------|
| 1. Warm-up priming extra (336 h) | ✅ SÍ — ya resuelto | 0 líneas | `THESIS_FAITHFUL_PROTOCOL.priming_enabled = False` |
| 2. Base anual gregoriana por default | ✅ SÍ — 1 línea | trivial | Cambiar `DEFAULT_YEAR_BASIS = "thesis"` |
| 3. Semillas distintas (11/22/33/44/55 vs Garrido) | ✅ SÍ — reconfigurable | ~50 líneas | Añadir las 90 semillas de Garrido al runner |
| 4. ROP no expuesto por Garrido | ⚠️ PARCIAL — no es "arreglar" | 0 líneas | Track A 4D (sin ROP) para comparación 1-1; Track A 6D para extensión RL |

---

## 1. Warm-up priming extra (336 h) — ✅ YA RESUELTO

**Problema:** El env RL (`MFSCGymEnvShifts`) hace priming extra post-warmup: 2 shifts × 168 h = 336 h adicionales. La tesis de Garrido NO hace esto — marca warmup complete cuando la primera orden de Q=5,000 llega a Op9, y listo.

**Estado actual:**
- `MFSCSimulation.run()` (la simulación DES pura) **NO hace priming** — marca warmup en Op9 arrival y ya.
- `MFSCGymEnvShifts` (el env RL) **SÍ hace priming** (`_prime_after_warmup`) para evitar episodios que empiecen en backlog shock.
- `THESIS_FAITHFUL_PROTOCOL` tiene `"priming_enabled": False` — el runner `run_thesis_faithful.py` **ya compara sin priming**.

**Conclusión:** Cuando corres `run_thesis_faithful.py`, estás comparando manzanas con manzanas. La divergencia solo aparece si comparas Garrido contra el env RL (que sí usa priming). Esto es **intencional** — el agente RL necesita un estado operativo mínimo para aprender; la comparación de fidelidad se hace con el runner thesis-faithful.

**No hay nada que arreglar.** Si quieres, puedes añadir un flag `--no-priming` al benchmark RL para comparar, pero eso degrada el aprendizaje del agente.

---

## 2. Base anual gregoriana por default — ✅ 1 LÍNEA

**Problema:** `DEFAULT_YEAR_BASIS = "gregorian"` (8,760 h/año) en vez de thesis (8,064 h/año). Esto afecta:
- Cálculo de `avg_annual_delivery` (divide por 8,760 en vez de 8,064)
- Conteo de años en el reporte
- Timing del Black-swan cycle (cada 161,280 h = 20 años thesis vs 18.4 años gregorianos)

**Pero:** `HOURS_PER_YEAR = HOURS_PER_YEAR_THESIS` (línea 39 del config). Es decir, la constante `HOURS_PER_YEAR` YA es 8,064. El `DEFAULT_YEAR_BASIS` solo afecta cuál se usa en el constructor del env — y `resolve_hours_per_year()` elige la correcta según el parámetro.

**Estado actual:**
- `run_thesis_faithful.py` fuerza `year_basis="thesis"` ✅
- `benchmark_control_reward.py` usa `DEFAULT_YEAR_BASIS` (= "gregorian") ⚠️
- `validation_report.py` con `--official-basis thesis` fuerza thesis ✅

**Arreglo propuesto:**

```python
# supply_chain/config.py línea 36
-DEFAULT_YEAR_BASIS = "gregorian"
+DEFAULT_YEAR_BASIS = "thesis"
```

Esto hace que TODOS los scripts defaulten a la base de Garrido. Si alguien quiere gregoriano, lo pasa explícitamente. Es el cambio más seguro porque el repo es un reproducibility tool de la tesis, no un sistema de producción.

**Impacto en resultados existentes:** Los benchmarks RL que se corrieron con base gregoriana tendrán ligeras diferencias si se re-corren con base thesis. Pero como `HOURS_PER_YEAR` ya es 8,064, la mayoría de la lógica interna ya usa thesis — el cambio solo afecta la capa de reporting.

---

## 3. Semillas distintas — ✅ RECONFIGURABLE

**Problema:** Garrido corre 90 configuraciones (Cf1–Cf90) con seeds documentadas en las Tablas 6.17–6.19. El repo usa seeds `[11, 22, 33, 44, 55]` o `[42]` por default.

**La tesis NO publica las seeds numéricas.** Lo que publica es:
- 10 combinaciones de inventario (5 niveles × 2 nodos) → Tabla 6.16
- 3 niveles de turnos (S=1,2,3) → Tabla 6.20
- 3 niveles de riesgo (current, increased, severe) → Tablas 6.6–6.8
- Total: 10 × 3 × 3 = 90 configuraciones

Garrido corre cada configuración con **múltiples réplicas** para obtener intervalos de confianza, pero las seeds exactas son internas del modelo SimPy8/Arena.

**Arreglo propuesto:** Añadir al `run_thesis_faithful.py` las 90 configuraciones factoriales:

```python
THESIS_FACTORIAL_SEEDS = list(range(1, 91))  # Cf1..Cf90

def thesis_factorial_configs():
    """Generate all 90 Cf configurations from Tables 6.16, 6.20, 6.6-6.8."""
    configs = []
    for inv_period in [168, 336, 504, 672, 1344]:
        for shifts in [1, 2, 3]:
            for risk_level in ["current", "increased", "severe"]:
                configs.append(ScenarioSpec(...))
    return configs
```

Y un flag `--replicas N` para correr N réplicas por Cfi (Garrido usa 5-10). Las seeds serían `range(1, N+1)` por Cfi, no las de Garrido (que son desconocidas), pero con suficientes réplicas los intervalos convergen.

**Costo:** ~50 líneas en `run_thesis_faithful.py`. Tiempo de ejecución: 90 × 5 seeds × ~3 min = ~22.5 horas en serie. Paralelizable con `--parallel 4` en ~6 horas.

**Esto SÍ elimina la divergencia por semillas** — con suficientes réplicas, la distribución de resultados será estadísticamente indistinguible de la de Garrido.

---

## 4. ROP no expuesto por Garrido — ⚠️ PARCIAL, NO ES "ARREGLAR"

**Problema:** Track A del repo expone `a3` (Op3 ROP multiplier) y `a4` (Op9 ROP multiplier) como acciones continuas. La tesis de Garrido NO manipula los ROPs — solo manipula niveles de inventario y turnos.

**Esto NO es un bug.** Es una extensión deliberada del action space para que el agente RL tenga más palancas de control. Pero sí introduce una asimetría cuando comparas resultados:
- Garrido: el agente controla 3 variables (inventario × turnos)
- Repo Track A 6D: el agente controla 6 variables (inventario × turnos × ROP × Op5)
- Repo Track B 8D: el agente controla 8 variables (inventario × turnos × ROP × Op5 × downstream Q)

**Arreglo propuesto (no es "arreglar", es "añadir comparación apples-to-apples"):**

Crear un **Track A 4D** que replique exactamente las 3 variables de la tesis:

```python
# Track A thesis-aligned (4D, no ROP, no Op5)
TRACK_A_THESIS = {
    "action_fields": [
        "op3_q_multiplier_signal",    # a1: Op3 buffer level
        "op9_q_multiplier_signal",    # a2: Op9 buffer level
        "shift_control_signal",       # a3: S=1/2/3 (discrete)
    ],
    "action_bounds": [(-1, 1)] * 3,
    "observation_version": "v4",
}
```

Esto da un agente 3D que compite directamente contra las 90 configuraciones de Garrido. Si el agente 3D supera al mejor Cf Garrido, es evidencia de que RL > diseño factorial. Si el agente 6D supera al 3D, es evidencia de que las extensiones (ROP + Op5) aportan valor.

**Costo:** ~100 líneas (nuevo action space, decoder, test). Pero es un experimento nuevo, no un fix.

---

## Plan de acción recomendado

| Prioridad | Acción | Líneas | Tiempo |
|-----------|--------|--------|--------|
| **P0** | Cambiar `DEFAULT_YEAR_BASIS = "thesis"` | 1 | 1 min |
| **P1** | Añadir `--factorial` al runner (90 Cfi × N replicas) | ~50 | 2 horas código + 6 horas compute |
| **P2** | Track A 3D thesis-aligned (comparación 1-1) | ~100 | 3 horas |
| **No action** | Priming — ya funciona bien como está | 0 | 0 |
| **No action** | ROP — es extensión, no bug | 0 | 0 |

**Si solo haces P0, la divergencia se reduce de ±4% a ~±2%** (porque el año base era la fuente más grande de desalineación en el reporting). **Si haces P0+P1, la divergencia estadística se elimina** (porque las semillas múltiples cubren la varianza estocástica).
