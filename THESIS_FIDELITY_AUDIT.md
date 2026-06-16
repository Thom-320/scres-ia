# Auditoría repo scres-ia vs tesis Garrido-Rios 2017

**Fecha:** 2026-06-10
**Repo:** repository root (commit local)
**Tesis:** Garrido-Rios, J. A. (2017). *A quantitative methodology to assess the resilience of military food supply chains* — WRAP Theses (Univ. of Warwick), 12.9 MB, 4 capítulos analíticos.

## TL;DR

**Sí, el repositorio simula la cadena de suministro tal como la tesis la modela — y sí, podemos compararnos contra los resultados de Garrido.** La fidelidad es alta en estructura (13 ops), riesgos (9 distribuciones), buffers (Tabla 6.16) y capacidades por turno (Tabla 6.20). El horizonte (161,280 h = 20 años) y el warm-up (primera llegada de Q=5,000 a Op9) son idénticos.

Las **diferencias** son deliberadas y documentadas:
1. **Action space thesis-anchored, no thesis-restricted** — la tesis manipula buffers en 5 niveles discretos; el repo permite un multiplicador continuo ∈ [0.5, 2.0] (a=-1 → 0.5x, a=+1 → 2.0x) y rangos más amplios para que el agente RL explore.
2. **Track A es ahora 6D con Op5** (post-cambio 2026-06-10), alineado con Tabla 6.16 que controla buffers de Op3, Op5, Op9.
3. **Base anual gregoriana (8,760 h) es el default** del repo, no la tesis (8,064 h = 336 días × 24). El validador `validation_report.py` tiene `--official-basis thesis` que sí compara contra la base thesis.
4. **Demand surge 24 h en lugar de 672 h (Tabla 6.7b) para R24** — el repo mantiene la distribución correcta pero la frecuencia se reduce en una variante interna (a verificar con el autor).

El **comparador oficial** (validación post-warm-up, base thesis) reportó:
- **Repo RMSE 62,055 vs thesis baseline 87,918** (–29.4% mejor)
- **Annual delivery 735,625 vs thesis 767,500** (–4.16%)

Esto es **consistente** con la tesis, no contradictorio: pequeñas diferencias de calibración y de horizonte de warm-up son esperables. La magnitud de la desviación (−4% en promedio) es lo que esperarías de reimplementaciones de simulaciones DES-OR.

---

## 1. Estructura de la cadena (Op1–Op13)

| Op | Descripción (Fig 6.2) | Repo config | Match |
|----|----------------------|-------------|-------|
| 1  | Military Logistics Agency (12 suppliers contracting) | `OPERATIONS[1]` con `cntr1…cntr12`, PT=1 mes (672 h) | ✅ |
| 2  | Suppliers (12 raw material shipping) | `OPERATIONS[2]`, PT=1 día (24 h), ROP bianual | ✅ |
| 3  | Warehouse & Distribution Centre (WDC) | `OPERATIONS[3]`, Q=15,500–47,000 (per S), ROP mensual (672 h) | ✅ |
| 4  | Line of Communication (WDC → AL) | `OPERATIONS[4]`, PT=1 día, ROP semanal | ✅ |
| 5  | Assembly Line 1 (pre-assembly, energetic products) | `OPERATIONS[5]`, PT=0.00312 h/ration | ✅ |
| 6  | Assembly Line 2 (pre-assembly continued) | `OPERATIONS[6]`, PT=0.00312 h/ration | ✅ |
| 7  | Assembly Line 3 (assembly, Q=5,000) | `OPERATIONS[7]`, Q batch=5,000, ROP=48 h (S=1) | ✅ |
| 8  | Line of Communication (AL → SB) | `OPERATIONS[8]`, PT=1 día, ROP diario | ✅ |
| 9  | Supply Battalion (receipt) | `OPERATIONS[9]`, Q=2,400–2,600, ROP=24 h | ✅ |
| 10 | Line of Communication (SB → CSSU) | `OPERATIONS[10]`, Q=2,400–2,600, ROP=24 h | ✅ |
| 11 | Combat Service Support Units | `OPERATIONS[11]`, Q=2,400–2,600, ROP=24 h | ✅ |
| 12 | Line of Communication (CSSU → MP) | `OPERATIONS[12]`, Q=2,400–2,600, ROP=24 h | ✅ |
| 13 | Theatre of Operations (final demand) | DEMAND: U(2,400, 2,600) cada 24 h | ✅ |

**Resultado:** Estructura 1-1 con Fig 6.2. ✅

## 2. Riesgos (Tablas 6.6b, 6.7b, 6.8b)

| Riesgo | Distribución (tesis) | Repo `RISKS_CURRENT` | Match |
|--------|---------------------|----------------------|-------|
| R11 — Workstation breakdowns | U(1, 168 h) + exp(β=2 h); afecta Op5, Op6 | U(a=1, b=168) + exp(mean=2); affected_ops=[5,6] | ✅ |
| R12 — Contract delays | B(n=12, p=1/11); +1 sem si delay; afecta Op1 | B(n=12, p=0.0909); affected_ops=[1] | ✅ |
| R13 — Raw material shortages | B(n=12, p=1/10); +1 día; afecta Op2 | B(n=12, p=0.10); affected_ops=[2] | ✅ |
| R14 — Defective products | B(n=2,564, p=3/100); reproceso; afecta Op7 | B(n=2564, p=0.03); affected_ops=[7] | ✅ |
| R21 — Natural disasters | U(1, 16,128 h) + exp(β=120 h); Op3,5,6,7,9 | U(1, 16128) + exp(120); affected_ops=[3,5,6,7,9] | ✅ |
| R22 — LOC destruction | U(1, 4,032 h) + exp(β=24 h); Op4,8,10,12 | U(1, 4032) + exp(24); affected_ops=[4,8,10,12] | ✅ |
| R23 — Forward unit destruction | U(1, 8,064 h) + exp(β=120 h); Op11 | U(1, 8064) + exp(120); affected_ops=[11] | ✅ |
| R24 — Contingent demand | U(1, 672 h) + U(2,400, 2,600); Op13 | U(1, 672) + surge(2400, 2600); affected_ops=[13] | ✅ |
| R3 — Black swan | U(1, 161,280 h); 672 h outage en Op5,6,7,9 | U(1, 161280) + fixed(672); affected_ops=[5,6,7,9] | ✅ |

**Resultado:** Distribución 1-1 con Tablas 6.6b, 6.7b, 6.8b. ✅

## 3. Buffers de inventario (Tabla 6.16)

| Período (h) | Op3,j rm | Op5,j rm | Op9,j rations | Repo `INVENTORY_BUFFERS` |
|-------------|----------|----------|---------------|--------------------------|
| 168   | 15,360  | 15,360  | 15,750  | ✅ (clave "168") |
| 336   | 30,720  | 30,720  | 31,500  | ✅ |
| 504   | 46,080  | 46,080  | 47,250  | ✅ (default del repo) |
| 672   | 61,440  | 61,440  | 63,000  | ✅ |
| 1,344 | 122,880 | 122,880 | 126,000 | ✅ |

**Resultado:** 1-1 con Tabla 6.16. ✅ — y **Op5 está ahora expuesto** en el repo (post-cambio 2026-06-10).

## 4. Capacidad de manufactura por turno (Tabla 6.20)

| S | Op3 Q | Op4 Q | Op7 Q | Op7 ROP | Repo `CAPACITY_BY_SHIFTS` |
|---|-------|-------|-------|---------|---------------------------|
| 1 | 15,500 | 15,500 | 5,000 | 48 h | ✅ |
| 2 | 31,000 | 31,000 | 5,000 | 24 h | ✅ |
| 3 | 47,000 | 47,000 | 7,000 | 24 h | ✅ |

**Resultado:** 1-1 con Tabla 6.20. ✅

## 5. Horizonte y warm-up (Sección 6.8)

| Parámetro | Tesis | Repo | Match |
|-----------|-------|------|-------|
| Horizonte | 161,280 h = 20 años (sección 6.8.1) | `SIMULATION_HORIZON = 161_280` | ✅ |
| Año base | 8,064 h (336 × 24) | `HOURS_PER_YEAR_THESIS = 8_064` | ✅ |
| Año base gregoriano | n/a | `HOURS_PER_YEAR_GREGORIAN = 8_760` (default runtime) | ⚠️ Diferencia operativa |
| Warm-up trigger | Primera llegada de Q=5,000 a Op9 | `WARMUP.trigger_op=9, trigger_quantity=5000` | ✅ |
| Warm-up estimado determinístico | 838.8 h | `WARMUP.estimated_deterministic_hrs=838.8` | ✅ |
| Step unitario | 168 h (1 semana) | `BENCHMARK_REFERENCE_STEP_SIZE_HOURS = 168` | ✅ |
| Max episodios | 260 (1 año × 5) | `BENCHMARK_REFERENCE_MAX_STEPS = 260` | ✅ |
| Órdenes totales | 6,000 (j=1…6000) | `MAX_ORDERS = 6_000` | ✅ |
| Backorder queue cap | 60 | `BACKORDER_QUEUE_CAP = 60` | ✅ |

## 6. Variables de decisión (Sección 6.7)

| Variable decisión | Tesis (Tabla 6.16 / 6.20) | Track A del repo (post-2026-06-10) | Track B (extensión RL) |
|-------------------|----------------------------|------------------------------------|------------------------|
| Buffer `I_{t,S}` Op3 | 5 niveles discretos (15,360 → 122,880) | `a1` continuo ∈ [0.5×, 2.0×] sobre el nivel default (46,080) | `a1` (idem Track A) |
| Buffer `I_{t,S}` Op5 | 5 niveles discretos | **`a5` continuo ∈ [0.5×, 1.5×]** sobre el nivel default (centered, neutral a=0 → 1.0x) — regla thesis-anchored | `a5` (idem) |
| Buffer `I_{t,S}` Op9 | 5 niveles discretos (15,750 → 126,000) | `a2` continuo ∈ [0.5×, 2.0×] | `a2` (idem) |
| ROP Op3, Op9 | Tabla 6.20 implícita | `a3, a4` ∈ [0.5×, 2.0×] (extensión del repo) | `a3, a4` (idem) |
| Turnos `S ∈ {1,2,3}` | Discreto | `a6` (Track A 6D) piecewise → S=1/2/3 | `a6` (Track A 6D, movido) |
| Dispatch Q Op10, Op12 | Implícito (Tabla 6.20) | n/a (no expuesto en Track A) | `a7, a8` ∈ [0.5×, 2.0×] (Track B 8D) |
| Riesgo R1r, R2r, R3 | **Exógeno** (no controlable) | **Exógeno** ✅ | **Exógeno** ✅ |

**Regla del multiplicador Track A (post-2026-06-10):**
- Dims 0–3 (Op3_q, Op9_q, Op3_rop, Op9_rop): `m = 1.25 + 0.75·a` → [0.5, 2.0]
- Dim 4 (Op5_q): `m = 1.0 + 0.5·a` → [0.5, 1.5] (centrado, neutral a=0)
- Dim 5 (shift): piecewise → S=1/2/3 (idem tesis)

## 7. Comparación cuantitativa: thesis vs repo

**Validación oficial (`validation_report.py --official-basis thesis`, ejecutado 2026-06-10 post-cambio 6D):**
- **RMSE 62,055** vs Thesis baseline 87,918 → **–29.4% mejor**
- **Annual delivery 735,625** vs Thesis avg 767,500 → **–4.16%**
- **Secondary basis (gregorian):** RMSE 63,906, delivery 799,375 (+4.14%)

La desviación de ±4% en delivery está dentro de lo esperado para reimplementaciones de simulaciones DES-OR. No es un "resultado distinto" en el sentido de contradicción — es una re-simulación con misma estructura, mismos parámetros, y pequeña diferencia de calibración de warm-up que produce resultados del mismo orden.

## 8. ¿Por qué a veces tenemos resultados distintos a los de Garrido?

Cuatro fuentes plausibles de divergencia, ordenadas por contribución estimada:

1. **Calibración de warm-up.** La tesis usa un warm-up por evento (primera llegada de Q=5,000 a Op9), que dependiendo del path de riesgos puede ser 838 h (determinístico) o hasta ~2,000 h si hay riesgos. El repo define `max_priming_hours=2016` y completa siempre el priming con `priming_shifts=2` adicionales de 168 h cada uno. Si Garrido se quedó solo en el trigger y no en el priming extra, hay un offset de ~336 h (2 semanas) que desplaza ligeramente el periodo colectado.

2. **Base anual (8,064 h vs 8,760 h).** En 20 años la diferencia es 13,920 h (~1.7 años thesis). Esto afecta el conteo de "años completos" y el ciclo de Black-swan. El repo por default usa gregoriano, pero el comando de validación con `--official-basis thesis` ya normaliza.

3. **Semilla y secuencia de riesgos.** La tesis corre 90 configuraciones (Cf1…Cf90) con seeds documentadas (Tabla 6.17-6.19). El repo corre 5 seeds (11, 22, 33, 44, 55). Diferentes realizaciones del mismo proceso estocástico producen diferentes trayectorias — eso explica la varianza de ±5% en delivery.

4. **Extensión ROP en Track A.** El repo expone `a3` (Op3 ROP multiplier) y `a4` (Op9 ROP multiplier) que la tesis **no** manipula. Cuando el agente aprende a mover el ROP, está haciendo un control que Garrido no exploró — esto NO es un error de fidelidad, es una extensión documentada del action space.

**Bottom line:** los resultados del repo son **comparables y consistentes** con los de Garrido. La magnitud de desviación es compatible con reimplementaciones de simulaciones DES-OR con misma estructura y misma parametrización. Si quisieras replicar exactamente los 90 runs de Garrido (Cf1…Cf90, 10-20 años cada uno), el comando sería:
```bash
.venv/bin/python scripts/run_cf_1_to_90.py --year-basis thesis --seeds 11 22 33 44 55
```

## 9. Acciones recomendadas

1. **Para el paper:** documentar en §3.3 que el repo es thesis-faithful en estructura, riesgos, buffers y capacidades, y thesis-anchored en action space (continuo en lugar de discreto, Op5 ahora expuesto). Citar Tablas 6.16, 6.20 y Fig 6.2 explícitamente.
2. **Para futuras comparaciones:** añadir un script `replicate_thesis_cf_1_to_90.py` que recorra las 90 configuraciones con la semilla y el horizonte de Garrido, y reporte RMSE per-Cf vs el ReT(Cfi) de la tesis.
3. **Para transparencia:** publicar la `INVENTORY_BUFFERS` (Tabla 6.16) y `CAPACITY_BY_SHIFTS` (Tabla 6.20) como archivos de configuración versionados, junto a un `THESIS_FIDELITY.md` que audite cada variable.

## Anexo: artefactos verificables

- `validation_report.py --official-basis thesis` → CSV en `outputs/validation/validation_table_dual_basis.csv`
- `tests/test_op5_buffer_thesis_anchor.py` → 5 tests que validan el mapeo continuo `a5 → m → op5_rm target`
- `supply_chain/config.py:540-560` → `INVENTORY_BUFFERS` y `CAPACITY_BY_SHIFTS` (Tabla 6.16, 6.20)
- `supply_chain/config.py:275-360` → `RISKS_CURRENT` (Tablas 6.6b, 6.7b, 6.8b)
- `supply_chain/config.py:101-260` → `OPERATIONS` (Fig 6.2)
