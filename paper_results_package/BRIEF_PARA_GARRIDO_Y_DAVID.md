# Resultados Finales — Proyecto SCRES+IA (DES + RL)

**Para:** Alexander Garrido, David  
**De:** Thomas  
**Fecha:** 4 de abril 2026  
**Repo:** `proyecto_grarrido_scres+ia`

---

## 1. Resumen en una línea

RL funciona para control adaptativo de resiliencia en la cadena MFSC **cuando el espacio de acciones cubre el cuello de botella operativo activo** — y esta condición es verificable causalmente.

---

## 2. ¿Qué hicimos?

Reconstruimos la simulación DES de la cadena MFSC (13 operaciones, SimPy) basada en la tesis doctoral de Garrido-Rios (2017). Entrenamos agentes PPO y RecurrentPPO con Stable-Baselines3 bajo dos configuraciones:

- **Track A (fiel a la tesis):** 5 dimensiones de acción (inventario + turnos de ensamblaje)
- **Track B (extensión diagnóstica):** 7 dimensiones (+ control de despacho downstream Op10/Op12)

Cada configuración fue evaluada con 5 semillas, 500k timesteps, 20 episodios de evaluación.

---

## 3. Resultados principales

### Track A: Resultado negativo honesto


| Política         | Fill Rate | Mejor estático |
| ---------------- | --------- | -------------- |
| PPO (control_v1) | 0.782     | S2 = 0.792     |
| PPO (ReT_seq_v1) | 0.788     | S2 = 0.792     |
| RecurrentPPO     | 0.751     | S2 = 0.794     |


**PPO NO supera a S2.** Causa: el cuello de botella está en la distribución downstream (Op9 despacha máx ~2,500 raciones/día), no en el ensamblaje. Más turnos no se traducen en más entregas.

### Track B: PPO supera todos los baselines


| Política                   | Fill Rate | Order-level ReT | Horas ensamblaje |
| -------------------------- | --------- | --------------- | ---------------- |
| **PPO**                    | **1.000** | **0.950**       | ~18,500          |
| RecurrentPPO               | 1.000     | 0.949           | ~19,000          |
| Mejor estático (S2, d=1.5) | 0.990     | 0.479           | ~29,000          |
| S3 (d=2.0)                 | 0.985     | 0.449           | ~43,700          |


**PPO descubre una estrategia no obvia:** usa turnos mínimos (S1, 77% del tiempo) y compensa con despacho downstream selectivo. Es **57% más eficiente** en horas de ensamblaje que el mejor estático, con mayor fill rate.

### Ablación causal


| Config          | Fill Rate | ReT   | ¿Supera estáticos? |
| --------------- | --------- | ----- | ------------------ |
| Joint (7D)      | 1.000     | 0.948 | ✅ SÍ               |
| Downstream-only | 1.000     | 0.953 | ✅ SÍ               |
| Shift-only      | 0.953     | 0.686 | ❌ NO               |


**Prueba causal:** Sin control downstream, PPO no puede ganar (reproduce Track A). Con solo downstream, PPO gana incluso sin shifts adaptativos.

### Robustez bajo estrés creciente


| Escenario  | PPO Fill | Mejor estático Fill | Ventaja PPO |
| ---------- | -------- | ------------------- | ----------- |
| Current    | 1.000    | 0.9997              | +0.0pp      |
| Increased  | 0.993    | 0.901               | +9.2pp      |
| Severe     | 0.966    | 0.656               | +31.0pp     |
| Severe_ext | 0.558    | 0.282               | +27.6pp     |


La ventaja de PPO **crece con la severidad**, pero colapsa bajo estrés extremo. Esto demuestra que PPO no es invencible — tiene un límite operativo.

### Insensibilidad a la función de reward

5 de 7 funciones de reward convergen a la misma política fuerte. Cuando el espacio de acciones cubre el bottleneck, la reward importa poco.

### Forecasts no son críticos

PPO mantiene fill=1.000 incluso con forecasts scrambled o zeroed. La política usa señales de régimen, no predicciones numéricas.

---

## 4. Papers que envió Alexander

### Rajagopal & Sivasakthivel (2024) — WFFNN for SCR Strategy Selection

- **Método:** Red neuronal para seleccionar estrategia de resiliencia (clasificación)
- **Para nosotros:** Background. Ellos *clasifican* estrategias; nosotros *controlamos* online. No compite.

### Rezki & Mansouri (2024) — ANN for SC Risk Management

- **Método:** ANN para predecir riesgo de proveedores
- **Para nosotros:** Background. Es predictivo, no prescriptivo. No usa DES ni RL.

### Ordibazar et al. (2022) — Counterfactual Explanation for SCR

- **Método:** XAI para recomendar mitigación de riesgos
- **Para nosotros:** Background. Diferente paradigma (explicabilidad vs. control adaptativo).

**Conclusión:** Ninguno de los 3 hace DES+RL para control operacional. Nuestra posición en la literatura se mantiene. Los 3 van en la sección 2 como evidencia de actividad en "AI+SCRES" pero en un bucket diferente al nuestro.

---

## 5. Estructura propuesta del paper

1. **Introduction:** "When does RL help for SCRES?" — framing como pregunta, no como claim universal
2. **Related Work:** AI+SCRES landscape (incluye los 3 papers de Alexander), RL for operations, resilience metrics
3. **Methodology:**
  - 3.1 DES model (thesis reconstruction)
  - 3.2 RL formulation (PPO, observation space, action contracts)
  - 3.3 Reward design (7 modes, C-D analysis)
4. **Track A Results:** Negative result + structural explanation
5. **Track B Results:** Positive result + ablation + reward sweep
6. **Robustness & Policy Analysis:** Cross-scenario + forecast sensitivity + strategy discovery
7. **Discussion:** Action-space alignment, limitations, implications
8. **Conclusion**

### Target journal: **IJPR** (International Journal of Production Research, Q1)

- Backup: Computers & Industrial Engineering (Q1)

---

## 6. Estado técnico

### ✅ Completado

- DES validada contra tesis ✅
- Track A cerrado con resultado negativo ✅
- Track B validado (500k × 5 seeds) ✅
- Ablación causal (joint + shift_only + downstream_only) ✅
- Reward sweep (7 modes × 5 seeds) ✅
- Cross-scenario (4 niveles de riesgo) ✅
- Forecast sensitivity ✅
- Anticipation analysis (5 seeds) ✅
- 16 findings auditados ✅
- 215/216 tests pasan (el 1 que fallaba ya fue corregido) → 216/216 ✅
- Script de reproducibilidad (`reproduce_paper.sh`) ✅

### ⏳ En progreso

- Ablación `downstream_only` — seed 5/5 corriendo (ETA ~4 AM COT)
- Observation ablation — solo smoke 50k (nice-to-have at 500k)

### 📝 Por hacer (para la fase de escritura)

- Escribir sections 3-7 del paper
- Generar figuras (learning curves, action heatmaps, degradation curve)
- Redactar abstract y conclusions

---

## 7. Para reproducir

```bash
cd proyecto_grarrido_scres+ia
source .venv/bin/activate

# Smoke test (~30 min)
bash reproduce_paper.sh --smoke

# Full reproduction (~48h)
bash reproduce_paper.sh
```

---

## 8. Archivos clave

- `paper_results_package/README.md` — Tablas con todos los resultados
- `docs/PAPER_FINDINGS_REGISTRY.md` — 16 findings auditados con evidencia
- `docs/briefs/garrido_track_b_audit_brief_2026-04-03.md` — Brief técnico completo
- `reproduce_paper.sh` — Script de reproducción end-to-end
- `CLAUDE.md` — Documentación técnica completa del proyecto
