# COMPREHENSIVE SYNTHESIS — SCRES-IA Project (2026-06-29)

## 1. LO QUE PPO REALMENTE APRENDIÓ (Auditoría per-métrica)

**Veredicto: PPO NO aprendió una política adaptativa efectiva.** Las 3 seeds divergieron en 3 estrategias ortogonales:
- Seed 1: ramp-up (S3 siempre, resource alto)
- Seed 2: regime-switching (deramp R13, ramp R14, bueno cualitativamente pero resource altísimo)
- Seed 3: deramp-all (S1 siempre, resource mínimo, peor Excel)

**Métricas donde PPO GANA vs best static: 0 de 22.** La única "ganancia" en `n_late` es un artefacto composicional (menos órdenes servidas).

**Métricas donde PPO EMPATA:**
- Fill rate (ambos 0.007 — la rama recovery domina)
- On-time fill (ambos 0.000 — delay=54h hace que todas las órdenes sean late)

**Métricas donde PPO PIERDE:** Excel ReT (-0.0041), flow fill (-0.014), lost orders (+0.74), backlog (+10k), service loss (+196k), CTj p99 (+407h), RPj p99 (+172h), DPj p99 (+457h), TTR p95 (+76h), CVaR95 (+36M), resource (+28%).

**Por qué falla:** El reward landscape es PLANO — ~50 políticas estáticas están a 0.001 del máximo Excel. PPO encontró diferentes mesetas planas pero no la Pareto-eficiente. Las 3 seeds cayeron en diferentes cuencas de atracción sin converger.

---

## 2. CÓMO SE CALCULA LA RESILIENCIA (ReT)

**Fórmula de Garrido (Eq. 5.5):**
```
ReT_j = {
    APj/LT           si CTj=LTj y hay riesgos       (autotomy — on-time despite risks)
    0.5 × (1/RPj)    si CTj>LTj y recuperando       (recovery — late but recovering)
    0                si CTj>LTj y no recuperando     (non-recovery)
    1-(Bt+Ut)/j      si no hay riesgos               (fill-rate branch)
}
```

**Componentes por orden j:**
- **APj** = tiempo que el SC absorbió la disrupción SIN retrasarse
- **RPj** = tiempo desde el primer riesgo hasta la recuperación
- **DPj** = CTj (cycle time total del pedido)
- **Bt** = backorders acumulados
- **Ut** = unattended (perdidos)

**Cómo incrementar ReT:**
1. Reducir RPj (recuperación más rápida) → más dispatch downstream
2. Reducir CTj (cycle time más corto) → menos cola
3. Reducir Bt+Ut (menos backorders/perdidos) → más throughput
4. Aumentar APj (absorber disrupciones sin retraso) → buffers pre-posicionados

El bottleneck para TODO esto es el **dispatch downstream (Op9→Op10→Op12→theatre)** fijo en U(2400,2600)/24h.

---

## 3. TRACK A — Variables de Garrido (buffer + shifts)

**Lo que funciona:**
- Per-op buffer contract con BC warm-start + PPO en campaña no-estacionaria
- Pareto win (mismo Excel con 44% menos recurso) en Run A (3 seeds)
- El win es real pero seed-frágil (Run B perdió)

**Lo que NO funciona:**
- Single-buffer continuous_its: muerto bajo CRN denso
- Discrete(18): colapsa a constante
- PPO sin BC warm-start: no explora suficiente en 4D

**Amplificación (#1):** Rediseñar campaña a R13+R24 puro (quitar R14 que es peso muerto), φ∈{1,2,4,8,16}. El conflicto real es supply crisis (→S1) vs demand surge (→S2/S3).

**Amplificación (#2):** Gate más denso (7 fracs, 0-0.50) para descubrir que buffer ALTO en Op9 sí diferencia regímenes.

---

## 4. TRACK B — Variables desde cero

**El bottleneck real es Op12 (last-mile dispatch a theatre).** Shifts son el lever EQUIVOCADO.

**5 variables óptimas para Track B:**
1. **Op12 dispatch rate** (last-mile, máxima sensibilidad)
2. **Op10 dispatch rate** (link intermedio, vulnerable a R22)
3. **Op9 dispatch rate** (alimenta el pipeline)
4. **Backorder queue depth** (menos lost orders)
5. **Op9 buffer placement** (buffer EN el cuello)

---

## 5. PAPER STRATEGY — Cómo lograr los 4 claims

### Estado del draft (v.0_neuralNet-scres.docx): ~5% escrito. Placeholders. Sin resultados.

### H1 — Learning Effect: "Dynamic beats static"
- **Actual:** Pareto win frágil (3 seeds, Run A gana, Run B pierde)
- **Para lograrlo:** 10+ seeds, campaña R13+R24, gate denso, CRN confirmation
- **Claim viable:** "Non-inferior resilience at lower resource" (Pareto), NO "shorter recovery times"

### H2 — Adaptation: "Learning curve across disruptions"
- **Actual:** NUNCA testeado
- **Para lograrlo:** `retention_transfer.py --track continuous` con 40+ campaign blocks
- **Métrica:** `E[ΔR_k]` vs `k` — pendiente positiva con CI > 0

### H3 — Volatility Reduction: "Lower variance across heterogeneous intensities"
- **Actual:** Datos del conflict campaign YA existen (φ∈{1,4,8})
- **Para lograrlo:** Computar CV = σ/μ del ReT across regímenes para dynamic vs static
- **Claim viable:** con los datos actuales, solo falta el análisis

### H4 — Path Dependency: "L_{t-1} improves resilience"
- **Actual:** NULL en Discrete(18). NUNCA testeado en continuous/per-op.
- **Para lograrlo:** `retention_transfer.py --track continuous --algo ppo` con 3 arms (frozen/reset/retained) y 10+ seeds
- **Este es el experimento de más alto valor.** Si confirma, es el claim central del paper. Si null, el paper pivota a "boundary characterization."

### Operationalización de R_t = f(S_t, D_t, L_{t-1}):
- S_t: estado DES (inventario, backorders, capacidad)
- D_t: shock de disrupción (φ, ψ, enabled_risks)
- L_{t-1}: **dos niveles** — belief memory b_t (hazard features) + parametric memory θ_{k-1} (retained weights)
- R_t: Excel ReT, CD sigmoid, CVaR

---

## 6. RESEARCH GAPS (lo que este proyecto llena)

| Gap | Estado en literatura | Nuestra posición |
|---|---|---|
| RL + SCRES DES | Sin trabajo publicado | **Primeros** |
| DES + neural learning | Garrido et al. (2024) propone, nadie implementa | **Implementamos** |
| L_{t-1} formalizado | No existe | **Formalizamos y testeamos (H4)** |
| Frontier-dependent learning | No en literatura | **Nuestro framing central** |
| BC + PPO para SCRES | Sin trabajo | **Exploramos** |
| Non-stationary DES + RL | Wang et al. (2023) en ABM, no DES | **Campañas de conflicto** |
