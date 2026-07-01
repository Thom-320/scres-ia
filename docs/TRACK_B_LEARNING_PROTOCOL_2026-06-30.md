# Track B Learning Protocol — Q1 Evidence Package

> **2026-07-01 status note:** Use
> `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` for final claim wording and
> reviewer-defense gates. This protocol remains useful for H4 and Q1 structure,
> but any "strict Pareto" or old Track B dimensionality language must be checked
> against the current 8D Track B contract and dense-CRN audit.

## Current Status

### ✅ DEMOSTRADO
1. **Track B win**: PPO > best static en ReT (+8%), flow fill (+46%), rolling fill (1.00 vs 0.85), cost (-32%). 24/24 episodios.
2. **Adaptive policy**: PPO aprendió shift mix 25/47/28 vs static 100% S3. Dispatch 1.29x/1.34x vs static 2.00x.
3. **Track A boundary**: Static frontier IS el óptimo global (136-pt random search). Frontier-dependent learning demostrado.

### ❌ CRÍTICO FALTANTE
4. **H4 (L_{t-1})**: Retained-vs-reset NUNCA testeado en Track B. En Track A fue NULL.
5. **Dense static frontier**: Track B solo tiene 9 statics (vs 63 de Track A). Sin CRN.
6. **Rich metrics panel**: Track B evaluado con métricas limitadas (no APj/RPj/DPj/CTj para statics).
7. **CI95**: Solo 3 seeds, sin bootstrap CI sobre ReT.
8. **Generalización**: Solo adaptive_benchmark_v2, no cross-regime.

---

## Protocolo para Q1 — 5 fases

### Fase 1: Dense Static Frontier + CRN (PRIORIDAD #1)
**Por qué:** Track A's "win" se evaporó cuando se evaluó con frontera densa + CRN. Track B tiene solo 9 statics. Esto es lo PRIMERO que debe hacerse antes de cualquier claim.

**Qué hacer:**
- 21 dispatch fractions (0.00, 0.05, ..., 1.00 step 0.05) × 3 shifts × 3 dispatch_op12 multipliers
- CRN eval con seed0=9000 (mismo para dynamic y static)
- 5 seeds por cell
- Reportar: ReT, flow_fill, rolling_fill, cost, CVaR95
- Output: dense_static_frontier_track_b.csv

**Tiempo:** ~2h local o 30 min Kaggle

### Fase 2: Full Metrics Panel Audit (PRIORIDAD #2)
**Por qué:** Track A's "win" evaporó cuando se computaron rich metrics. Debemos auditar ANTES de claimar.

**Qué hacer:**
- Re-evaluar PPO y top-5 statics con `compute_episode_metrics` (APj, RPj, DPj, CTj, TTR, CVaR, CD)
- Comparar PPO vs best static en 22+ métricas
- Reportar: tabla completa estilo Garrido
- Si PPO gana en ≥15/22 métricas → claim fuerte

**Tiempo:** ~30 min (modelos ya existen)

### Fase 3: H4 Retained-vs-Reset en Track B (PRIORIDAD #3 — EL CLAIM DE APRENDIZAJE)
**Por qué:** Es el experimento de más alto valor. Track B tiene non-stationarity real (Markov regime process). Si H4 confirma, el paper tiene L_{t-1} como contribución teórica.

**Qué hacer:**
- Usar `scripts/retention_track_b.py` (ya existe, necesita escalar)
- Escalar: max_steps=104, cycles=30, seeds=10, online_timesteps=8k
- Dos condiciones: obs_full (regime visible) vs obs_hidden (regime masked)
- 3 arms: frozen / reset / retained
- Primary: ΔR_memory = ReT_retained − ReT_reset, seed-clustered CI95
- Negative controls: shuffle, no-update, wrong-history

**Tiempo:** ~4-6h en Kaggle (one-seed-per-kernel con partial writes)

**Veredicto:**
- Si ΔR_memory > 0 con CI95 > 0 en obs_hidden → **H4 CONFIRMADO** → L_{t-1} es real
- Si ΔR_memory ≈ 0 → H4 NULL → paper pivota a "boundary characterization"

### Fase 4: Statistical Q1 Upgrades (PRIORIDAD #4)
**Por qué:** Q1 reviewers exigen effect sizes + Pareto framing + learning curve.

**Qué hacer:**
1. **Cohen's d** para cada comparación PPO vs best static (siguiendo ReflectiChain)
2. **Pareto frontier figure**: ReT vs flow_fill, points = policies, "strictly dominates" wording (siguiendo SeqRoute)
3. **Learning curve**: ReT vs training step, con CI band (siguiendo Li & Karunarathne)
4. **Action entropy curve**: exploración vs convergencia
5. **Per-regime behavior breakdown**: qué hace PPO en nominal vs disrupted

**Tiempo:** ~2h análisis (sin simulación nueva)

### Fase 5: Generalization (PRIORIDAD #5)
**Por qué:** Q1 exige que el win no sea específico a un régimen.

**Qué hacer:**
- Entrenar en adaptive_benchmark_v2, evaluar en:
  - risk_level=current (Garrido faithful)
  - risk_level=increased
  - risk_level=severe
  - h52 (short) y h260 (long)
- Si PPO mantiene non-inferiority → generalization confirmada

**Tiempo:** ~3h Kaggle

---

## Claims del Paper (reformulados para Q1)

### Primary Claim (H1)
> A neural policy controlling downstream dispatch multipliers and manufacturing shifts learns a resource-efficient adaptive strategy that **strictly Pareto-dominates** the best static policy on resilience (ReT +8%), service continuity (flow fill +46%, rolling fill 1.00), and resource cost (-32%) under recurring heterogeneous disruptions.

### Secondary Claim (Boundary)
> With Garrido's exact decision variables (buffer + shift only), the static frontier is the **global optimum** — demonstrated by exhaustive search — proving that neural learning value is **frontier-dependent**: learning improves resilience only when the action space reaches the binding constraint.

### Tertiary Claim (H4 — si confirma)
> Retained policy knowledge (L_{t-1}) accumulated across disruption campaigns improves cold-start resilience on unseen campaigns when the disruption regime is not directly observable, providing the first empirical evidence for Garrido et al.'s (2024) "Alzheimer effect" hypothesis.

### Open Question (H4 — si null)
> Retained learning across campaigns remains an open hypothesis. The non-stationary regime process in Track B provides the structural conditions for L_{t-1}, but the current training budget may be insufficient to demonstrate it. Future work should explore meta-learning (MAML) and recurrent architectures (GTrXL) for cross-campaign knowledge transfer.

---

## Must-Cite Papers (de la búsqueda Q1)

1. **Garrido, Pongutá & Adarme (2024)** — "Alzheimer effect" gap → nuestro anchor
2. **Garrido, Pongutá & García-Reyes (2024)** IJPR — CD resilience index → nuestra métrica secundaria
3. **Gil et al. (arXiv:2509.06853)** — BC+RL para bioprocess → justifica nuestro warm-start
4. **SeqRoute (arXiv:2605.25424)** — Pareto dominance framing → "strictly dominates"
5. **ReflectiChain (arXiv:2606.10359)** — Statistical reporting (p + Cohen's d) → nuestro estándar
6. **Bekal et al. (arXiv:2503.19212)** — BWT/FWT para retained learning → nuestra métrica H4
7. **ParallelCBF (arXiv:2605.15509)** — Checkpoint selection → único precedente
8. **Ghasemloo & Eckman (arXiv:2605.27556)** — Surrogate DES acceleration → justifica slow DES
9. **Li & Karunarathne (arXiv:2507.18398)** — DES+PPO learning curve → nuestro formato
10. **Che et al. (arXiv:2409.03740)** — Differentiable DES → future work
