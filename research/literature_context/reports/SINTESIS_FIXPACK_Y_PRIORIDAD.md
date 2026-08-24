# SÍNTESIS FINAL — Correcciones autorizadas + prioridad science-backed

**Fecha:** 2026-08-24 · **Decide:** Thommy (PI) · **Ejecutan:** Hermes + Codex + Claude + OpenCode

## 0. La corrección más importante: qué pasó realmente en Program Q

Tu lectura "media buena, cola mala" era **incompleta**. Los datos del adjudicator (verificados por Claude Opus, `SECOND_OPINION_CLAUDE.md`):

| celda | worst_product_fill vs classical (t) | vs open-loop |
|---|---|---|
| rho75_share90 | −0.0104 (**−1.82**) | **+0.146** (+5.96) |
| rho90_share75 | −0.0157 (**−3.41**) | **+0.196** (+8.16) |
| rho90_share90 | −0.0045 (−0.45) | **+0.427** (+11.7) |

La verdad de tres capas:
1. El learner **mejora la cola enormemente vs open-loop** (feedback funciona).
2. Pero el belief-MPC/clásico con feedback **también** la mejora — y el learner compra fill agregado desbalanceando el producto débil (sustitución media↔mínimo; `max_backlog_age` +123, `service_loss_auc` +908k vs clásico).
3. Codex encontró además que el token `STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION` es engañoso: H_OL y equivalencia pasaron 3/3 con 10/10 semillas positivas; lo único que falló fue la guardrail de cola (`adjudicate_program_q.py:21-30`). La adaptación SÍ se replicó.

**No se mata el proyecto: se separa en dos ejes — efficacy (pasó) y safety (falló) — y se corrige la cola.**

## 1. Qué se corrige (fix-pack v2, todo prospectivo)

| # | Supresor | Corrección | ¿CLI hoy? |
|---|---|---|---|
| a | Obs pobre | MFSC: posterior bayesiano compartido learner↔clásico (ablación raw+LSTM vs posterior+MLP). En Q no cambiar obs (ya expone belief_c). | parcial (v10 env-ready; parche 1 línea en choices) |
| b | Reward terminal esparsa | **PBRS ya existente** (`control_v1_pbrs`, potencial Q21 congelado CV-R² 0.44-0.50) + shift de potencial lineal (Müller 2025). NO shaping ad-hoc que rompa óptimo. | ✅ hoy |
| c | γ=0.99 / 260 semanas | γ=1.0 en Q2 (terminal, longitud fija); en MFSC γ=0.95-0.97 o freezing slow states (Wang 2023). | ✅ hoy (--gamma) |
| d | Pasos insuficientes | Checkpoints 50k/100k/200k; continuar a 400k solo si mejora 100k→200k >0.002. Curvas, no fe. | ✅ hoy |
| e | LSTM limitado | Ablación única 64→128 (Q usa LSTM 64, corrección de REPORT_B), mantener MLP [64,64]. | ✅ hoy |
| f | Comparador con modelo exacto | NO degradar el primario post-hoc (p-hacking). Añadir baselines secundarios sin-modelo y desagregar deltas por familia (los ganadores Q fueron min_cost_flow__2 y max_pressure__0, ¡ni siquiera belief-MPC!). | análisis |
| g | Potencia N | Diseño secuencial KN/FHN con IZ δ=0.01 (Hong 2021, Cheng 2023): varianza está en SEMILLAS (σ_seed≈0.032), no tapes → comprar semillas, recortar tapes. | diseño |

**Anti p-hacking (Claude, línea roja):** preregistro con hash antes de abrir semillas; margen de cola fijado prospectivamente (δ=SESOI=0.01 como no-inferioridad, nunca ≥0.027 post-hoc); reselección de comparador dentro del bootstrap se toca JAMÁS; el guardrail de cola no se elimina por ser el único que falla; publicar el fallo v4 gane o pierda.

## 2. Smoke artefacto-vs-física ≤48 CPU-h (diseño Codex sol-xhigh)

- **Gate 0 sin entreno (~6h):** frontera completa 4^8 + 10 clásicos sobre 128 tapes nuevos/celda → `G_PI = mean_t[max calendar] − max_c[mean ReT]`. Abandono físico inmediato si UCB95(G_PI)<0.01 en alguna celda.
- **B0 vs B1 pareado (3 seeds):** B0 = réplica exacta Q; B1 = PBRS-Q21 + γ=1 + LSTM 128. Éxito-promoción: LCB95(D_fix)>0, media Δ_N(B1)≥0.010 por celda, cola ≥−0.020. Abandono honesto: UCB95(D_fix)<0.005 → física, cerrar.
- Advertencia Claude: con déficit verdadero μ≤−0.003 en cola, ningún N flipea el gate — aceptarlo por escrito ANTES.

## 3. Prioridad según el propio repo

`PROMISING_LANES_REGISTRY.md` dice textualmente: el action contract es el lever #1, recurrente ya perdió, Track A conserva headroom real (+0.006 oracle) pero PPO lo erosiona (0/5 seeds, BC convergente → RL drift). Con Garrido 2024 (demanda CON variabilidad cv/smoothing — su §3.2; la i.i.d. es la tesis 2017) y la frontera 2021–2026:

**Prioridad 1 — Fix-pack smoke (arriba).** Barato, decide artefacto-vs-física.
**Prioridad 2 — Lane topológica pequeña (8–13 nodos)** con filling/repairing/recruiting, ahora science-backed: Kim 2023 IISE (MAPPO transshipment Dec-POMDP), Fan 2023 (GCN repair recursos escasos), Akashi 2023 (repair early-stage scarcity), Burtea & Tsay 2024 (CMDP costes explícitos + masking), Mousa 2024 (CTDE falla sin belief en obs), Kotecha 2025 (GNN+MARL 8-13 nodos). Gate previo: H_PI screen + placebo fungible (H≈0 obligatorio) + belief-MPC con las MISMAS acciones topológicas.
**Prioridad 3 — Outer-loop Alzheimer (UCB1-transfer ya ganó)** como companion paper, fiel al SCRES+AI de Garrido.

## 4. Estado de infraestructura (todo verificado)

- **Suite certificada**: 2256 passed / 38 failed (todos gobernanza-inventario local, no ciencia) + 9/9 war_risk tras instalar ema-workbench → `SUITE_CERTIFICACION.md`.
- **Segundas opiniones**: Codex sol-xhigh (`SECOND_OPINION_CODEX.md`), Claude Opus (`SECOND_OPINION_CLAUDE.md`), OpenCode muse-spark (`FIXPACK_FEASIBILITY_OPENCODE.md`) — los tres en `/home/ubuntu/scres-sources/reports/`.
- **Frontera descargada**: 15 PDFs 2021–2026 verificados en `pdfs_frontier/` + informe completo `REPORT_FRONTERA_2021-2026.md` (20 papers, cada decisión de diseño citada). 10 MANUAL para ti vía CRAI (Kim IISE, Kotecha, Liu POM, Burtea, Mousa, Cheng EJOR...).
- Pendiente menor: sincronizar `pdfs_frontier/` a tu Mac (siguiente comando rsync).

## 5. Próximos pasos propuestos (esperando tu OK)

1. Prerregistrar `program_q2_fixpack_v1.json` (estimandos efficacy/safety/authorization separados, margen cola δ=0.01, semillas vírgenes rango nuevo).
2. Lanzar Gate 0 (~6h) — sin entrenar nada.
3. Si Gate 0 pasa → smoke B0/B1 pareado (42h CPU).
4. En paralelo: diseñar entorno topológico mínimo y su H_PI screen.
