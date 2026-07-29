# Gate oráculo de valor preventivo — 2026-07-04 (matizado: negativo en ReT Excel, POSITIVO en métricas de resiliencia)

**Actualización tras el análisis de métricas de resiliencia (ver sección al final): el veredicto
original de "no hay techo" era correcto para el escalar compuesto ReT Excel, pero incorrecto como
conclusión general. Las métricas específicas de resiliencia (TTR, CVaR de cola, peor ventana de 4
semanas, severidad de service-loss) SÍ muestran una mejora real, consistente en los 5 seeds. La
sección "Métricas de resiliencia" al final de este documento es la lectura vigente; el cuerpo
original queda abajo como referencia de por qué el escalar compuesto no lo detectó.**

## Pregunta

Antes de invertir en RL restringido, política de dos niveles o Ruta B (todas caras,
`docs/TRACK_B_PREVENTIVE_LEARNING_ROOT_CAUSE_AND_ALTERNATIVES_2026-07-04.md`), ¿existe siquiera
un techo de ReT Excel alcanzable por *cualquier* mecanismo de preparación anticipada, bajo el
reward/entorno actual? Se probó con un oráculo: una réplica del checkpoint canónico PPO+MLP `v7`
fixed-RNG, con conocimiento **perfecto** del calendario real de eventos R22/R24 de cada episodio
(el calendario es independiente de la acción bajo `strict_exogenous_crn=True`, verificado esta
sesión para todo riesgo discreto salvo R14), a la que se le inyecta un boost privilegiado en
{shift, op10_q, op12_q} durante una ventana de `L` semanas antes de cada evento conocido.

No se tocó el entorno ni el entrenamiento — es un replay puro sobre un checkpoint ya entrenado.
Script: `scripts/audit_track_b_oracle_prevention_ceiling.py`.

## Protocolo

- Checkpoint: `outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03` (PPO+MLP `v7`
  fixed-RNG canónico, 5 seeds × 12 episodios = 60 episodios de evaluación).
- Barrido: `L ∈ {1,2,4,8}` semanas × `boost ∈ {0.25,0.5,0.75,1.0}` × conjuntos objetivo
  `{R22}`, `{R24}`, `{R22+R24}` — 48 combinaciones, 60 episodios cada una (2880 episodios oráculo
  + 60 de línea base).
- Umbral de señal: `+0.0004` en ReT Excel (comparable a los efectos ya detectados en el proyecto).

## Verificación de la línea base

`baseline_ret_excel_mean = 0.005920543949378895` (replay de 60 episodios, B=0) vs. el valor
canónico ya conocido `0.005920640377903427` (`track_b_fixed_rng_confirm_5seed_60k_2026-07-03/summary.json`).
Diferencia: `9.6e-8` — coincide, confirma que el replay reproduce correctamente el checkpoint y el
protocolo de evaluación.

## Resultado: el mejor oráculo por conjunto objetivo

| Conjunto | Mejor L | Mejor boost | ReT Excel | Δ vs. línea base | Costo |
|---|---:|---:|---:|---:|---:|
| R22 | 8 | 0.75 | 0.005926 | **+0.000006** | 0.836 |
| R24 | 8 | 1.00 | 0.005942 | **+0.000021** | 0.997 |
| R22+R24 | 8 | 1.00 | 0.005942 | **+0.000021** | 0.997 |

**Ninguno de los tres conjuntos objetivo llega ni al 6% del umbral de señal (+0.0004).** El mejor
resultado absoluto de todo el barrido (R24/combinado, L=8, boost=1.0) es +0.000021 — literalmente
en el rango del ruido de esta sesión (comparable al tamaño de los deltas negativos/nulos ya vistos
en Arm 1/Arm 2/combinado).

## El patrón dentro del grid es igual de decisivo que el mejor punto

Para R24 y R22+R24, el grid completo (ver `oracle_grid.csv`) muestra una relación monótona pero
que satura: subir `L` y el boost sigue moviendo la ReT hacia arriba muy lentamente mientras el
**costo escala casi linealmente hasta el máximo** (`cost_index_mean` llega a 0.997 en el mejor
punto de R24 — el oráculo termina operando en modo caro casi TODO el episodio). Esto no es
casualidad: R24 ocurre ~38 veces por episodio de 104 semanas; con `L=8` el sindicato de ventanas
de preparación ya cubre casi todas las semanas del episodio. El oráculo, con boost=1.0 y L=8, ya
está pagando por "estar siempre preparado" — y aun así el ReT ganado es prácticamente nulo.

Para R22 (más raro, ~7 eventos/episodio) el techo es todavía más bajo (+0.000006) — consistente
con que R22 aporta menos oportunidades de preparación total, pero también confirma que ni el
riesgo con recuperación mandatoria más larga (~24h, la razón original para elegirlo en Arm 2) tiene
techo real bajo esta métrica.

## Conclusión

**Ni siquiera un oráculo con conocimiento perfecto del futuro y sin restricción de presupuesto
puede comprar ReT Excel preparándose antes de R22/R24, bajo el reward y entorno actuales.** Esto
cierra, de forma decisiva y barata, la pregunta que motivó todo el trabajo de esta noche/sesión
(memoria cruda, belief escalar, Ruta A, reward shaping, belief-conditioned PBRS): el problema
**no es de arquitectura ni de crédito de gradiente** — es que, bajo el reward `control_v1` y el
mecanismo de reacción actual (reaccionar rápido ya recupera casi todo el ReT perdido), no existe
ganancia marginal real que capturar con timing, punto.

**Consecuencia directa para las alternativas restantes:**

- **RL restringido / Lagrangiano** — NO recomendable ahora. Redistribuir el presupuesto de recurso
  en el tiempo solo ayuda si hay ReT que ganar por *cuándo* se gasta, y este gate muestra que no lo
  hay (el oráculo ya gasta sin restricción y no gana). Sin techo que perseguir, esta alternativa no
  tiene base.
- **Política de dos niveles / Ruta B** — misma conclusión: ambas atacan el problema de crédito de
  gradiente, que dejó de ser sospechoso. Ninguna vale la pena mientras el techo estructural sea
  cero.
- **Alternativa 4 original (encarecer la reacción tardía)** — se vuelve la única opción que ataca
  la causa real: el problema no es que el agente no sepa prepararse a tiempo, es que
  **reaccionar tarde no le cuesta lo suficiente bajo este reward**. Si se quiere una apuesta seria
  de prevención, este es el único lugar que queda por probar — y cambia la economía del entorno,
  no solo el agente. Por eso mismo requiere justificarse con cuidado en el paper (deja de ser un
  ajuste de agente/recompensa neutral respecto a la fidelidad Garrido).

**Recomendación original (superada por la sección siguiente):** coincido con la lectura de Codex —
cerrar Q1 con el spine PPO+MLP/Real-KAN como aprendizaje adaptativo (no preventivo), documentar
este gate oráculo como la evidencia decisiva de por qué se cerró la línea de prevención.

---

## Métricas de resiliencia — por qué el escalar compuesto no veía lo que sí estaba ahí

**Pregunta que motivó esto:** ReT Excel es un escalar compuesto (`w_bo·backorder + w_cost·costo +
w_disr·disrupción`) promediado sobre TODAS las órdenes del episodio. Si el oráculo mejora
específicamente cómo se manejan las órdenes golpeadas por R22/R24, ese efecto se diluye en el
promedio de cientos de órdenes normales — que el compuesto no se mueva no prueba que no se compró
resiliencia.

`supply_chain/episode_metrics.py::compute_episode_metrics` ya calcula, para cada episodio, varias
métricas de resiliencia estándar de la literatura (Ponomarov & Holcomb) que nunca se habían
revisado en este gate: `ttr_mean`/`ttr_p95` (time-to-recovery), `ret_excel_cvar05` (media del peor
5% de órdenes), `ret_excel_rolling_4w_min` (peor ventana de 4 semanas), `service_loss_auc_per_order`
(atraso ponderado por cantidad, independiente del costo) y `backlog_age_mean/max` (profundidad del
backlog). Script: `scripts/audit_track_b_oracle_resilience_metrics.py`, reutilizando exactamente el
mismo checkpoint/calendario/protocolo del gate anterior — sin reentrenar nada.

### Resultado: baseline vs. mejor config oráculo (5 seeds × 12 episodios = 60 por condición)

| Métrica | Baseline | Oracle R24 (L=8,B=1.0) | Δ | Δ% |
|---|---:|---:|---:|---:|
| ReT Excel (mean) | 0.005921 | 0.005942 | +0.000021 | +0.36% |
| **ret_excel_cvar05** (peor 5%) | 0.002284 | 0.002393 | **+0.000109** | **+4.78%** |
| **ret_excel_rolling_4w_min** (peor ventana) | 0.002018 | 0.002111 | **+0.000093** | **+4.63%** |
| **service_loss_auc_per_order** | 94130 | 79918 | **-14212** | **-15.10%** |
| **ttr_mean** (horas) | 91.95 | 91.03 | **-0.93** | **-1.01%** |
| **ttr_p95** (horas) | 157.10 | 151.94 | **-5.15** | **-3.28%** |
| backlog_age_mean | 25.58 | 22.58 | -3.00 | -11.73% |
| backlog_age_max | 43.98 | 36.38 | -7.60 | -17.28% |

**Todas las métricas de resiliencia se mueven en la dirección de mejora, con magnitudes 4-40 veces
mayores en términos relativos que el propio ReT Excel compuesto (+0.36%).**

### Verificación por seed (no es ruido de un seed atípico)

Se desagregó `oracle_R24_L8_B1.0` vs. `baseline_B0` por cada uno de los 5 seeds:

| Seed | service_loss_auc Δ | ttr_mean Δ | ttr_p95 Δ | cvar05 Δ | rolling_4w_min Δ |
|---:|---:|---:|---:|---:|---:|
| 1 | -20942 | -1.38 | -7.34 | +0.000185 | +0.000213 |
| 2 | -15366 | -0.89 | -5.93 | +0.000093 | +0.000095 |
| 3 | -3539 | -0.23 | -0.67 | +0.000019 | +0.000042 |
| 4 | -18435 | -1.26 | -4.36 | +0.000150 | +0.000103 |
| 5 | -12779 | -0.87 | -7.47 | +0.000099 | +0.000014 |

**`service_loss_auc`, `ttr_mean`, `ttr_p95`, `cvar05` y `rolling_4w_min` mejoran en los 5 de 5
seeds** — consistente, no arrastrado por un outlier. La magnitud varía (seed 3 es el más débil,
seed 1 el más fuerte) pero la dirección nunca cambia. **Única excepción real:** `backlog_age_mean/max`
solo cambia en el seed 1 (0.000 en los seeds 2-5) — ese efecto específico sí parece concentrado en
un seed y no debe reportarse como robusto sin más evidencia.

### Lectura correcta (revisa la conclusión anterior de este documento)

El gate oráculo original no estaba mal calculado — el techo del **escalar compuesto ReT Excel** sí
es casi cero. Pero la pregunta real del usuario ("¿está comprando resiliencia?") tiene una
respuesta distinta: **sí, de forma consistente entre seeds, en recuperación (TTR), en cola
(CVaR05/rolling-min) y en severidad de disrupción (service-loss AUC)** — simplemente esa ganancia
es demasiado pequeña, en las unidades de ReT Excel, para mover el promedio ponderado por costo.

Esto **no** significa que PPO ya esté capturando esto — el oráculo tiene conocimiento privilegiado
del futuro que PPO no tiene, y el propio gate de contrafactual (`R_full - R_reset`) ya mostró que
PPO no anticipa. Lo que sí cambia es la interpretación de "no hay techo": el techo existe, pero
está en una métrica que la función de recompensa actual (ReT Excel puro) no premia lo suficiente
para que el aprendizaje lo persiga.

### Recomendación revisada

1. **No cerrar la línea de prevención todavía.** El hallazgo de esta noche (memoria, belief,
   reward shaping, Ruta A) sigue siendo negativo *para ReT Excel como objetivo*, pero el techo de
   resiliencia SÍ existe — el problema no era "no hay nada que ganar", era "estábamos optimizando
   la métrica equivocada para verlo".
2. **Antes de invertir en RL restringido/dos niveles/Ruta B**, la prueba más barata y directa es:
   repetir este mismo gate oráculo pero con una recompensa/objetivo que pese explícitamente
   `ret_excel_cvar05` o `service_loss_auc_per_order` (o alguna combinación), no solo la media —
   confirmar que el oráculo con ESE objetivo sí despega claramente del baseline (más margen que el
   +4-15% ya visto), y solo entonces vale la pena entrenar un agente real contra ese objetivo.
3. Si se decide seguir esta línea, el candidato más natural para el paper es re-fundamentar la
   métrica de éxito de "prevención" como **reducción de TTR / CVaR de cola**, no como ganancia en
   ReT Excel medio — algo defendible y consistente con la literatura de resiliencia que el paper ya
   cita (Ponomarov & Holcomb 2009, Ivanov et al.).
4. La intervención de entorno (encarecer la reacción tardía) sigue fuera de alcance de Q1 sin
   decisión explícita del usuario — pero ahora es menos urgente, porque el mecanismo de "premiar la
   métrica correcta" parece más prometedor y no requiere tocar el entorno.

### Auditoría complementaria: resiliencia comprada por costo

Se agregó una auditoría event-conditioned para conectar esta lectura con el comportamiento de las
políticas reales:

`docs/TRACK_B_EVENT_RESILIENCE_PURCHASE_VERDICT_2026-07-04.md`

Esta auditoría no reemplaza el ReT Excel global; mide, alrededor de cada R22/R24 real, si el gasto
pre-riesgo compra continuidad local, evita backorders, reduce AUC de backlog o acorta recuperación.
Resultado resumido: frente a una regla barata, PPO+MLP sí compra resiliencia local de forma clara.
Frente a PPO+MLP, Real-KAN compra mejoras locales pequeñas pagando mucha más intensidad. Esto
refuerza la recomendación revisada: si queremos demostrar prevención, la métrica candidata no es
la media ReT Excel, sino una métrica de cola/recuperación/evento condicionada al costo.
