# Preregistro — Fase 1A: ¿la contención con dientes crea headroom?

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_contention_headroom_v1.py`.
Plan: `Fase 1A`. Objetivo raíz: las dos preguntas de Garrido 2024 (`CLAUDE.md`).

## La hipótesis, y de dónde sale

Program O midió `H_PI = 0,1515` (LCB95 0,1156) sobre un recurso compartido **no fungible**, y
—control causal— **exactamente 0** al hacer el mismo recurso fungible. La contención es el
mecanismo. En el envolvente thesis-native medimos `~1e-4` en todo: resolución continua, buffers,
nodos nuevos aguas abajo, mezcla de regímenes, orientación por interacción.

El DES completo **ya tiene** dos CSSU en disputa (`cssu_topology_mode="split_v1"`), y R23 ya
destruye **una** de las dos. Pero la disputa está **desdentada por tres vías**, y las tres eran
decisiones nuestras, no de la tesis:

1. **La capacidad no fungible nunca se probó.** `reallocate_unused` estaba **cableado a `True`**
   en `supply_chain.py`: lo que un destino no usa fluye al otro. Eso es *precisamente* la
   condición bajo la que Program O midió cero.
2. **El reparto sólo admitía tres puntos** (`0,25 / 0,50 / 0,75`). Un reparto que siga a *qué
   unidad está caída* no cabe en tres puntos.
3. **R23 corre a frecuencia de tiempo de paz.** Es una cadena militar; la propia justificación
   del proyecto es que funcione en guerra.

> **H1 — contención.** Con la capacidad **no fungible**, el mejor reparto constante **varía con
> el régimen de riesgo**: `H_regime > 0` con `LCB95 > 0`.
>
> **H2 — mecanismo.** Ese headroom es **atribuible a la no fungibilidad**: `H_regime` no
> fungible **>** `H_regime` fungible. Si el fungible iguala o supera, H1 no mide contención.
>
> **H3 — escala.** El headroom **crece con la escalada de R23**. Si es plano, la destrucción de
> una unidad no es la fuente y hay que buscar en otro sitio.

## Diseño

* **Palanca**: `cssu_allocation_a` continuo, 9 niveles en `[0,1, 0,9]`. Constantes, no política:
  esto mide si **hay** headroom, no si alguien lo captura.
* **Regímenes**: 6 = {R23 base, R23 ×3 frecuencia, R23 ×3 frecuencia ×2 impacto} × {con R1r, sin
  R1r}. La escalada usa `risk_frequency_multipliers_by_id` / `risk_impact_multipliers_by_id`,
  el permiso explícito de Garrido.
* **Brazos de fungibilidad**: `cssu_reallocate_unused ∈ {True, False}` — el control causal.
* **Regla de servicio**: las tres (`SPT_FULL`, `FIFO_PARTIAL`, `R24_AGE_PARTIAL`); el reparto
  actúa distinto en cada una y no quiero elegir a ciegas.
* **Semillas**: `5 200 001…` vírgenes, CRN — **las mismas** en todas las celdas.
* **Métricas**: `ret_excel_risk_conditional` (primaria, la de toda la campaña),
  `ret_excel_visible_clipped_0_1` (acotada) y `flow_fill_rate` al lado. Cobb-Douglas se calcula
  en la Fase 2, que es donde se decide la métrica.
* **`H_regime` = `mean_r[max_a] − max_a[mean_r]`**, con LCB95 por bootstrap agrupado por semilla.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_the_lever_actually_moves_the_system` | si `ret` no varía con `allocation_a`, la palanca es inerte y `H_regime` mide ruido |
| `f2_non_fungible_actually_binds` | `unused > 0` debe aparecer con `False` y ser ~0 con `True`; si no, el flag no hizo nada y H2 es vacua |
| `f3_escalation_actually_escalates` | más eventos R23 con los multiplicadores altos; si no, H3 no se ha probado |
| `f4_crn_is_common` | semillas distintas por celda serían más muestreo, no CRN |
| `f5_H_regime_is_non_negative` | por construcción `mean[max] ≥ max[mean]`; un negativo sería un bug de agregación |
| `f6_seeds_are_virgin` | reutilizar semillas invalidaría cualquier confirmación posterior |

## Regla de lectura, fijada de antemano

* **`H_regime` no fungible ≥ 0,01, `LCB95 > 0`, y mayor que el fungible** →
  `CONTENTION_HEADROOM_FOUND`. Autoriza la Fase 3 (escalera MLP vs RL) sobre esta palanca.
* **`H_regime` entre `1e-3` y `0,01`** → `CONTENTION_HEADROOM_SUBCRITICAL`: hay señal pero no
  llega a la barra; se reporta y se combina con la Fase 1B/1C antes de entrenar nada.
* **`H_regime < 1e-3` o el fungible empata** → la contención **tampoco** abre la puerta en esta
  cadena, y eso es un resultado fuerte: sería el mismo mecanismo que dio 0,1515 en Program O
  fallando aquí, lo que localizaría la causa en la **topología serial**, no en la ausencia de
  disputa.

**Lo que un PASS NO dice:** que RL gane. Sólo que existe una decisión dependiente del estado.
La escalera empieza por constantes y umbrales, y el MLP se entrena en serio.
