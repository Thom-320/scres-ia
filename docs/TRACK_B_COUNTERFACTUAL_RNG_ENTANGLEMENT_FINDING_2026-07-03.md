# Hallazgo crítico: el contrafactual R_full - R_reset(w) no es válido en este simulador — 2026-07-03

## Contexto

Antes de ejecutar el contrafactual `R_full - R_reset(pre-risk, risk_id)` con anclas precisas del
`risk_event_ledger.csv` (el paso acordado tras "ejecutalo"), verifiqué el supuesto en el que se
apoya TODO el diseño: que sustituir acciones en una ventana y continuar la política no cambia qué
riesgos ocurren después (solo cambia el comportamiento). Esto se había verificado antes solo para
la parte "temporizador" de los riesgos (`risk_rng` es un stream separado de `self.rng`). **Esa
verificación era incompleta.**

## El hallazgo

`_adaptive_regime_controller` (supply_chain.py:1748-1759) revisa el régimen cada
`ADAPTIVE_BENCHMARK_REVIEW_HOURS = 48` horas y decide el siguiente régimen con
`self.adaptive_regime = str(self.rng.choice(next_regimes, p=probs))` — usando `self.rng`, el
stream **compartido**, no un stream de régimen dedicado. Ese mismo `self.rng` también se usa en:

- `_pt()` (supply_chain.py:1824): ruido de tiempo de procesamiento, `self.rng.triangular(...)`,
  se dispara en cada operación procesada — la frecuencia depende de cuánto se produce, que
  depende de las acciones.
- `_risk_R14` (línea 2361): `self.rng.binomial(produced, p)`, y `produced` es la producción real
  del día — directamente dependiente de la política.
- Ruido del forecast (`_update_adaptive_forecasts`, líneas 1738/1743).

Como todos estos consumidores comparten un único generador (`self.rng`), la posición en el stream
en el momento en que se dispara la revisión de régimen depende de cuántas otras veces se llamó a
`self.rng` antes — y eso depende de las acciones tomadas. Es decir: **el régimen (y por lo tanto
la intensidad de riesgo que modula `_get_risk_p`/`_get_risk_recovery_mean` para R11, R12, R13,
R21, R22, R23, R24, R3) no es independiente de la política**, aunque los temporizadores de riesgo
sí usan `risk_rng` de forma separada.

## Verificación empírica

Comparé el ledger ya generado (mismo `eval_seed`, seed=1, episodio=1) entre `ppo_mlp` y
`real_kan` — dos políticas distintas deberían, bajo el supuesto de riesgo-independiente-de-acción,
producir la MISMA secuencia exacta de eventos de riesgo (mismo `risk_id`, mismo `start_time`).

- Los primeros 15 eventos (hasta t=1200h) coinciden exactamente.
- Por octavo de episodio, el acuerdo exacto cae a **86% en el primer octavo** (t=0-2400h) y baja
  a **68-78%** en el resto — nunca vuelve a ser 100%, y la divergencia ya es visible desde el
  principio, consistente con que cada revisión de régimen (cada 48h) es una oportunidad de
  decorrelación.
- Conteos totales por riesgo, mismo seed/episodio: R13 75 (ppo_mlp) vs 66 (real_kan), R12 4 vs 1,
  R14 611 vs 637 — divergencias grandes en riesgos de conteo bajo, exactamente donde un cambio de
  intensidad de régimen tiene más efecto relativo.

## Por qué esto invalida el contrafactual por ventana

El diseño `R_full - R_reset(w)` asume: misma trayectoria de riesgos en ambas ramas, solo cambia el
comportamiento en la ventana. Pero en cuanto la acción sustituida difiere de la que habría tomado
la política real, el consumo de `self.rng` (vía `_pt`) diverge inmediatamente, lo cual hace
diverger el régimen en la siguiente revisión (máximo 48h después), lo cual cambia la intensidad de
riesgo real para el resto del episodio. El delta de ReT medido después de la ventana queda
contaminado por una diferencia REAL en qué riesgos ocurrieron, no solo por el cambio de
comportamiento — exactamente el tipo de confusión que el diseño pretendía evitar.

Esto **retroactivamente explica mejor** la inestabilidad encontrada en las Rondas 1-4
(`docs/TRACK_B_PREVENTION_COUNTERFACTUAL_VALIDATION_2026-07-03.md`): no era solo un problema de
referencia de reset o ancho de ventana — es que la técnica de "empalme" (splice) dentro del mismo
episodio es estructuralmente inválida para este simulador, sin importar qué tan bien se elija la
ventana o la referencia.

## Qué sigue siendo válido

- El event-study descriptivo (`risk_event_aligned_action_study.csv`,
  `risk_event_aligned_by_risk_study.csv`) **no** tiene este problema: solo describe la trayectoria
  real ya simulada, sin empalmar ni sustituir nada. Sigue siendo evidencia descriptiva legítima.
- Cualquier comparación de **rollouts completos e independientes** (política entrenada vs.
  heurística, política entrenada vs. estática, con/sin observación privilegiada vía reentrenamiento
  o wrapper aplicado desde t=0) sigue siendo válida — es exactamente el protocolo CRN ya usado en
  E1-E6 y en el propio no-forecast confirm run. La limitación conocida (la correlación entre pares
  se degrada después de la primera divergencia) ya se acepta en todo el proyecto y se maneja con
  múltiples seeds/episodios e IC95%, no invalida la comparación de medias.

## Qué NO se debe hacer sin aprobación explícita

Arreglar esto en el simulador (dar al controlador de régimen, al ruido de `_pt` y a R14 sus propios
streams dedicados, independientes de las acciones) es un cambio de código en una parte ya validada
del simulador (pasa el gate de fidelidad H2/H3). Cualquier cambio ahí exige re-validar el gate de
fidelidad completo y probablemente invalida la comparabilidad con resultados ya generados. No lo
haría sin que el usuario lo pida explícitamente.

## Recomendación

1. **Abandonar la técnica de empalme dentro del mismo episodio** (`R_full - R_reset(w)`) para la
   pregunta preventivo/reactivo — no es arreglable ajustando ventana o referencia, es estructural.
2. Para aislar causalmente "el pequeño aumento anticipatorio observado, ¿tiene valor en ReT?", usar
   en su lugar una comparación de **rollouts completos independientes**: evaluar las heurísticas de
   referencia (`heur_forecast_threshold`, `heur_downstream_reactive`, estática) como políticas
   completas bajo el mismo protocolo (5 seeds × 12 episodios, `adaptive_benchmark_v2`), y comparar
   tanto ReT Excel como el patrón de `action_intensity` alineado a eventos reales (ya construido)
   contra PPO+MLP/Real-KAN. Esto no aísla "esta ventana específica de esta política" pero sí
   responde honestamente "¿el patrón de la política entrenada se parece más al de una referencia
   diseñada como preventiva o a una reactiva, y hay una heurística preventiva completa que gane en
   ReT Excel?" — sin el confound de RNG.
3. Mantener la ruta de memoria histórica + cabeza auxiliar (ya en curso, `TRACK_B_RISK_BELIEF_AUX_HEAD_IMPLEMENTATION_PLAN_2026-07-03.md`)
   como la vía principal para demostrar prevención real, precisamente porque no depende de esta
   técnica de empalme inválida — usa predicción supervisada + comparación de ReT Excel entre
   políticas completas, evaluadas de forma independiente.
