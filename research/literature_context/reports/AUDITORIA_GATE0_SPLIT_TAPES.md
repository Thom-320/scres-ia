# Auditoría del diseño de Gate 0 — split de tapes antes de ejecución

**Fecha:** 2026-08-24 · **Estado:** diseño no ejecutado; no modifica contratos ni código científico.

## Hecho verificado

- `SINTESIS_FIXPACK_Y_PRIORIDAD.md:38` propone: `G_PI = mean_t[max calendar] − max_c[mean ReT]`, con 4^8 calendarios y 128 tapes/celda.
- La búsqueda en `scres-ia-expanded-v2` no encontró una función ejecutada que implemente literalmente ese `G_PI`; el Gate 0 está propuesto, no adjudicado.
- El contrato ya ejecutado de Program O usa otro artefacto y declara `static_reselected_over_65536_in_every_resample=true`; no debe mezclarse ese resultado con el Gate 0 prospectivo.

## Riesgo metodológico

Si `max calendar` se elige mirando los mismos tapes con los que luego se promedia, el estimando es `E_t[max_k X_{t,k}]`. Esto es distinto de seleccionar un calendario en una muestra A y evaluarlo en una muestra B: `E_B[X_{B,k*(A)}]`. El primero puede inflar el headroom por selección; la desigualdad `E[max] >= max E` describe la dirección, pero el tamaño del sesgo aún no está medido.

## Diseño corregido antes de gastar CPU

1. **Tapes A (selección):** evaluar los 65.536 calendarios y elegir `k* = argmax_k mean_A(X_{A,k})` por celda.
2. **Tapes B (evaluación):** congelar `k*` y evaluar solo `X_{B,k*}`; calcular `G_PI_split = mean_B(X_{B,k*}) - max_c mean_B(X_{B,c})`.
3. Reportar ambos como diagnósticos: `G_PI_naive` y `G_PI_split`, además de su diferencia de selección.
4. Fijar la regla de promoción en el preregistro: usar `G_PI_split`, no el ingenuo.
5. Si se desea conservar potencia, usar A/B dentro de los 128 tapes (p. ej. 64/64) solo si el contrato declara esa partición antes de ver resultados; mejor ampliar tapes vírgenes si el presupuesto lo permite.

## Gate propuesto

- Si `UCB95(G_PI_split) < 0.01`, cerrar la lane sin entrenar.
- Si `G_PI_naive - G_PI_split >= 0.01`, documentar que el diseño sin split habría sobreestimado la oportunidad.
- Si ambos pasan, abrir el smoke de aprendizaje con el estimando separado.

**No es una re-adjudicación:** el análisis solo corrige un diseño futuro; no toca O, O-R, Q ni el resultado Program O ya ejecutado.
