# Enmienda instrumental v2 — reloj operativo y R24 bajo CRN natural

**Escrita después del humo v1 negativo y antes de ejecutar v2.** La v1 se conserva en
`results/garrido_v0_recovery_gate_v1/smoke_result.json` con
`STOP_V0_RECOVERY_SURFACE_GATE_FAILED`. Esta enmienda no cambia duraciones, magnitudes, posturas,
ventana de recuperación ni umbrales de respuesta.

## Dos defectos causales observados

1. El evento común en la semana 8 cae entre ciclos de Op1 y Op2. Op1 usa `op1_rop = 4.032 h` y
   Op2 `op2_rop = 672 h`; bajar Op1/Op2 cuando ninguno inicia trabajo hace que R12/R13 sean
   decorativos aunque su duración sea correcta.
2. `demand_source="excel_order_tape"` pone `_contingent_demand_pending = 0` antes de cada pedido,
   porque presupone que la cinta ya incluye R24. La cinta del gate fue generada **sin** R24, de
   modo que el pulso se borraba y R24 no podía fallar ni pasar.

Estos son defectos de soporte y replay, no resultados desfavorables. Aumentar magnitudes para
rescatarlos queda prohibido.

## Correcciones congeladas

- Todos los eventos comienzan en `4.031 h`, una hora antes del ciclo común de Op1/Op2; horizonte
  36 semanas. `tau = 8 semanas` permanece intacto.
- Se usa demanda natural con `strict_exogenous_crn=True`. El RNG de demanda está separado del RNG
  general/riesgos; el runner persiste el hash de la cinta de pedidos realmente generada y exige
  identidad entre las seis posturas dentro de cada `(semilla, riesgo)`. En R24 la demanda puede
  diferir del placebo —esa diferencia es el mecanismo—, pero debe ser idéntica entre posturas.
- El gate de impacto deja de usar la **mediana entre posturas**. Esa regla declaraba «no hay
  choque» precisamente cuando las posturas altas lo absorbían. Un contexto es físicamente vivo si
  al menos 25 % de sus celdas centinela muestran pérdida incremental y existe respuesta entre
  posturas. El umbral global sigue siendo 6 de 8 riesgos.

## Lo que no cambia

- ocho riesgos R11–R24;
- medianas empíricas redondeadas ya congeladas;
- seis posturas centinela;
- `RTTR_tau = min(TTR, tau)`, con cero sólo para absorción y `tau` para censura;
- replay de semillas quemadas `5.300.001–002`;
- respuesta de postura: 24 h en RTTR o 0,01 de AUC incremental normalizado;
- al menos 25 % de celdas impactadas recuperan dentro de `tau`;
- si v2 falla, no se vuelve a mover el reloj, la demanda, los riesgos ni los umbrales mirando los
  resultados.
