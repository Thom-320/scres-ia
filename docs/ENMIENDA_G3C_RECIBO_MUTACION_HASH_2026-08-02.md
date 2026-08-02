# Enmienda G3c — recibo reproducible del falsador del hash dorado

**Estado:** `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`
**Contrato padre:** `contracts/g3c_temporal_coupling_v2.json`
**Recibo machine-readable:** `contracts/g3c_mutation_receipt_v1.json`

Esta enmienda no abre semillas ni modifica los resultados de ningún experimento. Formaliza la
evidencia de que el falsador del brazo nulo puede detectar un defecto en el camino de producción,
no sólo una mutación de un diccionario ya construido.

## Fixture congelada

El brazo nulo usa el tape de prueba quemado `5.200.001`, horizonte de ocho semanas, pasos de 24
horas, solicitudes alternadas `0,9/0,1`, `min_dwell_days=1` y coste de cambio cero. Su hash de
payload científico es:

```text
be9d1bc227d498cb093f654014b791066ea945ad5c71cfc7cf74b2d9a4df9c37
```

El hash incluye órdenes, eventos de riesgo, acciones, ledgers y métricas; no incluye metadata de
provenance.

## Mutación ejecutable

`test_runtime_activation_drift_is_caught_by_the_golden_hash` inyecta, mediante `monkeypatch`, una
deriva deliberada de `+0,001` en `cssu_allocation_a` después de cada activación real. El test exige
que el hash mutado difiera del fixture congelado. La mutación no abre una semilla científica y se
ejecuta únicamente sobre el fixture unitario quemado.

Esto prueba el límite correcto: el guardarraíl detecta una deriva física en producción. No prueba
potencia, headroom, prima neural ni autorización para ejecutar G3c.

## Regla de re-fijación

Si cambia deliberadamente la física nula, primero debe emitirse otra enmienda con la razón, el
nuevo fixture y su hash. No se actualiza el hash dorado silenciosamente.
