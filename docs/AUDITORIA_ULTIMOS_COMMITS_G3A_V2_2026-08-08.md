# Auditoría de los últimos commits G3a-v2

**Artefacto:** `results/g3a_boundary_v2/result.full34.json`

## Opinión ejecutiva

El negativo es científicamente informativo: al restaurar los 34 controladores, la celda
persistente/uniforme con cuota rígida conserva `H_obs=+0,002789`, IC95
`[−0,007635,+0,012373]`; el mejor placebo no pierde y FIFO global permanece exactamente nulo. El
recorte inicial a 14 controladores ya no explica la no reproducción.

No debe describirse todavía como un paquete contractual completo. La auditoría estática encontró:

1. el preregistro enumera `f1–f9`, pero el runner/result omite `f3_mass_conserves` y
   `f5_belief_uses_one_common_model`;
2. `f9_forfeiture_is_measured` exige `forfeited >= 0`, condición que también pasa si el atributo
   falta y el `getattr(..., 0.0)` devuelve cero;
3. se conservan los 30 outcomes held-out por controlador y celda, pero sólo la media de selección,
   no las filas por seed de selección;
4. el run de 34 brazos reutiliza exactamente las 60 seeds del run de 14 brazos. Es una enmienda de
   alcance válida, no una réplica independiente.

## Etiqueta de lectura

```text
HEADROOM_NEGATIVE_STANDS__CONTRACT_COMPLIANCE_INCOMPLETE
```

Esto no reabre la asignación A/B como lane de aprendizaje. Sí exige que cualquier paquete C&IE
regenere filas completas y materialice masa, modelo de creencia y forfeiture con falsadores que
puedan fallar.
