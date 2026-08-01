# Preregistro — liveness CSSU en dos gates

**Contrato:** `garrido_wrap_scres_ai_v1`
**Estado:** `GATE_A_PASS_GATE_B_HOLD`
**Propósito:** separar la liveness de la interfaz de acción de cualquier afirmación sobre
capacidad o tiempo físico de Op11.

## Gate A — interfaz de reasignación

La acción pública es:

```text
{"cssu_allocation_a": alpha, "cssu_service_rule": rule}
```

El gate pasa únicamente si, bajo `cssu_topology_mode="split_v1"`:

1. `alpha` se valida en `[0, 1]` y se puede cambiar fuera de la rejilla histórica;
2. la acción queda programada con la latencia contractual de 24 horas;
3. la acción se activa después de la latencia y no antes;
4. dos acciones distintas producen distintos libros de despacho por CSSU;
5. la demanda y el despacho conservan masa;
6. el modo `aggregate` rechaza la acción;
7. la observación no contiene riesgo futuro ni duración futura;
8. una CSSU caída no puede ser atravesada por la acción.

El alcance del gate es únicamente:

> La reasignación de una capacidad compartida en el DES split es una acción computacionalmente
> viva.

No demuestra manejo finito en Op11.

## Gate B — física de Op11

Estado preregistrado: `HOLD_OP11_PHYSICS_UNSPECIFIED`.

No se fijan retrospectivamente una distribución, una capacidad por orden, una capacidad por
lote ni un tiempo exacto a partir de la frase de la tesis “less than one hour”. Un nuevo contrato
será obligatorio antes de implementar esta extensión.

El nuevo contrato deberá especificar:

- unidad de servicio y capacidad finita;
- tiempo de manejo y su distribución;
- conexión con el camino `op9_linked`;
- interacción con R23 y acciones CSSU;
- conservación de stock, tránsito y backorders;
- precio de fidelidad frente a `thesis_1to1`.

El gate B sólo podrá pasar si un falsador con manejo cero detecta una diferencia, la acción no
atraviesa una CSSU caída, no se crea inventario y el carril agregado permanece bitwise separado.

## Regla de publicación

Los resultados de contención pueden usar el gate A para hablar de una acción viva. Sólo un gate B
posterior permitiría describir la intervención como contención física de manejo en Op11.
