# Preregistro — Program V sobre el DES completo

**Fecha:** 2026-08-09. **Congelado antes de escribir el runner.**
**Rol:** `DEVELOPMENT_PORT_NO_LEARNER_AUTHORIZED`.

## 1. Qué se porta y por qué

Program V identificó, en una abstracción de 780 filas, el primer mecanismo de **memoria causal
falsable** del proyecto: comprometer una bolsa fija entre tres proveedores una semana antes de la
entrega, bajo degradación latente persistente, aviso imperfecto y yields parciales. `H_ret` (Bayes
retenido − reset) dio **+0,0413 [+0,0266, +0,0561]**.

La abstracción no es el DES. Tiene inventario escalar, una demanda semanal y ninguna de las trece
operaciones, riesgos, colas ni transporte de la MFSC. Portarlo responde la única pregunta que la
abstracción no puede: **¿sobrevive el valor de la historia retenida cuando la señal tiene que
atravesar Op1–Op13, riesgos recurrentes y las colas reales?**

## 2. Qué se conserva idéntico, a propósito

**Las mismas tapes y las mismas trece políticas**, importadas del módulo de la abstracción, no
reimplementadas. Si se reescribieran, una diferencia de resultado sería inseparable de una
diferencia de código. Esto convierte el port en una comparación limpia: **misma decisión, misma
información, misma historia; física distinta.**

## 3. La física añadida al DES, declarada como extensión nuestra

`supplier_portfolio_mode="v1"`, inerte por defecto. En cada ciclo de entrega de Op2 el volumen
contratado se multiplica por `Σ_i alloc_i · yield_i` de la semana en curso. Lo que no llega
**nunca entró al sistema**: se contabiliza como rechazo del proveedor y jamás como inventario
destruido — la distinción que ya nos costó una retractación hoy.

* **Compromiso con una semana de antelación.** La asignación que gobierna la semana `t` se escribe
  en `t−1` y no se puede mover después de observar el yield.
* **Yields exógenos con clave de hash**, no del RNG del simulador, para que la misma tape conserve
  los mismos riesgos y el mismo orden de extracciones bajo cualquier política.
* La tesis no modela una cartera adaptativa de proveedores; declara disponibilidad dada y
  colocación instantánea. Nada de esto se atribuye a Garrido-Ríos (2017).

## 4. Endpoint

**Fill rate de teatro** (`entregado / demandado`) sobre el ledger real del DES — la misma cantidad
que la abstracción llamó `service`, ahora medida a través de la red completa. Secundarios: backlog
AUC, unidades rechazadas por proveedor, residual de masa.

**No se usa `ret_excel`.** Está medido que premia el abandono, y un endpoint que recompensa no
servir no puede arbitrar una decisión de aprovisionamiento.

## 5. Falsadores, y por qué cada uno puede fallar

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_portfolio_is_inert_by_default` | con `mode="none"` el resultado es idéntico al DES congelado | si no lo es, la extensión cambió la física base |
| `f2_allocation_moves_arrivals` | cambiar la asignación cambia lo recibido | una acción inerte haría ruido de todo lo demás |
| `f3_rejected_is_never_destroyed_stock` | rechazado = pedido − recibido, y el ledger de masa cierra | es exactamente el defecto retractado esta mañana |
| `f4_same_tape_same_risks` | recuento de pedidos idéntico entre políticas | los yields no deben consumir RNG |
| `f5_commitment_lead_binds` | la asignación de la semana `t` se fijó en `t−1` | sin lead, la decisión ve el yield que debía anticipar |
| `f6_H_priv_material` | `LCB95 ≥ 0,02` privilegiado vs mejor constante | **puede fallar**: el DES puede absorber la señal |
| `f7_H_obs_material` | `LCB95 ≥ 0,01` mejor observable vs mejor constante | puede fallar aunque `f6` pase |
| `f8_H_ret_positive` | `LCB95 > 0` retenido menos reset | **la pregunta del port**; puede fallar y ése sería el resultado |
| `f9_retained_beats_both_placebos` | vence a retardado y barajado | si empata, lo medido es cadencia |

## 6. Reglas de lectura, fijadas de antemano

1. **Primero la inercia y la masa.** Si `f1`, `f3` o `f4` fallan → `BLOCKED_INSTRUMENT`.
2. `f8` es la pregunta. Si el valor de la historia retenida **no** sobrevive al DES completo, el
   veredicto es `RETAINED_VALUE_DID_NOT_SURVIVE_THE_FULL_DES`, y eso **acota** el hallazgo de la
   abstracción en vez de anularlo: seguiría siendo cierto en su propio contrato.
3. Si `f8` pasa, el estado es `RETAINED_VALUE_SURVIVES_THE_FULL_DES`, y aun así **no autoriza
   entrenar**: el residuo del privilegiado sobre Bayes se reporta, y en la abstracción era
   `+0,00076` con UCB95 `+0,0023`.

## 7. Semillas

Bloque de desarrollo ya quemado `8600001–8600060`, 30 selección / 30 evaluación. **No se abre
ningún bloque virgen.** No hay grado confirmatorio posible aquí y no se reclamará.
