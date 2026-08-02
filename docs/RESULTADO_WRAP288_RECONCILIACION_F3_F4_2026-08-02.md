# Resultado — WRAP-288 reconciliado bajo el contrato v2, con un `f3` que ya puede fallar

**Artefacto:** `results/garrido_q2_des288_reconciled_v2/result.json` (sello `67473d2590f03ee6…`)
· contrato **v2** `docs/PREREGISTRO_META_APRENDIZ_V2_2026-08-01.md` (`91614d39…`)
· `audit_status = RUNTIME_F3_F4_RECONCILIATION_NOT_A_NEW_CONFIRMATION`
· `replay_of = garrido_q2_des288`, bloque `5.300.001–12` · **ninguna semilla nueva**.

## 1. Qué queda reconciliado

El artefacto DES-288 existía y se reproducía, pero **nunca había pasado los checks `f3`/`f4` del
contrato v2**: estaba sellado contra el preregistro anterior. Ahora sí, **re-ejecutando** en vez de
inspeccionando — la reconciliación post hoc era imposible porque el artefacto viejo sólo conserva
`surface_sha256` y las secuencias visitadas, no la superficie de valores que `f4` necesita
sustituir.

| | |
|---|---|
| reproducción vs el artefacto original | **14 de 14 cantidades, 0 difieren** |
| los seis falsadores | **cinco PASAN**, `f6` **NO APLICA** (`DECLARED_REPLAY`) |
| efecto Alzheimer, **sellado** | **+7,9028** [+6,8750 · +8,9306] |
| memoria vs OFAT | +5,43 [+4,01 · +6,78] |
| memoria vs aleatorio | +12,56 [+10,65 · +14,56] |

## 2. El defecto que había que arreglar antes, y por qué la primera corrida se mató

El control negativo de `f3` era:

```python
f3_negative_control_detected = memory_arm_contract != dict(reset_arm_contract, budget=budget+1)
```

Los dos contratos **ya difieren en `rho_policy`** (`carry` frente a `reset`), así que ese `!=` era
**verdadero por construcción** y lo seguía siendo con el presupuesto **idéntico**. No comprobaba
que una diferencia de presupuesto fuera detectada: comprobaba que dos diccionarios distintos son
distintos. **Tautológico**, la misma forma que el `passed: True` cableado que dejó pasar una fuga
real en julio.

**La corrida en vuelo se mató en vez de dejarla terminar.** Un artefacto sellado con un falsador
que no puede fallar tiene apariencia de validez y ningún contenido.

## 3. El arreglo, y la demostración de que ahora sí puede fallar

El control alimenta **el mismo comparador** con un brazo de presupuesto manipulado y exige que lo
**rechace**:

| | manipulado | sin manipular |
|---|---|---|
| forma vieja | `True` | **`True`** ← tautológico |
| **forma nueva** | `True` | `False` ← distingue |

Y con el defecto que vigila inyectado —sacar `budget` de las claves comparadas— el control devuelve
`False` y **`f3` falla**. Se añadió además un segundo control: el checker de forma de traza recibe
una `visited_sequence` con una visita de más y debe rechazarla.

En el artefacto quedan sellados los tres, por separado:

```
negative_control_detected  True
budget_tamper_rejected     True
trace_tamper_rejected      True
```

**Registrar los sub-controles importa**: un único booleano agregado no habría dejado ver cuál de
los dos mutantes se caza.

## 4. Una acusación mía que retiro

Dije que los `f3`/`f4` viejos «eran prosa». **Demasiado duro y no exacto.** El `f3` viejo **era
estructural** —registraba semillas compartidas y orden de contextos— y el `f4` viejo **sí
ejecutaba** una búsqueda con presupuesto cero. Lo que les faltaba era el **control negativo del
contrato v2**, que es una acusación más estrecha y correcta.

## 5. Lo que esto NO es

* **No es una confirmación nueva.** El `audit_status` lo dice: es una reconciliación en tiempo de
  ejecución sobre un bloque **ya usado**, con `f6 = DECLARED_REPLAY`.
* **No convierte el DES-288 en cobertura contractual completa de Q2.** Reconcilia `f3`/`f4`; el
  resto del contrato v2 sigue siendo lo que sea que diga, y el bloque `5.300.001–12` sigue con
  `n = 12`, que es la corrida **exploratoria**, no una campaña con potencia.
* **El efecto Alzheimer sigue sin entrar al manuscrito por esta vía.** Aquí vale como valor de
  desarrollo reconciliado; el citable con potencia es el de H3′ a `n = 120`.

## 6. Estado del bloque

`garrido_q2_des288` pasa de `ARTIFACT_PRESENT_PENDING_CANONICAL_CUSTODY` a **reconciliado en
`f3`/`f4` bajo el contrato v2**, y sigue siendo desarrollo.
