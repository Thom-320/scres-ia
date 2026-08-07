# Enmienda — se estrecha el alcance del veredicto agregado y se adjudica DDMRP con lo ya medido

Sucede a `docs/ENMIENDA_PASO3_GUARDARRAIL_INEXPRESABLE_2026-08-07.md`, cuyo diagnóstico
(`results/step3_expressiveness/result.json`,
`BOTH_STEP3_FALSIFIERS_FAIL_ON_DOMAIN_EXPRESSIVENESS_NOT_ON_A_DEFECT`) es la evidencia. **No se
edita ningún artefacto fechado.**

## 1. El alcance de `NO_STRUCTURED_CONTROLLER_CONVERTS`, estrechado por escrito

`results/step3_pooled/result.json` sigue vigente **con esta lectura y no otra**:

> Ningún controlador estructurado convierte **en el contrato agregado de un solo reclamante**,
> puntuado con `ret_excel_full_ledger`.

Lo que ya no se puede decir de él: que pasó el screen que su preregistro definió. **No lo pasó**,
porque `f4` exigía `worst_product_fill` y en ese contrato **no existe la dimensión**: 141 pedidos
por familia, `cssu_destination = None`, ningún atributo de producto. Un solo reclamante hace que
`worst_product_fill` sea `flow_fill_rate` por construcción.

**Lo que sostiene el veredicto igualmente:** la métrica es `ret_excel_full_ledger`, que puntúa a
**cero** todos los pedidos generados y no servidos. El abandono ya está pagado, así que el negativo
no puede deberse a un controlador que gane dejando de servir. Es un negativo estrecho y honesto,
no uno inflado.

El screen **con** el guardarraíl corre en paralelo bajo `docs/PREREGISTRO_PASO3_SPLIT_V1_2026-08-07.md`
y acompañará a éste; no lo supersede, porque `split_v1` reproduce `ret_excel_full_ledger` al
`0,000e+00`.

## 2. DDMRP queda adjudicado — y no hacía falta re-correr

`f6` del artefacto sellado falla con `n_distinct_postures = 1`, postura `[1344, 1344, 504]`, y su
propio texto teme lo peor:

> *"if the projection saturates, the arm emits ONE posture and the paired contrast against the best
> static is zero by construction. Then we are not measuring DDMRP, we are measuring 'saturate the
> buffers', and no claim about DDMRP is supported in either direction."*

**Esa disyuntiva ya estaba resuelta por una medición que existía.**
`results/buffer_saturation_diagnostic/result.json` (métrica `ret_excel_full_ledger`, multiplicadores
`×0, ×0,5, ×2, ×10` sobre la postura de referencia):

| nodo | `delta_up_at_10x` | `saturated_upward` | `has_downward_authority` |
|---|---:|---|---|
| `op3_rm` | **0,000000** | `true` | `true` (bajar a cero cuesta −1,81e−05) |
| `op5_rm` | **0,000000** | `true` | `true` (−1,25e−04 al bajar) |
| `op9_rations` | **0,000000** | `true` | `true` |

Y su propia retractación deja claro que el mecanismo **no** es que la escritura no funcione:

> *"An earlier reading of the bit-identical unprojected DDMRP was that the target write had no
> effect. It does: zeros give 0.1420 and 5,000,000 gives 0.2164 on the same tape. The write works;
> the system is saturated."*

**Consecuencia:** por encima del techo del dominio la métrica es **plana exactamente**. Por tanto la
postura proyectada de DDMRP es **equivalente en métrica** a su objetivo sin proyectar — que es lo
mismo que `results/ddmrp_unprojected_v1/` midió por el otro lado: **+1,02 M (R1r) / +1,27 M (R2r)**
unidades de más para un `ret_excel_full_ledger` **bit a bit idéntico**.

### La adjudicación, que es la que Garrido pidió

> **En el contrato del paso 3, DDMRP degenera observacionalmente a una postura constante situada en
> el techo del dominio, y ese techo cae en una región donde la métrica es exactamente plana. Su
> contraste pareado contra el mejor estático es cero, y eso es un hecho sobre DDMRP en esta cadena,
> no un artefacto del actuador.**

Se levanta la restricción `"supports NO claim about DDMRP"` **en esta dirección concreta**: sí
sostiene el negativo. Lo que sigue sin sostener es cualquier afirmación **positiva** sobre DDMRP,
que necesitaría una cadena cuyo techo esté por encima de su objetivo — y esta no lo está.

## 3. Lo que esta enmienda NO hace

* No supersede `results/step3_pooled/result.json` ni lo edita.
* No convierte el negativo en un screen preregistrado aprobado: para eso corre `split_v1`.
* No autoriza entrenar nada.
* No dice que DDMRP sea malo en general — dice que **esta cadena está saturada por encima de donde
  DDMRP apunta**, y que en una cadena saturada cualquier método que pida más buffer es
  indistinguible de una constante.

## 4. Por qué esto es un resultado y no una excusa

El registro de huecos pedía **cinco horas de cómputo** para persistir un campo. Comprobar el
supuesto costó dos minutos y descubrió que la dimensión no existía; y la adjudicación de DDMRP que
las cinco horas iban a producir **ya estaba medida en otro artefacto**. Las tres horas que sí se
gastan (`split_v1`) compran algo que ninguna de las dos rutas anteriores daba: **el guardarraíl
real, en un contrato que puede expresarlo**.
