# Preflight G3c sobre la rejilla viva — el instrumento aguanta, y el candidato no gana

**Contrato:** `contracts/g3c_burned_preflight_v2.json` (`{1, 6, 12}`, derivada de
`results/headroom/g3c_dwell_inertia/result.json`). **Semillas:** bloque quemado `5.200.001–16`,
réplica declarada. **Artefacto:** `results/headroom/g3c_preflight_grid_v2/result.json`.

**Estado: `STOP_G3C_GUARDRAIL`.** Ocho falsadores pasan, `f8` **no aplica** (réplica declarada, no
«pasa»), y **`f9` falla**.

## 1. Esta vez el mecanismo sí ata

La rejilla anterior murió porque `dwell=3` no retenía ni una acción. La nueva se derivó de una
medición con un criterio doble —retener en **todas** las cintas **y** suprimir ≥ 10 % de las
conmutaciones—, y `f2` pasa en los cuatro contrastes. **La restricción existe.**

## 2. El contraste primario, con potencia de sobra

`histéresis − miope` en `worst_claimant_fill`, pareado por semilla:

| celda | media | LCB95 | UCB95 | `n*` |
|---|---:|---:|---:|---:|
| base · dwell=6 | −0,00157 | −0,00571 | +0,00282 | 11 |
| base · dwell=12 | **−0,00469** | **−0,00868** | **−0,00097** | 10 |
| freq3 · dwell=6 | +0,00084 | −0,00305 | +0,00589 | 12 |
| freq3 · dwell=12 | −0,00121 | −0,00656 | +0,00498 | 18 |

**El candidato no gana en ninguna celda**, y en la más restrictiva del régimen base **pierde con el
intervalo entero por debajo de cero**. El SESOI es `+0,010`; el mejor punto estimado es `+0,0008`,
**doce veces por debajo**.

Y la potencia **no es la excusa**: `n*` va de 10 a 18 contra un presupuesto de 96. Con la rejilla
viva, **el diseño tiene potencia y aun así no hay nada que detectar**.

## 3. Por qué falla `f9`

Daño pareado contra el incumbente, `UCB95(daño) ≤ δ`:

| celda | `backorder_qty_final` relativo | UCB95 | δ |
|---|---:|---:|---:|
| base · dwell=6 | +0,00699 | **+0,01164** | 0,010 |
| base · dwell=12 | +0,00574 | **+0,01028** | 0,010 |
| freq3 · ambos | ≈ 0 | ≤ +0,0034 | 0,010 |

`flow_fill_rate` y `lost_orders` pasan holgadamente en las cuatro. Lo que falla es **el backlog
final en el régimen base**: la histéresis, al sostener el reparto, **deja más pedido pendiente al
cierre del horizonte**. Es pequeño y roza el margen —los UCB95 lo cruzan por 1,6 y 0,3 milésimas—
pero el margen estaba firmado antes, y cruzarlo por poco sigue siendo cruzarlo.

## 4. Qué queda dicho

**La conjetura que justificó reabrir G3c no encuentra apoyo.** El argumento era que la permanencia
mínima saca al incumbente miope de su clase de optimalidad; **con la restricción realmente
mordiendo, la regla miope sigue ganando**, y el compromiso intertemporal se paga en backlog.

Con cuidado sobre lo que esto **no** es: un preflight sobre tapes quemados **no adjudica**, y un
`STOP` por guardarraíl no es una refutación de la clase. Lo defendible es lo estrecho:

> Sobre 16 cintas quemadas, con la rejilla derivada de la inercia medida y potencia suficiente,
> **una histéresis de dos estados no supera a la regla miope bajo permanencia mínima, y viola el
> margen de backlog en el régimen base.**

**Ninguna semilla nueva. Ninguna adjudicación. Ningún learner.** El contrato sigue en
`DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT` para raíces frescas, y ahora hay menos
razón que ayer para gastarlas aquí.

## 5. Defecto reparado de paso

El runner canónico se caía con `NameError` **exactamente en la rama de fallo** —`f1` sin definir en
el selector de veredicto—, así que el camino que importa era el único no ejercitado. Corregido, y
es lo que permitió que este `STOP` se sellara en vez de reventar.
