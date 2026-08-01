# Resultado — DETENIDO por `f7`, con una señal grande y una cualificación que la reduce

**Artefacto:** `results/headroom/contention_policy_class/result.json` (sello `158f199cbd98eb60…`)
· preregistro `docs/PREREGISTRO_REAUDITORIA_CLASE_DE_POLITICA_2026-08-01.md` (commiteado antes)
· semillas **quemadas** `5.200.001–16`, **ninguna nueva** · `claim_status =
HALTED_FALSIFIER_FAILED`.

> **Nada de lo que sigue se promueve.** `f7` falló, la regla dice que nada se promueve y todo se
> registra, y esto es el registro.

## 1. La señal, en las cuatro celdas

Endpoint primario escalar `worst_claimant_fill`; contraste primario **clarividente − placebo no
informado**, que es el único que aísla el estado.

| celda | mejor constante | **valor incremental del estado** | LCB95 |
|---|---:|---:|---:|
| `FIFO_PARTIAL \| base` | 0,5 → 0,7868 | **+0,0372** | +0,0212 |
| `FIFO_PARTIAL \| freq3_imp2` | 0,5 → 0,7833 | **+0,0406** | +0,0267 |
| `R24_AGE_PARTIAL \| base` | 0,5 → 0,7868 | **+0,0362** | +0,0211 |
| `R24_AGE_PARTIAL \| freq3_imp2` | 0,5 → 0,7833 | **+0,0402** | +0,0264 |

**El placebo no informado pierde contra la constante en las cuatro celdas** (−0,011 a −0,013), así
que variar por variar **no** es la explicación. Y apuntar al reclamante **equivocado** cuesta
**−0,62**: la dirección importa, no sólo la cadencia.

**Mecanismo medido:** el reclamante estresado **alterna de forma persistente** —A domina los
primeros meses del episodio, B después—, porque el objetivo de cada riesgo se sortea por evento.
El barrido sellado probó **sólo constantes** (su propio docstring lo dice). **Una constante no
puede ser 0,9 y 0,1 a la vez**, así que el valor equivariante se cancelaba en la agregación
**antes de que la física tuviera ocasión de mostrarse**.

## 2. La cualificación que reduce el hallazgo, y es lo más importante del documento

**El fill agregado no se mueve**: `flow_fill_rate` 0,7962 (clarividente) vs 0,7955 (placebo) vs
~0,796 en todas partes. **La ganancia es REDISTRIBUTIVA, no productiva.** La política no crea
servicio: se lo pasa al reclamante que va peor.

Eso es exactamente lo que un endpoint max-min premia, así que el resultado **no puede venderse
como «más resiliencia»**. La afirmación defendible es más estrecha:

> Bajo un objetivo que protege al peor reclamante, conocer **cuál** está estresado permite una
> reasignación que una constante no puede expresar. El sistema no entrega más raciones; las
> reparte mejor.

## 3. Por qué se detuvo, y por qué NO voy a arreglarlo a posteriori

`f7_no_gain_by_abandonment` compara `lost_orders` del clarividente contra el placebo **a margen
cero sobre estimados puntuales**:

| celda | clarividente | placebo | diferencia |
|---|---:|---:|---:|
| `FIFO \| base` | 1,4375 | 1,3750 | **+0,0625** |
| `R24_AGE \| base` | 2,4375 | 2,3750 | **+0,0625** |
| ambas `freq3_imp2` | 0,5000 / 1,0625 | 0,5000 / 1,1875 | 0 / **−0,125** |

**0,0625 sobre 16 semillas es UN pedido en UNA semilla.** Es ruido Monte Carlo, y las revisiones
externas lo habían nombrado **de forma prospectiva**:

> *«Exigir no deterioro estadístico a margen cero es una prueba de superioridad disfrazada.»*

**Lo advirtieron, y yo envié el instrumento defectuoso igualmente.** Ése es el hallazgo de
proceso, y me lo apunto.

**Lo que NO hago:** ni aflojar el margen, ni cambiar el comparador de `f7` a la constante, ni
re-correr hasta que pase. Un falsador que se reajusta después de ver su resultado no falsa nada.
El artefacto queda `HALTED`, y un margen de no inferioridad **con potencia** se preregistra en un
contrato sucesor, **antes** de correr.

## 4. Y una laguna que declaro en vez de taparla

`f3b`: cinco revisiones pidieron una prueba de equivarianza A↔B. **No es construible en
`split_v1`** — el destino es un bit de hash (`digest[0] & 1`) y el objetivo del riesgo un
`rng.choice(("A","B"))` sin pesos. Con etiquetas indistinguibles, la equivarianza es cierta por
construcción y **no transporta información**. Queda registrada como
`NOT_EXPRESSIBLE_IN_split_v1_DEFERRED_TO_G3A`, no como un falsador aprobado.

## 5. Lo que esto cambia para G3a

El diagnóstico de G3a cambia de sujeto, y a mi costa:

* **Mi hipótesis de simetría, en su forma fuerte, ya estaba refutada** por las revisiones
  (simetría ⇒ equivarianza, no ⇒ `argmax = 0,5`).
* Ahora se mide algo distinto: **la restricción vinculante era la CLASE DE POLÍTICA**, no la
  simetría de la física. El nulo `H_regime = 1,5e-04` es correcto **para constantes** y no dice
  nada sobre políticas equivariantes.
* Por tanto **G3a debería probar acoplamiento temporal y estado, no asimetría de demanda** — la
  asimetría no era el ingrediente que faltaba. Eso **adelanta G3c** por delante de G3a.

**Pero nada de esto está confirmado**: es desarrollo, detenido, sobre tapes quemados, con un techo
clarividente que **no es un headroom desplegable** y que **no autoriza entrenar nada**. La
gobernanza de `authority_ladder_v1` sigue intacta: `f5` verifica que no se abrió una sola semilla.
