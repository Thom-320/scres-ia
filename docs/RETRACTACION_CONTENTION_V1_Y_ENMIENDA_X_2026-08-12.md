# Retractación — el `+0,0136` de `contention_v1` no existe, y con él cae la enmienda `d_min`

**Fecha:** 2026-08-12 · **Autor de los errores:** yo
**Detectado por:** cinco revisiones externas independientes sobre `9f712330`; verificado contra los
artefactos antes de aceptarlo.
**Artefacto de autoridad:** `results/audit_positive_validation/result.json`
**Retracta:** `docs/ENMIENDA_PROGRAM_X_PERMANENCIA_MINIMA_2026-08-10.md` y el contrato
`contracts/program_x_o_scale_amortized_control_v2.json` (commit `d33a6802`)
**No edita nada en sitio.** El contrato v2 y la enmienda se conservan y quedan etiquetados.

---

## 1. El número que cité no aparece en ningún artefacto

Durante toda la sesión del 2026-08-09/10 afirmé, en el briefing, en la respuesta al auditor y en la
enmienda a Program X:

> *en `contention_v1` el aprendiz batió al belief-MPC por **+0,0136 [LCB95 +0,0124]***

**Falso.** El artefacto dice, en la celda `positive`:

```
learner_vs_best_structured   mean +0,011477   lcb95 +0,009135   ucb95 +0,013831   51/60
SESOI                        0,010000
converts                     false
claim_status  AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED
```

La media supera el SESOI; **la cota inferior no lo cruza**. Por la regla congelada del propio
artefacto, la dirección positiva **no quedó demostrada**. Convertí un resultado que no pasó su gate
en la única victoria de control del proyecto, y lo hice el pilar de un contrato.

De dónde salió el `+0,0136`: de ninguna parte. No es el `ucb95` (0,013831 se le parece, pero el
intervalo que cité, `[+0,0124]`, no coincide con ningún campo). **Es un número que no puedo
reconstruir desde el artefacto, y eso lo hace peor, no mejor.**

## 2. Y la dirección del argumento se invierte

La enmienda `d_min` sostenía que la permanencia mínima era la **única** diferencia estructural, y el
**único** sitio donde un aprendiz ha ganado. Es falso por tres vías, todas leídas del mismo fichero:

**2.1 · El aprendiz también «gana» sin permanencia mínima.**

| celda | `min_dwell` | `rho` | learner − mejor estructurado |
|---|---|---|---|
| `positive` | **4** | 0,90 | +0,011477 [+0,009135, +0,013831], 51/60 |
| `no_memory` | **1** | 0,50 | +0,009066 [+0,007931, +0,010230], 58/60 |

Ninguna de las dos cruza el SESOI por abajo. La celda **sin** dwell tiene además **más tapas
favorables** (58/60 frente a 51/60).

**2.2 · Dwell y persistencia están confundidos.** Las dos celdas difieren en `min_dwell` **y** en
`rho` (0,90 vs 0,50). No existe ninguna celda `min_dwell=4, rho=0,5` ni `min_dwell=1, rho=0,9` que
los separe. Cualquier diferencia entre ellas es atribuible a la persistencia con la misma
legitimidad que a la permanencia. **Lo señaló un revisor y tiene razón.**

**2.3 · Contra el brazo de modelo verdadero, la ventaja es MAYOR sin dwell.**

```
learner_vs_oracle_model_mpc
  positive    mean +0,010323   lcb95 +0,007559   49/60
  no_memory   mean +0,019374   lcb95 +0,017430   60/60
```

Exactamente al revés de lo que mi enmienda necesitaba.

## 3. Mi propio gate `G4b` ya falla con los datos existentes

`G4b` exigía que, con `d_min > 1`, el filtro de primer orden rindiera **mediblemente peor** que el
filtro semi-Markov exacto. Es medible ahora, sin correr nada:

```
structured_means_test, celda positive
  oracle_model_mpc   0,906880
  belief_mpc         0,905726      diferencia  0,001154
```

**Un orden de magnitud por debajo del SESOI 0,01.** La mala especificación **no es material**, que
es precisamente la condición que `G4b` existía para comprobar. La rama de calidad de Program X v2
se cerraría en su propio gate.

Y en `no_memory` el filtro **correcto** es **peor** que el mal especificado (0,875682 frente a
0,885990). Eso sólo puede ocurrir porque la decisión es miope: una creencia mejor no garantiza una
acción miope mejor.

## 4. Los dos brazos llamados «MPC» no planifican

`supply_chain/contention_bench_v1.py`: tanto `belief_mpc_policy` (línea 153) como
`oracle_model_mpc_policy` (línea 174) terminan llamando a `_myopic_split`, un reparto de **un solo
periodo**. Ninguno tiene horizonte, rollout ni programación dinámica.

Es el mismo defecto que ya documentamos en `strong_mpc`, `techo` y `amortization_eligible`: **un
nombre puesto por el papel esperado, sin una medición que lo respalde** — y esta vez estaba en el
banco que yo mismo llamé «nuestro único contraejemplo», en código que leí durante la sesión y no
señalé.

## 5. Qué se retracta y qué sobrevive

**Se retracta:**

* el número `+0,0136 [+0,0124]` en todas sus apariciones;
* la frase «el único sitio donde un aprendiz batió a un belief-MPC»;
* la afirmación de que la permanencia mínima es la diferencia estructural decisiva;
* **la enmienda `d_min` completa** y el contrato v2 que la implementa.

**Sobrevive:**

* que `contention_v1` es un banco con headroom conocido por construcción — `H_PI = 0,156425`
  [LCB95 0,147734] en la celda `positive`, y **exactamente 0** en el nulo fungible. Eso es sólido y
  es lo que el banco fue construido para validar;
* que el aprendiz bate a su propio placebo (`+0,020038 [+0,018080]`), es decir, que usa la historia;
* el veredicto del propio artefacto: **la auditoría se detiene correctamente**, que era su objetivo.

## 6. Consecuencia para Program X

`contracts/program_x_o_scale_amortized_control_v1.json` **vuelve a ser el contrato vigente**. El v2
se conserva, etiquetado como superado por premisa fallida.

Esto **no** rehabilita la rama de calidad de v1: sigue siendo cierto que, con el HMM exacto conocido,
el posterior es estadística suficiente —lo dice el propio §7 de su preregistro— y por tanto v1 puede
producir un claim de **coste** y no de **calidad**. Lo que cae es mi propuesta de reparación, no el
diagnóstico de la limitación.

**La ruta al claim de calidad queda abierta y sin candidato.** Decirlo así es más honesto que
sostener un contrato apoyado en un número que no existe.

## 7. La regla que deja

Ya teníamos escrito que un falsador debe poder fallar y poder pasar. Faltaba ésta:

> **Un número citado en un documento debe ser reconstruible desde el campo exacto del artefacto que
> lo produce.** Si no se puede señalar el campo, no se cita.

Cité durante dos días un valor que no aparece en ningún `result.json` del árbol, y lo llevé hasta un
contrato congelado y hasta una respuesta enviada a un revisor externo. Ninguno de mis falsadores
podía detectarlo, porque ninguno mira los documentos: sólo miran las corridas.
