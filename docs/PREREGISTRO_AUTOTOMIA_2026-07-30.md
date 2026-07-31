# Preregistro — la autotomía: el piso de `CTj` y el predicado de la rama

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Ninguna constante cambiada. Toda cifra congelada permanece como fue reportada.

---

## 1. El defecto, con causa medida

`autotomy_share = 0,000` en ambas familias, contra una referencia de 0,004 (`d_k` 11,2 en
R1r y 4,6 en R2r). Es la única brecha abierta cuya causa está localizada.

Nuestra rama de autotomía exige `CTj <= LTj` (`supply_chain.py:5940`). Medido:

    LT = 48    GARRIDO_FULFILLMENT_DELAY_HOURS = 54,0
    nuestro CTj: min = 54,00   p1 = 54,00   p50 = 54,00
    órdenes con CTj <= 48:  0 / 416

**Ninguna orden puede calificar jamás.** El piso lo fija una constante ajustada en
2026-06-26, documentada entonces como *«the smallest tested value that crosses the LT=48
cliff»* y etiquetada *«provisional reproduction default»*. Se eligió, literalmente, para
cruzar el acantilado que hace inalcanzable esta rama.

## 2. Lo que hacen sus datos, medido antes de declarar nada

21.667 filas, nueve hojas R1r.

**Su piso también está por encima de `LT`.** `min(CTj) = 48,0074`, y **0 filas** tienen
`CTj <= 48`. Su rama de autotomía dispara con `CTj > LTj`, no con `CTj <= LTj`.

**Y la separación es casi perfecta:**

| banda de `CTj` | n | `APj > 0` | `RPj > 0` |
|---|---:|---:|---:|
| [48,0074, 48,06] — el piso | 98 | **96** | 2 |
| (48,06, 54] | 2.594 | **0** | 2.592 |

Las 96 filas de autotomía están **todas** dentro del piso; `CTj − 48` toma nueve valores
distintos entre **0,0074 y 0,048**; ninguna tiene `RPj > 0` simultáneo. Inmediatamente
encima del piso no hay ni una sola autotomía entre 2.594 filas.

**Esto corrige un hallazgo previo.** El contrato `paper_b_independent_calibration_v2`
afirma que *«no band on CTj reproduces his classification»*, porque filas de no-autotomía
también empiezan en 0,00744. Es cierto que existen — son **2 de 98**. Una banda en 0,05
clasifica **96/98 = 98%** correctamente. El obstáculo era real pero se sobrestimó.

## 3. Los brazos

El arreglo tiene dos partes y **hay que separarlas**, porque una sin la otra no hace nada:

| brazo | piso (`delay`) | predicado de autotomía |
|---|---|---|
| **A** (statu quo) | 54,0 | `CTj <= LTj` |
| **B** (solo piso) | **48,0074** | `CTj <= LTj` |
| **C** (piso + banda) | **48,0074** | `CTj − LTj <= tol` |

`tol ∈ {0,01, 0,05, 0,10}`, **las tres se reportan, ninguna se elige por su resultado**.
`tol = 0,05` es la **primaria declarada**: es el valor más pequeño que cubre la banda
observada [0,0074, 0,048] sin extenderse más allá.

`LT = 48` **no se toca** en ningún brazo (tesis §6.8.2 p.111). El piso 48,0074 es
`min(CTj)` de sus hojas, no un valor barrido.

## 4. Predicción, en `d_k`, la misma escala que la regla

1. **Brazo B deja `autotomy_share` exactamente en 0,000.** Es una predicción dura: con
   `CTj = 48,0074 > 48`, el predicado `<=` sigue siendo falso. **Si B enciende la
   autotomía, la implementación está mal y la corrida se detiene.** B existe para aislar el
   efecto del piso del efecto del predicado, no porque espere que funcione.
2. **Brazo C mejora `d_k(autotomy_share)` en ambas familias.** Dirección declarada,
   magnitud no.
3. **Riesgo declarado, adverso, y afecta a los tres brazos con piso nuevo.** `ReT = 0,5/RPj`
   en la rama de recuperación, y `RPj ≈ CTj` para el grueso. Bajar el piso de 54 a 48,0074
   sube ese término **~12%** para la mayoría de las órdenes. `ret_mean` en R1r está hoy en
   0,007 contra una referencia de 0,006 — **ya por encima**, así que B y C lo empujan en la
   dirección equivocada. **Espero que `ret_mean` se degrade.** Por eso conserva veto.
4. **No predigo nada sobre `rpj_p95`.** El piso mueve el extremo bajo de la distribución, no
   la cola, y la saturación sigue sin mecanismo.

## 5. Falsadores del instrumento

1. **Brazo A reproduce lo congelado.** Sobre las raíces 2.400.001–12 debe dar
   `rpj_p95 = 2440,6` y `ret_mean = 0,007` de
   `results/metric_audit/rpj_onset_admission_v1/result.json`. Si no, no se reporta nada.
2. **Brazo B da `autotomy_share = 0,000` exacto** (§4.1).
3. **En B y C, `min(CTj) = 48,0074`** para las órdenes no bloqueadas, y **ninguna orden
   tiene `CTj < LT`**. Sus 21.667 filas cumplen esto último sin excepción.
4. **En C, ninguna orden con `APj > 0` tiene `RPj > 0`.** Sus 96 filas lo cumplen 96/96.

## 6. Criterio de aceptación

Dominancia sobre los seis momentos, referencia `fidelity_reference_v3`, `EPSILON = 0,5`,
**ambas familias objetivo** — el piso las afecta por igual, no hay control.

**C se adopta si y solo si:**

* `d_k(autotomy_share)` **mejora en ambas familias**; **y**
* `d_k(ret_mean)` **no empeora más de `EPSILON` en ninguna de las dos**; **y**
* ningún otro momento empeora más allá de `EPSILON` en ninguna familia; **y**
* los cuatro falsadores de §5 pasan.

**Si C enciende la autotomía y degrada `ret_mean`, no se adopta**, y el intercambio se
reporta como medido. Dado §4.3 ese es el desenlace más probable, y es publicable: diría que
el piso ajustado de 54 h y la magnitud de ReT de Garrido son mutuamente inconsistentes —
que es el mismo tipo de tensión que ya encontramos con el clamp, y empieza a ser un patrón
que el paper debe reportar en vez de resolver.

**Prohibido** elegir brazo o `tol` por el `H_PI` que produzca, por el signo de cualquier
contraste MPC-contra-estático, por que una familia cruce un umbral de servicio, o por que el
resultado sea publicable.

## 7. Lo que este preregistro reconoce como ajuste, y por qué es admisible

Dos números vienen de sus datos y no de la tesis: el piso **48,0074** (`min(CTj)`) y la
tolerancia **0,05** (cubre su banda observada). **Eso es un ajuste y lo digo.**

Es admisible por una razón concreta y limitada: ambos reproducen **su clasificación
publicada**, no un resultado favorable nuestro. La banda se juzga por cuántas de sus 98
filas del piso etiqueta igual que él (96), no por qué le hace a `ReT`. Y el barrido de `tol`
se reporta completo, sin selección.

**No es admisible extender esto.** Si `tol = 0,05` no reproduce su clasificación, la
respuesta es reportarlo, no ampliar la banda hasta que lo haga.

## 8. Declarado por adelantado

| ítem | valor |
|---|---|
| constante **no** tocada | `LEAD_TIME_PROMISE = 48` (tesis §6.8.2 p.111) |
| piso | 54,0 (A) contra 48,0074 (B, C) — de `min(CTj)` de sus hojas |
| `tol` | {0,01, **0,05**, 0,10}, las tres reportadas, primaria en negrita |
| raíces | **2.500.001–2.500.012**, disjuntas de todo bloque previo |
| raíces de regresión | 2.400.001–12, solo para el falsador 1 |
| familias | R1r y R2r, ambas objetivo |
| configuración | `S = 1`, buffers 0, nivel «+» de la Tabla 6.12 |
| modo RPj | `elapsed`, `procurement_delay_accumulation = serial`, `rpj_onset_admission = clamped` (los defaults) |
| criterio | dominancia seis-momentos, `EPSILON = 0,5`, `ret_mean` con veto |
| predicción | §4, en `d_k` |

## 9. Alcance

**Nada se reetiqueta.** Program Q, la confirmación H2/H3, el buffer gate, las 90
configuraciones y la frontera conjunta conservan sus cifras. Si C se adopta, **abre un cuerpo
de resultados nuevo**.

**Fuera de alcance:** `rpj_p95` y la saturación (sin mecanismo, cuatro hipótesis refutadas
hoy), `ret_above_one_share`, y el clamp (medido y no adoptado, `a0912bd`).

## 10. Firma

Requiere aprobación del PI antes de ejecutar.

La decisión que no me corresponde: si adoptar un piso derivado de sus datos aun cuando
degrade `ret_mean`. Mi lectura, y es solo eso: el 54 fue ajustado contra **un** observable y
rompe **otro** — el mismo patrón que este proyecto ya documentó seis veces — así que
reemplazarlo por `min(CTj)` de sus hojas es estrictamente mejor fundado. Pero si el precio
es el endpoint del manuscrito, esa es una decisión de proyecto.
