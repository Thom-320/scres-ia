# Enmienda — la potencia se fijó después del SESOI, y eso fue un defecto de diseño

**Fecha:** 2026-08-08. Enmienda a `docs/PREREGISTRO_VALIDACION_POSITIVA_AUDIT_2026-08-08.md`.
El artefacto de la primera corrida **se conserva** (`results/audit_positive_validation/result.json`,
`AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED`) y esta enmienda **no lo reescribe**.

## 1. Qué pasó

La mitad negativa validó de forma limpia y exacta. La positiva quedó **a 0,00087 de la barra**:
aprendiz menos mejor estructurado `+0,011477`, `LCB95 +0,009135`, contra un SESOI de `+0,01`.

## 2. El defecto es mío y es de diseño, no de resultado

**Congelé un SESOI sin comprobar que el diseño pudiera resolverlo.** Con `n = 60` la media anchura
del intervalo es `0,00235`, es decir **casi una cuarta parte del propio SESOI**. Un diseño así no
puede separar «no hay efecto de 0,01» de «hay uno y no lo veo», y de hecho no lo hace: el
`UCB95 = +0,01383` está **por encima** del SESOI. Por la regla que este proyecto ya tiene escrita —
la ausencia exige `UCB95 < δ`, y `LCB95 < δ` **no** es ausencia— el resultado correcto no es
negativo sino **irresoluble**.

Un preflight de potencia debió estar en el preregistro. No lo estuvo.

## 3. Qué se cambia, y qué NO

**Se cambia sólo el número de tapes.** La resolución objetivo se fija por un principio de diseño,
no por el resultado observado: **media anchura ≤ SESOI/10 = 0,001**. Con la desviación medida
(`0,00928`) eso exige `n ≥ 331`, y se fija **`n = 340`** por celda.

**No se cambia nada más.** Ni el SESOI, ni la barra de headroom, ni las celdas, ni la frontera
estructurada, ni el aprendiz, ni las reglas de lectura, ni los ocho falsadores. Elegir el `n`
mínimo que haría cruzar la barra sería exactamente la ingeniería del resultado que este banco
existe para hacer imposible; `SESOI/10` es una regla de resolución que no depende de dónde cayó la
estimación.

**Semillas nuevas y disjuntas:** `9100121–9100800` (340 de entrenamiento, 340 de test). Las
`9100001–9100120` quedan consumidas.

## 4. El brazo de divulgación no hacía lo que dije

Escribí que el MPC de modelo-oráculo impide vender como prima sobre optimalidad lo que es prima
sobre mala especificación. **Eso era falso y la corrida lo demostró:** el aprendiz le gana por
`+0,0103`. El oráculo lo es en *estimación de estado* —conoce `k` y `ρ`— pero su decisión es
**miope**, así que no es una cota superior de nada. Se conserva como referencia informativa y se
le retira la función de cota. Sigue faltando el óptimo decisional, y el documento lo dice en vez de
insinuar que lo tiene.

## 5. La lectura si vuelve a quedar corto

Sin cambio: `AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED`, y **no habrá una
tercera corrida sobre este banco**. Con la resolución en `SESOI/10`, quedar corto entonces sí
distinguiría ausencia de falta de potencia, y sería un resultado y no un empate.
