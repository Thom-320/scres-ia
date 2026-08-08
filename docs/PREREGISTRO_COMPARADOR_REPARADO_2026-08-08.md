# Preregistro — dos comparadores de reemplazo para el replay marginal

**Congelado:** 2026-08-08, antes de escribir una línea del runner y antes de mirar ningún resultado
de los brazos nuevos.

## Por qué

`results/comparator_drift/result.json` (`COMPARATOR_DRIFT_DIRECTIONAL_IN_4_OF_4_RESOLVED_IN_2_OF_4`)
estableció que el comparador llamado *state-blind marginal replay* tiene dos defectos:

1. **Contaminación del caso actual.** Las 24 visitas que el portador acaba de elegir sobre esta
   superficie entran al histograma antes de que el replay muestree de él. Cota exacta: 0,52 % de la
   masa en la primera evaluación, 0,18 % en la última.
2. **Acumulación entre casos.** El histograma nace con 4.608 pseudo-cuentas y termina con 8.640
   visitas reales, el 65 % de la masa. El comparador es prácticamente búsqueda aleatoria al principio
   y un histograma informado al final. Los cuatro brazos marginales son los cuatro más
   negativamente correlacionados con el orden de corrida de los doce; ningún brazo frío deriva.

Ninguno de los dos defectos se repara reinterpretando el artefacto sellado. Se reparan corriendo
comparadores que no los tengan.

## Los dos brazos

### A · `frozen_prior` — prior de niveles congelado ex ante

Durante la fase de entrenamiento sobre la rejilla de 288 —que ya ocurre y no se modifica— se cuentan
las visitas del portador **por nivel de factor**. Ese conteo se congela al terminar el entrenamiento,
**antes de tocar la rejilla extendida**, y se extiende a las 4.608 configuraciones por producto sobre
los factores, con los dos factores nuevos (`op3_rm`, `op5_rm`) **uniformes** — la misma regla con la
que `extend_state` extiende el estado del portador, para que la única diferencia entre brazos sea
qué se conserva y no cómo se extiende.

Propiedades, y son la razón de existir del brazo: no contiene el caso actual, no acumula durante la
evaluación, es idéntico para las seis contextos de una semilla, y **es desplegable sin correr el
portador sobre el caso objetivo**. Es, literalmente, un *prior de frecuencias de nivel transportable*.

### B · `loo_marginal` — el histograma acumulado menos el caso actual

Idéntico al comparador original salvo que la actualización `visits[...] += 1` ocurre **después** del
replay y no antes. Conserva la acumulación entre casos y elimina exactamente la contaminación del
caso en curso. Existe para separar los dos defectos: si `loo_marginal` se comporta como el original,
la contaminación del caso actual es irrelevante y el problema es la acumulación; si difiere, no.

Ambos se puntúan contra los mismos brazos `transfer` y `cold`, sobre las mismas semillas, con el
mismo presupuesto 24 y los mismos cachés.

## Grado — y esto no es negociable

Las semillas `8200001–8200060` están **quemadas**: este bloque ya se abrió para la confirmación. Por
tanto esta corrida es **REPLAY / DEVELOPMENT** y **no puede**:

* elevar el grado de RQ2a;
* sustituir la confirmación preregistrada;
* nombrar un ganador distinto.

Sólo puede **calificar** la interpretación de RQ2a y adjudicar la pregunta que el artefacto sellado
no puede responder. No quedan bloques vírgenes y no se abre ninguno.

## Estimandos

Por familia *f* ∈ {ucb1, neuron, gp, ofat}, pareado por semilla, menor AUC es mejor:

    δ_frozen(f) = AUC(frozen_prior_f) − AUC(transfer_f)
    δ_loo(f)    = AUC(loo_marginal_f) − AUC(transfer_f)

Primario: **δ_frozen(ucb1)**. Los demás son secundarios y descriptivos.

## Reglas de lectura, fijadas antes del dato

| resultado sobre δ_frozen(ucb1) | lectura autorizada |
|---|---|
| media > 0 y LCB95 > 0 | la ventaja de UCB1 **sobrevive contra un comparador transportable y desplegable**. Ese pasa a ser el comparador que encabeza RQ2 y la calificación de RQ2a se levanta. La afirmación «lo que transfiere es más que un prior de frecuencias de nivel» queda **sostenida en desarrollo**, nunca confirmada |
| intervalo cruza cero | **indistinguible de un prior de niveles congelado.** RQ2a se reporta sólo contra el comparador realmente implementado, con la calificación permanente, y la afirmación de que un prior de niveles basta queda **no refutada** |
| media < 0 y UCB95 < 0 | el prior congelado **gana**. Se reporta como tal: lo transportable es el histograma de niveles y el condicionamiento secuencial cuesta. Sería el resultado más interesante y el más incómodo |

Los tres se publican. Ninguno se convierte en «hace falta otra corrida».

## Falsadores, con su modo de fallo

| falsador | falla cuando |
|---|---|
| `f1_transfer_and_cold_reproduce_the_sealed_values_exactly` | añadir brazos perturbó los existentes; cada brazo usa su propio `default_rng`, así que un cambio aquí significa que los flujos se contaminaron |
| `f2_frozen_prior_does_not_drift_with_run_order` | el prior congelado deriva; si lo hace, no está congelado y el brazo no es lo que dice ser. **Puede fallar** |
| `f3_loo_still_drifts_with_run_order` | quitar el caso actual eliminó la deriva; entonces el mecanismo diagnosticado (acumulación entre casos) era el equivocado. **Puede fallar y refutaría el diagnóstico de ayer** |
| `f4_loo_differs_from_the_original_by_no_more_than_the_mass_bound` | la diferencia excede lo que 0,18–0,52 % de la masa puede explicar; entonces la aritmética de contaminación está mal |
| `f5_frozen_prior_puts_mass_on_every_extended_configuration` | el prior deja las 4.320 configuraciones nuevas en cero y el replay no puede alcanzarlas — sería un comparador tullido, no uno justo |
| `f6_budgets_are_matched` | algún brazo nuevo no gasta exactamente 24 evaluaciones por contexto y semilla |
| `f7_no_seed_outside_the_burned_block` | se tocó una semilla fuera de `8200001–8200060` |

`f2` y `f3` pueden fallar en direcciones opuestas y **el preregistro no se salva de ninguna**: si
`f3` falla, el diagnóstico de deriva de ayer estaba mal y hay que decirlo.

## Coste

La corrida original tardó 63,4 min con cuatro brazos por familia. Dos brazos más por familia son
~+50 % de replay sobre caché. Se lanza cuando la superficie extendida libere núcleos; no compite con
ella.

## Lo que este preregistro no autoriza

Ni entrenar, ni abrir semillas, ni añadir familias, ni cambiar presupuesto, normalizador, orden de
contextos o física. Un resultado adverso **no** habilita una tercera versión del comparador.
