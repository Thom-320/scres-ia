# Enmienda — la Puerta B se adjudica contra el mejor comparador NO neuronal

**Fecha:** 2026-08-09 · **Autoriza:** PI · **Alcance:** Programa N, Puerta B (las cuatro corridas)
**Enmienda a:** `docs/PREREGISTRO_PUERTA_B_SUPERFICIE_CD_2026-08-09.md`
**Semillas abiertas: ninguna.** Es re-adjudicación de `per_fold` ya sellados.
**No edita ningún artefacto sellado.** Los cuatro se conservan tal cual; esta corrida los cita.

## El defecto

El contrato marco dice, y lo dice desde la Fase 0:

> *la red debe batir al **mejor comparador NO neuronal**, no a la constante*

La Puerta A2 lo implementa (`vs_best_nonneural`) y por eso pudo matar a la Puerta A. **La Puerta B
nunca lo implementó**: su `f5` compara sólo contra el baseline primario `linear_interactions`.

Lo destapó la sensibilidad sobre `ret_excel`: un **árbol de regresión** pasa el mismo criterio que
el KAN y por margen mayor (+0,0705 vs +0,0676). El veredicto impreso,
`SENSITIVITY_PREMIUM_HOLDS_ON_LEGACY_SURFACE`, describe entonces que el primario no es el mejor
clásico de esa superficie — no que exista prima neural.

Es defecto de instrumento mío, heredado del preregistro de B.

## La partición, declarada ANTES de calcular

La regla es por **conjunto de información**, no por conveniencia. Un brazo sólo compite contra
comparadores que ven lo mismo que él.

**Clase A — features de configuración** (`buffer`, `family`, `escalation`):

* neuronales: `mlp_tuned`, `kan_tuned`
* no neuronales: `constant`, `linear_additive`, `linear_interactions`, `spline_buffer`, `tree`

**Clase B — además la resiliencia de la configuración x−1** (la activación de la Fig. 5):

* neuronal: `recurrent`
* no neuronal: `linear_lagged`

**`train_cell_mean_comparator` queda FUERA de ambas clases y no puede ganar nada.** Usa la
identidad de celda `(familia, escalación, buffer)`, que ningún otro brazo recibe. Se declaró como
techo y como techo se reporta. Incluirlo entre los comparadores sería cambiarle el papel después de
ver que en `ret_excel` encabeza; excluirlo de los brazos juzgados sería regalarle la victoria a la
red. No se hace ninguna de las dos: se reporta aparte, y se reporta también que en `ret_excel`
**dos brazos lo superan**, de modo que en esa superficie no acota nada y el nombre «techo» es
incorrecto.

## El estimando y su criterio

Para cada corrida y cada brazo neuronal de su clase:

```
Delta_bnn = R2(red) - R2(mejor no neuronal de su clase)
```

pareado por fold, con el criterio **ya congelado** de la Puerta B, sin cambiarlo:
**media ≥ SESOI 0,05 y el IC95 excluye cero**.

El mejor no neuronal se elige **por su media en la propia corrida**, lo cual sesga a favor del
comparador —es la elección conservadora y es deliberada—. Se registra cuál fue en cada caso.

## Falsadores

* **g1_the_partition_is_by_information_set** — falla si algún brazo de clase B aparece como
  comparador de un brazo de clase A, o si `train_cell_mean_comparator` aparece como comparador de
  cualquiera. *Puede fallar:* es la tentación exacta que hace ganar a la red por construcción.
* **g2_every_artifact_contributes_its_own_folds** — falla si dos corridas comparten arreglo de
  `per_fold`, lo que significaría que leí el mismo artefacto dos veces.
* **g3_the_criterion_is_the_frozen_one** — falla si el SESOI o la regla de IC difieren de los del
  preregistro original. *Puede fallar:* re-adjudicar es la ocasión ideal para aflojar el umbral.
* **g4_a_control_must_change_the_verdict** — se re-adjudica también contra el baseline primario
  original. Falla si el veredicto es idéntico bajo los dos comparadores en las cuatro corridas, lo
  que significaría que la enmienda no mide nada.
* **custody** — `NOT_APPLICABLE`: no se abre ninguna semilla y no se ejecuta ningún episodio.

## Lo que esta enmienda NO puede hacer

No puede convertir un `BLOCKED_INSTRUMENT` en resultado. La corrida `gate_b_confirmation_v2` se
re-adjudica y se reporta **como evidencia de desarrollo**, nunca como confirmación: su bloque se
perdió por falsador y ese hecho no lo cambia un cálculo posterior.

No puede subir de grado a la sensibilidad `ret_excel`, que sigue siendo replay declarado sobre las
mismas tapas.

La única corrida con grado confirmatorio es `gate_b_confirmation_v3`, y es la única cuyo veredicto
re-adjudicado puede sostener o tumbar el `SURFACE_PREMIUM_CAPTURED`.
