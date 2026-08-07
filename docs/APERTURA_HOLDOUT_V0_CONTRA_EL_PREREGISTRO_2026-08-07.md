# Apertura del holdout v0 — contra la regla de salida, por decisión explícita del PI

**Escrito ANTES de correr.** Bloque: `5.300.007–5.300.012`, la mitad retenida de
`docs/PREREGISTRO_GARRIDO_V0_RECOVERY_SURFACE_V1_2026-08-06.md`.

## 1. Lo que el preregistro dice, literalmente

> `STOP_NO_RECOVERY_LEARNING_HEADROOM`: cualquier gate falla.
> Sólo el primer veredicto autoriza desarrollar comparadores sobre las seis semillas de desarrollo.
> **No autoriza abrir semillas nuevas ni mirar el holdout.**

Y antes:

> No se construirá la mitad retenida hasta congelar por escrito el algoritmo, hiperparámetros,
> secuencia de campañas, estimandos y placebos.

**Las dos condiciones se incumplen.** El desarrollo dio `STOP_NO_RECOVERY_LEARNING_HEADROOM`
(`results/garrido_v0_surface_gates_v1/`): **G2 falla** (1 de 6 contextos contra 4 exigidos) y
**G3 falla en cero exacto** (`H_regime_TTR = 0,0`, IC [0 · 0], con la misma postura `[0, 672, 168]`
elegida en los seis contextos y en los seis folds). Y **no existe aprendiz**, así que no hay nada
congelado que el holdout pueda validar.

## 2. Quién decide y qué se decidió

Levanté la objeción explícitamente, con las dos vías por las que el preregistro lo prohíbe y con el
precedente de Program O. **El PI la reafirmó: «ábrelo».** Se ejecuta.

Esto no es una relectura del preregistro ni una excepción que el propio documento contemple. **Es
una apertura en contra de su regla de salida**, y se registra como tal.

## 3. Lo que esta apertura cuesta, dicho antes de tener los números

**El bloque `5.300.007–012` queda quemado.** No podrá usarse después como confirmación prospectiva
de ninguna hipótesis de esta familia, gane o pierda hoy.

**Y el resultado no será una confirmación, sea cual sea:**

* **Si el holdout confirma el STOP** —lo esperable— es una **réplica del negativo**, informativa y
  barata, pero no añade autoridad: el desarrollo ya lo había dicho.
* **Si el holdout contradice el STOP** y aparece señal, **NO es un positivo citable.** Sería un
  resultado seleccionado tras un STOP, sobre el único bloque que quedaba para validarlo, sin
  aprendiz congelado. La lectura correcta en ese caso es *«el desarrollo y el holdout discrepan; la
  hipótesis queda sin resolver y ya no quedan semillas limpias para resolverla»*, y obligaría a un
  bloque nuevo con autorización propia.

El artefacto llevará `claim_status` que empiece por `OPENED_AGAINST_PREREGISTRATION_` para que
ningún lector futuro —ni un revisor, ni nosotros dentro de un mes— pueda confundirlo con una
confirmación limpia.

## 4. Qué se corre

Las mismas seis semillas de la mitad retenida, con el **mismo builder, mismo contrato, mismos 216
postes, mismos 8 contextos y misma ventana de recuperación** que el desarrollo. Después, los mismos
gates G0–G3 sin cambiar ni un umbral.

**No se toca ningún umbral, ni el estimando, ni la lista de contextos vivos.** Si algo de eso se
moviera, esto dejaría de ser una réplica y pasaría a ser una búsqueda.

## 5. Registro de custodia

El bloque se marca en `research/seed_custody_registry.json` como
`BURNED_OPENED_AGAINST_PREREGISTRATION`, con este documento como `source` y la decisión del PI
citada. **Un estado nuevo, no uno existente**, porque ninguno de los existentes describe lo que
pasó aquí.
