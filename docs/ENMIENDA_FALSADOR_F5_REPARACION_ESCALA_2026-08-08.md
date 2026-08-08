# Enmienda — `f5` del preregistro de reparación de escala era inmedible

**Enmienda a** `docs/PREREGISTRO_COBB_DOUGLAS_REPARACION_ESCALA_2026-08-08.md`, escrita **antes de
medir ninguna variante** y antes de escribir el runner. No edita aquel documento: lo sucede en el
punto que aquí se declara, y todo lo demás sigue vigente sin cambio.

## Qué decía `f5` y por qué no se puede medir

> `f5_negative_control_stays_at_zero` — en la rejilla de 288 donde una configuración es óptima en
> las seis regiones, **toda** variante nueva debe dar `H = 0`.

**Es falso de partida, y lo dice un artefacto ya sellado nuestro.** En
`results/cobb_douglas_variant_family/result.json`, medido sobre esa misma rejilla de 288, las
cuatro variantes con exponentes `published` dan `H_regime` hasta **0,06396** — no cero. El cero de
la rejilla de 288 no es una propiedad de la rejilla: es una propiedad de la rejilla **bajo la regla
de exponentes aplicada a nuestros máximos**, que es donde el argmax se queda en la misma
configuración en los seis contextos.

Un falsador que exige un cero que el artefacto sellado ya contradice no puede fallar de forma
informativa: fallaría siempre, por una razón que no tiene nada que ver con la reparación. Sustituir
un falsador roto por otro medible antes de correr es lo correcto; hacerlo después de ver el
resultado no lo sería, y de ahí la fecha de esta enmienda.

## Qué lo sustituye

**`f5a_baseline_reproduces_the_sealed_family` — el ancla.** La celda base del nuevo diseño,
`garrido_c1 × at_max × his_five × within`, **es** la variante `ours × his_five × within` de la
familia sellada: mismo vector de costes (los siete a 1), misma regla de exponentes sobre los mismos
máximos, mismas variables, mismo conjunto de κ̇. Debe devolver el mismo número, a `1e-9`:

```
H_regime = 0.0     lcb95 = 0.0     respects_share_bound = True     max_term_magnitude = 0.2
```

**Puede fallar**, y es el único falsador que hace comparables los 30 números nuevos con los 158
anteriores: si la reimplementación no reproduce la sellada, la tabla combinada no significa nada y
el resto del artefacto se retira. El `max_term_magnitude = 0,2` exacto es parte del ancla — es la
cota de share tocada justo en el límite, que es lo que la regla de Garrido produce por
construcción.

**`f5b_manufactured_headroom_is_declared_as_misscaling`.** Si alguna variante nueva levanta
`H_regime` por encima de 0 en la rejilla de 288 **rompiendo su cota de share**, se clasifica como
mal escalada —tratamiento idéntico al de las cuatro anteriores— y no como hallazgo. Si lo levanta
**respetando** la cota, es un hallazgo y se reporta como tal con su mecanismo.

**Puede fallar en ambas direcciones**, que es justo lo que `f5` no podía hacer.

## Lo que no cambia

Los defectos A y B, sus números medidos, los ejes D y E, `K = 188`, la corrección de Holm, la cota
de share, las tres reglas de lectura y los falsadores `f1`–`f4` y `f6`–`f8` siguen exactamente como
se preregistraron. En particular sigue en pie **`f4`**, la predicción declarada de que `over_range`
**baja** el headroom — la dirección que va en contra de lo que nos conviene.
