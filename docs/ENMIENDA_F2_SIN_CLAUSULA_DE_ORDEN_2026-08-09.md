# Enmienda — f2 sin cláusula de orden, y apertura del bloque 9600001–9600008

**Fecha:** 2026-08-09 · **Autoriza:** PI · **Alcance:** Programa N, Puerta B (confirmación)
**Enmienda a:** `docs/ENMIENDA_F2_CONFIRMACION_Y_APERTURA_C_2026-08-09.md`
**No edita ningún contrato congelado ni artefacto fechado.** Los dos bloques quemados se conservan.

## Qué se enmienda y por qué

La forma de confirmación de `f2` exigía que el orden de los brazos clásicos se conservara entre el
desarrollo y las tapas frescas:

```
spline_buffer >= linear_interactions >= linear_additive >= constant
```

Escribí en el propio falsador que un reordenamiento *«no puede ocurrir por variación muestral»*.
**Es falso.** En el bloque 9500001–9500008, `linear_interactions` (0,6884) adelantó a
`spline_buffer` (0,6731) por **0,0153**, con IC pareado **[−0,0388, +0,0082]** — una cantidad que
cruza cero. El falsador hizo un test de signo sobre ella y quemó el bloque.

Es exactamente la regla que el cierre de Program L dejó escrita (R6/R7): *nunca un test de signo
sobre una cantidad que cruza cero*. Es el segundo defecto mío en la misma puerta; el primero
comparaba niveles entre tapas distintas y quemó el bloque 9400001–9400008.

## La forma nueva

La cláusula de orden **se elimina**. `f2` conserva la mitad que valía y la complementa con una
condición que el ruido no puede fabricar:

1. **Identidad del instrumento** — `module_manifest` byte-idéntico al artefacto de desarrollo.
2. **Ninguna reversión significativa** — para cada par consecutivo del orden de desarrollo, se
   calcula la diferencia pareada por fold; falla sólo si el par se invierte **con su propio IC95
   excluyendo cero**.

**Por qué puede fallar:** un hash de módulo distinto (yo toqué algo entre desarrollo y
confirmación), o una reversión clásica cuyo IC excluye cero — que sí implica que el instrumento
cambió, porque los brazos clásicos son deterministas dada la muestra.

**Por qué ya no falla por ruido:** dos brazos indistinguibles pueden ordenarse como quieran sin
disparar nada.

## Validación del falsador reparado (obligatoria, ambos sentidos)

Sobre el artefacto ya quemado `results/program_n/gate_b_confirmation_v2/result.json`, sin reabrir
ninguna semilla:

**Control negativo — habría pasado:**

```
linear_interactions - spline_buffer   +0.0153  [-0.0082, +0.0388]   ok
linear_additive - linear_interactions -0.0260  [-0.0688, +0.0168]   ok
constant - linear_additive            -0.6740  [-0.9102, -0.4378]   ok
f2 reparado -> PASA
```

**Control positivo — dispara ante una reversión real.** Invirtiendo el orden esperado a propósito,
el falsador detecta `constant` vs `linear_additive` con LCB95 **+0,4378**. El instrumento tiene
potencia; no pasa por construcción.

## Apertura de bloque

**Bloque `9600001–9600008`**, verificado libre: 0 apariciones en
`research/seed_custody_registry.json`, 0 colisiones en artefactos sellados bajo `results/`.

Es el **tercer** bloque que consume esta misma hipótesis. Se registra el precio explícitamente:
dos de los tres se perdieron por defectos de falsador míos, no por el dato. Si este también se
pierde por instrumento, la Puerta B se cierra y se reporta con los tres bloques a la vista.

## Lo que NO se enmienda

`f1`, `f3`, `f4`, `f5`, `f6`, `f7`, el criterio congelado del premio (`media ≥ SESOI 0,05` **y**
IC excluyendo cero), el baseline primario `linear_interactions` preexistente, la escalera de
comparadores, el pareo del brazo recurrente contra `linear_lagged`, y el endpoint Cobb-Douglas.

## Lo que el bloque quemado ya midió, y que esta corrida debe replicar

En tapas frescas 9500001–9500008, contra el baseline primario:

```
kan_tuned      +0.0650  [+0.0296, +0.1005]   cumple el criterio congelado
mlp_tuned      -0.0127  [-0.0828, +0.0574]
spline_buffer  -0.0153  [-0.0388, +0.0082]
tree           -0.0260  [-0.1145, +0.0626]
```

`f4` falló allí y **eso es dato, no instrumento**: `min(mlp, kan) − linear = −0,0127`. La hipótesis
«nuestras redes eran débiles» se sostiene sólo para el KAN. Esta corrida no cambia f4 ni su umbral.
