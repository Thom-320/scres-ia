# Enmienda de alcance — cierre del buffer bajo posturas estáticas

**Fecha:** 2026-08-08
**Predecesor:** `docs/CIERRE_FAMILIA_BUFFER_ESTRATEGICO_2026-08-08.md`
**Artefacto auditado:** `results/budget_expiry_priced/result.json`
**Regla:** el predecesor se conserva; esta enmienda sustituye únicamente su afirmación de alcance.

## Corrección

El resultado sí demuestra que el precio mueve la mejor **postura estática** y que ninguna de las
12 tapes de test selecciona otra postura estática dentro de las 27 enumeradas. No demuestra que
ninguna tape prefiera una **secuencia temporal** distinta.

El runner importa:

```text
POSTURES = {0, 0.5, 1}^3
```

y `play()` aplica la misma acción `[op3, op5, op9, -1]` en los 26 pasos. No enumera calendarios,
reglas de feedback, tiempos de activación, ni políticas dependientes de inventario, backlog,
caducidad o presupuesto. Su `per-tape min` es, por tanto, un mínimo sobre 27 posturas constantes,
no un oráculo secuencial.

### Etiqueta corregida

```text
STATIC_BUFFER_POSTURE_CLASS_CLOSED__NO_TAPE_HETEROGENEITY_ON_27_CONSTANTS
```

La etiqueta original
`STRATEGIC_BUFFER_FAMILY_CLOSED__NO_PRICED_SEQUENTIAL_HEADROOM` queda **superseded en alcance**, no
en sus números. Las 24 combinaciones conservan gap 0, `p=1` y un óptimo estático distinto por tape
igual a uno.

## Segundo límite: no se preservó la matriz cruda

El JSON conserva agregados por celda y λ, pero no las matrices `tape × postura` de `L`, coste,
caducidad, presupuesto y reposición. El runner permite regenerarlas mientras el entorno actual sea
reproducible, pero el artefacto por sí solo no atestigua las filas exactas. Esto incumple el estándar
que motivó la reconstrucción G3a y debe corregirse en cualquier sucesor.

## Tercer límite: escala de coste con información de test

La normalización se define como `cost.max()` después de apilar train y test. Aunque no selecciona
una postura usando el outcome de test, sí usa el bloque de test para fijar la escala del endpoint.
Eso es fuga de evaluación y vuelve los λ dependientes de las tapes observadas. Un sucesor debe usar
una cota física fijada a priori o, como mínimo, una escala derivada exclusivamente de train.

## Qué permanece válido

1. La física conservativa no destruye stock en el camino evaluado.
2. Presupuesto y caducidad se activan en sus celdas.
3. El coste separa posturas.
4. El precio cambia la postura train-selected de `[0,0,0.5]` a `[0,0,0]`.
5. Dentro de la clase de 27 constantes, no hay interacción tape-postura detectable.

La lectura útil es más estrecha y todavía fuerte: **el actuador se comporta como una decisión de
diseño dentro de la clase estática probada**. No autoriza un aprendiz y tampoco prueba un teorema
sobre todas las políticas secuenciales del actuador.

## Regla de sucesión

No se ejecutará otra rejilla de precios ni otra rejilla de posturas para rescatar este actuador. El
programa sucesor cambia de física y derecho de decisión: cartera de proveedores y capacidad aguas
arriba con lead times, compromisos y riesgo persistente. La reserva A/B de transporte no cuenta como
sucesor porque ya fue Program M.
