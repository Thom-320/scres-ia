# Enmienda — la rejilla de umbrales contenía una política que no actúa

**Escrita ANTES de reejecutar.** Enmienda a `scripts/run_observable_conversion_v1.py`, cuyo
resultado `results/observable_conversion/result.json` (sello `39e6047d…`) se **conserva** con su
`BLOCKED_INSTRUMENT`.

## 1. El defecto, medido

`f3` falló porque la regla no bate al placebo desinformado. La causa no era el entorno: el umbral
seleccionado en entrenamiento fue `θ = 200.000` raciones y **en las seis tapes de test la regla
sostuvo el buffer cero semanas**. Una política que nunca actúa **es** su propio placebo, así que el
contraste era `0,000000 [0, 0]` por construcción.

Medido sobre las 312 semanas-episodio del sondeo con el buffer apagado:

| percentil del backlog | valor |
|---|---:|
| p0 | 79.374 |
| p25 | 140.878 |
| p50 | 156.876 |
| p90 | 160.211 |
| **p100** | **163.986** |

**`θ = 200.000` está por encima del máximo observado.** No podía disparar nunca, y mi rejilla lo
incluía como candidato legítimo. Semanas sostenidas por umbral: `θ = 0` → 26,0 · `50.000` → 26,0 ·
`100.000` → 24,9 · `150.000` → 17,3 · **`200.000` → 0,0**.

## 2. El arreglo, declarado antes de correr

**La rejilla se ancla en percentiles del backlog observado, calculados SÓLO en tapes de
entrenamiento**, de modo que cada miembro actúa por construcción y ninguno queda fuera del soporte:

```
theta ∈ { p10, p25, p50, p75, p90 }   del backlog medido en las tapes de entrenamiento
```

Percentiles y no valores redondos porque el soporte del backlog es una propiedad **medida** del
entorno, no una intuición mía; y sólo en entrenamiento porque calcularlos sobre las tapes de test
sería seleccionar el instrumento contra los datos que lo puntúan — el defecto que ya hundió al
benchmark.

## 3. El falsador nuevo, que puede fallar

**`f6_every_threshold_acts`**: cada `θ` de la rejilla debe sostener el buffer **al menos una
semana** en cada tape de entrenamiento. Si alguno no actúa, la rejilla vuelve a contener una
política de no-hacer-nada y el instrumento queda bloqueado **antes** de medir, en vez de después.

Puede fallar: si el soporte del backlog fuera degenerado —todo por debajo de p10— ningún umbral
discriminaría, y eso debe verse.

## 4. Lo que NO cambia

`λ = 0,35` sigue siendo el titular **y sigue siendo un pico seleccionado sobre estas mismas
tapes**; la banda entera 0,275–0,500 se reporta al lado. El calendario comparador se sigue
eligiendo sólo en entrenamiento. El placebo sigue emparejado al número de semanas que la regla
realmente sostuvo. Y los cuatro veredictos posibles no se tocan — incluido
`OBSERVABLE_POLICY_IS_WORSE_THAN_THE_FIXED_SCHEDULE`, que sigue siendo un resultado admisible.

**El techo de `d5e0b9bf` no se toca**: 0,045103 [LCB95 +0,028482] en λ = 0,35.
