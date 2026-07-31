# Por qué restringir a la ventana no cambió el conteo — y retracta mi diagnóstico de la fuga

**Status:** `RETRACTION_AND_MECHANISM_RESOLVED`. Nada implementado.

## 1. Mi diagnóstico de «fuga por mínimo global» era falso

Afirmé que `_consume_ret_quantity_risk_for_order` siembra `R⁰` con el **mínimo sobre todos
los refs jamás registrados**, y que por eso casi toda orden heredaba el primer `R14` de la
corrida.

Medido, `sim._ret_quantity_risk_refs['R14']` al final de una corrida de 52 semanas:

    refs globales de R14 registrados: 25
    rango [8040, 8712]   separación mediana = 24,0 h

**Veinticinco refs, todos en las últimas cuatro semanas.** No es un histórico acumulado: es
una **cola viva que se drena** a medida que se consume. Así que su mínimo, en el momento de
atribuir, **ya es reciente por construcción** — nunca fue un mínimo sobre toda la historia.

**Por eso filtrar «dentro de la ventana» no excluyó nada**: el ref pendiente ya estaba en la
ventana de la orden que lo consumía.

## 2. El mecanismo real, medido

Sobre las órdenes con `RPj > 0`, distancia de `OPTj` al primer ref de `R14` en su ventana:

| | |
|---|---:|
| p10 | **0,00 h** |
| p50 | **0,00 h** |
| p90 | 67,20 h |
| fracción a menos de 24 h | **69,6%** |

**Para la mitad de las órdenes el origen `R⁰` cae exactamente en `OPTj`.** Con
`RPj = OATj − R⁰`, eso da `RPj = CTj` de forma exacta.

No hay fuga que cerrar. `R14` se modela como una **compuerta persistente** —su bucket no se
agota a propósito, documentado en el código como réplica del comportamiento de las columnas
Excel— y esa compuerta «se manifiesta» al inicio de cada orden. Ahí es donde `RPj` colapsa
sobre `CTj`.

## 3. Y esto reformula la brecha contra Garrido

Su `R14` toca el **98,1%** de las órdenes —igual que el nuestro— y sin embargo **su `RPj`
satura cerca de 400** mientras el nuestro sigue a `CTj` sin cota.

Luego la diferencia no es la frecuencia de `R14`, ni la ventana de atribución, ni un mínimo
global. Es que **en su modelo la compuerta `R14` no siembra `R⁰` al inicio de la orden**. Su
`RPj` correlaciona 0,88 con el **conteo** de riesgos y solo 0,37 con `CTj`; el nuestro es
`CTj` por construcción.

## 4. Lo que retracto, explícitamente

* «`ref_start` usa el mínimo global sobre todos los refs» — **falso**, la lista se drena.
* «el bucket de `R14` que no se agota hace heredar el primer evento de la corrida» — **falso**
  por lo mismo; el bucket persistente afecta a *qué* órdenes se marcan, no a *cuándo* empezó.
* «la fuga está localizada con precisión» (`04654aa` §2) — **lo estaba mal**. La localización
  correcta es §2 de aquí.

Lo que **sí** se sostiene de aquel documento: que ~14% de las violaciones de `f3` eran error
de mi falsador, y que mi arreglo `order_window` no movió el conteo y degradó `ret_mean` de
0,38 a 1,79 — ahora se entiende **por qué** no lo movió.

## 5. Estado

Nada implementado. El resultado del cruce no cambia: `L` mejor en `ret_mean` (0,29), `C`
mejor en `RPj` (nivel 672, residuo real 2,07), los dos ejes no se componen, ningún brazo
adoptable.

Lo que cambia es **qué habría que atacar**: no una fuga de atribución, sino la decisión de
que la compuerta `R14` siembre `R⁰` en `OPTj`. Eso es una pregunta de modelado con texto de
la tesis detrás (Algoritmo 2 exige que el impacto *se manifieste dentro* del intervalo, y una
compuerta persistente manifestándose al inicio de toda orden cumple la letra pero vacía la
condición de contenido).
