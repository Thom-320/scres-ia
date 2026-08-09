# Enmienda — la fórmula del presupuesto y su propósito se contradicen en la misma frase

**Fecha:** 2026-08-08, **antes de correr el mapa**. Enmienda a
`docs/PREREGISTRO_PRESUPUESTO_COMPARTIDO_Y_CADUCIDAD_2026-08-08.md` §3.

## El conflicto

El preregistro define «ajustado» así, en una sola frase:

> la bolsa que **impide preposicionar los tres nodos a la vez**:
> `presupuesto = 0,5 × (suma de los tres objetivos máximos) / 26`, sin arrastre.

**Las dos mitades no describen el mismo número.** «Impedir llenar los tres a la vez» es una
condición **por periodo**; dividir además entre 26 convierte esa cantidad en una vigésimo-sexta
parte de sí misma. Medido: con `/26` el presupuesto es 2.817 unidades kit-equivalentes por semana
frente a las ~19.578 que la política sin restricción gasta, y no impide preposicionar los tres
nodos — **impide preposicionar nada**, rechazando 7,5 millones de unidades y reponiendo 4.359.

Eso no es contención: es inanición, y una celda así no mide competencia entre nodos, mide un
entorno apagado.

## Qué se corrige y con qué criterio

**Manda el propósito, no la aritmética.** El propósito es la especificación —dice qué debe hacer la
restricción— y el `/26` es un error mío de unidades: dividí una cantidad por periodo entre el número
de periodos.

```
presupuesto por periodo = 0,5 × (op3_rm/12 + op5_rm/12 + op9_rations)  [máximos I1344]
                        = 0,5 × 146.480 = 73.240 unidades kit-equivalentes
```

Con eso, llenar los tres nodos al máximo en un periodo es **imposible por construcción** —que es
literalmente lo que la frase pedía— y llenar uno o dos sí es posible, que es lo que crea la
decisión de asignación.

**Este número se deriva de los máximos declarados y de la condición escrita, no de ningún
resultado.** No se ha corrido el mapa. El falsador `f3_budget_binds_when_tight` sigue exigiendo que
la restricción se active de verdad, y sigue pudiendo fallar.

## Lo que NO cambia

Ni las cuatro celdas, ni la vida útil inerte de 156 semanas como control fiel, ni la barra de 0,01,
ni la exigencia de que `f6` y `f7` pasen **juntos**, ni las semillas quemadas, ni las reglas de
lectura. Sólo el valor de una constante cuya definición era internamente inconsistente.
