# Resultado — en el mejor punto del sistema, la política **pierde** contra la constante

**Artefacto:** `results/sensitivity/op12_conditioned_policy_v1/result.json` (sello
`95efc4a1ca666254…`) · **los seis falsadores pasan** · métrica
`ret_excel_risk_conditional` · 7 regímenes × 6 semillas de ajuste + 6 de prueba, **disjuntas**.

## La medición

Se puso la variable donde `S_ij` la señaló —**el periodo de despacho de `op12`**, acoplado al
impacto de R1r con `S_ij = 0,219`— y se hizo **condicionable al estado de riesgo realizado**
(eventos R1r en las últimas 336 h, el pasado, nunca el futuro).

| política | valor fuera de muestra | captura del headroom |
|---|---:|---:|
| **constante** (la mejor, `rop = 21`) | 0,005866 | — (línea base) |
| **oráculo por régimen** | 0,006104 | **100%** (techo, brecha 0,000238) |
| **reactiva sobre el observable** | 0,005817 | **−20,5%** |
| **placebo** (traza de otra semilla) | 0,005810 | −23,6% |

**La regla reactiva no captura nada: pierde el 20% de la brecha.** Es *peor* que la mejor
constante fuera de muestra.

**La señal sí vale algo —pero no lo suficiente.** La reactiva supera al placebo (−20,5% contra
−23,6%), así que el observable **contiene información**: `f2` lo confirma. Lo que no consigue es
convertirla en ganancia frente a una constante.

## Lo que esto decide

El contrato de este experimento declaraba la puerta por adelantado:

> **Si la regla condicionada más simple no captura nada aquí, no se entrena nada más caro.**
> Un aprendizaje por refuerzo no puede extraer un acoplamiento que un umbral ajustado sobre el
> mismo observable no ve.

Y esto no es un punto cualquiera del sistema: es **el mejor punto que el análisis de
sensibilidad completo encontró** — el par con mayor interacción decisión × riesgo de los 55
pares evaluados, aguas abajo, sobre la métrica con 65× más resolución de headroom que la
canónica. **Si aquí no hay conversión, el «no entrenar» deja de ser una sospecha y pasa a ser
una medición en el punto más favorable disponible.**

## Dos veces me lo dijeron los falsadores antes de dejarme concluir

`f6` —«la regla ajustada realmente conmuta»— **detuvo dos corridas**:

1. Con umbrales fijados a mano `(1,2,3,4)`, el observable casi nunca bajaba de 1 y la regla
   quedaba **siempre encendida**: reactiva y placebo daban el mismo número a seis decimales.
2. Con umbrales derivados de cuantiles **agrupando los siete regímenes**, el percentil 25 salió
   **0** —R2r y R3 puros no tienen ningún evento R1r— y el ajuste **eligió el umbral 0**, otra
   constante disfrazada.

La corrección exigida por `f6`: derivar los umbrales **solo de los regímenes que contienen R1r**
y admitir al ajuste únicamente candidatos que conmutan entre el 5% y el 95% de los pasos. La
regla final conmuta de verdad (umbral 16) — y entonces pierde.

**Una constante disfrazada de política habría dado «captura 0,0%» y parecido un empate.** Es
justo el resultado que se publica sin querer.

## Límites

* Clase de política **mínima**: un umbral de dos niveles sobre un observable. Una política más
  rica podría hacerlo mejor — pero la puerta declarada dice que ése es exactamente el gasto que
  este resultado no autoriza.
* El observable es el **conteo de eventos R1r recientes**; otro estadístico (severidad, tiempo
  desde el último) podría llevar más señal.
* La brecha del oráculo es **2,4e-4**: incluso capturada al 100% sería pequeña. Lo que se midió
  es que ni siquiera se toca.
