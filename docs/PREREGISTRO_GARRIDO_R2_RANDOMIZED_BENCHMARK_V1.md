# Preregistro — el benchmark que Garrido pidió: R1 quieto, R2 aleatorizado, KAN contra MLP

**Escrito y commiteado ANTES de escribir el runner y ANTES de entrenar nada.** Custodia: no hay
bloques vírgenes (`ENMIENDA_4`), así que es **desarrollo sobre réplica declarada** y no adjudica.
Las semillas de optimizador **no son semillas de cinta** y se registran aparte (`TAPE_FLOOR`).

## 0. La petición, literal

> *«Los R1 los dejamos quietos y los R2 los modificamos. Y corremos otra vez el análisis.»*

Con siete puntos: R1r (`R11–R14`) fijos; R2r (`R21–R24`) más aleatorios modificando distribución,
frecuencia e impacto; repetir KAN–MLP; **si empatan, elegir por parsimonia**; si aparece una ventaja
de **5–10 %**, confirmarla; **repetir con menos parámetros** para ver qué arquitectura se degrada
primero; y encender/apagar riesgos uno a uno graficando su comportamiento temporal.

Los cuatro mecanismos, verificados en `config.py:469-499`:

| riesgo | mecanismo | parámetros modificables |
|---|---|---|
| R21 | desastres naturales sobre ops 3, 5, 6, 7, 9 **simultáneas** | frecuencia y recuperación |
| R22 | destrucción de líneas de comunicación (ops 4, 8, 10, 12) | frecuencia y recuperación |
| R23 | destrucción de unidad avanzada (op 11) | frecuencia y recuperación |
| R24 | demanda contingente (op 13), 2.400–2.600 raciones | frecuencia y magnitud del surge |

## 1. La divulgación que va ANTES del resultado, no después

**El entorno no contiene un problema de asignación entre sus dos palancas, y está medido.**
`results/actuator_complementarity_screen/result.json` (sello `81169a2e…`,
`PERFECT_SUBSTITUTES_EVERYWHERE_ON_THE_SCREENED_GRID`): sobre 18 configuraciones —frecuencia ×1/×2/×4,
impacto ×1/×2/×4, presión de demanda ×1,00/×1,15— la complementariedad
`min(solo-turnos, solo-buffer) − ambos` es **0,000000 en quince celdas** y marginalmente negativa
en dos. Las palancas tienen autoridad (`ninguno` es peor por 0,11–0,15 en todas), pero son
**intercambiables**.

Program O midió que la contención sobre un recurso **no fungible** lleva `H_PI = 0,1515` y que
hacerlo fungible lo lleva a **exactamente 0**. Dos palancas perfectamente fungibles y saturantes
**son** el caso fungible.

**Esto no bloquea el benchmark: lo enmarca.** Se corre igual, porque la pregunta de Garrido es
sobre arquitecturas y merece respuesta empírica. Pero la conclusión *«ninguna arquitectura extrae
valor»* quedará **explicada por el entorno** y no atribuida a las redes, y eso se declara aquí para
que no pueda leerse al revés después.

## 2. Dos entornos congelados

**Baseline** — R1r en `current`; R2r en los niveles de la fuente; R3 apagado; misma demanda y mismo
presupuesto físico que el modificado.

**R2 modificado** — R1r **idéntico** al baseline; **sólo** R21–R24 cambian; frecuencia e impacto
**sorteados por episodio desde una distribución congelada**, con niveles anclados en
`off / current / increased / severe`; **los parámetros del sorteo quedan ocultos a la política**; y
**las mismas tapes** para KAN, MLP y todos los comparadores.

En el manuscrito no se llama «fine-tuning de riesgos». Se llama
**`source-anchored randomized R2 stress design`**, porque cada nivel está anclado en la fuente.

## 3. La elegibilidad del entorno se decide SIN mirar KAN–MLP

Un entorno es elegible sólo si: el endpoint **no** está saturado ni colapsado; **al menos dos
políticas son óptimas en tapes distintas**; existe headroom temporal con presupuesto igual; el
placebo temporal **no** captura el efecto; y cada riesgo tiene incidencia y actuador documentados.

**Esa evaluación se ejecuta y se sella antes de entrenar la primera red.** Si se mirara primero
KAN–MLP y luego se subiera R24 o se apagara R22 hasta conseguir un 10 %, el resultado sería
circular: *KAN gana en el entorno seleccionado porque KAN ganaba en ese entorno.* Autorización para
hacer stress testing no es licencia para buscar el escenario donde gane una arquitectura.

## 4. El contraste primario es la INTERACCIÓN

```
Δ = (KAN − MLP)_R2modificado  −  (KAN − MLP)_baseline
```

Eso responde la pregunta que Garrido hizo —**¿la complejidad de R2 beneficia diferencialmente a
KAN?**— y no la que no hizo, que es si una red ganó una corrida. Una ventaja de KAN presente en
**ambos** entornos no es evidencia sobre incertidumbre R2; es una diferencia de arquitectura sin
relación con el estímulo.

**SESOI = 5 %**, el número que él mismo puso, declarado aquí y no ajustado después.

**Comparadores, los cuatro:** MLP · KAN · una regla simple o belief-MPC · el mejor calendario
open-loop. Sin los dos últimos, un empate KAN–MLP no distingue *«ambas capturan el valor»* de
*«no hay valor que capturar»* — que es exactamente lo que §1 hace probable.

## 5. Emparejamiento, o el benchmark no significa nada

**Tres presupuestos paramétricos: 25 %, 50 % y 100 %**, que es el «repetir con menos parámetros»
que él pidió — y una **curva de capacidad simétrica**, no reducir el MLP hasta que se rompa. Con,
en todos los casos:

* **igual número de interacciones con el DES** (el recurso que de verdad escasea);
* **igual presupuesto de búsqueda de hiperparámetros**;
* parámetros aproximadamente igualados entre arquitecturas dentro de cada nivel;
* **latencia y memoria reportadas** en todos los casos.

Sin esto, «KAN gana» puede significar sólo «KAN recibió más búsqueda».

## 6. Los cuatro resultados posibles, fijados de antemano

| resultado | lectura |
|---|---|
| interacción ≥ 5 % con `LCB95 > 0` | **KAN aporta valor bajo incertidumbre R2.** Se confirma con benchmarking, como él pidió |
| equivalencia (TOST dentro del SESOI) | **Se elige MLP por parsimonia**, que es su propia regla |
| ambas se degradan al reducir parámetros | la hipótesis de que KAN resiste mejor la reducción **no se sostiene** |
| ninguna bate a la regla ni al calendario open-loop | **no hay headroom físico**: se detiene el benchmark adaptativo y se reporta que ninguna arquitectura puede extraer una decisión que el entorno no contiene — con §1 como mecanismo |

## 7. La sensibilidad que pidió, riesgo por riesgo

Para cada `R21`–`R24`: apagarlo individualmente · activarlo en nivel fuente · activarlo en nivel
extremo congelado. Y en cada caso **graficar a lo largo del episodio** eventos, backlog, `L*`,
inventario y acción, reportando contribución marginal e interacción con la arquitectura. Es el
punto 7 de su lista y es lo que convierte una tabla en un diagnóstico.

## 8. Endpoint y disciplina heredada

Primario `L* = Σ qᵢ[eᵢ−(OPTᵢ+LTᵢ)]₊ / Σ qᵢ[T−(OPTᵢ+LTᵢ)]₊`, adimensional en `[0,1]`, con denominador
invariante a la política; abandonar no puede mejorarlo. Secundarios: `ES10(L*)`, raciones
entregadas/demandadas, demanda perdida, horas-turno y horas-inventario. **`ret_excel` no entra.**

Se heredan las reglas que costaron cinco familias aprender: **no hay rama `STOP`** —una ausencia
exige `UCB95 < δ` sobre una clase enumerada—; **los falsadores fijan el `claim_status`**, no sólo
el código de salida; el veredicto **nombra la clase de política que realmente se buscó**; y **cero
retuning tras ver el resultado** — cualquier variante posterior es una familia nueva con su
multiplicidad pagada.

## 9. Alcance

`DEVELOPMENT_ON_DECLARED_REPLAY`. No abre semillas de cinta, no adjudica, y una ventaja de KAN aquí
autorizaría **diseñar** una confirmación con semillas propias, nunca reclamarla.
