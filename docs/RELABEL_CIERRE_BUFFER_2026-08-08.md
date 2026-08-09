# Reetiquetado — el cierre del buffer cubría menos de lo que su nombre decía

**Fecha:** 2026-08-08. Corrige la etiqueta de `results/budget_expiry_priced/result.json`
(commit `103ca162`). El artefacto **se conserva**; lo que cambia es el alcance que su nombre
reclama.

## Qué probé y qué dije que probé

Escribí `STRATEGIC_BUFFER_FAMILY_CLOSED__NO_PRICED_SEQUENTIAL_HEADROOM` y hablé de «valor
secuencial». **La clase evaluada son 27 posturas CONSTANTES** —`(op3, op5, op9)` sobre
`{0, 0,5, 1}`— sostenidas las 26 semanas. Ninguna política de la clase cambia con el estado ni con
el tiempo.

Por tanto lo medido es si **la mejor postura constante varía de tape en tape**, y la respuesta es
que no: un único óptimo en las doce tapes, en las cuatro celdas y en los seis λ. Eso es un
resultado real y es más estrecho que «no hay headroom secuencial»: **una política que reacciona
dentro del episodio nunca se corrió.**

## Etiqueta correcta

```
STATIC_BUFFER_POSTURE_CLASS_CLOSED__NO_TAPE_HETEROGENEITY_ON_27_CONSTANTS
```

## Un segundo defecto, menor pero real

El coste se normaliza con el **máximo sobre train y test juntos**. El normalizador es común a
todas las posturas, así que no puede favorecer a una sobre otra ni mover el argmin — pero deja
entrar información del bloque de test en una cantidad usada para elegir en train, y eso no debería
ocurrir por principio. Un sucesor normaliza con el máximo constructivo de la clase, que no depende
de ningún bloque.

## Lo que sigue en pie sin cambios

* la física conserva: cero unidades destruidas en las 24 combinaciones;
* el coste separa la clase: 22 niveles distintos, 4 puntos no dominados;
* **el precio mueve la decisión**: la postura elegida pasa de `[0, 0, 0.5]` a `[0, 0, 0]` en λ = 0,5;
* la caducidad retira 352.352 unidades y el presupuesto se activa;
* el control fiel de 156 semanas es inerte y plano.

## Lo que esto reabre, y lo que no

**No reabre** la familia como vía de aprendizaje: para que una política secuencial pagara aquí
haría falta que el óptimo se moviera **dentro** del episodio, y no hay ninguna señal de que se
mueva **entre** tapes, que es la condición más débil de las dos.

**Sí obliga** a que cualquier sucesor evalúe una clase que de verdad reaccione —umbral sobre
backlog, regla sobre la fase estacional, o MPC de creencia— antes de volver a usar la palabra
«secuencial». Y a decir en el manuscrito qué clase se probó, no sólo qué actuador.
