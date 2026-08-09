# Preregistración de la prima neural — congelada 2026-08-08

**Por qué existe.** «Intentemos lograr prima neural» está mal formulado. Una prima neural no se
logra: se construye un contrato en el que exista una decisión secuencial material, el estado sea
parcialmente observable, la historia contenga información incremental, las acciones tengan autoridad
física y el control estructurado **no** absorba todo el valor — y entonces se mide si una red captura
el residual. Mover ruido, capacidad, demanda, observación o riesgo hasta que un aprendiz gane es
ingeniería del resultado.

Este documento congela el estimando, la frontera de comparadores, los presupuestos y las reglas de
parada **antes** de que exista ningún contrato nuevo, para que ninguna respuesta de dominio pueda
llegar acompañada de una definición conveniente.

## 1. El estimando

Para un presupuesto físico y computacional `B`:

```
Delta_N(B) = max{ V(pi) : pi en S u N, R(pi) <= B }
           - max{ V(pi) : pi en S,     R(pi) <= B }
```

* `S` — la familia estructurada y model-based.
* `N` — la familia neural.
* `R(pi)` — recursos: derechos físicos, llamadas al DES, parámetros, FLOPs y wall-clock.
* `V` — el endpoint congelado del contrato.

La resta contra **el mejor miembro de `S` bajo el mismo presupuesto** es lo que elimina el truco
habitual: declarar prima neural porque una red venció a una regla mediocre.

## 2. SESOI y promoción

```
SESOI                      +0.01 en el endpoint congelado
promoción                  LCB95(Delta_N) >= +0.01
                           sin violación de cola, de peor producto ni de recursos
bloque                     held-out fresco, nunca abierto
derechos                   físicos e informacionales idénticos entre brazos
tapes                      las mismas del DES para todos los brazos
presupuesto de evaluación  idéntico entre brazos
```

## 3. La frontera estructurada, que se corre ANTES que cualquier red

```
mejor postura fija
regla de umbral / política de índice
earliest-deadline-first y su versión ponderada
MILP de horizonte rodante
MPC bayesiano (creencia)
MPC robusto
programación dinámica en instancias pequeñas, donde sea exacta
```

Si ninguna de estas convierte la señal, una red no descubrirá estructura: aprenderá ruido con
mucha convicción.

## 4. Los brazos neurales, en este orden

```
MLP
MLP recurrente / PPO recurrente
KAN
híbrido MPC-neural (opcional)
```

Para **KAN frente a MLP** se reportan tres comparaciones **separadas**, nunca una: parámetros
emparejados, FLOPs emparejados, y wall-clock más número de llamadas al DES emparejados. Un KAN de
500 parámetros contra un MLP absurdamente estrecho no demuestra arquitectura; demuestra que el
presupuesto se convirtió en una trampa.

## 5. Los gates, en orden, y cada uno puede detener

| gate | qué exige | criterio |
|---|---|---|
| **1 · autoridad física** | la acción cambia el flujo previsto, conserva masa, respeta latencia y no crea recursos | falsadores de conservación |
| **2 · headroom privilegiado** | diferencia material entre decidir con información perfecta y la mejor postura fija | `LCB95(H_PI) >= 0.02` |
| **3 · convertibilidad observable** | una política no anticipativa simple convierte parte | `LCB95(H_obs) >= 0.01` |
| **4 · saturación estructurada** | queda residual tras la frontera de §3 | `LCB95(H_res) > 0` |
| **5 · retención** | el estado retenido aporta sobre su gemelo sin memoria | `LCB95(H_ret) > 0` |
| **6 · prima neural** | §1 y §2 | `LCB95(Delta_N) >= +0.01` |

**Regla de parada.** Sin headroom observable en el gate 3, o con la frontera estructurada absorbiendo
el residual en el gate 4, **no se entrena**. Ese es el resultado, no un contratiempo.

## 6. Falsadores obligatorios de todo el protocolo

Cada gate viene con un control que debe **poder fallar** y ser visto fallando:

* **placebo desinformado** en toda medición de headroom — una señal barajada por el mismo mapa;
* **nulo de interacción** para cualquier hueco clarividente, con los residuos permutados sobre el
  modelo aditivo, nunca dentro de fila;
* **control de historia equivocada** para cualquier claim de memoria;
* **conservación exacta** de masa y recursos entre brazos;
* **un control que debe diferir**, para probar que el comparador puede ver una diferencia.

Este último no es retórico: el 2026-08-08 una prueba de inercia reportó «0 diferencias» porque
ambos lados fallaban idénticamente, y sólo lo cazó un control obligado a diferir.

## 7. Lo que este documento NO autoriza

No autoriza entrenar nada. No autoriza abrir un bloque de semillas. No autoriza un contrato nuevo.
Un contrato nuevo requiere una respuesta de dominio del árbol de decisión de
`docs/DOMAIN_QUESTION_ADJUDICATION_2026-08-08.md`, su propia preregistración y su propia
autorización del PI.
