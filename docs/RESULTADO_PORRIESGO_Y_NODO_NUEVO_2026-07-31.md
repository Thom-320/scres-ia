# Resultado — el nodo nuevo aporta **cero**, y la descomposición por riesgo no converge

**Artefacto:** `results/sensitivity/perrisk_and_new_node_v1/result.json` (sello
`ad2c108cf5a08967…`, `HALTED_FALSIFIER_FAILED` por la etapa A) ·
**Runner:** `scripts/run_perrisk_and_new_node.py` · métrica `ret_excel_risk_conditional`, con
`ret_excel` al lado.

## Hueco 2 — el nodo aguas abajo que su modelo no tiene: **cerrado, y es un cero**

Garrido pidió **añadir buffers en nodos que su modelo no considera, aguas abajo**. El simulador
ya lleva exactamente eso, desactivado: la **reserva de emergencia de teatro** — un stock finito
situado **detrás del corredor aguas abajo**, reabastecido desde Op9 con lead time real y
**bloqueado mientras cualquier operación de ruta esté caída**. No es un buffer de juguete: es
conservador en masa y sensible a la ruta.

Se encendió, con sus propias variables de decisión (capacidad, retardo de emisión), y se midió
lo único que importa: **¿tener el nodo sube el valor de conocer el régimen?**

| | `H_regime` | nivel (mejor común) |
|---|---:|---:|
| **sin el nodo** | 0,000008 | 0,005533 |
| **con el nodo** | 0,000008 | 0,005533 |
| **delta** | **+0,000000** | **+0,000000** |

**Cero en headroom y cero en nivel.** Y no es que el nodo no haga nada: `f2` **verifica que sí
cambia la trayectoria** (las raciones entregadas varían). Cambia el sistema y **no cambia ni la
resiliencia ni el valor de conocer el estado**.

Estable en **tres corridas independientes** con distintos tamaños de muestra. Es el resultado
más limpio de toda la campaña de sensibilidad.

> **La instrucción del 28 de julio queda contestada en sus dos mitades, y ambas son negativas:**
> variables continuas — probado, no aporta; **nodos nuevos aguas abajo — probado, aporta cero.**

## Hueco 1 — por riesgo: **no converge, y no publico sus números**

Los 18 factores —frecuencia e impacto de cada uno de los nueve riesgos, que es el permiso que
Garrido nos dio, no la escala global que usábamos— **no convergen**. `f1` detiene el artefacto a
`N = 128` **y también a `N = 512`**, es decir con **10.240 corridas**.

**Y la forma del fallo diagnostica la causa.** A `N = 512`:

    sum(S1) = 0,035          <- prácticamente NADA de varianza de primer orden
    S_T = 0,40–0,72 en casi todos los factores, sumando muy por encima de 1
    solo 3 de 18 índices fuera de [0,1] (contra muchos a N=128)

Cuadruplicar la muestra **redujo el desbordamiento pero no cambió el cuadro**: ningún factor
actúa solo y el «total» de casi todos es alto. Eso no es interacción masiva — es **ruido por
episodio que el estimador atribuye como efecto total**.

**La conclusión es sobre el DISEÑO, no sobre los riesgos:** con una corrida por punto, este
simulador no resuelve 9 riesgos × 2 ejes. El arreglo no es más `N`, es **promediar semillas por
punto de diseño** (varias réplicas CRN por fila de A, B y AB) antes de estimar. Eso multiplica el
coste por el número de réplicas y es la corrida que cierra este hueco de verdad.

**No reporto el ranking como medición.** Sí anoto una señal débil que vale una prueba dirigida:
`freq_R24` e `impact_R24` —demanda contingente— encabezan `S_T` **de forma estable entre
`N = 128` y `N = 512`**. Estabilidad entre tamaños de muestra no es convergencia, pero sí es la
primera hipótesis que probaría con el diseño corregido.

## Lo que queda abierto, dicho sin adornos

* **Por riesgo**: pendiente de convergencia. No hay conclusión.
* **Clase de política**: sigue siendo un umbral de dos niveles sobre un observable. No se ha
  probado una política más rica — y la puerta declarada dice que ese gasto **no está autorizado**
  mientras la mínima no capture nada.
* **Un solo observable**: conteo de eventos R1r recientes. Severidad, tiempo desde el último
  evento, o estado de inventario podrían llevar más señal.

## Y el cuadro completo, que sí está cerrado

| pregunta | respuesta medida |
|---|---|
| ¿resolución continua? | no aporta (lineal **subió** a 0,982) |
| ¿nodos de buffer aguas arriba? | inertes, `S_T ≈ 0,006` |
| ¿nodo nuevo aguas abajo? | **cero**, headroom y nivel |
| ¿mezclar familias de riesgo? | **+50% de headroom**, y sigue en 1,8e-4 |
| ¿dónde está el acoplamiento? | `op12_rop` × `impact_R1r`, `S_ij = 0,219` |
| ¿lo captura una política? | **no: −20,5%** frente a la mejor constante |
