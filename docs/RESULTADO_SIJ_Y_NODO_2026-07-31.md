# Resultado — `S_ij` dice dónde va el nodo: **op12, y la variable es el PERIODO, no la cantidad**

**Artefacto:** `results/sensitivity/second_order_risk_search_v1/result.json` (sello
`8ea65af8f227bac4…`) · 6.552 corridas · métrica **`ret_excel_risk_conditional`** (65× más
resolución de headroom que `ret_excel`) · **los cinco falsadores pasan**.

## 1. Dónde va el nodo

Los índices de **segundo orden** cruzados —decisión × riesgo— son los únicos que hablan de
política: un par alto significa que **el ajuste correcto de esa palanca depende del entorno**.

| par decisión × riesgo | `S_ij` |
|---|---:|
| **`op12_rop` × `impact_R1r`** | **+0,219** |
| `op9_rop` × `impact_R1r` | +0,180 |
| `op10_rop` × `impact_R1r` | +0,122 |
| `op12_rop` × `freq_R2r` | +0,112 |
| `op9_rop` × `freq_R2r` | +0,088 |
| `op12_q_max` × `impact_R1r` | +0,074 |
| `op12_rop` × `impact_R2r` | +0,065 |

**El patrón es inequívoco y tiene dos partes:**

1. **La interacción está en los PERIODOS de despacho (`_rop`), no en las cantidades.**
   `op12_rop` encabeza y aparece tres veces en el top siete; `op12_q_max` cae al sexto puesto con
   un tercio del acoplamiento.
2. **El acoplamiento es con el IMPACTO de R1r**, no con la frecuencia y no con R3. Tres de los
   cuatro primeros pares llevan `impact_R1r`.

> **El nodo va en `op12` —el último tramo aguas abajo— y la variable de decisión es su
> PERIODO de despacho, condicionada al impacto de R1r.**

**Esto corrige la lectura de primer orden.** El mapa anterior nombró `op12_q_max` (la
**cantidad**) como el único candidato, porque miraba `S_T − S1`. El segundo orden dice que quien
se acopla con el riesgo es el **periodo**. Son preguntas distintas —«¿cuánta interacción tiene?»
contra «¿con quién?»— y la segunda es la que decide dónde poner una variable de decisión.

Coherente además con la física: un periodo de despacho es *cuándo* sales, y ante una disrupción
lo que hay que cambiar es el **momento**, no el tamaño del envío.

## 2. La búsqueda sobre los hiperparámetros de riesgo

Tres ejes, como pediste: **activación, ocurrencia e impacto**. Activación y ocurrencia son el
mismo eje en su límite —un multiplicador de frecuencia en 0 es un riesgo desactivado— y `f2` lo
**verifica**: frecuencia `1e-6` da exactamente el mismo resultado que `risks_enabled=False`.

Sobre 24 configuraciones muestreadas: **`H_regime` = 5,1e-5**.

**Y la configuración que más aporta no es la extrema.** Es de rango medio en todo:

    freq_R1r 0,62   freq_R2r 0,84   freq_R3 0,86
    impact_R1r 1,12  impact_R2r 0,98  impact_R3 1,30

**Subir los riesgos no maximiza el headroom.** Un régimen moderado —frecuencias por debajo de 1,
impactos ligeramente por encima— es el que más valor da a conocer el estado. Tiene sentido: con
riesgos saturados todo se degrada por igual y la decisión deja de importar; con riesgos nulos no
hay nada que decidir.

## 3. Lo que NO se puede concluir

* **Los 5,1e-5 de aquí NO se comparan con los 1,8e-4 de la mezcla de familias.** Distinta
  métrica (condicional a riesgo contra `ret_excel`), distinta definición de régimen (24
  configuraciones de hiperparámetros contra 7 conjuntos de familias) y rejilla más gruesa. Decir
  «bajó» sería un error de lectura.
* `sum(S1) = 0,569` — **la superficie ya no es aditiva** bajo esta métrica y este espacio, al
  contrario que el 0,985 sobre `ret_excel`. Es coherente con que la métrica condicional a riesgo
  exponga acoplamiento que la canónica promedia.
* `H_regime` sigue suponiendo el régimen **observado**: techo, no alcanzable.

## 4. El siguiente paso, ya concreto

Añadir en `op12` una variable de decisión de **periodo de despacho** que pueda **condicionarse
al estado de riesgo**, y medir si una política —MPC o RL— captura parte de ese acoplamiento de
`S_ij = 0,219` frente a la mejor constante. Es la primera vez en toda la campaña que hay un
**par nombrado** al que apuntar en vez de un espacio que explorar.
