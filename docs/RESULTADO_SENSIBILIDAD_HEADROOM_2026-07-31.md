# Resultado — el mapa de headroom: ni resolución ni buffers. El régimen de riesgo lo decide todo

**Artefacto:** `results/sensitivity/headroom_map_v1/result.json` (sello `6a16e3263e1ccaf6…`) ·
**Contrato:** `docs/PREREGISTRO_SENSIBILIDAD_HEADROOM_2026-07-31.md` · 20 factores, **7.732
corridas**, los **seis falsadores pasan**.

## 0. Antes del resultado: la primera corrida se auto-refutó

`ret_excel` es de cola pesada por construcción —la rama `0,5/RPj` no está acotada, y su propia
CF12 lleva una fila en **160,26**—. Una descomposición de varianza sobre una salida así la
dominan un puñado de sorteos, y eso fue exactamente lo que pasó: **`S1 = −5,75` para `op9_q_max`
y `sum(S1) = −5,08`**, ambos muy fuera de `[0,1]`.

El estimador no es el culpable: reproduce **Ishigami** con error **8e-4**, incluido el caso de
pura interacción (`S1 = 0`, `S_T = 0,244`). **La métrica lo es.** Se repite la descomposición
sobre el **transformado por rangos** —remedio estándar: conserva toda relación monótona y quita
apalancamiento a la cola—. Los índices crudos quedan en el artefacto **como la evidencia de por
qué no se pueden usar**, y `f5` (índices dentro de `[0,1]`) es el falsador que faltaba en mi
propio contrato.

> **Hallazgo metodológico para el paper:** un análisis de sensibilidad basado en varianza sobre
> `ret_excel` **sin transformar no es interpretable**. Cualquiera que publique índices de Sobol
> sobre esta métrica sin decirlo está reportando ruido de cola.

## 1. El mapa

`sum(S1) = 0,985` — **la superficie es aditiva al 98,5%.**

| factor | grupo | `S1` | `S_T` | `S_T − S1` | desplazamiento `argmax` |
|---|---|---:|---:|---:|---:|
| **`risk_family_selector`** | **riesgo** | **0,736** | **0,755** | 0,019 | — |
| `op10_q_max` | aguas abajo | 0,035 | 0,078 | 0,042 | 0,25 |
| **`op12_q_max`** | **aguas abajo** | 0,008 | 0,069 | **0,061** | **0,50** |
| `op9_q_max` | batallón | 0,060 | 0,068 | 0,007 | 0,25 |
| `demand_level` | demanda | 0,027 | 0,052 | 0,025 | — |
| `op10_rop` / `op12_rop` / `op9_rop` | despacho | ≈0,03 | ≈0,045 | 0,014–0,039 | 0,12–0,38 |
| `risk_frequency_scale` | riesgo | −0,005 | 0,021 | 0,026 | — |
| **`op3_rm`, `op5_rm`, `op9_rations`** | **buffers** | ≈0,01 | **≈0,006** | ≈0 | 0,38–0,88 |
| `op1_rop`, `op2_q`, `op2_rop`, `op3_q`, `op3_rop`, `batch_size`, `assembly_shifts` | resto | ≈0,005 | ≈0,004 | ≈0 | — |

## 2. Las tres respuestas que se pidieron

**¿Nodos o continuidad? Ninguno de los dos, en este espacio.** Con el 98,5% de la varianza en
efectos de primer orden, ni más resolución ni una política aprendida pueden batir a una
constante por factor. Y —esto es lo que no esperaba tan rotundo— **los tres buffers son
prácticamente inertes**: `S_T ≈ 0,006`, tres órdenes de magnitud por debajo del factor de
riesgo. **Añadir más nodos de buffer no puede generar headroom**, porque los que ya hay no mueven
la métrica.

**¿Dónde poner la próxima variable de decisión?** Bajo la regla declarada (interacción > 0,05 **y**
`argmax` que se mueve > 20% de su rango entre regímenes), **sobrevive exactamente uno**:

> **`op12_q_max`** — la cantidad de despacho del **último tramo aguas abajo**, con interacción
> 0,061 y un óptimo que se desplaza el **50%** de su rango entre familias de riesgo.

Segundo escalón, sin cumplir ambos criterios: `op10_q_max` (interacción 0,042) y `op9_rop`
(0,039). **Todo el headroom que existe está aguas abajo**, en el despacho, no aguas arriba ni en
inventario.

**¿Qué riesgos importan?** **Cuál familia está activa es el 75,5% de la varianza total** — más
que los diecinueve factores restantes juntos. En cambio **escalar frecuencia (0,021) o impacto
(0,006) casi no aporta nada**. Con su permiso para editar riesgos, la palanca **no** es subir
multiplicadores: es **qué riesgos coexisten**.

## 3. Predicciones: dos aciertos, una fallida, y la fallida es la útil

| # | predicción | resultado |
|---|---|---|
| 1 | `Σ S1 > 0,85` (superficie aditiva) | ✅ **0,985** |
| 2 | los factores de **riesgo** tendrán la mayor `S_T − S1` | ❌ **falló**: `risk_family_selector` tiene interacción de solo 0,019. Las mayores interacciones son de **despacho aguas abajo** |
| 3 | buffers con `S_T ≈ S1`, aditivos | ✅ y más fuerte: son **inertes** |
| 4 | el `argmax` no se moverá entre regímenes | ✅ **parcialmente**: se mueve poco en casi todo, pero **sí** en `op12_q_max` (0,50) y `op5_rm` (0,88) |

La 2 es la que aporta: **el riesgo domina el nivel pero no la interacción**. Manda el resultado,
no crea acoplamiento con la decisión. Por eso una política dependiente del estado no encuentra
qué explotar: el régimen mueve el suelo, no la pendiente.

## 4. Lo que esto implica para la instrucción de Garrido

Pidió **añadir nodos aguas arriba y abajo**. El mapa dice, medido:

* **aguas arriba (`op1`, `op2`, `op3`, buffers): `S_T ≤ 0,006`.** Nodos nuevos ahí serían inertes
  igual que los existentes. **No es donde hay que gastar.**
* **aguas abajo (`op9`, `op10`, `op12`): toda la interacción que existe.** Si se añade un nodo,
  va **ahí** — y la variable concreta con headroom es la **cantidad de despacho del último
  tramo**.
* **la palanca más grande que nadie ha tocado es de régimen, no de topología:** Garrido corre
  **una familia de riesgo por vez**. **Mezclar familias** —R1r y R2r simultáneas— es el cambio
  que más varianza mueve, está dentro de sus permisos, y **no existe en su diseño ni en el
  nuestro**.

## 5. Límites, declarados

* Horizonte **52 semanas**, no sus 10–20 años: los valores **no** se comparan con los suyos.
* Métrica **`ret_excel`** transformada por rangos. **El índice Cobb-Douglas queda pendiente**: su
  `CobbDouglasRecorder` exige muestrear por periodos, lo que multiplica el coste por ~300, y el
  contrato lo declaró para el subdiseño Morris. **No se corrió**, y decirlo es parte del
  resultado.
* La regla de lectura era preregistrada; `op12_q_max` la cumple por poco (0,061 contra 0,05).
  **Un candidato marginal es un candidato**, no una conclusión.
