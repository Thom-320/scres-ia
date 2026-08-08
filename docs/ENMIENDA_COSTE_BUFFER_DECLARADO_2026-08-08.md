# Enmienda — el precio del buffer, declarado como asunción nuestra y con su precio de fidelidad

**Escrita ANTES de correr.** Runner: `scripts/run_priced_buffer_gate_v1.py`. Custodia: réplica
declarada, sin semillas nuevas. Usa `supply_chain/falsifiers.py`, así que las comprobaciones
aprendidas hoy se heredan en vez de reescribirse, y un `passed` literal **no se puede construir**.

## 1. Por qué el coste puede existir ahora y no antes

Hasta que aterrizó la ruta de liberación, apagar el objetivo estratégico dejaba las unidades ya
entregadas en su sitio: `K = 4` y `K = 26` daban un **idéntico 0,302680**. Cobrar por horas-inventario
entonces habría puesto precio a algo que la política **no podía controlar** — peor que no cobrarlo.

Con liberación, el mismo bloque va de **0,541315 a 0,376187**. La duración del sostenimiento es una
decisión real, y ponerle precio significa algo.

## 2. El coste es asunción nuestra, y NO es una tasa monetaria

Garrido-Ríos (2017) **excluye el coste deliberadamente** (p.147) y lo lista como extensión futura
**sin valores** (p.148). Inventar una cifra en moneda sería fabricar procedencia — el modo de fallo
que este proyecto ya midió al copiar los exponentes de Cobb-Douglas entre escalas.

El coste entra en **las unidades del propio endpoint**:

```
J(λ) = L*  +  λ · (horas_inventario / max_horas_inventario)
```

`λ` es un **tipo de cambio** entre inventario sostenido y exposición de demanda no servida.

**La referencia `λ = 1` es la convención de igualdad en el máximo**: sostener el buffer máximo
posible durante todo el horizonte cuesta exactamente lo mismo que la exposición total. Es la lógica
*«cada argumento equiparado en su máximo»* de la propia §3.4 de Garrido (IJPR 2024), aplicada donde
**sí** es defendible —a dos cantidades que medimos nosotros— y no a cinco variables sobre máximos
prestados, que es como esa misma regla acabó dando a κ̇ un peso 7,38× el de ζ.

## 3. Y la respuesta se reporta como función de λ, no en un punto

Se barre `λ ∈ {0 · 0,25 · 0,5 · 1 · 2 · 4}` y se reporta el `K` óptimo en cada uno. **El λ de
equilibrio —donde cambia la duración óptima— se DERIVA de los datos, no se asume.** Un lector que
rechace nuestra referencia puede leer la suya en la misma tabla.

## 4. El precio de fidelidad, declarado

**Ni la liberación ni el lead time tienen evento fuente.** La tesis repone periódicamente (p.107) y
nunca describe qué ocurre al bajar el objetivo; sus 48 h (p.111) son el lead time de **entrega al
usuario**, no de reconstrucción del buffer. Las 336 h que usamos son **nuestras**.

Por tanto: **todo resultado bajo este contrato es nuestro y nunca se presenta como reproducción de
Garrido-Ríos (2017).** El modo por defecto del simulador sigue siendo `release_mode = "none"`, así
que la física congelada no cambia para ningún artefacto anterior.

## 5. Qué decide el gate, y qué no

**No es un resultado.** Es elegibilidad: dice si existe una decisión que medir, no si hay headroom
observable ni si alguna arquitectura lo captura.

| falsador | por qué puede fallar |
|---|---|
| `p1`–`p4` (heredados) | endpoint muerto · espacio de un bit · reset que consume el horizonte · escenario que no es el declarado |
| `f5_pareto_front_is_wide_enough` | un frente de uno o dos puntos es «sostener o no», el espacio colapsado que ya midió el benchmark |
| `f6_price_actually_moves_the_optimum` | si el `K` óptimo es el mismo en todo λ, **el precio es inerte** y el coste es decoración, no variable de decisión |
| `f7_release_actually_fired` | con liberación activada algún calendario debe liberar stock; cero liberado significaría que el cambio de física es inerte y todo el precio descansa sobre nada |

Las tres divulgaciones —la ausencia de procedencia del coste, la convención de λ y la custodia—
van en **campo propio y no cuentan** en el total de falsadores. Ésa es la regla que hoy se rompió
tres veces.
