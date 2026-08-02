# Enmienda 3 a G3c — los márgenes, re-derivados de operación en vez de del tamaño muestral

Repara el **bloqueador 2** de `docs/ENMIENDA_G3C_TRES_BLOQUEADORES_2026-08-01.md`. Ningún dato se
ha visto contra G3c: sigue `DESIGN_ONLY`.

## 1. El error, que es de tipo y es mío

Justifiqué `δ = 0,25` en `lost_orders` como **«4× la granularidad Monte Carlo»** del halt espurio
(0,0625 = 1 pedido / 16 semillas).

**Eso deja que el instrumento defina el criterio.** La resolución muestral informa **qué puedo
detectar** (potencia); no dice nada sobre **qué daño es aceptable** (margen). Con 64 semillas el
«margen justificado» se encogería cuatro veces sin que la operación cambiara en nada — y con 8
semillas se cuadruplicaría. Un margen que se mueve con `n` no es un margen.

## 2. Los márgenes re-derivados, cada uno en sus propias unidades

La unidad de operación es el **batallón-semana**: la cadena entrega raciones a dos CSSU durante 52
semanas, con una demanda anual observada de ~380.000–430.000 raciones por reclamante.

| guardarraíl | `δ` | derivación operacional, **sin referencia a `n`** |
|---|---:|---|
| `worst_claimant_fill` *(primario, SESOI)* | **0,010** | **un punto porcentual** del peor reclamante. Sobre ~400.000 raciones/año son ~4.000 raciones — más de **medio día** de consumo de un batallón. Por debajo de eso, ningún planificador cambiaría una decisión |
| `flow_fill_rate` | **0,005** | medio punto del agregado. Se fija **a la mitad** del primario a propósito: un candidato no puede comprar un punto en el peor reclamante hundiendo el agregado más de medio |
| `lost_orders` | **0,50 pedidos/episodio** | un pedido perdido es una **entrega no realizada a una unidad**. Medio pedido por episodio de 52 semanas ≈ **un pedido perdido cada dos años**. Ésa es la frecuencia por debajo de la cual el hecho no cambia una decisión de mando; **no** deriva de 1/16 |
| `backorder_qty_final` | **1,0 % relativo** | el backlog final varía de escala por celda, así que el margen es relativo. Un 1 % del backlog del titular es ruido operativo dentro del cierre de un ciclo de reposición |
| masa, capacidad creada, recursos programados | **0,0 exacto** | **identidades algebraicas**, no cantidades estocásticas. Único sitio donde el margen cero es legítimo, y por eso se conserva |

**`lost_orders` sube de 0,25 a 0,50** — y sube por una razón operacional, no para que algo pase.
El cambio se declara **antes** de correr G3c, sobre datos ya vistos sólo en un artefacto
**detenido**.

## 3. La granularidad Monte Carlo, en su sitio correcto

Se sigue **reportando**, pero como **límite de detección**, nunca como fundamento:

> Con `n` réplicas, la diferencia mínima resoluble en `lost_orders` es `1/n` pedidos. Si
> `1/n > δ`, el guardarraíl **no puede evaluarse** y el veredicto es `UNDERPOWERED`, **no** un
> pase.

Con `δ = 0,50` eso exige `n ≥ 2`, así que ya no es el binding constraint — que es exactamente la
señal de que el margen ahora lo fija la operación y no la muestra.

## 4. Alcance

Estos márgenes rigen **G3c** y **cualquier sucesor de G3-obs**, que heredó los anteriores. El
artefacto `results/headroom/g3_obs_conversion/result.json` **no se toca**: está `HALTED` y sus
guardarraíles pasaron con el margen mal fundamentado, cosa que su propio documento ya declara.

**Sigue pendiente** el bloqueador 1 (factorial `min_dwell` × `switch_cost` con niveles
especificados) y el 3 (verificar la identidad del brazo nulo con payload canónico). G3c sigue
`BLOCKED`.
