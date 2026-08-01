# Resultado — **H1′ sostenida**, H3′ no. Y el camino hasta aquí es parte del resultado

**Artefacto:** `results/manuscript/h1_h3_v2/result.json` (sello `312f91a548d8639c…`,
`H1_SUPPORTED__H3_NOT_SUPPORTED`) · **los siete falsadores PASAN** · preregistro
`docs/PREREGISTRO_H1_H3_V2_2026-08-01.md`, commiteado antes de correr.

## 1. H1′ — sostenida, y con margen

Servicio perdido acumulado (`service_loss_auc_ration_hours`, **menor es mejor**), evaluando lo que
**cada estrategia desplegó realmente en cada celda** (contexto × réplica), pareado por semilla:

| brazo | ración-hora perdidas |
|---|---:|
| **híbrido** (neurona con memoria) | **45.358.777** |
| reinicio | 79.746.843 |
| **estático** (OFAT, el diseño de la tesis) | **107.033.239** |

| contraste | ventaja del híbrido | IC95 | n |
|---|---:|---|---:|
| **primario — las 72 celdas** | **+61.674.462** | **[+38.932.760, +87.248.013]** | 360 |
| secundario — las 42 celdas con configuraciones distintas | +105.727.650 | [+68.638.033, +148.946.420] | 210 |
| híbrido vs reinicio | +34.388.066 | [+18.072.049, +53.864.222] | 360 |

El primario **incluye las 30 celdas donde ambos despliegan lo mismo**, que aportan exactamente
cero y arrastran el contraste hacia el nulo. Aun así el `LCB95` está muy lejos del cero.
**H1′ sostenida en el criterio primario**, no sólo en el subconjunto favorable.

**Lo que H1′ NO dice, y va en el paper con estas palabras:** `service_loss_auc` **no es un tiempo
de recuperación**. Es la **integral del servicio perdido**, que mezcla magnitud y duración. Es una
operacionalización defendible de *«se recupera mejor»*; no es el estimando que `H1` enuncia.

## 2. H3′ — no sostenida, y el orden es lo interesante

Varianza del **coste de búsqueda entre los seis contextos** (menor es mejor):

| estrategia | varianza |
|---|---:|
| **memoria** | **39,92** |
| reinicio | 47,78 |
| OFAT | 58,22 |
| aleatoria | 70,66 |

| contraste | diferencia | IC95 |
|---|---:|---|
| memoria vs reinicio | +7,86 | **[−14,72, +28,31]** |
| memoria vs OFAT | +18,29 | **[−5,11, +41,05]** |

**Ambos IC cruzan el cero: `H3′` NO sostenida** bajo la regla fijada de antemano.

El orden es **monótono en la dirección esperada** —memoria < reinicio < OFAT < aleatoria— y eso es
sugerente, pero **con 12 réplicas no alcanza**. El limitante es potencia, no signo. Lo digo así y
no lo titulo como tendencia: la regla decía `LCB95 > 0` y no se cumple.

## 3. Tres intentos de `f3`, y los tres fallos son informativos

El falsador que comprueba que la métrica no está censurada **falló dos veces por mi culpa antes de
estar bien**, y merece quedar escrito:

| intento | qué comprobaba | por qué estaba mal |
|---|---|---|
| 1 | `n_served + n_lost == n_orders` | identidad contable **no relacionada**: un pedido puede no ser ninguna de las dos cosas — sigue pendiente al horizonte, y **ésos son justo los que la integral debe capturar** |
| 2 | recalcular sobre **todos** los `sim.orders` | fallo **mío**: el panel excluye legítimamente los pedidos anteriores al fin del warm-up, *«so the metric reflects only the period the policy could influence»* — 33 de 311, y cargan enorme tardanza porque la cadena aún no había arrancado |
| 3 ✅ | replicar **la población exacta del panel**, exigir igualdad, **y** que los pedidos nunca completados estén **dentro** con aporte positivo | la propiedad que de verdad importa |

**El arreglo fue hacer el test correcto, no más laxo:** la versión final exige **dos** cosas, y la
segunda es la que separa esta métrica de `ret_excel` y de `system_ttr` — **los pedidos que nunca
se completan viven dentro de la población puntuada**, en vez de desaparecer de ella.

Es el **tercer** falsador que corrijo en dos días por el mismo motivo de fondo: comprobar la
propiedad *adyacente* en vez de la que da sentido al número. El patrón que sí funciona quedó
claro: **recalcular la cantidad de forma independiente y comparar**, nunca inferirla de un
invariante vecino.

## 4. Estado de las cuatro hipótesis del borrador

| | estado |
|---|---|
| **H1′** servicio perdido acumulado | **SOSTENIDA** — +61,7 M ración-hora, `LCB95` +38,9 M |
| **H2** curva de aprendizaje | **medida** — ventaja +0,00 → +10,00 entre contextos |
| **H3′** varianza del coste de búsqueda | **NO sostenida** — orden monótono, IC cruza cero |
| **H4** dependencia de trayectoria `L_{t−1}` | **medida** — +7,90 corridas [+6,88, +8,93] |

`H1` y `H3` en su redacción **original** siguen siendo **no evaluables** en este entorno, por las
razones de `docs/RESULTADO_H1_H3_2026-08-01.md`: `system_ttr` censurado al 100 % y un óptimo que no
se mueve. Las dos versiones primadas son **reformulaciones declaradas**, no reparaciones, y el
manuscrito tiene que presentarlas así.

## 5. Un resultado sobre el diseño de Garrido que no buscaba

En **30 de las 72 celdas (42 %)** el diseño OFAT de su tesis converge **exactamente** a la misma
configuración que el aprendiz. Su procedimiento un-factor-a-la-vez **encuentra el mismo óptimo casi
la mitad de las veces** — lo que cuesta es *cuándo* lo encuentra (12,42 corridas contra 6,99) y
*cuánto pierde por el camino* en las celdas donde no converge.

Eso es un resultado favorable a su método, y va dicho aunque `H1′` salga a favor del híbrido.
