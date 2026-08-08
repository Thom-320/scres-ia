# Enmienda de alcance — el gate estacional midió mucho menos de lo que su nombre dice

**El resultado se conserva y no se reejecuta.** `results/seasonal_r2_headroom_gate/result.json`
(sello `5bb556d3…`, commit `223c9d6d`) sigue en el repositorio con sus 3.600 episodios y sus ocho
falsadores. Lo que se retira es **su titular y tres frases mías**, no su medición.

Un auditor externo señaló los tres defectos. Los tres los verifiqué en el código y **los tres son
ciertos**. El crédito es suyo.

## 1. Lo que se retira, literalmente

| retirado | por qué |
|---|---|
| `STOP_NO_HEADROOM_UNDER_GARRIDO_PHYSICS` | falso en los tres términos: no es «no headroom», no es «bajo la física de Garrido», y no es un techo general |
| *«es un techo: ninguna política observable puede batir a una que ya conoce el régimen»* | **falso**. El techo es sobre posturas **constantes dentro del episodio** indexadas por el régimen. Una política que actúa por paso —S1 en el valle, S2/S3 en demanda alta, preposicionar antes de R21, reservar surge para R24, reaccionar al backlog— **no pertenece a esa clase**. `max` sobre posturas constantes no acota `max` sobre secuencias de acciones |
| *«si ni siquiera un oráculo inflado supera la barra, ninguna política observable puede»* | misma falla, y era la frase que sostenía toda la fuerza del STOP |

## 2. El defecto que anula el titular, medido

`flow_fill_rate` **no cobra ningún recurso**: ni horas-turno, ni inventario mantenido, ni capacidad
ociosa. Medido en `D1|R_fixed`, contexto `R1r+R2r`, semilla 8600001:

| buffer | señal turno | `flow_fill_rate` | horas-turno |
|---:|---:|---:|---:|
| 0,00 | −1,00 | 0,7214 | **4.368** |
| 0,00 | 0,00 | 0,8404 | 8.736 |
| 0,00 | +1,00 | 0,8404 | **13.104** |
| 0,50 | −1,00 | 0,8404 | 4.368 |
| 1,00 | +1,00 | 0,8404 | 13.104 |

**La misma meseta de servicio se compra con 4.368 horas o con 13.104 — tres veces el recurso.**
Sólo el rincón `(buffer 0, S1)` es malo; todo lo demás satura. Con un endpoint que regala recursos,
la postura maximalista domina **con independencia del estado**, una sola constante se sienta en la
meseta en todos los regímenes, y `H_regime = 0` sale **por construcción del endpoint**, no por una
propiedad del entorno.

Ése es el defecto, y no se repara añadiendo un peso de costes elegido después de ver el resultado.
Se repara con una **restricción física**: todas las políticas reciben exactamente las mismas
`shift_hours` e `inventory_hours`. Las tres cantidades ya existen en el panel
(`episode_metrics.py:327-329`), así que la reparación no necesita física nueva.

## 3. `D1` no es la física de Garrido, y el nombre tiene que decirlo

Verificado en `supply_chain.py:5494-5496`: la demanda **realizada** es
`U(2400,2600) × demand_mean_multiplier × perfil_periódico(now)`, donde el perfil es nuestro, de 12
semanas, once de meseta y un valle. `alpha` y `gamma` sólo alimentan `observe()` → el **pronóstico**
`GR`, que **nunca toca la senda realizada**. La Ec. (1) del paper presenta `GR` como el
gross-requirements que alimenta las decisiones APP, no como esa demanda multiplicada por un perfil
reconstruido.

`D1` pasa a llamarse **`researcher_defined_periodic_demand_v1`**. Sigue siendo física nueva y sigue
siendo nuestra asunción declarada — que es la regla de la casa — pero **no se le atribuye a
Garrido**. Implementar una senda realizada verdaderamente dependiente de `(α, γ)` es trabajo aparte
y se declara como tal.

## 4. `flow_fill_rate` tampoco es flujo

`episode_metrics.py:218`: `delivered = sum(o.quantity for o in served)` — suma la cantidad
**original de los pedidos completamente servidos**, no las raciones parcialmente entregadas. Es un
**completion-weighted service ratio**, no throughput. Se renombra en el reporte y el sucesor usa
demanda no servida integrada.

## 5. `f9` es demasiado débil, y así queda dicho

`f9` pasa porque existe **un** rincón claramente malo, `(buffer 0, S1)`. El resto es meseta. La
existencia de spread no garantiza una frontera decisional útil. El sucesor necesita:

> al menos tres calendarios **con presupuesto igual** son no dominados, y el desempeño no se explica
> por un único rincón de baja capacidad.

## 6. El titular correcto

> **`STOP_NO_REGIME_SELECTION_VALUE_AMONG_UNBUDGETED_CONSTANT_POSTURES_UNDER_A_RESOURCE_FREE_ENDPOINT`**

En prosa, y es lo único que la medición sostiene:

> Conocer el régimen no ayuda a elegir entre 25 posturas constantes cuando turnos e inventario son
> gratuitos. **No adjudica** políticas que varían en el tiempo, calendarios con presupuesto igual,
> ni aprendices que actúan dentro del episodio.

## 7. Y por qué «añadir riesgos» no lo arregla por sí solo

Si S3 y el buffer máximo siguen siendo gratuitos, también dominarán bajo R1, R2, escalada o demanda
estacional. Los riesgos crean headroom sólo si **el recurso es escaso**, **su valor marginal cambia
en el tiempo**, y **la política tiene una señal observable para asignarlo**. Las tres condiciones,
no una.

Eso también explica, sin apelar a nada más, por qué `R_esc` no cambió nada: escalar el riesgo no
crea escasez cuando la capacidad no tiene precio.

## 8. Qué queda vivo

El gate v2 con panel de cinco métricas y seis celdas **sigue siendo válido en lo que mide**, con el
mismo alcance corregido: sus dos endpoints que deciden (`service_deficit`, `service_deficit_es10`)
**sí** cobran el retraso y **no** se pueden mejorar abandonando —`f10` lo mide en −1,0—, pero
**tampoco cobran el recurso**. Se reporta con el alcance de §6.

El sucesor es una familia nueva —presupuesto congelado × timing × riesgos con actuador alineado— y
**no se corre hasta congelar presupuesto, riesgos, endpoint, clases de política y los falsadores de
no-dominación**.
