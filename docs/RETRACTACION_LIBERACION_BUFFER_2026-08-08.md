# Retractación — la «liberación» del buffer no liberaba, y el coste no era coste

**Fecha:** 2026-08-08. Afecta a los commits `c9dec266`, `46e94f2d` y `b901789`, todos de hoy y todos
míos. Los artefactos se **conservan y se etiquetan**; no se borran.

## 1. Los dos defectos, reproducidos antes de escribir esto

**Con objetivos activos libera exactamente cero.** `_buffer_container_keys()` recorría las **claves
del contrato** (`op3_rm`, `op5_rm`, `op9_rations`) con `getattr`. Esas claves no son atributos de
contenedor — los contenedores se llaman `raw_material_wdc`, `raw_material_al`, `rations_sb` — así
que la lista salía vacía justo cuando había buffer que liberar:

```
targets: {'op5_rm': 600000.0, 'op9_rations': 20000.0}
claves recorridas: []
liberado: 0.0
```

**Sin objetivos declarados destruye inventario operativo.** El fallback vaciaba `rations_al`,
`rations_sb` y `rations_cssu` **a cero**, con objetivo por defecto 0,0. Medido sobre una corrida de
26 semanas:

```
stock al cierre: rations_sb 2600, rations_cssu 88
destruido por el fallback: 2688.0  ->  todo a cero
```

Ese stock nunca fue buffer estratégico.

**Y el coste tampoco era coste.** `inventory_hours` sumaba semanas con el interruptor encendido por
la longitud del paso. Nunca tocaba inventario: cobraba lo mismo por un nodo con diez unidades que
por uno con cien mil, y no cobraba nada por el stock que seguía en mano después de apagar.

## 2. Qué queda retractado

`PRICED_DECISION_SPACE_ELIGIBLE` (`46e94f2d`) y el techo con precio (`b901789`) **no establecen un
espacio de decisión con precio**. El trade-off que medían venía de **destruir inventario**, no de
dejar de sostenerlo, contra un proxy de coste mal nombrado.

**Incluida una frase mía de esta misma sesión.** Al retractar el techo clarividente escribí que
«el espacio priced sí sobrevive: 21 niveles, 6 puntos no dominados, óptimo que se mueve con λ». **Esa
frase cae también.** Sobrevivió tres mensajes porque miré el techo y no la física que lo sostenía.

Lo que **no** se ve afectado: la validación positiva del audit
(`results/audit_positive_validation/result.powered.json`) corre sobre el banco sintético
`contention_v1` y no importa `supply_chain.py`. Y la prueba de inercia de pines pasó legítimamente
**porque el modo por defecto es `none`** y esta ruta nunca se ejecutó.

## 3. El sucesor, y por qué es conservativo

**Se rechaza el valor en vez de repararlo en silencio.** `strategic_buffer_release_mode="immediate"`
ahora **lanza** un `ValueError` que nombra esta retractación. Un modo que hizo lo contrario de lo que
decía no debe seguir siendo aceptado con otra implementación detrás del mismo nombre.

**Bajar el objetivo detiene la reposición y no evapora nada.** Es la física que ya existía: el
top-up sólo añade hasta el objetivo, así que apagar deja de inyectar y el stock se consume por la
vía normal. **La duración de la ventana sigue importando** — más semanas encendido son más
inyecciones y más unidad-horas — así que la clase de decisión no colapsa; simplemente deja de
apoyarse en la destrucción.

**El coste pasa a ser exacto y atribuible:** unidades kit-equivalentes que la política **repuso de
verdad**, `raw/12 + raciones`, leídas de los acumuladores del propio simulador. La integral física
cantidad×tiempo (`strategic_inventory_unit_hours`) viaja al lado **como sensibilidad, no como
precio**, porque el reparto de la tenencia entre buffer y operación no es limpio y fingir que lo es
fue exactamente el error anterior.

Ninguna de las dos magnitudes es una tasa monetaria, y ninguna viene de la tesis. Son asunciones
nuestras declaradas.

## 4. El gate sucesor, congelado antes de correrlo

* **Clase:** ventanas `(inicio, duración)` sobre 26 semanas, enumeradas y sin alias del
  «sin buffer».
* **Endpoint:** `L*` adimensional, sin cambios.
* **Precio:** `J(λ) = L* + λ · (unidades repuestas / máximo constructivo)`.
* **Frente de Pareto**, no un λ elegido. Ningún veredicto puede depender de escoger λ.
* **Train/test disjuntos**, con el control fijo elegido **sólo en train** — el benchmark anterior lo
  elegía sobre las mismas trayectorias donde medía el hueco.
* **Semillas de desarrollo ya quemadas** (`8600001+`). No se abre ningún bloque virgen.

**Falsadores que pueden fallar:** cero unidades destruidas; conservación de masa; la clase tiene más
de un nivel de coste distinto; una ventana apagada cuesta exactamente cero; el hueco clarividente
contra su nulo de interacción; y el control fijo seleccionado en train evaluado en test.

**Si no queda hueco material, no se entrena.** Eso es un resultado, no un contratiempo.
