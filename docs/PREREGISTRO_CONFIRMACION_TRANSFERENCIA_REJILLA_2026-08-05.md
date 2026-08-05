# Preregistro — confirmación de transferencia entre rejillas 288 → 4.608

**Escrito antes de abrir el bloque virgen.** Este documento fija la confirmación del resultado de
desarrollo `results/grid_transfer_v2/result.json`; no es el manuscrito ni una reinterpretación de
los resultados anteriores. La autoridad de apertura es el PI. El bloque queda reservado en
`research/seed_custody_registry.json` como `RESERVED_NOT_OPENED` antes de ejecutar cualquier
simulación.

## 1. Pregunta y alcance

La pregunta confirmatoria es si un estado retenido de una política de búsqueda puede cruzar un
cambio explícito de espacio de diseño y conservar una ventaja que no sea solamente la distribución
marginal de configuraciones visitadas.

El resultado de desarrollo seleccionó **UCB1 factor-wise** como brazo confirmatorio: fue el único de
los cuatro comparadores que superó simultáneamente su arranque en frío y su réplica marginal. La
selección queda congelada antes de abrir las semillas. La confirmación no es una prueba de RL, de
PPO, de una prima neural, ni de transferencia entre regímenes de riesgo: el gate contextual
`H_regime` quedó por debajo de 0,05. Es una prueba de transferencia de estado de búsqueda entre
dos rejillas con una representación factorizada y clásica.

## 2. Diseño congelado

* **Entrenamiento:** seis contextos, en este orden fijo: `R1r`, `R2r`, `R1r+R2r`, `R1r|esc`,
  `R2r|esc`, `R1r+R2r|esc`, sobre la rejilla de 288 configuraciones.
* **Transferencia:** el mismo estado UCB1 busca sobre la rejilla
  `wrap288_compat_extended_v1` de 4.608 configuraciones.
* **Control frío:** UCB1 inicializado desde cero sobre la rejilla extendida.
* **Placebo decisivo:** réplica marginal de las visitas del brazo transferido, sin leer ni usar el
  estado retenido ni el contexto para elegir adaptativamente.
* **Presupuesto:** 24 evaluaciones por contexto y brazo. No hay reemplazo dentro de una búsqueda;
  si una propuesta factor-wise ya fue visitada, el protocolo usa el desempate aleatorio congelado
  del runner y registra la visita efectiva.
* **Física:** 52 semanas, CRN estricto y la misma semántica DES de
  `scripts/build_transfer_confirmation_cache_v1.py`. La extensión usa
  `op3_rm, op5_rm ∈ {0, 17.500, 70.000, 140.000}`; el subgrid con ambos factores en cero conserva
  la rejilla de 288.

Se ejecutan también neurona, OFAT y GP-EI para conservar la escalera comparativa. Sus resultados
son secundarios y exploratorios en esta confirmación; no pueden reemplazar el brazo UCB1 ni
seleccionar otro ganador después de abrir las semillas.

## 3. Estimandos y regla primaria

Para cada semilla `s`, se promedia el AUC de regret sobre los seis contextos. Menor AUC es mejor.

```text
δ_M(s) = AUC_marginal(s) − AUC_UCB1_transfer(s)
δ_C(s) = AUC_UCB1_cold(s) − AUC_UCB1_transfer(s)
```

El estimando primario es `E[δ_M]`: una ventaja positiva indica que el estado transferido supera a
su placebo estado-ciego. El estimando secundario es `E[δ_C]`: una ventaja positiva indica que la
transferencia supera el arranque frío. La unidad de inferencia es la semilla, que agrupa los seis
contextos; no se tratan contextos como réplicas independientes.

La inferencia usa bootstrap pareado por semilla, 5.000 remuestras y percentil 2,5 % como LCB95,
con la semilla de bootstrap fijada en el runner. No hay decisiones intermedias, extensión del
bloque, reasignación de semillas ni cambio del estimando después de inspeccionar resultados.

**Éxito confirmatorio:** LCB95(`δ_M`) > 0 **y** LCB95(`δ_C`) > 0, con todos los falsadores de
custodia, completitud y presupuesto pasando. Si sólo pasa `δ_M`, se reporta transferencia frente
al placebo pero no superioridad frente al frío. Si falla `δ_M`, se cierra la afirmación de
transferencia de forma confirmatoria.

## 4. Potencia y bloque

El artefacto de desarrollo `results/grid_transfer_v2/result.json` tiene 12 semillas y, para UCB1,

```text
media de δ_M = 0,0365475
SD pareada    = 0,0421540
```

Antes de abrir el bloque se fija un efecto de diseño conservador `δ* = 0,015` en unidades de AUC.
Con la SD de desarrollo, un test t aproximado unilateral al 5 % tiene potencia aproximada de
0,86 para `n = 60` (la potencia real puede ser menor si la SD o el efecto cambian). Se fija por
adelantado el bloque virgen de **60 semillas `8.100.001–8.100.060`**. El cálculo no convierte la
estimación de desarrollo en evidencia confirmatoria: sólo dimensiona el experimento.

El bloque fue comprobado contra el registro y los artefactos antes de reservarlo con el resultado
`NO_KNOWN_COLLISION`; como el inventario central se declara incompleto, esto no se presenta como
prueba matemática de virginidad. Cualquier ejecución parcial consume la condición de virginidad
de las semillas que haya tocado; no se rescata una corrida parcial como confirmación.

## 5. Custodia y falsadores

| falsador | condición de fallo |
|---|---|
| `f_seed_block_is_reserved_and_unopened` | el bloque no está reservado antes de abrir la primera simulación |
| `f_no_known_seed_collision` | aparece solapamiento con registro o artefacto sellado |
| `f_cache_is_complete` | faltan contextos, semillas o las 4.608 celdas de una rebanada |
| `f_null_subgrid_is_identity` | el subgrid `op3_rm=op5_rm=0` no reproduce la proyección de 288 |
| `f_budgets_are_matched` | el log de accesos no muestra 24 evaluaciones por contexto y brazo |
| `f_no_unrun_value_is_read` | una política consulta un valor distinto de la celda que acaba de visitar |
| `f_source_manifest_is_identical` | las rebanadas no comparten el manifiesto de física y runner fijado |
| `f_primary_placebo_is_state_blind` | la réplica marginal usa el estado retenido o resultados futuros |

Los falsadores se publican aunque el resultado primario sea positivo. `not_applicable` sólo puede
usarse para un falsador de semillas en una réplica declarada; esta confirmación no es una réplica.

## 6. Límites de lectura

Un resultado positivo autoriza únicamente la frase: **UCB1 factor-wise conserva una ventaja de
búsqueda al transferir estado de 288 a 4.608 configuraciones bajo este protocolo**. No autoriza
llamarlo RL, aprendizaje neural, adaptación contextual, ni afirmar que supera a BO, porque el GP
es un comparador y su transferencia se mide descriptivamente.

Un resultado nulo o un falsador fallido cierra la afirmación confirmatoria correspondiente. No se
abrirán PPO, MLP, KAN ni una nueva rejilla para rescatarlo dentro de este bloque.
