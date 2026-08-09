# Re-atestación de pines de origen — 2026-08-08

Veinte tests estaban en rojo. **Catorce ya lo estaban al empezar la sesión** (verificado corriendo
la misma selección en un worktree en `6656dd92`), así que la deriva es anterior a este trabajo.
Ninguna es una regresión de comportamiento, y eso se **establece**, no se afirma.

## 1. `supply_chain/supply_chain.py` — probado inerte, no supuesto

**Pin viejo:** `d8fd9347…` = el árbol de `5cb8fb82` (2026-07-31 21:40). Desde entonces el fichero
cambió por almacenamiento CSSU finito, el motor estacional, las familias de ocurrencia de riesgo y
la ruta de liberación del buffer.

**La prueba:** `scripts/verify_source_pin_inertness_v1.py` corre **los mismos episodios bajo los dos
árboles** —el atestiguado, en un worktree, y el actual— por el mismo punto de entrada
(`arm_runner.episode_moments`), un subproceso por árbol para que ninguno pueda ensombrecer al otro.
**21 celdas × 6 momentos = 126 comparaciones, 0 diferencias, tolerancia EXACTA (0.0).** Artefacto:
`results/source_pin_inertness/result.json`,
`SOURCE_CHANGE_IS_BEHAVIOURALLY_INERT_ON_THE_TESTED_PATHS`.

**Y la primera vez falló, que es la parte que importa.** Con los kwargs mal adivinados, ambos
árboles lanzaban `TypeError` de forma idéntica y el comparador reportaba **«0 diferencias»** — un
verde perfecto sobre cero trabajo. Lo cazaron `f1` (toda celda corrió bajo ambos árboles) y `f4` (el
control perturbado `shifts` 1 vs 3 **debe** diferir). Sin esos dos controles habría re-atestiguado
nueve ficheros de custodia contra una prueba vacía.

`f3` exige además que los dos árboles sean **ficheros distintos**: comparar un árbol consigo mismo
también da acuerdo perfecto.

**Lo que la prueba NO dice:** que una funcionalidad con puerta esté bien **encendida**. Sólo que
dejarla en su valor por defecto reproduce el árbol atestiguado. Eso es exactamente lo que un pin
afirma, y sacarle más sería la sobre-lectura que el pin existe para impedir.

`scripts/reattest_source_pins.py` movió el pin en **tres barridos** hasta punto fijo, tocando **9
ficheros** — más profundo de lo que sugiere el DAG de `content_sha256`, porque hay una segunda clase
de arista: el sha256 de bytes completos de una atestación pinchado dentro de los `source_bindings`
de otra. Los congelados de ejecución de Program O quedan intactos por diseño.

## 2. `.gitignore` — otra clase de pin, otra justificación

**Pin viejo:** `04bc7200…` = el árbol de `8f31f410`. El diff completo desde entonces son **dos
reglas de ignorado**: `.claude/` pasa a `.claude/*` con una excepción `!.claude/settings.json`,
para que el hook que guarda la salida de cada agente viaje con el repositorio en vez de ser
preferencia local.

**La causa aquí no es una medición y sería deshonesto presentarla como tal.** `.gitignore` no está
en ninguna ruta de ejecución: no lo importa ningún módulo, no lo lee ningún runner, y no puede
entrar en un episodio. La justificación es el diff, que es verificable en un vistazo y está
enteramente compuesto de directivas para git. Se re-atestigua citando **este documento**, no un
artefacto de resultados, porque inventar una medición para un fichero que no se ejecuta sería peor
que decir por qué no hace falta.

## 3. El transductor exacto rechazando atributos sin clasificar

No es una atestación obsoleta: es el instrumento haciendo su trabajo. Quince atributos vivos del
simulador —conmutación CSSU, presupuesto de expedición, grafo LOC, motor estacional, liberación de
buffer— no estaban clasificados, y el transductor **se niega a certificar completitud de Markov**
mientras haya estado vivo que no sabe si debe llevar en la clave.

El criterio de clasificación es el real, no el cómodo: **¿se muta después de `__init__`?** Un
recorrido del AST de la clase busca reasignaciones, asignaciones aumentadas, escrituras por
subíndice y llamadas mutadoras.

* **9 → `IMMUTABLE_CONTRACT_FIELDS`**: escalares y cadenas ligadas una vez y nunca reasignadas, que
  por tanto **no pueden distinguir dos estados dentro de una corrida**.
* **14 → `INERT_FROZEN_FIELDS`**, que **se serializa dentro de la clave**: todo lo mutado, más los
  casos que un escaneo de reasignación habría llamado inmutables por error — un **objeto** cuyo
  estado interno cambia sin que el atributo se reasigne (`_cssu_capacity_ledger` que admite, un
  grafo LOC que pierde arcos, un motor estacional con su propia fase).

Esa asimetría es deliberada y va en la dirección segura: **no poder demostrar que un campo no separa
dos estados es razón para MANTENERLO en la clave, nunca para quitarlo.** Puede reducir compresión;
no puede crear una fusión falsa.

`classification_complete` pasa a `True`, con `static_live_reads_unclassified` y
`unclassified_live_attributes` ambos vacíos.

## 4. Portabilidad

Una transcripción de agente traía una ruta absoluta del `$HOME` del autor. El saver ya redacta en origen; este fichero se
escapó. Redactado. `test_repo_portability` era exactamente el guardarraíl que debía cazarlo, y lo
cazó.
