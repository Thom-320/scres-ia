# Resultado — la fuga de determinismo era una línea, y el arnés queda reparado

Sucede a `results/determinism_diagnostic/result.json` (`DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT`), que
**no se edita**. Runner de control: `scripts/run_determinism_repair_control_v1.py`.

## 1. Por qué el diagnóstico no pudo nombrarla, y por qué eso estuvo bien

El diagnóstico comparó tres configuraciones —8 envs con hilos libres, 8 envs con hilos fijados a 1,
y un solo env— y **las tres divergieron**, así que devolvió `DEEPER_THAN_BOTH` y **se negó a nombrar
una causa**.

Ese veredicto era correcto. **El defecto vivía en una capa que las tres configuraciones
compartían**, así que ningún contraste entre ellas podía aislarlo. Su regla de lectura hizo
exactamente lo que debía: negarse a reclamar un arreglo que no había demostrado. Nombrarlo exigía
otro test.

## 2. El test que sí lo nombró — sacar al aprendiz del medio

| prueba | resultado |
|---|---|
| entorno con `reset(seed=k)` y secuencia fija de acciones | **bit-idéntico**, `max\|Δ\| = 0,000e+00` |
| dos `reset(seed=k)` seguidos | observación inicial idéntica |
| **`reset()` SIN semilla, como hace el vec env** | episodio 1 idéntico · **episodio 2: 48,674 contra 51,820** |

Los vec env siembran **sólo el primer reset**. Del segundo episodio en adelante,
`MFSCGymEnvShifts.reset` pasaba **`seed=None`** directo a `MFSCSimulation`, que entonces **se
sembraba de la entropía del sistema**.

> Con ~1.900 episodios por entrenamiento, ese único argumento es **toda** la dispersión de 2,363
> puntos a semilla fija — mayor que los 2,102 que el bake-off mostraba **entre** semillas distintas.

## 3. El arreglo, y el falso comienzo que también se registra

La semilla de episodio se deriva de `self.np_random`, que `super().reset()` sí siembra y que
**persiste entre episodios**.

**La misma línea existe en `MFSCGymEnv` y en `MFSCGymEnvShifts`.** Arreglé primero la clase base y
no cambió nada, porque `track_b_v1` usa la subclase. Queda escrito porque es exactamente el tipo de
cosa que parece «el arreglo no funcionó».

## 4. Los controles

| control | resultado |
|---|---|
| cinco episodios consecutivos, dos corridas | **idénticos** |
| semilla distinta | **trayectoria distinta** — el arreglo no fija la cinta |
| **entrenamiento completo, 20k pasos × 8 envs, dos réplicas** | **93,973236562416 / 93,973236562416 · Δ = 0,0 exacto** |

El segundo control importa tanto como el primero: **un arreglo que hiciera todos los episodios
idénticos sería peor que el defecto** —el aprendiz entrenaría sobre una sola cinta para siempre— y
`f2` existe para impedirlo.

## 5. Lo que esto cambia, y lo que no

**Cambia:** la semilla vuelve a ser unidad de réplica. La banda de ±2,4 se colapsa. **La prima
neural de `track_b_v1` vuelve a ser MEDIBLE**, y con ello se retira el motivo que di —«el
instrumento no la resuelve»— como razón terminal del `NO-GO` de C1.

**No cambia:**

* los artefactos ya emitidos siguen siendo **irreproducibles**; nacieron bajo el defecto y ningún
  arreglo posterior los rescata. `results/architecture_bakeoff*` sigue degradado a nota al pie;
* **C1 sigue sin poder confirmarse**: no queda ningún bloque virgen. Medible ≠ confirmable;
* toda comparación **entre** corridas anteriores al arreglo sigue siendo inválida.

## 6. Lo que habilita

Cualquier medición futura de Track B es reproducible, así que las comparaciones entre corridas
vuelven a ser legítimas **a partir de aquí**. Si el PI autoriza semillas nuevas, C1 pasa de
«irresoluble» a «pendiente de una confirmación».
