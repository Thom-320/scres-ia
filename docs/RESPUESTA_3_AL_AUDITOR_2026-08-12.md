# Tercera respuesta al auditor — tu R2 mató un cuarto del titular y tu R1 me refutó a mí

**Fecha:** 2026-08-12 · **HEAD:** `67d81b71` · **Supersede:**
`docs/RESPUESTA_2_AL_AUDITOR_2026-08-12.md` en la parte de ruta, no en las concesiones.
**Estado completo:** `docs/BRIEFING_REVISION_EXTERNA_2026-08-12.md`.

---

## 1. Ejecuté tu ranking en el orden que dijiste, y tenías razón en el orden

Pediste **R2 antes que R1** — *«media hora de nulo de Jensen decide si el 0,15151 es un premio o un
artefacto del estimador; es la prueba más barata y más letal que queda»*. Lo hice así.

### R2 — el nulo de Jensen sobre `H_PI`

Tu diagnóstico era exacto: el nulo fungible de O es un nulo de **física**, no de **estimador**. Lo
confirmé con el número: la varianza intra-tapa entre calendarios bajo fungibilidad es
`0.000e+00`, así que un `H_PI` de exactamente 0 **no podía** cotizar sesgo de selección.

Precondición antes de testear nada: reproduje la cifra sellada desde las matrices crudas, **bit a
bit** (`0.15275340823389597` y `0.15151378920653932`). 1.000 permutaciones del eje de calendarios
dentro de cada tapa, moviendo todas las métricas a la vez para que la máscara de seguridad siguiera
coherente:

```
observado  safe_h_pi  +0.151514
nulo        media +0.114431   sd 0.004191   p95 +0.120352   p=0.0000
```

**Sobrevive.** Pero el **75,5 %** del titular era el sesgo de tomar un máximo sobre 65.536
calendarios. El headroom corregido es **+0,037083**, que aún supera la barra por 3,7×.

**Consecuencia editorial, y la asumo:** toda cita futura de `H_PI` dice **0,0371**, no 0,1515. La
cifra que circuló durante un mes era cuatro veces el efecto.

### R1 — y aquí me refuté a mí mismo

Yo sostenía que O murió porque la selección maximizaba la media y nunca miraba la cola. Enumeré
**exhaustivamente las 16 configuraciones** de la clase declarada sobre el bloque de ajuste.

**`p4_las_dos_reglas_pueden_diferir` FALLA.** `S_mean` y `S_cvar` eligen **la misma configuración en
las cuatro celdas**. La política óptima en media **ya era factible en cola**, y con margen: deltas de
`ret_visible_cvar10` de +0,023428 [+0,012287], +0,024263 [+0,016090] y +0,125967 [+0,100733].

Meter la restricción en el objetivo **no cambia nada, porque nunca se estaba violando**.

Y los puntos estimados **no cambian de signo** entre ajuste y validación. Lo que voltea el veredicto
es la inferencia: t(47) unilateral de **1,6779** en el ajuste contra un crítico **simultáneo de
2,8357** sobre 69 estimandos en la validación.

> **Era un problema de potencia, no de especificación.** Mi hipótesis del «objetivo sin la
> restricción» —que te mandé como la apuesta principal— está muerta.

---

## 2. Y entonces corrimos la réplica con potencia

`results/program_o/powered_replication_v1/result.json` →
**`OBSERVABLE_CONVERSION_SURVIVES_AT_ADEQUATE_POWER`**, 7/7 falsadores.

**288 tapas vírgenes por celda**, excepción PI documentada, bloque `7500001–7500288`, apertura única.

```
celda              cola punto   cola LCB sim   primario punto   primario LCB
rho75_share90       +0.038786      +0.021591        +0.091016      +0.077131
rho90_share75       +0.024439      +0.010936        +0.076211      +0.064356
rho90_share90       +0.127313      +0.103989        +0.117207      +0.099978

critico simultaneo 2.8770   (el que mato a O fue 2.8357)
```

**Cómo se ejecutó, porque importa para tu objeción de «segundo rescate»:**

* el runner congelado **rechaza por diseño** cualquier bloque que no sea de 48 tapas. Ese guard es
  correcto y **no se tocó**. Se ejecutó **byte-idéntico seis veces** sobre sub-bloques disjuntos,
  con el `sha256` verificado antes y después;
* los seis contratos hijos se verificaron programáticamente: difieren del correctivo en **exactamente
  un campo**, `validation_tapes.range`;
* el comparador estático congelado es idéntico en los seis;
* el agrupado **es código nuevo** —`joint_bootstrap` fija `n_tapes = 48`— y por eso lleva su propio
  falsador: aplicado a un solo sub-bloque devuelve la estimación y la cota **selladas de ese
  sub-bloque, sin cambiar**.

**El único grado de libertad usado es el tamaño de muestra**, que la lista `no_post_failure_changes`
del contrato correctivo **no contiene**. Controlador, hiperparámetros, celdas, comparador, física,
métrica, placebos, umbrales y guardarraíles: intactos.

**Y no se tomó ninguna de las tres laxitudes disponibles.** No estreché la familia de multiplicidad
—habría bajado la n necesaria de 154 a 93—, no metí margen tolerante, no toqué el SESOI. Cruzar el
listón original vale más que cruzar uno movido.

**La firma de que era potencia está en los sub-bloques:** `[STOP, STOP, STOP, PASS, STOP, STOP]`.

**Lo que esto NO es**, y te lo digo antes de que lo preguntes:

* **no promueve a Program O**, que sigue cerrado e inmutable. Es un programa nuevo que hereda su
  física;
* **no es una prima neural.** La política es un **belief-MPC clásico**. Es una conversión observable
  de un controlador **estructurado**.

---

## 3. Corrí también tus dos gates baratos, y los dos cerraron en negativo

**Clase comparadora ampliada (tu cargo principal).** Añadí GBDT, random forest, GP y kernel ridge,
con sus versiones con lag, sobre las mismas siete features y los mismos folds:

```
mlp_tuned  vs gaussian_process  +0.0342 [-0.1030, +0.1715]   no
recurrent  vs gbdt_lagged       -0.0300 [-0.1113, +0.0513]   no
```

**Tenías razón**: contra `linear_interactions` el MLP daba +0,1081 y el recurrente +0,1487; contra
una clase completa no pasa ninguno. El número era real y su interpretación no.

Dos veces me pillé montando hombres de paja en el proceso —un RBF a `gamma=1` sobre one-hots crudos
daba −1,08, y `KernelRidge` no centra el target y daba −6,90—. Los dos reparados: un comparador que
pierde por culpa mía no prueba nada, que es justo lo que reprochabas.

**El bucle externo, separado en dos estimandos.** `RETENTION_YES_NEURAL_CARRIER_NO`: la retención
vive (+0,0607, 6/6 simultáneo) y el portador neural es **−0,007010 [−0,024399, +0,013955]**, con
`ucb1_transfer` por delante. El mejor portador clásico se elige **dentro de cada remuestreo
bootstrap** para no regalarle a la red un margen de maldición del ganador.

**Y la decisión de configuración no vale nada**: un oráculo al que se le da el mejor buffer de cada
contexto compra **+0,000065** contra una barra de 0,01, y contra un nulo cuya media es **+0,003978**.

---

## 4. Lo que sigo rechazando, con código

Tu cargo de la **superficie analítica** sigue sin sostenerse, y lo repito porque es el único que
verifiqué como falso: ningún brazo recibe los cinco drivers del Cobb-Douglas. Reciben **siete
números de configuración** (`base_features`/`rich_features`), y el mapa configuración → drivers pasa
por un DES SimPy de 13 operaciones con cuatro flujos RNG. Un regresor log-lineal sobre los drivers
sería tautológico.

---

## 5. Dónde te pido que apuntes ahora

1. **La réplica con potencia**: ¿aceptas que cambiar sólo `n` no es un segundo rescate, dado que
   `no_post_failure_changes` no lo lista? Si no lo aceptas, dime qué lo convertiría en aceptable,
   porque el resultado está sellado y prefiero saberlo antes de escribirlo.
2. **El agrupado**: uso el máximo de los seis críticos simultáneos, argumentando que pooling de
   réplicas independientes deja la estructura de correlación intacta. ¿Es defendible, o quieres el
   max-t recomputado sobre las 288 tapas desde las matrices?
3. **`H_PI = 0,0371`**: ¿hay más cantidades en el árbol con forma de media-de-máximos que deban
   pasar por el mismo nulo? Yo veo el techo de Program T y el `cd_spatial` de G.
4. **El claim final.** Con la prima neural caída en predicción, decisión y control, y con un
   positivo clásico con custodia completa, ¿el paper es «operacionalización con control
   estructurado» con el negativo neuronal como aparato? Es lo que yo escribiría.
