# Dónde podría haber una prima neural — condiciones necesarias, y por qué el entorno actual no cumple ninguna

Tres cosas se cierran hoy y una se abre. Las tres cerradas son requisitos previos; la abierta es
el programa que el PI pide: **encontrar un entorno donde la prima neural exista, y entender por
qué en éste no existe.**

## 1. Lo que se cierra hoy

### a) Ningún resultado quedó obsoleto por los cambios al DES

**Medido, no afirmado** — `results/metric_audit/des_change_differential/result.json` (sello
`335ca8c812483917…`): árbol pre-cambio `d89f6d2` contra el actual, 10 configuraciones
preexistentes × 3 semillas × 61 métricas = **1.830 comparaciones, 0 diferencias**. Cobertura de
`split_v1`, las tres reglas de servicio y los tres niveles de reparto heredados.

**Deuda saldada:** usé exactamente esa frase como `--cause` al re-atestiguar los pins y **no tenía
registro de haberla corrido**. Afirmar una sonda que no corriste es peor que no sondear.

### b) El rerun de Q2 sobre 288 **ya está hecho**, no pendiente

`results/garrido_meta_learner_v2/result.json`: 288 configuraciones, presupuesto 24, 12 réplicas,
**DES realmente ejecutado** (20.736 episodios de superficie), seis falsadores PASA con el `f5`
validado por control, y **reproducido bit a bit** en arm64 y x86-64. `+7,90 [+6,88, +8,93]`.

La revisión externa lo lista como pendiente; **no lo está**. Lo que sí es replay algorítmico es la
superficie thesis-native de 90, y ésa está correctamente etiquetada como tal por quien la produjo.

### c) Alcance del actuador CSSU — y una corrección retroactiva

`results/metric_audit/cssu_liveness_scope/result.json` (sello `5f9e89dafb5bab87…`), cuatro
falsadores PASA:

| regla de servicio | fracción de epochs donde el reparto **puede** cambiar algo |
|---|---:|
| **`SPT_FULL`** | **0,0000** |
| `FIFO_PARTIAL` | 0,7308 |
| `R24_AGE_PARTIAL` | 0,7292 |

> **Bajo `SPT_FULL` el actuador está MUERTO.** Los dos destinos nunca son simultáneamente
> factibles, así que el reparto no selecciona nunca entre ellos.

**Y eso corrige una lectura mía.** Atribuí el `H_regime = 0,000000` de `SPT_FULL|fungible` al
mecanismo de fungibilidad reproduciendo el nulo de Program O. **Parte de ese cero es simplemente
que la palanca no existe en esa regla.** El nulo fungible sigue siendo real —se reproduce también
en `FIFO_PARTIAL` y `R24_AGE_PARTIAL`, donde la palanca sí vive— pero la celda `SPT_FULL` no era
evidencia de nada y no debí reportarla al lado de las otras sin decirlo.

**Declaración de alcance:** todo resultado de headroom sobre esta palanca está acotado por su
fracción viva. Con `FIFO_PARTIAL` el techo es 73 % de los epochs; con `SPT_FULL`, cero.

## 2. Lo que se abre: cuándo puede existir una prima neural

Una prima para un aprendiz no lineal sobre un baseline lineal exige que **la superficie de
respuesta sea no lineal en las entradas que el aprendiz ve**. Eso no ocurre por accidente: hay
**cuatro generadores** de no linealidad, y **el entorno actual no tiene ninguno**.

| generador | qué crea | ¿presente hoy? | por qué no |
|---|---|---|---|
| **G1 · óptimo interior con penalización a dos lados** | curvatura: poco **y** mucho hacen daño | ❌ | el inventario **no tiene coste**: más es débilmente mejor, así que la superficie es monótona |
| **G2 · umbral duro en el objetivo** | discontinuidad que un lineal no representa | ❌ | la rama de **autotomía** (`CTj ≤ LT`) está **muerta**: `delay = 54 > LT = 48`, 0 de 416 pedidos |
| **G3 · asignación combinatoria (≥3 reclamantes no sustituibles)** | superficie tipo mochila | ❌ | sólo **2** CSSU, **simétricas** por construcción (hash 50/50) |
| **G4 · observabilidad parcial con inferencia** | el mapa observación→acción es no lineal aunque el valor sea lineal en el estado | ❌ | el aprendiz ve `ρ` **directamente**; no hay nada que inferir |

**Ésa es la explicación completa del `R² = 0,982`.** No es que las redes fallen: es que **no hay
función no lineal que aprender**. Y decirlo así convierte un negativo en una caracterización.

## 3. Los cuatro generadores, ordenados por coste y por probabilidad de funcionar

### G1 — inventario con precio *(el más barato, y el más faithful)*
**No hay que inventar nada.** El índice Cobb-Douglas del propio Garrido (IJPR 2024, Eq. 5) ya
cobra el inventario: `ζ` entra positivo pero `κ` incluye el coste de mantener `c_i`. Es decir,
**su propia métrica tiene la estructura a dos lados que a `ret_excel` le falta** — y **nunca hemos
barrido los buffers bajo Cobb-Douglas.**

> **Predicción falsable:** bajo Cobb-Douglas el óptimo de buffer es **interior** (no en el
> extremo), y **se mueve con la frecuencia de riesgo**. Si es así, hay curvatura *y* dependencia
> del régimen — las dos condiciones — sin salir de la métrica de Garrido.

Coste: un barrido. **Es lo primero que haría.**

### G2 — resucitar la autotomía
El brazo `FDB` **está implementado y medido**: revive la autotomía 10× en R1r, está en el conjunto
no dominado, y su precio de fidelidad está cuantificado en **0,95 SE de `ret_mean`**. Adoptarlo
enciende la rama de peso 1,0 que hoy es idénticamente cero e introduce un **umbral duro en
`CTj = 48`**. Un umbral es exactamente lo que un lineal no puede representar y una red sí.

### G3 — tres o más reclamantes asimétricos
La contención con dos reclamantes **simétricos** dio 1,5e-04; con asimetría en el cuello subió
monótonamente hasta 2,1e-03 — **la dirección era correcta** — pero el guardarraíl la declaró
defecto de métrica. Con **tres** reclamantes de criticidad distinta la asignación pasa a ser
combinatoria. Requiere construcción real.

### G4 — observabilidad parcial
El único lane del proyecto con señal observable positiva y estable fue **belief-MPC bajo reloj
fijo**: `+0,025…+0,073`. Ahí el aprendiz **tenía que inferir**. Barato de reconstruir: ocultar la
etiqueta de régimen y forzar la inferencia desde conteos ruidosos de eventos.

## 4. El diseño experimental que convierte esto en resultado

Para cada generador, encendido de forma **independiente**, se miden **cuatro** cosas — y sólo si
las dos primeras se mueven tienen sentido las dos últimas:

1. **curvatura** — `R²` de un modelo lineal sobre la superficie de respuesta *(la prueba directa
   de la premisa)*;
2. **movilidad del óptimo** — `H_regime`, ¿se mueve el `argmax` con el régimen?;
3. **prima neural** — MLP y KAN contra el lineal en predicción held-out, con el **SESOI
   preregistrado de 0,05 en `R²`** y particiones agrupadas por semilla;
4. **valor de política** — ¿una política condicionada al estado bate a la mejor constante?

**Un generador "funciona" sólo si mueve (1) y (2).** Si ninguno de los cuatro los mueve, el
resultado del paper es una **condición necesaria demostrada por agotamiento constructivo**, que es
mucho más fuerte que «no encontramos». Si alguno los mueve y aun así (3) no aparece, el resultado
es **más profundo todavía**: habría curvatura sin prima, y eso hay que explicarlo.

## 5. Orden de trabajo — acuerdo con la revisión externa, con dos enmiendas

| paso | estado |
|---|---|
| 1. cerrar custodia y retirar cifras contaminadas | **hecho** — retracción en `43e3546`, sonda diferencial sellada |
| 2. liveness del CSSU y su alcance | **hecho hoy** — y `SPT_FULL` resulta ser un actuador muerto |
| 3. métrica contra abandono, endpoints separados | **hecho** — `v2` contratada y sellada; servicio y cola siguen separados |
| 4. rerun de Q2 sobre 90 y 288 | **90 hecho** (replay, etiquetado); **288 hecho** (DES ejecutado, sin fuga, reproducido en dos arquitecturas) |
| 5. H1–H4 con etiquetas | **H1′ sostenida · H2 medida · H3′ corriendo (n=120) · H4 medida** |
| 6. reescribir el v0 para C&IE | pendiente — es el hueco real del artículo |
| **7. MLP/PPO sólo si pasan fidelidad, métrica y Q2** | **enmienda:** añadir *«y sólo si algún generador produce curvatura»*. Entrenar sobre una superficie con `R² = 0,98` no puede producir una prima aunque todo lo demás pase |

**Segunda enmienda, sobre el encuadre:** la revisión propone «no hay prima neural práctica» como
respuesta a Q1. Estoy de acuerdo, **pero incompleta**. La respuesta con contenido es:

> **No hay prima neural porque no hay no linealidad que aprender, y eso es una propiedad
> construible del entorno, no un veredicto sobre las redes.** Aquí están las cuatro condiciones
> que la generarían, aquí está cuáles cumple el entorno de la tesis (ninguna), y aquí está qué
> pasa cuando se encienden.

Eso responde Q1 **y** le dice a Garrido dónde su propuesta sí aplicaría. Es una contribución
metodológica, que es lo que C&IE publica.
