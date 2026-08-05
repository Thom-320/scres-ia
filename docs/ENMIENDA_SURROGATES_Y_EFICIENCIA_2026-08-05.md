# Enmienda — KAN y MLP como surrogates de búsqueda, y `Delta_efficiency`

**Escrita ANTES de correr.** Runner: `scripts/run_search_surrogates_v1.py`. Misma caché, mismo
bloque quemado `5.300.001–012`, mismo presupuesto `B = 24`. **Sin semillas nuevas.**

## 1. Por qué aquí y no en el controlador

Garrido pidió el 28 de julio **KAN vs MLP vs MPC en parámetros, velocidad y convergencia**. Este
proyecto ya midió que dentro del episodio no hay prima neural en cuatro contratos distintos —y que
un MLP quedó **peor que un lineal** (R² 0,5548 vs 0,6826). Así que la pregunta se traslada al sitio
donde una red sí puede tener trabajo: **como aproximador de la superficie de diseño dentro de la
búsqueda entre corridas**, que es exactamente el papel que la Fig. 5 le da.

La escalera v2 ya midió lo esencial: **el ingrediente es la retención, no el aproximador** — un
bandido por niveles con memoria (AUC 0,04253) empata o supera a la neurona (0,04975). Falta saber
si un aproximador **más expresivo** convierte esa retención en más ventaja, o en ninguna.

## 2. Los brazos nuevos

Todos comparten el bucle de la neurona: puntúan las configuraciones no visitadas, eligen el máximo,
observan y actualizan. Sólo cambia el aproximador, y **todos retienen sus pesos entre contextos**,
igual que `ρ`.

| brazo | aproximador | parámetros |
|---|---|---|
| `neuron_memory` | logístico lineal sobre 4 coordenadas + sesgo | **5** |
| `surrogate_mlp` | MLP `[4→16→16→1]`, tanh | ~369 |
| `surrogate_kan` | KAN `width=[4,4,1]`, `grid=3`, `k=3` | pykan, contado en ejecución |

Objetivo de regresión: el valor **normalizado por prefijo dentro de cada contexto**, idéntico al de
la neurona. Entrenamiento **incremental** —los pesos continúan, no se reinicializan por paso—,
porque reinicializar convertiría el brazo en uno sin memoria y volveríamos a la tautología que ya
corregimos.

## 3. `Delta_efficiency`, medido y no prometido

El contrato `contracts/garrido_expanded_des_e_star_v1.json:134` lo define como *«calidad y
guardarraíles igualados, con menor latencia online o menos llamadas al DES»*. Aquí las llamadas al
DES **están igualadas por construcción** (`B = 24` para todos, verificado desde el log de accesos),
así que lo que queda es el **coste por decisión**:

* segundos de reloj por decisión, mediana y p95, cronometrados alrededor de la selección;
* número de parámetros del aproximador;
* coste total de búsqueda por contexto.

**La lectura declarada de antemano**: si dos brazos empatan en AUC, **gana el más barato**. Y si el
más barato es el de 5 parámetros, ése es el resultado — no un consuelo.

## 4. Por qué esto puede fallar

* **`surrogate_kan` o `surrogate_mlp` baten a la neurona con LCB95 > 0** → la expresividad sí
  compra algo en el bucle externo, y sería la primera prima neural medida en este proyecto.
* **empatan** → confirma la v2 desde otro ángulo: **el ingrediente es la retención**, y entonces
  el coste decide.
* **pierden** → una superficie con interacciones pareadas y 288 puntos no da para 369 parámetros
  con 24 observaciones por contexto, y hay que escribirlo así.

## 5. Lo que NO se puede afirmar con esto

`H_regime` = +0,0038 sobre esta superficie: **el óptimo es común a los seis contextos**. Ningún
resultado de esta enmienda puede presentarse como adaptación al régimen. Y ninguno autoriza
entrenar una política ni abrir una semilla.
