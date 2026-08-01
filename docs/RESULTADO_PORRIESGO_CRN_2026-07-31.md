# Resultado — el por-riesgo converge, **R24 domina**, y mi diagnóstico era falso

**Artefacto:** `results/sensitivity/perrisk_crn_v1/result.json` (sello `ee7b825cad6ede51…`,
`HALTED_FALSIFIER_FAILED` — **por la atribución, no por la medición**) · **19.200 corridas**,
18 factores, `N = 192`, **5 réplicas CRN por punto** · métrica `ret_excel_risk_conditional`.

## 1. Lo primero: el falsador tumbó mi propia explicación

Dije que la no convergencia venía de **ruido por episodio**, y que la cura era promediar
réplicas CRN por punto de diseño. Puse eso a prueba en vez de asumirlo:
`f_replication_reduces_the_estimator_overflow` calcula los mismos índices **también con una sola
réplica** y exige que promediar **reduzca** los índices fuera de `[0,1]`.

    fuera de [0,1]:   1 réplica  0   ->   5 réplicas  0

**Con una sola réplica ya convergen.** El falsador **FALLA**, y hace bien: **mi diagnóstico no
está demostrado.** Las violaciones de rango anteriores (3 a `N = 512`) no eran «ruido por punto»
en el sentido que afirmé — eran **inestables entre semillas de diseño**, y desaparecen sin
necesidad de replicar.

Lo mantengo escrito porque es exactamente lo que la réplica estaba para descubrir. Las 19.200
corridas siguen siendo válidas; lo que se cae es **la razón que les puse**.

## 2. Y ahora sí: la sensibilidad por riesgo, convergida

`f1` (índices dentro de `[0,1]`), `f2` (CRN común) y `f4` (cobertura) **pasan**. Los índices son
medición válida:

| factor | `S1` | `S_T` | `S_T − S1` |
|---|---:|---:|---:|
| **`freq_R24`** | **0,414** | **0,528** | 0,113 |
| **`impact_R24`** | **0,350** | **0,422** | 0,071 |
| `freq_R23` | −0,026 | 0,119 | 0,144 |
| `freq_R11` | 0,004 | 0,100 | 0,096 |
| resto (14 factores) | ≈0 | ≤ 0,08 | ≤ 0,08 |

**R24 —demanda contingente— domina todo el espacio de riesgo**, en frecuencia **y** en impacto.
Los dos juntos se llevan **~0,76 de `S1`**; los otros diecisiete factores suman prácticamente
nada.

**Y confirma la señal débil que anoté sin poder medirla.** En las corridas no convergidas dejé
escrito que `freq_R24` e `impact_R24` encabezaban `S_T` de forma estable entre `N = 128` y
`N = 512`, marcándolo como hipótesis a probar con el diseño corregido. **Se probó y es cierta.**

## 3. El intento de aumentar el headroom: falló

Con los índices convergidos, la etapa 2 construyó los regímenes **con información** —a partir de
los dos factores de mayor interacción, `freq_R23` y `freq_R24`— en vez de muestrear a ciegas.

    H_regime dirigido        0,000028
    H_regime mezclando familias  0,000182   (6,5x mayor)

**Dirigir por interacción dio MENOS headroom que mezclar familias.** No es lo que esperaba: la
interacción alta identifica dónde el efecto no es aditivo, pero **no** garantiza que el óptimo
se mueva entre los regímenes que construyes con ella. Son dos propiedades distintas y aquí se
separan.

Lo que sigue siendo el máximo medido en toda la campaña es **mezclar familias de riesgo:
1,8e-4** — y sigue ~55× bajo la barra de 0,01.

## 4. El estado final de los huecos

| hueco | estado |
|---|---|
| `S_ij` (dónde el nodo) | **cerrado**: `op12_rop` × `impact_R1r`, `S_ij = 0,219` |
| nodos nuevos aguas abajo | **cerrado**: aportan **cero** |
| riesgos por-riesgo | **cerrado**: R24 domina (`S1` 0,414 + 0,350) |
| aumentar el headroom | **probado y sin éxito**: continuo (no), nodos (no), dirigido por interacción (no). Solo mezclar familias sube, y a 1,8e-4 |
| clase de política | **abierto**: umbral de dos niveles, y la puerta declarada no autoriza más gasto |
| observable único | **abierto**: solo conteo de eventos R1r recientes |

## 5. Lo que esto deja para el paper de Garrido

Una respuesta, no un vacío. **El acoplamiento decisión × riesgo existe y está localizado**
(`op12_rop` × `impact_R1r`), **el riesgo que manda está identificado** (R24, demanda
contingente), y **en ese punto óptimo una política condicionada pierde contra la mejor
constante**. Cuatro intervenciones distintas para crear headroom —resolución continua, nodos
nuevos, mezcla de regímenes, orientación por interacción— y la mayor deja el techo en 1,8e-4.

Eso es «cuándo NO cerrar el lazo», medido en el punto más favorable que el propio sistema
ofrece, y con el placebo que lo separa del ruido.
