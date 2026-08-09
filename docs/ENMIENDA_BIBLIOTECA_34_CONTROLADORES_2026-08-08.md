# Enmienda — la biblioteca vuelve a los 34 controladores del original

**Fecha:** 2026-08-08. Enmienda a `docs/PREREGISTRO_G3A_V2_RECONSTRUCCION_2026-08-08.md` §5.
El artefacto de 14 controladores (`results/g3a_boundary_v2/result.json`,
`G3A_DID_NOT_REPRODUCE`) **se conserva** y no se reescribe.

## 1. Por qué

Reduje la biblioteca de 34 a 14 «porque el propósito es la frontera entre familias y no el barrido
dentro de cada una». Es un argumento razonable y **no es comprobable**: si el titular del paquete
lo capturaba un controlador que yo no incluí, mi no-reproducción mide mi recorte, no su resultado.
La única forma de cerrar esa duda es correr la biblioteca completa.

## 2. La biblioteca, enumerada aquí y cerrada, siguiendo la descripción del original

El manuscrito enumera: «nueve constantes; lookups de aviso; placebos retrasado y barajado;
políticas de creencia con estado y reset; reglas de demanda rezagada; umbrales de backlog;
compuestos creencia-backlog; y una política de estado verdadero predeclarada».

| familia | n | detalle |
|---|---|---|
| constantes | **9** | `allocation_a ∈ {0,10 … 0,90}` paso 0,10 |
| lookups de aviso | **4** | directo e invertido, en ganancia suave (0,6/0,4) y fuerte (0,8/0,2) |
| placebos | **4** | aviso barajado (dos semillas) y aviso retrasado (uno y dos periodos) |
| creencia | **4** | con estado y reset, en ganancia suave y fuerte |
| demanda rezagada | **3** | umbrales 0, ±0,5, ±1 |
| umbral de backlog | **4** | `θ ∈ {0,45, 0,50, 0,55, 0,60}` |
| compuestos creencia-backlog | **5** | mezcla `w ∈ {0,2 … 0,8}` de creencia y umbral |
| estado verdadero **privilegiado** | **1** | diagnóstico; no desplegable, no óptimo, no cota |
| **total** | **34** | |

## 3. Lo que esto le hace al resultado, dicho antes de correr

**Más brazos elegidos sobre el bloque de entrenamiento significa más maldición del ganador**, no
menos. La selección busca el máximo sobre 33 brazos desplegables en vez de 13, así que el brazo
elegido está más sesgado al alza **en entrenamiento** — y precisamente por eso se evalúa en el
bloque retenido, donde ese sesgo no viaja.

De ahí se sigue la lectura importante: **si con 34 controladores el `H_obs` retenido sigue sin
separarse de cero, el recorte de biblioteca queda descartado como explicación.** Y si aparece,
habrá que mirar qué familia lo trae y si sobrevive a su propio placebo.

## 4. Lo que no cambia

Ni las nueve celdas, ni el endpoint, ni las semillas `8800001–8800060` con su partición 30/30, ni
la barra, ni los falsadores, ni la regla de que `f7` y `f8` definen el hallazgo **juntos**, ni el
grado: sigue siendo desarrollo, no confirmación.

`f6` se amplía a lo que corresponde con más brazos: el **mejor** brazo de aviso debe batir al
**mejor** placebo, no a uno cualquiera — con cuatro placebos, comparar contra el peor sería
regalarse el falsador.
