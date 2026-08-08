# El caché se reproduce bit a bit entre arquitecturas, y eso no lo suponíamos

**Fecha:** 2026-08-08 · **Estado:** medido, pendiente de sellar por la fase de agregación del
certificado v2. **No inventar un artefacto a mano para esto** — la evidencia son los propios shards.

---

## La pregunta, y por qué era un riesgo real

El certificado de equivalencia compara **bit a bit** y su control de mutación planta **1e−12**
exigiendo detectarlo. Eso significa que **no admite tolerancia**: cualquier umbral por encima de
1e−12 desarma el control y el certificado deja de probar nada.

El caché lo produjo la M1 Pro local — `cache_custody.runtime.platform` de cada rebanada dice
`macOS-27.0-arm64-arm-64bit`, Python `3.11.15`. Repartir shards a máquinas x86 arriesgaba deltas que
fueran **artefacto de arquitectura** (contracción FMA, `libm` distinta para `exp`/`log`) y no deriva
de código: un `RERUN_REQUIRED` falso sobre la confirmación de la que depende el titular del paper.

Iba a medirlo con **una rebanada** en el VPS. No hizo falta.

## Lo medido

La superficie base ya estaba corriendo en `ovh-agent-lab`. Consultado directamente:

| | caché (productor) | VPS (verificador) |
|---|---|---|
| plataforma | `macOS-27.0-arm64-arm-64bit` | `Linux-7.0.0-28-generic-x86_64-with-glibc2.43` |
| arquitectura | **arm64** | **x86_64** |
| Python | **3.11.15** | **3.14.4** |
| libc | Apple | glibc 2.43 |

```
rebanadas replicadas   105
celdas                 30.240
mismatches             0
max_abs_delta          0.0        (cero exacto, no «cero a tolerancia»)
```

## Qué establece, y qué no

**Establece** que el DES reproduce el caché **cruzando arquitectura, sistema operativo, versión de
libc y versión menor de Python** —arm64/macOS/3.11.15 → x86_64/Linux/glibc 2.43/3.14.4— sobre 30.240
celdas, con delta exactamente cero en valor y panel completo.

Eso es más fuerte de lo que el certificado necesitaba y **es una frase que el paper necesita de todas
formas** en su declaración de reproducibilidad: un revisor que clone el repositorio en otra máquina
obtiene los mismos números, y no de palabra.

**No establece** que toda la superficie extendida se comporte igual —eso lo dirá la agregación— ni
que cualquier futura versión de numpy/scipy preserve la propiedad. Se cita con su alcance: estas dos
plataformas, estas versiones, estas 30.240 celdas.

## Consecuencia operativa: la flota se abre

La prueba de una rebanada que planeaba queda **superada por 30.240 celdas**, y el enrutado deja de
estar clavado a arm64.

Con 0,139 s/celda medido bajo carga real (1,66 M celdas ≈ **64 h-CPU**):

| pool | workers | reloj |
|---|---:|---:|
| sólo local | 8 | **8,0 h** |
| local + VPS | 14 | **4,6 h** |
| local + VPS + Kaggle | 18 | 3,6 h |

**Recomendación:** cuando el VPS termine la base (~1 h), pasarlo a shards de la extendida — corta el
resto casi a la mitad. **Kaggle no vale la pena aquí**: subir el caché extendido de 649 MB como
dataset, más el tope de 9 h por sesión, por pasar de 4,6 h a 3,6 h. Kaggle rinde más en el panel
estacional de la Fase 2, que es auto-contenido y no arrastra caché grande.

## Nota de método

Los números de arriba salieron de una consulta directa por `ssh`, que **no es el pipeline**. Se
registran aquí como hallazgo operativo y **el artefacto sellado debe salir de la fase de agregación
del certificado v2**, que ya lee esos mismos shards. No se fabrica un `result.json` a mano para esto:
sería exactamente el defecto que este proyecto lleva un año cazando.
