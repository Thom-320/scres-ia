# Lectura de la confirmación de la Puerta B — `BLOCKED_INSTRUMENT`, y el defecto es de diseño mío

**Artefacto:** `results/program_n/gate_b_confirmation/result.json` ·
**Veredicto:** `BLOCKED_INSTRUMENT` ·
**Regla:** `docs/PREREGISTRO_CONFIRMACION_PUERTA_B_2026-08-09.md` §4.1, escrita antes de abrir una
sola semilla: *«Si `f2` falla … el veredicto es `BLOCKED_INSTRUMENT`. **Nada más se lee.**»*

## 1. Qué falló

`f2_classical_arms_reproduce` exige que los brazos clásicos —código intacto— caigan dentro de
`0,02` de sus valores de desarrollo. Cuatro de seis lo exceden:

| brazo | desarrollo | confirmación | desviación |
|---|---|---|---|
| `train_cell_mean_comparator` | +0,6931 | +0,5994 | **0,0937** |
| `tree` | +0,6225 | +0,5382 | **0,0843** |
| `linear_interactions` | +0,6306 | +0,5905 | **0,0401** |
| `spline_buffer` | +0,6365 | +0,6022 | **0,0343** |
| `linear_additive` | +0,6062 | +0,5865 | 0,0197 |
| `constant` | −0,0167 | −0,0132 | 0,0035 |

## 2. El defecto es mío, y lo vi venir sin actuar

`f2` se escribió para el **desarrollo**, donde las tapas eran **las mismas** y su pregunta era
*«¿cambié algo más que el ajuste neuronal?»*. Es una comprobación de **identidad de código sobre
datos idénticos**, y allí reprodujo a `4,9e-05`.

Llevarla sin cambios a una **confirmación sobre tapas distintas** la convierte en otra pregunta:
*«¿producen ocho semillas nuevas el mismo R² que ocho semillas viejas?»*. **No hay ninguna razón
para que eso pase.** La variación de muestreo de un R² fuera de fold sobre ocho semillas mueve mucho
más de 0,02, y los brazos clásicos son deterministas dado el dato: se movieron porque **el dato es
otro**, que es exactamente lo que una tapa fresca significa.

**Y lo escribí en el preregistro antes de correr:**

> *«en desarrollo la reproducción fue a 4,9e-05 porque eran las mismas tapas. Aquí son otras, así
> que se espera una desviación real; 0,02 es la tolerancia que ya estaba escrita en el runner y no
> se relaja ahora.»*

Vi la tensión, la anoté, y **congelé el falsador equivocado en lugar de arreglarlo**. Congelar no es
una virtud cuando lo que congelas está mal: la disciplina era arreglar `f2` **antes** de abrir el
bloque, no documentar que iba a fallar y abrirlo igual.

## 3. El bloque está quemado

`9400001–9400008` queda `BURNED_INSTRUMENT_DEFECT`. La puerta era de un solo sentido y el
preregistro lo decía: *«si el instrumento resulta defectuoso al correr, el bloque queda quemado
igualmente»*. Un sucesor necesita otras ocho semillas.

## 4. Lo que NO se lee

Por la regla, **nada más**. El artefacto contiene los contrastes neuronales y **no se interpretan**.
Se conservan en el sello por custodia y se declara aquí que **no se usaron para inferencia,
selección ni adjudicación**.

Lo digo explícito porque la tentación es real y va contra mí: en esas tapas **el MLP pasa el gate y
el KAN no**, al revés que en desarrollo. Leer eso sería elegir qué resultado cuenta después de ver
cuál salió, que es el mecanismo exacto que este proyecto lleva dos días desmontando.

## 5. Lo que sigue en pie

**El desarrollo no queda tocado.** `results/program_n/gate_b_cd_surface/result.json` =
`SURFACE_PREMIUM_CAPTURED` sigue siendo válido en su propio alcance: **grado desarrollo**, ocho
semillas, cinco folds. Esta corrida **no lo confirma y no lo refuta** — no llegó a preguntárselo.

Lo que hoy sabemos con certeza es más estrecho de lo que parecía hace una hora: **la prima de
superficie está medida en desarrollo y sigue sin confirmar.**

## 6. El sucesor, con `f2` arreglada

`f2` se sustituye por dos comprobaciones que **sí** responden a la pregunta que debía hacer:

1. **identidad de código** — el `module_manifest` de la confirmación debe coincidir **hash a hash**
   con el de desarrollo. Eso responde «¿cambié algo más que el ajuste?» sin depender del dato;
2. **preservación de orden** — el ranking de los brazos clásicos entre sí debe conservarse
   (`spline ≥ linear_interactions ≥ linear_additive`, `constant` último). Eso comprueba que la
   superficie nueva es del mismo tipo, sin exigir que sus niveles coincidan.

Ninguna de las dos puede fallar por variación de muestreo, y ambas pueden fallar si de verdad
cambié el instrumento. Bloque nuevo, y esta vez la comprobación se piensa **para la confirmación**
y no se hereda del desarrollo.
