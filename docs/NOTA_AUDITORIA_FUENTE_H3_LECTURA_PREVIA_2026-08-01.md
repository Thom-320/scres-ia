# Nota — cómo debe leerse la auditoría de fuente H3′, escrita ANTES de que aterrice

> **Corrección 2026-08-01, antes de que exista el artefacto.** Dos frases de esta nota eran
> defectuosas, y ambas se reparan **en la ejecución**, no en el texto:
>
> **(1) `f6` no estaba realmente en `NOT_APPLICABLE`.** La corrida se lanzó **antes de que
> existiera el flag `--replay-of`**, así que la réplica quedaba declarada en esta nota y **no en
> el artefacto**. Eso es precisamente la reinterpretación post hoc que la nota dice evitar.
> **Corrección aplicada: corrida detenida y relanzada con `--replay-of garrido_h3_vps`**, de modo
> que el estado `DECLARED_REPLAY` queda **sellado dentro del resultado**.
>
> **(2) «Si difieren, H3′ se queda con las 90 locales» es FALSO.** Verificado: **ambas** rebanadas
> están selladas contra `PREREGISTRO_META_APRENDIZ_2026-07-31.md` (`a24b164d…`), **no** contra el
> contrato H3′ (`576d02b5…`). Ninguna de las dos es un artefacto H3′ contratado. La lectura
> correcta: **ambas rebanadas permanecen no promovibles hasta resolver contrato y procedencia**,
> y el efecto Alzheimer sigue **pendiente de custodia** en los dos desenlaces.
>
> **(3) Y el hash del script no bastaría** para identidad de fuente: el runner importa
> `supply_chain`, así que dos rebanadas pueden compartir un entry point idéntico y haber
> ejecutado física distinta — que es exactamente lo que pasó con el snapshot del VPS. Añadido
> `module_manifest()`, sellado dentro del payload. Hace decidible `f_merge_source_is_identical`
> para rebanadas **futuras**; **no rescata retroactivamente** las selladas sin él.

**Escrita mientras la corrida está en vuelo y `results/garrido_meta_learner_h3power_vps_local_replay/result.json`
NO existe.** Declarar la lectura después de ver el artefacto sería reinterpretación post hoc;
por eso se fija aquí.

## 1. Qué es esta corrida, y qué no es

Re-ejecución **local** de las semillas `6.000.091–120` —las de la rebanada VPS, **ya quemadas**—
con el checkout actual. Sellada contra `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md`
(`576d02b5…`), no contra el preregistro del meta-aprendiz.

**No abre raíces frescas.** Es una **auditoría de custodia**, no un experimento.

## 2. `f6_seeds_are_virgin` debe leerse `NOT_APPLICABLE`

El runner evalúa `f6` contra una lista `PRIOR_SEEDS` **interna y antigua**, mientras
`research/seed_custody_registry.json` ya marca `6.000.091–120` como **usadas**
(`USED_PENDING_SOURCE_AUDIT`).

**Cualquiera que sea su valor booleano, el falsador no significa aquí lo que su nombre dice.**
Estas semillas **no son vírgenes y no deben serlo**: reproducirlas es el objetivo. Por tanto:

> En este artefacto `f6` se interpreta **`NOT_APPLICABLE`**, y **no** cuenta ni entre los
> falsadores que pasan ni entre los que fallan — la misma regla que apliqué a `f3b` en G3-obs
> tras el sobrealcance de «los ocho falsadores pasan».

Defecto de instrumento derivado, que anoto: **`f6` consulta una lista interna en vez del registro
central**. Mientras siga así, ningún runner de este repo puede certificar virginidad — sólo
ausencia de colisión con una lista congelada. Debe apuntarse al registro.

## 3. El techo de lo que esta corrida puede concluir

El contrato H3′ exige `f_merge_source_is_identical`, que **compara el hash del script**. Una
réplica conductual **no puede satisfacerlo por construcción**. En consecuencia:

| desenlace | lectura permitida |
|---|---|
| las cifras coinciden | **`BEHAVIORAL_REPRODUCIBILITY_FOR_H3_ESTIMAND`** — el estimando se reproduce bajo el checkout actual. **NO** `MERGE_VALID`, **NO** 120 réplicas |
| las cifras difieren | la rebanada VPS queda **anulada**; H3′ se queda con las **90 locales** |

**En ninguno de los dos casos** entran `+5,04` o `+4,92` al manuscrito, y el estado canónico
sigue siendo `ARTIFACTS_PRESENT_MERGE_PENDING`.

## 4. Dos correcciones de lenguaje que acepto de la revisión

**(a) «Garrido pidió la prima de cómputo» es impreciso, y la imprecisión es mía.** Lo que pidió el
28 de julio es una **ventaja comparativa de eficiencia**: **parámetros, velocidad de entrenamiento
y convergencia**. **Latencia online, número de llamadas al DES y amortización son extensiones
NUESTRAS** y deben etiquetarse como tales, no atribuírsele. La distinción no es cosmética: eficiencia
de *entrenamiento* y coste de *inferencia* son estimandos distintos y se congelan por separado:

1. **menos parámetros** a igual calidad y seguridad;
2. **menos tiempo o muestras** para converger;
3. **menor latencia/coste online** a igual calidad ← **nuestra**, no suya.

**(b) «Hemos probado las tres familias y respondido Q1» necesita su alcance.** El panel WRAP, la
superficie Cobb-Douglas, G2 y Program Q **no son un único benchmark comparable**: son contratos
distintos, con endpoints, físicas y comparadores distintos. La formulación defendible:

> En **cuatro contratos separados** no apareció prima neural material sobre los controles
> estructurados **probados en cada uno**. Eso responde Q1 **condicionalmente**, no
> universalmente — y el precio del efecto Alzheimer sigue **pendiente de custodia H3′**.

**(c) Y «MPC» no puede ser una etiqueta para cualquier búsqueda estructurada.** En E\* debe fijar
horizonte, solver, observaciones, presupuesto de cómputo y restricciones antes de correr. Ya
tenemos el precedente: `run_k3_strong_mpc.py` es una búsqueda en rejilla que lleva «MPC» en el
nombre.
