# Preregistro — sucesor del techo monótono, con LCB, multiplicidad y prueba de potencia

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_monotone_transform_family_v2.py`.
Predecesor: `results/monotone_transform_ceiling/result.json`
(`A_MONOTONE_RESCALING_REACHES_THE_BAR_WITH_SIGNAL_INTACT`, `H = 0,0742`).
Semillas: bloque quemado `garrido_q2_des288`, réplica declarada. **Ninguna nueva.**

## 1. Los tres defectos del predecesor que esto repara

1. **Sin LCB.** `0,0742` era un estimador puntual. Aquí cada transformación lleva **LCB95 por
   bootstrap sobre semillas**, que es la unidad de replicación.
2. **Sin multiplicidad.** Era un máximo sobre ~2.500 transformaciones. Aquí la familia está
   **enumerada de antemano** y se corrige por **Holm** sobre ella entera.
3. **Un proxy de señal que no podía caer.** Usé pares ordenados, y ninguna función estrictamente
   creciente puede desordenar un par salvo por empates numéricos: el falsador pasó sin poder
   fallar. Se sustituye por una **razón señal-ruido** que sí puede caer, **y se valida
   reintroduciendo el defecto**.

## 2. La familia, enumerada aquí y cerrada

| subfamilia | parámetros | n |
|---|---|---:|
| identidad | — | 1 |
| logística | 25 umbrales (cuantiles 0,02–0,98) × 20 nitideces (`β` de 0,05 a 500) | 500 |
| potencia | `((v−lo)/(hi−lo))^γ`, `γ` de 0,1 a 10 | 21 |
| escalón | 99 umbrales por cuantil | 99 |

**`K = 621`.** No hay muestreo aleatorio: una familia que no se puede enumerar no se puede
corregir. Las monótonas aleatorias del predecesor existían sólo para falsar el argumento del
escalón, y ya lo hicieron.

## 3. El proxy de señal, y cómo se valida

```
SNR(f) = mediana sobre contextos de [ SD_configs(media_semillas f(V)) / media_configs(SD_semillas f(V)) ]
```

Es literalmente lo que importa para entrenar: **cuánta variación entre configuraciones sobrevive
por encima del ruido entre semillas**. Una logística muy nítida satura las colas, la variación
entre configuraciones se colapsa y el ruido no, así que **SNR cae**. El orden por pares no podía.

**Validación obligatoria:** un escalón extremo debe dar `SNR < 0,5 × SNR(identidad)`. Si no cae ahí,
el proxy nuevo es tan inerte como el viejo y **este preregistro se declara fallido**.

## 4. La prueba de potencia, y por qué va antes del veredicto

La rejilla extendida tiene **3 semillas**. Un bootstrap sobre 3 semillas es un instrumento débil, y
lo digo antes de mirar: **un nulo podría significar «no hay efecto» o «no hay potencia», y sin
distinguirlos el nulo no vale nada.**

Por eso se planta una superficie sintética con un óptimo **distinto por régimen**, calibrada por
bisección a `H ≈ 0,10`, con el ruido entre semillas **observado** en los datos reales, y se le pasa
la misma maquinaria de bootstrap.

* **Si el LCB de la superficie plantada no cruza el umbral, el veredicto es
  `UNDERPOWERED_NO_VERDICT`**, gane quien gane el resto. No se reporta como negativo.

## 5. Reglas de lectura, fijadas antes de mirar

Umbral `GATE = 0,05`. Una transformación **califica** si cumple **las tres**:

```
LCB95 >= 0,05      y      p ajustado por Holm sobre K=621 < 0,05      y      SNR >= 0,90 x SNR(identidad)
```

* potencia insuficiente → **`UNDERPOWERED_NO_VERDICT`**
* alguna califica → **`A_MONOTONE_RESCALING_SURVIVES_LCB_AND_MULTIPLICITY`**
* ninguna califica pero alguna tenía `LCB ≥ 0,05` antes de Holm →
  **`THE_CEILING_DOES_NOT_SURVIVE_MULTIPLICITY`**
* ninguna llega ni con `LCB` → **`NO_MONOTONE_RESCALING_SURVIVES_LCB`**

**Y la regla que no cambia:** aunque alguna califique, **no se adopta**. Un `H_regime` que sube
reescalando es una propiedad de la **curvatura de la métrica**, no de la física de la cadena — el
orden de configuraciones es idéntico por construcción. La adopción exigiría mecanismo declarado y
confirmación en bloque virgen.

## 6. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_identity_reproduces_the_sealed_scalar` | `H(identidad)` debe igualar `scalar_h_regime` sellado a 1e-9. **Ancla externa** |
| `f2_the_signal_proxy_can_actually_fall` | un escalón extremo debe dar `SNR < 0,5 ×` la identidad. **Falla si el proxy nuevo tampoco puede caer**, que es el defecto que este preregistro existe para reparar |
| `f3_the_instrument_has_power` | la superficie plantada a `H ≈ 0,10` debe dar `LCB95 ≥ 0,05`. Falla con 3 semillas si el bootstrap no alcanza, y entonces **no hay veredicto** |
| `f4_multiplicity_is_applied_over_the_declared_family` | `K` debe ser exactamente 621 y ningún `LCB` puede compararse contra el umbral sin Holm. Falla si la familia crece durante la corrida |
| `f5_the_base_grid_stays_at_zero` | la rejilla de 288 debe seguir dando 0 en toda la familia. **Control negativo**: falla si el instrumento nuevo inventa headroom donde el argmax es universal |
| `f6_no_fresh_seeds` | custodia central, réplica declarada |

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica, no autoriza
aprendices, **no cambia la primaria del contrato** y **no adopta ninguna transformación**.
