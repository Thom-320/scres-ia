# Enmienda — `f2` reparada, bloque nuevo, y la condición de autoridad de la Puerta C levantada

**Fecha:** 2026-08-09. Enmienda a `docs/PREREGISTRO_CONFIRMACION_PUERTA_B_2026-08-09.md`.
El artefacto bloqueado (`results/program_n/gate_b_confirmation/result.json`) **se conserva** y su
veredicto `BLOCKED_INSTRUMENT` **no se reescribe**.

## 1. La reparación de `f2`, y su control

La forma de desarrollo preguntaba *«¿cambié algo más que el ajuste?»* comparando **niveles** sobre
**las mismas tapas**. Sobre tapas distintas esa comparación se convierte en *«¿reproducen ocho
semillas nuevas el R² de ocho viejas?»*, que no tiene por qué cumplirse: los brazos clásicos son
deterministas dado el dato y se movieron **porque el dato se movió**.

La forma de confirmación hace la pregunta que corresponde **sin depender del dato**:

1. **identidad de código** — `module_manifest` idéntico hash a hash al del desarrollo;
2. **preservación de orden** — `spline ≥ linear_interactions ≥ linear_additive ≥ constant`.

**Ninguna puede fallar por variación de muestreo. Ambas pueden fallar si cambié el instrumento.**

**Control de la reparación, calculado sobre el bloque ya quemado:** la `f2` reparada **habría
pasado** allí — el orden se conservó (0,6022 ≥ 0,5905 ≥ 0,5865 ≥ −0,0132). Eso confirma que
distingue lo que debe distinguir, y confirma también que **aquel bloque se perdió por mi falsador y
por nada más**.

Esto **no reabre** el artefacto bloqueado. Su veredicto se queda; sólo valida la reparación.

## 2. Bloque nuevo

`9500001–9500008`, verificado libre: cero colisiones con el registro y cero con semillas de
artefactos sellados. Ocho semillas, como en desarrollo, para que la única diferencia siga siendo la
tapa. Puerta de un solo sentido, igual que la anterior.

**Todo lo demás sigue congelado**: mismo runner, mismo objetivo, misma rejilla, mismo `SESOI`, mismo
baseline primario, mismos folds, mismo horizonte, re-selección de hiperparámetro por fold sobre la
validación interna de las tapas nuevas.

## 3. La condición de autoridad de la Puerta C, levantada por el PI

`contracts/garrido_expanded_des_e_star_v2_hcompute.json` exige:

```
status: DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT
authority.submission_a_receipt_required_before_scientific_execution: true
authority.garrido_written_approval_required: true
```

**El PI la levanta explícitamente el 2026-08-09.** La justificación está en el propio repositorio:
`docs/DONDE_PODEMOS_SER_LAXOS_2026-08-02.md` §5 registra que **«el bloqueo real de Submission A es
editorial y humano, no experimental»**. Es decir, la condición no protege ninguna propiedad
científica: retrasa la ejecución hasta un hito de calendario.

**Qué se levanta y qué NO:**

* **Se levanta:** el requisito del recibo de Submission A y de la aprobación escrita previa para
  ejecutar la Puerta C.
* **NO se levanta:** ni un solo gate científico. Siguen vigentes `H_compute` (ya pasado), la no
  inferioridad en servicio y recursos, `r2_is_not_a_control_gate`, la escalera de comparadores con
  la red al final, y `quality_authorization: LCB95(Delta_obs) >= SESOI` para cualquier afirmación de
  **calidad**.
* **La Puerta C queda autorizada sólo en su carril de AMORTIZACIÓN**, que es el que el contrato
  permite sin residual de calidad: *«no quality residual is required to test a surrogate of an
  expensive planner»*.

**Y queda escrito lo que esto cuesta:** ejecutar antes del recibo significa que, si Garrido objeta
el constructo más adelante, el resultado de la Puerta C nace con una objeción de alcance que no
tendría si se hubiera esperado. El PI acepta ese precio; el artefacto lo declarará.
