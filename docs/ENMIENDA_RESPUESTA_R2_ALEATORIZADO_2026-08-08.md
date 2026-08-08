# Enmienda — la inferencia central de `RESPUESTA_GARRIDO_R2_ALEATORIZADO` es falsa

**Sucede a** `docs/RESPUESTA_GARRIDO_R2_ALEATORIZADO_2026-08-08.md`. No lo edita: aquel documento
se conserva con su fecha y su contenido, y esta enmienda retira **una** afirmación suya.

## Lo que se retira

> *«Aleatorizar frecuencia e impacto **promedia** sobre perfiles… Escalar cada riesgo uno a uno
> **aísla** el perfil… Si no se mueve —o si moverse no paga— bajo escalada sistemática de cada
> riesgo de la familia, no lo hará bajo sorteos de esa misma familia.»*

**No se sostiene, y la escribí yo.** El screen de 4.860 evaluaciones que la respaldaba cubrió:

* perfiles **fijos durante todo el episodio**;
* **18 posturas constantes**;
* el antiguo `ret_excel`, del que después medimos que premia el abandono;
* variación **entre** perfiles, no inferencia temporal **dentro** del episodio.

De ahí no se sigue nada sobre interacciones entre riesgos, regímenes ocultos a la política,
realizaciones que cambian dentro de la corrida, ni políticas secuenciales con presupuesto. Una
escalada uno-a-uno de perfiles fijos es **un diseño distinto**, no un diseño más severo del mismo
tipo. El crédito de haberlo visto es de un auditor externo.

## El estado correcto

```text
R2_FIXED_PROFILE_CONSTANT_POSTURES      SCREENED_DEVELOPMENT
R2_RANDOMIZED_ARCHITECTURE_BENCHMARK    NOT_RUN
```

## Lo que no cambia

Todo lo demás de aquel documento sigue en pie, incluida su corrección más importante: que
`unique_profile_optima` vale 1, 2 o 3 según la fila, y que **el óptimo sí se mueve entre a lo sumo
tres posturas — lo que no compra nada es moverse con él** (`6,93e−05` contra una barra de 0,01).

Y sigue en pie el hueco que aquel documento nombró con precisión y que ahora se cierra por la vía
correcta: el benchmark que Garrido pidió, preregistrado en
`docs/PREREGISTRO_GARRIDO_R2_RANDOMIZED_BENCHMARK_V1.md`.
