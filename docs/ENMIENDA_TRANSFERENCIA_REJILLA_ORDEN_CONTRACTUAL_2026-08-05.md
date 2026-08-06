# Enmienda — replay ordenado de la transferencia 288 → 4.608

**Escrita antes del replay.** Esta enmienda no abre semillas: solo reejecuta el análisis sobre las
cachés selladas del bloque quemado `5.300.001–5.300.012`.

## Motivo

El resultado exploratorio `results/grid_transfer_v2/result.json` fue calculado recorriendo los
contextos en el orden producido por `rglob`:

```text
R1r, R1r+R2r, R1r+R2r|esc, R1r|esc, R2r, R2r|esc
```

El orden contractual de una carrera de experiencias es:

```text
R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc
```

Como el estado UCB1, de la neurona, OFAT y GP cruza contextos, el orden altera el estimando. El
resultado exploratorio queda **SUPERSEDED_ORDERING**, no se borra y no se cita como confirmación.

## Regla

El runner debe exigir los seis contextos y recorrerlos exactamente en el segundo orden. Se
mantienen idénticos el presupuesto `B=24`, las cuatro familias, el placebo marginal, el bootstrap
por semilla y todos los datos de las cachés. El replay se sella como análisis de desarrollo y no
autoriza RL, PPO, MLP, KAN ni nuevas semillas.

El artefacto citable de esta enmienda será el resultado que declare el orden contractual en su
campo `contexts`. Si el ranking o los intervalos cambian, se reemplazan las cifras anteriores en
el informe; no se promedian ambos órdenes.
