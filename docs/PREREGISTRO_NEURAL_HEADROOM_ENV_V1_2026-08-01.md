# Preregistro — screen de headroom para prima neural

**Contrato:** `garrido_wrap_scres_ai_v1`
**Estado inicial:** `E0_E1_SCREEN_ONLY`
**Propósito:** decidir si existe una superficie de decisión suficientemente no lineal antes de
entrenar MLP, PPO o PPO recurrente.

## Entornos

### E0 — WRAP thesis-native

Superficie original de buffers, shifts y riesgo. Se usa para falsar la prima neural en el panel
de Garrido. La tarea `drivers → ReT` queda excluida por identidad.

### E1 — CSSU split no fungible

Extensión separada de `thesis_1to1`:

- dos reclamantes CSSU;
- capacidad compartida no fungible;
- `alpha ∈ [0, 1]` continuo;
- R23 como outage local y R24 como demanda localizada;
- endpoint `service_first_v2`;
- placebo sin información de régimen;
- constantes, reglas de umbral y superficie completa antes de cualquier red.

El resultado actualmente sellado de E1 da `H_regime = 0` en el componente líder y no autoriza
entrenamiento.

### E2 — CSSU parcialmente observable

E2 sólo se puede abrir mediante un preregistro sucesor si E1 demuestra headroom. El sucesor debe
fijar antes de correr:

- régimen latente persistente;
- observación retrasada o incompleta;
- historial observable de entregas, backorders y fallos;
- decisión diaria y latencia de acción;
- tapes comunes y evaluación en regímenes nuevos.

La hipótesis de E2 no es “RL siempre gana”, sino:

> un estado recurrente puede aportar cuando el estado relevante está aliasado antes de actuar y
> una regla estática/finita no alcanza el oráculo.

## Gate de apertura

E2 y cualquier entrenamiento requieren simultáneamente:

1. `H_obs >= 0.01`;
2. `LCB95(H_obs) > 0`;
3. el placebo no captura la ganancia;
4. el `argmax` cambia entre estados o regímenes;
5. la acción es físicamente viva;
6. el mejor control clásico no alcanza el oráculo;
7. no hay ganancia causada por abandono;
8. calibración y evaluación usan semillas disjuntas.

## Escalera de modelos

```text
constante → umbral → regla finita → lineal/logística → árbol/tabular → MLP → PPO → PPO recurrente
```

PPO recurrente sólo se abre si el test de aliasado muestra que la observación feed-forward no
identifica el estado y si PPO feed-forward no resuelve el problema.

## Prima neural

Para predicción, la prima exige `ΔR² >= 0.05` frente al lineal con IC95 pareado excluyendo cero.
Para control, exige `LCB95(Δ worst_claimant_fill) >= 0.01` o una reducción de al menos una
corrida en regret con intervalo favorable, sin empeorar fill agregado, backorders, recursos ni
el peor reclamante.

Si una regla finita o MPC alcanza el oráculo, el resultado es `NO_GO_NEURAL_PREMIUM`: la
no-linealidad existe, pero no necesita una red.
