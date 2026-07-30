# Preregistro — la duración de R12 y R13: lectura paralela contra serial

**Estado:** `PREREGISTRATION_NOTHING_APPLIED`. Requiere firma del PI.
Ninguna constante cambiada. Toda cifra congelada permanece como fue reportada.

---

## 1. Corrección de premisa: no falta ningún parámetro

En `docs/RPJ_P95_R1R_DIAGNOSIS_2026-07-30.md` escribí que R12 y R13 «no tienen
distribución de duración» y que había que proponerla contra los datos de Garrido.
**Eso está mal, y lo corrijo antes de preregistrar nada.**

La tesis las especifica textualmente, Tabla 6.6b, supuesto de modelado (3):

> **R12** — *«if one of the contracting process is delayed, **one week (168 hours)**
> is added to MLA processing time»*
>
> **R13** — *«if one delivery is delayed, **one day (24 hours)** is added to each
> supplier's processing time»*

No son distribuciones: son **constantes deterministas**. Y el código ya las tiene
(`supply_chain.py:5278`, `:5313`). **No hay nada que ajustar.** Este preregistro no
propone un parámetro; resuelve una **ambigüedad de lectura** de esa frase.

## 2. La ambigüedad, que es la única pregunta abierta

```python
delayed = self._risk_rng_for("R12").binomial(n, p)   # n = 12 contratos
if delayed > 0:
    delay = delayed * 168                             # <-- multiplica
```

`delayed` es el **número** de contratos demorados de los 12. El código multiplica la
semana por ese número, de modo que 8 contratos demorados bloquean Op1 durante 1.344 h.

Las dos lecturas de la frase de la tesis:

| brazo | lectura | duración del evento |
|---|---|---|
| **S** (statu quo) | las demoras se **acumulan en serie** | `k · 168` (R12), `k · 24` (R13) |
| **P** (paralela) | cada contrato demorado llega una semana tarde, **simultáneamente** | `168` si `k ≥ 1` (R12), `24` si `k ≥ 1` (R13) |

**La lectura P es la literal.** La frase está en singular y describe qué le pasa a *un*
contrato («if **one** of the contracting process is delayed»). Los 12 contratos son de
12 materias primas distintas con 12 proveedores distintos, y la propia tabla los declara
**independientes entre sí** (supuesto (2): *«which are considered independent of each
other»*). Procesos independientes que se demoran a la vez se demoran **en paralelo**; la
suma serial requiere una dependencia que la tesis niega explícitamente en la misma celda.

Una tercera lectura —el máximo sobre los contratos demorados— **coincide numéricamente
con P**, porque todas las demoras valen lo mismo. Se anota y no se corre por separado.

**Ninguno de los dos brazos tiene un parámetro libre.** 168 y 24 son de la tesis y no se
tocan en ningún brazo. Esto es una decisión de lectura, no una calibración, y por eso es
preregistrable sin riesgo de ajuste a resultado.

## 3. Por qué esta es la hipótesis, declarada antes de correr

De `RPJ_P95_R1R_DIAGNOSIS_2026-07-30.md` (medido, ya commiteado en 884c035):

* la cola de R1r es **enteramente** aprovisionamiento — toda orden por encima del p95 de
  Garrido lleva **R13**, y con **R12** presente la mediana salta a 3.000 h;
* **R2r reproduce a 0,7 SD**, y R2r no contiene ni R12 ni R13. Sus riesgos usan
  duraciones exponenciales y **ningún multiplicador**.

Es decir: los dos únicos riesgos con el multiplicador son los dos que producen la cola, y
la familia sin multiplicador es la que ya reproduce.

Bajo el nivel «+» de la tesis (Tabla 6.12, `R12: B(12, 4/11)`), la media de contratos
demorados es 4,36, de modo que el brazo S produce un evento de **733 h**; y R13
(`B(12, 4/10)`) produce **115 h** por evento, con solapamiento entre eventos.

## 4. Predicción declarada por adelantado

**Declaro el signo antes de medir**, que es lo que hace falsable esto:

1. El brazo **P reduce `rpj_p95` en R1r** desde 1.869,6 h. **No declaro que lo lleve a
   456,5.** Si lo cerrara exactamente sería sospechoso, no confirmatorio.
2. El brazo P **no puede empeorar** `rpj_p95` en R1r. Si lo empeora, mi mecanismo está mal
   y el preregistro se cierra en negativo.
3. `ret_mean` en R1r está hoy en **1,6 SD** y **no debe degradarse**. Es el endpoint que
   reporta el manuscrito y tiene prioridad sobre `rpj_p95`.

## 5. Falsador del instrumento

**R2r no contiene ni R12 ni R13.** Por tanto los seis momentos de R2r deben salir
**bit-idénticos** entre los brazos S y P. Si alguno se mueve, la implementación tocó algo
que no debía y la corrida se detiene sin reportar.

Segundo falsador: en el brazo P, la duración de todo evento R12 debe ser exactamente 168,0
y la de todo evento R13 exactamente 24,0. Se verifica sobre `sim.risk_events`.

## 6. Criterio de aceptación — multi-momento, no un solo blanco

Esta es la sexta vez que el proyecto enfrenta la tentación de cerrar un observable y romper
otro. La aceptación es por **dominancia sobre los seis momentos**, con la referencia
`fidelity_reference_v3` y el `d_k` ya definido en `supply_chain/fidelity_moments.py`:

**P se adopta si y solo si:**

* `d_k(ret_mean)` en R1r **no empeora más de 0,5** (la `EPSILON` ya declarada); **y**
* `d_k(rpj_p95)` en R1r **mejora**; **y**
* ningún otro momento de R1r empeora más allá de `EPSILON`; **y**
* los seis momentos de R2r son bit-idénticos (el falsador de §5).

**Si P mejora `rpj_p95` pero degrada `ret_mean`**, no se adopta y se reporta como
compensación medida. Ese resultado es publicable y no es un fracaso.

**Prohibido** elegir el brazo por el `H_PI` que produzca, por el signo de cualquier
contraste MPC-contra-estático, por que una familia cruce un umbral de servicio, o por que
el resultado sea publicable.

## 7. Declarado por adelantado

| ítem | valor |
|---|---|
| constantes barridas | **ninguna**; 168 y 24 son de la tesis y quedan fijas |
| brazos | S (`k·168`, `k·24`) contra P (`168`, `24`) |
| raíces | **2.300.001–2.300.012**, disjuntas de todo bloque previo |
| familias | R1r (blanco) y R2r (control/falsador) |
| configuración | `S = 1`, buffers 0, nivel «+» de la Tabla 6.12 |
| modo RPj | `elapsed` (ya migrado, `config.py:141`) |
| referencia | `fidelity_reference_v3` (sha `31ecf9f9dae8058a`) |
| criterio | dominancia sobre los seis momentos, `EPSILON = 0,5` |
| resultado esperado | declarado en §4 |

## 8. Alcance — lo que este preregistro NO toca

* **R14** también carece de duración, pero la tesis lo dice a propósito: el ítem defectuoso
  *«is returned to the previous operation for re-processing»*. Es retrabajo, no una demora
  con longitud. **Fuera de alcance.**
* **`autotomy_share = 0,000`** en ambas familias sigue abierto. Si P lo mueve, se reporta
  como efecto observado; **no** es criterio de aceptación aquí, porque no lo declaré como
  predicción y adoptarlo después sería seleccionar por resultado.
* **Nada se reetiqueta.** Program Q, la confirmación H2/H3, el buffer gate, las 90
  configuraciones y la frontera conjunta conservan sus cifras. Si P se adopta, **abre un
  cuerpo de resultados nuevo**, y ambos se reportan con su lectura declarada.

## 9. Firma

Requiere aprobación del PI antes de ejecutar.

La decisión que no me corresponde: si adoptar la lectura paralela cambia el modelo
publicado o se conserva como brazo de sensibilidad. Mi recomendación, y es solo eso: la
lectura P es la que dice el texto y la que es consistente con el supuesto de independencia
declarado en la misma celda de la Tabla 6.6b, así que el brazo S debería considerarse un
**defecto de implementación** y no una variante legítima — pero eso lo decide la medición
de §6, no este párrafo.
