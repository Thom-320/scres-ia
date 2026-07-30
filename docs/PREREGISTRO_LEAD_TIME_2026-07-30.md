# Preregistro — alineación del lead time y el delay de cumplimiento

**Estado:** `PREREGISTRATION_DRAFT_NOTHING_APPLIED`. Ninguna constante cambiada. Toda cifra
congelada permanece como fue reportada.

Este preregistro **corrige una decisión que yo mismo tomé mal** el 2026-07-30 en
`DECISIONES_SIN_GARRIDO_2026-07-30.md`, y la corrección viene de la tesis.

---

## 1. Lo que decidí mal, y la evidencia que lo refuta

Argumenté que `LT` es el tiempo de entrega programado, que el nuestro es de 54 h, y que
por tanto `LEAD_TIME_PROMISE` debía pasar a 54.

**La tesis fija 48 explícitamente.** §6.8.2, p.111:

> *«the availability of finished products at this point allows troops to be supplied within
> a **pre-set lead-time of 48 hours**»*

Es un lead time **preestablecido**, condicionado a que el batallón de abasto tenga
existencias. Mi cita de `config.py:118` a §6.3.4 estaba equivocada —esa sección es
«Demand for combat rations»— pero el valor 48 es correcto y su fuente es §6.8.2.

**Consecuencia: lo desalineado es nuestro pipeline, no la métrica.** La tesis afirma que
con stock en el batallón la tropa se abastece en 48 h; nuestro modelo tarda 54 h aun con
stock. `LT` se queda en 48.

Sostengo lo que sí verifiqué: la tesis parte sobre **`CTj = LTj`** (igualdad) contra
`CTj > LTj`, sin caso `CTj < LTj` (§5.5.1, Algoritmo 1 p.68, p.72), y nuestro código usa
`CTj <= LTj` en `supply_chain.py:5810`.

## 2. El segundo hallazgo: la cola no acotada es de él, no nuestra

**Figura 6.8a (p.115)** es el histograma de `ReT(Cf1)` y su eje horizontal **va de 0 a
120**. El Q-Q plot adjunto muestra observaciones cerca de 120. Los tests KS y SW corren
con **df = 4.241**, que es exactamente el número de filas de Cf1.

Su propio ReT toma valores hasta dos órdenes de magnitud por encima de 1.

Esto es doblemente importante:

1. **Valida nuestra implementación.** Reproducimos su no-acotamiento, no lo introdujimos.
2. **Debilita el clipping como reparación.** §5.6.3 dice que *«ReT is normalized on a 0 to
   1 scale»*, pero los datos publicados no lo están. La tesis es internamente
   inconsistente entre su especificación y su salida. Acotar a [0,1] **se aparta del
   comportamiento publicado**; no lo restaura.

Esto **no revierte** la confirmación prospectiva ya adjudicada —ahí canónica y acotada
coincidieron (+0,01252 contra +0,01247)— pero sí cambia el argumento de §2 del preregistro
de la cola, que decía que el clipping «impone un rango que la especificación ya declara».
La especificación lo declara; los datos del autor no lo cumplen. Queda anotado allí.

## 3. La tensión real que este preregistro debe resolver

`GARRIDO_FULFILLMENT_DELAY_HOURS = 54` **no es un tiempo derivado de nuestro pipeline**.
Es una constante ajustada, documentada el 2026-06-26 como *«the smallest tested value that
crosses the LT=48 cliff and reproduces the Garrido raw-Excel order of magnitude for ReT»* y
etiquetada *«provisional reproduction default, not a complete behavioral calibration»*.

Se ajustó para reproducir **la magnitud de ReT**. Pero ese ajuste fuerza `CTj ≥ 54 > 48`
siempre, y por tanto:

| | nuestro modelo | evidencia de Garrido |
|---|---|---|
| magnitud de ReT | reproduce el orden | ✓ criterio de ajuste original |
| casos de autotomía | **0,00% siempre** | `Media APj = 0,4486` en Cf1 |
| rango de ReT | no acotado | no acotado (Fig. 6.8a, hasta ~120) |

**Un solo criterio de ajuste seleccionó un valor que rompe otro momento observable.** Ese
es el defecto a corregir, y corregirlo exige ajustar contra **varios** momentos, no uno.

## 4. Lo que se barre, y el criterio de selección

**Se barre el delay de cumplimiento**, no `LT`. Valores declarados:

    delay ∈ {24, 36, 42, 47, 48, 49, 54, 60}  horas

Y en paralelo, el predicado de la rama:

    predicado ∈ { CTj <= LTj  (actual),  CTj = LTj ± tol  (letra de la tesis) }

con `tol` declarada en 1e-9 h.

### El criterio es fidelidad multi-momento, declarado antes de correr

La selección **no** se hace por magnitud de ReT sola —ese fue el error original— ni por
headroom, tamaño de efecto, o cualquier cantidad que dependa de qué controlador gana.

Se puntúa cada combinación `(delay, predicado)` por su distancia a los momentos observables
de Garrido, con pesos iguales sobre momentos estandarizados:

| momento | fuente de referencia | por qué |
|---|---|---|
| fracción de casos de autotomía | `Media APj > 0` en Cf1..Cf12 | el momento que el ajuste actual rompe |
| media y p95 de `RPj` | columnas `Media RPj` / `Máximo RPj` | la rama que hoy carga toda la señal |
| media de `ReT` por familia | Cf1–Cf10 (R1r), Cf11–Cf20 (R2r) | el criterio original, retenido pero no único |
| fracción de `ReT > 1` | derivable de Fig. 6.8a y de las columnas `Máximo Re` | la cola, ahora que sabemos que es suya |
| conteo de órdenes puntuadas | filas por hoja | población, no solo valores |

**Fuente canónica:** los tres workbooks reales verificados (0/47.546 discrepancias de
fórmula), **no** `Rsult_1.xlsx`, que ya establecimos que no es la data final de la tesis
(sus 12 configuraciones difieren entre −1.949 y +735 filas). Si esos workbooks no están
disponibles en el momento de correr, el preregistro se detiene: no se sustituye la
referencia por la que haya a mano.

### Prohibición explícita

Queda prohibido elegir `delay` o predicado por:

- el `H_PI` que produzca, a nivel de tape o de época;
- el signo o tamaño de cualquier contraste MPC-vs-estático;
- que una familia pase o no un umbral de servicio;
- que el resultado sea publicable.

Esta prohibición está aquí porque **ya cometí ese error dos veces hoy**: argumenté que
bajar el delay «eliminaría la prima neural» a partir de un `H_PI` de nivel-tape, y antes
que la planitud de medias acotaba el valor adaptativo. Las dos inferencias eran inválidas y
las dos apuntaban a elegir un instrumento por su resultado.

## 5. Qué se re-corre y qué no

**Se re-corre:** el barrido de fidelidad, sobre raíces nuevas declaradas
(2.000.001–2.000.012 por familia), en las tres familias de riesgo R1r/R2r/R3, con la
configuración base de la tesis (`S = 1`, buffers 0) para que la comparación sea contra sus
Cf1–Cf30 y no contra un régimen nuestro.

**No se re-corre ni se reetiqueta:** nada. Program Q, la confirmación H2/H3, el buffer
gate, la reproducción de 90 configuraciones, la confirmación prospectiva de ReT y la
frontera conjunta de 648 conservan sus cifras bajo la métrica con la que se calcularon. Si
el barrido cambia el delay, **eso abre un cuerpo de resultados nuevo, no reescribe el
viejo**, y ambos se reportan con su constante declarada.

## 6. Declarado por adelantado

| ítem | valor |
|---|---|
| constante barrida | `demand_on_hand_fulfillment_delay` |
| constante **no** barrida | `LEAD_TIME_PROMISE = 48` (tesis §6.8.2) |
| predicado | `<=` actual contra `=` de la tesis, ambos medidos |
| raíces | 2.000.001–2.000.012, disjuntas de todo bloque previo |
| familias | R1r, R2r, R3 |
| criterio | distancia multi-momento a los workbooks reales, pesos iguales |
| desempate | el delay **menor** entre los que queden dentro del 10% del mejor puntaje |
| resultado esperado | **no declarado** — a diferencia del preregistro de la cola, aquí el signo no se conoce, y decirlo importa |

**Falsador del instrumento:** si `delay = 54` con predicado `<=` **no** reproduce la
magnitud de ReT que ya sabemos que reproduce, el barrido está mal implementado y se
detiene. Es la única celda cuyo resultado conocemos de antemano, y sirve de compuerta.

## 7. Qué autoriza y qué no

**Autorizará:** reportar qué combinación `(delay, predicado)` reproduce mejor el conjunto
de momentos observables de Garrido, y con qué distancia residual.

**No autorizará:** entrenar nada; reemplazar cifras congeladas; ni afirmar que el modelo
nuevo es «más fiel» sin declarar en qué momentos mejora y en cuáles empeora — porque es
casi seguro que habrá de los dos.

## 8. Firma

Requiere aprobación del PI antes de ejecutar. La decisión que no me corresponde: si el
cuerpo de resultados del proyecto se migra a la constante ganadora, o si la constante
actual se conserva como línea de reproducción y la nueva se abre como estudio paralelo.
