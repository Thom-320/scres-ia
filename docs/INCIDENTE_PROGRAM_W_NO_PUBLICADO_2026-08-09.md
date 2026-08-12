# Incidente de reproducibilidad — Program W no quedó publicado

**Fecha:** 2026-08-09

**Estado:** `ORIGINAL_PROGRAM_W_OBJECTS_NOT_LOCATED_IN_CURRENT_CLONE_OR_INVENTORIED_REMOTE_REFS`

## Hecho verificable

La conversación y el preregistro de escasez mencionan un informe local de Program W y el objeto
`e761ef4`. En el clon actual ese objeto no existe (`git cat-file` falla) y no aparece en las refs
remotas inventariadas. Dentro de ese alcance tampoco se localizaron un runner, contrato, filas
crudas y artefacto sellado que permitan reconstruir su identidad exacta.

La ausencia no autoriza a recrear commits y presentarlos como los originales. Program W queda
fuera del inventario cuantitativo verificable de esta auditoría. La única proposición conservada es una narración
prospectiva escrita antes del barrido posterior: predecía `H_ret=0`; el barrido de almacenamiento
obtuvo cero, aunque porque `H_priv` también era cero.

## Qué puede y qué no puede afirmarse

| afirmación | estado |
|---|---|
| «Program W produjo un resultado confirmatorio» | prohibida: no hay objeto ni artefacto verificable |
| «Una nota previa atribuida a W predijo H_ret=0» | permitida como procedencia narrativa, no como resultado |
| «El barrido posterior obtuvo H_ret=0» | verificable en su propio artefacto, independiente de W |
| «W puede reabrirse sobre sus semillas originales» | no, hasta recuperar el clon/bundle original |
| «Puede construirse un sucesor» | sí, con nombre, contrato, seeds y estatus nuevos |

## Rutas de recuperación

1. Buscar un clon, bundle, reflog exportado o máquina que todavía contenga `e761ef4` y verificar el
   SHA completo antes de incorporarlo.
2. Si el objeto no aparece, reconstruir desde los documentos sobrevivientes sólo con la etiqueta
   `FORENSIC_RECONSTRUCTION_NOT_ORIGINAL_NOT_CONFIRMATORY`.
3. No bloquear el trabajo nuevo en esa recuperación: Program X es un diseño candidato independiente,
   no hereda ningún resultado de W y su preflight no autoriza seeds ni learner.

Esta nota evita dos errores simultáneos: perder conocimiento de que W fue intentado y convertir una
memoria de conversación en evidencia científica.
