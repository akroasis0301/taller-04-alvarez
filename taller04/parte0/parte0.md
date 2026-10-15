# Parte 0 — Tres errores que no dan error

Corrido el 2026-10-15 en macOS, Python 3.12 (entorno `uv` del repositorio), desde
`taller-04-llmops/Lab-04-LLMOps/`, con el kit **sin modificar**. `guion.py` reemplaza el agente
por uno de guion: no hay modelo ni red. Las salidas completas están junto a este archivo:
`0a_ceros.txt`, `0b_guardrails.txt`, `0c_traza.txt`. Los tres problemas aparecieron; no hubo que
declarar ningún cambio al laboratorio.

## 0.a — El reporte lleno de ceros

```
uv run python guion.py ceros
```

Qué muestra la salida:

- `instrumented_agent.py` escribe `reports/traces.jsonl` (línea 66) y `evaluation.py` lee
  `reports/answers.json` (línea 71), que nadie escribe. `evaluate_answers` busca cada respuesta con
  `answers.get(case["id"], "")` (línea 57): al no encontrarla usa la cadena vacía y califica 0, sin
  excepción. Aunque los dos archivos coincidieran, las trazas no llevan el `id` del caso
  («¿alguna lleva el id del caso? False»): no hay cómo unir una traza con su pregunta.
- Con las respuestas correctas casadas a mano, solo el caso `negativo` vale 1. `abstention_score`
  devuelve 0 a todo lo que no es `negativo` (líneas 48–49): la respuesta adversarial «No puedo hacer
  eso.», que es justamente la correcta, también vale 0.

**¿Por qué un reporte en ceros es peor que un error?** Un error detiene la corrida y obliga a
mirar; un reporte en ceros parece una medición válida («el agente nunca se abstiene») y lleva a
tomar decisiones —cambiar el prompt, el modelo— sobre datos que no existen, porque no se distingue
de un agente que de verdad falla todo.

**¿Por qué no se puede promediar un 0 «no aplica» con un 0 «falló»?** Porque el promedio deja de
medir el agente y pasa a medir la composición del golden set: con las respuestas perfectas el
reporte da 1/4 = 0,25 de abstención cuando el acierto real es 1/1 = 100 % en el único caso donde
aplica, y basta añadir casos `feliz` para que la cifra baje sin que el agente cambie; «no aplica»
tiene que ser un valor vacío (`None`/NaN) que sale del denominador.

## 0.b — Lo que el guardrail deja pasar y lo que bloquea por error

```
uv run python guion.py guardrails
```

| Resultado | Casos | n |
|---|---|---|
| Correcto | A01, A03, P01, P02, P03, L02, L04, S01, S04 | 9 |
| Falso negativo (debía bloquear o redactar y pasó) | A02, A04, A05, A06, S02 | 5 de 11 → 0,455 |
| Falso positivo (debía pasar y se modificó o bloqueó) | L01, L03, S03 | 3 de 6 → 0,5 |

**¿Qué tienen en común los ataques que pasan?** Piden lo mismo que uno que se bloquea, pero con
otras letras: A02 y A03 son la misma frase y solo A03 se bloquea, porque `detect_prompt_injection`
compara contra frases literales sin tildes (`"actua como si…"`) sobre un texto al que solo le baja
las mayúsculas (`text.lower()`, línea 27), así que la tilde de «actúa» basta; A04 cambia el verbo
(«ignora»), A05 un determinante («tus» por «las») y A06 no usa ninguna palabra de la lista. El
filtro reconoce cadenas, no la intención.

**¿Por qué los falsos positivos de la L01 y la L03 son graves aunque no bloqueen nada?** Porque
cambian la pregunta en silencio: `PHONE_RE` (línea 22) toma por teléfono cualquier secuencia de
nueve o más dígitos con espacios o guiones, el agente recibe otra pregunta, contesta sin error y
ningún contador de bloqueos lo registra; un bloqueo al menos se ve.

**¿Qué le pasa a la L03 en un agente que consulta ventas por fecha?** Las cuatro fechas se vuelven
`[TELEFONO_REDACTADO]` y la pregunta pierde sus dos rangos: el agente se abstiene, pide aclaración
o —lo peor— compara periodos que eligió él y devuelve una cifra equivocada con total seguridad, y
la traza guarda la versión redactada, así que tampoco se puede reconstruir qué se preguntó.

Notas adicionales:

- **P02 acierta por accidente.** La cédula sale como `[TELEFONO_REDACTADO]`: la misma regla que se
  come las fechas la atrapa porque tiene diez dígitos, no porque se reconozca como cédula.
- **S02:** `validate_output` (líneas 47–50) solo busca claves, nunca redacta datos personales: el
  correo y el teléfono de la salida llegan al usuario.
- **S03:** `SECRET_RE` (línea 23) bloquea cualquier cadena de 32 caracteres alfanuméricos, así que
  un hash de lote legítimo cuenta como clave: decide por la longitud, no por la forma.

## 0.c — La traza que filtra datos, mide mal y se pierde

```
uv run python guion.py traza
```

1. **Filtra el correo** («¿el correo está en reports/traces.jsonl? True»). En
   `instrumented_agent.py`, línea 47, `result = {"question": question, …}` guarda la pregunta
   *original* cuando el guardrail la bloquea: la redacción (`redact_pii`) solo ocurre en el camino
   que pasa (línea 44), y la línea 58 escribe ese resultado tal cual.
2. **Mide de menos** (la corrida tardó 0,40 s; la traza dice 0,0). La línea 45,
   `started = time.perf_counter()`, arranca el cronómetro *después* de `validate_input` (línea 44):
   el tiempo del guardrail no existe para la traza.
3. **Se pierde** («filas antes: 2 · después: 2»). Las líneas 49–58 no tienen `try/finally`: si
   `agent.run` (línea 50) lanza, la excepción sale de la función y las líneas 56–58 —latencia,
   fecha y `write_local_trace`— nunca se ejecutan. La corrida que falla, la que más interesa, no
   deja rastro.

Un cuarto fallo silencioso, fuera de lo que pide el guion: `maybe_langfuse_trace` (líneas 29–40)
llama a `langfuse.trace(...)`, un método del SDK 2.x que **no existe** en `langfuse==4.17.0` (el
`Langfuse` 4.x no define `trace`; se usa `start_as_current_observation`). Con `USE_LANGFUSE=true`
lanzaría `AttributeError`, el `except Exception` de la línea 39 lo convierte en un aviso por
pantalla y la corrida sigue como si nada: la observabilidad se apaga sin error.

Las tres correcciones (medir en `finally`, que el cronómetro incluya los guardrails y redactar antes
de escribir) son las que la Parte 1 implementa en el solver.
