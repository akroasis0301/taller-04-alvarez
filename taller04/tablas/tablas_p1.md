# Parte 1 — Observabilidad

### Resumen (base)

| métrica | valor |
|---|---|
| corridas | 6 |
| corridas sin span «corrida» | 0 |
| tokens de entrada | 305317 |
| tokens de salida (incluye razonamiento) | 565581 |
| costo real (H200) | 0 USD |
| costo de referencia | 0.3851 USD — fila propietario_economico (gpt-4o-mini), verificada el 2026-08-26 |
| latencia p50 por corrida | 468.8 s |
| latencia p95 por corrida | 1534.1 s |
| spans con error (sin la raíz) | 2 |

### Por corrida (base)

| tarea | duración (s) | llamadas al modelo | tokens entrada | tokens salida | status | traza_id |
|---|---|---|---|---|---|---|
| A | 407.8 | 12 | 29958 | 55610 | completado | 5fa56ad897e2 |
| B | 304.8 | 12 | 29847 | 44553 | completado | cd9a032a3228 |
| C | 859.1 | 13 | 51913 | 116147 | completado | affc55114a65 |
| S1 | 1,759.1 | 22 | 93492 | 240851 | completado | bdf64164acd1 |
| H | 529.8 | 15 | 62429 | 60000 | completado | e1626a32e6cb |
| P | 162.9 | 13 | 37678 | 48420 | completado | 9fb47b05e437 |

### Por agente (base) — el cuello de botella es el de mayor TIEMPO TOTAL

| agente | spans | tiempo total (s) | parte del tiempo | tiempo medio (s) | llamadas | tokens entrada | tokens salida | parte de los tokens |
|---|---|---|---|---|---|---|---|---|
| programador | 23 | 2,957.5 | 73.5 % | 128.6 | 26 | 103483 | 392808 | 57.0 % |
| redactor | 9 | 433.0 | 10.8 % | 48.1 | 9 | 69400 | 56329 | 14.4 % |
| critico | 23 | 340.8 | 8.5 % | 14.8 | 21 | 114631 | 43991 | 18.2 % |
| indexador | 6 | 166.1 | 4.1 % | 27.7 | 25 | 10391 | 61335 | 8.2 % |
| planificador | 6 | 91.0 | 2.3 % | 15.2 | 6 | 7412 | 11118 | 2.1 % |
| ejecutor | 23 | 24.1 | 0.6 % | 1 | 0 | 0 | 0 | 0.0 % |
| investigador | 20 | 9.4 | 0.2 % | 0.5 | 0 | 0 | 0 | 0.0 % |
| lector | 6 | 0.4 | 0.0 % | 0.1 | 0 | 0 | 0 | 0.0 % |

Suma de los spans de agente: 4,022.3 s; suma de las corridas: 4,023.5 s (los agentes son secuenciales: deben coincidir).

### Por tipo de span, dentro de los agentes (base)

| tipo de span | spans | tiempo total (s) | medio (ms) | respecto del reloj de las corridas |
|---|---|---|---|---|
| llm | 87 | 4,284.0 | 49,241.9 | 106.5 % |
| herramienta | 29 | 24.3 | 838.2 | 0.6 % |
| embedding | 6 | 9.6 | 1,599.2 | 0.2 % |
| retriever | 40 | 9.3 | 231.7 | 0.2 % |
| guardrail | 53 | 0.1 | 2.7 | 0.0 % |

Los `llm` pueden sumar más que el reloj: el indexador llama al modelo en paralelo (8 hilos).
