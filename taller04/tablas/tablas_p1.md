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

### Resumen (mejora)

| métrica | valor |
|---|---|
| corridas | 6 |
| corridas sin span «corrida» | 0 |
| tokens de entrada | 247638 |
| tokens de salida (incluye razonamiento) | 357898 |
| costo real (H200) | 0 USD |
| costo de referencia | 0.2519 USD — fila propietario_economico (gpt-4o-mini), verificada el 2026-08-26 |
| latencia p50 por corrida | 285.1 s |
| latencia p95 por corrida | 606.5 s |
| spans con error (sin la raíz) | 0 |

### Por corrida (mejora)

| tarea | duración (s) | llamadas al modelo | tokens entrada | tokens salida | status | traza_id |
|---|---|---|---|---|---|---|
| A | 645.3 | 15 | 46104 | 72220 | completado | afd26539106f |
| B | 175.8 | 12 | 26566 | 29880 | completado | 3ecf7e5baead |
| C | 394.4 | 12 | 46436 | 78221 | completado | b909e81cb73e |
| S1 | 490.2 | 18 | 69433 | 89152 | completado | 39acfbb16389 |
| H | 169.1 | 13 | 31059 | 49083 | completado | 468ba03470e1 |
| P | 126.5 | 12 | 28040 | 39342 | completado | 7b8bb79a0db0 |

### Por agente (mejora) — el cuello de botella es el de mayor TIEMPO TOTAL

| agente | spans | tiempo total (s) | parte del tiempo | tiempo medio (s) | llamadas | tokens entrada | tokens salida | parte de los tokens |
|---|---|---|---|---|---|---|---|---|
| programador | 21 | 783.6 | 39.2 % | 37.3 | 21 | 80314 | 178617 | 42.8 % |
| indexador | 6 | 402.9 | 20.1 % | 67.2 | 25 | 10391 | 63398 | 12.2 % |
| investigador | 20 | 337.4 | 16.9 % | 16.9 | 0 | 0 | 0 | 0.0 % |
| redactor | 9 | 210.1 | 10.5 % | 23.3 | 9 | 59026 | 56803 | 19.1 % |
| critico | 21 | 206.4 | 10.3 % | 9.8 | 21 | 90495 | 49042 | 23.0 % |
| planificador | 6 | 39.1 | 2.0 % | 6.5 | 6 | 7412 | 10038 | 2.9 % |
| ejecutor | 21 | 20.4 | 1.0 % | 1 | 0 | 0 | 0 | 0.0 % |
| lector | 6 | 0.4 | 0.0 % | 0.1 | 0 | 0 | 0 | 0.0 % |

Suma de los spans de agente: 2,000.3 s; suma de las corridas: 2,001.3 s (los agentes son secuenciales: deben coincidir).

### Por tipo de span, dentro de los agentes (mejora)

| tipo de span | spans | tiempo total (s) | medio (ms) | respecto del reloj de las corridas |
|---|---|---|---|---|
| llm | 82 | 1,559.6 | 19,019.2 | 77.9 % |
| retriever | 40 | 337.2 | 8,430.1 | 16.8 % |
| embedding | 6 | 266.1 | 44,341.7 | 13.3 % |
| herramienta | 27 | 20.6 | 762.8 | 1.0 % |
| guardrail | 51 | 0.2 | 3.1 | 0.0 % |

Los `llm` pueden sumar más que el reloj: el indexador llama al modelo en paralelo (8 hilos).
