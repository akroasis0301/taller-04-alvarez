# Parte 3 — Guardrails

### Guardrails como clasificadores

| validadores y banco | casos | correctos | falsos negativos | falsos positivos | otra acción | errores |
|---|---|---|---|---|---|---|
| nuestros, banco (26) | 26 | 26 | 0/16 | 0/10 | 0 | — |
| ejemplo del kit, banco (26) | 26 | 15 | 8/16 | 3/10 | 0 | A02, A04, A05, A06, L01, L03, S02, S03, B01, B02, B03 |
| nuestros, fuera del banco (12) | 12 | 7 | 5/7 | 0/5 | 0 | F01, F02, F03, F05, F06 |

### Guardrails en el agente (base): costo y decisiones en el golden

| guardrail | spans | latencia media (ms) | máx. (ms) | decisiones distintas de ok |
|---|---|---|---|---|
| entrada_inyeccion | 6 | 1.67 | 2 | ok: 5; redactada: 1 |
| entrada_pii | 6 | 0.67 | 1 | ok: 5; redactada: 1 |
| guarda_ast | 23 | 4 | 8 | — |
| salida_pii | 9 | 2.89 | 4 | ok |
| salida_secretos | 9 | 1.22 | 2 | ok |

### Guardrails en el agente (mejora): costo y decisiones en el golden

| guardrail | spans | latencia media (ms) | máx. (ms) | decisiones distintas de ok |
|---|---|---|---|---|
| entrada_inyeccion | 6 | 1.83 | 3 | ok: 5; redactada: 1 |
| entrada_pii | 6 | 1.17 | 2 | ok: 5; redactada: 1 |
| guarda_ast | 21 | 4.38 | 6 | — |
| salida_pii | 9 | 3.33 | 5 | ok |
| salida_secretos | 9 | 1.78 | 3 | ok |
