# Parte 4 — Datos para las alertas

### Mediciones para los umbrales de las alertas (Parte 4.b)

| corrida | corridas | latencia p50 (s) | latencia p95 (s) | latencia máx. (s) | tokens p50 por corrida | tokens máx. por corrida | resolución (1 corrida) |
|---|---|---|---|---|---|---|---|
| base | 6 | 469 | 1534 | 1759 | 104,264 | 334,343 | 16.7 % |

Con n corridas, cada una vale 100/n % de cualquier tasa: un umbral por debajo de esa resolución salta con el primer caso.
