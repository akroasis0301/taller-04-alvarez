## Objetivo

Comparar un recuperador léxico (TF-IDF con similitud coseno) contra una proyección semántica latente (LSA mediante TruncatedSVD con 4 componentes) sobre un corpus de 10 documentos y 6 consultas, cada una con un documento relevante. Se miden Hit@1, Hit@3 y MRR, se identifican las consultas donde cada método falla (relevante fuera del top 3) y se explica el porqué a partir de los términos que comparten la consulta y su documento relevante.

## Método

**Datos.** 10 documentos (d01–d10) y 6 consultas (q1–q6) con un juicio de relevancia positivo cada una, según el enunciado.

**Línea base léxica.** `TfidfVectorizer` de scikit-learn con parámetros por defecto, ajustado solo sobre los 10 documentos; las consultas se transforman con ese vocabulario (términos ajenos se ignoran). Similitud coseno consulta-documento y ordenamiento de los 10 documentos por consulta; empates resueltos a favor del documento de menor índice.

**Semántica latente.** `TruncatedSVD(n_components=4, random_state=0)` sobre la misma matriz TF-IDF (verificada idéntica a la de la Parte 1, diferencia máxima 1.1102e-16); similitud coseno en el espacio latente y misma evaluación.

**Métricas.** Hit@k = proporción de consultas cuyo relevante queda entre las k primeras posiciones; MRR = media de 1/rank del relevante. Análisis de fallos: relevante en posición ≥ 4; términos compartidos según el vocabulario TF-IDF (tokenizador por defecto, sin stopwords ni stemming).

## Resultados

**Métricas agregadas sobre las 6 consultas:**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (coseno) | 0.8333 | 0.8333 | 0.8611 |
| LSA (k=4, coseno) | 0.6667 | 0.8333 | 0.75 |
| Δ (LSA − TF-IDF) | −0.1667 | 0.0 | −0.1111 |

![Comparación de Hit@1, Hit@3 y MRR entre TF-IDF y LSA](figuras/T2_comparacion_metricas.png)

**Posición del documento relevante por consulta:**

| Consulta | Relevante | Rank TF-IDF | Rank LSA |
|---|---|---|---|
| q1 | d04 | 1 | 1 |
| q2 | d03 | 1 | 1 |
| q3 | d08 | 1 | 1 |
| q4 | d05 | 1 | 3 |
| q5 | d09 | 6 | 6 |
| q6 | d02 | 1 | 1 |

Fallos (relevante fuera del top 3): TF-IDF falla solo en q5; LSA también falla solo en q5. LSA además degrada q4 (d05 pasa del rank 1 al 3, sin salir del top 3): en el espacio latente d01 (0.9220) y d10 (0.9206) superan a d05 (0.8984), mientras que en TF-IDF d05 encabezaba con 0.4569 frente a 0.2475 de d01.

**Caso q5 (único fallo de ambos métodos).** La consulta comparte con d09 un único término del vocabulario, la palabra funcional "la" (peso TF-IDF 0.2440 en la consulta, 0.3238 en d09): ningún término de contenido. En TF-IDF d09 alcanza solo 0.0790 y el top 3 lo ocupan d03, d06 y d04 (d03 comparte "al", "la", "modelo"); en LSA d09 queda en rank 6 con 0.4999, detrás de d04 (0.9427), d01 (0.8066) y d03 (0.6141).

![Términos compartidos entre cada consulta y su documento relevante](figuras/T3_T3_terminos_compartidos.png)

## Discusión

**Por qué fallan (Parte 3).** Las consultas resueltas comparten entre 2 y 6 términos de contenido con su relevante (q1 comparte "frecuencia", "función", "léxica", "pondera", "ranking", "términos"; q6 comparte "autoatención", "capas", "feed", "forward"), lo que da a TF-IDF una señal léxica directa. En q5 esa señal no existe: "determinista" y "siguiente palabra" no aparecen en d09, que habla de "temperatura", "logits", "softmax" y "probabilidad"; el único término común es "la". LSA tampoco lo resuelve porque sus 4 dimensiones latentes se aprenden de las co-ocurrencias de solo 10 documentos: no hay evidencia en el corpus que asocie "determinista" con "temperatura". La proyección además comprime los márgenes (en q1, d04 pasa de 0.6891 a 0.9542 pero d01 sube de 0.2410 a 0.9262), lo que explica la pérdida de Hit@1 (−0.1667) y de MRR (−0.1111) sin ganancia en Hit@3.

**Recuperador denso y su costo (Parte 4).** Un recuperador denso entrenado colocaría cerca del mismo punto del espacio vectorial "más determinista al elegir la siguiente palabra" y "la temperatura reescala los logits… concentran la probabilidad", porque aprende en grandes corpus que ambas frases describen la misma relación causa-efecto sobre la selección del siguiente token: captura sinonimia y paráfrasis, no coincidencia de términos. Eso resolvería justamente q5, el fallo común de TF-IDF y LSA aquí. El costo frente a TF-IDF es alto: exige datos y cómputo de entrenamiento, o depender de un modelo preentrenado con posible ajuste fino al dominio; codificar consultas y documentos requiere pasar cada texto por una red neuronal (GPU en la práctica), frente a un índice invertido disperso; el índice denso almacena vectores continuos y suele necesitar búsqueda aproximada (ANN); y el resultado es menos interpretable y sensible a la versión del modelo. TF-IDF, en cambio, se ajusta en una pasada sobre el corpus, no requiere datos etiquetados y cada score es trazable a términos compartidos, como muestra el análisis de q5.
