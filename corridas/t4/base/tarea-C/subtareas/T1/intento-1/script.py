#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica: TF-IDF (TfidfVectorizer, parámetros por defecto)
       + similitud coseno, ranking completo de 10 documentos por consulta
       y evaluación Hit@1, Hit@3 y MRR sobre las 6 consultas.

Salidas:
  - resultados.json               (contrato: rankings completos y métricas)
  - t1_similitud_coseno.png       (mapa de calor de similitudes)
  - t1_rank_relevante.png         (rank del documento relevante por consulta)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------
# 1. Corpus: 10 documentos y 6 consultas con juicios de relevancia
# ----------------------------------------------------------------------
DOC_IDS = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]

DOCS = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; "
    "la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con "
    "conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade "
    "al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la "
    "longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno "
    "compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el "
    "resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal "
    "entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos "
    "originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran "
    "la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
]

QUERIES = {
    "q1": {"texto": "¿Qué función de ranking léxica pondera la frecuencia de términos?",
           "relevante": "d04"},
    "q2": {"texto": "¿Cómo se añaden fragmentos recuperados al prompt?",
           "relevante": "d03"},
    "q3": {"texto": "¿Qué técnica entrena matrices de bajo rango?",
           "relevante": "d08"},
    "q4": {"texto": "¿Cómo se compara la orientación de dos vectores de texto?",
           "relevante": "d05"},
    "q5": {"texto": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
           "relevante": "d09"},
    "q6": {"texto": "¿Qué arquitectura combina autoatención con capas feed-forward?",
           "relevante": "d02"},
}
Q_IDS = ["q1", "q2", "q3", "q4", "q5", "q6"]

# ----------------------------------------------------------------------
# 2. Vectorización TF-IDF con parámetros por defecto.
#    El vectorizador se ajusta SOLO con los documentos; las consultas
#    únicamente se transforman (no se reajusta nada con las consultas).
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto
X_docs = vectorizer.fit_transform(DOCS)                                  # (10, V)
X_queries = vectorizer.transform([QUERIES[q]["texto"] for q in Q_IDS])   # (6, V)
vocab_size = len(vectorizer.vocabulary_)

# ----------------------------------------------------------------------
# 3. Similitud coseno consulta-documento
# ----------------------------------------------------------------------
S = cosine_similarity(X_queries, X_docs)   # forma (6, 10)

# ----------------------------------------------------------------------
# 4. Ranking completo de los 10 documentos por consulta
#    (orden descendente por similitud; empates resueltos por orden d01..d10)
# ----------------------------------------------------------------------
rankings = {}
for i, qid in enumerate(Q_IDS):
    order = np.argsort(-S[i], kind="stable")
    rankings[qid] = [DOC_IDS[j] for j in order]

# ----------------------------------------------------------------------
# 5. Métricas: Hit@1, Hit@3 y MRR sobre las 6 consultas
# ----------------------------------------------------------------------
per_query = {}
hit1_list, hit3_list, rr_list = [], [], []
for i, qid in enumerate(Q_IDS):
    rel = QUERIES[qid]["relevante"]
    rank = rankings[qid].index(rel) + 1
    h1 = int(rank == 1)
    h3 = int(rank <= 3)
    rr = 1.0 / rank
    per_query[qid] = {
        "consulta": QUERIES[qid]["texto"],
        "relevante": rel,
        "rank_del_relevante": int(rank),
        "hit@1": h1,
        "hit@3": h3,
        "reciprocal_rank": float(rr),
        "ranking_completo": rankings[qid],
        "similitudes_por_documento": {DOC_IDS[j]: float(S[i, j]) for j in range(len(DOC_IDS))},
    }
    hit1_list.append(h1)
    hit3_list.append(h3)
    rr_list.append(rr)

hit_at_1 = float(np.mean(hit1_list))
hit_at_3 = float(np.mean(hit3_list))
mrr = float(np.mean(rr_list))

# ----------------------------------------------------------------------
# 6. Escritura del contrato resultados.json
# ----------------------------------------------------------------------
params = {}
for k, v in vectorizer.get_params().items():
    if v is None or isinstance(v, (int, float, str, bool, tuple, list)):
        params[k] = v
    else:
        params[k] = str(v)

results = {
    "subtarea": "T1_linea_base_lexica_tfidf",
    "metodo": "TfidfVectorizer (parámetros por defecto) + similitud coseno",
    "n_documentos": len(DOC_IDS),
    "n_consultas": len(Q_IDS),
    "tam_vocabulario": int(vocab_size),
    "vectorizer_params": params,
    "metricas": {
        "Hit@1": hit_at_1,
        "Hit@3": hit_at_3,
        "MRR": mrr,
    },
    "por_consulta": per_query,
    "rankings": rankings,
    "similitud_coseno": {
        qid: {DOC_IDS[j]: float(S[i, j]) for j in range(len(DOC_IDS))}
        for i, qid in enumerate(Q_IDS)
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Figuras (PNG, dpi=120)
# ----------------------------------------------------------------------
# 7.1 Mapa de calor de similitudes, marcando el documento relevante
fig, ax = plt.subplots(figsize=(9.5, 4.6))
im = ax.imshow(S, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(S.max(), 0.1))
ax.set_xticks(range(len(DOC_IDS)))
ax.set_xticklabels(DOC_IDS, fontsize=8)
ax.set_yticks(range(len(Q_IDS)))
ax.set_yticklabels(Q_IDS, fontsize=9)
for i in range(len(Q_IDS)):
    for j in range(len(DOC_IDS)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                color="white", fontsize=7)
    j_rel = DOC_IDS.index(QUERIES[Q_IDS[i]]["relevante"])
    ax.add_patch(Rectangle((j_rel - 0.5, i - 0.5), 1, 1, fill=False,
                           edgecolor="red", linewidth=2))
ax.set_title("T1: similitud coseno consulta-documento (TF-IDF por defecto)\n"
             "cuadro rojo = documento relevante", fontsize=10)
fig.colorbar(im, ax=ax, label="similitud coseno")
fig.tight_layout()
fig.savefig("t1_similitud_coseno.png", dpi=120)
plt.close(fig)

# 7.2 Rank del documento relevante por consulta
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ranks = [per_query[q]["rank_del_relevante"] for q in Q_IDS]
colors = ["#2a9d8f" if r <= 3 else "#e76f51" for r in ranks]
ax.bar(Q_IDS, ranks, color=colors)
ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(3, color="gray", linestyle="--", linewidth=0.8)
ax.set_ylim(0, 10.5)
ax.set_ylabel("rank del documento relevante")
ax.set_title(f"T1: rank del relevante por consulta\n"
             f"Hit@1={hit_at_1:.3f}  Hit@3={hit_at_3:.3f}  MRR={mrr:.3f}",
             fontsize=10)
for qi, r in zip(Q_IDS, ranks):
    ax.text(qi, r + 0.15, str(r), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("t1_rank_relevante.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 8. Resumen por consola
# ----------------------------------------------------------------------
print("=" * 64)
print("T1 — Línea base léxica: TF-IDF + similitud coseno")
print("=" * 64)
print(f"Vocabulario TF-IDF (ajustado solo con documentos): {vocab_size} términos")
print("-" * 64)
for qid in Q_IDS:
    pq = per_query[qid]
    print(f"{qid} (rel={pq['relevante']}): rank={pq['rank_del_relevante']:>2}  "
          f"Hit@1={pq['hit@1']}  Hit@3={pq['hit@3']}  RR={pq['reciprocal_rank']:.4f}")
    print(f"    ranking: {' > '.join(pq['ranking_completo'])}")
print("-" * 64)
print(f"Hit@1 = {hit_at_1:.4f}   Hit@3 = {hit_at_3:.4f}   MRR = {mrr:.4f}")
print("Archivos escritos: resultados.json, t1_similitud_coseno.png, t1_rank_relevante.png")
