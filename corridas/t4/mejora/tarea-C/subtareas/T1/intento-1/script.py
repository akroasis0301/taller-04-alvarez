import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Datos literales del enunciado ----------
documentos = {
    "d01": "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "d02": "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "d03": "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "d04": "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "d05": "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "d06": "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "d07": "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "d08": "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "d09": "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "d10": "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
}

consultas = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = list(documentos.keys())      # d01..d10 en orden
query_ids = list(consultas.keys())     # q1..q6 en orden
doc_texts = [documentos[d] for d in doc_ids]
query_texts = [consultas[q][0] for q in query_ids]
relevantes = {q: consultas[q][1] for q in query_ids}

assert set(relevantes.values()) <= set(doc_ids), "relevante fuera del corpus"
assert len(doc_ids) == 10 and len(query_ids) == 6

# ---------- Vectorización TF-IDF (parámetros por defecto) ----------
vectorizer = TfidfVectorizer()                   # defaults de scikit-learn
X_docs = vectorizer.fit_transform(doc_texts)     # fit SOLO sobre el corpus
X_queries = vectorizer.transform(query_texts)    # las consultas solo se transforman

S = cosine_similarity(X_queries, X_docs)         # forma (6, 10)
assert S.shape == (6, 10)

# ---------- Ranking por consulta y métricas ----------
rankings = {}
por_consulta = {}
hits1, hits3, recips = [], [], []
for i, qid in enumerate(query_ids):
    orden = np.argsort(-S[i], kind="stable")     # score descendente; empates por orden d01..d10
    ranked = [doc_ids[j] for j in orden]
    rank = ranked.index(relevantes[qid]) + 1
    h1 = int(rank == 1)
    h3 = int(rank <= 3)
    rr = 1.0 / rank
    hits1.append(h1)
    hits3.append(h3)
    recips.append(rr)
    rankings[qid] = ranked
    por_consulta[qid] = {
        "consulta": query_texts[i],
        "relevante_esperado": relevantes[qid],
        "ranking": ranked,
        "scores_ordenados": [float(S[i, j]) for j in orden],
        "rank_del_relevante": int(rank),
        "hit@1": h1,
        "hit@3": h3,
        "reciprocal_rank": rr,
    }

hit_at_1 = float(np.mean(hits1))
hit_at_3 = float(np.mean(hits3))
mrr = float(np.mean(recips))

resultados = {
    "subtarea": "T1_linea_base_tfidf",
    "metodo": "TfidfVectorizer (scikit-learn, parámetros por defecto) + similitud coseno",
    "hit_at_1": hit_at_1,
    "hit_at_3": hit_at_3,
    "mrr": mrr,
    "num_consultas": len(query_ids),
    "num_documentos": len(doc_ids),
    "rankings_por_consulta": rankings,
    "detalle_por_consulta": por_consulta,
    "matriz_similitud": {
        "filas": query_ids,
        "columnas": doc_ids,
        "valores": [[float(S[i, j]) for j in range(len(doc_ids))] for i in range(len(query_ids))],
    },
    "supuestos": [
        "El TfidfVectorizer se ajusta (fit) únicamente sobre los 10 documentos; las consultas se transforman con ese vocabulario y los términos de consulta ajenos al vocabulario se ignoran.",
        "Los empates de similitud se resuelven con orden estable a favor del documento de menor índice (d01 antes que d02, etc.).",
        "Hit@k = proporción de consultas cuyo documento relevante aparece dentro de las k primeras posiciones; MRR = media de 1/rank del documento relevante.",
    ],
}

Path("resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- Figura: mapa de calor de similitudes ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(S, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(doc_ids)))
ax.set_xticklabels(doc_ids)
ax.set_yticks(range(len(query_ids)))
ax.set_yticklabels(query_ids)
for i in range(len(query_ids)):
    for j in range(len(doc_ids)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
fig.colorbar(im, ax=ax, label="similitud coseno")
ax.set_title("T1: similitud coseno consulta-documento (TF-IDF)")
fig.tight_layout()
fig.savefig("t1_similitud_coseno.png", dpi=120)
plt.close(fig)

# ---------- Resumen ----------
print("T1 - Línea base TF-IDF (TfidfVectorizer por defecto + similitud coseno)")
for qid in query_ids:
    d = por_consulta[qid]
    print(f"{qid}: relevante={d['relevante_esperado']} rank={d['rank_del_relevante']} "
          f"top3={d['ranking'][:3]} hit@1={d['hit@1']} hit@3={d['hit@3']} RR={d['reciprocal_rank']:.4f}")
print(f"Agregado: Hit@1={hit_at_1:.4f}  Hit@3={hit_at_3:.4f}  MRR={mrr:.4f}")
print("Salidas: resultados.json, t1_similitud_coseno.png")
