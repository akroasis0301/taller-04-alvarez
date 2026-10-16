# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la matriz TF-IDF de la Parte 1 y evaluación comparada.

- Carga de entrada/T1/resultados.json las métricas y rankings TF-IDF (línea base: no se recalculan).
- Reconstruye la matriz TF-IDF (mismo corpus, TfidfVectorizer por defecto) porque TruncatedSVD
  necesita la matriz explícita; la convención de ajuste se verifica contra la matriz de similitud
  guardada por T1 para garantizar que es LA MISMA matriz TF-IDF de la Parte 1.
- Proyecta a 4 dimensiones con TruncatedSVD(n_components=4, random_state=0), calcula similitud
  coseno en el espacio latente, repite la evaluación (Hit@1, Hit@3, MRR) y construye UNA tabla
  con los dos métodos, más el ranking por consulta de LSA (insumo de la Parte 3).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# 0) Corpus, consultas y juicios (datos del enunciado, idénticos a la Parte 1)
# ------------------------------------------------------------------
docs_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]
docs_textos = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
]
queries_ids = ["q1", "q2", "q3", "q4", "q5", "q6"]
queries_textos = [
    "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "¿Cómo se añaden fragmentos recuperados al prompt?",
    "¿Qué técnica entrena matrices de bajo rango?",
    "¿Cómo se compara la orientación de dos vectores de texto?",
    "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "¿Qué arquitectura combina autoatención con capas feed-forward?",
]
relevante = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}

# ------------------------------------------------------------------
# 1) Línea base TF-IDF: métricas y rankings CARGADOS de la subtarea T1 (no se recalculan)
# ------------------------------------------------------------------
t1 = json.loads((Path("entrada") / "T1" / "resultados.json").read_text(encoding="utf-8"))
metricas_nombres = ["Hit@1", "Hit@3", "MRR"]
metricas_tfidf = {"Hit@1": float(t1["hit_at_1"]),
                  "Hit@3": float(t1["hit_at_3"]),
                  "MRR": float(t1["mrr"])}
rankings_tfidf = {q: list(r) for q, r in t1["rankings_por_consulta"].items()}

# ------------------------------------------------------------------
# 2) Misma matriz TF-IDF que T1: se ajusta el vectorizador y se verifica contra
#    la matriz de similitud guardada por T1 (se elige la convención que coincide)
# ------------------------------------------------------------------
vec_docs = TfidfVectorizer().fit(docs_textos)                    # convención A: fit sobre documentos
vec_joint = TfidfVectorizer().fit(docs_textos + queries_textos)  # convención B: fit sobre docs + consultas
S_a = cosine_similarity(vec_docs.transform(queries_textos), vec_docs.transform(docs_textos))
S_b = cosine_similarity(vec_joint.transform(queries_textos), vec_joint.transform(docs_textos))

ms = t1["matriz_similitud"]
filas_t1, cols_t1 = list(ms["filas"]), list(ms["columnas"])
S_t1 = np.asarray(ms["valores"], dtype=float)[np.ix_([filas_t1.index(q) for q in queries_ids],
                                                     [cols_t1.index(d) for d in docs_ids])]
diff_a = float(np.max(np.abs(S_a - S_t1)))
diff_b = float(np.max(np.abs(S_b - S_t1)))
if diff_b < diff_a:
    vectorizer, convencion_tfidf, diff_t1 = vec_joint, "fit sobre documentos + consultas", diff_b
else:
    vectorizer, convencion_tfidf, diff_t1 = vec_docs, "fit sobre documentos", diff_a
print(f"Convención TF-IDF elegida: {convencion_tfidf} (diff máx vs matriz de similitud de T1 = {diff_t1:.2e})")

X_docs = vectorizer.transform(docs_textos)        # 10 x V  (misma matriz TF-IDF que la Parte 1)
X_queries = vectorizer.transform(queries_textos)  # 6 x V

# ------------------------------------------------------------------
# 3) LSA: TruncatedSVD a 4 dimensiones y similitud coseno en el espacio latente
# ------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
docs_lsa = svd.fit_transform(X_docs)     # 10 x 4
queries_lsa = svd.transform(X_queries)   # 6 x 4 (las consultas se proyectan con el MISMO SVD)
S_lsa = cosine_similarity(queries_lsa, docs_lsa)  # 6 x 10

# ------------------------------------------------------------------
# 4) Evaluación de LSA (misma mecánica que la Parte 1; empates: orden estable por id)
# ------------------------------------------------------------------
def evaluar_matriz(S):
    detalle = {}
    for i, qid in enumerate(queries_ids):
        orden = np.argsort(-S[i], kind="stable")
        ranking = [docs_ids[j] for j in orden]
        rank_rel = ranking.index(relevante[qid]) + 1
        detalle[qid] = {
            "consulta": queries_textos[i],
            "relevante_esperado": relevante[qid],
            "ranking": ranking,
            "scores_ordenados": [float(S[i, j]) for j in orden],
            "rank_del_relevante": int(rank_rel),
            "hit@1": int(rank_rel == 1),
            "hit@3": int(rank_rel <= 3),
            "reciprocal_rank": 1.0 / rank_rel,
        }
    metricas = {
        "Hit@1": float(np.mean([d["hit@1"] for d in detalle.values()])),
        "Hit@3": float(np.mean([d["hit@3"] for d in detalle.values()])),
        "MRR": float(np.mean([d["reciprocal_rank"] for d in detalle.values()])),
    }
    return metricas, detalle

metricas_lsa, detalle_lsa = evaluar_matriz(S_lsa)

# ------------------------------------------------------------------
# 5) Tabla única comparativa + figuras
# ------------------------------------------------------------------
tabla_metricas = [
    {"metodo": "TF-IDF (coseno, Parte 1)",
     "Hit@1": metricas_tfidf["Hit@1"], "Hit@3": metricas_tfidf["Hit@3"], "MRR": metricas_tfidf["MRR"]},
    {"metodo": "LSA (TruncatedSVD k=4, coseno)",
     "Hit@1": metricas_lsa["Hit@1"], "Hit@3": metricas_lsa["Hit@3"], "MRR": metricas_lsa["MRR"]},
]
tabla = pd.DataFrame(tabla_metricas)
tabla.to_csv("tabla_comparativa.csv", index=False, encoding="utf-8")

# Figura 1: barras comparativas de las tres métricas
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(metricas_nombres)); w = 0.35
b1 = ax.bar(x - w / 2, [metricas_tfidf[m] for m in metricas_nombres], w, label="TF-IDF", color="#4C72B0")
b2 = ax.bar(x + w / 2, [metricas_lsa[m] for m in metricas_nombres], w, label="LSA (k=4)", color="#DD8452")
for barras in (b1, b2):
    for b in barras:
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(metricas_nombres)
ax.set_ylim(0, 1.12); ax.set_ylabel("valor (6 consultas)")
ax.set_title("Recuperación: TF-IDF vs LSA (TruncatedSVD, 4 dimensiones)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("comparacion_metricas.png", dpi=120); plt.close(fig)

# Figura 2: mapa de calor de la similitud coseno LSA (consulta x documento)
vmin, vmax = float(S_lsa.min()), float(S_lsa.max())
fig, ax = plt.subplots(figsize=(8.5, 4.6))
im = ax.imshow(S_lsa, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(docs_ids))); ax.set_xticklabels(docs_ids, rotation=90)
ax.set_yticks(range(len(queries_ids))); ax.set_yticklabels(queries_ids)
for i in range(len(queries_ids)):
    for j in range(len(docs_ids)):
        es_rel = docs_ids[j] == relevante[queries_ids[i]]
        ax.text(j, i, f"{S_lsa[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if S_lsa[i, j] < (vmin + vmax) / 2 else "black",
                fontweight="bold" if es_rel else "normal")
    j_rel = docs_ids.index(relevante[queries_ids[i]])
    ax.add_patch(Rectangle((j_rel - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
fig.colorbar(im, ax=ax, label="similitud coseno (LSA)")
ax.set_title("Similitud coseno en el espacio latente LSA (rojo = documento relevante)")
fig.tight_layout(); fig.savefig("mapa_calor_lsa.png", dpi=120); plt.close(fig)

# ------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2_lsa_truncated_svd_k4",
    "metodo": "TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF de la Parte 1 + similitud coseno en el espacio latente",
    "convencion_tfidf": convencion_tfidf,
    "verificacion_vs_T1_diff_max": diff_t1,
    "hit_at_1": metricas_lsa["Hit@1"],
    "hit_at_3": metricas_lsa["Hit@3"],
    "mrr": metricas_lsa["MRR"],
    "tabla_metricas": tabla_metricas,
    "deltas_lsa_vs_tfidf": {m: metricas_lsa[m] - metricas_tfidf[m] for m in metricas_nombres},
    "rankings_por_consulta": {q: detalle_lsa[q]["ranking"] for q in queries_ids},
    "rankings_por_consulta_tfidf_de_T1": rankings_tfidf,
    "detalle_por_consulta": detalle_lsa,
    "matriz_similitud_lsa": {
        "filas": queries_ids,
        "columnas": docs_ids,
        "valores": [[float(v) for v in fila] for fila in S_lsa],
    },
    "vectores_latentes": {
        "documentos": {docs_ids[i]: [float(v) for v in docs_lsa[i]] for i in range(len(docs_ids))},
        "consultas": {queries_ids[i]: [float(v) for v in queries_lsa[i]] for i in range(len(queries_ids))},
    },
    "varianza_explicada_por_componente": [float(v) for v in svd.explained_variance_ratio_],
    "varianza_explicada_total": float(np.sum(svd.explained_variance_ratio_)),
    "num_consultas": len(queries_ids),
    "num_documentos": len(docs_ids),
    "dimension_latente": 4,
    "supuestos": [
        "Las métricas y rankings TF-IDF de la tabla se cargan de entrada/T1/resultados.json (subtarea previa) sin recalcularlos.",
        "La matriz TF-IDF se reconstruye con el mismo corpus y TfidfVectorizer por defecto porque TruncatedSVD necesita la matriz explícita; la convención de ajuste (solo documentos, o documentos+consultas) se elige comprobando cuál reproduce la matriz de similitud guardada por T1.",
        "Las consultas se proyectan al espacio latente con el mismo TruncatedSVD ajustado sobre los documentos (svd.transform).",
        "La similitud coseno en el espacio latente se calcula sobre los vectores latentes sin normalización adicional (cosine_similarity normaliza internamente).",
        "Los empates de similitud se resuelven con orden estable por id de documento (d01..d10).",
    ],
}
Path("resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")

# ------------------------------------------------------------------
# 7) Resumen
# ------------------------------------------------------------------
print("\n=== T2: TF-IDF vs LSA (TruncatedSVD k=4) — 6 consultas ===")
print(tabla.to_string(index=False))
print(f"\nVarianza explicada por las 4 componentes latentes: {resultados['varianza_explicada_total']:.4f}")
print("\nRanking LSA por consulta ([...] = documento relevante):")
for q in queries_ids:
    rel = relevante[q]
    cadena = " > ".join(f"[{d}]" if d == rel else d for d in detalle_lsa[q]["ranking"])
    print(f"  {q}: {cadena}  (rank del relevante = {detalle_lsa[q]['rank_del_relevante']})")
print("\nArchivos escritos: resultados.json, tabla_comparativa.csv, comparacion_metricas.png, mapa_calor_lsa.png")
