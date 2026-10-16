#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Semántica latente (LSA) sobre la matriz TF-IDF de la Parte 1 y tabla comparativa.

- Misma matriz TF-IDF (TfidfVectorizer con parámetros por defecto) y mismos juicios
  de relevancia de la Parte 1 (q1→d04, q2→d03, q3→d08, q4→d05, q5→d09, q6→d02).
- Proyección a 4 dimensiones: TruncatedSVD(n_components=4, random_state=0) (LSA).
  Se ajusta SOLO con los 10 documentos; las consultas se proyectan con las mismas
  componentes (folding-in estándar de LSA).
- Ranking por similitud coseno en el espacio latente y evaluación Hit@1, Hit@3, MRR.
- Una única tabla comparativa TF-IDF vs LSA (CSV + PNG) y rankings LSA por consulta.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# 1) Datos del enunciado: 10 documentos, 6 consultas, juicio relevante
# ------------------------------------------------------------------
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

doc_ids = sorted(documentos)          # d01..d10
q_ids = sorted(consultas)             # q1..q6
doc_texts = [documentos[d] for d in doc_ids]
q_texts = [consultas[q][0] for q in q_ids]
relevante = {q: consultas[q][1] for q in q_ids}

# ------------------------------------------------------------------
# 2) Misma matriz TF-IDF que la Parte 1 (parámetros por defecto)
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)
X_queries = vectorizer.transform(q_texts)

S_tfidf = cosine_similarity(X_queries, X_docs)  # (6, 10)

# ------------------------------------------------------------------
# 3) LSA: TruncatedSVD(n_components=4, random_state=0)
# ------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)
X_queries_lsa = svd.transform(X_queries)

S_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)  # (6, 10)

# ------------------------------------------------------------------
# 4) Rankings por similitud coseno (empates: orden estable por id de documento)
# ------------------------------------------------------------------
def rankear(S):
    return {
        q: [doc_ids[j] for j in np.argsort(-S[i], kind="stable")]
        for i, q in enumerate(q_ids)
    }

def evaluar(rankings):
    por_q, h1, h3, rr = {}, [], [], []
    for q in q_ids:
        r = rankings[q].index(relevante[q]) + 1
        h1.append(float(r == 1))
        h3.append(float(r <= 3))
        rr.append(1.0 / r)
        por_q[q] = {
            "consulta": consultas[q][0],
            "relevante": relevante[q],
            "rank_del_relevante": int(r),
            "hit@1": int(r == 1),
            "hit@3": int(r <= 3),
            "reciprocal_rank": 1.0 / r,
        }
    metricas = {
        "Hit@1": float(np.mean(h1)),
        "Hit@3": float(np.mean(h3)),
        "MRR": float(np.mean(rr)),
    }
    return metricas, por_q

rank_tfidf = rankear(S_tfidf)
rank_lsa = rankear(S_lsa)
m_tfidf, por_q_tfidf = evaluar(rank_tfidf)
m_lsa, por_q_lsa = evaluar(rank_lsa)

# ------------------------------------------------------------------
# 5) Tabla única comparativa (2 métodos × Hit@1, Hit@3, MRR)
# ------------------------------------------------------------------
tabla = pd.DataFrame(
    [m_tfidf, m_lsa],
    index=["TF-IDF", "LSA (TruncatedSVD, k=4)"],
    columns=["Hit@1", "Hit@3", "MRR"],
)
tabla.to_csv("tabla_comparativa.csv", encoding="utf-8")

fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.axis("off")
celdas = [[f"{v:.4f}" for v in fila] for fila in tabla.values]
tbl = ax.table(
    cellText=celdas,
    colLabels=list(tabla.columns),
    rowLabels=list(tabla.index),
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.8)
ax.set_title("Recuperación sobre 6 consultas: TF-IDF vs LSA (k=4)", fontsize=12, pad=14)
plt.savefig("tabla_comparativa.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# 6) Verificación opcional contra la línea base de T1 (entrada/T1)
# ------------------------------------------------------------------
t1_path = Path("entrada") / "T1" / "resultados.json"
verif = {"archivo": str(t1_path), "encontrado": t1_path.exists(), "coincide": None}
if t1_path.exists():
    try:
        with t1_path.open(encoding="utf-8") as fh:
            t1 = json.load(fh)
        m1 = t1.get("metricas", {})
        verif["metricas_T1"] = {k: m1.get(k) for k in ("Hit@1", "Hit@3", "MRR")}
        verif["coincide"] = all(
            np.isclose(m_tfidf[k], float(m1[k]), rtol=0, atol=1e-12)
            for k in ("Hit@1", "Hit@3", "MRR")
            if k in m1
        )
    except Exception as exc:
        verif["error_lectura"] = repr(exc)

# ------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2_semantica_latente_lsa_comparativa",
    "metodo_tfidf": "TfidfVectorizer (parámetros por defecto) + similitud coseno",
    "metodo_lsa": "TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF + similitud coseno en el espacio latente",
    "n_documentos": len(doc_ids),
    "n_consultas": len(q_ids),
    "n_componentes_lsa": 4,
    "random_state": 0,
    "varianza_explicada_por_componente": svd.explained_variance_ratio_.tolist(),
    "varianza_explicada_total": float(svd.explained_variance_ratio_.sum()),
    "metricas_tfidf": m_tfidf,
    "metricas_lsa": m_lsa,
    "diferencia_lsa_menos_tfidf": {k: m_lsa[k] - m_tfidf[k] for k in ("Hit@1", "Hit@3", "MRR")},
    "tabla_comparativa": [
        {"metodo": "TF-IDF", **m_tfidf},
        {"metodo": "LSA (TruncatedSVD, k=4)", **m_lsa},
    ],
    "rankings_tfidf": rank_tfidf,
    "rankings_lsa": rank_lsa,
    "por_consulta_tfidf": por_q_tfidf,
    "por_consulta_lsa": por_q_lsa,
    "verificacion_con_T1": verif,
    "archivos_generados": ["tabla_comparativa.csv", "tabla_comparativa.png", "resultados.json"],
}
with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 8) Resumen
# ------------------------------------------------------------------
print("T2 — LSA (TruncatedSVD k=4, random_state=0) vs TF-IDF (línea base Parte 1)")
print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"Varianza explicada por las 4 componentes LSA: {svd.explained_variance_ratio_.sum():.4f}")
print("\nRankings LSA por consulta (relevante entre corchetes):")
for q in q_ids:
    rk = rank_lsa[q]
    marcado = [f"[{d}]" if d == relevante[q] else d for d in rk]
    print(f"  {q} (rel={relevante[q]}, rank={por_q_lsa[q]['rank_del_relevante']}): {' > '.join(marcado)}")
print("\nArchivos generados: resultados.json, tabla_comparativa.csv, tabla_comparativa.png")
