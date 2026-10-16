import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ----------------------------------------------------------------------
# Datos del enunciado (corpus de 10 documentos y 6 consultas con su relevante)
# ----------------------------------------------------------------------
DOCS = {
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
QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}
doc_ids = [f"d{i:02d}" for i in range(1, 11)]
q_ids = [f"q{i}" for i in range(1, 7)]

# ----------------------------------------------------------------------
# Vectorización TF-IDF (misma convención que T1/T2: fit sobre los 10 documentos,
# parámetros por defecto). Se reconstruye SOLO para obtener el vocabulario y los
# pesos de los términos; los rankings se cargan de las subtareas previas.
# ----------------------------------------------------------------------
vec = TfidfVectorizer()
X_docs = vec.fit_transform([DOCS[d] for d in doc_ids]).toarray()
X_q = vec.transform([QUERIES[q][0] for q in q_ids]).toarray()
vocab = set(vec.get_feature_names_out())
analyzer = vec.build_analyzer()


def ranking_desde_scores(scores):
    return [doc_ids[j] for j in np.argsort(-scores, kind="stable")]


def rankings_tfidf():
    sims = cosine_similarity(X_q, X_docs)
    return {q: ranking_desde_scores(sims[i]) for i, q in enumerate(q_ids)}


def rankings_lsa():
    svd = TruncatedSVD(n_components=4, random_state=0)
    Ld = svd.fit_transform(X_docs)
    Lq = svd.transform(X_q)
    sims = cosine_similarity(Lq, Ld)
    return {q: ranking_desde_scores(sims[i]) for i, q in enumerate(q_ids)}


def cargar_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


fuente = {}
try:
    rank_tfidf = cargar_json("entrada/T1/resultados.json")["rankings_por_consulta"]
    fuente["tfidf"] = "entrada/T1/resultados.json"
except (OSError, KeyError, ValueError):
    rank_tfidf = rankings_tfidf()
    fuente["tfidf"] = "recalculado (T1 no disponible)"

try:
    rank_lsa = cargar_json("entrada/T2/resultados.json")["rankings_por_consulta"]
    fuente["lsa"] = "entrada/T2/resultados.json"
except (OSError, KeyError, ValueError):
    rank_lsa = rankings_lsa()
    fuente["lsa"] = "recalculado (T2 no disponible)"

# Verificación blanda: el TF-IDF recalculado coincide con el ranking cargado de T1
rank_tfidf_check = rankings_tfidf()
coincide_t1 = all(rank_tfidf_check[q] == rank_tfidf[q] for q in q_ids)

# ----------------------------------------------------------------------
# Fallos: relevante fuera del top 3 (posición >= 4)
# ----------------------------------------------------------------------
def rank_de(ranking, doc_id):
    return ranking.index(doc_id) + 1


ranks = {q: {"tfidf": rank_de(rank_tfidf[q], QUERIES[q][1]),
             "lsa": rank_de(rank_lsa[q], QUERIES[q][1])} for q in q_ids}

FALLO_MIN_RANK = 4
fallos = {m: [q for q in q_ids if ranks[q][m] >= FALLO_MIN_RANK] for m in ("tfidf", "lsa")}

# ----------------------------------------------------------------------
# Términos compartidos consulta <-> documento relevante (vocabulario TF-IDF)
# ----------------------------------------------------------------------
FUNCIONALES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a",
    "y", "o", "u", "que", "qué", "se", "su", "sus", "como", "cómo", "más", "mas",
    "para", "por", "con", "sin", "entre", "según", "ante", "antes", "cada", "en",
    "es", "son", "sea", "ser", "sido", "está", "están", "hace", "hacer",
    "cuál", "cuáles", "dónde", "cuándo", "quién",
}


def terminos_de(texto):
    return set(t for t in analyzer(texto) if t in vocab)


terminos_compartidos = {}
for q in q_ids:
    texto_q, rel = QUERIES[q]
    shared = sorted(terminos_de(texto_q) & terminos_de(DOCS[rel]))
    contenido = [t for t in shared if t not in FUNCIONALES]
    i_d, i_q = doc_ids.index(rel), q_ids.index(q)
    pesos_doc = {t: float(X_docs[i_d, vec.vocabulary_[t]]) for t in shared}
    pesos_q = {t: float(X_q[i_q, vec.vocabulary_[t]]) for t in shared}
    terminos_compartidos[q] = {
        "consulta": texto_q,
        "relevante": rel,
        "terminos_compartidos": shared,
        "n_terminos_compartidos": len(shared),
        "terminos_de_contenido": contenido,
        "n_terminos_de_contenido": len(contenido),
        "peso_tfidf_en_relevante": pesos_doc,
        "peso_tfidf_en_consulta": pesos_q,
    }

# ----------------------------------------------------------------------
# Detalle de los fallos: con qué términos compiten los documentos del top 3
# ----------------------------------------------------------------------
detalle_fallos = {}
for m in ("tfidf", "lsa"):
    rankings = rank_tfidf if m == "tfidf" else rank_lsa
    det = {}
    for q in fallos[m]:
        texto_q, rel = QUERIES[q]
        top3 = rankings[q][:3]
        det[q] = {
            "rank_relevante": ranks[q][m],
            "relevante": rel,
            "top3_recuperado": top3,
            "terminos_compartidos_con_relevante": terminos_compartidos[q]["terminos_compartidos"],
            "terminos_compartidos_con_top3": {
                d: sorted(terminos_de(texto_q) & terminos_de(DOCS[d])) for d in top3
            },
        }
    detalle_fallos[m] = det

# Consultas degradadas (rank 2-3) que NO cuentan como fallo
degradados = {m: {q: ranks[q][m] for q in q_ids
                  if 1 < ranks[q][m] <= 3 and q not in fallos[m]}
              for m in ("tfidf", "lsa")}

# Verificación de métricas agregadas reconstruidas desde los ranks
hit1 = {m: float(np.mean([ranks[q][m] == 1 for q in q_ids])) for m in ("tfidf", "lsa")}
hit3 = {m: float(np.mean([ranks[q][m] <= 3 for q in q_ids])) for m in ("tfidf", "lsa")}

# ----------------------------------------------------------------------
# Figura: solapamiento consulta-relevante y marcas de fallo
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(q_ids))
tot = [terminos_compartidos[q]["n_terminos_compartidos"] for q in q_ids]
con = [terminos_compartidos[q]["n_terminos_de_contenido"] for q in q_ids]
ax.bar(x - 0.2, tot, width=0.4, color="#4C72B0", label="términos compartidos (todos)")
ax.bar(x + 0.2, con, width=0.4, color="#DD8452", label="términos de contenido")
primero_t = primero_l = True
for i, q in enumerate(q_ids):
    if q in fallos["tfidf"]:
        ax.scatter(i - 0.2, tot[i] + 0.15, marker="x", color="red", s=110, zorder=3,
                   label="fallo TF-IDF" if primero_t else None)
        primero_t = False
    if q in fallos["lsa"]:
        ax.scatter(i + 0.2, con[i] + 0.15, marker="v", color="darkviolet", s=80, zorder=3,
                   label="fallo LSA" if primero_l else None)
        primero_l = False
ax.set_xticks(x)
ax.set_xticklabels(q_ids)
ax.set_ylabel("nº de términos compartidos con el relevante")
ax.set_title("T3: solapamiento consulta–relevante y fallos (relevante fuera del top 3)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("T3_terminos_compartidos.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T3_analisis_fallos_por_consulta",
    "criterio_de_fallo": "el documento relevante queda fuera del top 3 (posición >= 4 en el ranking de 10 documentos)",
    "fuente_rankings": fuente,
    "fallos_tfidf": fallos["tfidf"],
    "fallos_lsa": fallos["lsa"],
    "num_fallos_tfidf": len(fallos["tfidf"]),
    "num_fallos_lsa": len(fallos["lsa"]),
    "rank_del_relevante_por_consulta": ranks,
    "hit_at_1_verificado_desde_ranks": hit1,
    "hit_at_3_verificado_desde_ranks": hit3,
    "verificacion_tfidf_vs_T1_rankings_iguales": bool(coincide_t1),
    "terminos_compartidos_q1_a_q6": terminos_compartidos,
    "detalle_fallos": detalle_fallos,
    "consultas_degradadas_sin_fallo": degradados,
    "supuestos": [
        "Fallo = posición del relevante >= 4 (fuera del top 3), según el enunciado de la Parte 3.",
        "Los rankings por consulta se cargan de entrada/T1/resultados.json (TF-IDF) y entrada/T2/resultados.json (LSA); solo si no estuvieran disponibles se recalculan con los mismos parámetros (TfidfVectorizer por defecto; TruncatedSVD k=4, random_state=0).",
        "T1 no exportó el vocabulario, así que se reconstruye con TfidfVectorizer por defecto ajustado solo sobre los 10 documentos (misma convención 'fit sobre documentos' de T1/T2) para tokenizar y obtener los pesos.",
        "La intersección de términos usa el tokenizador del propio TfidfVectorizer (minúsculas, sin stopwords, sin stemming, patrón por defecto) filtrada por el vocabulario TF-IDF.",
        "La separación entre 'términos de contenido' y palabras funcionales usa una lista manual mínima de palabras funcionales del español; la lista completa de términos compartidos se reporta sin filtrar.",
        "q4 en LSA queda en el puesto 3: no es un fallo (dentro del top 3) aunque pierde Hit@1; se reporta en 'consultas_degradadas_sin_fallo'.",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
print("T3 — Análisis de fallos por consulta (criterio: relevante fuera del top 3)")
print(f"Fallos TF-IDF: {fallos['tfidf']}  |  Fallos LSA: {fallos['lsa']}")
print(f"Rankings cargados desde: {fuente}  |  TF-IDF recalculado coincide con T1: {coincide_t1}")
for q in q_ids:
    tc = terminos_compartidos[q]
    print(f"{q} -> {tc['relevante']} (rank tfidf={ranks[q]['tfidf']}, lsa={ranks[q]['lsa']}) | "
          f"compartidos ({tc['n_terminos_compartidos']}): {tc['terminos_compartidos']} | "
          f"contenido: {tc['terminos_de_contenido']}")
print(f"Degradados sin fallo: {degradados}")
print("Hit@1 verificado:", hit1, "| Hit@3 verificado:", hit3)
print("Salidas: resultados.json, T3_terminos_compartidos.png")
