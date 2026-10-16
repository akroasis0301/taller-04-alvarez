# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Tarea C · MMIA 6013).

Con los rankings de TF-IDF (T1) y LSA (T2):
  1) identifica en qué consultas falla cada método (documento relevante
     fuera del top 3);
  2) calcula, para las 6 consultas, la intersección de tokens entre la
     consulta y su documento relevante —con la misma tokenización de
     TfidfVectorizer por defecto usada en T1/T2— y la posición del
     relevante en cada ranking, como base para explicar los fallos.

Salidas en la carpeta actual:
  resultados.json · T3_tabla_terminos_compartidos.csv · T3_terminos_compartidos.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 3
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_T1 = Path("entrada/T1/resultados.json")

# ---------------------------------------------------------------------------
# 1) Corpus y consultas del enunciado (juicios: q1→d04, q2→d03, q3→d08,
#    q4→d05, q5→d09, q6→d02)
# ---------------------------------------------------------------------------
DOCUMENTOS = {
    "d01": ("El mecanismo de atención pondera cada token según su similitud con la consulta; "
            "la atención escalada divide por la raíz de la dimensión."),
    "d02": ("Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones "
            "residuales y normalización."),
    "d03": ("La recuperación aumentada con generación busca fragmentos relevantes y los añade al "
            "prompt del modelo."),
    "d04": ("BM25 es una función de ranking léxica que pondera la frecuencia de términos y la "
            "longitud del documento."),
    "d05": ("Los embeddings densos representan textos como vectores; la similitud coseno compara su "
            "orientación."),
    "d06": ("Un agente con herramientas decide en cada paso qué función llamar y observa el resultado."),
    "d07": ("ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre "
            "intentos."),
    "d08": ("El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales."),
    "d09": ("La temperatura reescala los logits antes del softmax; valores bajos concentran la "
            "probabilidad."),
    "d10": ("La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria."),
}

CONSULTAS = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

DOC_IDS = list(DOCUMENTOS)
Q_IDS = list(CONSULTAS)

# ---------------------------------------------------------------------------
# 2) Rankings de TF-IDF y LSA: se toman de T2 (respaldos: T1 y recálculo)
# ---------------------------------------------------------------------------
def _cargar_json(ruta):
    try:
        with ruta.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _rankings_validos(r):
    return (isinstance(r, dict) and set(r) == set(Q_IDS) and all(
        isinstance(v, list) and len(v) == len(DOC_IDS) and set(v) == set(DOC_IDS)
        for v in r.values()))


def _extraer_rankings(res):
    if not isinstance(res, dict):
        return None, None
    rt = res.get("rankings_tfidf", res.get("rankings"))
    rl = res.get("rankings_lsa")
    return (rt if _rankings_validos(rt) else None,
            rl if _rankings_validos(rl) else None)


def _recalcular():
    """Respaldo: repite T1 (TF-IDF por defecto) y T2 (LSA k=4, random_state=0)."""
    vect = TfidfVectorizer()  # parámetros por defecto, idéntico a T1
    X = vect.fit_transform([DOCUMENTOS[d] for d in DOC_IDS])
    Q = vect.transform([CONSULTAS[q][0] for q in Q_IDS])
    svd = TruncatedSVD(n_components=4, random_state=0)  # idéntico a T2
    Xl, Ql = svd.fit_transform(X), svd.transform(Q)

    def rankings_de(sim):
        return {q: [DOC_IDS[j] for j in sorted(range(len(DOC_IDS)),
                                               key=lambda j: (-sim[i, j], j))]
                for i, q in enumerate(Q_IDS)}

    return rankings_de(cosine_similarity(Q, X)), rankings_de(cosine_similarity(Ql, Xl))


fuente = None
rank_tfidf, rank_lsa = None, None
res_t2 = _cargar_json(RUTA_T2)
if res_t2 is not None:
    rank_tfidf, rank_lsa = _extraer_rankings(res_t2)
    if rank_tfidf and rank_lsa:
        fuente = str(RUTA_T2)
if rank_tfidf is None:
    res_t1 = _cargar_json(RUTA_T1)
    if res_t1 is not None:
        rt, _ = _extraer_rankings(res_t1)
        if rt:
            rank_tfidf = rt
            fuente = f"{RUTA_T1} (solo TF-IDF; LSA recalculado)"
if rank_tfidf is None or rank_lsa is None:
    rt, rl = _recalcular()
    rank_tfidf = rank_tfidf if rank_tfidf else rt
    rank_lsa = rank_lsa if rank_lsa else rl
    fuente = fuente or ("recalculado in situ (TfidfVectorizer por defecto + "
                        "TruncatedSVD k=4, random_state=0)")

# ---------------------------------------------------------------------------
# 3) Tokenización idéntica a la del vectorizador TF-IDF de T1/T2
# ---------------------------------------------------------------------------
analizador = TfidfVectorizer().build_analyzer()  # minúsculas + (?u)\b\w\w+\b, sin stopwords


def tokens_unicos(texto):
    vistos, salida = set(), []
    for t in analizador(texto):
        if t not in vistos:
            vistos.add(t)
            salida.append(t)
    return salida


tokens_docs = {d: tokens_unicos(t) for d, t in DOCUMENTOS.items()}
conj_docs = {d: set(ts) for d, ts in tokens_docs.items()}

# Palabras funcionales (solo para separar contenido vs. función en la explicación)
FUNCIONALES = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
    "cuando", "de", "del", "desde", "donde", "el", "ella", "ellas", "ellos", "en",
    "entre", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estas", "este",
    "esto", "estos", "ha", "han", "hasta", "la", "las", "le", "les", "lo", "los",
    "más", "me", "mi", "mis", "muy", "no", "nos", "o", "otra", "otras", "otro",
    "otros", "para", "pero", "por", "porque", "que", "qué", "se", "según", "ser",
    "si", "sin", "sobre", "su", "sus", "también", "tanto", "te", "un", "una",
    "unas", "unos", "usted", "y", "ya",
}

# ---------------------------------------------------------------------------
# 4) Análisis por consulta: intersección de tokens y posición del relevante
# ---------------------------------------------------------------------------
def rango(rankings, q, relevante):
    return rankings[q].index(relevante) + 1  # posición 1-based


detalle = {}
filas_tabla = []
for q in Q_IDS:
    texto_q, rel = CONSULTAS[q]
    toks_q = tokens_unicos(texto_q)
    compartidos = [t for t in toks_q if t in conj_docs[rel]]          # intersección (orden de la consulta)
    contenido = [t for t in compartidos if t not in FUNCIONALES]      # solo términos de contenido
    r_t, r_l = rango(rank_tfidf, q, rel), rango(rank_lsa, q, rel)
    fallo_t, fallo_l = r_t > TOP_K, r_l > TOP_K

    por_doc = {}
    for d in DOC_IDS:
        inter = [t for t in toks_q if t in conj_docs[d]]
        por_doc[d] = {"n": len(inter), "terminos": inter}

    detalle[q] = {
        "consulta": texto_q,
        "relevante": rel,
        "texto_relevante": DOCUMENTOS[rel],
        "tokens_consulta": toks_q,
        "n_tokens_consulta": len(toks_q),
        "tokens_relevante": tokens_docs[rel],
        "n_tokens_relevante": len(tokens_docs[rel]),
        "terminos_compartidos": compartidos,
        "n_terminos_compartidos": len(compartidos),
        "terminos_compartidos_de_contenido": contenido,
        "n_terminos_compartidos_de_contenido": len(contenido),
        "rank_tfidf": r_t,
        "rank_lsa": r_l,
        "fallo_tfidf": fallo_t,
        "fallo_lsa": fallo_l,
        "top3_tfidf": rank_tfidf[q][:TOP_K],
        "top3_lsa": rank_lsa[q][:TOP_K],
        "terminos_compartidos_con_cada_documento": por_doc,
    }
    filas_tabla.append({
        "consulta": q,
        "texto_consulta": texto_q,
        "relevante": rel,
        "terminos_compartidos": ", ".join(compartidos) if compartidos else "(ninguno)",
        "terminos_compartidos_lista": compartidos,
        "n_terminos_compartidos": len(compartidos),
        "n_terminos_de_contenido": len(contenido),
        "rank_tfidf": r_t,
        "rank_lsa": r_l,
        "fallo_tfidf": int(fallo_t),
        "fallo_lsa": int(fallo_l),
    })

falladas_tfidf = [q for q in Q_IDS if detalle[q]["fallo_tfidf"]]
falladas_lsa = [q for q in Q_IDS if detalle[q]["fallo_lsa"]]


def metricas(rankings):
    n = len(Q_IDS)
    h1 = sum(rango(rankings, q, CONSULTAS[q][1]) == 1 for q in Q_IDS)
    h3 = sum(rango(rankings, q, CONSULTAS[q][1]) <= TOP_K for q in Q_IDS)
    mrr = sum(1.0 / rango(rankings, q, CONSULTAS[q][1]) for q in Q_IDS)
    return {"Hit@1": h1 / n, "Hit@3": h3 / n, "MRR": mrr / n}


# ---------------------------------------------------------------------------
# 5) Explicación de los fallos (texto + datos estructurados)
# ---------------------------------------------------------------------------
def _como_lista(ts):
    return ", ".join(f"'{t}'" for t in ts) if ts else "ninguno"


explicacion_fallos = {}
for q in sorted(set(falladas_tfidf) | set(falladas_lsa)):
    dd = detalle[q]
    rel = dd["relevante"]
    metodos = [m for m, f in (("TF-IDF", dd["fallo_tfidf"]), ("LSA", dd["fallo_lsa"])) if f]
    frases = [
        f"{q} falla en {' y en '.join(metodos)}: el relevante {rel} queda en la posición "
        f"{dd['rank_tfidf']} de TF-IDF y en la {dd['rank_lsa']} de LSA (fallo = fuera del top-{TOP_K}).",
        f"La consulta comparte con {rel} únicamente {dd['n_terminos_compartidos']} token(s) "
        f"[{_como_lista(dd['terminos_compartidos'])}], de los cuales "
        f"{dd['n_terminos_compartidos_de_contenido']} son de contenido "
        f"[{_como_lista(dd['terminos_compartidos_de_contenido'])}].",
    ]
    if dd["n_terminos_compartidos_de_contenido"] == 0:
        contenido_rel = [t for t in tokens_docs[rel] if t not in FUNCIONALES]
        frases.append(
            "Sin ningún término de contenido compartido, el relevante no puede puntuar por "
            f"solapamiento léxico (sus términos de contenido son: {_como_lista(contenido_rel)}); "
            "la consulta lo parafrasea y los documentos que sí comparten tokens con ella "
            "—aunque sean funcionales o de otros temas— la superan."
        )
    superan = {}
    for metodo, rankings in (("tfidf", rank_tfidf), ("lsa", rank_lsa)):
        pos = rankings[q].index(rel)
        arriba = rankings[q][:pos]
        superan[metodo] = [
            {"doc": d,
             "n_tokens_compartidos_con_consulta": dd["terminos_compartidos_con_cada_documento"][d]["n"],
             "terminos": dd["terminos_compartidos_con_cada_documento"][d]["terminos"]}
            for d in arriba
        ]
        if arriba:
            trozos = []
            for d in arriba:
                info = dd["terminos_compartidos_con_cada_documento"][d]
                trozos.append(f"{d} [{info['n']}: {_como_lista(info['terminos'])}]")
            frases.append(f"En {metodo.upper()}, los documentos situados por delante de {rel} son: "
                          + "; ".join(trozos) + ".")
    explicacion_fallos[q] = {
        "falla_en": metodos,
        "rank_tfidf": dd["rank_tfidf"],
        "rank_lsa": dd["rank_lsa"],
        "terminos_compartidos_con_relevante": dd["terminos_compartidos"],
        "n_terminos_compartidos": dd["n_terminos_compartidos"],
        "n_terminos_de_contenido": dd["n_terminos_compartidos_de_contenido"],
        "documentos_que_superan_al_relevante": superan,
        "nota": " ".join(frases),
    }

cambios = {q: {"rank_tfidf": detalle[q]["rank_tfidf"],
               "rank_lsa": detalle[q]["rank_lsa"],
               "delta_lsa_menos_tfidf": detalle[q]["rank_lsa"] - detalle[q]["rank_tfidf"]}
           for q in Q_IDS}

observaciones = []
for q in Q_IDS:
    dd = detalle[q]
    if not (dd["fallo_tfidf"] or dd["fallo_lsa"]) and dd["rank_lsa"] != dd["rank_tfidf"]:
        observaciones.append(
            f"{q}: LSA retrasa el relevante {dd['relevante']} del puesto {dd['rank_tfidf']} (TF-IDF) "
            f"al {dd['rank_lsa']} (LSA); no cuenta como fallo (sigue en el top-3), pero con solo "
            f"{dd['n_terminos_compartidos']} tokens compartidos "
            f"({dd['n_terminos_compartidos_de_contenido']} de contenido) la proyección latente k=4 "
            "diluye parte del matching léxico."
        )

# ---------------------------------------------------------------------------
# 6) Figura: términos compartidos y desglose de la consulta fallada
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

# Panel 1: intersección consulta–relevante + posición en cada ranking
ax = axes[0]
y = np.arange(len(Q_IDS))[::-1]
tot = [detalle[q]["n_terminos_compartidos"] for q in Q_IDS]
con = [detalle[q]["n_terminos_compartidos_de_contenido"] for q in Q_IDS]
col = ["#c0392b" if (detalle[q]["fallo_tfidf"] or detalle[q]["fallo_lsa"]) else "#4c72b0"
       for q in Q_IDS]
ax.barh(y, tot, height=0.62, color=col, alpha=0.9, edgecolor="black", linewidth=0.6,
        label="Tokens compartidos (total)")
ax.barh(y, con, height=0.28, color="#f2a65a", edgecolor="black", linewidth=0.6,
        label="De contenido (sin funcionales)")
ax.set_yticks(y)
ax.set_yticklabels([f"{q} → {CONSULTAS[q][1]}" for q in Q_IDS])
ax.set_xlim(0, max(tot) + 6.0)
for yi, q, t in zip(y, Q_IDS, tot):
    ax.text(t + 0.12, yi, f"TF-IDF r={detalle[q]['rank_tfidf']} | LSA r={detalle[q]['rank_lsa']}",
            va="center", fontsize=8.5)
ax.set_xlabel("N.º de tokens compartidos con el documento relevante")
ax.set_title("Intersección de tokens consulta–relevante\n(rojo: relevante fuera del top-3 en algún método)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# Panel 2: desglose por documento de la consulta fallada
ax = axes[1]
falladas_union = sorted(set(falladas_tfidf) | set(falladas_lsa))
q_focus = falladas_union[0] if falladas_union else min(
    Q_IDS, key=lambda q: detalle[q]["n_terminos_compartidos"])
rel = CONSULTAS[q_focus][1]
por_doc = detalle[q_focus]["terminos_compartidos_con_cada_documento"]
y2 = np.arange(len(DOC_IDS))[::-1]
vals = [por_doc[d]["n"] for d in DOC_IDS]
cols2 = ["#c0392b" if d == rel else "#9bb7d4" for d in DOC_IDS]
ax.barh(y2, vals, height=0.62, color=cols2, edgecolor="black", linewidth=0.6)
ax.set_yticks(y2)
ax.set_yticklabels(DOC_IDS)
ax.set_xlim(0, max(vals) + 6.0)
for yi, d in zip(y2, DOC_IDS):
    n = por_doc[d]["n"]
    ts = ", ".join(por_doc[d]["terminos"])
    ax.text(n + 0.1, yi, f"{n}" + (f"  ({ts})" if ts else ""), va="center", fontsize=8)
ax.set_xlabel("N.º de tokens compartidos con la consulta")
ax.set_title(f"{q_focus} (relevante: {rel}) — tokens compartidos con cada documento\n"
             f"posición del relevante: TF-IDF r={detalle[q_focus]['rank_tfidf']}, "
             f"LSA r={detalle[q_focus]['rank_lsa']}")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig("T3_terminos_compartidos.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7) Salidas: CSV y resultados.json
# ---------------------------------------------------------------------------
tabla_df = pd.DataFrame(filas_tabla)
tabla_df.to_csv("T3_tabla_terminos_compartidos.csv", index=False, encoding="utf-8-sig")

resultados = {
    "subtarea": "T3_analisis_fallos_y_terminos_compartidos",
    "descripcion": ("Consultas en las que TF-IDF y LSA colocan el documento relevante fuera del "
                    "top 3, e intersección de tokens consulta–documento relevante (tokenización de "
                    "TfidfVectorizer por defecto, la misma de T1/T2) junto con la posición del "
                    "relevante en cada ranking."),
    "criterio_fallo": "rank_del_relevante > 3 (fuera del top 3)",
    "fuente_rankings": fuente,
    "tokenizacion": {
        "metodo": "TfidfVectorizer().build_analyzer()",
        "detalle": ("minúsculas + token_pattern (?u)\\b\\w\\w+\\b (tokens de 2+ caracteres), "
                    "sin lista de stopwords — idéntica a la vectorización de T1/T2"),
        "interseccion_sobre": "tokens únicos de cada texto; lista en orden de aparición en la consulta",
        "funcionales_para_separar_contenido": sorted(FUNCIONALES),
    },
    "metricas_contexto": {
        "tfidf": metricas(rank_tfidf),
        "lsa": metricas(rank_lsa),
        "nota": "recalculadas a partir de los rankings usados; coherentes con T1/T2",
    },
    "consultas_falladas": {
        "tfidf": [{"consulta": q, "texto": CONSULTAS[q][0], "relevante": CONSULTAS[q][1],
                   "rank_del_relevante": detalle[q]["rank_tfidf"],
                   "top3_devuelto": detalle[q]["top3_tfidf"]} for q in falladas_tfidf],
        "lsa": [{"consulta": q, "texto": CONSULTAS[q][0], "relevante": CONSULTAS[q][1],
                 "rank_del_relevante": detalle[q]["rank_lsa"],
                 "top3_devuelto": detalle[q]["top3_lsa"]} for q in falladas_lsa],
    },
    "resumen_fallos": {"tfidf": falladas_tfidf, "lsa": falladas_lsa},
    "tabla_terminos_compartidos": filas_tabla,
    "cambios_rank_tfidf_a_lsa": cambios,
    "observaciones": observaciones,
    "explicacion_fallos": explicacion_fallos,
    "detalle_por_consulta": detalle,
    "rankings_usados": {"tfidf": rank_tfidf, "lsa": rank_lsa},
    "archivos_generados": ["resultados.json", "T3_tabla_terminos_compartidos.csv",
                           "T3_terminos_compartidos.png"],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 8) Resumen
# ---------------------------------------------------------------------------
print("=" * 78)
print("T3 · Análisis por consulta — fallos (relevante fuera del top 3) y")
print("términos compartidos consulta–documento relevante")
print("=" * 78)
print(f"Fuente de los rankings : {fuente}")
print(f"Falladas en TF-IDF     : {', '.join(falladas_tfidf) if falladas_tfidf else 'ninguna'}")
print(f"Falladas en LSA        : {', '.join(falladas_lsa) if falladas_lsa else 'ninguna'}")
print()
print("Tabla: términos compartidos consulta–relevante y posición del relevante")
print(tabla_df.to_string(index=False))
print()
for q, exp in explicacion_fallos.items():
    print(f"[{q}] {exp['nota']}")
print()
for obs in observaciones:
    print(f"[nota] {obs}")
print()
print("Archivos generados: resultados.json, T3_tabla_terminos_compartidos.csv, "
      "T3_terminos_compartidos.png")
