import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================
# T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución
#      de T = 1 obtenida en T1 (Parte 1).
# ==================================================================

P = 0.9
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
FIGURA = "t2_topp_nucleo_p09.png"


def softmax_con_temperatura(logits, T):
    """Softmax con temperatura, numéricamente estable (resta del máximo)."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ------------------------------------------------------------------
# 1) Distribución de T = 1: se toma de T1 (entrada/T1/resultados.json)
#    si está disponible; en su defecto se recalcula de forma idéntica.
# ------------------------------------------------------------------
p_t1, fuente = None, None
ruta_t1 = Path("entrada") / "T1" / "resultados.json"
if ruta_t1.exists():
    try:
        with open(ruta_t1, "r", encoding="utf-8") as f:
            t1 = json.load(f)
        p_t1 = np.asarray(t1["distribuciones"]["T=1"], dtype=float)
        fuente = str(ruta_t1)
    except Exception:
        p_t1 = None
if p_t1 is None:
    p_t1 = softmax_con_temperatura(LOGITS, 1.0)
    fuente = "recalculada con softmax numéricamente estable (T = 1)"

# ------------------------------------------------------------------
# 2) Ordenar tokens por probabilidad descendente y acumular.
# ------------------------------------------------------------------
orden = np.argsort(-p_t1, kind="stable")
probs_ord = p_t1[orden]
tokens_ord = [TOKENS[i] for i in orden]
acum = np.cumsum(probs_ord)

# ------------------------------------------------------------------
# 3) Conjunto MÍNIMO cuya probabilidad acumulada ALCANZA p = 0.9.
#    Interpretación inclusiva (la que pide el enunciado): se incluye
#    el token en cuyo paso la acumulada cruza/alcanza el umbral.
# ------------------------------------------------------------------
k = int(np.searchsorted(acum, P, side="left")) + 1
k = max(1, min(k, len(acum)))
idx_nucleo = orden[:k]
idx_fuera = orden[k:]
acum_alcanzada = float(acum[k - 1])
acum_previa = float(acum[k - 2]) if k >= 2 else 0.0

# ------------------------------------------------------------------
# 4) Renormalizar sobre los tokens supervivientes.
# ------------------------------------------------------------------
p_nucleo = p_t1[idx_nucleo]
p_renorm = p_nucleo / p_nucleo.sum()
suma_renorm = float(p_renorm.sum())

tokens_nucleo = [TOKENS[i] for i in idx_nucleo]
tokens_fuera = [TOKENS[i] for i in idx_fuera]

# ------------------- tabla ordenada (4 decimales) -----------------
en_nucleo_set = set(idx_nucleo.tolist())
filas, md_rows = [], []
for r, (tok, pv, av) in enumerate(zip(tokens_ord, probs_ord, acum), start=1):
    dentro = TOKENS.index(tok) in en_nucleo_set
    filas.append([r, tok, f"{pv:.4f}", f"{av:.4f}", "sí" if dentro else "no"])
    md_rows.append(f"| {r} | {tok} | {pv:.4f} | {av:.4f} | {'sí' if dentro else 'no'} |")
markdown_tabla = ("| rank | token | prob | prob_acumulada | en_nucleo |\n"
                  "|---|---|---|---|---|\n" + "\n".join(md_rows))

# ------------------- justificación del corte ----------------------
if k >= 2:
    justificacion = (
        f"Con los tokens ordenados de mayor a menor probabilidad "
        f"({' > '.join(tokens_ord)}), la probabilidad acumulada con los primeros "
        f"{k - 1} tokens ({', '.join(tokens_ord[:k - 1])}) es {acum_previa:.4f} < {P}, "
        f"y al añadir {tokens_ord[k - 1]} alcanza {acum_alcanzada:.4f} >= {P}. "
        f"Por tanto, el conjunto mínimo cuya probabilidad acumulada alcanza p = {P} "
        f"es {{ {', '.join(tokens_nucleo)} }} (k = {k} tokens) y el corte se justifica "
        f"con la probabilidad acumulada alcanzada = {acum_alcanzada:.4f}."
    )
else:
    justificacion = (
        f"El token más probable {tokens_ord[0]} ya alcanza una acumulada "
        f"{acum_alcanzada:.4f} >= {P}; el núcleo es {{{tokens_ord[0]}}}."
    )

# ------------------------------------------------------------------
# Resultados (contrato resultados.json)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T2_topp_nucleo_p09",
    "p": P,
    "temperatura_base": 1.0,
    "fuente_distribucion_T1": fuente,
    "tokens": TOKENS,
    "logits": LOGITS.tolist(),
    "distribucion_T1": {t: float(v) for t, v in zip(TOKENS, p_t1)},
    "interpretacion_corte": ("El enunciado pide conservar el conjunto mínimo cuya "
                             "probabilidad acumulada ALCANZA p = 0.9, por lo que se "
                             "incluye el token en cuyo paso se cruza el umbral "
                             "(corte inclusivo)."),
    "orden_descendente": tokens_ord,
    "probs_ordenadas": probs_ord.tolist(),
    "prob_acumulada_ordenada": acum.tolist(),
    "tabla_ordenada_4_decimales": {
        "columnas": ["rank", "token", "prob", "prob_acumulada", "en_nucleo"],
        "filas": filas,
    },
    "markdown_tabla_ordenada": markdown_tabla,
    "tokens_sobrevivientes": tokens_nucleo,
    "tokens_excluidos": tokens_fuera,
    "num_tokens_nucleo": k,
    "prob_acumulada_alcanzada": acum_alcanzada,
    "prob_acumulada_alcanzada_4_decimales": f"{acum_alcanzada:.4f}",
    "prob_acumulada_antes_del_ultimo_nucleo": acum_previa,
    "justificacion_corte": justificacion,
    "distribucion_renormalizada": {t: float(v) for t, v in zip(tokens_nucleo, p_renorm)},
    "distribucion_renormalizada_orden_tokens": tokens_nucleo,
    "distribucion_renormalizada_valores": p_renorm.tolist(),
    "distribucion_renormalizada_4_decimales": {
        t: f"{v:.4f}" for t, v in zip(tokens_nucleo, p_renorm)
    },
    "suma_probs_renormalizadas": suma_renorm,
    "suma_probs_renormalizadas_4_decimales": f"{suma_renorm:.4f}",
    "verificaciones": {
        "suma_renormalizada_es_1": bool(abs(suma_renorm - 1.0) < 1e-9),
        "acumulada_alcanza_p": bool(acum_alcanzada >= P),
        "acumulada_previa_menor_que_p": bool(acum_previa < P),
        "nucleo_es_conjunto_minimo": bool(acum_previa < P <= acum_alcanzada),
    },
    "figura": FIGURA,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# Figura: distribución T=1, acumulación con corte y renormalizada
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

ax = axes[0]
mascara = np.zeros(len(TOKENS), dtype=bool)
mascara[idx_nucleo] = True
colores = ["#2a9d8f" if m else "#d64545" for m in mascara]
ax.bar(TOKENS, p_t1, color=colores, edgecolor="black", linewidth=0.6)
for i, v in enumerate(p_t1):
    ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=8)
ax.set_ylim(0, float(p_t1.max()) * 1.22)
ax.set_ylabel("p_i  (T = 1)")
ax.set_title("Distribución T=1\n(verde = núcleo, rojo = excluido)", fontsize=10)

ax = axes[1]
ranks = np.arange(1, len(TOKENS) + 1)
ax.step(ranks, acum, where="post", marker="o", color="#264653", label="prob. acumulada")
ax.axhline(P, color="#e9c46a", linestyle="--", linewidth=1.5, label=f"p = {P}")
ax.axvline(k, color="#d64545", linestyle=":", linewidth=1.5, label=f"corte: k = {k}")
ax.scatter([k], [acum_alcanzada], color="#d64545", zorder=5)
ax.annotate(f"{acum_alcanzada:.4f}", (k, acum_alcanzada),
            textcoords="offset points", xytext=(10, -12),
            fontsize=9, color="#d64545")
ax.set_xticks(ranks)
ax.set_xticklabels(tokens_ord)
ax.set_ylim(0, 1.06)
ax.set_xlabel("tokens ordenados (prob. descendente)")
ax.set_ylabel("probabilidad acumulada")
ax.set_title("Acumulación y corte del núcleo", fontsize=10)
ax.legend(fontsize=8, loc="lower right")

ax = axes[2]
ax.bar(tokens_nucleo, p_renorm, color="#2a9d8f", edgecolor="black", linewidth=0.6)
for i, v in enumerate(p_renorm):
    ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=8)
ax.set_ylim(0, float(p_renorm.max()) * 1.22)
ax.set_ylabel("p_i renormalizada")
ax.set_title(f"Distribución renormalizada\n(suma = {suma_renorm:.4f})", fontsize=10)

fig.suptitle(f"Top-p (nucleus sampling) con p = {P} sobre la distribución de T = 1",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIGURA, dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------
print("=" * 66)
print("T2 — Top-p (nucleus sampling) con p = 0.9 sobre la distribución T = 1")
print("=" * 66)
print(f"Fuente distribución T=1     : {fuente}")
print("Orden descendente           : "
      + " > ".join(f"{t}({v:.4f})" for t, v in zip(tokens_ord, probs_ord)))
print("Prob. acumulada por rango   : "
      + " | ".join(f"{a:.4f}" for a in acum))
print(f"Tokens supervivientes (k={k})  : {tokens_nucleo}")
print(f"Tokens excluidos            : {tokens_fuera}")
print(f"Prob. acumulada alcanzada   : {acum_alcanzada:.6f}  (>= p = {P}; "
      f"previa = {acum_previa:.6f} < p)")
print("Distribución renormalizada  : "
      + ", ".join(f"{t}={v:.6f}" for t, v in zip(tokens_nucleo, p_renorm)))
print(f"Suma renormalizada          : {suma_renorm:.12f}")
print(f"Figura                      : {FIGURA}")
print("Resultados                  : resultados.json")
