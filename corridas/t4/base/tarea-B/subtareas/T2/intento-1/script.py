# T2 — Muestreo de núcleo (top-p) con p=0.9 sobre la distribución T=1 de la Parte 1 (T1)
# Solo usa numpy/matplotlib; lee el resultado previo de entrada/T1/resultados.json.

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Parámetros fijados por el enunciado ----------------
P = 0.9
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])


def softmax_con_temperatura(logits, T):
    """Softmax con temperatura, numéricamente estable (resta el máximo)."""
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropia_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---------------- 1) Distribución T=1 obtenida en T1 ----------------
dist_T1 = None
ruta_T1 = Path("entrada/T1/resultados.json")
if ruta_T1.exists():
    with open(ruta_T1, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    probs = t1.get("probabilidades_sin_redondear", {}).get("T=1")
    if probs is not None:
        dist_T1 = np.array(probs, dtype=float)

if dist_T1 is None:  # respaldo: recalcular con la softmax estable de la Parte 1
    dist_T1 = softmax_con_temperatura(LOGITS, 1.0)

# ---------------- 2) Ordenar tokens de mayor a menor probabilidad ----------------
orden = np.argsort(-dist_T1, kind="stable")
tokens_ordenados = [TOKENS[i] for i in orden]
probs_ordenadas = dist_T1[orden]
acumuladas = np.cumsum(probs_ordenadas)

# ---------------- 3) Conjunto mínimo cuya suma acumulada alcanza p=0.9 ----------------
# Se incluyen los tokens hasta que la suma acumulada alcanza (>=) 0.9 por primera vez.
k = int(np.searchsorted(acumuladas, P, side="left") + 1)
k = min(k, len(TOKENS))

sobrevivientes_idx = orden[:k]
excluidos_idx = orden[k:]

tokens_sobrevivientes = [TOKENS[i] for i in sobrevivientes_idx]
probs_sobrevivientes = dist_T1[sobrevivientes_idx]
suma_sobrevivientes = float(probs_sobrevivientes.sum())

tokens_excluidos = [TOKENS[i] for i in excluidos_idx]
probs_excluidos = dist_T1[excluidos_idx]
masa_excluida = float(probs_excluidos.sum())

# ---------------- 4) Renormalizar las probabilidades del núcleo ----------------
probs_renorm = probs_sobrevivientes / suma_sobrevivientes
suma_renorm = float(probs_renorm.sum())

entropia_original = entropia_bits(dist_T1)
entropia_renorm = entropia_bits(probs_renorm)

# ---------------- resultados.json (contrato de la subtarea) ----------------
resultados = {
    "p": P,
    "temperatura": 1.0,
    "tokens": TOKENS,
    "distribucion_T1_original": {t: float(p) for t, p in zip(TOKENS, dist_T1)},
    # Lista ordenada de tokens con sus probabilidades y acumuladas
    "lista_ordenada_tokens": tokens_ordenados,
    "probabilidades_ordenadas": [float(x) for x in probs_ordenadas],
    "probabilidad_acumulada": [float(x) for x in acumuladas],
    "tabla_ordenada": {
        "posicion": list(range(1, len(TOKENS) + 1)),
        "token": tokens_ordenados,
        "probabilidad": [float(x) for x in probs_ordenadas],
        "probabilidad_acumulada": [float(x) for x in acumuladas],
    },
    # Conjunto mínimo de supervivientes
    "num_tokens_sobrevivientes": k,
    "tokens_sobrevivientes": tokens_sobrevivientes,
    "probabilidades_sobrevivientes_ordenadas": [float(x) for x in probs_sobrevivientes],
    "suma_acumulada_sobrevivientes": suma_sobrevivientes,
    "alcanza_p": bool(suma_sobrevivientes >= P),
    "tokens_excluidos": tokens_excluidos,
    "probabilidades_excluidos": [float(x) for x in probs_excluidos],
    "masa_excluida": masa_excluida,
    # Distribución renormalizada
    "distribucion_renormalizada": {t: float(p) for t, p in zip(tokens_sobrevivientes, probs_renorm)},
    "probabilidades_renormalizadas": [float(x) for x in probs_renorm],
    "suma_renormalizada": suma_renorm,
    "suma_renormalizada_es_1": bool(abs(suma_renorm - 1.0) < 1e-12),
    # Extras informativos
    "entropia_bits_original_T1": entropia_original,
    "entropia_bits_renormalizada": entropia_renorm,
    "tabla_renormalizada_4_decimales": {
        "token": tokens_sobrevivientes,
        "probabilidad_original": [round(float(x), 4) for x in probs_sobrevivientes],
        "probabilidad_renormalizada": [round(float(x), 4) for x in probs_renorm],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------- Figuras ----------------
# Figura 1: distribución original vs núcleo renormalizado
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
x_orig = np.arange(len(TOKENS))
ax[0].bar(x_orig, dist_T1, color="#4C72B0")
ax[0].set_xticks(x_orig)
ax[0].set_xticklabels(TOKENS)
ax[0].set_ylim(0, float(dist_T1.max()) * 1.25)
ax[0].set_title("Distribución T=1 (softmax original)")
ax[0].set_ylabel("probabilidad")
for i, p in enumerate(dist_T1):
    ax[0].text(i, p + 0.01, f"{p:.4f}", ha="center", fontsize=8)

x_surv = np.arange(k)
ax[1].bar(x_surv, probs_renorm, color="#55A868", label="renormalizada")
ax[1].bar(x_surv, probs_sobrevivientes, color="#4C72B0", alpha=0.45, label="original")
ax[1].set_xticks(x_surv)
ax[1].set_xticklabels(tokens_sobrevivientes)
ax[1].set_ylim(0, 1.05)
ax[1].set_title(f"Top-p (p={P}): núcleo de {k} tokens")
ax[1].set_ylabel("probabilidad")
ax[1].legend(fontsize=8)
for i, p in enumerate(probs_renorm):
    ax[1].text(i, p + 0.02, f"{p:.4f}", ha="center", fontsize=8)

fig.suptitle(f"Muestreo de núcleo top-p: suma del núcleo = {suma_sobrevivientes:.4f} >= {P}")
fig.tight_layout()
fig.savefig("top_p_nucleus.png", dpi=120)
plt.close(fig)

# Figura 2: curva de probabilidad acumulada y corte
fig2, ax2 = plt.subplots(figsize=(7, 4.2))
ax2.plot(range(1, len(TOKENS) + 1), acumuladas, marker="o", color="#C44E52")
ax2.axhline(P, color="gray", linestyle="--", label=f"p = {P}")
ax2.axvline(k, color="green", linestyle=":", label=f"corte: {k} tokens")
ax2.set_xticks(range(1, len(TOKENS) + 1))
ax2.set_xticklabels(tokens_ordenados)
ax2.set_xlabel("tokens ordenados por probabilidad descendente")
ax2.set_ylabel("probabilidad acumulada")
ax2.set_title("Probabilidad acumulada y corte top-p")
ax2.legend()
fig2.tight_layout()
fig2.savefig("top_p_acumulada.png", dpi=120)
plt.close(fig2)

# ---------------- Resumen ----------------
print("=== T2: Top-p (nucleus sampling) con p=0.9 sobre la distribución T=1 ===")
print("Orden descendente:", ", ".join(f"{t}={p:.4f}" for t, p in zip(tokens_ordenados, probs_ordenadas)))
print("Acumuladas:       ", ", ".join(f"{a:.4f}" for a in acumuladas))
print(f"Núcleo mínimo ({k} tokens): {tokens_sobrevivientes} | suma acumulada = {suma_sobrevivientes:.6f} >= {P}")
print(f"Excluidos: {tokens_excluidos} | masa de probabilidad excluida = {masa_excluida:.6f}")
print("Distribución renormalizada:", ", ".join(f"{t}={p:.6f}" for t, p in zip(tokens_sobrevivientes, probs_renorm)))
print(f"Suma de probabilidades renormalizadas = {suma_renorm:.12f}")
print(f"Entropía (bits): original T=1 = {entropia_original:.4f} | renormalizada = {entropia_renorm:.4f}")
print("Salidas: resultados.json, top_p_nucleus.png, top_p_acumulada.png")
