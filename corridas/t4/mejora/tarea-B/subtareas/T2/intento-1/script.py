import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# T2: Top-p (nucleus) con p = 0.9 sobre la distribución de T = 1
# La distribución de T=1 se CARGA de la subtarea previa (entrada/T1/),
# no se recalcula.
# ------------------------------------------------------------------

with open("entrada/T1/resultados.json", "r", encoding="utf-8") as f:
    t1_res = json.load(f)

tokens = list(t1_res["tokens"])
probs = np.array([t1_res["distribuciones"]["T=1.0"][t] for t in tokens], dtype=float)

# 1) Ordenar tokens por probabilidad descendente
orden = np.argsort(-probs)
tokens_ord = [tokens[i] for i in orden]
probs_ord = probs[orden]
acum = np.cumsum(probs_ord)

# 2) Conjunto mínimo cuya probabilidad acumulada alcanza p = 0.9
p = 0.9
idx = int(np.searchsorted(acum, p))   # primer índice con acum >= p
k = min(idx + 1, len(tokens))         # número de tokens conservados
sobreviven = tokens_ord[:k]
descartados = tokens_ord[k:]
acum_nucleo = float(acum[k - 1])
probs_nucleo = probs_ord[:k]

# 3) Renormalizar las probabilidades del núcleo
probs_renorm = probs_nucleo / probs_nucleo.sum()

# Verificaciones (garantizadas por la construcción anterior)
assert acum_nucleo >= p, "la masa del núcleo debe alcanzar p"
if k > 1:
    assert acum[k - 2] < p, "minimalidad: sin el último token no se alcanza p"
assert abs(probs_renorm.sum() - 1.0) < 1e-12, "la renormalizada debe sumar 1"

# 4) resultados.json (contrato de la subtarea)
resultados = {
    "p": p,
    "temperatura": 1.0,
    "distribucion_T1_ordenada": {t: float(pr) for t, pr in zip(tokens_ord, probs_ord)},
    "probabilidades_acumuladas_ordenadas": {t: float(a) for t, a in zip(tokens_ord, acum)},
    "tokens_sobrevivientes": sobreviven,
    "num_tokens_conservados": k,
    "probabilidad_acumulada_nucleo": acum_nucleo,
    "distribucion_renormalizada": {t: float(pr) for t, pr in zip(sobreviven, probs_renorm)},
    "suma_distribucion_renormalizada": float(probs_renorm.sum()),
    "tokens_descartados": descartados,
    "masa_descartada": float(probs_ord[k:].sum()),
    "tabla": [
        {
            "token": t,
            "prob_original_T1": float(pr),
            "prob_acumulada": float(a),
            "conservado": bool(i < k),
            "prob_renormalizada": float(probs_renorm[i]) if i < k else None,
        }
        for i, (t, pr, a) in enumerate(zip(tokens_ord, probs_ord, acum))
    ],
    "supuestos": [
        "El conjunto mínimo se define incluyendo el primer token (en orden descendente) cuya "
        "probabilidad acumulada alcanza o supera p=0.9; los tokens posteriores se descartan, "
        "aunque la masa del núcleo quede por encima de 0.9.",
        "La distribución de T=1 se carga de entrada/T1/resultados.json (subtarea T1) y no se recalcula.",
        "La renormalización divide cada probabilidad conservada entre la masa total del núcleo, "
        "de modo que las probabilidades resultantes suman exactamente 1.0.",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# 5) Figura: distribución original vs. renormalizada tras el corte top-p
renorm_full = np.zeros(len(tokens))
for t, pr in zip(sobreviven, probs_renorm):
    renorm_full[tokens.index(t)] = pr

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(tokens))
ax.bar(x - 0.2, probs, width=0.4, label="T=1 original")
ax.bar(x + 0.2, renorm_full, width=0.4, label="Top-p (p=0.9) renormalizada")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_title("Muestreo de núcleo top-p (p=0.9) sobre la distribución de T=1")
ax.legend()
fig.tight_layout()
fig.savefig("top_p_T1.png", dpi=120)
plt.close(fig)

# 6) Resumen breve
print("Distribución T=1 ordenada (desc):",
      {t: round(float(pr), 4) for t, pr in zip(tokens_ord, probs_ord)})
print("Probabilidades acumuladas:", [round(float(a), 4) for a in acum])
print(f"Tokens que sobreviven al corte top-p (p={p}): {sobreviven}")
print(f"Probabilidad acumulada del núcleo: {acum_nucleo:.6f} (>= {p})")
print("Distribución renormalizada:",
      {t: round(float(pr), 6) for t, pr in zip(sobreviven, probs_renorm)})
print(f"Suma de la distribución renormalizada: {float(probs_renorm.sum()):.12f}")
print("Tokens descartados:", descartados,
      "| masa descartada:", round(float(probs_ord[k:].sum()), 6))
print("Resultados escritos en resultados.json; figura en top_p_T1.png")
