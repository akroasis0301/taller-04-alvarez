# -*- coding: utf-8 -*-
"""
SUBTAREA T3 — Comprobación empírica del muestreo top-p (Parte 3 de la Tarea B)

Pasos:
1. Cargar de entrada/T2/resultados.json los tokens del núcleo (top-p, p=0.9, T=1)
   y su distribución renormalizada (con recálculo de respaldo desde los logits).
2. Tomar 10,000 muestras con numpy.random.default_rng(0).
3. Calcular frecuencias relativas observadas por token.
4. Gráfico de barras: frecuencias observadas vs probabilidades teóricas (solo núcleo).
5. Divergencia KL en bits de la distribución empírica respecto de la teórica.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Distribución teórica renormalizada (resultado de T2)
# ----------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
t2 = None
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        t2 = json.load(f)

if t2 is not None:
    tokens_nucleo = list(t2["tokens_sobrevivientes"])
    p_teorica = np.array(
        [float(t2["distribucion_renormalizada"][tk]) for tk in tokens_nucleo],
        dtype=float,
    )
else:
    # Respaldo: recalcular softmax(T=1) + top-p (p=0.9) desde los logits del enunciado
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()  # softmax numéricamente estable
    p_full = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p_full)
    acum = np.cumsum(p_full[orden])
    k = int(np.searchsorted(acum, 0.9) + 1)  # conjunto mínimo que alcanza 0.9
    idx_nucleo = np.sort(orden[:k])
    tokens_full = [f"t{i}" for i in range(len(logits))]
    tokens_nucleo = [tokens_full[i] for i in idx_nucleo]
    p_teorica = p_full[idx_nucleo] / p_full[idx_nucleo].sum()

assert abs(p_teorica.sum() - 1.0) < 1e-12, "La distribución renormalizada debe sumar 1"

# ----------------------------------------------------------------------
# 2) Muestreo: 10,000 muestras con numpy.random.default_rng(0)
# ----------------------------------------------------------------------
SEMILLA = 0
N_MUESTRAS = 10_000

rng = np.random.default_rng(SEMILLA)
indices = rng.choice(len(tokens_nucleo), size=N_MUESTRAS, p=p_teorica)

# ----------------------------------------------------------------------
# 3) Frecuencias relativas observadas
# ----------------------------------------------------------------------
conteos = np.array([(indices == i).sum() for i in range(len(tokens_nucleo))], dtype=int)
frec_obs = conteos / N_MUESTRAS

# ----------------------------------------------------------------------
# 4) Gráfico de barras: observado vs teórico (solo tokens del núcleo)
# ----------------------------------------------------------------------
nombre_figura = "T3_frecuencias_observadas_vs_teoricas.png"
x = np.arange(len(tokens_nucleo))
ancho = 0.38

fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - ancho / 2, frec_obs, ancho,
            label="Frecuencia observada (empírica)",
            color="#4C72B0", edgecolor="black")
b2 = ax.bar(x + ancho / 2, p_teorica, ancho,
            label="Probabilidad teórica (renormalizada)",
            color="#DD8452", edgecolor="black")
for barras in (b1, b2):
    for b in barras:
        h = b.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(tokens_nucleo)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia relativa")
ax.set_title(
    "Top-p (p=0.9, T=1): frecuencias observadas vs probabilidades teóricas\n"
    f"{N_MUESTRAS} muestras, numpy.random.default_rng({SEMILLA})"
)
ax.set_ylim(0.0, max(frec_obs.max(), p_teorica.max()) * 1.18)
ax.legend()
fig.tight_layout()
fig.savefig(nombre_figura, dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 5) Divergencia KL (en bits) de la empírica respecto de la teórica
#    D_KL(q || p) = sum_i q_i * log2(q_i / p_i);  términos con q_i = 0 valen 0
# ----------------------------------------------------------------------
mask = frec_obs > 0
kl_bits = float(np.sum(frec_obs[mask] * np.log2(frec_obs[mask] / p_teorica[mask])))
kl_nats = float(np.sum(frec_obs[mask] * np.log(frec_obs[mask] / p_teorica[mask])))

entropia_empirica_bits = float(-np.sum(frec_obs[mask] * np.log2(frec_obs[mask])))
entropia_teorica_bits = float(-np.sum(p_teorica * np.log2(p_teorica)))

# ----------------------------------------------------------------------
# 6) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T3",
    "descripcion": (
        "Comprobación empírica: 10,000 muestras de la distribución renormalizada "
        "top-p (p=0.9, T=1) con numpy.random.default_rng(0); frecuencias observadas "
        "vs probabilidades teóricas y divergencia KL en bits (empírica || teórica)."
    ),
    "semilla": SEMILLA,
    "num_muestras": N_MUESTRAS,
    "tokens_nucleo": tokens_nucleo,
    "probabilidades_teoricas_renormalizadas": {
        tk: float(p) for tk, p in zip(tokens_nucleo, p_teorica)
    },
    "conteos_observados": {tk: int(c) for tk, c in zip(tokens_nucleo, conteos)},
    "frecuencias_observadas": {
        tk: float(fr) for tk, fr in zip(tokens_nucleo, frec_obs)
    },
    "tabla_comparativa": {
        "token": tokens_nucleo,
        "probabilidad_teorica": [float(v) for v in p_teorica],
        "frecuencia_observada": [float(v) for v in frec_obs],
        "conteo_observado": [int(v) for v in conteos],
        "diferencia_obs_menos_teorica": [float(v) for v in (frec_obs - p_teorica)],
    },
    "divergencia_KL_bits": kl_bits,
    "divergencia_KL_nats": kl_nats,
    "entropia_bits_empirica": entropia_empirica_bits,
    "entropia_bits_teorica_renormalizada": entropia_teorica_bits,
    "figura": nombre_figura,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
print("=" * 64)
print("T3 — Comprobación empírica del muestreo top-p (p=0.9, T=1)")
print("=" * 64)
print(f"Muestras: {N_MUESTRAS} | Semilla: {SEMILLA} (numpy.random.default_rng)")
print(f"Tokens del núcleo: {tokens_nucleo}")
print("\nToken | Teórica (renorm.) | Observada | Conteo")
for tk, pt, fo, c in zip(tokens_nucleo, p_teorica, frec_obs, conteos):
    print(f"  {tk}  |      {pt:.4f}       |  {fo:.4f}  | {c}")
print(f"\nDivergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
print(f"  (equivalente en nats: {kl_nats:.6f})")
print(f"Entropía empírica: {entropia_empirica_bits:.6f} bits | "
      f"Entropía teórica: {entropia_teorica_bits:.6f} bits")
print(f"Figura guardada: {nombre_figura}")
print("Resultados escritos en resultados.json")
