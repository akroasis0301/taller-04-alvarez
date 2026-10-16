# -*- coding: utf-8 -*-
"""
Subtarea T3 — Comprobación empírica del muestreo del núcleo (top-p, p = 0.9, T = 1)

1) Carga la distribución renormalizada de T2 (entrada/T2/resultados.json).
2) Toma 10 000 muestras con numpy.random.default_rng(0).
3) Grafica frecuencias observadas vs. probabilidades teóricas (diagrama de barras).
4) Calcula la divergencia KL en bits de la distribución empírica respecto de la teórica.
5) Escribe resultados.json (contrato de la subtarea).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) Distribución renormalizada de T2 (distribución teórica de referencia)
# ----------------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        t2 = json.load(f)
    tokens_nucleo = list(t2["distribucion_renormalizada_orden_tokens"])
    p_teorica = np.array(t2["distribucion_renormalizada_valores"], dtype=float)
    fuente = "entrada/T2/resultados.json"
else:
    # Respaldo: recalcular desde el enunciado (softmax T=1 + top-p p=0.9, corte inclusivo)
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    z = logits - logits.max()
    p_full = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p_full)
    acum = np.cumsum(p_full[orden])
    k = int(np.searchsorted(acum, 0.9) + 1)  # conjunto mínimo que ALCANZA p=0.9
    nucleo_idx = np.sort(orden[:k])
    tokens_nucleo = [tokens[i] for i in nucleo_idx]
    p_teorica = p_full[nucleo_idx] / p_full[nucleo_idx].sum()
    fuente = "recalculada desde logits (softmax T=1, top-p p=0.9 inclusivo)"

p_teorica = p_teorica / p_teorica.sum()  # seguridad numérica

# ----------------------------------------------------------------------------
# 2) Muestreo: 10 000 muestras con numpy.random.default_rng(0)
# ----------------------------------------------------------------------------
N_MUESTRAS = 10_000
SEMILLA = 0
rng = np.random.default_rng(SEMILLA)
indices = rng.choice(len(tokens_nucleo), size=N_MUESTRAS, p=p_teorica)

# ----------------------------------------------------------------------------
# 3) Distribución empírica (frecuencias observadas)
# ----------------------------------------------------------------------------
conteos = np.bincount(indices, minlength=len(tokens_nucleo)).astype(int)
frec_obs = conteos / N_MUESTRAS

# ----------------------------------------------------------------------------
# 4) Divergencia KL en bits: KL(empírica || teórica) = Σ q_i log2(q_i / p_i)
# ----------------------------------------------------------------------------
mask = frec_obs > 0
kl_bits = float(np.sum(frec_obs[mask] * np.log2(frec_obs[mask] / p_teorica[mask])))

# ----------------------------------------------------------------------------
# 5) Diagrama de barras: frecuencias observadas vs. probabilidades teóricas
# ----------------------------------------------------------------------------
nombre_figura = "t3_frecuencias_vs_teoricas.png"
x = np.arange(len(tokens_nucleo))
ancho = 0.38

fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - ancho / 2, frec_obs, width=ancho,
            label=f"Frecuencia observada ({N_MUESTRAS:,} muestras)".replace(",", " "),
            color="#4C72B0", edgecolor="black")
b2 = ax.bar(x + ancho / 2, p_teorica, width=ancho,
            label="Probabilidad teórica (top-p renormalizada)",
            color="#DD8452", edgecolor="black")
for rect in list(b1) + list(b2):
    h = rect.get_height()
    ax.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width() / 2, h),
                ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(tokens_nucleo)
ax.set_ylabel("Probabilidad / frecuencia relativa")
ax.set_xlabel("Token del núcleo")
ax.set_title(
    "Parte 3 — Comprobación empírica del muestreo top-p (T=1, p=0.9)\n"
    f"n = 10 000, rng = default_rng(0) · KL(empírica ‖ teórica) = {kl_bits:.6f} bits"
)
ax.set_ylim(0, max(frec_obs.max(), p_teorica.max()) * 1.18)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(nombre_figura, dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6) Contrato: resultados.json
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T3_comprobacion_empirica_kl_bits",
    "n_muestras": N_MUESTRAS,
    "semilla": SEMILLA,
    "generador": "numpy.random.default_rng(0)",
    "metodo_muestreo": "rng.choice(k, size=10000, p=distribucion_renormalizada)",
    "fuente_distribucion_teorica": fuente,
    "tokens_nucleo": tokens_nucleo,
    "p_teorica_valores": p_teorica.tolist(),
    "p_teorica": {tok: float(p) for tok, p in zip(tokens_nucleo, p_teorica)},
    "p_teorica_4_decimales": {tok: f"{p:.4f}" for tok, p in zip(tokens_nucleo, p_teorica)},
    "conteos_observados_valores": conteos.tolist(),
    "conteos_observados": {tok: int(c) for tok, c in zip(tokens_nucleo, conteos)},
    "frecuencias_observadas_valores": frec_obs.tolist(),
    "frecuencias_observadas": {tok: float(fr) for tok, fr in zip(tokens_nucleo, frec_obs)},
    "frecuencias_observadas_4_decimales": {tok: f"{fr:.4f}" for tok, fr in zip(tokens_nucleo, frec_obs)},
    "suma_frecuencias_observadas": float(frec_obs.sum()),
    "formula_kl": "KL(q||p) = sum_i q_i * log2(q_i / p_i), q = empírica, p = teórica (bits)",
    "kl_bits_empirica_respecto_teorica": kl_bits,
    "kl_bits": kl_bits,
    "kl_bits_4_decimales": f"{kl_bits:.4f}",
    "figura": nombre_figura,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------------
print("=== Subtarea T3 — Comprobación empírica (top-p p=0.9, T=1) ===")
print(f"Muestras: {N_MUESTRAS} con numpy.random.default_rng({SEMILLA})")
print("Token | conteo | frec. observada | prob. teórica")
for tok, c, fo, pt in zip(tokens_nucleo, conteos, frec_obs, p_teorica):
    print(f"  {tok}  | {c:6d} |      {fo:.4f}     |    {pt:.4f}")
print(f"Suma frecuencias observadas: {frec_obs.sum():.6f}")
print(f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
print(f"Figura guardada: {nombre_figura}")
print("Contrato escrito: resultados.json")
