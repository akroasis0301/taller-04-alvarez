# SUBTAREA T3: Comprobación empírica del muestreo top-p (Parte 3)
# Toma 10 000 muestras de la distribución renormalizada de T2 con
# numpy.random.default_rng(0), compara frecuencias observadas vs probabilidades
# teóricas en un gráfico de barras y calcula la divergencia KL en bits.

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Cargar la distribución renormalizada de T2 (no se recalcula)
# ------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
with open(ruta_t2, "r", encoding="utf-8") as f:
    res_t2 = json.load(f)

dist_renorm = res_t2["distribucion_renormalizada"]  # dict token -> prob
tokens = list(dist_renorm.keys())                   # ["t0", "t1", "t2", "t3"]
p_teorica = np.array([dist_renorm[t] for t in tokens], dtype=float)

assert abs(p_teorica.sum() - 1.0) < 1e-12, "La distribución teórica debe sumar 1"

# ------------------------------------------------------------------
# 2) Muestreo: 10 000 muestras con default_rng(0)
# ------------------------------------------------------------------
rng = np.random.default_rng(0)
n_muestras = 10_000
indices = rng.choice(len(tokens), size=n_muestras, p=p_teorica)

conteos = np.bincount(indices, minlength=len(tokens)).astype(float)
frecuencias_obs = conteos / n_muestras

# ------------------------------------------------------------------
# 3) Divergencia KL (empírica || teórica) en bits
#    KL = Σ_i q_i * log2(q_i / p_i); términos con q_i = 0 aportan 0
# ------------------------------------------------------------------
mascara = frecuencias_obs > 0
kl_bits = float(np.sum(
    frecuencias_obs[mascara]
    * np.log2(frecuencias_obs[mascara] / p_teorica[mascara])
))

# ------------------------------------------------------------------
# 4) Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# ------------------------------------------------------------------
x = np.arange(len(tokens))
ancho = 0.38

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - ancho / 2, frecuencias_obs, width=ancho,
            label="Frecuencia observada (10 000 muestras)", color="#4C72B0")
b2 = ax.bar(x + ancho / 2, p_teorica, width=ancho,
            label="Probabilidad teórica (top-p renormalizada)", color="#DD8452")

for barras in (b1, b2):
    for b in barras:
        ax.annotate(f"{b.get_height():.4f}",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_title("Top-p (p=0.9): frecuencias observadas vs probabilidades teóricas\n"
             f"10 000 muestras, default_rng(0) — KL = {kl_bits:.6f} bits")
ax.legend()
ax.set_ylim(0, max(frecuencias_obs.max(), p_teorica.max()) * 1.15)
fig.tight_layout()
plt.savefig("T3_frecuencias_vs_teoricas.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) Escribir resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T3: comprobación empírica del muestreo top-p (10 000 muestras, default_rng(0))",
    "semilla": 0,
    "generador": "numpy.random.default_rng(0)",
    "n_muestras": n_muestras,
    "tokens": tokens,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens, p_teorica)},
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(f) for t, f in zip(tokens, frecuencias_obs)},
    "divergencia_KL_bits": kl_bits,
    "figura": "T3_frecuencias_vs_teoricas.png",
    "supuestos": [
        "La distribución renormalizada se carga de entrada/T2/resultados.json (subtarea T2) sin recalcularla.",
        "El muestreo usa rng.choice sobre los 4 tokens del núcleo con las probabilidades renormalizadas de T2.",
        "La KL se calcula como KL(empírica || teórica) = Σ q_i log2(q_i / p_i); los tokens con frecuencia empírica 0 aportan 0 (aquí todos los tokens del núcleo aparecen en las muestras).",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6) Resumen
# ------------------------------------------------------------------
print("=== T3: Comprobación empírica (top-p, p=0.9) ===")
print(f"Muestras: {n_muestras} | Generador: default_rng(0)")
print(f"{'token':<6}{'teórica':>12}{'observada':>12}{'conteo':>10}")
for t, pt, fo, c in zip(tokens, p_teorica, frecuencias_obs, conteos):
    print(f"{t:<6}{pt:>12.6f}{fo:>12.6f}{int(c):>10d}")
print(f"Divergencia KL (empírica || teórica) = {kl_bits:.6f} bits")
print("Figura guardada: T3_frecuencias_vs_teoricas.png")
print("Resultados escritos en resultados.json")
