# SUBTAREA T3: Comprobación empírica de la distribución renormalizada (top-p, p=0.9, T=1)
# - 10 000 muestras con numpy.random.default_rng(0)
# - Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# - Divergencia KL (empírica || teórica) en bits

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Cargar la distribución renormalizada producida por T2
# ------------------------------------------------------------------
ruta_t2 = Path("entrada/T2/resultados.json")
if ruta_t2.exists():
    with open(ruta_t2, "r", encoding="utf-8") as f:
        res_t2 = json.load(f)
    tokens = list(res_t2["distribucion_renormalizada"].keys())
    p_teorica = np.array([res_t2["distribucion_renormalizada"][t] for t in tokens],
                         dtype=float)
else:
    # Fallback defensivo: reconstruir desde los logits del enunciado
    # (solo se usa si entrada/T2/resultados.json no existe)
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0])
    tokens_todos = [f"t{i}" for i in range(6)]
    z = logits - logits.max()
    p_t1 = np.exp(z) / np.exp(z).sum()
    orden = np.argsort(-p_t1)
    acum = np.cumsum(p_t1[orden])
    n_nucleo = int(np.searchsorted(acum, 0.9) + 1)
    idx_nucleo = np.sort(orden[:n_nucleo])
    tokens = [tokens_todos[i] for i in idx_nucleo]
    p_teorica = p_t1[idx_nucleo] / p_t1[idx_nucleo].sum()

p_teorica = p_teorica / p_teorica.sum()  # garantiza suma exactamente 1
n_tokens = len(tokens)

# ------------------------------------------------------------------
# 2) Muestrear 10 000 veces con la semilla exigida
# ------------------------------------------------------------------
N = 10_000
rng = np.random.default_rng(0)
indices = np.arange(n_tokens)
muestras = rng.choice(indices, size=N, p=p_teorica)

conteos = np.bincount(muestras, minlength=n_tokens)
frecuencias_obs = conteos / N

# ------------------------------------------------------------------
# 3) Divergencia KL(empírica || teórica) en bits
#    KL = Σ q_i * log2(q_i / p_i); términos con q_i = 0 valen 0
# ------------------------------------------------------------------
mascara = frecuencias_obs > 0
kl_bits = float(np.sum(frecuencias_obs[mascara] *
                       np.log2(frecuencias_obs[mascara] / p_teorica[mascara])))

# ------------------------------------------------------------------
# 4) Gráfico de barras: frecuencias observadas vs probabilidades teóricas
# ------------------------------------------------------------------
x = np.arange(n_tokens)
ancho = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - ancho / 2, frecuencias_obs, width=ancho,
       label="Frecuencia observada (10 000 muestras)", color="#4C72B0")
ax.bar(x + ancho / 2, p_teorica, width=ancho,
       label="Probabilidad teórica (renormalizada)", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_title("Parte 3: frecuencias observadas vs. probabilidades teóricas\n"
             "muestreo top-p (p=0.9, T=1), 10 000 muestras, rng(0)")
ax.legend()
for i, (fo, pt) in enumerate(zip(frecuencias_obs, p_teorica)):
    ax.text(i - ancho / 2, fo + 0.005, f"{fo:.4f}", ha="center", fontsize=8)
    ax.text(i + ancho / 2, pt + 0.005, f"{pt:.4f}", ha="center", fontsize=8)
ax.set_ylim(0, max(p_teorica.max(), frecuencias_obs.max()) * 1.15)
fig.tight_layout()
plt.savefig("parte3_frecuencias_vs_teorica.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 5) Escribir resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "n_muestras": N,
    "semilla": 0,
    "tokens": tokens,
    "probabilidades_teoricas": {t: float(p) for t, p in zip(tokens, p_teorica)},
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(f) for t, f in zip(tokens, frecuencias_obs)},
    "kl_bits_empirica_vs_teorica": kl_bits,
    "figura": "parte3_frecuencias_vs_teorica.png",
    "supuestos": [
        "La distribución renormalizada se carga de entrada/T2/resultados.json (subtarea T2); "
        "el muestreo se realiza sobre los tokens sobrevivientes del núcleo (t0-t3) con sus "
        "probabilidades renormalizadas.",
        "La divergencia KL se calcula como KL(empírica || teórica) = Σ q_i log2(q_i / p_i) "
        "sobre los tokens del núcleo, en bits (log base 2).",
        "Las 'frecuencias observadas' son conteos divididos entre 10 000, de modo que son "
        "directamente comparables con las probabilidades teóricas en el gráfico de barras.",
    ],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6) Resumen breve
# ------------------------------------------------------------------
print("=== SUBTAREA T3: comprobación empírica (top-p p=0.9, T=1) ===")
print(f"Muestras: {N} | Semilla: 0 | Tokens del núcleo: {tokens}")
for t, c, fo, pt in zip(tokens, conteos, frecuencias_obs, p_teorica):
    print(f"  {t}: conteo={c:6d}  freq_obs={fo:.6f}  p_teorica={pt:.6f}")
print(f"Divergencia KL(empírica || teórica) = {kl_bits:.6f} bits")
print("Figura guardada: parte3_frecuencias_vs_teorica.png")
print("Resultados escritos en resultados.json")
