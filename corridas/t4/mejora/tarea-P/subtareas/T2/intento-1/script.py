import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# T2: Top-p (nucleus) con p=0.9 sobre la distribución de T=1 de T1
# ------------------------------------------------------------------

# 1) Cargar la distribución de T=1 calculada en T1 (no se recalcula)
ruta_t1 = Path("entrada/T1/resultados.json")
with open(ruta_t1, "r", encoding="utf-8") as f:
    t1 = json.load(f)

tokens = list(t1["tokens"])
p_t1 = np.array(t1["distribuciones"]["T=1.0"], dtype=float)

# 2) Ordenar tokens de mayor a menor probabilidad
orden = np.argsort(-p_t1)
tokens_ordenados = [tokens[i] for i in orden]
probs_ordenadas = p_t1[orden]

# 3) Conjunto mínimo cuya probabilidad acumulada alcanza p = 0.9
p_corte = 0.9
acumulada = np.cumsum(probs_ordenadas)
# searchsorted (side='left') devuelve el primer índice i con acumulada[i] >= p_corte
k = int(np.searchsorted(acumulada, p_corte) + 1)  # número de tokens conservados

tokens_nucleo = tokens_ordenados[:k]
probs_nucleo = probs_ordenadas[:k]
acumulada_nucleo = float(acumulada[k - 1])
tokens_descartados = tokens_ordenados[k:]

# 4) Renormalizar las probabilidades del núcleo
probs_renorm = probs_nucleo / probs_nucleo.sum()

# Verificaciones (garantizadas por la construcción anterior)
assert acumulada_nucleo >= p_corte, "la acumulada del núcleo debe alcanzar p=0.9"
if k > 1:
    assert acumulada[k - 2] < p_corte, "el conjunto debe ser mínimo (sin el último token no llega a p)"
assert abs(probs_renorm.sum() - 1.0) < 1e-12, "la renormalizada debe sumar 1"

# 5) Tabla resumen (original, acumulada, renormalizada)
tabla = []
for i, tok in enumerate(tokens_ordenados):
    fila = {
        "token": tok,
        "p_T1_original": float(probs_ordenadas[i]),
        "acumulada": float(acumulada[i]),
        "en_nucleo": i < k,
        "p_renormalizada": float(probs_renorm[i]) if i < k else None,
    }
    tabla.append(fila)

# 6) Escribir resultados.json (contrato de la subtarea)
resultados = {
    "subtarea": "T2: muestreo de núcleo (top-p, p=0.9) sobre la distribución de T=1",
    "p": p_corte,
    "temperatura": 1.0,
    "distribucion_T1_original": {tok: float(p) for tok, p in zip(tokens, p_t1)},
    "tokens_ordenados_desc": tokens_ordenados,
    "probs_ordenadas_desc": [float(x) for x in probs_ordenadas],
    "acumulada_ordenada": [float(x) for x in acumulada],
    "tokens_sobrevivientes": tokens_nucleo,
    "probs_nucleo_sin_renormalizar": [float(x) for x in probs_nucleo],
    "probabilidad_acumulada_nucleo": acumulada_nucleo,
    "distribucion_renormalizada": {tok: float(p) for tok, p in zip(tokens_nucleo, probs_renorm)},
    "tokens_descartados": tokens_descartados,
    "suma_distribucion_renormalizada": float(probs_renorm.sum()),
    "tabla_completa": tabla,
    "supuestos": [
        "La distribución de T=1 se carga de entrada/T1/resultados.json (subtarea T1) sin recalcularla.",
        "El conjunto mínimo se construye conservando tokens en orden decreciente de probabilidad hasta que la suma alcance o supere p=0.9 por primera vez; el último token incluido puede hacer que la acumulada supere ligeramente 0.9 (aquí 0.9688).",
        "Los tokens fuera del núcleo (t4, t5) se descartan y no participan en la renormalización.",
    ],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# 7) Figura: distribución original de T=1 vs. renormalizada del núcleo
fig, ax = plt.subplots(figsize=(8, 4.5))
x_orig = np.arange(len(tokens))
ax.bar(x_orig - 0.2, p_t1, width=0.4, label="T=1 original", color="#4C72B0")
x_nuc = np.arange(len(tokens_nucleo))
ax.bar(x_nuc + 0.2, probs_renorm, width=0.4, label="Top-p renormalizada", color="#DD8452")
ax.set_xticks(x_orig)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_title(f"Top-p (p=0.9) sobre T=1: sobreviven {tokens_nucleo} (acumulada={acumulada_nucleo:.4f})")
ax.legend()
fig.tight_layout()
fig.savefig("top_p_T1.png", dpi=120)
plt.close(fig)

# 8) Resumen breve
print("=== T2: Top-p con p=0.9 sobre la distribución de T=1 ===")
print("Tokens ordenados (mayor a menor):", tokens_ordenados)
print("Probabilidades ordenadas:", np.round(probs_ordenadas, 6).tolist())
print("Acumulada:", np.round(acumulada, 6).tolist())
print(f"Tokens sobrevivientes al núcleo: {tokens_nucleo} (k={k})")
print(f"Probabilidad acumulada del núcleo: {acumulada_nucleo:.6f} (>= 0.9)")
print("Distribución renormalizada:")
for tok, p in zip(tokens_nucleo, probs_renorm):
    print(f"  {tok}: {p:.6f}")
print(f"Suma de la renormalizada: {probs_renorm.sum():.12f}")
print("Tokens descartados:", tokens_descartados)
print("Resultados escritos en resultados.json; figura en top_p_T1.png")
