# T1: Softmax con temperatura (NumPy, numéricamente estable) + entropía en bits
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]
etiquetas = ["T=0.5", "T=1", "T=2"]

# ----------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable (se resta el máximo)
# p_i = exp(z_i / T) / sum_j exp(z_j / T)
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    z = np.asarray(z, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    escalados = z / T
    escalados = escalados - np.max(escalados)  # estabilidad numérica
    e = np.exp(escalados)
    return e / np.sum(e)

def entropia_bits(p):
    p = np.asarray(p, dtype=float)
    p_pos = p[p > 0]  # evita log2(0)
    return float(-np.sum(p_pos * np.log2(p_pos)))

# ----------------------------------------------------------------------
# Cálculo de las tres distribuciones y sus entropías
# ----------------------------------------------------------------------
distribuciones = {
    etq: softmax_con_temperatura(logits, T)
    for etq, T in zip(etiquetas, temperaturas)
}
entropias = {etq: entropia_bits(p) for etq, p in distribuciones.items()}

# ----------------------------------------------------------------------
# Tabla: una fila por distribución (6 probabilidades + entropía), 4 decimales
# ----------------------------------------------------------------------
tabla = pd.DataFrame(
    {etq: {tok: float(distribuciones[etq][i]) for i, tok in enumerate(tokens)}
     for etq in etiquetas}
).T
tabla["entropia_bits"] = pd.Series(entropias)
tabla_redondeada = tabla.round(4)
tabla_redondeada.to_csv("tabla_softmax_temperatura.csv")

# ----------------------------------------------------------------------
# Figura: distribuciones para las tres temperaturas
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(tokens))
ancho = 0.25
for k, etq in enumerate(etiquetas):
    ax.bar(x + (k - 1) * ancho, distribuciones[etq], ancho, label=etq)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura sobre los logits de t0..t5")
ax.legend(title="Temperatura")
fig.tight_layout()
plt.savefig("softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# Verificaciones del criterio de éxito
# ----------------------------------------------------------------------
sumas = {etq: float(np.sum(distribuciones[etq])) for etq in etiquetas}
orden_ok = bool(entropias["T=0.5"] < entropias["T=1"] < entropias["T=2"])

# ----------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": {tok: float(z) for tok, z in zip(tokens, logits)},
    "temperaturas": temperaturas,
    "probabilidades_sin_redondear": {
        etq: [float(v) for v in distribuciones[etq]] for etq in etiquetas
    },
    "probabilidades_por_token_sin_redondear": {
        etq: {tok: float(distribuciones[etq][i]) for i, tok in enumerate(tokens)}
        for etq in etiquetas
    },
    "entropia_bits_sin_redondear": {etq: entropias[etq] for etq in etiquetas},
    "tabla_4_decimales": {
        "filas": etiquetas,
        "columnas": tokens + ["entropia_bits"],
        "valores": [
            [float(v) for v in tabla_redondeada.loc[etq].tolist()]
            for etq in etiquetas
        ],
    },
    "entropias_bits_4_decimales": {
        etq: round(entropias[etq], 4) for etq in etiquetas
    },
    "suma_probabilidades_por_distribucion": sumas,
    "orden_entropias_T05_menor_T1_menor_T2": orden_ok,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
print("Softmax con temperatura (estable: se resta el maximo de los logits):")
print(tabla_redondeada.to_string())
print("\nEntropias en bits:")
for etq in etiquetas:
    print(f"  {etq}: {entropias[etq]:.6f} bits  (redondeada: {round(entropias[etq], 4)})")
print("\nSuma de cada fila de probabilidades:", sumas)
print("Orden de entropias T=0.5 < T=1 < T=2:", orden_ok)
print("\nArchivos generados: resultados.json, tabla_softmax_temperatura.csv, softmax_temperatura.png")
