# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Softmax con temperatura (numéricamente estable) y entropía en bits.

Implementa en NumPy p_i = exp(z_i/T) / Σ_j exp(z_j/T) restando el máximo de los
logits escalados (estabilidad numérica), calcula la distribución sobre los seis
tokens para T = 0.5, 1 y 2, la entropía en bits de cada distribución, y presenta
todo en una tabla con cuatro decimales. Guarda los resultados en resultados.json
y una figura PNG con las tres distribuciones.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]


# ----------------------------------------------------------------------
# 2. Softmax con temperatura, numéricamente estable
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    """
    Softmax con temperatura p_i = exp(z_i/T) / Σ_j exp(z_j/T).

    Estabilidad numérica: se resta el máximo de los logits escalados (z/T)
    antes de la exponencial, de modo que el mayor exponente es 0 y no hay
    desbordamiento (overflow) en exp.
    """
    z = np.asarray(z, dtype=float)
    x = z / float(T)
    x = x - np.max(x)          # truco de estabilidad: max(x) se vuelve 0
    e = np.exp(x)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon H = -Σ p_i log2(p_i), en bits."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]               # evita log2(0); 0*log2(0) = 0 por convención
    return float(-np.sum(p * np.log2(p)))


# ----------------------------------------------------------------------
# 3. Cálculo de las tres distribuciones y sus entropías
# ----------------------------------------------------------------------
distribuciones = {}
entropias = {}
sumas = {}
for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)
    sumas[T] = float(np.sum(p))

# ----------------------------------------------------------------------
# 4. Tabla con cuatro decimales (generada por el código, no a mano)
# ----------------------------------------------------------------------
filas = []
for T in temperaturas:
    filas.append(
        [f"{T:.1f}"]
        + [f"{v:.4f}" for v in distribuciones[T]]
        + [f"{entropias[T]:.4f}", f"{sumas[T]:.4f}"]
    )
tabla_df = pd.DataFrame(
    filas, columns=["T"] + tokens + ["entropia_bits", "suma_probs"]
)

# Versión markdown de la tabla (para resultados.json / notebook)
encabezado = "| " + " | ".join(tabla_df.columns) + " |"
separador = "|" + "|".join(["---"] * len(tabla_df.columns)) + "|"
cuerpo = "\n".join("| " + " | ".join(fila) + " |" for fila in filas)
tabla_markdown = "\n".join([encabezado, separador, cuerpo])

print("Softmax con temperatura (distribuciones y entropías, 4 decimales):\n")
print(tabla_df.to_string(index=False))
print("\nTabla markdown:\n")
print(tabla_markdown)

# ----------------------------------------------------------------------
# 5. Figura PNG: las tres distribuciones
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ancho = 0.25
posiciones = np.arange(len(tokens))
for i, T in enumerate(temperaturas):
    ax.bar(
        posiciones + (i - 1) * ancho,
        distribuciones[T],
        width=ancho,
        label=f"T = {T:g}",
    )
ax.set_xticks(posiciones)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_xlabel("Token")
ax.set_title("Softmax con temperatura — logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
plt.savefig("t1_distribuciones_temperatura.png", dpi=120)
plt.close(fig)

# ----------------------------------------------------------------------
# 6. resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------
claves_T = {0.5: "T=0.5", 1.0: "T=1", 2.0: "T=2"}
resultados = {
    "subtarea": "T1_softmax_temperatura",
    "tokens": tokens,
    "logits": [float(v) for v in logits],
    "temperaturas": [float(T) for T in temperaturas],
    "distribuciones": {
        claves_T[T]: [float(v) for v in distribuciones[T]] for T in temperaturas
    },
    "entropias_bits": {claves_T[T]: entropias[T] for T in temperaturas},
    "sumas_probabilidades": {claves_T[T]: sumas[T] for T in temperaturas},
    "tabla_4_decimales": {
        "columnas": list(tabla_df.columns),
        "filas": filas,
        "markdown": tabla_markdown,
    },
    "figura": "t1_distribuciones_temperatura.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Resumen breve
# ----------------------------------------------------------------------
print("\nResumen:")
for T in temperaturas:
    print(
        f"  T={T:g}: entropía = {entropias[T]:.4f} bits, "
        f"suma de probabilidades = {sumas[T]:.6f}, "
        f"token más probable = {tokens[int(np.argmax(distribuciones[T]))]}"
    )
print("Archivos generados: resultados.json, t1_distribuciones_temperatura.png")
