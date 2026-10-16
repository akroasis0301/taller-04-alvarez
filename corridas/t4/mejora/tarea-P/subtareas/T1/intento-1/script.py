import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Celda inicial del notebook: datos del estudiante (según enunciado)
# ------------------------------------------------------------------
datos_estudiante = {
    "nombre": "[NOMBRE]",
    "correo": "[CORREO]",
    "cedula": "[CEDULA]",
    "telefono": "[TELEFONO]",
}

# ------------------------------------------------------------------
# Datos del problema: logits de t0..t5
# ------------------------------------------------------------------
tokens = [f"t{i}" for i in range(6)]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

# ------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable (max-shift)
# ------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    z = np.asarray(z, dtype=float)
    z_shift = z - z.max()          # estabilidad: restar el máximo antes de exponenciar
    e = np.exp(z_shift / T)
    return e / e.sum()

temperaturas = [0.5, 1.0, 2.0]
dists = {T: softmax_con_temperatura(logits, T) for T in temperaturas}

def entropia_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

entropias = {T: entropia_bits(dists[T]) for T in temperaturas}

# Verificaciones (garantizadas por la construcción de la softmax)
for T in temperaturas:
    assert abs(dists[T].sum() - 1.0) < 1e-12
    assert (dists[T] > 0).all()

# ------------------------------------------------------------------
# Tabla: tres distribuciones y tres entropías, cuatro decimales
# ------------------------------------------------------------------
tabla = pd.DataFrame(
    {
        "logit": logits,
        **{f"p_T={T}": dists[T] for T in temperaturas},
    },
    index=tokens,
)
tabla_redondeada = tabla.round(4)

entropias_fila = pd.Series(
    {f"p_T={T}": round(entropias[T], 4) for T in temperaturas},
    name="entropia_bits",
)

# ------------------------------------------------------------------
# Figura: barras agrupadas de las tres distribuciones
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(tokens))
ancho = 0.25
for k, T in enumerate(temperaturas):
    ax.bar(x + (k - 1) * ancho, dists[T], width=ancho, label=f"T={T}")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("probabilidad")
ax.set_title("Softmax con temperatura (logits t0..t5)")
ax.legend()
fig.tight_layout()
plt.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# resultados.json (contrato de la subtarea)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1: softmax con temperatura y entropía en bits",
    "celda_inicial_notebook": datos_estudiante,
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {f"T={T}": dists[T].tolist() for T in temperaturas},
    "suma_distribuciones": {f"T={T}": float(dists[T].sum()) for T in temperaturas},
    "entropias_bits": {f"T={T}": entropias[T] for T in temperaturas},
    "tabla_4_decimales": {
        "filas": tokens,
        "columnas": list(tabla_redondeada.columns),
        "valores": tabla_redondeada.values.tolist(),
        "entropias_bits_4dec": {f"T={T}": round(entropias[T], 4) for T in temperaturas},
    },
    "supuestos": [
        "Los datos personales de la primera celda se dejan con los marcadores [NOMBRE], [CORREO], [CEDULA], [TELEFONO] del enunciado, pues el script no conoce los datos reales del estudiante.",
        "La entropía se calcula en bits (log2) sobre la distribución completa; los seis tokens tienen probabilidad positiva para las tres temperaturas.",
    ],
}

Path("resultados.json").write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------
print("Distribuciones (4 decimales):")
print(tabla_redondeada.to_string())
print("\nEntropías en bits (4 decimales):")
print(entropias_fila.to_string())
print("\nSuma de cada distribución:",
      {f"T={T}": float(dists[T].sum()) for T in temperaturas})
print("Figura guardada: t1_softmax_temperatura.png")
print("Resultados guardados: resultados.json")
