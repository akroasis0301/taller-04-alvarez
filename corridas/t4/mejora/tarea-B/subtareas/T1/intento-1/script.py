import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- Datos del enunciado -----
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]


def softmax_con_temperatura(z, T):
    """Softmax con temperatura p_i = exp(z_i/T) / sum_j exp(z_j/T),
    numéricamente estable: se resta el máximo de los logits escalados."""
    z = np.asarray(z, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura debe ser positiva.")
    escalados = z / T
    escalados = escalados - escalados.max()  # estabilidad numérica
    e = np.exp(escalados)
    return e / e.sum()


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -sum p_i * log2(p_i)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ----- Cálculo de distribuciones y entropías -----
filas = []
entropias = {}
for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    H = entropia_bits(p)
    entropias[T] = H
    fila = {"T": T}
    for tok, pi in zip(tokens, p):
        fila[tok] = round(float(pi), 4)  # cuatro decimales
    fila["entropia_bits"] = round(H, 4)
    filas.append(fila)

tabla = pd.DataFrame(filas)

# ----- Verificaciones del criterio de éxito -----
sumas = [float(np.sum(softmax_con_temperatura(logits, T))) for T in temperaturas]
assert all(abs(s - 1.0) < 1e-12 for s in sumas), "Cada distribución debe sumar 1.0"
Hs = [entropias[T] for T in temperaturas]
assert Hs[0] < Hs[1] < Hs[2], "La entropía debe crecer con T: H(0.5) < H(1) < H(2)"

# ----- resultados.json (contrato de la subtarea) -----
resultados = {
    "tokens": tokens,
    "logits": logits.tolist(),
    "temperaturas": temperaturas,
    "distribuciones": {
        f"T={T}": {tok: float(pi) for tok, pi in
                   zip(tokens, softmax_con_temperatura(logits, T))}
        for T in temperaturas
    },
    "sumas_distribuciones": {f"T={T}": s for T, s in zip(temperaturas, sumas)},
    "entropias_bits": {f"T={T}": H for T, H in zip(temperaturas, Hs)},
    "tabla_4_decimales": tabla.to_dict(orient="records"),
    "supuestos": [
        "La entropía se calcula en bits (log2) sobre la distribución completa; "
        "los seis tokens tienen probabilidad positiva, no se requiere corrección.",
        "La estabilidad numérica se logra restando el máximo de los logits ya "
        "escalados por 1/T antes de la exponenciación.",
    ],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----- Figura: efecto de la temperatura -----
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(tokens))
ancho = 0.25
for i, T in enumerate(temperaturas):
    p = softmax_con_temperatura(logits, T)
    ax.bar(x + (i - 1) * ancho, p, width=ancho, label=f"T={T}")
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_ylabel("Probabilidad")
ax.set_title("Softmax con temperatura (logits 2.0, 1.0, 0.5, 0.2, -1.0, -3.0)")
ax.legend()
fig.tight_layout()
fig.savefig("t1_softmax_temperatura.png", dpi=120)
plt.close(fig)

# ----- Resumen -----
print("Tabla (4 decimales):")
print(tabla.to_string(index=False))
print("\nEntropías en bits:")
for T in temperaturas:
    print(f"  T={T}: {entropias[T]:.4f}")
print("\nCada distribución suma 1.0 y H(0.5) < H(1) < H(2): verificado.")
print("Archivos escritos: resultados.json, t1_softmax_temperatura.png")
