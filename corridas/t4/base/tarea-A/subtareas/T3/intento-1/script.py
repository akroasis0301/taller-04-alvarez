# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje: Naive Bayes gaussiano vs. regresión logística.

Sobre la MISMA división 70/30 estratificada de T1 (random_state=42) y los MISMOS
dos modelos de T2 (GaussianNB y LogisticRegression con atributos estandarizados;
el escalador se ajusta solo con los datos de entrenamiento), se entrena con
submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y 100 % del entrenamiento
(random_state=42) y se evalúa cada tamaño SIEMPRE sobre el mismo conjunto de
prueba. Se grafica exactitud en prueba contra número de ejemplos de
entrenamiento (una curva por modelo) y se guarda la figura en PNG.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEMILLA = 42
PORCENTAJES = [5, 10, 25, 50, 100]
FRACCIONES = [p / 100.0 for p in PORCENTAJES]

# Conteos reportados por T1: sirven para validar que usamos EXACTAMENTE la
# misma división 70/30 estratificada (random_state=42).
T1_ESPERADO = {"n_train": 398, "n_test": 171, "n_atributos": 30,
               "train_malignant": 148, "train_benign": 250,
               "test_malignant": 64, "test_benign": 107}


def _valida_division(Xtr, ytr, Xte, yte):
    return (
        Xtr.shape[0] == T1_ESPERADO["n_train"]
        and Xte.shape[0] == T1_ESPERADO["n_test"]
        and Xtr.shape[1] == T1_ESPERADO["n_atributos"]
        and Xte.shape[1] == T1_ESPERADO["n_atributos"]
        and int((ytr == 0).sum()) == T1_ESPERADO["train_malignant"]
        and int((ytr == 1).sum()) == T1_ESPERADO["train_benign"]
        and int((yte == 0).sum()) == T1_ESPERADO["test_malignant"]
        and int((yte == 1).sum()) == T1_ESPERADO["test_benign"]
    )


def _xy_de_df(df):
    col_y = next(c for c in ("target", "y", "label", "clase", "target_name")
                 if c in df.columns)
    y = df[col_y]
    if not np.issubdtype(y.dtype, np.number):
        y = y.map({"malignant": 0, "benign": 1})  # 0 = malignant, 1 = benign
    y = y.astype(int).to_numpy()
    X = (df.drop(columns=[col_y])
           .select_dtypes(include=[np.number])
           .to_numpy(dtype=float))
    return X, y


def obtener_division_t1():
    """Recupera la división 70/30 estratificada (random_state=42) fijada en T1.

    Intenta cargar los archivos que T1 dejó guardados; si no están disponibles
    o no coinciden con los conteos reportados por T1, reconstruye la división
    de forma determinista (load_breast_cancer + train_test_split(test_size=0.3,
    stratify=y, random_state=42)), que la reproduce exactamente.
    """
    # 1) split.npz de T1
    for base in (Path("entrada/T1"), Path(".")):
        ruta = base / "datos" / "split.npz"
        if ruta.exists():
            try:
                z = np.load(ruta, allow_pickle=False)
                if {"X_train", "y_train", "X_test", "y_test"} <= set(z.files):
                    Xtr, ytr = z["X_train"].astype(float), z["y_train"].astype(int)
                    Xte, yte = z["X_test"].astype(float), z["y_test"].astype(int)
                    if _valida_division(Xtr, ytr, Xte, yte):
                        return Xtr, ytr, Xte, yte, f"archivos de T1: {ruta}"
            except Exception:
                pass
    # 2) parquet train/test de T1
    for base in (Path("entrada/T1"), Path(".")):
        rt, re_ = base / "datos" / "train.parquet", base / "datos" / "test.parquet"
        if rt.exists() and re_.exists():
            try:
                Xtr, ytr = _xy_de_df(pd.read_parquet(rt))
                Xte, yte = _xy_de_df(pd.read_parquet(re_))
                if _valida_division(Xtr, ytr, Xte, yte):
                    return Xtr, ytr, Xte, yte, f"archivos de T1: {rt} y {re_}"
            except Exception:
                pass
    # 3) reconstrucción determinista (idéntica a la división de T1)
    datos = load_breast_cancer()
    X, y = datos.data.astype(float), datos.target.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEMILLA
    )
    origen = ("reconstrucción determinista de la división de T1: "
              "load_breast_cancer + train_test_split(test_size=0.3, "
              "stratify=y, random_state=42)")
    assert _valida_division(Xtr, ytr, Xte, yte), \
        "La división reconstruida no coincide con los conteos de T1"
    return Xtr, ytr, Xte, yte, origen


def fabrica_modelos():
    """Los mismos dos modelos (y mismos hiperparámetros) que en T2."""
    return {
        "naive_bayes_gaussiano": ("Naive Bayes gaussiano", GaussianNB()),
        "regresion_logistica": ("Regresión logística",
                                LogisticRegression(max_iter=5000,
                                                   random_state=SEMILLA)),
    }


def main():
    X_train, y_train, X_test, y_test, origen = obtener_division_t1()
    print(f"División usada ({origen})")
    print(f"  train: {X_train.shape[0]} ejemplos | "
          f"test: {X_test.shape[0]} ejemplos | atributos: {X_train.shape[1]}")

    filas = []
    for pct, frac in zip(PORCENTAJES, FRACCIONES):
        # Submuestra ESTRATIFICADA del entrenamiento (random_state=42).
        # El 100 % corresponde al entrenamiento completo.
        if frac >= 1.0:
            X_sub, y_sub = X_train, y_train
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train,
                train_size=frac, stratify=y_train, random_state=SEMILLA,
            )
        n_sub = int(len(y_sub))

        for clave, (nombre, estimador) in fabrica_modelos().items():
            # Mismo esquema de T2: escalador ajustado SOLO con los datos de
            # entrenamiento de cada punto (la submuestra). El conjunto de
            # prueba no se usa para ajustar nada.
            pipe = Pipeline([
                ("escalador", StandardScaler()),
                ("modelo", clone(estimador)),
            ])
            pipe.fit(X_sub, y_sub)
            y_pred = pipe.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            f1m = float(f1_score(y_test, y_pred, average="macro"))
            filas.append({
                "modelo_clave": clave,
                "modelo": nombre,
                "porcentaje_entrenamiento": pct,
                "n_train": n_sub,
                "accuracy_prueba": acc,
                "f1_macro_prueba": f1m,
            })
            print(f"  {pct:>3}% (n={n_sub:>3}) | {nombre:<22} "
                  f"accuracy={acc:.4f}  f1_macro={f1m:.4f}")

    tabla = pd.DataFrame(filas)
    tabla.to_csv("t3_curva_aprendizaje.csv", index=False)

    # ------------------------------ Figura ------------------------------
    tamanos = sorted(int(t) for t in tabla["n_train"].unique())
    estilos = {
        "naive_bayes_gaussiano": {"color": "tab:blue", "marker": "o", "dy": -14},
        "regresion_logistica":   {"color": "tab:orange", "marker": "s", "dy": 9},
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for clave, est in estilos.items():
        sub = tabla[tabla["modelo_clave"] == clave].sort_values("n_train")
        ax.plot(sub["n_train"], sub["accuracy_prueba"],
                marker=est["marker"], color=est["color"], linewidth=2,
                label=str(sub["modelo"].iloc[0]))
        for _, r in sub.iterrows():
            ax.annotate(f"{r['accuracy_prueba']:.3f}",
                        (r["n_train"], r["accuracy_prueba"]),
                        textcoords="offset points", xytext=(0, est["dy"]),
                        ha="center", fontsize=8, color=est["color"])
    ax.set_xticks(tamanos)
    ax.set_xticklabels([str(t) for t in tamanos])
    ax.set_xlabel("Número de ejemplos de entrenamiento")
    ax.set_ylabel("Exactitud en prueba (accuracy)")
    ax.set_title("Curva de aprendizaje: NB gaussiano vs. regresión logística\n"
                 "(división 70/30 de T1; evaluación siempre sobre el mismo conjunto de prueba)")
    y_lo = min(0.5, float(tabla["accuracy_prueba"].min()) - 0.05)
    ax.set_ylim(y_lo, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("t3_curva_aprendizaje.png", dpi=120)
    plt.close(fig)

    # -------------------------- resultados.json --------------------------
    def puntos(clave):
        sub = tabla[tabla["modelo_clave"] == clave].sort_values("n_train")
        return [{
            "porcentaje_entrenamiento": int(r["porcentaje_entrenamiento"]),
            "n_train": int(r["n_train"]),
            "accuracy_prueba": float(r["accuracy_prueba"]),
            "f1_macro_prueba": float(r["f1_macro_prueba"]),
        } for _, r in sub.iterrows()]

    claves = list(fabrica_modelos().keys())
    accuracy_matriz = {"n_train": tamanos}
    for clave in claves:
        accuracy_matriz[clave] = [
            float(tabla[(tabla["modelo_clave"] == clave) &
                        (tabla["n_train"] == t)]["accuracy_prueba"].iloc[0])
            for t in tamanos
        ]

    ganador_por_tamano = []
    for t in tamanos:
        f_nb = tabla[(tabla["modelo_clave"] == "naive_bayes_gaussiano") &
                     (tabla["n_train"] == t)].iloc[0]
        f_lr = tabla[(tabla["modelo_clave"] == "regresion_logistica") &
                     (tabla["n_train"] == t)].iloc[0]
        ganador_por_tamano.append({
            "n_train": int(t),
            "mejor_modelo": ("naive_bayes_gaussiano"
                             if f_nb["accuracy_prueba"] >= f_lr["accuracy_prueba"]
                             else "regresion_logistica"),
            "diferencia_accuracy": float(abs(f_nb["accuracy_prueba"] -
                                             f_lr["accuracy_prueba"])),
        })

    resultados = {
        "subtarea": "T3",
        "descripcion": ("Curva de aprendizaje sobre la división 70/30 estratificada de T1 "
                        "(random_state=42): Naive Bayes gaussiano y regresión logística de T2 "
                        "entrenados con submuestras estratificadas del 5 %, 10 %, 25 %, 50 % y "
                        "100 % del entrenamiento (random_state=42) y evaluados siempre sobre el "
                        "mismo conjunto de prueba."),
        "division_usada": {
            "origen": origen,
            "test_size": 0.3,
            "estratificada_por_clase": True,
            "random_state": SEMILLA,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_atributos": int(X_train.shape[1]),
        },
        "preprocesamiento": {
            "escalador": "StandardScaler",
            "ajustado_solo_con": ("la submuestra de entrenamiento de cada punto "
                                  "(nunca con el conjunto de prueba)"),
            "aplicado_a": ["Naive Bayes gaussiano", "Regresión logística"],
        },
        "submuestreo": {
            "metodo": ("train_test_split(train_size=fraccion, stratify=y_train, "
                       "random_state=42)"),
            "porcentajes": PORCENTAJES,
            "n_train_por_porcentaje": {str(p): int(t)
                                       for p, t in zip(PORCENTAJES, tamanos)},
        },
        "modelos": {
            "naive_bayes_gaussiano": {"clase": "sklearn.naive_bayes.GaussianNB",
                                      "hiperparametros": {}},
            "regresion_logistica": {"clase": "sklearn.linear_model.LogisticRegression",
                                    "hiperparametros": {"max_iter": 5000,
                                                        "random_state": SEMILLA}},
        },
        "tamanos_train": tamanos,
        "accuracy_prueba_por_tamano": accuracy_matriz,
        "curva_aprendizaje": {clave: puntos(clave) for clave in claves},
        "modelo_ganador_por_tamano": ganador_por_tamano,
        "archivos_generados": {
            "figura": "t3_curva_aprendizaje.png",
            "tabla_csv": "t3_curva_aprendizaje.csv",
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\nResumen T3 — exactitud en prueba por tamaño de entrenamiento:")
    pivote = tabla.pivot(index="n_train", columns="modelo",
                         values="accuracy_prueba")
    print(pivote.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nArchivos generados: t3_curva_aprendizaje.png, "
          "t3_curva_aprendizaje.csv, resultados.json")


if __name__ == "__main__":
    main()
