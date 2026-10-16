import json
from pathlib import Path

import numpy as np
import pandas as pd


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names) asociados a sus columnas."""

    def __init__(self, values, feature_names):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("values debe ser una matriz bidimensional.")
        if values.shape[1] != len(feature_names):
            raise ValueError(
                "El número de columnas de values no coincide con la cantidad "
                "de nombres de características."
            )
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("Los datos de values no son numéricos.")
        self.values = values
        self.feature_names = tuple(feature_names)

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")


def build_measurements(df):
    """Selecciona gpu_utilization, cpu_utilization y memory_gb (en ese orden),
    las convierte a una matriz NumPy flotante y retorna un ServerMeasurements."""
    values = df[list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def lanza_value_error(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def main():
    # Lectura del CSV con parse_dates, según el enunciado
    df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])

    m = build_measurements(df)
    print(repr(m))

    # Las tres validaciones exigidas deben lanzar ValueError
    validaciones = {
        "no_bidimensional_lanza_valueerror": lanza_value_error(
            lambda: ServerMeasurements(np.zeros(5), ("a",))
        ),
        "columnas_distintas_de_nombres_lanza_valueerror": lanza_value_error(
            lambda: ServerMeasurements(np.zeros((4, 3)), ("a", "b"))
        ),
        "no_numericos_lanza_valueerror": lanza_value_error(
            lambda: ServerMeasurements(
                np.array([["a", "b", "c"], ["d", "e", "f"]]), ("a", "b", "c")
            )
        ),
    }

    # Comprobaciones que el propio script garantiza
    assert m.values.ndim == 2
    assert m.values.shape[1] == 3
    assert np.issubdtype(m.values.dtype, np.floating)
    assert m.feature_names == FEATURES
    assert all(validaciones.values())

    estadisticas = {
        nombre: {
            "media": float(m.values[:, i].mean()),
            "desv_std": float(m.values[:, i].std()),
            "min": float(m.values[:, i].min()),
            "max": float(m.values[:, i].max()),
        }
        for i, nombre in enumerate(m.feature_names)
    }

    resultados = {
        "repr": repr(m),
        "shape": [int(m.values.shape[0]), int(m.values.shape[1])],
        "feature_names": list(m.feature_names),
        "dtype": str(m.values.dtype),
        "n_filas": int(m.values.shape[0]),
        "n_columnas": int(m.values.shape[1]),
        "validaciones_valueerror": validaciones,
        "estadisticas_descriptivas": estadisticas,
        "supuestos": [
            "La validación de datos numéricos se hace sobre el dtype de la matriz "
            "(np.issubdtype(dtype, np.number)); un array de dtype object se rechaza.",
            "El CSV se lee con parse_dates=['timestamp']; la columna timestamp no "
            "forma parte de la matriz de características.",
            "output/server_analysis.parquet es entregable de una subtarea posterior "
            "(tabla_resumen_servidores) y no se genera en T1.",
        ],
    }

    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Validaciones ValueError:", validaciones)
    print("Shape:", m.values.shape, "| dtype:", m.values.dtype)
    print("Estadísticas:", json.dumps(estadisticas, indent=2))
    print("resultados.json escrito.")


if __name__ == "__main__":
    main()
