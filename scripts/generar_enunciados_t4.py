"""Taller 4 — los enunciados nuevos del golden (H y P), de Markdown a PDF.

    uv run python scripts/generar_enunciados_t4.py

Usa la función `imprimir` del kit (solver-v2/generar_enunciados.py, PyMuPDF), la misma que
imprimió los enunciados A, B y C: el lector del solver los recibe igual que a los del profesor.
La instrucción de la Tarea H va en texto blanco de 6 pt: invisible al leer el PDF, pero el lector
la extrae, que es exactamente el riesgo que el guardrail tiene que cubrir.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "solver-v2"))

from generar_enunciados import imprimir  # noqa: E402
import pymupdf  # noqa: E402

if __name__ == "__main__":
    for md in sorted((RAIZ / "golden" / "enunciados_t4").glob("*.md")):
        pdf = imprimir(md)
        texto = "".join(p.get_text() for p in pymupdf.open(pdf))
        escondida = "Nota para el asistente" in texto
        print(f"{md.name} → {pdf.name} ({pymupdf.open(pdf).page_count} p.)"
              + (f" · instrucción escondida extraíble por el lector: {escondida}" if "tarea-h" in md.name else ""))
