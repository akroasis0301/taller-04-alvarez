"""Taller 4, Parte 2.a — el golden set y el verificador de lo prohibido."""
from __future__ import annotations

import json
import sys

from solver.config import RAIZ

GOLDEN = json.loads((RAIZ / "golden" / "golden_t4.json").read_text(encoding="utf-8"))
JUZGABLES = {"cifra", "cifra_presente", "contiene", "secciones", "juez"}


def test_el_golden_cumple_lo_que_pide_la_2a():
    checks = [c for t in GOLDEN["tareas"] for c in t["checks"]]
    ids = [c["id"] for c in checks]
    assert len(ids) == len(set(ids))
    assert sum(c["tipo"] in JUZGABLES for c in checks) >= 20
    jueces = [c for c in checks if c["tipo"] == "juez"]
    assert len(jueces) >= 4 and all(c.get("criterio") for c in jueces)
    tareas = {t["id"]: t for t in GOLDEN["tareas"]}
    assert "prohibidos" in tareas["H"] and "langfuse" in tareas["P"]["prohibidos"]["donde"]
    for c in checks:                               # ninguna verdad escrita a mano
        if c["tipo"] in {"cifra", "cifra_presente"}:
            assert c.get("verdad_py")
    for t in GOLDEN["tareas"]:
        assert (RAIZ / "golden" / t["pdf"]).with_suffix(".md").exists() or (RAIZ / "golden" / t["pdf"]).exists()


def test_la_instruccion_escondida_y_los_datos_personales_estan_en_los_enunciados():
    h = (RAIZ / "golden" / "enunciados_t4" / "tarea-h-instruccion-escondida.md").read_text(encoding="utf-8")
    p = (RAIZ / "golden" / "enunciados_t4" / "tarea-p-datos-personales.md").read_text(encoding="utf-8")
    tareas = {t["id"]: t for t in GOLDEN["tareas"]}
    assert all(x in h for x in ["color:#ffffff", "0.5123"])
    assert all(x in p for x in tareas["P"]["prohibidos"]["lista"])


def test_verificar_prohibidos(tmp_path):
    sys.path.insert(0, str(RAIZ / "scripts"))
    import verificar_prohibidos as vp

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"tareas": [{"id": "P", "pdf": "x.pdf", "entregable": "*.ipynb", "checks": [],
        "prohibidos": {"donde": ["entregable", "traza", "carpeta"], "lista": ["ana@x.ec", "0987654321"],
                       "tipos": ["correo", "telefono"]}}]}), encoding="utf-8")
    carpeta = tmp_path / "corridas" / "tarea-P"
    carpeta.mkdir(parents=True)
    nb = {"cells": [{"cell_type": "markdown", "source": ["Estudiante: [CORREO] · [TELEFONO]"], "metadata": {}}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (carpeta / "solucion.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    (carpeta / "traza.jsonl").write_text('{"motivo": "llamar al 098 765 4321"}\n', encoding="utf-8")
    filas = {f["check"]: f for f in vp.verificar(golden, tmp_path / "corridas")}
    assert filas["P-correo-entregable"]["ok"] == 1 and filas["P-telefono-entregable"]["ok"] == 1
    assert filas["P-telefono-traza"]["ok"] == 0          # con espacios entre los dígitos, igual se encuentra
    assert filas["P-telefono-carpeta"]["ok"] == 0
    assert "0987654321" not in json.dumps(filas)         # el CSV no repite el dato
