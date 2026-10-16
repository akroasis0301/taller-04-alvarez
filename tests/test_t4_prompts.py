"""Taller 4, 2.c — el prompt versionado: v1 es exactamente el de la corrida base; v2 lo extiende."""
from __future__ import annotations

from solver import prompts
from solver.agentes import programador


def test_v1_es_el_prompt_de_la_base():
    assert programador.SISTEMA == prompts.PROGRAMADOR_V1 == prompts.programador("v1")


def test_v2_conserva_v1_y_anade_las_cuatro_reglas():
    v2 = prompts.programador("v2")
    assert v2 != prompts.PROGRAMADOR_V1 and v2.count("\n- ") == prompts.PROGRAMADOR_V1.count("\n- ") + 4
    assert v2.endswith("Devuelve SOLO el script, dentro de un bloque ```python ... ```.")
    assert set(prompts.CHANGELOG) == set(prompts.PROGRAMADOR)


def test_el_programador_usa_la_version_activa(monkeypatch):
    s = {"id": "T1", "objetivo": "x", "criterio": "y"}
    monkeypatch.setenv("SOLVER_PROMPT_VERSION", "v2")
    assert "Reglas añadidas en v2" in programador.mensajes(s, {"texto": "t"}, None, None)[0]["content"]
    monkeypatch.setenv("SOLVER_PROMPT_VERSION", "v1")
    assert "Reglas añadidas en v2" not in programador.mensajes(s, {"texto": "t"}, None, None)[0]["content"]
