"""Taller 4, Parte 2.c — sube a Langfuse las versiones del prompt del programador, con su changelog.

    uv run python scripts/registrar_prompts.py

Cada versión de solver/prompts.py se registra como una versión del prompt «solver-programador»
con la etiqueta igual a su nombre (v1, v2) y el changelog como commit_message. Las generations del
programador se ligan a la versión con la que corren (solver/observabilidad.prompt_langfuse). Si la
versión ya existe con el mismo texto, no se duplica.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from solver import prompts  # noqa: E402
from solver.config import Config  # noqa: E402,F401  (carga el .env)


def main() -> int:
    from langfuse import get_client
    lf = get_client()
    if not lf.auth_check():
        raise SystemExit("Langfuse no responde o las claves no valen: revisa LANGFUSE_* en .env.")
    nombre = prompts.NOMBRE_LANGFUSE["programador"]
    for version, texto in prompts.PROGRAMADOR.items():
        try:
            actual = lf.get_prompt(nombre, label=version, cache_ttl_seconds=0)
            if actual.prompt == texto:
                print(f"{nombre} [{version}] ya registrado (versión {actual.version}), sin cambios")
                continue
        except Exception:
            pass
        p = lf.create_prompt(name=nombre, type="text", prompt=texto, labels=[version],
                             commit_message=prompts.CHANGELOG[version])
        print(f"{nombre} [{version}] registrado como versión {p.version}: {prompts.CHANGELOG[version][:80]}…")
    lf.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
