"""Taller 4 — Guardrails del solver (Parte 3). Esta es la primera pieza: los datos personales.

Este módulo no importa nada del solver: lo usan la traza local (traza.py), la máscara de
Langfuse (observabilidad.py) y, en la Parte 3, la entrada y la salida del solver.

Datos personales que se redactan (correo, cédula, teléfono), con su marcador:

  correo     ana.perez@mail.com              → [CORREO]
  cédula     1710034065 · 171003406-5         → [CEDULA]
  teléfono   0991234567 · 099 123 4567 ·      → [TELEFONO]
             +593 99 123 4567 · (02) 234-5678

Lo que NO se toca, y por qué la regla es más estrecha que la del andamiaje (Parte 0.b):

  - Fechas: 2025-01-01, 01/03/2026. Un teléfono ecuatoriano empieza por 0 (o +593) y no
    lleva el año delante; una fecha ISO empieza por el año.
  - Números normales: «tiendas 101 102 103», «845.520,02», un timestamp 1791922743, un hash.
    Una cédula tiene diez dígitos, provincia 01–24 o 30 y tercer dígito 0–5; además, o su
    dígito verificador (módulo 10) cuadra, o va precedida de una palabra que la nombra
    («cédula», «CI», «identificación»). Sin esa doble condición, cualquier código de diez
    dígitos sería una cédula.

Límite declarado: no hay modelo de entidades, así que los NOMBRES de personas se escapan
(«Ana Pérez» pasa tal cual). Ver el informe, Parte 3.
"""
from __future__ import annotations

import re
import unicodedata

CORREO = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# Bordes de un número suelto: sin otro dígito pegado y sin formar un decimal («0,991234567»,
# «1234567890.5»). Una coma o un punto de la prosa («…5678, y») sí cuentan como borde.
_ANTES = r"(?<!\d)(?<!\d[.,])"
_DESPUES = r"(?!\d|[.,]\d)"

# Diez dígitos, o nueve + guion + uno.
CEDULA = re.compile(_ANTES + r"(\d{9})-?(\d)" + _DESPUES)
PALABRA_CEDULA = re.compile(r"(?:c[eé]dula|\bc\.?\s?i\b\.?|identificaci[oó]n|\bdni\b)[^\d\n]{0,25}$",
                            re.IGNORECASE)

# Móvil: 09 + 8 dígitos, o +593 9 + 8 dígitos, con espacios o guiones opcionales.
MOVIL = re.compile(_ANTES + r"(?:\+?593[\s-]?|0)9\d(?:[\s-]?\d){7}" + _DESPUES)
# Fijo: 0[2-7] + 7 dígitos (o +593 [2-7] + 7), con paréntesis, espacios o guiones opcionales.
FIJO = re.compile(_ANTES + r"(?:\+?593[\s-]?|\(?0)[2-7]\)?[\s-]?\d{3}[\s-]?\d{4}" + _DESPUES)

MARCADORES = {"correo": "[CORREO]", "cedula": "[CEDULA]", "telefono": "[TELEFONO]"}


def sin_tildes(texto: str) -> str:
    """Minúsculas y sin tildes: la forma en que los guardrails comparan texto."""
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def cedula_valida(digitos: str) -> bool:
    """Dígito verificador de la cédula ecuatoriana (módulo 10, coeficientes 2-1-2-1…)."""
    if len(digitos) != 10 or not digitos.isdigit():
        return False
    provincia, tercero = int(digitos[:2]), int(digitos[2])
    if not (1 <= provincia <= 24 or provincia == 30) or tercero > 5:
        return False
    suma = 0
    for i, d in enumerate(digitos[:9]):
        v = int(d) * (2 if i % 2 == 0 else 1)
        suma += v - 9 if v > 9 else v
    return (10 - suma % 10) % 10 == int(digitos[9])


def _forma_de_cedula(digitos: str) -> bool:
    provincia, tercero = int(digitos[:2]), int(digitos[2])
    return (1 <= provincia <= 24 or provincia == 30) and tercero <= 5


def _cedulas(texto: str, hallazgos: list[str]) -> str:
    def cambiar(m: re.Match) -> str:
        digitos = m.group(1) + m.group(2)
        if not _forma_de_cedula(digitos):
            return m.group(0)
        nombrada = bool(PALABRA_CEDULA.search(texto[max(0, m.start() - 40):m.start()]))
        if cedula_valida(digitos) or nombrada:
            hallazgos.append("cedula")
            return MARCADORES["cedula"]
        return m.group(0)
    return CEDULA.sub(cambiar, texto)


def redactar_pii(texto: str) -> tuple[str, list[str]]:
    """(texto sin datos personales, tipos encontrados). Nunca lanza.

    El orden importa: el correo primero (sus dígitos no deben leerse como teléfono), luego la
    cédula (una de Guayas empieza por 09, como un móvil: si cuadra como cédula se marca como
    tal; si no, la regla del móvil la redacta igual) y al final los teléfonos."""
    if not isinstance(texto, str) or not texto:
        return texto, []
    hallazgos: list[str] = []

    def marcar(tipo: str):
        def f(_m: re.Match) -> str:
            hallazgos.append(tipo)
            return MARCADORES[tipo]
        return f

    texto = CORREO.sub(marcar("correo"), texto)
    texto = _cedulas(texto, hallazgos)
    texto = MOVIL.sub(marcar("telefono"), texto)
    texto = FIJO.sub(marcar("telefono"), texto)
    return texto, hallazgos
