"""Taller 4 — todas las tablas del informe, desde los archivos crudos (nada se copia a mano).

    uv run python scripts/tablas_t4.py                 # base y, si existe, mejora
    uv run python scripts/tablas_t4.py --versiones base

Lee: taller04/parte1/spans_<v>.jsonl (exportados de Langfuse), resultados/t4/*.csv (evaluar_solver,
juez_solver, verificar_prohibidos), corridas/t4/<v>/tarea-*/traza.jsonl y traza_id.txt, las
etiquetas humanas y los CSV de guardrails. Escribe taller04/tablas/tablas.md y un CSV por tabla.

Dos decisiones de medición, explicadas en el informe:
  - El tiempo NO se suma entre niveles: un span `llm` vive DENTRO de su agente, y las llamadas del
    indexador corren en paralelo (su suma supera al reloj). El reparto «por agente» usa solo los
    spans `agent` (secuenciales, suman la corrida); el de «tipos de span» se lee dentro de ellos.
  - El costo y los percentiles usan las funciones del kit (evaluar_llmops.precio, _percentil): las
    mismas cifras que resumir_trazas.py.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LAB04 = RAIZ / "taller-04-llmops" / "Lab-04-LLMOps"
TAREAS = ["A", "B", "C", "S1", "H", "P"]
AGENTES = ["lector", "indexador", "planificador", "investigador", "programador", "ejecutor", "critico", "redactor"]


def _kit():
    sys.path.insert(0, str(LAB04))
    spec = importlib.util.spec_from_file_location("evaluar_llmops", LAB04 / "evaluar_llmops.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EV = _kit()


# ============================================================================ utilidades
def leer_csv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def leer_jsonl(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def tabla(cabecera: list[str], filas: list[list]) -> str:
    def c(x):
        if isinstance(x, float):
            return f"{x:,.4f}".rstrip("0").rstrip(".") if abs(x) < 10 else f"{x:,.1f}"
        return str(x)
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "---|" * len(cabecera)]
    lineas += ["| " + " | ".join(c(x) for x in f) + " |" for f in filas]
    return "\n".join(lineas)


def guardar_csv(nombre: str, cabecera: list[str], filas: list[list], destino: Path) -> None:
    with (destino / f"{nombre}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecera)
        w.writerows(filas)


def pct(a, b) -> str:
    return f"{100 * a / b:.1f} %" if b else "—"


def tarea_de_traza(version: str) -> dict[str, str]:
    m = {}
    for t in TAREAS:
        f = RAIZ / "corridas" / "t4" / version / f"tarea-{t}" / "traza_id.txt"
        if f.exists():
            m[f.read_text().strip()] = t
    return m


def resultados(version: str) -> list[dict]:
    corr = RAIZ / "resultados" / "t4" / f"resultados_solver_{version}_corregido.csv"
    return leer_csv(corr if corr.exists() else RAIZ / "resultados" / "t4" / f"resultados_solver_{version}.csv")


# ============================================================================ Parte 1
def parte1(version: str, md: list[str], destino: Path) -> dict:
    spans = leer_jsonl(RAIZ / "taller04" / "parte1" / f"spans_{version}.jsonl")
    if not spans:
        md.append(f"_Sin spans exportados para «{version}»._\n")
        return {}
    a_tarea = tarea_de_traza(version)
    resumen = {r["tarea"]: r for r in leer_csv(RAIZ / "resultados" / "t4" / f"resumen_solver_{version}.csv")}
    p_ent, p_sal, origen = EV.precio("propietario_economico")

    raices = [s for s in spans if s["span"] == "corrida"]
    dur = [s["duracion_ms"] / 1000 for s in raices if s.get("duracion_ms") is not None]
    llm = [s for s in spans if s["span"] == "llm"]
    ent, sal = sum(s.get("tokens_entrada") or 0 for s in llm), sum(s.get("tokens_salida") or 0 for s in llm)
    errores = [s for s in spans if s.get("error") and s["span"] != "corrida"]
    general = [["corridas", len(raices)], ["corridas sin span «corrida»", 0 if len(raices) == len({s['traza'] for s in spans}) else "revisar"],
               ["tokens de entrada", ent], ["tokens de salida (incluye razonamiento)", sal],
               ["costo real (H200)", "0 USD"],
               ["costo de referencia", f"{(ent * p_ent + sal * p_sal) / 1e6:.4f} USD — {origen}"],
               ["latencia p50 por corrida", f"{EV._percentil(dur, 0.50):.1f} s"],
               ["latencia p95 por corrida", f"{EV._percentil(dur, 0.95):.1f} s"],
               ["spans con error (sin la raíz)", len(errores)]]
    md += [f"### Resumen ({version})", "", tabla(["métrica", "valor"], general), ""]
    guardar_csv(f"p1_resumen_{version}", ["metrica", "valor"], general, destino)

    por_tarea = []
    for r in raices:
        t = a_tarea.get(r["traza"], "?")
        ll = [s for s in llm if s["traza"] == r["traza"]]
        por_tarea.append([t, round(r["duracion_ms"] / 1000, 1), len(ll), sum(s.get("tokens_entrada") or 0 for s in ll),
                          sum(s.get("tokens_salida") or 0 for s in ll), resumen.get(t, {}).get("status", ""),
                          r["traza"][:12]])
    por_tarea.sort(key=lambda f: TAREAS.index(f[0]) if f[0] in TAREAS else 99)
    cab = ["tarea", "duración (s)", "llamadas al modelo", "tokens entrada", "tokens salida", "status", "traza_id"]
    md += [f"### Por corrida ({version})", "", tabla(cab, por_tarea), ""]
    guardar_csv(f"p1_por_tarea_{version}", cab, por_tarea, destino)

    ag = defaultdict(lambda: {"n": 0, "ms": 0.0, "ent": 0, "sal": 0, "llamadas": 0})
    for s in spans:
        if s["span"] == "agent":
            ag[s["nombre"]]["n"] += 1
            ag[s["nombre"]]["ms"] += s.get("duracion_ms") or 0
        if s["span"] == "llm":
            ag[s["nombre"]]["ent"] += s.get("tokens_entrada") or 0
            ag[s["nombre"]]["sal"] += s.get("tokens_salida") or 0
            ag[s["nombre"]]["llamadas"] += 1
    t_ag = sum(v["ms"] for v in ag.values()) or 1
    tok = (ent + sal) or 1
    filas = sorted(([k, v["n"], round(v["ms"] / 1000, 1), pct(v["ms"], t_ag), round(v["ms"] / v["n"] / 1000, 1) if v["n"] else 0,
                     v["llamadas"], v["ent"], v["sal"], pct(v["ent"] + v["sal"], tok)] for k, v in ag.items() if k in AGENTES),
                   key=lambda f: -f[2])
    cab = ["agente", "spans", "tiempo total (s)", "parte del tiempo", "tiempo medio (s)", "llamadas", "tokens entrada",
           "tokens salida", "parte de los tokens"]
    md += [f"### Por agente ({version}) — el cuello de botella es el de mayor TIEMPO TOTAL", "", tabla(cab, filas), "",
           f"Suma de los spans de agente: {t_ag / 1000:,.1f} s; suma de las corridas: {sum(dur):,.1f} s "
           "(los agentes son secuenciales: deben coincidir).", ""]
    guardar_csv(f"p1_por_agente_{version}", cab, filas, destino)

    tipos = defaultdict(lambda: [0, 0.0])
    for s in spans:
        if s["span"] not in {"corrida", "agent", "chain"}:
            tipos[s["span"]][0] += 1
            tipos[s["span"]][1] += s.get("duracion_ms") or 0
    filas = sorted(([k, v[0], round(v[1] / 1000, 1), round(v[1] / v[0], 1) if v[0] else 0, pct(v[1], sum(dur) * 1000)]
                    for k, v in tipos.items()), key=lambda f: -f[2])
    cab = ["tipo de span", "spans", "tiempo total (s)", "medio (ms)", "respecto del reloj de las corridas"]
    md += [f"### Por tipo de span, dentro de los agentes ({version})", "", tabla(cab, filas), "",
           "Los `llm` pueden sumar más que el reloj: el indexador llama al modelo en paralelo (8 hilos).", ""]
    guardar_csv(f"p1_por_tipo_{version}", cab, filas, destino)
    cuello = max(ag.items(), key=lambda kv: kv[1]["ms"] if kv[0] in AGENTES else -1)
    return {"dur": dur, "ent_por_corrida": [f[3] + f[4] for f in por_tarea], "cuello": cuello[0],
            "cuello_parte": cuello[1]["ms"] / t_ag, "spans": spans}


# ============================================================================ Parte 2.b
def metricas_det(filas: list[dict]) -> list[list]:
    por = defaultdict(lambda: [0, 0])
    for r in filas:
        if r["tipo"] == "juez":
            continue
        por[r["tipo"]][0] += 1
        por[r["tipo"]][1] += r["ok"] == "1"
    return [[k, v[0], v[1], pct(v[1], v[0])] for k, v in sorted(por.items())]


def kappa_entre(a: dict[str, str], b: dict[str, str]) -> dict:
    """Acuerdo y kappa entre dos calificadores aprueba/falla, con la fórmula del kit."""
    filas = [{"id": k, "juez_veredicto": v} for k, v in b.items()]
    return EV.calibrar(filas, a)


def parte2b(versiones: list[str], md: list[str], destino: Path) -> None:
    for v in versiones:
        filas = resultados(v)
        if not filas:
            continue
        m = metricas_det(filas)
        tot = [sum(f[1] for f in m), sum(f[2] for f in m)]
        md += [f"### Verificador automático por tipo de comprobación ({v}, golden corregido)", "",
               tabla(["tipo", "n", "aprobadas", "tasa"], m + [["**total (sin «juez»)**", tot[0], tot[1], pct(tot[1], tot[0])]]), ""]
        guardar_csv(f"p2b_deterministico_{v}", ["tipo", "n", "aprobadas", "tasa"], m, destino)
        pt = defaultdict(lambda: [0, 0])
        for r in filas:
            if r["tipo"] != "juez":
                pt[r["tarea"]][0] += 1
                pt[r["tarea"]][1] += r["ok"] == "1"
        proh = leer_csv(RAIZ / "resultados" / "t4" / f"prohibidos_{v}.csv")
        ph = defaultdict(lambda: [0, 0])
        for r in proh:
            ph[r["tarea"]][0] += 1
            ph[r["tarea"]][1] += r["ok"] == "1"
        f2 = [[t, pt[t][1], pt[t][0], f"{ph[t][1]}/{ph[t][0]}" if t in ph else "—"] for t in TAREAS if t in pt]
        md += [f"### Por tarea ({v})", "", tabla(["tarea", "aprobadas", "comprobaciones", "prohibidos ausentes"], f2), ""]
        guardar_csv(f"p2b_por_tarea_{v}", ["tarea", "aprobadas", "comprobaciones", "prohibidos_ausentes"], f2, destino)

        juez = leer_csv(RAIZ / "resultados" / "t4" / f"juez_solver_{v}.csv")
        if not juez:
            continue
        pj = defaultdict(lambda: Counter())
        for r in juez:
            pj[r["tipo"]][r["juez_veredicto"]] += 1
            pj[r["tipo"]]["_p"] += int(r["juez_puntaje"] or 0)
        fj = [[k, sum(c[x] for x in c if x != "_p"), c["aprueba"], c["falla"], c["sin_parsear"] + c["sin_entregable"],
               round(c["_p"] / max(1, sum(c[x] for x in c if x != "_p")), 2)] for k, c in sorted(pj.items())]
        md += [f"### Juez ({juez[0]['juez_modelo']}, prompt {juez[0]['juez_version']}) por tipo ({v})", "",
               tabla(["tipo", "n", "aprueba", "falla", "sin respuesta", "puntaje medio"], fj), ""]
        guardar_csv(f"p2b_juez_{v}", ["tipo", "n", "aprueba", "falla", "sin_respuesta", "puntaje_medio"], fj, destino)

        jv = {r["id"]: r["juez_veredicto"] for r in juez if r["juez_veredicto"] in {"aprueba", "falla"}}
        det_corr = {r["check"]: ("aprueba" if r["ok"] == "1" else "falla") for r in filas
                    if r["tipo"] != "juez" and r["check"] in jv}
        cal = [["juez vs. verificador (golden corregido)", *_cal(kappa_entre(det_corr, jv))]]
        if v == "base":
            orig = leer_csv(RAIZ / "resultados" / "t4" / "resultados_solver_base.csv")
            det_orig = {r["check"]: ("aprueba" if r["ok"] == "1" else "falla") for r in orig
                        if r["tipo"] != "juez" and r["check"] in jv}
            cal.insert(0, ["juez vs. verificador (golden ORIGINAL)", *_cal(kappa_entre(det_orig, jv))])
            etq = EV.leer_etiquetas(RAIZ / "taller04" / "parte2" / "etiquetas.csv")
            cal.append(["juez vs. persona (etiquetas.csv)", *_cal(kappa_entre(etq, jv))])
            det_todo = {r["check"]: ("aprueba" if r["ok"] == "1" else "falla") for r in filas if r["tipo"] != "juez"}
            det_o = {r["check"]: ("aprueba" if r["ok"] == "1" else "falla") for r in orig if r["tipo"] != "juez"}
            cal.append(["verificador (corregido) vs. persona", *_cal(kappa_entre(etq, {k: x for k, x in det_todo.items() if k in etq}))])
            cal.append(["verificador (original) vs. persona", *_cal(kappa_entre(etq, {k: x for k, x in det_o.items() if k in etq}))])
            cambios = [[k, det_o[k], det_todo[k]] for k in det_o if det_o[k] != det_todo.get(k)]
            md += ["### Correcciones del golden (base re-evaluada sin volver a correr el solver)", "",
                   tabla(["comprobación", "golden original", "golden corregido"], cambios or [["—", "", ""]]), "",
                   "A06/H06: el verificador original aprobaba **por accidente** (tomaba el 0.9474 de la fila de 39 "
                   "ejemplos); la revisión humana decidió que la estandarización aplica a ambos modelos y la verdad pasó "
                   "a ser el NB estandarizado (0.9357). S103: falso negativo del verificador (regla de longitud del "
                   "encabezado). Ver `_correccion` en golden/golden_t4.json.", ""]
        cab = ["calibración", "pares", "acuerdo", "kappa", "discrepancias"]
        md += [f"### Calibración del juez ({v})", "", tabla(cab, cal), ""]
        guardar_csv(f"p2b_calibracion_{v}", cab, cal, destino)


def _cal(c: dict) -> list:
    return [c.get("pares", 0), c.get("acuerdo", "—"), c.get("kappa", "—"), ", ".join(c.get("discrepancias", [])) or "—"]


# ============================================================================ Parte 2.c
def de_la_traza(version: str, tarea: str) -> dict:
    F = leer_jsonl(RAIZ / "corridas" / "t4" / version / f"tarea-{tarea}" / "traza.jsonl")
    prog = [f for f in F if f["tipo"] == "llamada" and f["agente"] == "programador"]
    return {"prog_llamadas": len(prog), "prog_salida": sum(f.get("tokens_salida") or 0 for f in prog),
            "prog_vacias": sum("vacío" in (f.get("error") or "") for f in prog),
            "rechazos": sum(f["tipo"] == "decision" and f.get("decision") == "rechazado" for f in F),
            "ejec_error": sum(f["tipo"] == "ejecucion" and bool(f.get("error")) for f in F)}


def parte2c(md: list[str], destino: Path) -> None:
    rb = {r["tarea"]: r for r in leer_csv(RAIZ / "resultados" / "t4" / "resumen_solver_base.csv")}
    rm = {r["tarea"]: r for r in leer_csv(RAIZ / "resultados" / "t4" / "resumen_solver_mejora.csv")}
    if not rm:
        md.append("_Aún no hay corrida mejorada._\n")
        return
    db, dm = resultados("base"), resultados("mejora")
    ap = lambda filas, t: (sum(r["ok"] == "1" for r in filas if r["tarea"] == t and r["tipo"] != "juez"),  # noqa: E731
                           sum(1 for r in filas if r["tarea"] == t and r["tipo"] != "juez"))
    filas, tot = [], defaultdict(lambda: [0, 0])
    for t in TAREAS:
        if t not in rb or t not in rm:
            continue
        tb, tm = de_la_traza("base", t), de_la_traza("mejora", t)
        (ab, nb), (am, nm) = ap(db, t), ap(dm, t)
        fila = [t, f"{ab}/{nb} → {am}/{nm}",
                f"{float(rb[t]['duracion_s']):.0f} → {float(rm[t]['duracion_s']):.0f}",
                f"{int(rb[t]['tokens_salida']):,} → {int(rm[t]['tokens_salida']):,}",
                f"{tb['prog_llamadas']} → {tm['prog_llamadas']}", f"{tb['prog_salida']:,} → {tm['prog_salida']:,}",
                f"{tb['prog_vacias']} → {tm['prog_vacias']}", f"{tb['rechazos']} → {tm['rechazos']}",
                f"{tb['ejec_error']} → {tm['ejec_error']}"]
        filas.append(fila)
        for k, (x, y) in {"dur": (float(rb[t]["duracion_s"]), float(rm[t]["duracion_s"])),
                          "sal": (int(rb[t]["tokens_salida"]), int(rm[t]["tokens_salida"])),
                          "prog_sal": (tb["prog_salida"], tm["prog_salida"]), "vacias": (tb["prog_vacias"], tm["prog_vacias"]),
                          "rech": (tb["rechazos"], tm["rechazos"]), "ejec": (tb["ejec_error"], tm["ejec_error"]),
                          "llam": (tb["prog_llamadas"], tm["prog_llamadas"])}.items():
            tot[k][0] += x
            tot[k][1] += y
    ab, am = sum(r["ok"] == "1" for r in db if r["tipo"] != "juez"), sum(r["ok"] == "1" for r in dm if r["tipo"] != "juez")
    nb, nm = sum(r["tipo"] != "juez" for r in db), sum(r["tipo"] != "juez" for r in dm)
    filas.append(["**total**", f"{ab}/{nb} → {am}/{nm}", f"{tot['dur'][0]:.0f} → {tot['dur'][1]:.0f}",
                  f"{tot['sal'][0]:,} → {tot['sal'][1]:,}", f"{tot['llam'][0]} → {tot['llam'][1]}",
                  f"{tot['prog_sal'][0]:,} → {tot['prog_sal'][1]:,}", f"{tot['vacias'][0]} → {tot['vacias'][1]}",
                  f"{tot['rech'][0]} → {tot['rech'][1]}", f"{tot['ejec'][0]} → {tot['ejec'][1]}"])
    cab = ["tarea", "comprobaciones", "duración (s)", "tokens salida", "llamadas programador",
           "tokens salida programador", "vacías por longitud", "rechazos del crítico", "ejecuciones con error"]
    md += ["### Antes (v1, base) → después (v2, mejora): mismo golden, verificador, juez y modelo", "", tabla(cab, filas), ""]
    guardar_csv("p2c_antes_despues", cab, filas, destino)

    ob = {r["check"]: r for r in db}
    cambios = [[r["check"], r["tipo"], ob[r["check"]]["ok"], r["ok"], r["detalle"][:90]] for r in dm
               if r["check"] in ob and ob[r["check"]]["ok"] != r["ok"] and r["tipo"] != "juez"]
    jb = {r["id"]: r["juez_veredicto"] for r in leer_csv(RAIZ / "resultados" / "t4" / "juez_solver_base.csv")}
    jm = {r["id"]: r["juez_veredicto"] for r in leer_csv(RAIZ / "resultados" / "t4" / "juez_solver_mejora.csv")}
    cambios += [[k, "juez", jb[k], jm[k], "veredicto del juez"] for k in jm if k in jb and jb[k] != jm[k]]
    md += ["### Comprobaciones que cambiaron entre las dos corridas", "",
           tabla(["comprobación", "tipo", "base", "mejora", "detalle (mejora)"], cambios or [["ninguna", "", "", "", ""]]), ""]
    guardar_csv("p2c_cambios", ["comprobacion", "tipo", "base", "mejora", "detalle"], cambios, destino)


# ============================================================================ 2.c — peores casos
PEORES = [("S1", ["T4", "T5"], "dos llamadas del programador vacías por longitud y la T5 rechazada"),
          ("C", ["T3"], "una llamada del programador vacía por longitud"),
          ("H", ["T1", "T3"], "dos scripts que fallaron al ejecutarse")]


def _corto(x, n=220) -> str:
    s = " ".join(str(x or "").split())
    return (s[:n] + "…") if len(s) > n else s


def peores_casos(version: str, md: list[str], destino: Path) -> None:
    """Los spans de los peores casos, desde la traza local (enmascarada, igual que en Langfuse)."""
    for tarea, subtareas, porque in PEORES:
        carpeta = RAIZ / "corridas" / "t4" / version / f"tarea-{tarea}"
        F = leer_jsonl(carpeta / "traza.jsonl")
        if not F:
            continue
        tid = (carpeta / "traza_id.txt").read_text().strip() if (carpeta / "traza_id.txt").exists() else "—"
        filas = []
        for f in F:
            if f["tipo"] == "decision" and f.get("agente") == "guardrail":
                filas.append([f["seq"], "guardrail", f["decision"], "", "", _corto(f.get("motivo"))])
            if f.get("subtarea") not in subtareas:
                continue
            if f["tipo"] == "llamada":
                filas.append([f["seq"], "llm", f"{f['agente']} · intento {f.get('intento')}",
                              f"{f.get('tokens_entrada')}+{f.get('tokens_salida')} · {f.get('latencia_s')} s · fin={f.get('fin')}",
                              _corto(f.get("error"), 80) or "—", _corto(f.get("salida"), 160) or "(vacío)"])
            elif f["tipo"] == "ejecucion":
                filas.append([f["seq"], "herramienta", f"sandbox · intento {f.get('intento')}",
                              f"returncode={f.get('returncode')} · {f.get('duracion_s')} s", _corto(f.get("error"), 120) or "—", ""])
            elif f["tipo"] == "decision":
                filas.append([f["seq"], "decisión", f"{f['agente']}: {f.get('decision')}", "", "", _corto(f.get("motivo"))])
        md += [f"### Peor caso: tarea {tarea}, subtareas {', '.join(subtareas)} ({version}) — {porque}", "",
               f"traza_id en Langfuse: `{tid}`", "",
               tabla(["#", "span", "quién / qué", "medida", "error", "qué respondió o decidió"], filas), ""]
        guardar_csv(f"p2c_peor_{tarea}_{version}", ["seq", "span", "que", "medida", "error", "respuesta"], filas, destino)


# ============================================================================ Parte 3
def parte3(versiones: list[str], md: list[str], destino: Path, spans_por_v: dict) -> None:
    filas = []
    for nombre, archivo in [("nuestros, banco (26)", "guardrails_nuestros.csv"),
                            ("ejemplo del kit, banco (26)", "guardrails_ejemplo.csv"),
                            ("nuestros, fuera del banco (12)", "guardrails_fuera_del_banco.csv")]:
        r = leer_csv(RAIZ / "taller04" / "parte3" / archivo)
        if not r:
            continue
        c = Counter(x["veredicto"] for x in r)
        ataques = [x for x in r if x["esperado"] != "pasar"]
        legit = [x for x in r if x["esperado"] == "pasar"]
        fn = sum(x["veredicto"] == "falso_negativo" for x in ataques)
        fp = sum(x["veredicto"] == "falso_positivo" for x in legit)
        filas.append([nombre, len(r), c["correcto"], f"{fn}/{len(ataques)}", f"{fp}/{len(legit)}", c["otra_accion"],
                      ", ".join(x["id"] for x in r if x["veredicto"] != "correcto") or "—"])
    cab = ["validadores y banco", "casos", "correctos", "falsos negativos", "falsos positivos", "otra acción", "errores"]
    md += ["### Guardrails como clasificadores", "", tabla(cab, filas), ""]
    guardar_csv("p3_bancos", cab, filas, destino)

    for v in versiones:
        spans = spans_por_v.get(v) or []
        g = defaultdict(list)
        for s in spans:
            if s["span"] == "guardrail":
                g[s["nombre"]].append(s.get("duracion_ms") or 0)
        decis = Counter()
        for t in TAREAS:
            for f in leer_jsonl(RAIZ / "corridas" / "t4" / v / f"tarea-{t}" / "traza.jsonl"):
                if f["tipo"] == "decision" and f.get("agente") == "guardrail":
                    decis[(f["decision"], f.get("motivo"))] += 1
                if f["tipo"] == "ejecucion" and (f.get("error") or "").startswith("guarda:"):
                    decis[("guarda_ast", "bloqueada")] += 1
        fg = [[k, len(x), round(sum(x) / len(x), 2), max(x),
               "; ".join(f"{d}: {n}" for (nom, d), n in sorted(decis.items()) if nom == k) or ("ok" if k != "guarda_ast" else "—")]
              for k, x in sorted(g.items())]
        md += [f"### Guardrails en el agente ({v}): costo y decisiones en el golden", "",
               tabla(["guardrail", "spans", "latencia media (ms)", "máx. (ms)", "decisiones distintas de ok"], fg), ""]
        guardar_csv(f"p3_en_el_agente_{v}", ["guardrail", "spans", "ms_medio", "ms_max", "decisiones"], fg, destino)


# ============================================================================ Parte 4
def parte4(datos: dict, md: list[str], destino: Path) -> None:
    filas = []
    for v, d in datos.items():
        if not d:
            continue
        dur, tok = d["dur"], d["ent_por_corrida"]
        filas.append([v, len(dur), f"{EV._percentil(dur, .5):.0f}", f"{EV._percentil(dur, .95):.0f}", f"{max(dur):.0f}",
                      f"{EV._percentil(tok, .5):,.0f}", f"{max(tok):,}", f"{100 / len(dur):.1f} %"])
    cab = ["corrida", "corridas", "latencia p50 (s)", "latencia p95 (s)", "latencia máx. (s)", "tokens p50 por corrida",
           "tokens máx. por corrida", "resolución (1 corrida)"]
    md += ["### Mediciones para los umbrales de las alertas (Parte 4.b)", "", tabla(cab, filas), "",
           "Con n corridas, cada una vale 100/n % de cualquier tasa: un umbral por debajo de esa resolución salta "
           "con el primer caso.", ""]
    guardar_csv("p4_mediciones", cab, filas, destino)


# ============================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versiones", nargs="*", default=["base", "mejora"])
    ap.add_argument("--salida", default=str(RAIZ / "taller04" / "tablas"))
    a = ap.parse_args()
    destino = Path(a.salida)
    destino.mkdir(parents=True, exist_ok=True)
    versiones = [v for v in a.versiones if (RAIZ / "taller04" / "parte1" / f"spans_{v}.jsonl").exists()
                 or (RAIZ / "resultados" / "t4" / f"resumen_solver_{v}.csv").exists()]
    md = ["# Tablas del Taller 04 (generadas por scripts/tablas_t4.py: no editar a mano)", ""]
    partes: dict[str, list[str]] = {}

    def parte(clave: str, titulo: str) -> list[str]:
        partes[clave] = [f"# {titulo}", ""]            # el informe incluye cada parte por separado
        return partes[clave]
    p1 = parte("p1", "Parte 1 — Observabilidad")        # una sola parte con todas las versiones
    datos = {v: parte1(v, p1, destino) for v in versiones}
    parte2b(versiones, parte("p2b", "Parte 2.b — Dos verificadores y un juez"), destino)
    p2c = parte("p2c", "Parte 2.c — Peores casos y mejora medida")
    if "base" in versiones:
        peores_casos("base", p2c, destino)
    parte2c(p2c, destino)
    parte3(versiones, parte("p3", "Parte 3 — Guardrails"), destino, {v: d.get("spans") for v, d in datos.items() if d})
    parte4(datos, parte("p4", "Parte 4 — Datos para las alertas"), destino)
    for clave, lineas in partes.items():
        (destino / f"tablas_{clave}.md").write_text("\n".join(lineas), encoding="utf-8")
        md += ["#" + lineas[0], ""] + lineas[2:]
    (destino / "tablas.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✓ {destino / 'tablas.md'} ({len(versiones)} versiones: {', '.join(versiones)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
