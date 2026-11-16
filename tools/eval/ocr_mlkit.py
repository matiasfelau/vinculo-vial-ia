"""Compara ML Kit (medido en un celular por el Frontend) con PARSeq sobre el set etiquetado de OCR.

Cruza por id `resultados/ocr-results.csv` (id, formato, luz, lectura, candidatos crudos, confianza, tiempo;
`--results` acepta varias versiones: `v1=...csv v2=...csv`) con las etiquetas del set (`etiquetas.xlsx` o
`etiquetas.csv`, solo las patentes legibles de auto) y con las lecturas de PARSeq que escribe
`ocr.py --labeled-set` (`readings.csv`). Informa, en total y por luz (día y noche):

- lectura exacta (aproxima U5: al menos 85 % de día y 70 % de noche), sin lectura (el usuario la tipea) y
  lectura equivocada (error silencioso si el usuario no la corrige), y si la patente correcta aparecía entre
  los candidatos crudos aunque la lectura elegida fuera otra;
- lo mismo para PARSeq con la regla del servicio (`OCR_MIN_CONF`, estructura válida), lado a lado;
- el error silencioso: para cada lectura equivocada de ML Kit, R3 (`plate_check`) tomándola como patente
  denunciada contra la lectura de PARSeq de la misma foto: patente distinta (RED con penalización a un usuario
  honesto), difiere en un carácter o ilegible (GREEN);
- el tiempo por recorte (mediana y p95) y, con varias versiones, qué cambió de una a la siguiente: lecturas
  nuevas (cuántas exactas y cuántas equivocadas), perdidas y cambiadas.

No corre ningún modelo. Escribe `mlkit.json` (solo cifras) y `mlkit_rows.csv` (id, luz y clases: nunca el
texto) en `--out-dir`. Uso: ver `tools/eval/README.md`.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ocr_labels
from rapidfuzz.distance import Levenshtein

from vvi_ia.domain.plate import TO_DIGIT, TO_LETTER, PlateCheck, canonicalize, plate_check

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIN_CONF = 0.78
RESULTS_PATH = Path("resultados") / "ocr-results.csv"
READINGS_FILE = "readings.csv"
RESULT_COLUMNS = ("id", "luz", "lectura", "candidatos", "confianza", "tiempo_ms")
ROW_COLUMNS = ("id", "luz", "mlkit", "mlkit_error", "candidato", "parseq", "r3")
LIGHTS = ("dia", "noche")
U5_MIN_PCT = {"dia": 85.0, "noche": 70.0}  # lectura exacta mínima, por luz
EXACT, NONE, WRONG = "exacta", "sin_lectura", "equivocada"
SEPARATOR = "|"
PERCENTILE = 0.95


@dataclass(frozen=True)
class Row:
    """Una patente legible de auto con lo que leyó cada lector. Los textos solo viven en RAM."""

    ident: str
    label: str
    light: str
    mlkit: str  # lectura elegida por ML Kit, canónica ('' si no hubo)
    candidates: tuple[str, ...]  # candidatos crudos, canónicos
    confidence: float | None
    time_ms: float | None
    parseq: str  # lo que leyó PARSeq, canónico
    parseq_conf: float


class ComparisonError(ValueError):
    """Los archivos no se pueden cruzar."""


# --- lectura ---------------------------------------------------------------------------------------


def read_results(path: Path) -> dict[str, dict[str, str]]:
    """`id` → fila del CSV de resultados de ML Kit."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in RESULT_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ComparisonError(f"{path.name} no tiene las columnas {', '.join(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            if row["id"] in rows:
                raise ComparisonError(f"{path.name}: el id {row['id']} está repetido")
            rows[row["id"]] = row
    return rows


def read_parseq(path: Path) -> dict[str, tuple[str, float]]:
    """`id` → (lectura, confianza) de `readings.csv` (lo que escribe `ocr.py --labeled-set`)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: (row["lectura"], float(row["confianza"] or 0)) for row in csv.DictReader(handle)}


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def build_rows(
    set_dir: Path, results: Mapping[str, Mapping[str, str]], parseq: Mapping[str, tuple[str, float]]
) -> list[Row]:
    """Cruza por id las etiquetas (legibles de auto) con ML Kit y PARSeq; falla si no coinciden los ids."""
    labels = [row for row in ocr_labels.read_label_rows(set_dir) if ocr_labels.is_legible_car(row)]
    ids = {row["id"] for row in labels}
    for name, found in (("ML Kit", set(results)), ("PARSeq", set(parseq))):
        if found != ids:
            raise ComparisonError(
                f"los ids de {name} no son los de las patentes legibles de auto del set "
                f"(faltan {len(ids - found)}, sobran {len(found - ids)})"
            )
    rows: list[Row] = []
    for label in labels:
        result = results[label["id"]]
        text, conf = parseq[label["id"]]
        rows.append(
            Row(
                ident=label["id"],
                label=canonicalize(label["texto"]),
                light=label["luz"],
                mlkit=canonicalize(result["lectura"]),
                candidates=tuple(
                    candidate
                    for candidate in (canonicalize(c) for c in result["candidatos"].split(SEPARATOR))
                    if candidate
                ),
                confidence=_number(result["confianza"]),
                time_ms=_number(result["tiempo_ms"]),
                parseq=canonicalize(text),
                parseq_conf=conf,
            )
        )
    return rows


# --- clasificación -----------------------------------------------------------------------------------


def mlkit_outcome(row: Row) -> str:
    if not row.mlkit:
        return NONE
    return EXACT if row.mlkit == row.label else WRONG


def wrong_kind(label: str, reading: str) -> str:
    """Qué tan lejos está una lectura equivocada de la etiqueta, según `plate_check` (sin umbral)."""
    check = plate_check(label, reading, 1.0, 0.0)
    return {
        PlateCheck.MATCH: "igual tras normalizar letra/dígito",
        PlateCheck.FUZZY: "difiere en un carácter",
        PlateCheck.MISMATCH: "patente distinta",
        PlateCheck.UNREADABLE: "sin la estructura de una patente",
    }[check]


def _normalized_equal(label: str, window: str) -> bool:
    """La ventana es la patente al leer cada carácter como dígito o letra según su posición (O/0, I/1...)."""
    return all(
        char.translate(TO_DIGIT if expected.isdigit() else TO_LETTER) == expected
        for char, expected in zip(window, label, strict=True)
    )


def candidate_match(label: str, candidates: Sequence[str]) -> str:
    """Si la patente correcta aparece entre los candidatos crudos (de más a menos estricto).

    `exacta`: un candidato contiene la patente tal cual. `normalizada`: una ventana del largo de la
    patente es la patente al normalizar letra/dígito por posición. `a_un_caracter`: una ventana difiere en
    un solo carácter.
    `no` en otro caso.
    """
    if any(label in candidate for candidate in candidates):
        return "exacta"
    size = len(label)
    windows = [
        candidate[start : start + size]
        for candidate in candidates
        for start in range(len(candidate) - size + 1)
    ]
    if any(_normalized_equal(label, window) for window in windows):
        return "normalizada"
    if any(Levenshtein.distance(label, window) <= 1 for window in windows):
        return "a_un_caracter"
    return "no"


def parseq_outcome(row: Row, min_conf: float) -> str:
    """Con la regla del servicio: ilegible (poca confianza o sin estructura) es "sin lectura" y una lectura
    aceptada que no es exacta es equivocada."""
    check = plate_check(row.label, row.parseq, row.parseq_conf, min_conf)
    if check == PlateCheck.UNREADABLE:
        return NONE
    return EXACT if row.parseq == row.label else WRONG


def r3_of_wrong_reading(row: Row, min_conf: float) -> str:
    """R3 con la lectura equivocada de ML Kit como denunciada y la lectura de PARSeq de la misma foto."""
    return plate_check(row.mlkit, row.parseq, row.parseq_conf, min_conf).value


# --- cifras ------------------------------------------------------------------------------------------


def pct(count: int, total: int) -> float | None:
    return round(100 * count / total, 1) if total else None


def percentile(values: Sequence[float], q: float) -> float:
    """El valor en la posición `q` de los ordenados (índice `floor(q * n)`): el mismo cálculo del Frontend."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def mlkit_block(rows: Sequence[Row]) -> dict[str, Any]:
    outcomes = {row.ident: mlkit_outcome(row) for row in rows}
    counts = Counter(outcomes.values())
    total = len(rows)
    matches = {row.ident: candidate_match(row.label, row.candidates) for row in rows}
    not_exact = [row for row in rows if outcomes[row.ident] != EXACT]
    by_outcome = {name: [row for row in rows if outcomes[row.ident] == name] for name in (EXACT, NONE, WRONG)}

    def confidences(name: str) -> float | None:
        values = [row.confidence for row in by_outcome[name] if row.confidence is not None]
        return round(statistics.median(values), 3) if values else None

    return {
        "recortes": total,
        "lectura_exacta": counts[EXACT],
        "lectura_exacta_pct": pct(counts[EXACT], total),
        "sin_lectura": counts[NONE],
        "sin_lectura_pct": pct(counts[NONE], total),
        "lectura_equivocada": counts[WRONG],
        "lectura_equivocada_pct": pct(counts[WRONG], total),
        "equivocada_detalle": dict(Counter(wrong_kind(row.label, row.mlkit) for row in by_outcome[WRONG])),
        "candidatos": {
            "la_correcta_aparece": sum(m == "exacta" for m in matches.values()),
            "la_correcta_aparece_pct": pct(sum(m == "exacta" for m in matches.values()), total),
            # Contando también la que aparece tras normalizar letra/dígito por posición (lo que hace v2).
            "la_correcta_aparece_normalizada": sum(m in ("exacta", "normalizada") for m in matches.values()),
            "la_correcta_aparece_normalizada_pct": pct(
                sum(m in ("exacta", "normalizada") for m in matches.values()), total
            ),
            "no_exactas": len(not_exact),
            "no_exactas_con_la_correcta_normalizada": sum(
                matches[row.ident] in ("exacta", "normalizada") for row in not_exact
            ),
            "no_exactas_con_la_correcta": sum(matches[row.ident] == "exacta" for row in not_exact),
            "sin_lectura_con_la_correcta": sum(matches[row.ident] == "exacta" for row in by_outcome[NONE]),
            "equivocadas_con_la_correcta": sum(matches[row.ident] == "exacta" for row in by_outcome[WRONG]),
            "no_exactas_con_la_correcta_o_a_un_caracter": sum(
                matches[row.ident] != "no" for row in not_exact
            ),
            "sin_candidatos": sum(not row.candidates for row in rows),
        },
        "confianza_mediana": {EXACT: confidences(EXACT), WRONG: confidences(WRONG)},
    }


def parseq_block(rows: Sequence[Row], min_conf: float) -> dict[str, Any]:
    outcomes = [parseq_outcome(row, min_conf) for row in rows]
    counts = Counter(outcomes)
    total = len(rows)
    wrong = [row for row, outcome in zip(rows, outcomes, strict=True) if outcome == WRONG]
    raw_exact = sum(row.parseq == row.label for row in rows)
    return {
        "recortes": total,
        "lectura_exacta": counts[EXACT],
        "lectura_exacta_pct": pct(counts[EXACT], total),
        "sin_lectura": counts[NONE],
        "sin_lectura_pct": pct(counts[NONE], total),
        "lectura_equivocada": counts[WRONG],
        "lectura_equivocada_pct": pct(counts[WRONG], total),
        "equivocada_detalle": dict(Counter(wrong_kind(row.label, row.parseq) for row in wrong)),
        # Sin aplicar el umbral: el texto es exacto aunque el servicio lo dé por ilegible (poca confianza).
        "texto_exacto_sin_umbral": raw_exact,
        "texto_exacto_sin_umbral_pct": pct(raw_exact, total),
    }


def silent_error(rows: Sequence[Row], min_conf: float) -> dict[str, Any]:
    """R3 con cada lectura equivocada de ML Kit como denunciada y la lectura de PARSeq de la misma foto."""
    wrong = [row for row in rows if mlkit_outcome(row) == WRONG]
    results = Counter(r3_of_wrong_reading(row, min_conf) for row in wrong)
    return {
        "lecturas_equivocadas": len(wrong),
        "patente_distinta_RED": results[PlateCheck.MISMATCH.value],
        "difiere_en_un_caracter_GREEN": results[PlateCheck.FUZZY.value],
        "ilegible_GREEN": results[PlateCheck.UNREADABLE.value],
        "coincide_GREEN": results[PlateCheck.MATCH.value],
        "patente_distinta_RED_pct": pct(results[PlateCheck.MISMATCH.value], len(wrong)),
        "de_las_cuales_parseq_lee_la_correcta": sum(
            row.parseq == row.label and row.parseq_conf >= min_conf for row in wrong
        ),
    }


def timing(rows: Sequence[Row]) -> dict[str, Any]:
    values = [row.time_ms for row in rows if row.time_ms is not None]
    if not values:
        return {"recortes": 0}
    return {
        "recortes": len(values),
        "mediana_ms": round(statistics.median(values), 2),
        "p95_ms": round(percentile(values, PERCENTILE), 2),
        "minimo_ms": round(min(values), 2),
        "maximo_ms": round(max(values), 2),
    }


def u5(report: Mapping[str, Any]) -> dict[str, Any]:
    """Lectura exacta de cada lector contra la aproximación de U5 (85 % de día y 70 % de noche)."""

    def check(blocks: Mapping[str, Any]) -> dict[str, Any]:
        return {
            light: {
                "lectura_exacta_pct": blocks[light]["lectura_exacta_pct"],
                "cumple": (blocks[light]["lectura_exacta_pct"] or 0.0) >= U5_MIN_PCT[light],
            }
            for light in LIGHTS
            if light in blocks
        }

    return {
        "umbral_pct": dict(U5_MIN_PCT),
        "mlkit": {version: check(blocks) for version, blocks in report["mlkit"].items()},
        "parseq": check(report["parseq"]),
    }


def groups_of(rows: Sequence[Row]) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = {"total": list(rows)}
    for light in LIGHTS:
        groups[light] = [row for row in rows if row.light == light]
    return {name: items for name, items in groups.items() if items}


def new_readings(before: Sequence[Row], after: Sequence[Row], min_conf: float) -> dict[str, Any]:
    """Qué cambió de una versión de ML Kit a la siguiente (mismos recortes).

    Lecturas nuevas (antes sin lectura): cuántas son exactas y cuántas equivocadas, y el R3 de estas.
    También las perdidas (antes había lectura) y las cambiadas (otra lectura distinta).
    """
    previous = {row.ident: row for row in before}
    if set(previous) != {row.ident for row in after}:
        raise ComparisonError("las dos versiones no tienen los mismos recortes")
    new = [row for row in after if not previous[row.ident].mlkit and row.mlkit]

    def part(items: Sequence[Row]) -> dict[str, Any]:
        wrong = [row for row in items if mlkit_outcome(row) == WRONG]
        return {
            "lecturas_nuevas": len(items),
            "exactas": sum(mlkit_outcome(row) == EXACT for row in items),
            "equivocadas": len(wrong),
            "equivocada_detalle": dict(Counter(wrong_kind(row.label, row.mlkit) for row in wrong)),
            "r3_de_las_equivocadas": silent_error(items, min_conf),
        }

    return {
        **part(new),
        "perdidas": sum(bool(previous[row.ident].mlkit) and not row.mlkit for row in after),
        "cambiadas": sum(
            bool(previous[row.ident].mlkit) and bool(row.mlkit) and previous[row.ident].mlkit != row.mlkit
            for row in after
        ),
        "por_luz": {light: part([row for row in new if row.light == light]) for light in LIGHTS},
    }


def build_report(versions: Mapping[str, Sequence[Row]], min_conf: float) -> dict[str, Any]:
    """`versions` mapea cada versión de ML Kit a sus filas (mismos recortes y lecturas de PARSeq)."""
    names = list(versions)
    first = versions[names[0]]
    parseq_groups = groups_of(first)
    candidates_equal = all(
        [row.candidates for row in versions[name]] == [row.candidates for row in first] for name in names
    )
    report: dict[str, Any] = {
        "min_conf": min_conf,
        "versiones": names,
        "candidatos_iguales_entre_versiones": candidates_equal,
        "mlkit": {
            name: {g: mlkit_block(items) for g, items in groups_of(rows).items()}
            for name, rows in versions.items()
        },
        "parseq": {g: parseq_block(items, min_conf) for g, items in parseq_groups.items()},
        "error_silencioso": {
            name: {g: silent_error(items, min_conf) for g, items in groups_of(rows).items()}
            for name, rows in versions.items()
        },
        "tiempo": {name: timing(rows) for name, rows in versions.items()},
        "lecturas_nuevas": {
            f"{later}_contra_{earlier}": new_readings(versions[earlier], versions[later], min_conf)
            for earlier, later in zip(names, names[1:], strict=False)
        },
    }
    report["u5"] = u5(report)
    return report


def write_rows(path: Path, versions: Mapping[str, Sequence[Row]], min_conf: float) -> None:
    """Una fila por recorte con las clases de cada versión (nunca el texto)."""
    names = list(versions)
    columns = ["id", "luz", "candidato", "parseq"]
    for name in names:
        columns += [f"mlkit_{name}", f"error_{name}", f"r3_{name}"]
    by_version = {name: {row.ident: row for row in rows} for name, rows in versions.items()}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in versions[names[0]]:
            out = {
                "id": row.ident,
                "luz": row.light,
                "candidato": candidate_match(row.label, row.candidates),
                "parseq": parseq_outcome(row, min_conf),
            }
            for name in names:
                mine = by_version[name][row.ident]
                outcome = mlkit_outcome(mine)
                out[f"mlkit_{name}"] = outcome
                out[f"error_{name}"] = wrong_kind(mine.label, mine.mlkit) if outcome == WRONG else ""
                out[f"r3_{name}"] = r3_of_wrong_reading(mine, min_conf) if outcome == WRONG else ""
            writer.writerow(out)


# --- línea de comandos ----------------------------------------------------------------------------------


def parse_results(values: Sequence[str] | None, set_dir: Path) -> dict[str, Path]:
    """`nombre=ruta` (o solo la ruta: el nombre es el del archivo); por defecto, `v1` con los del set."""
    if not values:
        return {"v1": set_dir / RESULTS_PATH}
    found: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator:
            name, path = Path(value).stem, value
        if name in found:
            raise ComparisonError(f"la versión {name} está repetida")
        found[name] = Path(path)
    return found


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--labeled-set", type=Path, required=True, help="set etiquetado de ocr_set.py")
    parser.add_argument(
        "--results",
        nargs="+",
        metavar="NOMBRE=RUTA",
        help=f"resultados de ML Kit, una o más versiones en orden (por defecto v1=<set>/{RESULTS_PATH})",
    )
    parser.add_argument(
        "--parseq-readings",
        type=Path,
        required=True,
        help=f"{READINGS_FILE} que escribe `ocr.py --labeled-set` (id, lectura, confianza)",
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=float(os.environ.get("OCR_MIN_CONF", DEFAULT_MIN_CONF)),
        help="OCR_MIN_CONF de PARSeq (por defecto el del entorno o 0,78)",
    )
    args = parser.parse_args(argv)
    if args.out_dir is None:
        parser.error("indicá --out-dir o definí EVAL_OUT")
    if not 0.0 <= args.min_conf <= 1.0:
        parser.error("--min-conf debe estar entre 0 y 1")
    try:
        args.results = parse_results(args.results, args.labeled_set)
    except ComparisonError as exc:
        parser.error(str(exc))
    return args


def cell(count: int, share: float | None) -> str:
    return f"{count} ({share} %)"


def print_table(report: Mapping[str, Any]) -> None:
    """Una tabla por grupo (total, día, noche): cada versión de ML Kit y PARSeq lado a lado."""
    versions = report["versiones"]
    for group in report["parseq"]:
        print(
            f"\n[{group}]".ljust(34) + "".join(f"{'ML Kit ' + v:>18}" for v in versions) + f"{'PARSeq':>18}"
        )
        for title, key in (
            ("lectura exacta", "lectura_exacta"),
            ("sin lectura", "sin_lectura"),
            ("lectura equivocada", "lectura_equivocada"),
        ):
            cells = [
                cell(report["mlkit"][v][group][key], report["mlkit"][v][group][key + "_pct"])
                for v in versions
            ]
            parseq = report["parseq"][group]
            cells.append(cell(parseq[key], parseq[key + "_pct"]))
            print(title.ljust(34) + "".join(f"{c:>18}" for c in cells))
        found = [
            cell(
                report["mlkit"][v][group]["candidatos"]["la_correcta_aparece"],
                report["mlkit"][v][group]["candidatos"]["la_correcta_aparece_pct"],
            )
            for v in versions
        ]
        print("la correcta entre los candidatos".ljust(34) + "".join(f"{c:>18}" for c in [*found, "-"]))
        normalized = [
            cell(
                report["mlkit"][v][group]["candidatos"]["la_correcta_aparece_normalizada"],
                report["mlkit"][v][group]["candidatos"]["la_correcta_aparece_normalizada_pct"],
            )
            for v in versions
        ]
        print("  y normalizada por posición".ljust(34) + "".join(f"{c:>18}" for c in [*normalized, "-"]))
        for title, key in (
            ("R3: patente distinta (RED)", "patente_distinta_RED"),
            ("R3: un carácter (GREEN)", "difiere_en_un_caracter_GREEN"),
            ("R3: ilegible (GREEN)", "ilegible_GREEN"),
            ("R3: coincide (GREEN)", "coincide_GREEN"),
        ):
            values = [str(report["error_silencioso"][v][group][key]) for v in versions]
            print(title.ljust(34) + "".join(f"{c:>18}" for c in [*values, "-"]))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name, path in (
        ("el set etiquetado", args.labeled_set),
        *((f"los resultados {version}", path) for version, path in args.results.items()),
        ("las lecturas de PARSeq", args.parseq_readings),
        ("la salida", args.out_dir),
    ):
        if path.resolve().is_relative_to(REPO_ROOT):
            print(
                f"{name} no puede estar dentro del repositorio: tiene patentes de terceros.", file=sys.stderr
            )
            return 1
    try:
        parseq = read_parseq(args.parseq_readings)
        versions = {
            version: build_rows(args.labeled_set, read_results(path), parseq)
            for version, path in args.results.items()
        }
        report = build_report(versions, args.min_conf)
    except (OSError, ComparisonError, ocr_labels.LabelsError, KeyError, ValueError) as exc:
        print(f"No se pudo cruzar: {exc}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mlkit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_rows(args.out_dir / "mlkit_rows.csv", versions, args.min_conf)
    print_table(report)
    print(f"\nCandidatos iguales entre versiones: {report['candidatos_iguales_entre_versiones']}")
    for name, delta in report["lecturas_nuevas"].items():
        print(
            f"Lecturas nuevas ({name}): {delta['lecturas_nuevas']} · exactas {delta['exactas']} · "
            f"equivocadas {delta['equivocadas']} {delta['equivocada_detalle']} · "
            f"perdidas {delta['perdidas']} · "
            f"cambiadas {delta['cambiadas']}"
        )
    for name, value in report["tiempo"].items():
        print(f"Tiempo por recorte ({name}): {value}")
    print(f"U5: {report['u5']}")
    print(f"Salida: {args.out_dir / 'mlkit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
