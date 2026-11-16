"""Compara ML Kit (medido en un celular por el Frontend) con PARSeq sobre el set etiquetado de OCR.

Cruza por id `resultados/ocr-results.csv` (id, formato, luz, lectura, candidatos crudos, confianza, tiempo)
con las etiquetas del set (`etiquetas.xlsx` o `etiquetas.csv`, solo las patentes legibles de auto) y con las
lecturas de PARSeq que escribe `ocr.py --labeled-set` (`readings.csv`). Informa, en total y por luz (día y
noche):

- lectura exacta (aproxima U5: al menos 85 % de día y 70 % de noche), sin lectura (el usuario la tipea) y
  lectura equivocada (error silencioso si el usuario no la corrige), y si la patente correcta aparecía entre
  los candidatos crudos aunque la lectura elegida fuera otra;
- lo mismo para PARSeq con la regla del servicio (`OCR_MIN_CONF`, estructura válida), lado a lado;
- el error silencioso: para cada lectura equivocada de ML Kit, R3 (`plate_check`) tomándola como patente
  denunciada contra la lectura de PARSeq de la misma foto: patente distinta (RED con penalización a un usuario
  honesto), difiere en un carácter o ilegible (GREEN);
- el tiempo por recorte (mediana y p95).

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

from vvi_ia.domain.plate import PlateCheck, canonicalize, plate_check

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


def candidate_match(label: str, candidates: Sequence[str]) -> str:
    """`exacta` si un candidato crudo contiene la patente correcta; `a_un_caracter` si una ventana del largo
    de la patente difiere en un solo carácter; `no` en otro caso."""
    if any(label in candidate for candidate in candidates):
        return "exacta"
    size = len(label)
    for candidate in candidates:
        for start in range(len(candidate) - size + 1):
            if Levenshtein.distance(label, candidate[start : start + size]) <= 1:
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
            "no_exactas": len(not_exact),
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
    out: dict[str, Any] = {"umbral_pct": dict(U5_MIN_PCT)}
    for reader in ("mlkit", "parseq"):
        out[reader] = {
            light: {
                "lectura_exacta_pct": report[reader][light]["lectura_exacta_pct"],
                "cumple": (report[reader][light]["lectura_exacta_pct"] or 0.0) >= U5_MIN_PCT[light],
            }
            for light in LIGHTS
            if light in report[reader]
        }
    return out


def build_report(rows: Sequence[Row], min_conf: float) -> dict[str, Any]:
    groups: dict[str, list[Row]] = {"total": list(rows)}
    for light in LIGHTS:
        groups[light] = [row for row in rows if row.light == light]
    groups = {name: items for name, items in groups.items() if items}
    report: dict[str, Any] = {
        "min_conf": min_conf,
        "mlkit": {name: mlkit_block(items) for name, items in groups.items()},
        "parseq": {name: parseq_block(items, min_conf) for name, items in groups.items()},
        "error_silencioso": {name: silent_error(items, min_conf) for name, items in groups.items()},
        "tiempo": timing(rows),
    }
    report["u5"] = u5(report)
    return report


def write_rows(path: Path, rows: Sequence[Row], min_conf: float) -> None:
    """Una fila por recorte con las clases (nunca el texto)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        for row in rows:
            outcome = mlkit_outcome(row)
            writer.writerow(
                {
                    "id": row.ident,
                    "luz": row.light,
                    "mlkit": outcome,
                    "mlkit_error": wrong_kind(row.label, row.mlkit) if outcome == WRONG else "",
                    "candidato": candidate_match(row.label, row.candidates),
                    "parseq": parseq_outcome(row, min_conf),
                    "r3": r3_of_wrong_reading(row, min_conf) if outcome == WRONG else "",
                }
            )


# --- línea de comandos ----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--labeled-set", type=Path, required=True, help="set etiquetado de ocr_set.py")
    parser.add_argument(
        "--results", type=Path, help=f"resultados de ML Kit (por defecto <labeled-set>/{RESULTS_PATH})"
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
    args.results = args.results or args.labeled_set / RESULTS_PATH
    return args


def print_table(report: Mapping[str, Any]) -> None:
    names = [name for name in ("total", "dia", "noche") if name in report["mlkit"]]
    print(f"{'':28}" + "".join(f"{name:>22}" for name in names))
    for title, key in (
        ("lectura exacta", "lectura_exacta"),
        ("sin lectura", "sin_lectura"),
        ("lectura equivocada", "lectura_equivocada"),
    ):
        for reader, label in (("mlkit", "ML Kit"), ("parseq", "PARSeq")):
            cells = [
                f"{report[reader][name][key]} ({report[reader][name][key + '_pct']} %)" for name in names
            ]
            print(f"{title + ' ' + label:28}" + "".join(f"{cell:>22}" for cell in cells))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name, path in (
        ("el set etiquetado", args.labeled_set),
        ("los resultados", args.results),
        ("las lecturas de PARSeq", args.parseq_readings),
        ("la salida", args.out_dir),
    ):
        if path.resolve().is_relative_to(REPO_ROOT):
            print(
                f"{name} no puede estar dentro del repositorio: tiene patentes de terceros.", file=sys.stderr
            )
            return 1
    try:
        rows = build_rows(args.labeled_set, read_results(args.results), read_parseq(args.parseq_readings))
    except (OSError, ComparisonError, ocr_labels.LabelsError, KeyError, ValueError) as exc:
        print(f"No se pudo cruzar: {exc}", file=sys.stderr)
        return 1
    report = build_report(rows, args.min_conf)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mlkit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_rows(args.out_dir / "mlkit_rows.csv", rows, args.min_conf)
    print_table(report)
    for name, block in report["error_silencioso"].items():
        print(
            f"Error silencioso ({name}): {block['lecturas_equivocadas']} lecturas equivocadas de ML Kit · "
            f"patente distinta (RED) {block['patente_distinta_RED']} · "
            f"un carácter {block['difiere_en_un_caracter_GREEN']} · "
            f"ilegible {block['ilegible_GREEN']} · coincide {block['coincide_GREEN']}"
        )
    print(f"Tiempo por recorte: {report['tiempo']}")
    print(f"U5: {report['u5']}")
    print(f"Salida: {args.out_dir / 'mlkit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
