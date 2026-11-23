"""Revisión a mano de los falsos positivos del detector (auto y patente) y precisión corregida.

`detector.py --dump-fp` escribe `falsos_positivos.json`. Muchos "falsos positivos" son autos o patentes
reales que el dataset no anotó (Open Images no anota todo lo que se ve). Dos pasos:

- `sheets`: arma hojas de contacto (fuera del repo) con cada falso positivo en rojo y las cajas reales en
  verde, más `indice.json` (número de panel → id). Con `--inherit-from` reutiliza lo ya revisado en otra
  corrida (misma foto, misma clase, IoU >= 0,5) y solo muestra lo nuevo.
- `corrected`: con `veredictos.json` (categoría → números de panel) calcula la precisión corregida,
  **aparte** de la oficial, que no se reemplaza.

Categorías: `real_sin_anotar` (hay un auto o una patente real y no está anotado), `otro_vehiculo` (camión,
ómnibus o furgoneta: Open Images los anota con otra clase), `duplicada_o_mal_ubicada` (sobre un objeto
anotado, con IoU < 0,5 o repetida), `error` (no hay auto ni patente) y `dudoso`.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audit_sheets import GREEN, RED, Panel, sheets, with_margin
from detector import iou

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL = "real_sin_anotar"
OTHER = "otro_vehiculo"
DUPLICATE = "duplicada_o_mal_ubicada"
ERROR = "error"
DOUBT = "dudoso"
CATEGORIES = (REAL, OTHER, DUPLICATE, ERROR, DOUBT)
INHERIT_IOU = 0.5
PER_SHEET = 12


def inherit(
    new: Sequence[Mapping[str, Any]], old: Sequence[Mapping[str, Any]], old_verdicts: Mapping[str, str]
) -> dict[str, str]:
    """Veredicto de cada falso positivo nuevo que coincide con uno ya revisado (misma foto y clase)."""
    out: dict[str, str] = {}
    for fp in new:
        for before in old:
            if (
                before["id"] in old_verdicts
                and before["foto"] == fp["foto"]
                and before["clase"] == fp["clase"]
                and iou(before["caja"], fp["caja"]) >= INHERIT_IOU
            ):
                out[fp["id"]] = old_verdicts[before["id"]]
                break
    return out


def resolve_verdicts(numbered: Mapping[str, Sequence[int]], index: Mapping[str, str]) -> dict[str, str]:
    """`{categoría: [números de panel]}` → `{id: categoría}`; falla si un número no existe o se repite."""
    out: dict[str, str] = {}
    for category, numbers in numbered.items():
        if category not in CATEGORIES:
            raise ValueError(f"categoría desconocida: {category}")
        for number in numbers:
            fp_id = index[str(number)]
            if fp_id in out:
                raise ValueError(f"el panel {number} está en dos categorías")
            out[fp_id] = category
    return out


def summarize(
    report_class: Mapping[str, Any], fps: Sequence[Mapping[str, Any]], verdicts: Mapping[str, str], name: str
) -> dict[str, Any]:
    """Cuenta por categoría los falsos positivos de una clase y calcula la precisión corregida."""
    mine = [fp for fp in fps if fp["clase"] == name]
    false_positives = report_class["falsos_positivos"]
    if len(mine) != false_positives:
        raise ValueError(f"{name}: {len(mine)} en el volcado y {false_positives} en el informe")
    missing = [fp["id"] for fp in mine if fp["id"] not in verdicts]
    if missing:
        raise ValueError(f"{name}: faltan veredictos de {len(missing)} falsos positivos")
    precision = report_class["precision"]
    tp = round(false_positives * precision / (1 - precision)) if precision < 1 else 0
    counts = {c: sum(verdicts[fp["id"]] == c for fp in mine) for c in CATEGORIES}
    return {
        "aciertos": tp,
        "falsos_positivos": false_positives,
        "precision_oficial": precision,
        "por_categoria": counts,
        "precision_corregida": round((tp + counts[REAL]) / (tp + false_positives), 4),
        "precision_corregida_con_dudosos": round(
            (tp + counts[REAL] + counts[DOUBT]) / (tp + false_positives), 4
        ),
        # Para v2, camión y van son clases de presencia (cuentan como auto): son aciertos.
        "precision_corregida_otros_vehiculos_son_aciertos": round(
            (tp + counts[REAL] + counts[OTHER]) / (tp + false_positives), 4
        ),
    }


def combine(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro-promedio de varias clases (autos + patentes)."""
    tp = sum(r["aciertos"] for r in rows)
    fp = sum(r["falsos_positivos"] for r in rows)
    real = sum(r["por_categoria"][REAL] for r in rows)
    doubt = sum(r["por_categoria"][DOUBT] for r in rows)
    other = sum(r["por_categoria"].get(OTHER, 0) for r in rows)
    return {
        "aciertos": tp,
        "falsos_positivos": fp,
        "precision_oficial": round(tp / (tp + fp), 4),
        "precision_corregida": round((tp + real) / (tp + fp), 4),
        "precision_corregida_con_dudosos": round((tp + real + doubt) / (tp + fp), 4),
        "precision_corregida_otros_vehiculos_son_aciertos": round((tp + real + other) / (tp + fp), 4),
    }


def command_sheets(args: argparse.Namespace) -> int:
    dump = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    dataset = Path(dump["dataset"])
    fps = dump["falsos_positivos"]
    inherited: dict[str, str] = {}
    if args.inherit_from:
        old = json.loads(Path(args.inherit_from).read_text(encoding="utf-8"))["falsos_positivos"]
        old_index = json.loads((Path(args.inherit_from).parent / "indice.json").read_text(encoding="utf-8"))
        old_verdicts = resolve_verdicts(
            json.loads((Path(args.inherit_from).parent / "veredictos.json").read_text(encoding="utf-8")),
            old_index,
        )
        old_verdicts.update(
            json.loads((Path(args.inherit_from).parent / "heredados.json").read_text(encoding="utf-8"))
            if (Path(args.inherit_from).parent / "heredados.json").is_file()
            else {}
        )
        inherited = inherit(fps, old, old_verdicts)
    pending = [fp for fp in fps if fp["id"] not in inherited]
    panels = []
    for number, fp in enumerate(pending, 1):
        height, width = fp["tam"]
        box = tuple(fp["caja"])
        boxes = [(tuple(t), GREEN) for t in fp["reales"]] + [(box, RED)]
        panels.append(
            Panel(
                dataset / fp["foto"],
                f"{number:03d} {fp['clase']} c{fp['confianza']:.2f} iou{fp['iou_max']:.2f}",
                crop=with_margin(box, (height, width), 0.8 if fp["clase"] == "car" else 1.5, 0.45),  # type: ignore[arg-type]
                boxes=boxes,  # type: ignore[arg-type]
            )
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = sheets(panels, out, columns=4, rows=3, cell=420, prefix="fp")
    (out / "indice.json").write_text(
        json.dumps({str(n): fp["id"] for n, fp in enumerate(pending, 1)}, indent=1), encoding="utf-8"
    )
    (out / "heredados.json").write_text(json.dumps(inherited, indent=1), encoding="utf-8")
    (out / "falsos_positivos.json").write_text(Path(args.dump).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(pending)} para revisar en {len(written)} hojas; {len(inherited)} heredados de la anterior")
    return 0


def load_run(folder: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    dump = json.loads((folder / "falsos_positivos.json").read_text(encoding="utf-8"))["falsos_positivos"]
    index = json.loads((folder / "indice.json").read_text(encoding="utf-8"))
    verdicts = resolve_verdicts(json.loads((folder / "veredictos.json").read_text(encoding="utf-8")), index)
    verdicts.update(json.loads((folder / "heredados.json").read_text(encoding="utf-8")))
    return dump, verdicts


def command_corrected(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    fps, verdicts = load_run(Path(args.audit))
    rows = {name: summarize(report["clases"][name], fps, verdicts, name) for name in ("car", "plate")}
    result = {"por_clase": rows, "auto_y_patente": combine(list(rows.values())), "mosaico": report["mosaico"]}
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, row in rows.items():
        print(f"{name}: {row}")
    print(f"auto + patente: {result['auto_y_patente']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("sheets", help="hojas de contacto de los falsos positivos")
    one.add_argument(
        "--dump", type=Path, required=True, help="falsos_positivos.json de detector.py --dump-fp"
    )
    one.add_argument("--out", type=Path, required=True, help="carpeta de salida (fuera del repo)")
    one.add_argument("--inherit-from", type=Path, help="falsos_positivos.json de otra corrida ya revisada")
    two = sub.add_parser("corrected", help="precisión corregida con los veredictos")
    two.add_argument("--report", type=Path, required=True, help="detector_report.json de la misma corrida")
    two.add_argument("--audit", type=Path, required=True, help="carpeta de `sheets` con veredictos.json")
    two.add_argument(
        "--out", type=Path, default=Path(os.environ.get("EVAL_OUT", ".")) / "precision_corregida.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = Path(args.out).resolve()
    if target.is_relative_to(REPO_ROOT):
        print("La salida no puede estar dentro del repositorio.", file=sys.stderr)
        return 1
    return command_sheets(args) if args.command == "sheets" else command_corrected(args)


if __name__ == "__main__":
    sys.exit(main())
