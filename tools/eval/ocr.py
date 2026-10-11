"""Evalúa el OCR (puerto `ocr`) sobre el export COCO del dataset "patentes" de Roboflow.

Por cada caja anotada recorta la foto igual que el servicio (`crop_box`, sin padding), la lee con
el adaptador seleccionado y compara el texto con la etiqueta (el nombre de la clase, normalizado y
sin espacios) usando `domain.plate.plate_check`. Descarta los marcadores de posición (patentes con
todos los dígitos en cero, como `AA000AA`).

Escribe en `EVAL_OUT`: `report.json` (solo cifras), `audit.csv`, `audit.html` y `audit/` con los
recortes que hay que mirar a ojo. Esas carpetas contienen patentes de terceros: nunca van al repo.
La consola muestra solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import csv
import html
import json
import math
import os
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import Box, EvidenceImage, EvidenceRole
from vvi_ia.domain.plate import PlateCheck, canonicalize, plate_check
from vvi_ia.service.ports import ImageDecodeError
from vvi_ia.service.validate_evidence import crop_box

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
ANNOTATIONS = "_annotations.coco.json"
DEFAULT_MIN_CONF = 0.60
SWEEP = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
PROPOSAL_FLOOR = 0.50
PROPOSAL_MARGIN = 0.01
PROPOSAL_CEIL = 0.99
DEFAULT_MATCH_SAMPLES = 20
FORMATS = {"LLDDDLL": "mercosur", "LLLDDD": "anterior"}
OTHER_FORMAT = "otro"
EXACT = "exacta"


@dataclass(frozen=True)
class Sample:
    """Una caja anotada: de qué foto sale, dónde está y qué patente es."""

    split: str
    file: str
    box: Box
    label: str


@dataclass(frozen=True)
class Reading:
    """Lo que leyó el OCR sobre una muestra (`text` vacío y `conf` 0 si no leyó nada)."""

    sample: Sample
    text: str
    conf: float
    failed: bool  # el adaptador devolvió `None`

    @property
    def content(self) -> PlateCheck:
        """Cómo compara el texto con la etiqueta, sin aplicar ningún umbral de confianza."""
        return plate_check(self.sample.label, self.text, 1.0, 0.0)

    def check(self, min_conf: float) -> PlateCheck:
        return plate_check(self.sample.label, self.text, self.conf, min_conf)

    @property
    def exact(self) -> bool:
        """El texto canónico coincide con la etiqueta sin corregir ninguna confusión."""
        return canonicalize(self.text) == self.sample.label


# --- carga del dataset ---------------------------------------------------------------------


def shape(label: str) -> str:
    """`AB123CD` → `LLDDDLL`."""
    return "".join("D" if char.isdigit() else "L" if char.isalpha() else char for char in label)


def is_placeholder(label: str) -> bool:
    """Marcador de posición: tiene dígitos y todos son ceros (`AA000AA`)."""
    digits = [char for char in label if char.isdigit()]
    return bool(digits) and all(char == "0" for char in digits)


def label_format(label: str) -> str:
    return FORMATS.get(shape(label), OTHER_FORMAT)


def load_samples(data_dir: Path, splits: Sequence[str]) -> tuple[list[Sample], Counter[str]]:
    """Cajas del dataset y cuántas se descartaron, por motivo."""
    samples: list[Sample] = []
    discarded: Counter[str] = Counter()
    for split in splits:
        path = data_dir / split / ANNOTATIONS
        if not path.is_file():
            raise FileNotFoundError(f"falta {split}/{ANNOTATIONS}")
        coco = json.loads(path.read_text(encoding="utf-8"))
        files = {image["id"]: image["file_name"] for image in coco["images"]}
        names = {category["id"]: category["name"] for category in coco["categories"]}
        for annotation in coco["annotations"]:
            label = canonicalize(names[annotation["category_id"]])
            if not label:
                discarded["etiqueta vacía"] += 1
            elif is_placeholder(label):
                discarded["marcador de posición"] += 1
            else:
                x, y, width, height = annotation["bbox"]
                box = Box(x1=x, y1=y, x2=x + width, y2=y + height)
                samples.append(Sample(split, files[annotation["image_id"]], box, label))
    return samples, discarded


# --- lectura -------------------------------------------------------------------------------


def read_samples(
    samples: Iterable[Sample], data_dir: Path, reader: Any
) -> tuple[list[tuple[Reading, Any]], Counter[str]]:
    """Lee cada muestra. Devuelve las lecturas con su recorte y cuántas se descartaron, por motivo."""
    decoder = CvImageDecoder()
    images: dict[tuple[str, str], EvidenceImage | None] = {}
    readings: list[tuple[Reading, Any]] = []
    discarded: Counter[str] = Counter()
    for sample in samples:
        key = (sample.split, sample.file)
        if key not in images:
            try:
                pixels = decoder.decode((data_dir / sample.split / sample.file).read_bytes())
                images[key] = EvidenceImage(role=EvidenceRole.PLATE, index=0, pixels=pixels)
            except (OSError, ImageDecodeError):
                images[key] = None
        image = images[key]
        if image is None:
            discarded["foto ilegible"] += 1
            continue
        crop = crop_box(image, sample.box)
        if crop is None:
            discarded["caja vacía"] += 1
            continue
        read = reader.read(crop)
        reading = Reading(sample, *(read if read is not None else ("", 0.0)), failed=read is None)
        readings.append((reading, crop))
    return readings, discarded


# --- métricas ------------------------------------------------------------------------------


def distribution(readings: Sequence[Reading], min_conf: float) -> dict[str, int]:
    counts = Counter(reading.check(min_conf).value for reading in readings)
    return {check.value: counts.get(check.value, 0) for check in PlateCheck}


def summarize(readings: Sequence[Reading], min_conf: float) -> dict[str, Any]:
    total = len(readings)
    exact = sum(reading.exact for reading in readings)
    dist = distribution(readings, min_conf)
    return {
        "lecturas": total,
        "fallas_del_modelo": sum(reading.failed for reading in readings),
        "coincidencia_exacta": exact,
        "coincidencia_exacta_pct": round(100 * exact / total, 1) if total else None,
        "distribucion": dist,
        "distribucion_pct": {
            key: round(100 * value / total, 1) if total else None for key, value in dist.items()
        },
    }


def sweep(readings: Sequence[Reading], thresholds: Iterable[float] = SWEEP) -> list[dict[str, Any]]:
    """Qué pasa con cada umbral: cuántas lecturas se aceptan y cuántas de ellas están mal."""
    rows = []
    for threshold in thresholds:
        dist = distribution(readings, threshold)
        accepted = len(readings) - dist[PlateCheck.UNREADABLE.value]
        rows.append(
            {
                "umbral": threshold,
                "aceptadas": accepted,
                "aceptadas_pct": round(100 * accepted / len(readings), 1) if readings else None,
                "match": dist[PlateCheck.MATCH.value],
                "fuzzy": dist[PlateCheck.FUZZY.value],
                "mismatch": dist[PlateCheck.MISMATCH.value],
            }
        )
    return rows


def propose_min_conf(readings: Sequence[Reading]) -> dict[str, Any]:
    """`OCR_MIN_CONF` más bajo que deja a todos los MISMATCH debajo del umbral.

    Un MISMATCH aceptado es una alerta RED (fraude) sobre una patente que en realidad es la correcta,
    así que el umbral tiene que quedar por encima de la confianza del peor MISMATCH. Sin ningún
    MISMATCH en los datos, se propone el piso.
    """
    wrong = [reading.conf for reading in readings if reading.content == PlateCheck.MISMATCH]
    fuzzy = [reading.conf for reading in readings if reading.content == PlateCheck.FUZZY]
    highest = max(wrong, default=None)
    if highest is None:
        value = PROPOSAL_FLOOR
    else:
        value = min(PROPOSAL_CEIL, max(PROPOSAL_FLOOR, math.ceil((highest + PROPOSAL_MARGIN) * 100) / 100))
    accepted = [reading for reading in readings if reading.text and reading.conf >= value]
    return {
        "OCR_MIN_CONF": value,
        "mayor_confianza_de_un_mismatch": None if highest is None else round(highest, 4),
        "mayor_confianza_de_un_fuzzy": None if not fuzzy else round(max(fuzzy), 4),
        "mismatch_totales": len(wrong),
        "lecturas_aceptadas": len(accepted),
        "lecturas_aceptadas_pct": round(100 * len(accepted) / len(readings), 1) if readings else None,
    }


def build_report(
    readings: Sequence[Reading],
    min_conf: float,
    model_version: str,
    discarded: Counter[str],
    total_boxes: int,
) -> dict[str, Any]:
    in_scope = [reading for reading in readings if label_format(reading.sample.label) != OTHER_FORMAT]
    groups: dict[str, dict[str, list[Reading]]] = {"por_split": {}, "por_formato": {}}
    for reading in readings:
        groups["por_split"].setdefault(reading.sample.split, []).append(reading)
        groups["por_formato"].setdefault(label_format(reading.sample.label), []).append(reading)
    return {
        "modelo": model_version,
        "min_conf": min_conf,
        "cajas_del_dataset": total_boxes,
        "descartadas": dict(discarded),
        "total": summarize(readings, min_conf),
        **{
            name: {key: summarize(items, min_conf) for key, items in sorted(group.items())}
            for name, group in groups.items()
        },
        "barrido_de_umbrales": sweep(readings),
        # Los formatos que no son Mercosur ni el anterior son, en este dataset, motos de dos líneas:
        # fuera de alcance. La propuesta se calcula sin ellas; la otra se muestra por transparencia.
        "propuesta": propose_min_conf(in_scope),
        "propuesta_con_todos_los_formatos": propose_min_conf(readings),
        "lecturas_fuera_de_alcance": len(readings) - len(in_scope),
    }


# --- auditoría -----------------------------------------------------------------------------


def audit_selection(
    items: Sequence[tuple[Reading, Any]], min_conf: float, match_samples: int, seed: int
) -> list[tuple[Reading, Any]]:
    """Todo lo que no coincide exacto, más una muestra al azar de lo que sí (para detectar etiquetas mal)."""

    def is_right(item: tuple[Reading, Any]) -> bool:
        return item[0].exact and item[0].check(min_conf) == PlateCheck.MATCH

    wrong = [item for item in items if not is_right(item)]
    right = [item for item in items if is_right(item)]
    picked = random.Random(seed).sample(right, min(match_samples, len(right)))  # noqa: S311
    return sorted(wrong + picked, key=lambda item: (item[0].sample.split, item[0].sample.file))


def write_audit(selection: Sequence[tuple[Reading, Any]], out_dir: Path, min_conf: float) -> None:
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for old in audit_dir.glob("*.jpg"):
        old.unlink()
    rows = []
    for number, (reading, crop) in enumerate(selection, start=1):
        name = f"{number:03d}.jpg"
        cv2.imwrite(str(audit_dir / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        rows.append(
            {
                "recorte": f"audit/{name}",
                "split": reading.sample.split,
                "foto": reading.sample.file,
                "etiqueta": reading.sample.label,
                "lectura": reading.text,
                "confianza": f"{reading.conf:.4f}",
                "resultado": reading.check(min_conf).value,
                "exacta": "si" if reading.exact else "no",
            }
        )
    with (out_dir / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["recorte"])
        writer.writeheader()
        writer.writerows(rows)
    cells = "".join(
        "<tr>"
        f'<td><img src="{html.escape(row["recorte"])}" height="48"></td>'
        + "".join(f"<td>{html.escape(row[key])}</td>" for key in list(row)[1:])
        + "</tr>"
        for row in rows
    )
    head = "".join(f"<th>{html.escape(key)}</th>" for key in (rows[0] if rows else {"recorte": ""}))
    (out_dir / "audit.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Auditoría del OCR</title>"
        "<style>body{font:14px sans-serif}td,th{border:1px solid #ccc;padding:4px 8px}"
        "table{border-collapse:collapse}</style>"
        f"<h1>Auditoría del OCR</h1><p>Contiene patentes de terceros: no compartir ni commitear.</p>"
        f"<table><tr>{head}</tr>{cells}</table>",
        encoding="utf-8",
    )


# --- línea de comandos ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="export COCO (EVAL_DATA_DIR)"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument(
        "--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="pesos (MODELS_DIR)"
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--model", default="parseq", help="modelo del puerto ocr (nombre del manifest)")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=float(os.environ.get("OCR_MIN_CONF", DEFAULT_MIN_CONF)),
        help="OCR_MIN_CONF con el que se arma la distribución (por defecto el del entorno o 0,60)",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=None, help="por defecto, los que existan en el dataset"
    )
    parser.add_argument(
        "--match-samples", type=int, default=DEFAULT_MATCH_SAMPLES, help="aciertos a incluir en la auditoría"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for flag, value in (
        ("--data-dir", args.data_dir),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    if not 0.0 <= args.min_conf <= 1.0:
        parser.error("--min-conf debe estar entre 0 y 1")
    return args


def inside_repo(path: Path) -> bool:
    return path.resolve().is_relative_to(REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name, path in (("el dataset", args.data_dir), ("la salida", args.out_dir)):
        if inside_repo(path):
            print(
                f"{name} no puede estar dentro del repositorio: tiene patentes de terceros.", file=sys.stderr
            )
            return 1
    splits = args.splits or [split for split in SPLITS if (args.data_dir / split / ANNOTATIONS).is_file()]
    if not splits:
        print(f"No hay ningún split con {ANNOTATIONS} en el dataset.", file=sys.stderr)
        return 1
    try:
        samples, discarded = load_samples(args.data_dir, splits)
        models = build_models({"ocr": args.model}, args.manifest, args.models_dir)
    except (FileNotFoundError, ModelSetupError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    total_boxes = len(samples) + sum(discarded.values())
    items, unreadable = read_samples(samples, args.data_dir, models.plate)
    discarded += unreadable
    readings = [reading for reading, _ in items]

    report = build_report(readings, args.min_conf, models.versions()["ocr"], discarded, total_boxes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selection = audit_selection(items, args.min_conf, args.match_samples, args.seed)
    write_audit(selection, args.out_dir, args.min_conf)

    total, proposal = report["total"], report["propuesta"]
    every_format = report["propuesta_con_todos_los_formatos"]
    print(f"Modelo {report['modelo']} · {total_boxes} cajas · descartadas {dict(discarded) or 'ninguna'}")
    print(f"Lecturas: {total['lecturas']} · fallas del modelo: {total['fallas_del_modelo']}")
    print(f"Coincidencia exacta: {total['coincidencia_exacta']} ({total['coincidencia_exacta_pct']} %)")
    print(f"Distribución con OCR_MIN_CONF={args.min_conf}: {total['distribucion']}")
    print(
        f"Propuesta (sin {report['lecturas_fuera_de_alcance']} lecturas de otros formatos): "
        f"OCR_MIN_CONF={proposal['OCR_MIN_CONF']} "
        f"({proposal['lecturas_aceptadas_pct']} % de lecturas aceptadas)"
    )
    print(f"Propuesta con todos los formatos: OCR_MIN_CONF={every_format['OCR_MIN_CONF']}")
    print(f"Auditoría: {len(selection)} recortes en {args.out_dir / 'audit.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
