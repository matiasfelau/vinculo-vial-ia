"""Falsos positivos del detector de daños: fotos de autos sin daño evidente frente al recall de CarDD.

Tres pasos, que se corren en este orden:

- `select`: elige de un subset de Open Images (el de `detector`) las fotos donde un auto ocupa buena parte
  del cuadro, que se parecen a las fotos de detalle del servicio, y las copia a `<data-dir>/damage-fp/images`.
- `sheet`: arma hojas de contacto para revisar a ojo y anotar en `excluidas.txt` (una foto por línea) las que
  sí tengan daño.
- `run`: corre el adaptador real sobre esas fotos y sobre el split de prueba de CarDD, y barre
  `DAMAGE_MIN_CONF`: qué fracción de las fotos sin daño tiene alguna caja (falso positivo por foto) frente
  a qué fracción de las fotos con daño de CarDD la tiene (recall por foto).

Las fotos y la salida se quedan fuera del repo; la consola y `damage_fp_report.json` tienen solo cifras.
Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from damage import CLASS_IDS
from damage import DATASET_NAME as CARDD_NAME
from detector import ANNOTATIONS, FLOOR, REPO_ROOT
from detector import DATASET_NAME as OPEN_IMAGES_NAME

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.adapters.models.rfdetr_damage import MODEL_NAME

FOLDER = "damage-fp"
EXCLUDED = "excluidas.txt"
THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MIN_CAR_FRACTION = 0.30
CELL = (320, 240)


def closeups(coco: dict[str, Any], min_fraction: float) -> list[str]:
    """Fotos donde algún auto ocupa al menos `min_fraction` del cuadro, ordenadas por nombre."""
    car_ids = {c["id"] for c in coco["categories"] if c["name"] == "car"}
    images = {i["id"]: i for i in coco["images"]}
    largest: dict[int, float] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in car_ids:
            info = images[ann["image_id"]]
            fraction = ann["bbox"][2] * ann["bbox"][3] / (info["width"] * info["height"])
            largest[ann["image_id"]] = max(largest.get(ann["image_id"], 0.0), fraction)
    return sorted(images[i]["file_name"] for i, fraction in largest.items() if fraction >= min_fraction)


def photo_rate(scores: Sequence[float], threshold: float) -> float:
    """Fracción de fotos cuya mayor confianza llega a `threshold` (0 si no hay fotos)."""
    return sum(score >= threshold for score in scores) / len(scores) if scores else 0.0


def sweep(
    clean: Sequence[float], damaged: Sequence[float], thresholds: Sequence[float] = THRESHOLDS
) -> list[dict[str, float]]:
    """Por umbral: FP por foto (sin daño) y recall por foto (con daño), de la mayor confianza de cada foto."""
    return [
        {
            "umbral": t,
            "fotos_sin_dano_con_caja": sum(score >= t for score in clean),
            "falsos_positivos_por_foto": round(photo_rate(clean, t), 4),
            "recall_por_foto": round(photo_rate(damaged, t), 4),
        }
        for t in thresholds
    ]


def propose(rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    """El umbral que más separa recall de falsos positivos (recall - FP por foto); en empate, el más bajo."""
    best = max(
        rows, key=lambda row: (row["recall_por_foto"] - row["falsos_positivos_por_foto"], -row["umbral"])
    )
    return {"DAMAGE_MIN_CONF": best["umbral"], **best}


def class_thresholds(per_class: dict[str, list[dict[str, float]]], cap: float) -> dict[str, float]:
    """Por clase, el umbral más bajo con a lo sumo `cap` de falsos positivos por foto (1,0 si ninguno)."""
    return {
        label: next((r["umbral"] for r in rows if r["falsos_positivos_por_foto"] <= cap), 1.0)
        for label, rows in per_class.items()
    }


def combined(
    clean: Sequence[dict[str, float]], damaged: Sequence[dict[str, float]], thresholds: dict[str, float]
) -> dict[str, float]:
    """FP y recall por foto cuando cada clase tiene su umbral: la foto cuenta si alguna clase lo supera."""

    def flagged(photo: dict[str, float]) -> bool:
        return any(conf >= thresholds[label] for label, conf in photo.items())

    return {
        "falsos_positivos_por_foto": round(sum(map(flagged, clean)) / len(clean), 4),
        "recall_por_foto": round(sum(map(flagged, damaged)) / len(damaged), 4),
    }


def read_excluded(folder: Path) -> set[str]:
    path = folder / EXCLUDED
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def contact_sheet(images: Sequence[Path], captions: Sequence[str], columns: int = 4) -> np.ndarray:
    """Hoja con las fotos a tamaño uniforme y un pie con el nombre (o dato) de cada una."""
    cells = []
    for path, caption in zip(images, captions, strict=True):
        pixels = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        cell = np.zeros((CELL[1] + 18, CELL[0], 3), np.uint8)
        if pixels is not None:
            scale = min(CELL[0] / pixels.shape[1], CELL[1] / pixels.shape[0])
            resized = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cell[: resized.shape[0], : resized.shape[1]] = resized
        cv2.putText(cell, caption, (2, CELL[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cells.append(cell)
    while len(cells) % columns:
        cells.append(np.zeros_like(cells[0]))
    return np.vstack([np.hstack(cells[i : i + columns]) for i in range(0, len(cells), columns)])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="elegir y copiar las fotos")
    select.add_argument("--min-car-fraction", type=float, default=MIN_CAR_FRACTION)
    sheet = sub.add_parser("sheet", help="hojas de contacto para revisar a ojo")
    sheet.add_argument("--count", type=int, default=24, help="fotos al azar por hoja")
    sheet.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run", help="barrer DAMAGE_MIN_CONF")
    run.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    run.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    run.add_argument("--model", default=MODEL_NAME)
    run.add_argument("--cardd-split", default="test", choices=("train", "valid", "test"))
    run.add_argument("--top", type=int, default=12, help="las fotos sin daño con más confianza, para mirar")
    run.add_argument(
        "--review-conf", type=float, default=0.40, help="umbral desde el que se sortean fotos a revisar"
    )
    run.add_argument("--review-count", type=int, default=24)
    run.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR")
    if args.out_dir is None:
        parser.error("indicá --out-dir o definí EVAL_OUT")
    if args.command == "run" and args.models_dir is None:
        parser.error("indicá --models-dir o definí MODELS_DIR")
    return args


def class_scores(models: Any, decoder: CvImageDecoder, paths: Sequence[Path]) -> list[dict[str, float]]:
    """Por foto, la mayor confianza de cada clase de daño (las clases sin cajas no aparecen)."""
    scores: list[dict[str, float]] = []
    for path in paths:
        best: dict[str, float] = {}
        for box in models.damage.detect(decoder.decode(path.read_bytes())) or []:
            best[box.label] = max(best.get(box.label, 0.0), box.conf)
        scores.append(best)
    return scores


def overall(scores: Sequence[dict[str, float]]) -> list[float]:
    """Mayor confianza de cualquier clase por foto (0 si no hay ninguna caja)."""
    return [max(photo.values(), default=0.0) for photo in scores]


def by_class(
    clean: Sequence[dict[str, float]],
    damaged: Sequence[dict[str, float]],
    damaged_classes: Sequence[set[str]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> dict[str, list[dict[str, float]]]:
    """Por clase y umbral: FP por foto sin daño y recall de las fotos que tienen esa clase anotada."""
    out: dict[str, list[dict[str, float]]] = {}
    for label in CLASS_IDS:
        positives = [
            photo.get(label, 0.0)
            for photo, labels in zip(damaged, damaged_classes, strict=True)
            if label in labels
        ]
        out[label] = [
            {
                "umbral": t,
                "falsos_positivos_por_foto": round(photo_rate([p.get(label, 0.0) for p in clean], t), 4),
                "recall_por_foto": round(photo_rate(positives, t), 4),
            }
            for t in thresholds
        ]
    return out


def do_select(args: argparse.Namespace, folder: Path) -> int:
    source = Path(args.data_dir) / OPEN_IMAGES_NAME
    names = []
    for split in ("train", "valid", "test"):
        annotations = source / split / ANNOTATIONS
        if annotations.is_file():
            coco = json.loads(annotations.read_text(encoding="utf-8"))
            names += [source / split / name for name in closeups(coco, args.min_car_fraction)]
    if not names:
        print(
            f"No hay fotos de autos en {source}. Corré antes tools/training/detector/download.py.",
            file=sys.stderr,
        )
        return 1
    (folder / "images").mkdir(parents=True, exist_ok=True)
    for path in names:
        shutil.copyfile(path, folder / "images" / f"{path.parent.name}_{path.name}")
    print(
        f"{len(names)} fotos con un auto de al menos {args.min_car_fraction:.0%} del cuadro "
        f"en {folder / 'images'}"
    )
    return 0


def do_sheet(args: argparse.Namespace, folder: Path) -> int:
    excluded = read_excluded(folder)
    photos = [p for p in sorted((folder / "images").glob("*.jpg")) if p.name not in excluded]
    sample = random.Random(args.seed).sample(photos, min(args.count, len(photos)))  # noqa: S311
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    target = Path(args.out_dir) / "damage_fp_sheet.jpg"
    cv2.imencode(".jpg", contact_sheet(sample, [p.name[:40] for p in sample]))[1].tofile(str(target))
    print(f"{len(sample)} fotos al azar de {len(photos)} en {target}")
    return 0


def do_run(args: argparse.Namespace, folder: Path) -> int:
    cardd = Path(args.data_dir) / CARDD_NAME / args.cardd_split
    if not (cardd / ANNOTATIONS).is_file():
        print(
            f"Falta el split {args.cardd_split} de CarDD en {cardd}. Corré antes prepare.py.", file=sys.stderr
        )
        return 1
    excluded = read_excluded(folder)
    clean = [p for p in sorted((folder / "images").glob("*.jpg")) if p.name not in excluded]
    if not clean:
        print(f"No hay fotos en {folder / 'images'}. Corré antes `select`.", file=sys.stderr)
        return 1
    try:
        models = build_models(
            {"damage": args.model}, args.manifest, args.models_dir, params={"damage_min_conf": FLOOR}
        )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    coco = json.loads((cardd / ANNOTATIONS).read_text(encoding="utf-8"))
    with_damage = {a["image_id"] for a in coco["annotations"]}
    names = {c["id"]: c["name"].replace(" ", "_") for c in coco["categories"]}
    classes: dict[int, set[str]] = {}
    for ann in coco["annotations"]:
        classes.setdefault(ann["image_id"], set()).add(names[ann["category_id"]])
    damaged_images = [i for i in coco["images"] if i["id"] in with_damage]
    damaged = [cardd / i["file_name"] for i in damaged_images]
    decoder = CvImageDecoder()
    clean_detail = class_scores(models, decoder, clean)
    damaged_detail = class_scores(models, decoder, damaged)
    clean_scores = overall(clean_detail)
    rows = sweep(clean_scores, overall(damaged_detail))
    report = {
        "modelo": models.versions()["damage"],
        "fotos_sin_dano": len(clean),
        "excluidas_por_dano": len(excluded),
        "fotos_con_dano_cardd": len(damaged),
        "cardd_split": args.cardd_split,
        "barrido": rows,
        "propuesta": propose(rows),
        "por_clase": by_class(clean_detail, damaged_detail, [classes[i["id"]] for i in damaged_images]),
    }
    report["umbrales_por_clase"] = {
        f"fp_maximo_{cap:.0%}": {
            "umbrales": (chosen := class_thresholds(report["por_clase"], cap)),
            **combined(clean_detail, damaged_detail, chosen),
        }
        for cap in (0.03, 0.05, 0.10)
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "damage_fp_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flagged = sorted(zip(clean_scores, clean, strict=True), key=lambda item: -item[0])[: args.top]
    cv2.imencode(
        ".jpg", contact_sheet([p for _, p in flagged], [f"{s:.2f} {p.name[:30]}" for s, p in flagged])
    )[1].tofile(str(Path(args.out_dir) / "damage_fp_top.jpg"))
    review = [item for item in zip(clean_scores, clean, strict=True) if item[0] >= args.review_conf]
    review = random.Random(args.seed).sample(review, min(args.review_count, len(review)))  # noqa: S311
    cv2.imencode(
        ".jpg", contact_sheet([p for _, p in review], [f"{s:.2f} {p.name[:30]}" for s, p in review])
    )[1].tofile(str(Path(args.out_dir) / "damage_fp_flagged.jpg"))
    print(
        f"Modelo {report['modelo']} · {len(clean)} fotos sin daño ({len(excluded)} excluidas) · "
        f"{len(damaged)} con daño"
    )
    print(f"{'umbral':>7}{'sin daño con caja':>19}{'FP por foto':>13}{'recall por foto':>17}")
    for row in rows:
        print(
            f"{row['umbral']:>7}{int(row['fotos_sin_dano_con_caja']):>19}"
            f"{row['falsos_positivos_por_foto']:>13.1%}{row['recall_por_foto']:>17.1%}"
        )
    print(f"Propuesta: {report['propuesta']}")
    print("Por clase (FP por foto sin daño / recall por foto de las que tienen esa clase):")
    print(f"{'clase':<15}" + "".join(f"{t:>12}" for t in THRESHOLDS))
    for label, class_rows in report["por_clase"].items():
        cells = "".join(
            f"{r['falsos_positivos_por_foto']:>6.0%}/{r['recall_por_foto']:<5.0%}" for r in class_rows
        )
        print(f"{label:<15}{cells}")
    print("Un umbral por clase (el más bajo con a lo sumo ese FP por foto en cada clase):")
    for name, entry in report["umbrales_por_clase"].items():
        print(
            f"  {name}: {entry['umbrales']} -> FP {entry['falsos_positivos_por_foto']:.1%}, "
            f"recall {entry['recall_por_foto']:.1%}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    folder = Path(args.data_dir) / FOLDER
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (args.data_dir, args.out_dir)):
        print("Las fotos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    return {"select": do_select, "sheet": do_sheet, "run": do_run}[args.command](args, folder)


if __name__ == "__main__":
    sys.exit(main())
