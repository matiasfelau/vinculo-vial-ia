"""Arma, a partir de CarDD (formato COCO), el dataset con la estructura que entrena RF-DETR.

CarDD (PIC Lab, CAS) se usa para investigación con consentimiento previo del titular. No se puede
redistribuir: el dataset nunca va al repositorio, a la imagen de Docker ni a la nube. Este script solo
reordena lo que ya está en tu disco: copia (o enlaza, en el mismo disco) las fotos a
`<data-dir>/cardd-detector/{train,valid,test}` y deja en cada carpeta un `_annotations.coco.json` con
las cajas, sin las máscaras. Uso: ver `tools/training/damage/README.md`.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ANNOTATIONS = "_annotations.coco.json"
DATASET_NAME = "cardd-detector"
SOURCE_NAME = Path("CarDD_release") / "CarDD_COCO"
SPLITS = {"train": "train2017", "valid": "val2017", "test": "test2017"}  # carpeta → carpeta de CarDD
# Las clases de CarDD, en el orden en que las lee el adaptador (`DAMAGE_LABELS`)
CLASSES = ("dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat")


def convert(coco: dict[str, Any]) -> dict[str, Any]:
    """COCO de CarDD solo con cajas. Falla si las categorías no son las de CarDD, en su orden."""
    categories = sorted(coco["categories"], key=lambda category: category["id"])
    if [category["name"] for category in categories] != list(CLASSES):
        raise ValueError(f"las categorías no son las de CarDD: {[c['name'] for c in categories]}")
    annotations = []
    for annotation in coco["annotations"]:
        _, _, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            continue
        annotations.append(
            {
                "id": annotation["id"],
                "image_id": annotation["image_id"],
                "category_id": annotation["category_id"],
                "bbox": annotation["bbox"],
                "area": annotation.get("area", width * height),
                "iscrowd": 0,
            }
        )
    return {
        "info": {"description": "CarDD (PIC Lab, CAS), solo cajas"},
        "images": [
            {key: image[key] for key in ("id", "file_name", "width", "height")} for image in coco["images"]
        ],
        "annotations": annotations,
        "categories": [{"id": c["id"], "name": c["name"], "supercategory": "none"} for c in categories],
    }


def place(source: Path, target: Path) -> None:
    """Enlace duro si el disco lo permite (no ocupa espacio de más); si no, copia."""
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare(source: Path, out_dir: Path) -> dict[str, Any]:
    """Arma los tres splits. Devuelve cifras: fotos y cajas por clase."""
    summary: dict[str, Any] = {}
    for folder_name, source_name in SPLITS.items():
        coco = convert(
            json.loads((source / "annotations" / f"instances_{source_name}.json").read_text("utf-8"))
        )
        folder = out_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for image in coco["images"]:
            place(source / source_name / image["file_name"], folder / image["file_name"])
        (folder / ANNOTATIONS).write_text(json.dumps(coco), encoding="utf-8")
        names = {category["id"]: category["name"] for category in coco["categories"]}
        counts = Counter(names[annotation["category_id"]] for annotation in coco["annotations"])
        summary[folder_name] = {
            "fotos": len(coco["images"]),
            **{name: counts.get(name, 0) for name in CLASSES},
        }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--source", type=Path, help=f"carpeta CarDD_COCO (por defecto <data-dir>/{SOURCE_NAME})"
    )
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR")
    args.source = args.source or Path(args.data_dir) / SOURCE_NAME
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.data_dir) / DATASET_NAME
    if Path(args.data_dir).resolve().is_relative_to(REPO_ROOT) or args.source.resolve().is_relative_to(
        REPO_ROOT
    ):
        print("CarDD no puede estar dentro del repositorio: no se puede redistribuir.", file=sys.stderr)
        return 1
    if not (args.source / "annotations").is_dir():
        print(f"No encuentro las anotaciones de CarDD en {args.source}.", file=sys.stderr)
        return 1
    try:
        summary = prepare(args.source, out_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"No se pudo armar el dataset: {exc}", file=sys.stderr)
        return 1
    for split, stats in summary.items():
        print(f"{split}: {stats}")
    print(f"Listo: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
