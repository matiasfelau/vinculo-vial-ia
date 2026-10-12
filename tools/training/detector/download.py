"""Baja de Open Images V7, con FiftyOne, un subconjunto para entrenar el detector y lo deja en COCO.

Las cuatro clases (Car, Wheel, Vehicle registration plate y Human face) se bajan en tres etapas por
split, para que las patentes, que son lo más escaso, no queden casi sin ejemplos: primero fotos con
patentes, después con rostros y después con autos. Cada foto conserva **todas** sus cajas de las cuatro
clases, no solo las de la etapa en la que se bajó: si no, quedarían rostros y patentes sin anotar.

El tamaño es configurable para que entre en la GTX 1660 y en Colab. Todo se guarda fuera del repo:
las fotos tienen rostros y patentes de terceros. Uso: ver `tools/training/detector/README.md`.
"""

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from dataset import STAGES, Annotation, ImageRecord, class_counts, to_coco
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "openimages-detector"
SPLITS = {"train": "train", "valid": "validation", "test": "test"}  # carpeta → split de Open Images
DEFAULT_COUNTS = {"train": (500, 400, 400), "valid": (100, 80, 80), "test": (100, 80, 80)}
ANNOTATIONS = "_annotations.coco.json"
LICENSE_NOTE = (
    "Open Images V7: anotaciones CC BY 4.0, imágenes CC BY 2.0. Citar: Kuznetsova et al., "
    "The Open Images Dataset V4, IJCV 2020. No se redistribuye."
)


def fetch_split(oi_split: str, counts: Sequence[int], seed: int) -> list[tuple[ImageRecord, Path]]:
    """Fotos de un split, con todas sus cajas, sin repetir las que aparecen en más de una etapa."""
    found: dict[str, tuple[ImageRecord, Path]] = {}
    for (stage, oi_class), count in zip(STAGES.items(), counts, strict=True):
        if count <= 0:
            continue
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=oi_split,
            label_types=["detections"],
            classes=[oi_class],
            max_samples=count,
            seed=seed,
            shuffle=True,
            dataset_name=f"vvi-{oi_split}-{stage}",
            drop_existing_dataset=True,
        )
        for sample in dataset:
            path = Path(sample.filepath)
            if path.name in found:
                continue
            with Image.open(path) as image:
                width, height = image.size
            annotations = [
                Annotation(
                    label=detection.label,
                    x=detection.bounding_box[0],
                    y=detection.bounding_box[1],
                    width=detection.bounding_box[2],
                    height=detection.bounding_box[3],
                    group_of=bool(detection["IsGroupOf"]),
                    depiction=bool(detection["IsDepiction"]),
                    inside=bool(detection["IsInside"]),
                )
                for detection in (sample.ground_truth.detections if sample.ground_truth else [])
            ]
            found[path.name] = (ImageRecord(path.name, width, height, annotations), path)
    return list(found.values())


def write_split(folder: Path, items: Sequence[tuple[ImageRecord, Path]]) -> dict[str, int]:
    """Copia las fotos y escribe `_annotations.coco.json`. Devuelve cifras del split."""
    folder.mkdir(parents=True, exist_ok=True)
    coco = to_coco(record for record, _ in items)
    kept = {image["file_name"] for image in coco["images"]}
    for record, source in items:
        if record.file_name in kept:
            shutil.copy2(source, folder / record.file_name)
    (folder / ANNOTATIONS).write_text(json.dumps(coco), encoding="utf-8")
    return {"fotos": len(coco["images"]), **class_counts(coco)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("TRAIN_DATA_DIR"),
        help=f"carpeta de datos de entrenamiento (TRAIN_DATA_DIR); el subconjunto va en {DATASET_NAME}/",
    )
    for split, counts in DEFAULT_COUNTS.items():
        parser.add_argument(
            f"--{split}",
            nargs=3,
            type=int,
            metavar=("PATENTES", "ROSTROS", "AUTOS"),
            default=counts,
            help=f"fotos de {split} por etapa (por defecto {' '.join(map(str, counts))})",
        )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="volver a armar los splits que ya existen")
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.data_dir) / DATASET_NAME
    if Path(args.data_dir).resolve().is_relative_to(REPO_ROOT):
        print("La carpeta de datos no puede estar dentro del repositorio.", file=sys.stderr)
        return 1
    fo.config.dataset_zoo_dir = str(Path(args.data_dir) / "fiftyone-cache")
    summary: dict[str, object] = {"licencia": LICENSE_NOTE, "semilla": args.seed}
    for folder_name, oi_split in SPLITS.items():
        folder = out_dir / folder_name
        if (folder / ANNOTATIONS).exists() and not args.force:
            print(f"{folder_name}: ya existe, se omite (usá --force para rehacerlo)")
            continue
        counts = getattr(args, folder_name)
        print(
            f"{folder_name}: bajando {counts[0]} fotos con patentes, {counts[1]} con rostros, "
            f"{counts[2]} con autos..."
        )
        stats = write_split(folder, fetch_split(oi_split, counts, args.seed))
        summary[folder_name] = stats
        print(f"{folder_name}: {stats}")
    if len(summary) > 2:
        (out_dir / "dataset_info.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"Listo: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
