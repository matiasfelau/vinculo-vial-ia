"""Arma los conjuntos de fotos para `face.py`: WIDER FACE (validación) y el split de prueba de Open Images.

- `wider`: baja la validación de WIDER FACE de Hugging Face (si no está) y deja una muestra al azar
  de fotos con caras válidas, con sus anotaciones en COCO, en `<eval-dir>/face-wider`. WIDER FACE es solo
  referencia académica (CC BY-NC-ND): nunca para entrenar y siempre fuera del repo.
- `openimages`: las fotos del split de prueba del detector (`<data-dir>/openimages-detector/test`) con solo la
  clase `face`, en `<eval-dir>/face-oi-test`.

Se descartan las caras que WIDER marca como inválidas y las de un lado menor a `--min-side` píxeles. La
consola muestra solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = "_annotations.coco.json"
WIDER_FOLDER = "face-wider"
OPEN_IMAGES_SOURCE = Path("openimages-detector") / "test"
OPEN_IMAGES_FOLDER = "face-oi-test"
WIDER_REPO = "CUHK-CSE/wider_face"
WIDER_PARQUET = "default/validation/0000.parquet"
WIDER_REVISION = "refs/convert/parquet"
DEFAULT_SAMPLE = 1000
DEFAULT_MIN_SIDE = 8
CATEGORIES = [{"id": 1, "name": "face"}]


def valid_boxes(faces: Mapping[str, Any], min_side: int) -> list[list[float]]:
    """Cajas `[x, y, ancho, alto]` de WIDER sin las inválidas ni las de un lado menor a `min_side`."""
    return [
        [float(v) for v in box]
        for box, invalid in zip(faces["bbox"], faces["invalid"], strict=True)
        if not invalid and min(box[2], box[3]) >= min_side
    ]


def coco_entry(
    images: list[dict[str, Any]], annotations: list[dict[str, Any]], file_name: str, size: tuple[int, int],
    boxes: Sequence[Sequence[float]],
) -> None:  # fmt: skip
    """Suma una foto y sus cajas a las listas de un COCO de una sola clase."""
    image_id = len(images) + 1
    images.append({"id": image_id, "file_name": file_name, "width": size[0], "height": size[1]})
    for x, y, w, h in boxes:
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )


def build_wider(parquet: Path, out: Path, sample: int, seed: int, min_side: int) -> tuple[int, int]:
    """Muestra `sample` fotos al azar del parquet; devuelve (fotos, caras) con caras válidas."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet)
    rows = [
        (group, i)
        for group in range(table.num_row_groups)
        for i in range(table.metadata.row_group(group).num_rows)
    ]
    chosen = set(random.Random(seed).sample(range(len(rows)), min(sample, len(rows))))  # noqa: S311
    (out / "images").mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    index = 0
    for group in range(table.num_row_groups):
        for item in table.read_row_group(group, columns=["image", "faces"]).to_pylist():
            if index in chosen and (boxes := valid_boxes(item["faces"], min_side)):
                data = item["image"]["bytes"]
                height, width = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2]
                name = f"w{index:05d}.jpg"
                (out / "images" / name).write_bytes(data)
                coco_entry(images, annotations, f"images/{name}", (width, height), boxes)
            index += 1
    (out / ANNOTATIONS).write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}), encoding="utf-8"
    )
    return len(images), len(annotations)


def build_open_images(source: Path, out: Path) -> tuple[int, int]:
    """Copia las fotos del split y deja solo la clase `face`; devuelve (fotos, caras)."""
    coco = json.loads((source / ANNOTATIONS).read_text(encoding="utf-8"))
    face_ids = {c["id"] for c in coco["categories"] if c["name"] == "face"}
    (out / "images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for info in coco["images"]:
        name = Path(info["file_name"]).name
        shutil.copyfile(source / info["file_name"], out / "images" / name)
        images.append({**info, "file_name": f"images/{name}"})
    annotations = [
        {**a, "category_id": 1, "iscrowd": 0} for a in coco["annotations"] if a["category_id"] in face_ids
    ]
    (out / ANNOTATIONS).write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}), encoding="utf-8"
    )
    return len(images), len(annotations)


def download_wider(target: Path) -> Path:
    """El parquet de validación de WIDER FACE (convertido por Hugging Face), si todavía no está."""
    if not target.is_file():
        from huggingface_hub import hf_hub_download

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            hf_hub_download(WIDER_REPO, WIDER_PARQUET, repo_type="dataset", revision=WIDER_REVISION), target
        )
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--eval-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    wider = sub.add_parser("wider", help="muestra de WIDER FACE (validación)")
    wider.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="fotos de la muestra")
    wider.add_argument("--seed", type=int, default=0)
    wider.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE, help="lado menor mínimo de una cara")
    sub.add_parser("openimages", help="split de prueba de Open Images, solo rostros")
    args = parser.parse_args(argv)
    if args.eval_dir is None:
        parser.error("indicá --eval-dir o definí EVAL_DATA_DIR")
    if args.command == "openimages" and args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR (ahí está el subset de Open Images)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.eval_dir)
    if root.resolve().is_relative_to(REPO_ROOT):
        print("Las fotos no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    if args.command == "wider":
        parquet = download_wider(root / WIDER_FOLDER / "validation.parquet")
        images, faces = build_wider(parquet, root / WIDER_FOLDER, args.sample, args.seed, args.min_side)
        print(f"WIDER FACE: {images} fotos y {faces} caras en {root / WIDER_FOLDER}")
        return 0
    source = Path(args.data_dir) / OPEN_IMAGES_SOURCE
    if not (source / ANNOTATIONS).is_file():
        print(
            f"Falta {ANNOTATIONS} en {source}. Corré antes tools/training/detector/download.py.",
            file=sys.stderr,
        )
        return 1
    images, faces = build_open_images(source, root / OPEN_IMAGES_FOLDER)
    print(f"Open Images (prueba): {images} fotos y {faces} caras en {root / OPEN_IMAGES_FOLDER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
