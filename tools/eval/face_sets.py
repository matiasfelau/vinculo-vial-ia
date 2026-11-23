"""Arma el conjunto de fotos de `face.py` (módulo experimental de rostros): el split de prueba de Open Images.

- `openimages`: las fotos del split de prueba del detector (`<data-dir>/openimages-detector/test`) con solo la
  clase `face`, en `<eval-dir>/face-oi-test`.

La consola muestra solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = "_annotations.coco.json"
OPEN_IMAGES_SOURCE = Path("openimages-detector") / "test"
OPEN_IMAGES_FOLDER = "face-oi-test"
CATEGORIES = [{"id": 1, "name": "face"}]


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--eval-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    sub = parser.add_subparsers(dest="command", required=True)
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
