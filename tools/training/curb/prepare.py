"""Arma el dataset de rebajes para RF-DETR-Seg a partir de la exportación de CVAT y de `fotos.csv`.

Toma las anotaciones (polígonos `curb` y `curb_cut`) de la exportación "COCO 1.0" de CVAT, reparte las
fotos en train, valid y test por calle (ver `splits.py`) y deja el dataset en `<curb>/dataset/`. Las fotos
se copian (o enlazan) desde `raw/`, que no se toca. Todo queda fuera del repo: son fotos de la calle con
rostros y patentes de terceros. Uso: ver `tools/training/curb/README.md`.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from photos import (
    CAR_NEXT_TO_RAMP,
    CLEAR_POSITIVE,
    MARGINAL_POSITIVE,
    RAMP_WITHOUT_CAR,
    CurbPhoto,
    PhotosError,
    load_photos,
)
from splits import Assignment, SplitError, assign, load_fixed_test, save_test, write_assignments

REPO_ROOT = Path(__file__).resolve().parents[3]
CURB_FOLDER = "curb"
ANNOTATIONS = "_annotations.coco.json"
CLASSES = ("curb", "curb_cut")  # el orden de las columnas del modelo y del adaptador
DATASET_FOLDER = "dataset"
MIN_POINTS = 6  # un polígono tiene al menos 3 puntos (x, y)
# Tipos de foto que muestran un rebaje: tienen que traer al menos un polígono `curb_cut`
SHOW_A_CUT = (CLEAR_POSITIVE, MARGINAL_POSITIVE, CAR_NEXT_TO_RAMP, RAMP_WITHOUT_CAR)


class ExportError(ValueError):
    """La exportación de CVAT no sirve."""


def find_export(folder: Path) -> Path:
    """El JSON de anotaciones de una exportación "COCO 1.0" de CVAT (descomprimida en `folder`)."""
    if folder.is_file():
        return folder
    candidates = sorted(folder.rglob("instances_*.json")) or sorted(folder.rglob("*.json"))
    if not candidates:
        raise ExportError(f"no encontré el JSON de anotaciones en {folder.name}")
    return candidates[0]


def usable_polygons(annotation: dict[str, Any]) -> list[list[float]] | None:
    """Los polígonos de una anotación, o `None` si no sirve (sin polígonos, una máscara RLE o sin área)."""
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return None  # una máscara RLE: CVAT la usa para anotaciones de tipo máscara, no polígono
    polygons = [list(map(float, polygon)) for polygon in segmentation if len(polygon) >= MIN_POINTS]
    return polygons or None


def convert(
    export: dict[str, Any], photos: list[CurbPhoto], names: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    """Anotaciones de cada foto, ya con las clases en el orden del modelo.

    `names` mapea el nombre de archivo de la foto al de la imagen en la exportación de CVAT. Devuelve
    `{foto: {"width", "height", "annotations": [...]}}` y cuántas anotaciones se descartaron, por motivo.
    """
    labels = {category["id"]: category["name"] for category in export["categories"]}
    missing = [name for name in CLASSES if name not in labels.values()]
    if missing:
        raise ExportError(
            f"la exportación no tiene las clases {', '.join(missing)} (tiene {sorted(labels.values())})"
        )
    class_id = {name: index + 1 for index, name in enumerate(CLASSES)}
    images = {image["id"]: image for image in export["images"]}
    wanted = {names[photo.file]: photo.file for photo in photos}
    result: dict[str, dict[str, Any]] = {}
    for image in images.values():
        if image["file_name"] in wanted:
            result[wanted[image["file_name"]]] = {
                "width": image["width"],
                "height": image["height"],
                "annotations": [],
            }
    discarded: Counter[str] = Counter()
    for annotation in export["annotations"]:
        image = images[annotation["image_id"]]
        target = result.get(wanted.get(image["file_name"], ""))
        if target is None:
            continue
        name = labels[annotation["category_id"]]
        if name not in class_id:
            discarded[f"clase {name}"] += 1
            continue
        polygons = usable_polygons(annotation)
        if polygons is None:
            discarded["sin polígono"] += 1
            continue
        target["annotations"].append(
            {
                "category_id": class_id[name],
                "segmentation": polygons,
                "bbox": annotation["bbox"],
                "area": annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]),
            }
        )
    return result, discarded


def missing_cuts(assignment: Assignment, converted: dict[str, dict[str, Any]]) -> list[str]:
    """Fotos que por su tipo muestran un rebaje y no tienen ningún `curb_cut` anotado."""
    cut = CLASSES.index("curb_cut") + 1
    return sorted(
        photo.file
        for photos in assignment.splits.values()
        for photo in photos
        if photo.kind in SHOW_A_CUT
        and not any(a["category_id"] == cut for a in converted[photo.file]["annotations"])
    )


def match_names(export: dict[str, Any], photos: list[CurbPhoto]) -> dict[str, str]:
    """Nombre de cada foto del CSV en la exportación (CVAT puede sumar una carpeta: se compara el base)."""
    by_base = {Path(image["file_name"]).name: image["file_name"] for image in export["images"]}
    missing = [photo.file for photo in photos if photo.file not in by_base]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise ExportError(f"{len(missing)} fotos de fotos.csv no están en la exportación de CVAT: {shown}")
    return {photo.file: by_base[photo.file] for photo in photos}


def place(source: Path, target: Path) -> None:
    """Enlace duro si el disco lo permite; si no, copia."""
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_dataset(
    assignment: Assignment, converted: dict[str, dict[str, Any]], raw_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Escribe `train/`, `valid/` y `test/` con sus fotos y su `_annotations.coco.json`."""
    summary: dict[str, Any] = {}
    for split, photos in assignment.splits.items():
        folder = out_dir / split
        folder.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for photo in photos:
            data = converted[photo.file]
            image_id = len(images) + 1
            images.append(
                {"id": image_id, "file_name": photo.file, "width": data["width"], "height": data["height"]}
            )
            for annotation in data["annotations"]:
                annotations.append(
                    {"id": len(annotations) + 1, "image_id": image_id, "iscrowd": 0, **annotation}
                )
            place(raw_dir / photo.file, folder / photo.file)
        coco = {
            "info": {"description": "Rebajes y cordones"},
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(CLASSES)
            ],
        }
        (folder / ANNOTATIONS).write_text(json.dumps(coco), encoding="utf-8")
        counts = Counter(CLASSES[a["category_id"] - 1] for a in annotations)
        summary[split] = {"fotos": len(images), **{name: counts.get(name, 0) for name in CLASSES}}
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument("--export", type=Path, help="exportación de CVAT (por defecto <curb>/cvat-export)")
    parser.add_argument(
        "--seed", type=int, default=0, help="semilla del reparto (el test ya guardado no cambia)"
    )
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí EVAL_DATA_DIR")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    curb = Path(args.data_dir) / CURB_FOLDER
    if curb.resolve().is_relative_to(REPO_ROOT):
        print("Las fotos y los resultados no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    try:
        photos = load_photos(curb / "fotos.csv", curb / "raw")
        export = json.loads(find_export(args.export or curb / "cvat-export").read_text(encoding="utf-8"))
        names = match_names(export, photos)
        converted, discarded = convert(export, photos, names)
        fixed = load_fixed_test(curb / "splits")
        assignment = assign(photos, args.seed, fixed)
    except PhotosError as exc:
        print(
            "fotos.csv tiene errores:\n" + "\n".join(f"  {problem}" for problem in exc.problems),
            file=sys.stderr,
        )
        return 1
    except (ExportError, SplitError, OSError, json.JSONDecodeError) as exc:
        print(f"No se pudo armar el dataset: {exc}", file=sys.stderr)
        return 1
    save_test(curb / "splits", assignment, args.seed)
    write_assignments(curb / "splits", assignment)
    summary = write_dataset(assignment, converted, curb / "raw", curb / DATASET_FOLDER)
    summary["split_por_calle"] = assignment.summary()
    summary["anotaciones_descartadas"] = dict(discarded)
    without_cut = missing_cuts(assignment, converted)
    summary["fotos_sin_curb_cut_esperado"] = len(without_cut)
    (curb / DATASET_FOLDER / "dataset_info.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for split in ("train", "valid", "test"):
        print(f"{split}: {summary[split]} · tipos {assignment.summary()[split]['tipos']}")
    if assignment.left_out:
        print(
            f"Atención: {len(assignment.left_out)} fotos nuevas son de calles del test y quedaron "
            "afuera de todo."
        )
    if discarded:
        print(f"Anotaciones descartadas: {dict(discarded)}")
    if without_cut:
        shown = ", ".join(without_cut[:5]) + (" ..." if len(without_cut) > 5 else "")
        print(
            f"Atención: {len(without_cut)} fotos de un tipo con rebaje no tienen ningún curb_cut "
            f"anotado: {shown}"
        )
    print(f"Listo: {curb / DATASET_FOLDER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
