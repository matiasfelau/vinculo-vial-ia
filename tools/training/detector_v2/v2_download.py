"""Arma el conjunto de entrenamiento del detector v2 con Open Images V7 (nada de COCO ni de otras fuentes).

Parte de los CSV de anotaciones que ya bajó FiftyOne para v1 (`<data-dir>/fiftyone-cache/open-images-v7`),
elige fotos por etapa (patentes, autos, camiones, vans, colectivos, cada confusora y fotos de fondo), baja
solo esas fotos del bucket público de Open Images y deja un COCO en `<data-dir>/openimages-detector-v2/`.

**Exclusiones** (nunca entran): la partición de prueba de v1 (`openimages-detector/test`), las 1.200 negativas
nuevas (`--fresh-negatives`) y las 489 negativas de v1 (fotos de `openimages-detector` sin `Car`, `Wheel` ni
patente anotados). Se excluyen **antes** de elegir y se **verifican después** sobre los archivos bajados:
por identificador, por SHA-256 del archivo y por el MD5 original que publica Open Images (que atrapa la
misma foto bajo otro identificador). El resultado queda en `exclusion_check.json`.

Las fotos de **fondo** no tienen ninguna caja de las clases de v2 ni de la familia `Vehicle` (ni ruedas,
neumáticos o patentes) anotada. Open Images no es exhaustivo: alguna puede tener un auto sin anotar.

Todo se guarda fuera del repo: son fotos de terceros. Uso: ver `tools/training/detector_v2/README.md`.
"""

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from v2_classes import (
    BACKGROUNDS,
    CLASSES,
    OPEN_IMAGES_CLASSES,
    STAGES,
    VEHICLE_FAMILY,
    VEHICLE_PARTS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "openimages-detector-v2"
V1_DATASET = "openimages-detector"
ANNOTATIONS = "_annotations.coco.json"
SPLITS = {"train": "train", "valid": "validation"}  # carpeta → split de Open Images
IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"
MIN_SIDE_PX = 8
FIRST_CATEGORY_ID = 1
V1_NEGATIVE_EXCLUDED = ("car", "plate", "wheel")  # lo que no puede tener anotado una negativa de v1
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_THREADS = 16
LICENSE_NOTE = (
    "Open Images V7: anotaciones CC BY 4.0, imágenes CC BY 2.0. Citar: Kuznetsova et al., "
    "The Open Images Dataset V4, IJCV 2020. No se redistribuye."
)


@dataclass(frozen=True)
class Box:
    """Una caja de Open Images con el nombre de su clase y sus atributos (coordenadas normalizadas)."""

    label: str
    x: float
    y: float
    width: float
    height: float
    group_of: bool = False
    depiction: bool = False
    inside: bool = False


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    width: int
    height: int
    boxes: Sequence[Box]


def is_usable(box: Box, width: int, height: int, min_side: int = MIN_SIDE_PX) -> bool:
    """Una caja entra al dataset si es de una clase de v2 y no es ambigua ni diminuta."""
    return (
        box.label in OPEN_IMAGES_CLASSES
        and not (box.group_of or box.depiction or box.inside)
        and box.width * width >= min_side
        and box.height * height >= min_side
    )


def to_coco(records: Iterable[ImageRecord], min_side: int = MIN_SIDE_PX) -> dict[str, Any]:
    """COCO con las cajas utilizables. Una foto sin cajas se queda **solo si es de fondo** (`background`).

    Las fotos de etapa que quedan sin ninguna caja utilizable se descartan; las de fondo se conservan sin
    anotaciones: así el entrenamiento aprende que ahí no hay nada.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_id = {name: FIRST_CATEGORY_ID + index for index, name in enumerate(CLASSES)}
    for record in records:
        kept = [b for b in record.boxes if is_usable(b, record.width, record.height, min_side)]
        has_any_box = bool(record.boxes)
        if not kept and has_any_box:
            continue
        image_id = len(images) + 1
        images.append(
            {"id": image_id, "file_name": record.file_name, "width": record.width, "height": record.height}
        )
        for box in kept:
            x1 = max(0.0, box.x * record.width)
            y1 = max(0.0, box.y * record.height)
            x2 = min(float(record.width), (box.x + box.width) * record.width)
            y2 = min(float(record.height), (box.y + box.height) * record.height)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id[OPEN_IMAGES_CLASSES[box.label]],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "area": round((x2 - x1) * (y2 - y1), 2),
                    "iscrowd": 0,
                }
            )
    return {
        "info": {"description": "Subconjunto de Open Images V7 para el detector v2"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id[name], "name": name, "supercategory": "none"} for name in CLASSES],
    }


def class_counts(coco: Mapping[str, Any]) -> dict[str, int]:
    names = {category["id"]: category["name"] for category in coco["categories"]}
    counts = Counter(names[a["category_id"]] for a in coco["annotations"])
    return {name: counts.get(name, 0) for name in CLASSES}


# --- anotaciones de Open Images ---------------------------------------------------------------------


def load_class_ids(classes_csv: Path) -> dict[str, str]:
    """Nombre de Open Images → MID (`/m/0k4j`)."""
    with classes_csv.open(encoding="utf-8", newline="") as handle:
        return {row[1]: row[0] for row in csv.reader(handle) if len(row) >= 2}


@dataclass
class Annotations:
    """Lo que hace falta del CSV de cajas para elegir fotos sin cargarlo entero."""

    boxes: dict[str, list[Box]]  # por foto, solo las cajas de las clases de v2
    vehicle_images: set[str]  # fotos con algo de la familia vehículo (o ruedas, neumáticos o patentes)
    all_images: set[str]  # toda foto que tiene alguna caja


def read_annotations(detections_csv: Path, class_ids: Mapping[str, str]) -> Annotations:
    """Recorre el CSV de cajas de un split (pandas, por tramos)."""
    import pandas as pd  # se importa acá: solo hace falta para bajar datos

    names = {class_ids[name]: name for name in OPEN_IMAGES_CLASSES if name in class_ids}
    blocked = {class_ids[name] for name in (*VEHICLE_FAMILY, *VEHICLE_PARTS) if name in class_ids}
    boxes: dict[str, list[Box]] = {}
    vehicle_images: set[str] = set()
    all_images: set[str] = set()
    columns = [
        "ImageID",
        "LabelName",
        "XMin",
        "XMax",
        "YMin",
        "YMax",
        "IsGroupOf",
        "IsDepiction",
        "IsInside",
    ]
    for chunk in pd.read_csv(detections_csv, usecols=columns, chunksize=2_000_000):
        all_images.update(chunk["ImageID"].unique())
        vehicle_images.update(chunk.loc[chunk["LabelName"].isin(blocked), "ImageID"].unique())
        for row in chunk[chunk["LabelName"].isin(names)].itertuples(index=False):
            boxes.setdefault(row.ImageID, []).append(
                Box(
                    label=names[row.LabelName],
                    x=float(row.XMin),
                    y=float(row.YMin),
                    width=float(row.XMax - row.XMin),
                    height=float(row.YMax - row.YMin),
                    group_of=bool(row.IsGroupOf),
                    depiction=bool(row.IsDepiction),
                    inside=bool(row.IsInside),
                )
            )
    return Annotations(boxes, vehicle_images, all_images)


# --- exclusiones ------------------------------------------------------------------------------------


def v1_negative_ids(v1_dir: Path) -> set[str]:
    """Las negativas de v1: fotos de sus tres splits sin `car`, `plate` ni `wheel` anotados (las 489)."""
    found: set[str] = set()
    for split in ("train", "valid", "test"):
        path = v1_dir / split / ANNOTATIONS
        if not path.is_file():
            continue
        coco = json.loads(path.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in coco["categories"]}
        present: dict[int, set[str]] = {}
        for annotation in coco["annotations"]:
            present.setdefault(annotation["image_id"], set()).add(names[annotation["category_id"]])
        found.update(
            Path(i["file_name"]).stem
            for i in coco["images"]
            if not present.get(i["id"], set()) & set(V1_NEGATIVE_EXCLUDED)
        )
    return found


def stems(folder: Path) -> set[str]:
    return {p.stem for p in folder.glob("*.jpg")} if folder.is_dir() else set()


def excluded_ids(data_dir: Path, fresh_negatives: Path | None) -> dict[str, set[str]]:
    """Los tres grupos de exclusión, por identificador de Open Images (el nombre del archivo sin `.jpg`)."""
    v1_dir = data_dir / V1_DATASET
    return {
        "prueba_de_v1": stems(v1_dir / "test"),
        "negativas_nuevas": stems(fresh_negatives) if fresh_negatives else set(),
        "negativas_de_v1": v1_negative_ids(v1_dir),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_md5s(image_ids_csvs: Iterable[Path], wanted: set[str]) -> dict[str, str]:
    """`OriginalMD5` (base64) de Open Images para los identificadores pedidos."""
    found: dict[str, str] = {}
    for path in image_ids_csvs:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["ImageID"] in wanted:
                    found[row["ImageID"]] = row["OriginalMD5"]
    return found


def verify_exclusion(
    selected: Mapping[str, Path],
    excluded: Mapping[str, set[str]],
    excluded_files: Mapping[str, Path],
    md5s: Mapping[str, str],
) -> dict[str, Any]:
    """Compara lo elegido con lo excluido por identificador, por SHA-256 y por MD5 original.

    `selected` y `excluded_files` son `{identificador: archivo}`; los archivos excluidos que no estén en
    disco quedan sin comparar por contenido y se informan (`sin_archivo`).
    """
    all_excluded = set().union(*excluded.values()) if excluded else set()
    by_id = sorted(set(selected) & all_excluded)
    excluded_hashes = {file_sha256(path): ident for ident, path in excluded_files.items() if path.is_file()}
    by_sha = sorted(ident for ident, path in selected.items() if file_sha256(path) in excluded_hashes)
    excluded_md5 = {md5s[i] for i in all_excluded if i in md5s}
    by_md5 = sorted(ident for ident in selected if md5s.get(ident) in excluded_md5)
    return {
        "elegidas": len(selected),
        "excluidas_por_grupo": {name: len(ids) for name, ids in excluded.items()},
        "excluidas_total": len(all_excluded),
        "archivos_excluidos_comparados_por_contenido": len(excluded_hashes),
        "excluidas_sin_archivo": sorted(i for i in all_excluded if i not in excluded_files)[:20],
        "excluidas_sin_archivo_cantidad": sum(i not in excluded_files for i in all_excluded),
        "coincidencias_por_identificador": by_id,
        "coincidencias_por_sha256": by_sha,
        "coincidencias_por_md5_original": by_md5,
        "md5_original_conocidos": {
            "elegidas": sum(i in md5s for i in selected),
            "excluidas": len(excluded_md5),
        },
        "ok": not (by_id or by_sha or by_md5),
    }


# --- elección y descarga ----------------------------------------------------------------------------


def stage_candidates(annotations: Annotations, names: Sequence[str]) -> list[str]:
    """Fotos con al menos una caja (no ambigua) de alguna de las clases de la etapa."""
    wanted = set(names)
    return sorted(
        image_id
        for image_id, boxes in annotations.boxes.items()
        if any(b.label in wanted and not (b.group_of or b.depiction or b.inside) for b in boxes)
    )


def background_candidates(annotations: Annotations) -> list[str]:
    """Fotos con alguna caja anotada pero ninguna de las clases de v2 ni de la familia vehículo."""
    return sorted(annotations.all_images - annotations.vehicle_images - set(annotations.boxes))


def select_images(
    annotations: Annotations,
    excluded: set[str],
    split_index: int,
    seed: int,
    scale: float = 1.0,
) -> dict[str, list[str]]:
    """Fotos por etapa de un split (0 = entrenamiento, 1 = validación), sin repetidas ni excluidas."""
    rng = random.Random(seed)  # noqa: S311 - muestreo de fotos, no criptografía
    taken: set[str] = set()
    chosen: dict[str, list[str]] = {}
    for stage, (names, *counts) in STAGES.items():
        pool = [i for i in stage_candidates(annotations, names) if i not in excluded and i not in taken]
        rng.shuffle(pool)
        chosen[stage] = pool[: max(1, round(counts[split_index] * scale))]
        taken.update(chosen[stage])
    pool = [i for i in background_candidates(annotations) if i not in excluded and i not in taken]
    rng.shuffle(pool)
    chosen["backgrounds"] = pool[: max(1, round(BACKGROUNDS[split_index] * scale))]
    return chosen


def download(image_id: str, split: str, target: Path) -> bool:
    """Baja una foto del bucket público de Open Images. `False` si no se pudo."""
    if target.is_file() and target.stat().st_size > 0:
        return True
    url = IMAGE_URL.format(split=split, image_id=image_id)
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - URL fija de Open Images
                data = response.read()
            target.write_bytes(data)
            return True
        except Exception:  # noqa: BLE001 - red: se reintenta y se informa
            time.sleep(1 + attempt)
    return False


def download_all(ids: Sequence[str], split: str, folder: Path) -> set[str]:
    folder.mkdir(parents=True, exist_ok=True)
    ok: set[str] = set()
    with concurrent.futures.ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        futures = {pool.submit(download, i, split, folder / f"{i}.jpg"): i for i in ids}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if future.result():
                ok.add(futures[future])
            if done % 200 == 0:
                print(f"  {done}/{len(ids)} bajadas", flush=True)
    return ok


def image_size(path: Path) -> tuple[int, int] | None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 - un archivo roto se descarta
        return None


def md5_base64(path: Path) -> str:
    return base64.b64encode(hashlib.md5(path.read_bytes(), usedforsecurity=False).digest()).decode()  # noqa: S324


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"))
    parser.add_argument(
        "--fresh-negatives", type=Path, required=True, help="carpeta de las 1.200 negativas nuevas"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="fracción del tamaño de cada etapa (0,05 para una prueba rápida)",
    )
    parser.add_argument("--force", action="store_true", help="rehacer un split que ya existe")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if data_dir.resolve().is_relative_to(REPO_ROOT):
        print("La carpeta de datos no puede estar dentro del repositorio.", file=sys.stderr)
        return 1
    cache = data_dir / "fiftyone-cache" / "open-images-v7"
    out_dir = data_dir / DATASET_NAME
    excluded = excluded_ids(data_dir, args.fresh_negatives)
    all_excluded = set().union(*excluded.values())
    print({name: len(ids) for name, ids in excluded.items()}, "excluidas en total:", len(all_excluded))
    summary: dict[str, Any] = {
        "licencia": LICENSE_NOTE,
        "semilla": args.seed,
        "escala": args.scale,
        "clases": list(CLASSES),
        "correspondencia_open_images": dict(OPEN_IMAGES_CLASSES),
        "exclusiones": {name: len(ids) for name, ids in excluded.items()},
    }
    selected_files: dict[str, Path] = {}
    for index, (folder_name, oi_split) in enumerate(SPLITS.items()):
        if folder_name not in args.splits:
            continue
        folder = out_dir / folder_name
        if (folder / ANNOTATIONS).is_file() and not args.force:
            print(f"{folder_name}: ya existe, se omite (usá --force para rehacerlo)")
            selected_files.update({p.stem: p for p in folder.glob("*.jpg")})
            continue
        class_ids = load_class_ids(cache / oi_split / "metadata" / "classes.csv")
        print(f"{folder_name}: leyendo las anotaciones de Open Images ({oi_split})...", flush=True)
        annotations = read_annotations(cache / oi_split / "labels" / "detections.csv", class_ids)
        chosen = select_images(annotations, all_excluded, index, args.seed, args.scale)
        ids = sorted({i for stage_ids in chosen.values() for i in stage_ids})
        print(
            f"{folder_name}: {len(ids)} fotos elegidas { ({s: len(v) for s, v in chosen.items()}) }",
            flush=True,
        )
        ok = download_all(ids, oi_split, folder)
        records: list[ImageRecord] = []
        backgrounds = set(chosen["backgrounds"])
        for image_id in ids:
            size = image_size(folder / f"{image_id}.jpg") if image_id in ok else None
            if size is None:
                continue
            boxes = [] if image_id in backgrounds else annotations.boxes.get(image_id, [])
            records.append(ImageRecord(f"{image_id}.jpg", size[0], size[1], boxes))
        coco = to_coco(records)
        kept = {i["file_name"] for i in coco["images"]}
        for path in folder.glob("*.jpg"):
            if path.name not in kept:
                path.unlink()
        (folder / ANNOTATIONS).write_text(json.dumps(coco), encoding="utf-8")
        selected_files.update({p.stem: p for p in folder.glob("*.jpg")})
        summary[folder_name] = {
            "fotos": len(coco["images"]),
            "fotos_por_etapa": {stage: len(v) for stage, v in chosen.items()},
            "fotos_de_fondo": len(backgrounds & {Path(n).stem for n in kept}),
            "no_se_pudieron_bajar": len(ids) - len(ok),
            **class_counts(coco),
        }
        print(f"{folder_name}: {summary[folder_name]}", flush=True)
    # Verificación posterior, sobre los archivos que quedaron en disco
    excluded_files: dict[str, Path] = {}
    for source in (
        data_dir / V1_DATASET / "test",
        args.fresh_negatives,
        *(data_dir / V1_DATASET / s for s in ("train", "valid")),
    ):
        excluded_files.update({p.stem: p for p in Path(source).glob("*.jpg") if p.stem in all_excluded})
    md5s = original_md5s(
        (cache / s / "metadata" / "image_ids.csv" for s in ("train", "validation", "test")),
        all_excluded | set(selected_files),
    )
    check = verify_exclusion(selected_files, excluded, excluded_files, md5s)
    (out_dir / "exclusion_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    summary["verificacion_de_exclusion"] = {k: check[k] for k in ("ok", "elegidas", "excluidas_total")}
    (out_dir / "dataset_info.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Verificación de exclusión: ok={check['ok']} · {check['elegidas']} elegidas · Listo: {out_dir}")
    return 0 if check["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
