"""Sets de evaluación de R1 con el encuadre de la app (el usuario fotografía a 1-2 m del vehículo).

Arma, con fotos de la **partición de prueba de Open Images V7** que no se usaron en ningún entrenamiento
(ni de v1 ni de v2), los conjuntos con los que se mide la presencia de vehículo (R1):

- **general**: el vehículo casi entero. **plate**: la trompa o la cola completa, con la patente al 8-15 % del
  ancho. **detail**: una zona del vehículo que ocupa buena parte de la foto. Se recortan de fotos de autos
  anotados con una ventana 16:9 cuyas proporciones salen de las fotos de referencia (`ref`, abajo).
- **heavy**: camiones, vans y colectivos válidos (`Truck`, `Van` o `Bus` anotado y sin `Car`), con el encuadre
  de general (y la foto entera aparte).
- **neg**: fotos sin ninguna clase de presencia anotada (`Car` y sus subclases, `Truck`, `Van`, `Bus`, `Taxi`,
  `Ambulance`), **sin exclusiones manuales**. Se parten en dos mitades con semilla fija (A: se eligen los
  umbrales; B: se reporta S1).

Cada recorte se lleva a 1600 px de lado mayor y a JPEG (lo que envía la app: lado mayor <= 1600 px y
<= 900 KB). `ref` mide el encuadre de las fotos de referencia (ancho de la patente y área del vehículo) y solo
imprime cifras: esas fotos no calibran umbrales ni se copian ni entran a ningún conjunto.

La salida (`--out-dir`) tiene los JPEG y `specs.json` (identificadores de Open Images y ventanas, sin patentes
ni nombres de archivo de terceros). Todo queda fuera del repo. Uso: ver `tools/eval/README.md`.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "detector_v2"))

CACHE = Path("fiftyone-cache") / "open-images-v7"
SOURCE_SPLIT = "test"
IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"

# Clases de presencia de Open Images (Car y sus subclases Limousine y Van; Truck, Bus, Taxi y Ambulance)
CAR_CLASSES = ("Car", "Limousine", "Taxi")
HEAVY_CLASSES = ("Truck", "Van", "Bus")
PRESENCE_CLASSES = ("Car", "Limousine", "Taxi", "Truck", "Van", "Bus", "Ambulance")
PLATE_CLASS = "Vehicle registration plate"
# No son parte de la definición de "sin presencia", pero delatan un vehículo: se marcan para otra lectura
FAMILY_CLASSES = ("Vehicle", "Land vehicle", "Wheel", "Tire", PLATE_CLASS, "Motorcycle", "Bicycle")

# Encuadre (fracciones de la foto de salida), medido en las fotos de referencia con `ref`
ASPECT = 16 / 9
OUT_LONG_SIDE = 1600
JPEG_QUALITY = 88
MAX_BYTES = 900 * 1024
GENERAL_CAR_WIDTH = (0.80, 0.98)  # ancho del vehículo (referencias: 0,82 a 0,98)
PLATE_WIDTH = (0.08, 0.15)  # ancho de la patente (pedido; referencias: 0,115 y 0,132)
PLATE_CENTER_X = (0.40, 0.70)  # dónde cae la patente en la foto
PLATE_CENTER_Y = (0.50, 0.70)
PLATE_TO_CAR_WIDTH = (0.10, 0.30)  # la patente respecto del vehículo (referencias: 0,18 y 0,20)
DETAIL_WINDOW_OF_CAR = (0.30, 0.55)  # el ancho de la zona respecto del vehículo
DETAIL_MIN_INSIDE = 0.85  # fracción de la foto que queda dentro de la caja del vehículo (referencia: 0,89)
MIN_WINDOW_PX = {"general": 480, "plate": 480, "detail": 320}
DRAWS = 24  # intentos de ventana por foto

TARGETS = {"plate": 300, "general": 400, "detail": 400}
HEAVY_PER_CLASS = 150
NEGATIVES = 2000
SEED = 0


@dataclass(frozen=True)
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Spec:
    """Una foto del conjunto: de qué imagen sale y con qué ventana (píxeles de la foto de origen)."""

    kind: str
    image_id: str
    window: tuple[int, int, int, int] | None = None
    detail: dict[str, Any] = field(default_factory=dict)


# --- geometría -----------------------------------------------------------------------------------


def window_at(cx: float, cy: float, width: float, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Ventana 16:9 de ese ancho centrada en (cx, cy) y desplazada para que quede dentro de la foto."""
    img_w, img_h = size
    height = width / ASPECT
    if width > img_w or height > img_h:
        return None
    x1 = min(max(cx - width / 2, 0), img_w - width)
    y1 = min(max(cy - height / 2, 0), img_h - height)
    return round(x1), round(y1), round(x1 + width), round(y1 + height)


def overlap_fraction(window: tuple[int, int, int, int], box: tuple[float, float, float, float]) -> float:
    """Qué fracción de la ventana cubre la caja."""
    wx1, wy1, wx2, wy2 = window
    ix = max(0.0, min(wx2, box[2]) - max(wx1, box[0]))
    iy = max(0.0, min(wy2, box[3]) - max(wy1, box[1]))
    return ix * iy / max(1.0, (wx2 - wx1) * (wy2 - wy1))


def general_window(
    car: tuple[float, float, float, float], size: tuple[int, int], rng: random.Random
) -> tuple[tuple[int, int, int, int], float] | None:
    """Ventana con el vehículo casi entero: su ancho ocupa `GENERAL_CAR_WIDTH` de la foto."""
    car_w, car_h = car[2] - car[0], car[3] - car[1]
    for _ in range(DRAWS):
        fraction = rng.uniform(*GENERAL_CAR_WIDTH)
        width = car_w / fraction
        if width < MIN_WINDOW_PX["general"] or car_h > 0.95 * width / ASPECT:
            continue
        jitter = rng.uniform(-0.03, 0.03) * width
        window = window_at((car[0] + car[2]) / 2 + jitter, (car[1] + car[3]) / 2, width, size)
        if window is not None:
            return window, fraction
    return None


def plate_window(
    plate: tuple[float, float, float, float],
    car: tuple[float, float, float, float],
    size: tuple[int, int],
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], float] | None:
    """Ventana con la trompa o la cola completa y la patente al `PLATE_WIDTH` del ancho."""
    plate_w = plate[2] - plate[0]
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    for _ in range(DRAWS):
        fraction = rng.uniform(*PLATE_WIDTH)
        width = plate_w / fraction
        height = width / ASPECT
        if width < MIN_WINDOW_PX["plate"] or width > size[0] or height > size[1]:
            continue
        x1 = cx - rng.uniform(*PLATE_CENTER_X) * width
        y1 = cy - rng.uniform(*PLATE_CENTER_Y) * height
        x1, y1 = min(max(x1, 0), size[0] - width), min(max(y1, 0), size[1] - height)
        # la trompa o la cola entera: el vehículo cabe a lo ancho y la ventana llega hasta su base
        if car[0] < x1 - 0.05 * width or car[2] > x1 + 1.05 * width or y1 + height < car[3] - 0.10 * height:
            continue
        return (round(x1), round(y1), round(x1 + width), round(y1 + height)), plate_w / (round(width) or 1)
    return None


def detail_window(
    car: tuple[float, float, float, float], size: tuple[int, int], rng: random.Random
) -> tuple[tuple[int, int, int, int], float] | None:
    """Ventana con una zona del vehículo: al menos `DETAIL_MIN_INSIDE` de la foto cae dentro de su caja."""
    car_w = car[2] - car[0]
    for _ in range(DRAWS):
        fraction = rng.uniform(*DETAIL_WINDOW_OF_CAR)
        width = car_w * fraction
        if width < MIN_WINDOW_PX["detail"]:
            continue
        cx = rng.uniform(car[0] + width / 2, car[2] - width / 2)
        cy = rng.uniform(car[1], car[3])
        window = window_at(cx, cy, width, size)
        if window is not None and overlap_fraction(window, car) >= DETAIL_MIN_INSIDE:
            return window, fraction
    return None


# --- anotaciones de Open Images --------------------------------------------------------------------


@dataclass
class Pool:
    """Lo que se necesita de las cajas de la partición de prueba."""

    boxes: dict[str, list[Box]]  # por foto, solo cajas de las clases de interés (normalizadas)
    presence: set[str]  # fotos con alguna clase de presencia anotada (cualquier caja)
    cars: set[str]  # fotos con `Car` o una subclase anotada (cualquier caja)
    family: set[str]  # fotos con algo de la familia vehículo
    annotated: set[str]  # fotos con alguna caja


def read_pool(detections_csv: Path, class_ids: Mapping[str, str]) -> Pool:
    import pandas as pd  # solo hace falta para armar los conjuntos

    names = {
        class_ids[n]: n for n in (*CAR_CLASSES, *HEAVY_CLASSES, PLATE_CLASS, "Ambulance") if n in class_ids
    }
    presence_ids = {class_ids[n] for n in PRESENCE_CLASSES}
    car_ids = {class_ids[n] for n in CAR_CLASSES}
    family_ids = {class_ids[n] for n in FAMILY_CLASSES if n in class_ids}
    boxes: dict[str, list[Box]] = defaultdict(list)
    presence: set[str] = set()
    cars: set[str] = set()
    family: set[str] = set()
    annotated: set[str] = set()
    columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction", "IsInside"]
    for chunk in pd.read_csv(detections_csv, usecols=columns, chunksize=1_000_000):
        annotated.update(chunk["ImageID"].unique())
        presence.update(chunk.loc[chunk["LabelName"].isin(presence_ids), "ImageID"].unique())
        cars.update(chunk.loc[chunk["LabelName"].isin(car_ids), "ImageID"].unique())
        family.update(chunk.loc[chunk["LabelName"].isin(family_ids), "ImageID"].unique())
        usable = chunk[
            chunk["LabelName"].isin(names)
            & (chunk["IsGroupOf"] == 0)
            & (chunk["IsDepiction"] == 0)
            & (chunk["IsInside"] == 0)
        ]
        for row in usable.itertuples(index=False):
            boxes[row.ImageID].append(Box(names[row.LabelName], row.XMin, row.YMin, row.XMax, row.YMax))
    return Pool(dict(boxes), presence, cars, family, annotated)


def load_class_ids(classes_csv: Path) -> dict[str, str]:
    import csv

    with classes_csv.open(encoding="utf-8", newline="") as handle:
        return {row[1]: row[0] for row in csv.reader(handle) if len(row) >= 2}


def training_ids(data_dir: Path) -> dict[str, set[str]]:
    """Fotos con las que se entrenó v1 (`train` y `valid`) y v2: no pueden entrar a ningún conjunto."""
    groups = {
        "v1_entrenamiento": data_dir / "openimages-detector" / "train",
        "v1_validacion": data_dir / "openimages-detector" / "valid",
        "v2_entrenamiento": data_dir / "openimages-detector-v2" / "train",
        "v2_validacion": data_dir / "openimages-detector-v2" / "valid",
    }
    return {name: {p.stem for p in folder.glob("*.jpg")} for name, folder in groups.items()}


# --- armado ---------------------------------------------------------------------------------------


def pixel_box(box: Box, size: tuple[int, int]) -> tuple[float, float, float, float]:
    return box.x1 * size[0], box.y1 * size[1], box.x2 * size[0], box.y2 * size[1]


def contains_center(
    outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]
) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def crop_to_jpeg(image: np.ndarray, window: tuple[int, int, int, int] | None) -> bytes:
    """Recorta, lleva el lado mayor a 1600 px y codifica a JPEG de <= 900 KB (como la app)."""
    if window is not None:
        x1, y1, x2, y2 = window
        image = image[y1:y2, x1:x2]
    scale = OUT_LONG_SIDE / max(image.shape[:2])
    if window is not None or scale < 1.0:
        size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        image = cv2.resize(image, size, interpolation=interpolation)
    quality = JPEG_QUALITY
    while True:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and (len(encoded) <= MAX_BYTES or quality <= 40):
            return bytes(encoded)
        quality -= 8


def image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else (image.shape[1], image.shape[0])


def plan(pool: Pool, excluded: set[str], rng: random.Random) -> dict[str, list[str]]:
    """Qué fotos bajar para cada conjunto (con margen: algunas no admiten una ventana)."""
    usable = [i for i in sorted(pool.boxes) if i not in excluded]
    with_plate = [i for i in usable if any(b.label == PLATE_CLASS for b in pool.boxes[i])]
    with_car = [i for i in usable if any(b.label in CAR_CLASSES for b in pool.boxes[i])]
    heavy: dict[str, list[str]] = {}
    for name in HEAVY_CLASSES:
        heavy[name] = [
            i for i in usable if any(b.label == name for b in pool.boxes[i]) and i not in pool.cars
        ]
    negatives = [i for i in sorted(pool.annotated) if i not in pool.presence and i not in excluded]
    for ids in (with_plate, with_car, negatives, *heavy.values()):
        rng.shuffle(ids)
    taken: set[str] = set()

    def take(ids: Sequence[str], n: int) -> list[str]:
        chosen = [i for i in ids if i not in taken][:n]
        taken.update(chosen)
        return chosen

    return {
        "plate": take(with_plate, len(with_plate)),  # las rarezas: todas las que haya
        "general": take(with_car, round(TARGETS["general"] * 3)),
        "detail": take(with_car, round(TARGETS["detail"] * 2)),
        **{f"heavy_{name}": take(heavy[name], len(heavy[name])) for name in HEAVY_CLASSES},
        "neg": negatives[:NEGATIVES],
    }


def best_car(boxes: Iterable[Box], size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    cars = [pixel_box(b, size) for b in boxes if b.label in CAR_CLASSES]
    return max(cars, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)


def make_specs(
    pool: Pool, ids: Mapping[str, Sequence[str]], sizes: Mapping[str, tuple[int, int]], rng: random.Random
) -> dict[str, list[Spec]]:
    """Las ventanas de cada foto candidata; se queda con las que entran hasta completar cada cupo."""
    out: dict[str, list[Spec]] = {name: [] for name in ids}
    for kind, candidates in ids.items():
        for image_id in candidates:
            if image_id not in sizes:
                continue
            size, boxes = sizes[image_id], pool.boxes.get(image_id, [])
            spec = None
            if kind == "plate":
                cars = [pixel_box(b, size) for b in boxes if b.label in CAR_CLASSES]
                for plate in (pixel_box(b, size) for b in boxes if b.label == PLATE_CLASS):
                    car = next(
                        (
                            c
                            for c in cars
                            if contains_center(c, plate)
                            and PLATE_TO_CAR_WIDTH[0]
                            <= (plate[2] - plate[0]) / (c[2] - c[0])
                            <= PLATE_TO_CAR_WIDTH[1]
                        ),
                        None,
                    )
                    found = None if car is None else plate_window(plate, car, size, rng)
                    if found:
                        spec = Spec("plate", image_id, found[0], {"plate_width": round(found[1], 3)})
                        break
            elif kind in ("general", "detail"):
                car = best_car(boxes, size)
                found = None
                if car is not None:
                    found = (
                        general_window(car, size, rng) if kind == "general" else detail_window(car, size, rng)
                    )
                if found:
                    spec = Spec(kind, image_id, found[0], {"fraction": round(found[1], 3)})
            elif kind.startswith("heavy_"):
                label = kind.removeprefix("heavy_")
                heavies = [pixel_box(b, size) for b in boxes if b.label == label]
                biggest = max(heavies, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)
                found = None if biggest is None else general_window(biggest, size, rng)
                if found:
                    spec = Spec(kind, image_id, found[0], {"fraction": round(found[1], 3), "whole": True})
            else:
                spec = Spec("neg", image_id, None, {"family": image_id in pool.family})
            if spec is not None:
                out[kind].append(spec)
    return out


def trim(specs: dict[str, list[Spec]]) -> dict[str, list[Spec]]:
    quotas = {**TARGETS, **{f"heavy_{n}": HEAVY_PER_CLASS for n in HEAVY_CLASSES}, "neg": NEGATIVES}
    return {name: items[: quotas[name]] for name, items in specs.items()}


def halves(count: int, seed: int) -> list[str]:
    """`A` o `B` por posición: la mitad de cada conjunto con semilla fija."""
    order = list(range(count))
    random.Random(seed).shuffle(order)  # noqa: S311 - partición de datos, no criptografía
    labels = ["A"] * count
    for position in order[count // 2 :]:
        labels[position] = "B"
    return labels


def build(args: argparse.Namespace) -> int:
    import v2_download  # tools/training/detector_v2

    cache = Path(args.data_dir) / CACHE / SOURCE_SPLIT
    pool = read_pool(cache / "labels" / "detections.csv", load_class_ids(cache / "metadata" / "classes.csv"))
    train = training_ids(Path(args.data_dir))
    excluded = set().union(*train.values())
    rng = random.Random(SEED)  # noqa: S311 - muestreo de fotos, no criptografía
    planned = plan(pool, excluded, rng)
    source = Path(args.out_dir) / "source"
    wanted = sorted({i for ids in planned.values() for i in ids})
    print(f"Fotos a bajar: {len(wanted)} (partición {SOURCE_SPLIT} de Open Images)", flush=True)
    ok = v2_download.download_all(wanted, SOURCE_SPLIT, source)
    sizes = {i: s for i in ok if (s := image_size(source / f"{i}.jpg")) is not None}
    specs = trim(make_specs(pool, {k: [i for i in v if i in sizes] for k, v in planned.items()}, sizes, rng))
    out = Path(args.out_dir)
    for name, items in specs.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for index, spec in enumerate(items):
            image = cv2.imdecode(
                np.fromfile(str(source / f"{spec.image_id}.jpg"), np.uint8), cv2.IMREAD_COLOR
            )
            (folder / f"{index:04d}.jpg").write_bytes(crop_to_jpeg(image, spec.window))
            if name.startswith("heavy_"):  # además, la foto entera
                whole = out / f"{name}_whole"
                whole.mkdir(exist_ok=True)
                (whole / f"{index:04d}.jpg").write_bytes(crop_to_jpeg(image, None))
    report = {
        "origen": f"Open Images V7, partición {SOURCE_SPLIT}",
        "semilla": SEED,
        "fotos": {name: len(items) for name, items in specs.items()},
        "mitades": {
            name: dict(
                zip(
                    ("A", "B"),
                    (halves(len(items), SEED).count("A"), len(items) - halves(len(items), SEED).count("A")),
                    strict=True,
                )
            )
            for name, items in specs.items()
        },
        "negativas_con_familia_vehiculo": sum(s.detail["family"] for s in specs["neg"]),
        "specs": {name: [asdict(s) for s in items] for name, items in specs.items()},
    }
    (out / "specs.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "specs"})
    return 0


# --- verificación de la exclusión --------------------------------------------------------------------


def verify(args: argparse.Namespace) -> int:
    import v2_download

    out = Path(args.out_dir)
    specs = json.loads((out / "specs.json").read_text(encoding="utf-8"))["specs"]
    selected = {
        s["image_id"]: out / "source" / f"{s['image_id']}.jpg" for items in specs.values() for s in items
    }
    train = training_ids(Path(args.data_dir))
    folders = {
        "v1_entrenamiento": Path(args.data_dir) / "openimages-detector" / "train",
        "v1_validacion": Path(args.data_dir) / "openimages-detector" / "valid",
        "v2_entrenamiento": Path(args.data_dir) / "openimages-detector-v2" / "train",
        "v2_validacion": Path(args.data_dir) / "openimages-detector-v2" / "valid",
    }
    excluded_files = {i: folders[g] / f"{i}.jpg" for g, ids in train.items() for i in ids}
    csvs = [
        Path(args.data_dir) / CACHE / s / "metadata" / "image_ids.csv"
        for s in ("test", "validation", "train")
    ]
    wanted = set(selected) | set(excluded_files)
    md5s = v2_download.original_md5s(csvs, wanted)
    result = v2_download.verify_exclusion(selected, train, excluded_files, md5s)
    (out / "exclusion_check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "excluidas_sin_archivo"}, indent=1))
    return 0 if result["ok"] else 2


# --- encuadre de las fotos de referencia -------------------------------------------------------------


def ref(args: argparse.Namespace) -> int:
    """Mide, en las fotos de referencia, el ancho de la patente y el área del vehículo. Solo cifras."""
    from vvi_ia.adapters.imaging import CvImageDecoder
    from vvi_ia.adapters.models.registry import build_models
    from vvi_ia.domain.model import BoxLabel

    models = build_models(
        {"detector": args.detector, "segmenter": "rfdetr_seg_coco"},
        args.manifest,
        args.models_dir,
        params={"detector_min_conf": 0.05, "redaction_min_conf": 0.05, "segmenter_min_conf": 0.05},
    )
    decoder, seen = CvImageDecoder(), set()
    rows = []
    for path in sorted(Path(args.ref_dir).glob("*.jpg")):
        data = path.read_bytes()
        digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
        if digest in seen:  # la misma foto repetida
            continue
        seen.add(digest)
        image = decoder.decode(data)
        height, width = image.shape[:2]
        row: dict[str, Any] = {"foto": len(rows) + 1, "ancho": width, "alto": height}
        detections = [b for b in models.vehicle.detect(image) or () if b.conf >= 0.30]
        cars = sorted((b for b in detections if b.label == BoxLabel.CAR), key=lambda b: -b.area)
        plates = sorted((b for b in detections if b.label == BoxLabel.PLATE), key=lambda b: -b.conf)
        if cars:
            row |= {
                "vehiculo_ancho": round((cars[0].x2 - cars[0].x1) / width, 3),
                "vehiculo_area": round(cars[0].area / (width * height), 3),
            }
        if plates:
            row["patente_ancho"] = round((plates[0].x2 - plates[0].x1) / width, 3)
        rows.append(row)
    print(json.dumps(rows, indent=1))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("ref", "build", "verify"))
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--ref-dir", type=Path, help="fotos de referencia (solo para `ref`)")
    parser.add_argument("--detector", default="rfdetr_openimages_v2")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        parser.error("indicá --out-dir o definí EVAL_OUT")
    if args.command != "ref" and args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR")
    if args.command == "ref" and (args.ref_dir is None or args.models_dir is None):
        parser.error("`ref` pide --ref-dir y --models-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.out_dir, args.data_dir, args.ref_dir):
        if path is not None and Path(path).resolve().is_relative_to(REPO_ROOT):
            print("Las fotos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
            return 1
    return {"ref": ref, "build": build, "verify": verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
