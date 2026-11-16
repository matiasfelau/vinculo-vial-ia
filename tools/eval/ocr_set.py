"""Arma el conjunto de evaluación del OCR con "Patentes Argentinas - Master" (Roboflow, Public Domain).

Parte del export COCO (cajas de patente) y deja, fuera del repo, 300 recortes para etiquetar a mano:

1. Descarta cajas chicas (alto o ancho en píxeles por debajo del mínimo): no hay nada que leer.
2. Descarta las fotos y las patentes que ya están en el dataset "patentes" (`--reference`), con el que se
   calibró el OCR: una patente que el OCR ya vio no mide nada.
3. Saca los duplicados y casi duplicados: la misma patente en otra foto, a otra escala o con otra compresión
   (se conserva el de mayor resolución).
4. Clasifica cada foto en día o noche por el brillo medio de la escena sin la patente.
5. Saca las probables motos (relación ancho/alto de la caja menor que 1,2) y las cajas inválidas (con varias
   patentes adentro): el servicio lee una sola línea y esas cajas no son una patente. Entre 1,2 y 1,5 las deja
   y las marca con `revisar_moto = sí` para que quien etiqueta lo confirme.
6. Elige `--n` con semilla fija, al menos `--night-min` de noche y como mucho 2 por foto, y los numera en
   orden aleatorio.

Escribe en `--out`: `recortes/NNN.png` (caja con 15 % de margen), `etiquetas.csv` (texto, formato y estado
vacíos), `hojas/hoja-NN.jpg` (hojas de contacto para etiquetar mirando), `informe.json` (solo cifras) y
`descartes.csv`.
No corre ningún OCR. La consola muestra solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_labels  # noqa: E402 - las etiquetas en csv o xlsx

SPLITS = ("train", "valid", "test")
ANNOTATIONS = "_annotations.coco.json"
DEFAULT_SOURCE = "ocr-ar"
DEFAULT_REFERENCE = "ocr-roboflow"
DEFAULT_OUT = "ocr-ar-set"

CROP_MARGIN = 0.15  # de ancho y de alto, a cada lado, recortado a la foto
MIN_BOX_HEIGHT = 24  # píxeles de la caja anotada, antes del margen
MIN_BOX_WIDTH = 40
DEFAULT_COUNT = 300
DEFAULT_NIGHT_MIN = 50
DEFAULT_SEED = 20261116
# Noche si la escena sin la patente tiene un brillo medio (gris 0-255) menor que esto. Mirando las fotos más
# oscuras, las de noche quedan bajo ~90 y de ahí para arriba se mezclan con sombras: por eso las confirma una
# persona.
NIGHT_BRIGHTNESS = 90.0
# Si la caja ocupa tanto de la foto, no hay escena para juzgar la luz (primer plano de la patente).
MAX_BOX_FRACTION = 0.4
SURROUND_EXPANSION = 0.2
SCENE_SIDE = 256

# Dos recortes son la misma patente si se parecen en bruto (NCC de 64x32 en gris) y, además, en los trazos:
# NCC con corrimiento de recortes en 128x64 sin las bajas frecuencias. Calibrado mirando pares: con 0,75 y 0,6
# todos los que pasan son la misma patente (las distintas, de igual diseño, dan menos de 0,6 en trazos); queda
# sin detectar la misma patente en una foto muy distinta. Misma foto (otra resolución): NCC de 32x32 de 0,95.
RAW_SIZE = (64, 32)
STROKE_SIZE = (128, 64)
STROKE_BLUR = 5.0
STROKE_SHIFT = (8, 16)  # alto y ancho de recorte para tolerar corrimientos
MIN_RAW_NCC = 0.6
MIN_STROKE_NCC = 0.75
MIN_SAME_PHOTO_NCC = 0.95
PHOTO_SIZE = (32, 32)

REVIEW_COLUMN = "revisar_moto"
REVIEW_VALUE = "sí"
LABEL_COLUMNS = ("id", "archivo", "recorte", "texto", "formato", "luz", "estado", REVIEW_COLUMN)
DISCARD_COLUMNS = ("archivo", "caja", "motivo", "detalle")
BOX_COLUMNS = ("id", "archivo", "caja", "relacion_aspecto")
REPLACEMENT_COLUMNS = ("id", "archivo", "caja", "motivo", "reemplazada_por")
LIGHT_DAY = "dia"
LIGHT_NIGHT = "noche"
MOTORCYCLE_ASPECT = 1.2  # ancho/alto de la caja: por debajo, probable moto (patente de dos líneas)
REVIEW_ASPECT = 1.5  # de MOTORCYCLE_ASPECT a esto queda en el set, con `revisar_moto`
MAX_PER_PHOTO = 2  # patentes del set que pueden salir de la misma foto
REASON_MOTORCYCLE = "probable_moto"
REASON_INVALID = "caja_invalida"
REASON_PHOTO_CAP = "exceso_por_foto"
# Una caja contiene a otra de la misma foto si tiene adentro al menos esto de su área y la otra es bastante
# más chica: es una hoja de patentes, un auto entero o una caja doble, no una patente.
NESTED_INSIDE = 0.8
NESTED_SIZE = 0.6
# Cajas con varias patentes sin otra adentro, vistas a ojo: (split/foto sin el hash de Roboflow, ancho, alto).
KNOWN_INVALID = (("train/17397969544621434_jpg", 142, 212),)  # pila de cinco patentes

TILE_W, TILE_H = 480, 200
CROP_AREA = (340, 172)
PHOTO_AREA = 130
SHEET_COLUMNS, SHEET_ROWS = 3, 6


@dataclass(frozen=True)
class Annotation:
    """Una caja de patente de una foto del export."""

    split: str
    file: str
    box: tuple[float, float, float, float]  # x, y, ancho, alto, en píxeles de la foto

    @property
    def name(self) -> str:
        return f"{self.split}/{self.file}"


@dataclass
class Described:
    """Una caja con lo que hace falta para compararla y clasificarla (sin guardar la foto)."""

    annotation: Annotation
    area: int
    raw: np.ndarray
    strokes: np.ndarray
    photo: np.ndarray
    surround: float | None  # brillo medio de la escena sin la patente; `None` si no hay escena que juzgar
    light: str = LIGHT_DAY
    reasons: list[str] = field(default_factory=list)

    @property
    def light_is_certain(self) -> bool:
        return self.surround is not None


def read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None


def read_color(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def load_annotations(root: Path) -> list[Annotation]:
    """Todas las cajas del export (train, valid y test), en un orden estable."""
    found: list[Annotation] = []
    for split in SPLITS:
        path = root / split / ANNOTATIONS
        if not path.is_file():
            continue
        coco = json.loads(path.read_text(encoding="utf-8"))
        images = {info["id"]: info["file_name"] for info in coco["images"]}
        for item in coco["annotations"]:
            x, y, w, h = (float(v) for v in item["bbox"])
            found.append(Annotation(split, images[item["image_id"]], (x, y, w, h)))
    return sorted(found, key=lambda a: (a.split, a.file, a.box))


def pixel_box(
    box: tuple[float, float, float, float], width: int, height: int, margin: float = 0.0
) -> tuple[int, int, int, int]:
    """Caja (x1, y1, x2, y2) entera, con `margin` de su ancho y alto a cada lado, recortada a la foto."""
    x, y, w, h = box
    x1 = max(0, int(round(x - w * margin)))
    y1 = max(0, int(round(y - h * margin)))
    x2 = min(width, int(round(x + w + w * margin)))
    y2 = min(height, int(round(y + h + h * margin)))
    return x1, y1, x2, y2


def _normalized(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    vector = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _strokes(crop: np.ndarray) -> np.ndarray:
    small = cv2.resize(crop, STROKE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return small - cv2.GaussianBlur(small, (0, 0), STROKE_BLUR)


def surround_brightness(gray: np.ndarray, box: tuple[float, float, float, float]) -> float | None:
    """Brillo medio de la foto (reducida) sin la patente ni su entorno; `None` si ocupa casi toda."""
    scale = SCENE_SIDE / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x, y, w, h = (v * scale for v in box)
    x1, y1, x2, y2 = pixel_box((x, y, w, h), small.shape[1], small.shape[0], SURROUND_EXPANSION)
    mask = np.ones(small.shape, dtype=bool)
    mask[y1 : y2 + 1, x1 : x2 + 1] = False
    if 1.0 - float(mask.mean()) >= MAX_BOX_FRACTION or int(mask.sum()) < 200:
        return None
    return float(small[mask].mean())


def estimate_light(surround: float | None, threshold: float = NIGHT_BRIGHTNESS) -> str:
    """Noche si hay escena y es oscura; sin escena (primer plano) no se puede saber y queda como día."""
    return LIGHT_NIGHT if surround is not None and surround < threshold else LIGHT_DAY


def describe(root: Path, annotations: Sequence[Annotation]) -> list[Described]:
    """Lee cada foto una sola vez y calcula las firmas de sus cajas. Salta las que no se pueden leer."""
    described: list[Described] = []
    by_photo: dict[tuple[str, str], list[Annotation]] = {}
    for annotation in annotations:
        by_photo.setdefault((annotation.split, annotation.file), []).append(annotation)
    for (split, file), items in by_photo.items():
        gray = read_gray(root / split / file)
        if gray is None:
            continue
        photo = _normalized(gray, PHOTO_SIZE)
        height, width = gray.shape
        for annotation in items:
            x1, y1, x2, y2 = pixel_box(annotation.box, width, height)
            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            described.append(
                Described(
                    annotation=annotation,
                    area=int(crop.shape[0] * crop.shape[1]),
                    raw=_normalized(crop, RAW_SIZE),
                    strokes=_strokes(crop),
                    photo=photo,
                    surround=surround_brightness(gray, annotation.box),
                )
            )
    return sorted(described, key=lambda d: (d.annotation.split, d.annotation.file, d.annotation.box))


def stroke_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Mejor NCC de una firma de trazos contra la otra con corrimiento, en los dos sentidos."""
    dy, dx = STROKE_SHIFT
    first = cv2.matchTemplate(a, b[dy:-dy, dx:-dx], cv2.TM_CCOEFF_NORMED).max()
    second = cv2.matchTemplate(b, a[dy:-dy, dx:-dx], cv2.TM_CCOEFF_NORMED).max()
    return float(max(first, second))


def same_plate(a: Described, b: Described) -> bool:
    """`True` si los dos recortes son la misma patente (copia o casi copia)."""
    if float(a.raw @ b.raw) < MIN_RAW_NCC:
        return False
    return stroke_ncc(a.strokes, b.strokes) >= MIN_STROKE_NCC


def same_photo(a: Described, b: Described) -> bool:
    return float(a.photo @ b.photo) >= MIN_SAME_PHOTO_NCC


def in_reference(item: Described, reference: Sequence[Described]) -> str | None:
    """`"foto"` si la foto ya está en la referencia, `"patente"` si lo está la patente, o `None`."""
    if any(same_photo(item, other) for other in reference):
        return "foto"
    if any(same_plate(item, other) for other in reference):
        return "patente"
    return None


def duplicate_groups(items: Sequence[Described]) -> list[list[int]]:
    """Grupos de índices de la misma patente (componentes conexas de `same_plate`), solo los de 2 o más."""
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if items:
        raw = np.stack([item.raw for item in items])
        candidates = np.argwhere(np.triu(raw @ raw.T >= MIN_RAW_NCC, k=1))
        for i, j in candidates:
            if find(int(i)) != find(int(j)) and same_plate(items[int(i)], items[int(j)]):
                parent[find(int(j))] = find(int(i))
    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)
    return [members for members in groups.values() if len(members) > 1]


def dedupe(items: Sequence[Described]) -> tuple[list[Described], list[tuple[Described, Described]]]:
    """Deja el de mayor resolución de cada grupo: (los que quedan, pares (descartado, conservado))."""
    dropped: list[tuple[Described, Described]] = []
    removed: set[int] = set()
    for members in duplicate_groups(items):
        keep = max(members, key=lambda i: (items[i].area, -i))  # a igual resolución, el primero
        for index in members:
            if index != keep:
                removed.add(index)
                dropped.append((items[index], items[keep]))
    return [item for index, item in enumerate(items) if index not in removed], dropped


def photo_of(annotation: Annotation) -> tuple[str, str]:
    return annotation.split, annotation.file


def _capped_sample(
    rng: random.Random, items: Sequence[Described], count: int, per_photo: int, already: Sequence[Described]
) -> list[Described]:
    """`count` al azar con como mucho `per_photo` por foto, contando las que ya estaban (`already`)."""
    taken = Counter(photo_of(item.annotation) for item in already)
    picked: list[Described] = []
    for item in rng.sample(list(items), len(items)):
        if len(picked) == count:
            break
        key = photo_of(item.annotation)
        if taken[key] < per_photo:
            picked.append(item)
            taken[key] += 1
    if len(picked) < count:
        raise ValueError(f"no alcanzan las candidatas con como mucho {per_photo} por foto")
    return picked


def select(
    pool: Sequence[Described], count: int, night_min: int, seed: int, per_photo: int | None = None
) -> list[Described]:
    """`count` al azar con semilla fija y al menos `night_min` de noche, en orden aleatorio.

    Con `per_photo`, como mucho esa cantidad de cada foto. Sin él, la selección es la de siempre.
    """
    nights = [item for item in pool if item.light == LIGHT_NIGHT]
    if len(nights) < night_min:
        raise ValueError(f"hay {len(nights)} de noche y se piden al menos {night_min}")
    if len(pool) < count:
        raise ValueError(f"hay {len(pool)} candidatas y se piden {count}")
    rng = random.Random(seed)  # noqa: S311 - muestreo de fotos, no criptografía
    if per_photo is None:
        chosen = rng.sample(nights, night_min)
        taken = {id(item) for item in chosen}
        chosen += rng.sample([item for item in pool if id(item) not in taken], count - night_min)
    else:
        chosen = _capped_sample(rng, nights, night_min, per_photo, [])
        taken = {id(item) for item in chosen}
        rest = [item for item in pool if id(item) not in taken]
        chosen += _capped_sample(rng, rest, count - night_min, per_photo, chosen)
    rng.shuffle(chosen)
    return chosen


def crop_with_margin(color: np.ndarray, annotation: Annotation) -> np.ndarray:
    height, width = color.shape[:2]
    x1, y1, x2, y2 = pixel_box(annotation.box, width, height, CROP_MARGIN)
    return color[y1:y2, x1:x2]


def _fit(image: np.ndarray, area: tuple[int, int]) -> np.ndarray:
    scale = min(area[0] / image.shape[1], area[1] / image.shape[0])
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def contact_tile(
    label: str, crop: np.ndarray, photo: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    """Un casillero: el recorte grande, la foto entera chica con la caja marcada y el número."""
    tile = np.full((TILE_H, TILE_W, 3), 32, np.uint8)
    fitted = _fit(crop, CROP_AREA)
    tile[24 : 24 + fitted.shape[0], 4 : 4 + fitted.shape[1]] = fitted
    scene = photo.copy()
    cv2.rectangle(scene, box[:2], box[2:], (0, 0, 255), max(2, max(scene.shape) // 150))
    small = _fit(scene, (PHOTO_AREA, PHOTO_AREA))
    tile[24 : 24 + small.shape[0], TILE_W - PHOTO_AREA - 4 : TILE_W - PHOTO_AREA - 4 + small.shape[1]] = small
    cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (90, 90, 90), 1)
    return tile


def sheet_from_tiles(tiles: Sequence[np.ndarray]) -> np.ndarray:
    blank = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    padded = list(tiles) + [blank] * (SHEET_COLUMNS * SHEET_ROWS - len(tiles))
    rows = [np.hstack(padded[r : r + SHEET_COLUMNS]) for r in range(0, len(padded), SHEET_COLUMNS)]
    return np.vstack(rows)


def write_png(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"no se pudo codificar {path.name}")
    path.write_bytes(data.tobytes())


def write_jpeg(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError(f"no se pudo codificar {path.name}")
    path.write_bytes(data.tobytes())


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(
    root: Path,
    chosen: Sequence[tuple[str, Described]],
    out: Path,
    previous: dict[str, dict[str, str]] | None = None,
) -> None:
    """Recortes, etiquetas.csv, cajas.csv y hojas de contacto por id; `previous` conserva filas existentes."""
    (out / "recortes").mkdir(parents=True, exist_ok=True)
    (out / "hojas").mkdir(parents=True, exist_ok=True)
    labels: list[dict[str, str]] = []
    boxes: list[dict[str, str]] = []
    tiles: list[np.ndarray] = []
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    for position, (ident, item) in enumerate(chosen, start=1):
        annotation = item.annotation
        color = read_color(root / annotation.split / annotation.file)
        if color is None:
            raise OSError(f"no se pudo leer {annotation.name}")
        crop = crop_with_margin(color, annotation)
        write_png(out / "recortes" / f"{ident}.png", crop)
        label = dict(
            (previous or {}).get(ident)
            or {
                "id": ident,
                "archivo": annotation.name,
                "recorte": f"recortes/{ident}.png",
                "texto": "",
                "formato": "",
                "luz": item.light,
                "estado": "",
            }
        )
        label[REVIEW_COLUMN] = REVIEW_VALUE if needs_moto_review(annotation) else ""
        labels.append(label)
        boxes.append(
            {
                "id": ident,
                "archivo": annotation.name,
                "caja": box_text(annotation.box),
                "relacion_aspecto": f"{aspect(annotation):.2f}",
            }
        )
        mark = "N" if item.light == LIGHT_NIGHT else "D"
        mark += "" if item.light_is_certain else "?"
        mark += "  moto?" if needs_moto_review(annotation) else ""
        height, width = color.shape[:2]
        tiles.append(contact_tile(f"{ident}  {mark}", crop, color, pixel_box(annotation.box, width, height)))
        if len(tiles) == per_sheet or position == len(chosen):
            write_jpeg(
                out / "hojas" / f"hoja-{(position - 1) // per_sheet + 1:02d}.jpg", sheet_from_tiles(tiles)
            )
            tiles = []
    write_csv(out / "etiquetas.csv", LABEL_COLUMNS, labels)
    write_csv(out / "cajas.csv", BOX_COLUMNS, boxes)


def box_text(box: tuple[float, float, float, float]) -> str:
    return " ".join(f"{v:.0f}" for v in box)


def aspect(annotation: Annotation) -> float:
    """Ancho sobre alto de la caja anotada (sin margen)."""
    _, _, w, h = annotation.box
    return w / h if h > 0 else 0.0


def nested_boxes(annotations: Sequence[Annotation]) -> set[Annotation]:
    """Cajas que tienen adentro otra caja de patente de la misma foto (no son una sola patente)."""
    by_photo: dict[tuple[str, str], list[Annotation]] = {}
    for annotation in annotations:
        by_photo.setdefault((annotation.split, annotation.file), []).append(annotation)

    def inside(inner: Annotation, outer: Annotation) -> float:
        ix, iy, iw, ih = inner.box
        ox, oy, ow, oh = outer.box
        width = max(0.0, min(ix + iw, ox + ow) - max(ix, ox))
        height = max(0.0, min(iy + ih, oy + oh) - max(iy, oy))
        return width * height / (iw * ih) if iw * ih > 0 else 0.0

    found: set[Annotation] = set()
    for group in by_photo.values():
        for outer in group:
            area = outer.box[2] * outer.box[3]
            if any(
                other is not outer
                and other.box[2] * other.box[3] < NESTED_SIZE * area
                and inside(other, outer) >= NESTED_INSIDE
                for other in group
            ):
                found.add(outer)
    return found


def invalid_boxes(annotations: Sequence[Annotation]) -> set[Annotation]:
    """Cajas que no son una sola patente: las que contienen otras y las vistas a ojo."""
    known = {
        annotation
        for annotation in annotations
        for prefix, width, height in KNOWN_INVALID
        if annotation.name.split(".rf.")[0] == prefix
        and round(annotation.box[2]) == width
        and round(annotation.box[3]) == height
    }
    return nested_boxes(annotations) | known


def classify(annotation: Annotation, invalid: set[Annotation]) -> str | None:
    """Por qué la caja no sirve para el set (inválida o probable moto), o `None` si es de auto."""
    if annotation in invalid:
        return REASON_INVALID
    return REASON_MOTORCYCLE if aspect(annotation) < MOTORCYCLE_ASPECT else None


def needs_moto_review(annotation: Annotation) -> bool:
    """Relación entre 1,2 y 1,5: se queda en el set y quien etiqueta confirma si es una moto."""
    return MOTORCYCLE_ASPECT <= aspect(annotation) < REVIEW_ASPECT


@dataclass
class Pool:
    """Las cajas candidatas tras los descartes de tamaño, referencia y duplicados."""

    annotations: list[Annotation]
    items: list[Described]
    discards: list[dict[str, str]]
    dropped: int


def build_pool(source: Path, reference: Path, night_brightness: float = NIGHT_BRIGHTNESS) -> Pool:
    annotations = load_annotations(source)
    discards: list[dict[str, str]] = []

    def discard(annotation: Annotation, reason: str, detail: str = "") -> None:
        discards.append(
            {
                "archivo": annotation.name,
                "caja": box_text(annotation.box),
                "motivo": reason,
                "detalle": detail,
            }
        )

    big_enough = []
    for annotation in annotations:
        _, _, w, h = annotation.box
        if h < MIN_BOX_HEIGHT or w < MIN_BOX_WIDTH:
            discard(annotation, "caja_chica", f"{w:.0f}x{h:.0f}")
        else:
            big_enough.append(annotation)
    sized = describe(source, big_enough)

    references = describe(reference, load_annotations(reference))
    clean = []
    for item in sized:
        found = in_reference(item, references)
        if found is None:
            clean.append(item)
        else:
            discard(item.annotation, f"{found}_en_dataset_de_calibracion")

    pool, dropped = dedupe(clean)
    for gone, kept in dropped:
        discard(
            gone.annotation,
            "duplicado",
            f"se conserva {kept.annotation.name} {box_text(kept.annotation.box)}",
        )
    for item in pool:
        item.light = estimate_light(item.surround, night_brightness)
    return Pool(annotations, pool, discards, len(dropped))


def chosen_stats(chosen: Sequence[Described]) -> dict[str, Any]:
    return {
        "elegidas": len(chosen),
        "elegidas_de_noche": sum(item.light == LIGHT_NIGHT for item in chosen),
        "elegidas_sin_escena_para_juzgar_la_luz": sum(not item.light_is_certain for item in chosen),
        "elegidas_probable_moto": sum(aspect(item.annotation) < MOTORCYCLE_ASPECT for item in chosen),
        "elegidas_revisar_moto": sum(needs_moto_review(item.annotation) for item in chosen),
        "elegidas_max_por_foto": max(
            Counter(photo_of(item.annotation) for item in chosen).values(), default=0
        ),
        "elegidas_por_split": {
            split: sum(item.annotation.split == split for item in chosen) for split in SPLITS
        },
    }


def parameters(count: int, night_min: int, seed: int, night_brightness: float) -> dict[str, Any]:
    return {
        "n": count,
        "noche_minima": night_min,
        "semilla": seed,
        "umbral_noche": night_brightness,
        "margen_recorte": CROP_MARGIN,
        "caja_minima": f"{MIN_BOX_WIDTH}x{MIN_BOX_HEIGHT}",
        "ncc_crudo_minimo": MIN_RAW_NCC,
        "ncc_trazos_minimo": MIN_STROKE_NCC,
        "relacion_aspecto_moto": MOTORCYCLE_ASPECT,
        "relacion_aspecto_revisar_moto": REVIEW_ASPECT,
        "maximo_por_foto": MAX_PER_PHOTO,
    }


def run(
    source: Path,
    reference: Path,
    out: Path,
    count: int = DEFAULT_COUNT,
    night_min: int = DEFAULT_NIGHT_MIN,
    seed: int = DEFAULT_SEED,
    night_brightness: float = NIGHT_BRIGHTNESS,
) -> dict[str, Any]:
    """Todo el armado, ya sin motos ni cajas inválidas; devuelve el informe (solo cifras)."""
    built = build_pool(source, reference, night_brightness)
    invalid = invalid_boxes(built.annotations)
    pool = []
    for item in built.items:
        reason = classify(item.annotation, invalid)
        if reason is None:
            pool.append(item)
        else:
            built.discards.append(
                {
                    "archivo": item.annotation.name,
                    "caja": box_text(item.annotation.box),
                    "motivo": reason,
                    "detalle": f"relacion {aspect(item.annotation):.2f}",
                }
            )
    chosen = select(pool, count, night_min, seed, MAX_PER_PHOTO)
    build_outputs(source, [(f"{n:03d}", item) for n, item in enumerate(chosen, start=1)], out)
    write_csv(out / "descartes.csv", DISCARD_COLUMNS, built.discards)

    def count_reason(reason: str) -> int:
        return sum(d["motivo"] == reason for d in built.discards)

    report = {
        "fotos": len({(a.split, a.file) for a in built.annotations}),
        "cajas": len(built.annotations),
        "caja_chica": count_reason("caja_chica"),
        "foto_en_dataset_de_calibracion": count_reason("foto_en_dataset_de_calibracion"),
        "patente_en_dataset_de_calibracion": count_reason("patente_en_dataset_de_calibracion"),
        "duplicados": built.dropped,
        "probable_moto": count_reason(REASON_MOTORCYCLE),
        "caja_invalida": count_reason(REASON_INVALID),
        "candidatas": len(pool),
        "candidatas_de_noche": sum(item.light == LIGHT_NIGHT for item in pool),
        **chosen_stats(chosen),
        "parametros": parameters(count, night_min, seed, night_brightness),
    }
    (out / "informe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


@dataclass(frozen=True)
class Swap:
    """Una caja del set anterior que sale y la candidata que ocupa su id."""

    ident: str
    removed: Described
    reason: str
    replacement: Described


def plan_swaps(
    first: Sequence[Described],
    pool: Sequence[Described],
    invalid: set[Annotation],
    seed: int,
    per_photo: int = MAX_PER_PHOTO,
) -> list[Swap]:
    """Reemplazos de las cajas que no sirven por candidatas de auto que no estaban elegidas.

    Salen las probables motos, las cajas inválidas y las que pasan el tope por foto (se conservan las de menor
    id). Con la misma semilla; primero las estimadas de noche, sin pasar el tope. Cada reemplazo toma el id de
    la caja que sale.
    """
    per_photo_count: Counter[tuple[str, str]] = Counter()
    removed: list[tuple[int, Described, str]] = []
    for position, item in enumerate(first):
        reason = classify(item.annotation, invalid)
        if reason is None:
            key = photo_of(item.annotation)
            if per_photo_count[key] >= per_photo:
                reason = REASON_PHOTO_CAP
            else:
                per_photo_count[key] += 1
        if reason is not None:
            removed.append((position, item, reason))
    taken = {id(item) for item in first}
    eligible = [item for item in pool if id(item) not in taken and classify(item.annotation, invalid) is None]
    nights = [item for item in eligible if item.light == LIGHT_NIGHT]
    days = [item for item in eligible if item.light != LIGHT_NIGHT]
    rng = random.Random(seed)  # noqa: S311 - muestreo de fotos, no criptografía
    rng.shuffle(nights)
    rng.shuffle(days)
    picks: list[Described] = []
    for candidate in nights + days:
        if len(picks) == len(removed):
            break
        key = photo_of(candidate.annotation)
        if per_photo_count[key] < per_photo:
            picks.append(candidate)
            per_photo_count[key] += 1
    if len(picks) < len(removed):
        raise ValueError(f"hay {len(picks)} candidatas de auto y se necesitan {len(removed)}")
    return [
        Swap(f"{position + 1:03d}", item, reason, pick)
        for (position, item, reason), pick in zip(removed, picks, strict=True)
    ]


def free_backup(out: Path) -> Path:
    """`<out>-v1`, o el primer `<out>-vN` que no exista."""
    number = 1
    while out.with_name(f"{out.name}-v{number}").exists():
        number += 1
    return out.with_name(f"{out.name}-v{number}")


def refresh(
    source: Path,
    reference: Path,
    out: Path,
    count: int = DEFAULT_COUNT,
    night_min: int = DEFAULT_NIGHT_MIN,
    seed: int = DEFAULT_SEED,
    night_brightness: float = NIGHT_BRIGHTNESS,
    dry_run: bool = False,
    original: Path | None = None,
) -> dict[str, Any]:
    """Limpia un set: reemplaza motos, cajas inválidas y el exceso por foto, y conserva los demás ids.

    Parte de `original` (por defecto `out`): reconstruye su selección (mismos filtros y semilla) y la compara
    con su `etiquetas.csv`. Escribe el set limpio en `out`; si `out` existe, antes lo copia a `<out>-vN`.
    No pisa etiquetas ya cargadas (en `etiquetas.csv` o en `etiquetas.xlsx`), ni en `original` ni en `out`.
    Si hay un `etiquetas.xlsx` sin etiquetas, lo regenera con las miniaturas nuevas.
    """
    original = original or out
    rows = ocr_labels.read_label_rows(original)
    if ocr_labels.has_labels(rows):
        raise ValueError("el set ya tiene etiquetas cargadas (en el csv o el xlsx): no se pisa")
    if original != out and (out / "etiquetas.csv").exists() and ocr_labels.labels_loaded(out):
        raise ValueError("el set de --out ya tiene etiquetas cargadas (en el csv o el xlsx): no se pisa")
    built = build_pool(source, reference, night_brightness)
    first = select(built.items, count, night_min, seed)
    if [row["archivo"] for row in rows] != [item.annotation.name for item in first]:
        raise ValueError("el set de la carpeta no es el que arman estos filtros y esta semilla")
    invalid = invalid_boxes(built.annotations)
    swaps = plan_swaps(first, built.items, invalid, seed)
    final = list(first)
    by_ident = {f"{n:03d}": n - 1 for n in range(1, len(first) + 1)}
    for swap in swaps:
        final[by_ident[swap.ident]] = swap.replacement
    nights = sum(item.light == LIGHT_NIGHT for item in final)
    if nights < night_min:
        raise ValueError(f"quedarían {nights} de noche y se piden al menos {night_min}")
    swapped = {swap.ident for swap in swaps}
    summary: dict[str, Any] = {
        "reemplazadas": len(swaps),
        "reemplazadas_probable_moto": sum(s.reason == REASON_MOTORCYCLE for s in swaps),
        "reemplazadas_caja_invalida": sum(s.reason == REASON_INVALID for s in swaps),
        "reemplazadas_exceso_por_foto": sum(s.reason == REASON_PHOTO_CAP for s in swaps),
        "reemplazadas_estimadas_de_noche": sum(s.removed.light == LIGHT_NIGHT for s in swaps),
        "reemplazos_estimados_de_noche": sum(s.replacement.light == LIGHT_NIGHT for s in swaps),
        "conservadas": len(first) - len(swaps),
        **chosen_stats(final),
    }
    if dry_run:
        summary["swaps"] = swaps
        return summary
    if out.exists():
        shutil.copytree(out, free_backup(out))
    previous = {row["id"]: row for row in rows if row["id"] not in swapped}
    build_outputs(
        source, [(f"{n:03d}", item) for n, item in enumerate(final, start=1)], out, previous=previous
    )
    write_csv(
        out / "reemplazos.csv",
        REPLACEMENT_COLUMNS,
        [
            {
                "id": swap.ident,
                "archivo": swap.removed.annotation.name,
                "caja": box_text(swap.removed.annotation.box),
                "motivo": swap.reason,
                "reemplazada_por": f"{swap.replacement.annotation.name} "
                f"{box_text(swap.replacement.annotation.box)}",
            }
            for swap in swaps
        ],
    )
    if original != out and (original / "descartes.csv").exists():
        shutil.copyfile(original / "descartes.csv", out / "descartes.csv")
    if (out / ocr_labels.XLSX_NAME).exists() or (original / ocr_labels.XLSX_NAME).exists():
        ocr_labels.build_xlsx(out, source)  # las miniaturas de los reemplazos tienen que verse
        summary["xlsx_regenerado"] = True
    report_path = original / "informe.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.pop("limpieza", None)
    report.update(chosen_stats(final))
    report["limpieza"] = {k: v for k, v in summary.items() if k.startswith(("reemplaz", "conserv"))}
    report["parametros"] = parameters(count, night_min, seed, night_brightness)
    (out / "informe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    eval_dir = os.environ.get("EVAL_DATA_DIR")
    base = Path(eval_dir) if eval_dir else None
    parser.add_argument(
        "--src", type=Path, default=base / DEFAULT_SOURCE if base else None, help="export COCO"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=base / DEFAULT_REFERENCE if base else None,
        help="export COCO del dataset 'patentes' (calibración del OCR)",
    )
    parser.add_argument(
        "--out", type=Path, default=base / DEFAULT_OUT if base else None, help="carpeta de salida"
    )
    parser.add_argument("--n", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--night-min", type=int, default=DEFAULT_NIGHT_MIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--night-brightness", type=float, default=NIGHT_BRIGHTNESS)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="limpia un set (motos, cajas inválidas y exceso por foto) y escribe el resultado en --out",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=None,
        help="con --refresh, el set del que parte (por defecto --out); --out se respalda en <out>-vN",
    )
    args = parser.parse_args(argv)
    for name in ("src", "reference", "out"):
        if getattr(args, name) is None:
            parser.error(f"indicá --{name} o definí EVAL_DATA_DIR")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [args.src, args.reference, args.out] + ([args.original] if args.original else [])
    if any(path.resolve().is_relative_to(REPO_ROOT) for path in paths):
        print("Las fotos y la salida no pueden estar dentro del repositorio: tienen patentes de terceros.")
        return 1
    existing = [name for name in ("etiquetas.csv", ocr_labels.XLSX_NAME) if (args.out / name).exists()]
    if not args.refresh and existing:
        print(f"{args.out / existing[0]} ya existe: no se pisa (puede tener etiquetas). Usá otra --out.")
        return 1
    try:
        if args.refresh:
            report = refresh(
                args.src,
                args.reference,
                args.out,
                args.n,
                args.night_min,
                args.seed,
                args.night_brightness,
                original=args.original,
            )
        else:
            report = run(
                args.src, args.reference, args.out, args.n, args.night_min, args.seed, args.night_brightness
            )
    except (ValueError, OSError, ImportError) as error:
        print(error)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
