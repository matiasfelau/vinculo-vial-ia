"""Evalúa el OCR (puerto `ocr`) sobre el export COCO del dataset "patentes" de Roboflow.

Por cada caja anotada recorta la foto igual que el servicio (`crop_box`, sin padding), la lee con
el adaptador seleccionado y compara el texto con la etiqueta (el nombre de la clase, normalizado y
sin espacios) usando `domain.plate.plate_check`. Descarta los marcadores de posición (patentes con
todos los dígitos en cero, como `AA000AA`).

Con `--labeled-set` evalúa en cambio el set de `ocr_set.py` etiquetado a mano: lee `etiquetas.xlsx` (o
`etiquetas.csv`) de esa carpeta y recorta de la foto original con la caja de `cajas.csv`. Solo cuentan las
filas con estado `legible`; el `formato` y la `luz` etiquetados arman los cortes del informe. El bloque
`en_alcance` del informe es el de las patentes legibles de auto (anterior y Mercosur): lectura exacta, la
distribución de cada lectura contra su propia etiqueta y U2 (patentes correctas marcadas como patente
distinta, con la cota de una cola al 95 %), en total y por luz. `--export-mlkit DIR` no corre ningún
modelo: guarda esos recortes con su id, sin el texto, para leerlos con otro lector.

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
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import Box, EvidenceImage, EvidenceRole
from vvi_ia.domain.plate import TO_DIGIT, PlateCheck, canonicalize, plate_check
from vvi_ia.service.ports import ImageDecodeError
from vvi_ia.service.validate_evidence import crop_box

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_labels  # noqa: E402 - el set etiquetado, en xlsx o csv
from presence import upper_bound  # noqa: E402 - cota de una cola

SPLITS = ("train", "valid", "test")
ANNOTATIONS = "_annotations.coco.json"
DEFAULT_MIN_CONF = 0.78
SWEEP = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
PROPOSAL_FLOOR = 0.50
PROPOSAL_MARGIN = 0.01
PROPOSAL_CEIL = 0.99
DEFAULT_MATCH_SAMPLES = 20
FORMATS = {"LLDDDLL": "mercosur", "LLLDDD": "anterior"}
IN_SCOPE_FORMATS = frozenset(FORMATS.values())  # los que lee el servicio: una sola línea
OTHER_FORMAT = "otro"
OVERRIDE_COLUMNS = ("archivo", "etiqueta", "motivo")
EXCLUDE = "EXCLUIR"
DATASET_DIR = "ocr-roboflow"
PHOTOS_DIR = "ocr-ar"  # el export COCO de las fotos del set etiquetado
OVERRIDES_PATH = Path("ocr") / "label_overrides.csv"


@dataclass(frozen=True)
class Sample:
    """Una caja anotada: de qué foto sale, dónde está y qué patente es."""

    split: str
    file: str
    box: Box
    label: str
    image_size: tuple[int, int] = (0, 0)  # (ancho, alto) de la foto completa
    corrected: bool = False  # la etiqueta viene del archivo de correcciones, no del dataset
    format: str = ""  # el formato etiquetado a mano (set de `ocr_set.py`); vacío: sale de la etiqueta
    light: str = ""  # `dia` o `noche`, etiquetada a mano (set de `ocr_set.py`)
    ident: str = ""  # el id de la fila en el set de `ocr_set.py`


@dataclass(frozen=True)
class Loaded:
    """Lo que se cargó del dataset."""

    samples: list[Sample]
    discarded: Counter[str]  # cajas que no se pueden evaluar, por motivo
    unreadable: int  # cajas excluidas por el archivo de correcciones: la patente no se lee ni a ojo


@dataclass(frozen=True)
class Reading:
    """Lo que leyó el OCR sobre una muestra (`text` vacío y `conf` 0 si no leyó nada)."""

    sample: Sample
    text: str
    conf: float
    failed: bool  # el adaptador devolvió `None`
    raw: str = ""  # lo que leyó el modelo antes de extraer la patente (si el adaptador lo ofrece)

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


def sample_format(sample: Sample) -> str:
    """El formato etiquetado a mano si lo hay; si no, el que sale de la forma de la etiqueta."""
    return sample.format or label_format(sample.label)


class OverridesError(ValueError):
    """El archivo de correcciones no es válido."""


def load_overrides(path: Path) -> dict[str, str]:
    """`archivo → etiqueta correcta` (o `EXCLUIR`) del CSV de correcciones; vacío si no existe.

    Columnas: `archivo`, `etiqueta` y `motivo`. El export original del dataset no se toca: las
    correcciones viven en este archivo, fuera del repo.
    """
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(OVERRIDE_COLUMNS) - set(reader.fieldnames):
            raise OverridesError(f"{path.name} necesita las columnas {', '.join(OVERRIDE_COLUMNS)}")
        overrides: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            file = (row["archivo"] or "").strip()
            value = (row["etiqueta"] or "").strip()
            label = EXCLUDE if value.upper() == EXCLUDE else canonicalize(value)
            if not file or not label:
                raise OverridesError(f"{path.name}, fila {row_number}: faltan el archivo o la etiqueta")
            if file in overrides:
                raise OverridesError(f"{path.name}, fila {row_number}: el archivo {file} está repetido")
            overrides[file] = label
    return overrides


def load_samples(data_dir: Path, splits: Sequence[str], overrides: dict[str, str] | None = None) -> Loaded:
    """Cajas del dataset con las correcciones aplicadas, y cuántas se descartaron.

    Una corrección vale para todas las cajas de esa foto. Si nombra una foto que no está en el
    dataset, es un error (casi seguro un nombre mal escrito).
    """
    overrides = overrides or {}
    samples: list[Sample] = []
    discarded: Counter[str] = Counter()
    unreadable = 0
    seen: set[str] = set()
    for split in splits:
        path = data_dir / split / ANNOTATIONS
        if not path.is_file():
            raise FileNotFoundError(f"falta {split}/{ANNOTATIONS}")
        coco = json.loads(path.read_text(encoding="utf-8"))
        images = {image["id"]: image for image in coco["images"]}
        names = {category["id"]: category["name"] for category in coco["categories"]}
        seen.update(image["file_name"] for image in images.values())
        for annotation in coco["annotations"]:
            image = images[annotation["image_id"]]
            file = image["file_name"]
            override = overrides.get(file)
            if override == EXCLUDE:
                unreadable += 1
                continue
            label = canonicalize(override if override else names[annotation["category_id"]])
            if not label:
                discarded["etiqueta vacía"] += 1
            elif is_placeholder(label):
                discarded["marcador de posición"] += 1
            else:
                x, y, width, height = annotation["bbox"]
                box = Box(x1=x, y1=y, x2=x + width, y2=y + height)
                size = (int(image["width"]), int(image["height"]))
                samples.append(Sample(split, file, box, label, size, corrected=override is not None))
    unknown = sorted(set(overrides) - seen)
    if unknown:
        raise OverridesError(
            f"las correcciones nombran fotos que no están en el dataset: {', '.join(unknown)}"
        )
    return Loaded(samples, discarded, unreadable)


def load_set_samples(set_dir: Path, incomplete: bool = False) -> Loaded:
    """Muestras del set etiquetado a mano con `ocr_set.py`: `etiquetas.xlsx` si existe, si no `etiquetas.csv`.

    La caja sale de `cajas.csv` y el recorte, de la foto original (igual que el servicio, sin el margen del
    recorte del set). Solo cuentan las filas con estado `legible`: `ilegible` se excluye como en el archivo de
    correcciones y `fuera_de_alcance` se descarta. Si falta etiquetar alguna fila, falla; con `incomplete`
    las salta.
    """
    rows = ocr_labels.read_label_rows(set_dir)
    invalid = [row["id"] for row in rows if ocr_labels.row_problems(row)]
    if invalid:
        shown = ", ".join(invalid[:10]) + (" ..." if len(invalid) > 10 else "")
        raise ocr_labels.LabelsError(f"hay valores no válidos en {len(invalid)} filas (ids {shown})")
    boxes = {row["id"]: row["caja"] for row in ocr_labels.read_csv_rows(set_dir / ocr_labels.BOXES_NAME)}
    pending = [
        row["id"]
        for row in rows
        if not row["estado"] or (row["estado"] == ocr_labels.STATE_LEGIBLE and not row["texto"])
    ]
    if pending and not incomplete:
        shown = ", ".join(pending[:10]) + (" ..." if len(pending) > 10 else "")
        raise ocr_labels.LabelsError(
            f"faltan etiquetas en {len(pending)} filas (ids {shown}): completalas o usá --incomplete"
        )
    samples: list[Sample] = []
    discarded: Counter[str] = Counter()
    unreadable = 0
    for row in rows:
        if row["id"] in pending:
            discarded["sin etiquetar"] += 1
        elif row["estado"] == "ilegible":
            unreadable += 1
        elif row["estado"] != ocr_labels.STATE_LEGIBLE:
            discarded["fuera de alcance"] += 1
        else:
            split, file = row["archivo"].split("/", 1)
            x, y, width, height = (float(part) for part in boxes[row["id"]].split())
            box = Box(x1=x, y1=y, x2=x + width, y2=y + height)
            samples.append(
                Sample(
                    split,
                    file,
                    box,
                    canonicalize(row["texto"]),
                    format=row["formato"],
                    light=row["luz"],
                    ident=row["id"],
                )
            )
    return Loaded(samples, discarded, unreadable)


# --- lectura -------------------------------------------------------------------------------


def read_samples(
    samples: Iterable[Sample], data_dir: Path, reader: Any
) -> tuple[list[tuple[Reading, Any]], Counter[str]]:
    """Lee cada muestra. Devuelve las lecturas con su recorte y cuántas se descartaron, por motivo."""
    decoder = CvImageDecoder()
    images: OrderedDict[tuple[str, str], EvidenceImage | None] = OrderedDict()
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
            while len(images) > IMAGE_CACHE:  # las fotos de celular decodificadas pesan decenas de MB
                images.popitem(last=False)
        else:
            images.move_to_end(key)
        image = images[key]
        if image is None:
            discarded["foto ilegible"] += 1
            continue
        crop = crop_box(image, sample.box)
        if crop is None:
            discarded["caja vacía"] += 1
            continue
        if sample.image_size == (0, 0):
            height, width = image.pixels.shape[:2]
            sample = replace(sample, image_size=(int(width), int(height)))
        read = reader.read(crop)
        text, conf = read if read is not None else ("", 0.0)
        raw_text = getattr(reader, "raw_text", None)
        raw = (raw_text(crop) if raw_text is not None else text) or ""
        reading = Reading(sample, text, conf, failed=read is None, raw=raw)
        readings.append((reading, crop))
    return readings, discarded


# --- exportación para otro lector ------------------------------------------------------------

MLKIT_INDEX = "indice.csv"
MLKIT_INDEX_COLUMNS = ("id", "formato", "luz")  # nunca el texto: es lo que el otro lector tiene que leer
MLKIT_CROPS = ("servicio", "set")


def export_crops(
    samples: Iterable[Sample], photos_dir: Path, out_dir: Path, set_dir: Path, crop: str = "servicio"
) -> dict[str, int]:
    """Guarda `<id>.png` de las patentes legibles de auto y un `indice.csv` (id, formato, luz), sin el texto.

    Con `crop = servicio` es el mismo recorte que lee el OCR (la caja sin margen, de la foto original); con
    `set`, el recorte con margen del set (`recortes/`). Reemplaza lo que una exportación anterior dejó en la
    carpeta.
    """
    if crop not in MLKIT_CROPS:
        raise ValueError(f"recorte desconocido: {crop}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in (*out_dir.glob("*.png"), out_dir / MLKIT_INDEX):
        old.unlink(missing_ok=True)
    decoder = CvImageDecoder()
    rows: list[dict[str, str]] = []
    skipped = 0
    for sample in samples:
        if sample_format(sample) not in IN_SCOPE_FORMATS or not sample.ident:
            continue
        target = out_dir / f"{sample.ident}.png"
        if crop == "set":
            source = set_dir / "recortes" / f"{sample.ident}.png"
            if not source.is_file():
                skipped += 1
                continue
            target.write_bytes(source.read_bytes())
        else:
            try:
                pixels = decoder.decode((photos_dir / sample.split / sample.file).read_bytes())
            except (OSError, ImageDecodeError):
                skipped += 1
                continue
            cut = crop_box(EvidenceImage(role=EvidenceRole.PLATE, index=0, pixels=pixels), sample.box)
            if cut is None:
                skipped += 1
                continue
            ok, data = cv2.imencode(".png", cv2.cvtColor(cut, cv2.COLOR_RGB2BGR))
            if not ok:
                skipped += 1
                continue
            target.write_bytes(data.tobytes())
        rows.append({"id": sample.ident, "formato": sample_format(sample), "luz": sample.light})
    with (out_dir / MLKIT_INDEX).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MLKIT_INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return {"exportados": len(rows), "salteados": skipped}


# --- causas --------------------------------------------------------------------------------

TWO_LINES = "dos líneas (moto)"
STRIPE = "lee la franja (REPÚBLICA ARGENTINA / MERCOSUR)"
CUT = "recorte cortado"
ANGLE = "ángulo"
BLUR = "desenfoque o baja resolución"
GLARE = "reflejo"
DOUBTFUL_LABEL = "etiqueta dudosa"
CONFUSION = "confusión de caracteres"
OTHER_CAUSE = "otras (revisar a ojo)"
CAUSES = (TWO_LINES, STRIPE, CUT, ANGLE, BLUR, GLARE, DOUBTFUL_LABEL, CONFUSION, OTHER_CAUSE)

STRIPE_WORDS = ("REPUBLICA", "ARGENTINA", "MERCOSUR")
STRIPE_GRAM = 4  # un fragmento de 4 letras seguidas de alguna de esas palabras
MAX_TWO_LINES_ASPECT = 1.4  # las patentes de una línea miden cerca de 3:1; las de moto, casi cuadradas
MAX_TILT_DEGREES = 10.0
MIN_PIXELS_WIDTH = 100  # recorte más chico que esto se agranda para llegar a la entrada del modelo
MIN_PIXELS_HEIGHT = 45
MIN_SHARPNESS = 200.0  # varianza del laplaciano con el recorte normalizado a 256 px de ancho
SHARPNESS_WIDTH = 256
WASHED_OUT_MEAN = 200  # un reflejo o una sobreexposición lava el recorte: claro y casi sin contraste
WASHED_OUT_STD = 40  # una patente blanca normal tiene los caracteres oscuros y mucho más contraste
MIN_CHARACTERS = 4  # componentes para estimar la inclinación de la línea de texto
CHAR_HEIGHT_RANGE = (0.2, 0.8)  # alto de un carácter respecto del alto del recorte
MAX_CHAR_WIDTH = 0.3  # ancho máximo de un carácter respecto del ancho del recorte
MAX_ALIGNMENT_RATIO = 0.02  # los centros de los caracteres de una línea de texto están casi en una recta
ADAPTIVE_BLOCK = 25
ADAPTIVE_OFFSET = 8
MIN_MISSING_CHARS = 2  # una sola letra de menos es un error de lectura, no un recorte cortado
MIN_CHAR_AREA = 20
DOUBTFUL_MIN_CONF = 0.90  # una lectura tan segura de una patente válida que difiere en ≥2 caracteres
EDGE_PIXELS = 1  # la caja toca el borde de la foto
IMAGE_CACHE = 4  # fotos enteras decodificadas que se guardan a la vez (300 fotos de 12 MP no entran en RAM)


# Caracteres que el OCR confunde por su forma. Las tablas de `domain/plate.py` solo corrigen las
# confusiones entre letra y dígito según la posición (O/0, I/1, B/8...); acá se agregan otras que
# aparecen en la práctica, como el 0 leído G en la tipografía Mercosur o la W leída M.
SIMILAR_FAMILIES = ("0OQDG", "6G", "1I", "2Z", "5S", "8B", "MWNH", "UVY")


def _similar_pairs() -> frozenset[tuple[str, str]]:
    pairs = {(a, b) for family in SIMILAR_FAMILIES for a in family for b in family}
    for letter_code, digit_code in TO_DIGIT.items():  # las tablas del dominio, letra ↔ dígito
        pairs.update({(chr(letter_code), chr(digit_code)), (chr(digit_code), chr(letter_code))})
    return frozenset(pairs)


SIMILAR = _similar_pairs()


def only_confusions(label: str, text: str) -> bool:
    """Mismo largo y cada diferencia es entre caracteres de forma parecida (O/0, I/1, M/W...)."""
    if len(label) != len(text) or label == text:
        return False
    return all(a == b or (a, b) in SIMILAR for a, b in zip(label, text, strict=True))


def has_stripe(raw: str) -> bool:
    """El texto crudo del OCR contiene un fragmento de la franja de la placa."""
    text = canonicalize(raw)
    return any(
        word[i : i + STRIPE_GRAM] in text for word in STRIPE_WORDS for i in range(len(word) - STRIPE_GRAM + 1)
    )


def text_tilt(crop: Any) -> float | None:
    """Inclinación en grados de la línea que forman los caracteres; `None` si no hay caracteres claros."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    gray = cv2.resize(
        gray,
        (SHARPNESS_WIDTH, max(16, round(SHARPNESS_WIDTH * height / width))),
        interpolation=cv2.INTER_AREA,
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (otsu > 0).mean() > 0.5:  # los caracteres son la clase minoritaria
        otsu = 255 - otsu
    adaptive = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK,
        ADAPTIVE_OFFSET,
    )
    low, high = (limit * gray.shape[0] for limit in CHAR_HEIGHT_RANGE)
    for binary in (otsu, adaptive):  # el umbral adaptativo rescata las placas con franja o marco oscuros
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
        keep = [
            i
            for i in range(1, count)
            if low <= stats[i, cv2.CC_STAT_HEIGHT] <= high
            and stats[i, cv2.CC_STAT_WIDTH] < gray.shape[1] * MAX_CHAR_WIDTH
            and stats[i, cv2.CC_STAT_HEIGHT] > 0.8 * stats[i, cv2.CC_STAT_WIDTH]
            and stats[i, cv2.CC_STAT_AREA] >= MIN_CHAR_AREA
        ]
        if len(keep) < MIN_CHARACTERS:
            continue
        points = centroids[keep] - centroids[keep].mean(axis=0)
        covariance = points.T @ points
        small, large = np.linalg.eigvalsh(covariance)
        if large <= 0 or small / large > MAX_ALIGNMENT_RATIO:
            continue  # los centros no forman una línea: no es texto, es el marco o la suciedad
        return float(
            0.5 * math.degrees(math.atan2(2 * covariance[0, 1], covariance[0, 0] - covariance[1, 1]))
        )
    return None


def sharpness(crop: Any) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    gray = cv2.resize(
        gray, (SHARPNESS_WIDTH, max(8, round(SHARPNESS_WIDTH * height / width))), interpolation=cv2.INTER_AREA
    )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass(frozen=True)
class Diagnosis:
    """Por qué una lectura no coincide exacto: la causa principal y todos los factores presentes."""

    cause: str
    factors: tuple[str, ...]


def diagnose(reading: Reading, crop: Any) -> Diagnosis:
    """Causa de una lectura sin coincidencia exacta.

    Las condiciones de la foto (moto, franja, recorte, inclinación, resolución, reflejo) explican más
    que el mecanismo del error y van primero; la etiqueta dudosa y la confusión de caracteres se
    asignan cuando no hay ninguna condición adversa. Los factores no son excluyentes.
    """
    sample, label, text = reading.sample, reading.sample.label, canonicalize(reading.text)
    box, (image_width, image_height) = sample.box, sample.image_size
    factors: list[str] = []
    if sample_format(sample) not in IN_SCOPE_FORMATS:
        factors.append(TWO_LINES)
    if has_stripe(reading.raw):
        factors.append(STRIPE)
    touches_edge = image_width > 0 and (
        box.x1 <= EDGE_PIXELS
        or box.y1 <= EDGE_PIXELS
        or box.x2 >= image_width - EDGE_PIXELS
        or box.y2 >= image_height - EDGE_PIXELS
    )
    missing = len(label) - len(text)
    ends_with_label = bool(text) and (label.startswith(text) or label.endswith(text))
    if (touches_edge and missing > 0) or (missing >= MIN_MISSING_CHARS and ends_with_label):
        factors.append(CUT)
    tilt = text_tilt(crop)
    if tilt is not None and abs(tilt) >= MAX_TILT_DEGREES:
        factors.append(ANGLE)
    height, width = crop.shape[:2]
    if width < MIN_PIXELS_WIDTH or height < MIN_PIXELS_HEIGHT or sharpness(crop) < MIN_SHARPNESS:
        factors.append(BLUR)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if gray.mean() >= WASHED_OUT_MEAN and gray.std() <= WASHED_OUT_STD:
        factors.append(GLARE)
    if (
        reading.content == PlateCheck.MISMATCH
        and reading.conf >= DOUBTFUL_MIN_CONF
        and label_format(text) != OTHER_FORMAT
    ):
        factors.append(DOUBTFUL_LABEL)
    if only_confusions(label, text):
        factors.append(CONFUSION)
    cause = next((name for name in CAUSES if name in factors), OTHER_CAUSE)
    return Diagnosis(cause, tuple(factors))


def causes_report(diagnoses: Sequence[Diagnosis]) -> dict[str, Any]:
    """Cifras: cuántas lecturas sin coincidencia exacta tiene cada causa principal y cada factor."""
    primary = Counter(d.cause for d in diagnoses)
    factors = Counter(factor for d in diagnoses for factor in d.factors)
    return {
        "lecturas_sin_coincidencia_exacta": len(diagnoses),
        "causa_principal": {cause: primary.get(cause, 0) for cause in CAUSES},
        "factores_presentes": {cause: factors.get(cause, 0) for cause in CAUSES if cause != OTHER_CAUSE},
    }


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


def u2(readings: Sequence[Reading], min_conf: float) -> dict[str, Any]:
    """U2: patentes correctas marcadas como patente distinta (`MISMATCH`, RED en el servicio), con su cota.

    Cada lectura se compara con su propia etiqueta. La cota es de una cola al 95 % (regla del 3 con 0 casos).
    """
    marked = sum(reading.check(min_conf) == PlateCheck.MISMATCH for reading in readings)
    total = len(readings)
    return {
        "marcadas": marked,
        "lecturas": total,
        "tasa": round(marked / total, 4) if total else None,
        "cota_95": upper_bound(marked, total),
    }


def scope_report(in_scope: Sequence[Reading], min_conf: float) -> dict[str, Any]:
    """Las patentes legibles de auto (anterior y Mercosur): lectura exacta, distribución y U2, y por luz.

    La distribución es la de cada lectura contra su propia etiqueta: coincide (`MATCH`), difiere en un
    carácter (`FUZZY`), ilegible (`UNREADABLE`: sin lectura, poco segura o sin la estructura de una patente)
    y patente distinta (`MISMATCH`).
    """
    report: dict[str, Any] = {**summarize(in_scope, min_conf), "U2": u2(in_scope, min_conf)}
    by_light: dict[str, list[Reading]] = {}
    for reading in in_scope:
        by_light.setdefault(reading.sample.light or "sin dato", []).append(reading)
    if any(reading.sample.light for reading in in_scope):
        report["por_luz"] = {
            light: {**summarize(items, min_conf), "U2": u2(items, min_conf)}
            for light, items in sorted(by_light.items())
        }
    return report


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


def cross_pairs(readings: Sequence[Reading], min_conf: float) -> dict[str, Any]:
    """R3 con pares cruzados: la lectura de una foto contra la etiqueta de OTRA patente.

    Son casos que deberían dar `PLATE_MISMATCH` (la patente de la foto no es la reportada). Se cuenta qué
    parte sale MISMATCH con el umbral `min_conf` (RED en el servicio) y cuál se escapa como GREEN: `FUZZY`
    (una etiqueta parecida), `UNREADABLE` (lectura poco segura) o `MATCH` (etiquetas parecidas en el texto).
    """
    labels = sorted({reading.sample.label for reading in readings})
    total: Counter[str] = Counter()
    sure: Counter[str] = Counter()
    for reading in readings:
        if reading.failed or not reading.text:
            continue
        for label in labels:
            if label == reading.sample.label:
                continue
            check = plate_check(label, reading.text, reading.conf, min_conf)
            total[check.value] += 1
            if reading.conf >= min_conf:
                sure[check.value] += 1

    def shares(counts: Counter[str]) -> dict[str, Any]:
        n = sum(counts.values())
        return {
            "pares": n,
            "mismatch": counts[PlateCheck.MISMATCH.value],
            "mismatch_pct": round(100 * counts[PlateCheck.MISMATCH.value] / n, 2) if n else None,
            "distribucion": {check.value: counts.get(check.value, 0) for check in PlateCheck},
        }

    return {"todos": shares(total), "con_lectura_segura": shares(sure)}


def build_report(
    readings: Sequence[Reading],
    min_conf: float,
    model_version: str,
    discarded: Counter[str],
    total_boxes: int,
    unreadable: int = 0,
    diagnoses: Sequence[Diagnosis] = (),
) -> dict[str, Any]:
    in_scope = [reading for reading in readings if sample_format(reading.sample) in IN_SCOPE_FORMATS]
    groups: dict[str, dict[str, list[Reading]]] = {"por_split": {}, "por_formato": {}}
    if any(reading.sample.light for reading in readings):
        groups["por_luz"] = {}
    for reading in readings:
        groups["por_split"].setdefault(reading.sample.split, []).append(reading)
        groups["por_formato"].setdefault(sample_format(reading.sample), []).append(reading)
        if "por_luz" in groups:
            groups["por_luz"].setdefault(reading.sample.light or "sin dato", []).append(reading)
    return {
        "modelo": model_version,
        "min_conf": min_conf,
        "cajas_del_dataset": total_boxes,
        "descartadas": dict(discarded),
        # Excluidas en el archivo de correcciones: la patente no se lee ni a ojo. No cuentan como fallas.
        "ilegibles": unreadable,
        "etiquetas_corregidas": sum(reading.sample.corrected for reading in readings),
        "total": summarize(readings, min_conf),
        "causas": causes_report(diagnoses),
        **{
            name: {key: summarize(items, min_conf) for key, items in sorted(group.items())}
            for name, group in groups.items()
        },
        "barrido_de_umbrales": sweep(readings),
        # Los formatos que no son Mercosur ni anterior son motos de dos líneas u otros: fuera de alcance.
        # (Los datos de `--labeled-set` traen el formato etiquetado a mano.)
        # fuera de alcance. La propuesta se calcula sin ellas; la otra se muestra por transparencia.
        "propuesta": propose_min_conf(in_scope),
        "propuesta_con_todos_los_formatos": propose_min_conf(readings),
        "lecturas_fuera_de_alcance": len(readings) - len(in_scope),
        "pares_cruzados": cross_pairs(in_scope, min_conf),
        "en_alcance": scope_report(in_scope, min_conf),
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


def write_audit(
    selection: Sequence[tuple[Reading, Any]],
    out_dir: Path,
    min_conf: float,
    diagnoses: dict[int, Diagnosis] | None = None,
) -> None:
    diagnoses = diagnoses or {}
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for old in audit_dir.glob("*.jpg"):
        old.unlink()
    rows = []
    for number, (reading, crop) in enumerate(selection, start=1):
        name = f"{number:03d}.jpg"
        cv2.imwrite(str(audit_dir / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        diagnosis = diagnoses.get(id(reading))
        rows.append(
            {
                "recorte": f"audit/{name}",
                "split": reading.sample.split,
                "foto": reading.sample.file,
                "etiqueta": reading.sample.label,
                "etiqueta_corregida": "si" if reading.sample.corrected else "no",
                "lectura": reading.text,
                "confianza": f"{reading.conf:.4f}",
                "resultado": reading.check(min_conf).value,
                "exacta": "si" if reading.exact else "no",
                "causa": diagnosis.cause if diagnosis else "",
                "factores": "; ".join(diagnosis.factors) if diagnosis else "",
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
        "--data-dir",
        type=Path,
        default=os.environ.get("EVAL_DATA_DIR"),
        help=f"carpeta de datos de evaluación (EVAL_DATA_DIR); el dataset va en {DATASET_DIR}/",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, help=f"export COCO (por defecto <data-dir>/{DATASET_DIR})"
    )
    parser.add_argument(
        "--overrides", type=Path, help=f"correcciones de etiquetas (por defecto <data-dir>/{OVERRIDES_PATH})"
    )
    parser.add_argument(
        "--labeled-set",
        type=Path,
        help="set etiquetado de ocr_set.py (lee etiquetas.xlsx o etiquetas.csv); en vez del dataset COCO",
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        help=f"con --labeled-set, el export con las fotos (por defecto <data-dir>/{PHOTOS_DIR})",
    )
    parser.add_argument(
        "--incomplete",
        action="store_true",
        help="con --labeled-set, saltar las filas sin etiquetar en vez de fallar",
    )
    parser.add_argument(
        "--export-mlkit",
        type=Path,
        metavar="DIR",
        help="con --labeled-set: guarda los recortes legibles de auto con su id, sin el texto (sin modelos)",
    )
    parser.add_argument(
        "--mlkit-crop",
        choices=MLKIT_CROPS,
        default="servicio",
        help="recorte a exportar: el del OCR (servicio, sin margen) o el del set (con margen del 15 %%)",
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
        help="OCR_MIN_CONF con el que se arma la distribución (por defecto el del entorno o 0,78)",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=None, help="por defecto, los que existan en el dataset"
    )
    parser.add_argument(
        "--match-samples", type=int, default=DEFAULT_MATCH_SAMPLES, help="aciertos a incluir en la auditoría"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.export_mlkit and not args.labeled_set:
        parser.error("--export-mlkit va con --labeled-set")
    required = [("--data-dir", args.data_dir)]
    if not args.export_mlkit:
        required += [("--out-dir", args.out_dir), ("--models-dir", args.models_dir)]
    for flag, value in required:
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    if not 0.0 <= args.min_conf <= 1.0:
        parser.error("--min-conf debe estar entre 0 y 1")
    args.dataset_dir = args.dataset_dir or args.data_dir / (PHOTOS_DIR if args.labeled_set else DATASET_DIR)
    if args.photos_dir:
        args.dataset_dir = args.photos_dir
    args.overrides = args.overrides or args.data_dir / OVERRIDES_PATH
    return args


def inside_repo(path: Path) -> bool:
    return path.resolve().is_relative_to(REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name, path in (
        ("el dataset", args.dataset_dir),
        ("las correcciones", args.overrides),
        *([("la salida", args.out_dir)] if args.out_dir else []),
        *([("el set etiquetado", args.labeled_set)] if args.labeled_set else []),
        *([("la exportación", args.export_mlkit)] if args.export_mlkit else []),
    ):
        if inside_repo(path):
            print(
                f"{name} no puede estar dentro del repositorio: tiene patentes de terceros.", file=sys.stderr
            )
            return 1
    splits = args.splits or [split for split in SPLITS if (args.dataset_dir / split / ANNOTATIONS).is_file()]
    if not splits and not args.labeled_set:
        print(f"No hay ningún split con {ANNOTATIONS} en el dataset.", file=sys.stderr)
        return 1
    if args.export_mlkit:
        try:
            loaded = load_set_samples(args.labeled_set, args.incomplete)
            exported = export_crops(
                loaded.samples, args.dataset_dir, args.export_mlkit, args.labeled_set, args.mlkit_crop
            )
        except (OSError, ValueError, ocr_labels.LabelsError, ImportError) as exc:
            print(f"No se pudo exportar: {exc}", file=sys.stderr)
            return 1
        print(
            f"Exportados {exported['exportados']} recortes (recorte {args.mlkit_crop}) en {args.export_mlkit}"
        )
        if exported["salteados"]:
            print(f"Salteados: {exported['salteados']}")
        return 0
    try:
        if args.labeled_set:
            loaded = load_set_samples(args.labeled_set, args.incomplete)
        else:
            loaded = load_samples(args.dataset_dir, splits, load_overrides(args.overrides))
        models = build_models({"ocr": args.model}, args.manifest, args.models_dir)
    except (FileNotFoundError, OverridesError, ModelSetupError, ocr_labels.LabelsError, ImportError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    discarded = loaded.discarded
    total_boxes = len(loaded.samples) + sum(discarded.values()) + loaded.unreadable
    items, skipped = read_samples(loaded.samples, args.dataset_dir, models.plate)
    discarded += skipped
    readings = [reading for reading, _ in items]

    diagnoses = {id(reading): diagnose(reading, crop) for reading, crop in items if not reading.exact}
    report = build_report(
        readings,
        args.min_conf,
        models.versions()["ocr"],
        discarded,
        total_boxes,
        loaded.unreadable,
        list(diagnoses.values()),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selection = audit_selection(items, args.min_conf, args.match_samples, args.seed)
    write_audit(selection, args.out_dir, args.min_conf, diagnoses)

    total, proposal = report["total"], report["propuesta"]
    every_format = report["propuesta_con_todos_los_formatos"]
    print(f"Modelo {report['modelo']} · {total_boxes} cajas · descartadas {dict(discarded) or 'ninguna'}")
    print(
        f"Ilegibles (excluidas): {report['ilegibles']} · "
        f"etiquetas corregidas: {report['etiquetas_corregidas']}"
    )
    print(f"Lecturas: {total['lecturas']} · fallas del modelo: {total['fallas_del_modelo']}")
    print(f"Coincidencia exacta: {total['coincidencia_exacta']} ({total['coincidencia_exacta_pct']} %)")
    print(f"Distribución con OCR_MIN_CONF={args.min_conf}: {total['distribucion']}")
    crossed = report["pares_cruzados"]
    print(
        f"R3, pares cruzados (patente de una foto contra la etiqueta de otra, solo formatos en alcance): "
        f"{crossed['todos']['mismatch_pct']} % sale MISMATCH de {crossed['todos']['pares']} pares "
        f"({crossed['todos']['distribucion']}); con lectura segura: "
        f"{crossed['con_lectura_segura']['mismatch_pct']} % de {crossed['con_lectura_segura']['pares']}"
    )
    scope = report["en_alcance"]
    print(
        f"Legibles de auto: {scope['lecturas']} · lectura exacta {scope['coincidencia_exacta']} "
        f"({scope['coincidencia_exacta_pct']} %) · distribución {scope['distribucion']} · "
        f"U2 {scope['U2']['marcadas']} de {scope['U2']['lecturas']} (cota 95 % {scope['U2']['cota_95']})"
    )
    for light, block in scope.get("por_luz", {}).items():
        print(
            f"  {light}: {block['lecturas']} lecturas · exacta {block['coincidencia_exacta']} "
            f"({block['coincidencia_exacta_pct']} %) · {block['distribucion']} · "
            f"U2 {block['U2']['marcadas']} de {block['U2']['lecturas']} (cota {block['U2']['cota_95']})"
        )
    print(
        f"Propuesta (sin {report['lecturas_fuera_de_alcance']} lecturas de otros formatos): "
        f"OCR_MIN_CONF={proposal['OCR_MIN_CONF']} "
        f"({proposal['lecturas_aceptadas_pct']} % de lecturas aceptadas)"
    )
    print(f"Propuesta con todos los formatos: OCR_MIN_CONF={every_format['OCR_MIN_CONF']}")
    print("Barrido de umbrales (todas las lecturas):")
    print("  umbral  aceptadas      %  match fuzzy mismatch")
    for row in report["barrido_de_umbrales"]:
        print(
            f"  {row['umbral']:>6}  {row['aceptadas']:>9}  {row['aceptadas_pct']!s:>5}  "
            f"{row['match']:>5} {row['fuzzy']:>5} {row['mismatch']:>8}"
        )
    causes = report["causas"]
    print(
        f"Causa principal de las {causes['lecturas_sin_coincidencia_exacta']} "
        "lecturas sin coincidencia exacta:"
    )
    for cause, count in causes["causa_principal"].items():
        print(f"  {count:>3}  {cause}")
    print(f"Auditoría: {len(selection)} recortes en {args.out_dir / 'audit.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
