"""Del formato de Open Images V7 al COCO que entrena RF-DETR (sin dependencias de FiftyOne).

Las cuatro clases salen en el orden de `LABELS` del adaptador: RF-DETR numera las clases por su posición
en la lista de categorías, y el adaptador lee las columnas del modelo en ese mismo orden. Cambiarlo
obliga a reentrenar.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vvi_ia.adapters.models.rfdetr_detector import LABELS

CLASSES: Sequence[str] = tuple(str(LABELS[column]) for column in sorted(LABELS))
OPEN_IMAGES_CLASSES: Mapping[str, str] = {
    "Car": "car",
    "Wheel": "wheel",
    "Vehicle registration plate": "plate",
    "Human face": "face",
}
# Lo que se baja primero para cada etapa: las patentes son lo más escaso, así que tienen su propia etapa
STAGES: Mapping[str, str] = {
    "plates": "Vehicle registration plate",
    "faces": "Human face",
    "cars": "Car",
}
MIN_SIDE_PX = 8  # una caja más chica que esto, en la foto original, no se puede aprender ni evaluar
FIRST_CATEGORY_ID = 1


@dataclass(frozen=True)
class Annotation:
    """Una caja de Open Images: coordenadas normalizadas `(x, y, ancho, alto)` y sus atributos."""

    label: str  # nombre de Open Images, por ejemplo "Vehicle registration plate"
    x: float
    y: float
    width: float
    height: float
    group_of: bool = False  # una caja que cubre un grupo de objetos
    depiction: bool = False  # un dibujo, una pintura o un juguete
    inside: bool = False  # la foto está tomada desde adentro del objeto


@dataclass(frozen=True)
class ImageRecord:
    """Una foto con todas sus cajas."""

    file_name: str
    width: int
    height: int
    annotations: Sequence[Annotation]


def is_usable(annotation: Annotation, width: int, height: int, min_side: int = MIN_SIDE_PX) -> bool:
    """Una caja entra al dataset si es de una de las cuatro clases y no es ambigua ni diminuta."""
    return (
        annotation.label in OPEN_IMAGES_CLASSES
        and not (annotation.group_of or annotation.depiction or annotation.inside)
        and annotation.width * width >= min_side
        and annotation.height * height >= min_side
    )


def to_coco(records: Iterable[ImageRecord], min_side: int = MIN_SIDE_PX) -> dict[str, Any]:
    """COCO con las cajas utilizables de cada foto. Las fotos sin ninguna se descartan."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_id = {name: FIRST_CATEGORY_ID + index for index, name in enumerate(CLASSES)}
    for record in records:
        kept = [a for a in record.annotations if is_usable(a, record.width, record.height, min_side)]
        if not kept:
            continue
        image_id = len(images) + 1
        images.append(
            {"id": image_id, "file_name": record.file_name, "width": record.width, "height": record.height}
        )
        for annotation in kept:
            x1 = max(0.0, annotation.x * record.width)
            y1 = max(0.0, annotation.y * record.height)
            x2 = min(float(record.width), (annotation.x + annotation.width) * record.width)
            y2 = min(float(record.height), (annotation.y + annotation.height) * record.height)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id[OPEN_IMAGES_CLASSES[annotation.label]],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "area": round((x2 - x1) * (y2 - y1), 2),
                    "iscrowd": 0,
                }
            )
    return {
        "info": {"description": "Subconjunto de Open Images V7 para el detector"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id[name], "name": name, "supercategory": "none"} for name in CLASSES],
    }


def class_counts(coco: Mapping[str, Any]) -> dict[str, int]:
    """Cajas por clase (en cifras: no hay nada que revele el contenido de las fotos)."""
    names = {category["id"]: category["name"] for category in coco["categories"]}
    counts = Counter(names[annotation["category_id"]] for annotation in coco["annotations"])
    return {name: counts.get(name, 0) for name in CLASSES}
