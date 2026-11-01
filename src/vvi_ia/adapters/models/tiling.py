"""Inferencia por mosaico: correr un detector sobre recortes (tiles) de la foto además de la foto completa.

Un detector que achica la foto a un lado fijo (512 px) pierde los objetos chicos: una cara de 20 px en una
foto de 1600 px queda de 6 px. Con tiles de tamaño parecido al de la entrada, el recorte casi no se achica y
esos objetos recuperan tamaño. A cambio hay una inferencia más por tile.

Los tiles se solapan para que un objeto cortado por un borde aparezca entero en otro. Las cajas que tocan un
borde **interior** de un tile se descartan (están cortadas: el mismo objeto sale entero en el tile vecino o en
la foto completa) y el resto se fusiona con NMS, por clase, con las de la foto completa.
"""

from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.domain.model import Box, Pixels

DEFAULT_TILE_PX = 640
DEFAULT_OVERLAP = 0.25
DEFAULT_NMS_IOU = 0.5
DEFAULT_MAX_SIDE_PX = 1600  # una foto más grande se achica a esto antes de cortar los tiles
EDGE_MARGIN_PX = 2.0  # una caja a menos de esto de un borde interior del tile se considera cortada
MIN_TILE_PX = 64
MAX_OVERLAP = 0.9
ARRAY_NMS_IOU = 0.4

TileBox = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Tiling:
    """Qué clases se buscan también por mosaico y cómo se cortan los tiles."""

    labels: frozenset[str]
    tile: int = DEFAULT_TILE_PX
    overlap: float = DEFAULT_OVERLAP
    nms_iou: float = DEFAULT_NMS_IOU
    max_side: int = DEFAULT_MAX_SIDE_PX


def _starts(length: int, tile: int, overlap: float) -> list[int]:
    """Posiciones iniciales de los tiles a lo largo de un eje, de modo que el último termine en el borde."""
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1.0 - overlap)))
    count = ceil((length - tile) / stride) + 1
    step = (length - tile) / (count - 1)
    return [round(index * step) for index in range(count)]


def tile_grid(height: int, width: int, tile: int, overlap: float) -> list[TileBox]:
    """Tiles cuadrados de lado `tile` que cubren la foto con el solape pedido.

    Si la foto cabe entera en un tile no hay nada que mosaiquear y devuelve una lista vacía. Un eje más
    corto que `tile` usa todo su largo.
    """
    if tile < MIN_TILE_PX or not 0.0 <= overlap <= MAX_OVERLAP:
        raise ValueError("tile o solape fuera de rango")
    if height <= tile and width <= tile:
        return []
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in _starts(height, tile, overlap)
        for x in _starts(width, tile, overlap)
    ]


def iou(a: Box, b: Box) -> float:
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = width * height
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: Iterable[Box], threshold: float = DEFAULT_NMS_IOU) -> list[Box]:
    """NMS por clase: de mayor a menor confianza, descarta lo que se superpone con una ya elegida."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: -item.conf):
        if all(other.label != box.label or iou(box, other) <= threshold for other in kept):
            kept.append(box)
    return kept


def nms_array(dets: NDArray[Any], threshold: float = ARRAY_NMS_IOU) -> list[int]:
    """NMS de SCRFD y RetinaFace (con el `+ 1` de los píxeles) sobre filas `x1, y1, x2, y2, score`."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        width = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        height = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        overlap = width * height / (areas[i] + areas[order[1:]] - width * height)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def _is_cut(box: Box, tile: TileBox, height: int, width: int) -> bool:
    """¿Toca un borde del tile que no es el borde de la foto?"""
    x1, y1, x2, y2 = tile
    return (
        (x1 > 0 and box.x1 <= EDGE_MARGIN_PX)
        or (y1 > 0 and box.y1 <= EDGE_MARGIN_PX)
        or (x2 < width and box.x2 >= x2 - x1 - EDGE_MARGIN_PX)
        or (y2 < height and box.y2 >= y2 - y1 - EDGE_MARGIN_PX)
    )


def detect_tiled(
    detect: Callable[[Pixels], Sequence[Box] | None],
    img: Pixels,
    labels: Collection[str],
    tile: int = DEFAULT_TILE_PX,
    overlap: float = DEFAULT_OVERLAP,
    nms_iou: float = DEFAULT_NMS_IOU,
    full: Sequence[Box] | None = None,
    max_side: int | None = None,
) -> list[Box] | None:
    """Detecciones de `labels` sobre la foto completa y sobre cada tile, fusionadas.

    `full` son las cajas de la foto completa si el llamador ya las tiene (no se vuelve a inferir). Con
    `max_side`, una foto más grande se achica a ese lado mayor antes de cortar los tiles (acota la cantidad
    de inferencias en fotos de celular de 12 megapíxeles); las cajas vuelven a la escala original. Devuelve
    `None` si falla la inferencia de la foto completa o de algún tile: con rostros y patentes no se admite
    una cobertura a medias, y el llamador decide qué hacer.
    """
    height, width = img.shape[:2]
    if full is None:
        first = detect(img)
        if first is None:
            return None
        full = first
    merged = [box for box in full if box.label in labels]
    work, factor = img, 1.0
    if max_side is not None and max(height, width) > max_side:
        factor = max_side / max(height, width)
        work = np.asarray(
            cv2.resize(img, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA),
            np.uint8,
        )
    work_height, work_width = work.shape[:2]
    for x1, y1, x2, y2 in tile_grid(work_height, work_width, tile, overlap):
        found = detect(work[y1:y2, x1:x2])
        if found is None:
            return None
        for box in found:
            if box.label not in labels or _is_cut(box, (x1, y1, x2, y2), work_height, work_width):
                continue
            merged.append(
                Box(
                    min(float(width), (box.x1 + x1) / factor),
                    min(float(height), (box.y1 + y1) / factor),
                    min(float(width), (box.x2 + x1) / factor),
                    min(float(height), (box.y2 + y1) / factor),
                    conf=box.conf,
                    label=box.label,
                )
            )
    return nms(merged, nms_iou)
