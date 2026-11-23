"""Hojas de contacto para revisar a mano fotos o recortes de las evaluaciones.

Llevan fotos de terceros (autos, patentes, rostros): se escriben **fuera del repositorio** y nunca se
suben a ningún lado. Cada panel muestra una foto, o un recorte con margen, con sus cajas dibujadas y un
título corto (ASCII: OpenCV no dibuja tildes).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.imaging import CvImageDecoder

Box = tuple[float, float, float, float]
Color = tuple[int, int, int]  # RGB
RED: Color = (255, 40, 40)
GREEN: Color = (40, 200, 40)
CAPTION_PX = 34


@dataclass(frozen=True)
class Panel:
    """Una foto (o un recorte de ella) con cajas `(x1, y1, x2, y2, color)` en píxeles de la foto original."""

    image: Path
    caption: str
    crop: Box | None = None
    boxes: Sequence[tuple[Box, Color]] = field(default_factory=tuple)


def with_margin(box: Box, size: tuple[int, int], margin: float = 0.6, min_fraction: float = 0.0) -> Box:
    """`box` agrandada `margin` (de su ancho y alto) por lado, sin salirse de la foto `(alto, ancho)`.

    `min_fraction` pide además que el recorte ocupe al menos esa fracción del ancho y del alto de la foto
    (centrado en la caja, desplazado si hace falta): una caja pegada al borde no queda como una tira.
    """
    x1, y1, x2, y2 = box
    height, width = float(size[0]), float(size[1])
    half_w = max((x2 - x1) * (0.5 + margin), min_fraction * width / 2)
    half_h = max((y2 - y1) * (0.5 + margin), min_fraction * height / 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = min(max(cx - half_w, 0.0), width), min(max(cy - half_h, 0.0), height)
    right, bottom = min(cx + half_w, width), min(cy + half_h, height)
    # Si la caja está contra un borde, el lado que sobró se suma del otro lado
    if cx - half_w < 0:
        right = min(width, right + (half_w - cx))
    if cx + half_w > width:
        left = max(0.0, left - (cx + half_w - width))
    if cy - half_h < 0:
        bottom = min(height, bottom + (half_h - cy))
    if cy + half_h > height:
        top = max(0.0, top - (cy + half_h - height))
    return (left, top, right, bottom)


def render(panel: Panel, cell: int, decoder: CvImageDecoder) -> NDArray[np.uint8]:
    """El panel como una imagen RGB de `cell` x `cell` píxeles (con el título abajo)."""
    pixels = np.ascontiguousarray(decoder.decode(panel.image.read_bytes()))
    for (x1, y1, x2, y2), color in panel.boxes:
        thickness = max(2, round(max(pixels.shape[:2]) / 250))
        cv2.rectangle(pixels, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if panel.crop is not None:
        x1, y1, x2, y2 = (int(v) for v in panel.crop)
        pixels = pixels[y1 : max(y2, y1 + 1), x1 : max(x2, x1 + 1)]
    side = cell - CAPTION_PX
    scale = min(side / pixels.shape[1], side / pixels.shape[0])
    size = (max(1, round(pixels.shape[1] * scale)), max(1, round(pixels.shape[0] * scale)))
    shown = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    out = np.full((cell, cell, 3), 245, np.uint8)
    top, left = (side - size[1]) // 2, (cell - size[0]) // 2
    out[top : top + size[1], left : left + size[0]] = shown
    cv2.putText(
        out, panel.caption[:46], (4, cell - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )
    return out


def sheets(
    panels: Sequence[Panel],
    out_dir: Path,
    columns: int = 4,
    rows: int = 3,
    cell: int = 400,
    prefix: str = "hoja",
) -> list[Path]:
    """Escribe las hojas JPEG (`columns` x `rows` paneles cada una) y devuelve sus rutas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder = CvImageDecoder()
    per_sheet = columns * rows
    written: list[Path] = []
    for number in range(math.ceil(len(panels) / per_sheet)):
        chunk = panels[number * per_sheet : (number + 1) * per_sheet]
        canvas = np.full((rows * cell, columns * cell, 3), 255, np.uint8)
        for index, panel in enumerate(chunk):
            row, column = divmod(index, columns)
            canvas[row * cell : (row + 1) * cell, column * cell : (column + 1) * cell] = render(
                panel, cell, decoder
            )
        path = out_dir / f"{prefix}-{number + 1:02d}.jpg"
        cv2.imwrite(str(path), np.ascontiguousarray(canvas[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(path)
    return written
