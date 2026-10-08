"""Imágenes: decodificación segura, redimensionado, relleno opaco y codificación JPEG sin EXIF.

Las fotos se validan por su contenido real (firma de bytes), no por extensión ni `Content-Type`,
y el tamaño se lee del encabezado JPEG antes de decodificar para no reservar memoria de más.
Los píxeles se manejan como RGB.
"""

import math
from collections.abc import Sequence

import cv2
import numpy as np

from vvi_ia.domain.model import Box, Pixels
from vvi_ia.service.ports import ImageDecodeError

MAX_DECODE_PIXELS = 40_000_000
JPEG_QUALITY = 90
OPAQUE_FILL = 0

_JPEG_SIGNATURE = b"\xff\xd8\xff"
_MARKER_PREFIX = 0xFF
_SOI_LENGTH = 2
_SOF_HEADER_MIN = 9
_NO_LENGTH_MARKERS = frozenset({0x01, *range(0xD0, 0xDA)})  # TEM, RSTn, SOI, EOI
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}  # sin DHT, JPG ni DAC
_SOS_MARKER = 0xDA
_BYTE_BITS = 8


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """(ancho, alto) leídos del encabezado SOF. Lanza `ImageDecodeError` si no es un JPEG válido."""
    if not data.startswith(_JPEG_SIGNATURE):
        raise ImageDecodeError("no es un JPEG")
    pos = _SOI_LENGTH
    while pos + 1 < len(data):
        if data[pos] != _MARKER_PREFIX:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == _MARKER_PREFIX:  # relleno
            pos += 1
            continue
        pos += _SOI_LENGTH
        if marker in _NO_LENGTH_MARKERS or marker == 0x00:
            continue
        if pos + _SOI_LENGTH > len(data):
            break
        length = (data[pos] << _BYTE_BITS) | data[pos + 1]
        if marker in _SOF_MARKERS:
            if length < _SOF_HEADER_MIN or pos + _SOF_HEADER_MIN > len(data):
                break
            height = (data[pos + 3] << _BYTE_BITS) | data[pos + 4]
            width = (data[pos + 5] << _BYTE_BITS) | data[pos + 6]
            return width, height
        if marker == _SOS_MARKER:
            break
        pos += length
    raise ImageDecodeError("JPEG sin encabezado de tamaño")


class CvImageDecoder:
    """Implementa `ImageDecoder` con OpenCV y un tope de píxeles."""

    def __init__(self, max_pixels: int = MAX_DECODE_PIXELS) -> None:
        self._max_pixels = max_pixels

    def decode(self, data: bytes) -> Pixels:
        """Decodifica a RGB (aplicando la orientación EXIF). Lanza `ImageDecodeError`."""
        width, height = jpeg_dimensions(data)
        if width <= 0 or height <= 0 or width * height > self._max_pixels:
            raise ImageDecodeError("dimensiones fuera de rango")
        try:
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            raise ImageDecodeError("JPEG corrupto") from None
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ImageDecodeError("JPEG corrupto")
        rgb: Pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb


def resize_max_side(pixels: Pixels, max_side: int) -> Pixels:
    """Reduce la imagen para que su lado mayor sea ≤ `max_side`; nunca la agranda."""
    height, width = pixels.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return pixels
    scale = max_side / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized: Pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    return resized


def fill_boxes(pixels: Pixels, boxes: Sequence[Box], value: int = OPAQUE_FILL) -> Pixels:
    """Copia de la imagen con las cajas cubiertas por un relleno opaco (nunca blur)."""
    height, width = pixels.shape[:2]
    out = pixels.copy()
    for box in boxes:
        x1, y1 = max(0, math.floor(box.x1)), max(0, math.floor(box.y1))
        x2, y2 = min(width, math.ceil(box.x2)), min(height, math.ceil(box.y2))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = value
    return out


def encode_jpeg(pixels: Pixels, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG sin metadatos EXIF (el codificador de OpenCV no los escribe)."""
    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise ImageDecodeError("no se pudo codificar el JPEG")
    return bytes(buffer.tobytes())


class JpegRenderer:
    """Implementa `RedactionRenderer`: relleno sobre la imagen original, luego resize y JPEG sin EXIF.

    El relleno va primero porque las cajas están en coordenadas de la foto original.
    """

    def render(self, pixels: Pixels, boxes: Sequence[Box], max_side: int) -> bytes:
        return encode_jpeg(resize_max_side(fill_boxes(pixels, boxes), max_side))
