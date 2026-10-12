"""Detector de rostros SCRFD (InsightFace) para el puerto `face`.

Es el ONNX de `det_10g` de InsightFace (SCRFD-10G con puntos de la cara, 9 salidas). Como en
`insightface/model_zoo/scrfd.py`: la foto se achica para que entre en un lienzo cuadrado de `INPUT_SIZE`
píxeles, pegada arriba a la izquierda y con relleno en cero, y se normaliza con `(x - 127,5) / 128` en orden
RGB. Por cada stride (8, 16 y 32) el modelo devuelve las confianzas, las distancias de cada ancla a los
cuatro bordes de la cara y los puntos; hay dos anclas por posición. Las cajas se decodifican, se pasan a la
escala de la foto y se les aplica NMS.

Los pesos se entrenaron con WIDER FACE (CC BY-NC-ND) y InsightFace los publica para investigación: el uso
comercial requiere su licencia (`commercial = "escalable"` en el manifest).

Mosaico opcional (`tiling.py`) con `face_tiling`: la foto grande se recorre también en tiles, lo que ayuda a
las caras chicas a cambio de una inferencia más por tile.

La confianza de SCRFD no es comparable con la de RF-DETR: con el mismo umbral deja pasar más o menos cajas.
`redaction_min_conf` es el piso con el que salen las cajas; el dominio decide cuáles tapar.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.models.onnx_session import InferenceSessionLike
from vvi_ia.adapters.models.registry import ModelContext, register
from vvi_ia.adapters.models.rfdetr_detector import (
    DEFAULT_REDACTION_MIN_CONF,
    FACE_TILING_PARAM,
    REDACTION_MIN_CONF_PARAM,
    TILE_MAX_SIDE_PARAM,
    TILE_OVERLAP_PARAM,
    TILE_PX_PARAM,
)
from vvi_ia.adapters.models.tiling import (
    DEFAULT_MAX_SIDE_PX,
    DEFAULT_OVERLAP,
    DEFAULT_TILE_PX,
    Tiling,
    detect_tiled,
    nms_array,
)
from vvi_ia.domain.model import Box, Pixels

logger = logging.getLogger(__name__)

MODEL_NAME = "scrfd_10g"
FACE = "face"
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2
OUTPUT_COUNTS = (6, 9)  # sin y con puntos de la cara
BOX_VALUES = 4
IMAGE_NDIM = 3
RGB_CHANNELS = 3
PIXEL_MEAN = 127.5
PIXEL_SCALE = 128.0
WARMUP_SIDE = 64


def letterbox(img: Pixels, size: int) -> tuple[NDArray[np.float32], float]:
    """La foto achicada para que entre en `size × size`, pegada arriba a la izquierda; y la escala usada."""
    height, width = img.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(img, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.zeros((size, size, RGB_CHANNELS), np.float32)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, scale


def preprocess(img: Pixels, size: int = INPUT_SIZE) -> tuple[NDArray[np.float32], float]:
    """Tensor `(1, 3, size, size)` normalizado y la escala de la foto al lienzo."""
    if img.ndim != IMAGE_NDIM or img.shape[2] != RGB_CHANNELS or min(img.shape[:2]) < 1:
        raise ValueError("la imagen debe ser RGB de alto y ancho positivos")
    canvas, scale = letterbox(img, size)
    blob = ((canvas - PIXEL_MEAN) / PIXEL_SCALE).transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(blob.astype(np.float32)), scale


def decode(
    outputs: Sequence[NDArray[Any]], scale: float, size: int = INPUT_SIZE, threshold: float = 0.0
) -> NDArray[np.float32]:
    """Filas `x1, y1, x2, y2, confianza` en píxeles de la foto, de más a menos confiables, ya con NMS."""
    count = len(STRIDES)
    if len(outputs) not in OUTPUT_COUNTS:
        raise ValueError("forma inesperada de las salidas")
    scores_list, boxes_list = [], []
    for index, stride in enumerate(STRIDES):
        scores = np.asarray(outputs[index], np.float32).reshape(-1)
        distances = np.asarray(outputs[index + count], np.float32).reshape(-1, BOX_VALUES) * stride
        cells = size // stride
        if scores.size != cells * cells * ANCHORS_PER_CELL or distances.shape[0] != scores.size:
            raise ValueError("forma inesperada de las salidas")
        if not (np.isfinite(scores).all() and np.isfinite(distances).all()):
            raise ValueError("la salida del modelo no es finita")
        columns, rows = np.meshgrid(np.arange(cells), np.arange(cells))  # columns[i, j] = j, rows[i, j] = i
        grid = np.stack([columns, rows], axis=-1).astype(np.float32)
        centers = np.stack([(grid * stride).reshape(-1, 2)] * ANCHORS_PER_CELL, axis=1).reshape(-1, 2)
        keep = np.where(scores >= threshold)[0]
        boxes = np.stack(
            [
                centers[:, 0] - distances[:, 0],
                centers[:, 1] - distances[:, 1],
                centers[:, 0] + distances[:, 2],
                centers[:, 1] + distances[:, 3],
            ],
            axis=-1,
        )
        scores_list.append(scores[keep])
        boxes_list.append(boxes[keep])
    if sum(len(s) for s in scores_list) == 0:
        return np.empty((0, 5), np.float32)
    dets = np.hstack([np.vstack(boxes_list) / scale, np.concatenate(scores_list)[:, np.newaxis]])
    dets = dets.astype(np.float32)
    dets = dets[np.argsort(-dets[:, 4], kind="stable")]
    return dets[nms_array(dets)]


def to_boxes(dets: NDArray[np.float32], height: int, width: int) -> list[Box]:
    """Cajas recortadas a la foto; se descartan las que quedan vacías."""
    boxes = []
    for x1, y1, x2, y2, score in dets:
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
        if x2 > x1 and y2 > y1:
            boxes.append(Box(float(x1), float(y1), float(x2), float(y2), conf=float(score), label=FACE))
    return sorted(boxes, key=lambda box: -box.conf)


class ScrfdFaceDetector:
    """Implementa `FaceDetector`."""

    def __init__(
        self,
        session: InferenceSessionLike,
        input_name: str,
        min_conf: float,
        version: str,
        tiling: Tiling | None = None,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._min_conf = min_conf
        self._tiling = tiling
        self.version = version

    def _infer(self, img: Pixels) -> list[Box]:
        tensor, scale = preprocess(img)
        outputs = self._session.run(None, {self._input_name: tensor})
        return to_boxes(decode(outputs, scale, INPUT_SIZE, self._min_conf), img.shape[0], img.shape[1])

    def detect(self, img: Pixels) -> list[Box] | None:
        try:
            boxes = self._infer(img)
            if self._tiling is not None:
                tiled = detect_tiled(
                    self._infer,
                    img,
                    {FACE},
                    self._tiling.tile,
                    self._tiling.overlap,
                    self._tiling.nms_iou,
                    full=boxes,
                    max_side=self._tiling.max_side,
                )
                if tiled is None:  # no pasa: `_infer` lanza ante un error, no devuelve `None`
                    raise ValueError("falló la inferencia por mosaico")
                boxes = sorted(tiled, key=lambda box: -box.conf)
            return boxes
        except Exception as exc:
            # Solo el tipo de error: el mensaje podría arrastrar contenido de la imagen.
            logger.warning("falló la inferencia del detector de rostros", extra={"error": type(exc).__name__})
            return None


def _tiling(params: Mapping[str, Any]) -> Tiling | None:
    if not float(params.get(FACE_TILING_PARAM, 0.0)):
        return None
    return Tiling(
        frozenset({FACE}),
        int(params.get(TILE_PX_PARAM, DEFAULT_TILE_PX)),
        float(params.get(TILE_OVERLAP_PARAM, DEFAULT_OVERLAP)),
        max_side=int(params.get(TILE_MAX_SIDE_PARAM, DEFAULT_MAX_SIDE_PX)),
    )


@register("face", MODEL_NAME)
def build(ctx: ModelContext) -> ScrfdFaceDetector:
    session = ctx.model.session
    if len(session.input_names) != 1 or len(session.output_names) not in OUTPUT_COUNTS:
        raise ValueError("el modelo no tiene la interfaz de SCRFD (6 o 9 salidas)")
    min_conf = float(ctx.params.get(REDACTION_MIN_CONF_PARAM, DEFAULT_REDACTION_MIN_CONF))
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError("el umbral de confianza debe estar entre 0 y 1")
    detector = ScrfdFaceDetector(
        session, session.input_names[0], min_conf, ctx.model.version, tiling=_tiling(ctx.params)
    )
    # Inferencia de prueba sobre una imagen gris: si el archivo no se comporta como el modelo esperado,
    # el servicio no arranca.
    blank: Any = np.full((WARMUP_SIDE, WARMUP_SIDE, RGB_CHANNELS), 128, np.uint8)
    if detector.detect(blank) is None:
        raise ValueError("el modelo no devolvió una salida válida")
    return detector
