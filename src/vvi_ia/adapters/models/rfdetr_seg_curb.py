"""Segmentador de cordón y rebaje: RF-DETR-Seg entrenado con el dataset propio (`curb` y `curb_cut`).

Mismo preproceso y mismo ONNX (`dets`, `labels`, `masks`) que `rfdetr_seg`. Las columnas de `labels` son
las dos clases, en el orden de `CLASSES` (el de `tools/training/curb/prepare.py`). Cada clase se devuelve como
**una sola máscara**: la unión de todas las instancias con confianza mayor a `curb_min_conf`, del tamaño de la
foto. Sin instancias, la máscara sale vacía: "miré y no encontré" no es una falla.

`curb` es el cordón visible, sin el rebaje; `curb_cut` es el tramo rebajado (garage y accesibilidad son la
misma clase), que es el que puede quedar tapado por un auto.
"""

import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.models.onnx_session import InferenceSessionLike
from vvi_ia.adapters.models.registry import ModelContext, register
from vvi_ia.adapters.models.rfdetr_seg import LOGITS_NDIM, NCHW_NDIM, OUTPUT_NAMES, preprocess, sigmoid
from vvi_ia.domain.model import Mask, Pixels

logger = logging.getLogger(__name__)

MODEL_NAME = "rfdetr_seg_curb"
CLASSES: Sequence[str] = ("curb", "curb_cut")
MIN_CONF_PARAM = "curb_min_conf"
DEFAULT_MIN_CONF = 0.50
WARMUP_SIDE = 64
RGB_CHANNELS = 3


def decode_class_masks(
    labels: NDArray[Any], masks: NDArray[Any], size: tuple[int, int], min_conf: float
) -> tuple[Mask, Mask]:
    """Una máscara por clase, de tamaño `size = (alto, ancho)`, con la unión de sus instancias."""
    if labels.ndim != LOGITS_NDIM or labels.shape[0] != 1 or labels.shape[2] < len(CLASSES):
        raise ValueError("forma inesperada de labels")
    if masks.ndim != NCHW_NDIM or masks.shape[:2] != labels.shape[:2]:
        raise ValueError("forma inesperada de masks")
    if not (np.isfinite(labels).all() and np.isfinite(masks).all()):
        raise ValueError("la salida del modelo no es finita")
    height, width = size
    result: list[Mask] = []
    for column in range(len(CLASSES)):
        union = np.zeros((height, width), np.bool_)
        scores = sigmoid(labels[0, :, column])
        for query in np.nonzero(scores > min_conf)[0]:
            logits = masks[0, query].astype(np.float32)
            union |= cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0
        result.append(union)
    return result[0], result[1]


class RfDetrSegCurbSegmenter:
    """Implementa `CurbSegmenter`. La sesión es segura entre hilos: corre en el pool de inferencia."""

    def __init__(
        self,
        session: InferenceSessionLike,
        input_name: str,
        input_size: tuple[int, int],
        min_conf: float,
        version: str,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._input_size = input_size  # (alto, ancho) del modelo
        self._min_conf = min_conf
        self.version = version

    def curb(self, img: Pixels) -> tuple[Mask, Mask] | None:
        try:
            tensor = preprocess(img, *self._input_size)
            _, labels, masks = self._session.run(OUTPUT_NAMES, {self._input_name: tensor})
            return decode_class_masks(labels, masks, (img.shape[0], img.shape[1]), self._min_conf)
        except Exception as exc:
            # Solo el tipo de error: el mensaje podría arrastrar contenido de la imagen.
            logger.warning(
                "falló la inferencia del segmentador de rebajes", extra={"error": type(exc).__name__}
            )
            return None


@register("curb", MODEL_NAME)
def build(ctx: ModelContext) -> RfDetrSegCurbSegmenter:
    session = ctx.model.session
    if session.output_names != OUTPUT_NAMES or len(session.input_names) != 1:
        raise ValueError("el modelo no tiene la interfaz de RF-DETR-Seg")
    shape = session.input_shape()
    if len(shape) != NCHW_NDIM or not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise ValueError("la entrada del modelo debe ser NCHW con alto y ancho fijos")
    min_conf = float(ctx.params.get(MIN_CONF_PARAM, DEFAULT_MIN_CONF))
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError("el umbral de confianza debe estar entre 0 y 1")
    segmenter = RfDetrSegCurbSegmenter(
        session, session.input_names[0], (shape[2], shape[3]), min_conf, ctx.model.version
    )
    # Inferencia de prueba sobre una imagen gris: si el archivo no se comporta como el modelo esperado,
    # el servicio no arranca.
    blank = np.full((WARMUP_SIDE, WARMUP_SIDE, RGB_CHANNELS), 128, np.uint8)
    if segmenter.curb(blank) is None:
        raise ValueError("el modelo no devolvió una salida válida")
    return segmenter
