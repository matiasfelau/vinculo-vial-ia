"""Segmentador de autos RF-DETR-Seg (COCO, sin entrenar) sobre ONNX Runtime.

Preproceso igual al de `RFDETR.predict()`: la foto completa se escala a la resolución del modelo
(sin conservar la proporción ni antialias) y se normaliza con ImageNet. Las tres salidas son `dets`
(cajas cxcywh normalizadas), `labels` (logits por clase, indexados por el id de COCO: `car` es 3) y
`masks` (logits de máscara por consulta, a una resolución menor). Se queda con las consultas cuya
confianza en `car` supera el umbral y escala cada máscara al tamaño de la foto. El ONNX lo genera
`tools/export/rfdetr_seg.py`.
"""

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.models.onnx_session import InferenceSessionLike
from vvi_ia.adapters.models.registry import ModelContext, register
from vvi_ia.domain.model import CarInstance, Pixels

logger = logging.getLogger(__name__)

MODEL_NAME = "rfdetr_seg_coco"
CAR_CLASS_ID = 3  # id de `car` en COCO; las columnas de `labels` se indexan por id
DEFAULT_MIN_CONF = 0.5
MIN_CONF_PARAM = "segmenter_min_conf"
OUTPUT_NAMES = ("dets", "labels", "masks")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_PIXEL = 255.0
IMAGE_NDIM = 3
RGB_CHANNELS = 3
LOGITS_NDIM = 3  # (lote, consultas, clases)
NCHW_NDIM = 4
LOGIT_CLIP = 88.0  # evita el desborde de exp() en float32
WARMUP_SIDE = 64


def preprocess(img: Pixels, height: int, width: int) -> NDArray[np.float32]:
    """Tensor `(1, 3, height, width)` normalizado con ImageNet."""
    if img.ndim != IMAGE_NDIM or img.shape[2] != RGB_CHANNELS or img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError("la imagen debe ser RGB de alto y ancho positivos")
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / MAX_PIXEL - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis])


def sigmoid(logits: NDArray[Any]) -> NDArray[np.float32]:
    clipped = np.clip(logits.astype(np.float32), -LOGIT_CLIP, LOGIT_CLIP)
    scores: NDArray[np.float32] = 1.0 / (1.0 + np.exp(-clipped))
    return scores


def decode_cars(
    labels: NDArray[Any], masks: NDArray[Any], size: tuple[int, int], min_conf: float
) -> list[CarInstance]:
    """Autos con confianza mayor a `min_conf`, de más a menos confiables, con máscara del tamaño `size`.

    `size` es `(alto, ancho)` de la foto original. Una máscara que queda vacía se descarta: no
    sirve para recortar ni para comparar.
    """
    if labels.ndim != LOGITS_NDIM or labels.shape[0] != 1 or labels.shape[2] <= CAR_CLASS_ID:
        raise ValueError("forma inesperada de labels")
    if masks.ndim != NCHW_NDIM or masks.shape[:2] != labels.shape[:2]:
        raise ValueError("forma inesperada de masks")
    if not (np.isfinite(labels).all() and np.isfinite(masks).all()):
        raise ValueError("la salida del modelo no es finita")
    height, width = size
    scores = sigmoid(labels[0, :, CAR_CLASS_ID])
    cars: list[CarInstance] = []
    for query in np.argsort(-scores, kind="stable"):
        if scores[query] <= min_conf:
            break
        logits = masks[0, query].astype(np.float32)
        mask = cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0
        if mask.any():
            cars.append(CarInstance(mask=mask, conf=float(scores[query])))
    return cars


class RfDetrSegCarSegmenter:
    """Implementa `CarSegmenter`. La sesión es segura entre hilos: corre en el pool de inferencia."""

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

    def cars(self, img: Pixels) -> list[CarInstance] | None:
        try:
            tensor = preprocess(img, *self._input_size)
            _, labels, masks = self._session.run(OUTPUT_NAMES, {self._input_name: tensor})
            return decode_cars(labels, masks, (img.shape[0], img.shape[1]), self._min_conf)
        except Exception as exc:
            # Solo el tipo de error: el mensaje podría arrastrar contenido de la imagen.
            logger.warning("falló la inferencia del segmentador", extra={"error": type(exc).__name__})
            return None


@register("segmenter", MODEL_NAME)
def build(ctx: ModelContext) -> RfDetrSegCarSegmenter:
    session = ctx.model.session
    if session.output_names != OUTPUT_NAMES or len(session.input_names) != 1:
        raise ValueError("el modelo no tiene la interfaz de RF-DETR-Seg")
    shape = session.input_shape()
    if len(shape) != NCHW_NDIM:
        raise ValueError("la entrada del modelo debe ser NCHW")
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("la entrada del modelo debe tener alto y ancho fijos")
    min_conf = float(ctx.params.get(MIN_CONF_PARAM, DEFAULT_MIN_CONF))
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError("el umbral de confianza debe estar entre 0 y 1")
    segmenter = RfDetrSegCarSegmenter(
        session, session.input_names[0], (height, width), min_conf, ctx.model.version
    )
    # Inferencia de prueba sobre una imagen gris: si el archivo no se comporta como el modelo
    # esperado, el servicio no arranca.
    if segmenter.cars(np.full((WARMUP_SIDE, WARMUP_SIDE, 3), 128, np.uint8)) is None:
        raise ValueError("el modelo no devolvió una salida válida")
    return segmenter
