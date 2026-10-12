"""Detector de daños RF-DETR entrenado con CarDD: cajas de abolladuras, rayones, roturas y más.

Mismo preproceso y misma decodificación que el detector de vehículos (`rfdetr_detector`): la foto completa
a la resolución del modelo, normalizada con ImageNet, y un ONNX con `dets` y `labels`. Las columnas de
`labels` siguen el orden de las categorías de CarDD (`DAMAGE_LABELS`), que `tools/training/damage/prepare.py`
conserva al armar el dataset: no se puede cambiar sin reentrenar.

El puerto solo pide "cajas de daño, con confianza"; la clase de cada caja va en `Box.label`. Se devuelven las
cajas con confianza mayor a `damage_min_conf`, el mismo umbral con el que el dominio decide si hay daño.

Los pesos se entrenaron con CarDD (PIC Lab, CAS): uso de investigación con consentimiento; el uso comercial
requiere autorización del PIC Lab y los pesos no se publican ni se redistribuyen.
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from vvi_ia.adapters.models.onnx_session import InferenceSessionLike
from vvi_ia.adapters.models.registry import ModelContext, register
from vvi_ia.adapters.models.rfdetr_detector import OUTPUT_NAMES, decode_boxes, preprocess
from vvi_ia.domain.model import Box, Pixels

logger = logging.getLogger(__name__)

MODEL_NAME = "rfdetr_cardd"
DAMAGE_LABELS: Mapping[int, str] = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass_shatter",
    4: "lamp_broken",
    5: "tire_flat",
}
DAMAGE_MIN_CONF_PARAM = "damage_min_conf"
DEFAULT_DAMAGE_MIN_CONF = 0.40
IMAGE_CHANNELS = 3
NCHW_NDIM = 4
WARMUP_SIDE = 64


class RfDetrDamageDetector:
    """Implementa `DamageDetector`. La sesión es segura entre hilos: corre en el pool de inferencia."""

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
        self._thresholds = dict.fromkeys(DAMAGE_LABELS.values(), min_conf)
        self.version = version

    def detect(self, img: Pixels) -> list[Box] | None:
        try:
            tensor = preprocess(img, *self._input_size)
            dets, labels = self._session.run(OUTPUT_NAMES, {self._input_name: tensor})
            return decode_boxes(dets, labels, (img.shape[0], img.shape[1]), self._thresholds, DAMAGE_LABELS)
        except Exception as exc:
            # Solo el tipo de error: el mensaje podría arrastrar contenido de la imagen.
            logger.warning("falló la inferencia del detector de daños", extra={"error": type(exc).__name__})
            return None


@register("damage", MODEL_NAME)
def build(ctx: ModelContext) -> RfDetrDamageDetector:
    session = ctx.model.session
    if session.output_names != OUTPUT_NAMES or len(session.input_names) != 1:
        raise ValueError("el modelo no tiene la interfaz de RF-DETR")
    shape = session.input_shape()
    if len(shape) != NCHW_NDIM or not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise ValueError("la entrada del modelo debe ser NCHW con alto y ancho fijos")
    min_conf = float(ctx.params.get(DAMAGE_MIN_CONF_PARAM, DEFAULT_DAMAGE_MIN_CONF))
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError("el umbral de confianza debe estar entre 0 y 1")
    detector = RfDetrDamageDetector(
        session, session.input_names[0], (shape[2], shape[3]), min_conf, ctx.model.version
    )
    # Inferencia de prueba sobre una imagen gris: si el archivo no se comporta como el modelo esperado,
    # el servicio no arranca.
    blank: Any = np.full((WARMUP_SIDE, WARMUP_SIDE, IMAGE_CHANNELS), 128, np.uint8)
    if detector.detect(blank) is None:
        raise ValueError("el modelo no devolvió una salida válida")
    return detector
