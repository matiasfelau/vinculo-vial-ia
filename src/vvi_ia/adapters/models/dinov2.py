"""Embedder DINOv2 ViT-S/14 sobre ONNX Runtime: el vector con el que se comparan dos fotos de un auto.

Preproceso: recorte de la caja de la máscara con el fondo en gris, resize a 224×224 y normalización
ImageNet. De la salida se toma el token CLS y se normaliza L2. `dinov2_vits14` se usa tal como lo publicó
Meta, sin entrenar (el ONNX lo genera `tools/export/dinov2.py`). `dinov2_vits14_reid` es el mismo modelo
ajustado con 3DRealCar (`tools/training/reid`): tiene la misma entrada y la misma salida, así que usa este
mismo adaptador. Sin segmentador, la máscara es la caja `car` del detector y el recorte es esa caja.
"""

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.models.onnx_session import InferenceSessionLike
from vvi_ia.adapters.models.registry import ModelContext, register
from vvi_ia.domain.model import Mask, Pixels, Vector
from vvi_ia.domain.reid import GRAY_FILL, masked_crop

logger = logging.getLogger(__name__)

MODEL_NAME = "dinov2_vits14"
REID_MODEL_NAME = "dinov2_vits14_reid"
REID_MODEL_NAME_V2 = "dinov2_vits14_reid_v2"
INPUT_SIDE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_PIXEL = 255.0
CLS_INDEX = 0
TOKENS_NDIM = 3  # (lote, tokens, dimensión)


def preprocess(img: Pixels, mask: Mask) -> NDArray[np.float32] | None:
    """Tensor `(1, 3, 224, 224)` del auto enmascarado, o `None` si la máscara está vacía."""
    if img.ndim != 3 or mask.shape != img.shape[:2]:
        raise ValueError("la máscara no tiene el tamaño de la imagen")
    crop = masked_crop(img, mask)
    if crop is None:
        return None
    shrinks = max(crop.shape[:2]) > INPUT_SIDE
    resized = cv2.resize(
        crop, (INPUT_SIDE, INPUT_SIDE), interpolation=cv2.INTER_AREA if shrinks else cv2.INTER_CUBIC
    )
    normalized = (resized.astype(np.float32) / MAX_PIXEL - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis])


def cls_embedding(tokens: NDArray[Any]) -> Vector | None:
    """Token CLS normalizado L2, o `None` si la salida no sirve como embedding."""
    if tokens.ndim != TOKENS_NDIM or tokens.shape[0] != 1 or tokens.shape[1] < 1:
        return None
    cls = tokens[0, CLS_INDEX].astype(np.float32)
    norm = float(np.linalg.norm(cls))
    if not np.isfinite(norm) or norm == 0.0:
        return None
    unit: Vector = cls / norm
    return unit


class DinoV2Embedder:
    """Implementa `Embedder`. La sesión es segura entre hilos: corre en el pool de inferencia."""

    def __init__(self, session: InferenceSessionLike, input_name: str, version: str) -> None:
        self._session = session
        self._input_name = input_name
        self.version = version

    def embed(self, img: Pixels, mask: Mask) -> Vector | None:
        try:
            tensor = preprocess(img, mask)
            if tensor is None:
                return None
            tokens = self._session.run(None, {self._input_name: tensor})[0]
            return cls_embedding(tokens)
        except Exception as exc:
            # Solo el tipo de error: el mensaje podría arrastrar contenido de la imagen.
            logger.warning("falló la inferencia del embedder", extra={"error": type(exc).__name__})
            return None


@register("embedder", MODEL_NAME)
@register("embedder", REID_MODEL_NAME)
@register("embedder", REID_MODEL_NAME_V2)
def build(ctx: ModelContext) -> DinoV2Embedder:
    session = ctx.model.session
    if len(session.input_names) != 1:
        raise ValueError("se esperaba un modelo con una sola entrada")
    embedder = DinoV2Embedder(session, session.input_names[0], ctx.model.version)
    # Inferencia de prueba sobre una imagen gris: si el archivo no se comporta como el modelo
    # esperado, el servicio no arranca.
    blank = np.full((INPUT_SIDE, INPUT_SIDE, 3), GRAY_FILL, np.uint8)
    if embedder.embed(blank, np.ones((INPUT_SIDE, INPUT_SIDE), np.bool_)) is None:
        raise ValueError("el modelo no devolvió un embedding")
    return embedder
