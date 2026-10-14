"""Detector RF-DETR entrenado con Open Images V7: autos, ruedas, patentes y rostros en una sola sesión.

Una misma sesión ONNX sirve a dos puertos: `detector` (`VehicleDetector`: `car`, `wheel` y `plate`) y
`face` (`FaceDetector`). Los dos adaptadores comparten un núcleo que corre el modelo una vez por foto:
si el servicio llama a los dos con la misma imagen, la segunda llamada reutiliza el resultado.

Preproceso igual al de `RFDETR.predict()` (la foto completa a la resolución del modelo, normalizada con
ImageNet). El ONNX (`tools/training/detector/export.py`) devuelve `dets` (cajas cxcywh normalizadas) y
`labels` (logits por clase). Las columnas de `labels` siguen el orden de `LABELS`, que es el orden en
que `tools/training/detector/dataset.py` declara las categorías: no se puede cambiar sin reentrenar.

Mosaico opcional (`tiling.py`): con `face_tiling` o `plate_tiling` la foto grande se recorre también en
tiles y se fusionan las cajas de rostros o de patentes. Es más lento (una inferencia por tile) y solo
mejora los objetos chicos; `car` y `wheel` siempre salen de la foto completa.

Umbrales: `car` y `wheel` usan `detector_min_conf`; `plate` y `face` salen desde `redaction_min_conf`, bajo
a propósito porque el dominio decide cuáles tapar y ante la duda se tapa.
"""

import logging
import threading
import weakref
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.models.onnx_session import InferenceSessionLike
from vvi_ia.adapters.models.registry import ModelContext, register
from vvi_ia.adapters.models.tiling import (
    DEFAULT_MAX_SIDE_PX,
    DEFAULT_OVERLAP,
    DEFAULT_TILE_PX,
    Tiling,
    detect_tiled,
)
from vvi_ia.domain.model import Box, BoxLabel, Pixels

logger = logging.getLogger(__name__)

MODEL_NAME = "rfdetr_openimages"
FACE = "face"
LABELS: Mapping[int, str] = {0: BoxLabel.CAR, 1: BoxLabel.WHEEL, 2: BoxLabel.PLATE, 3: FACE}
VEHICLE_LABELS = frozenset({BoxLabel.CAR, BoxLabel.WHEEL, BoxLabel.PLATE})
OUTPUT_NAMES = ("dets", "labels")
DETECTOR_MIN_CONF_PARAM = "detector_min_conf"
REDACTION_MIN_CONF_PARAM = "redaction_min_conf"
DEFAULT_DETECTOR_MIN_CONF = 0.40
DEFAULT_REDACTION_MIN_CONF = 0.10
FACE_TILING_PARAM = "face_tiling"
PLATE_TILING_PARAM = "plate_tiling"
TILE_PX_PARAM = "tiling_tile_px"
TILE_OVERLAP_PARAM = "tiling_overlap"
TILE_MAX_SIDE_PARAM = "tiling_max_side_px"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_PIXEL = 255.0
IMAGE_NDIM = 3
RGB_CHANNELS = 3
BOXES_NDIM = 3  # (lote, consultas, 4)
BOX_VALUES = 4
NCHW_NDIM = 4
LOGIT_CLIP = 88.0  # evita el desborde de exp() en float32
WARMUP_SIDE = 64


def preprocess(img: Pixels, height: int, width: int) -> NDArray[np.float32]:
    """Tensor `(1, 3, height, width)` normalizado con ImageNet."""
    if img.ndim != IMAGE_NDIM or img.shape[2] != RGB_CHANNELS or min(img.shape[:2]) < 1:
        raise ValueError("la imagen debe ser RGB de alto y ancho positivos")
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / MAX_PIXEL - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis])


def decode_boxes(
    dets: NDArray[Any],
    labels: NDArray[Any],
    size: tuple[int, int],
    thresholds: Mapping[str, float],
    names: Mapping[int, str] = LABELS,
) -> list[Box]:
    """Cajas en píxeles de la foto original `size = (alto, ancho)`, de más a menos confiables."""
    if dets.ndim != BOXES_NDIM or dets.shape[0] != 1 or dets.shape[2] != BOX_VALUES:
        raise ValueError("forma inesperada de dets")
    if labels.ndim != BOXES_NDIM or labels.shape[:2] != dets.shape[:2] or labels.shape[2] <= max(names):
        raise ValueError("forma inesperada de labels")
    if not (np.isfinite(dets).all() and np.isfinite(labels).all()):
        raise ValueError("la salida del modelo no es finita")
    height, width = size
    scale = np.array([width, height, width, height], np.float32)
    boxes: list[Box] = []
    for column, name in names.items():
        scores = 1.0 / (
            1.0 + np.exp(-np.clip(labels[0, :, column].astype(np.float32), -LOGIT_CLIP, LOGIT_CLIP))
        )
        for query in np.nonzero(scores > thresholds[name])[0]:
            cx, cy, box_width, box_height = dets[0, query]
            corners = np.array(
                [cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2], np.float32
            )
            x1, y1, x2, y2 = np.clip(corners * scale, 0.0, scale)
            if x2 > x1 and y2 > y1:
                boxes.append(
                    Box(float(x1), float(y1), float(x2), float(y2), conf=float(scores[query]), label=name)
                )
    return sorted(boxes, key=lambda box: -box.conf)


class RfDetrDetections:
    """Núcleo compartido: corre el modelo y recuerda el último resultado de cada hilo.

    La memoria es de una sola imagen por hilo y se guarda con una referencia débil al arreglo: solo
    sirve si la segunda llamada recibe **el mismo objeto** y nadie lo modificó entre las dos.
    """

    def __init__(
        self,
        session: InferenceSessionLike,
        input_name: str,
        input_size: tuple[int, int],
        thresholds: Mapping[str, float],
        names: Mapping[int, str] = LABELS,
        tiling: Tiling | None = None,
    ) -> None:
        self._session = session
        self._input_name = input_name
        self._input_size = input_size  # (alto, ancho) del modelo
        self._thresholds = thresholds
        self._names = names
        self._tiling = tiling
        self._local = threading.local()

    def _infer(self, img: Pixels) -> list[Box]:
        tensor = preprocess(img, *self._input_size)
        dets, labels = self._session.run(OUTPUT_NAMES, {self._input_name: tensor})
        return decode_boxes(dets, labels, (img.shape[0], img.shape[1]), self._thresholds, self._names)

    def detect_all(self, img: Pixels) -> list[Box] | None:
        """Un solo pase sobre la foto entera (memoria de una imagen por hilo). `None` si el modelo falla."""
        remembered = getattr(self._local, "last", None)
        if remembered is not None and remembered[0]() is img:
            return list(remembered[1])
        try:
            boxes = self._infer(img)
        except Exception as exc:
            # Solo el tipo de error: el mensaje podría arrastrar contenido de la imagen.
            logger.warning("falló la inferencia del detector", extra={"error": type(exc).__name__})
            return None
        self._local.last = (weakref.ref(img), boxes)
        return list(boxes)

    def detect_tiled(self, img: Pixels) -> list[Box] | None:
        """El pase entero (reutilizado de `detect_all`) más los tiles de las clases con mosaico.

        Sin mosaico configurado es lo mismo que `detect_all`. Es más caro (una inferencia por tile): sirve
        para buscar patentes y rostros chicos al redactar, no para decidir la presencia.
        """
        if self._tiling is None:
            return self.detect_all(img)
        remembered = getattr(self._local, "last_tiled", None)
        if remembered is not None and remembered[0]() is img:
            return list(remembered[1])
        boxes = self.detect_all(img)
        if boxes is None:
            return None
        try:
            tiled = detect_tiled(
                self._infer,
                img,
                self._tiling.labels,
                self._tiling.tile,
                self._tiling.overlap,
                self._tiling.nms_iou,
                full=boxes,
                max_side=self._tiling.max_side,
            )
            if tiled is None:  # no pasa: `_infer` lanza ante un error, no devuelve `None`
                raise ValueError("falló la inferencia por mosaico")
        except Exception as exc:
            logger.warning("falló la inferencia por mosaico", extra={"error": type(exc).__name__})
            return None
        merged = sorted(
            [b for b in boxes if b.label not in self._tiling.labels] + tiled, key=lambda b: -b.conf
        )
        self._local.last_tiled = (weakref.ref(img), merged)
        return list(merged)


class RfDetrVehicleDetector:
    """Implementa `VehicleDetector`: `car`, `wheel` y `plate`."""

    def __init__(self, core: RfDetrDetections, version: str) -> None:
        self._core = core
        self.version = version

    def detect(self, img: Pixels) -> list[Box] | None:
        boxes = self._core.detect_all(img)
        return None if boxes is None else [box for box in boxes if box.label in VEHICLE_LABELS]

    def detect_plates(self, img: Pixels) -> list[Box] | None:
        """El pase entero más las patentes que encuentra el mosaico (solo con `PLATE_TILING`)."""
        boxes = self._core.detect_tiled(img)
        return None if boxes is None else [box for box in boxes if box.label in VEHICLE_LABELS]


class RfDetrFaceDetector:
    """Implementa `FaceDetector`."""

    def __init__(self, core: RfDetrDetections, version: str) -> None:
        self._core = core
        self.version = version

    def detect(self, img: Pixels) -> list[Box] | None:
        boxes = self._core.detect_tiled(
            img
        )  # los rostros solo se buscan al redactar: con mosaico si está activo
        return None if boxes is None else [box for box in boxes if box.label == FACE]


def _thresholds(params: Mapping[str, Any]) -> dict[str, float]:
    detector = float(params.get(DETECTOR_MIN_CONF_PARAM, DEFAULT_DETECTOR_MIN_CONF))
    redaction = float(params.get(REDACTION_MIN_CONF_PARAM, DEFAULT_REDACTION_MIN_CONF))
    if not (0.0 <= detector <= 1.0 and 0.0 <= redaction <= 1.0):
        raise ValueError("los umbrales de confianza deben estar entre 0 y 1")
    return {BoxLabel.CAR: detector, BoxLabel.WHEEL: detector, BoxLabel.PLATE: redaction, FACE: redaction}


def _tiling(params: Mapping[str, Any]) -> Tiling | None:
    labels = frozenset(
        label
        for label, param in ((FACE, FACE_TILING_PARAM), (BoxLabel.PLATE, PLATE_TILING_PARAM))
        if float(params.get(param, 0.0))
    )
    if not labels:
        return None
    return Tiling(
        labels,
        int(params.get(TILE_PX_PARAM, DEFAULT_TILE_PX)),
        float(params.get(TILE_OVERLAP_PARAM, DEFAULT_OVERLAP)),
        max_side=int(params.get(TILE_MAX_SIDE_PARAM, DEFAULT_MAX_SIDE_PX)),
    )


def _core(ctx: ModelContext) -> RfDetrDetections:
    """El núcleo del modelo: se construye una vez y lo comparten los dos puertos."""
    if "detector_core" not in ctx.model.shared:
        session = ctx.model.session
        if session.output_names != OUTPUT_NAMES or len(session.input_names) != 1:
            raise ValueError("el modelo no tiene la interfaz de RF-DETR")
        shape = session.input_shape()
        if len(shape) != NCHW_NDIM or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError("la entrada del modelo debe ser NCHW con alto y ancho fijos")
        core = RfDetrDetections(
            session,
            session.input_names[0],
            (shape[2], shape[3]),
            _thresholds(ctx.params),
            tiling=_tiling(ctx.params),
        )
        # Inferencia de prueba sobre una imagen gris: si el archivo no se comporta como el modelo
        # esperado, el servicio no arranca.
        if core.detect_all(np.full((WARMUP_SIDE, WARMUP_SIDE, RGB_CHANNELS), 128, np.uint8)) is None:
            raise ValueError("el modelo no devolvió una salida válida")
        ctx.model.shared["detector_core"] = core
    core = ctx.model.shared["detector_core"]
    assert isinstance(core, RfDetrDetections)
    return core


@register("detector", MODEL_NAME)
def build_vehicle(ctx: ModelContext) -> RfDetrVehicleDetector:
    return RfDetrVehicleDetector(_core(ctx), ctx.model.version)


@register("face", MODEL_NAME)
def build_face(ctx: ModelContext) -> RfDetrFaceDetector:
    return RfDetrFaceDetector(_core(ctx), ctx.model.version)
