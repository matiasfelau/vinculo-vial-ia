"""Compara detectores de rostros: RF-DETR (el elegido), SCRFD, RetinaFace-R50 y YuNet.

Mide, sobre fotos propias con las caras anotadas, el recall con IoU 0,5 o más, el recall en caras chicas, los
falsos positivos por foto y la latencia en CPU. Para tapar rostros manda el **recall**: el servicio trabaja en
modo *fail-closed* y un rostro que no se detecta queda a la vista. El criterio de `docs/modelos.md` es que se
queda RF-DETR si su recall queda cerca del de RetinaFace-R50.

Licencias (la columna `commercial` del manifest):

- RF-DETR: `yes`. Es el que entra al servicio.
- SCRFD: `escalable`. Solo puede entrar al servicio si gana la comparación **y** se confirma con InsightFace
  que puede usarse comercialmente.
- RetinaFace-R50 y YuNet: `no` (se entrenaron con WIDER FACE, CC BY-NC-ND). Son solo referencia en `tools/`.

Los detectores que no son RF-DETR se cargan desde un ONNX que vos tenés: no se bajan ni se redistribuyen. La
decodificación de SCRFD y RetinaFace sigue el código de sus repositorios y está probada con tensores
sintéticos; todavía no se corrió con los pesos reales. Fotos con rostros de terceros: todo queda fuera del
repo y la consola muestra solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models import scrfd_face
from vvi_ia.adapters.models.onnx_session import OnnxSession
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.adapters.models.tiling import DEFAULT_OVERLAP, DEFAULT_TILE_PX, detect_tiled
from vvi_ia.domain.model import Box, Pixels

REPO_ROOT = Path(__file__).resolve().parents[2]
FACE_FOLDER = "face"
ANNOTATIONS = "_annotations.coco.json"
IOU = 0.5
SMALL_SIDE_PX = 32  # una cara con algún lado menor a esto, en la foto original, es chica
MEDIUM_SIDE_PX = 96  # chica: menos de 32 px; mediana: de 32 a 96; grande: más de 96 (por su lado menor)
SIZES = ("chicas", "medianas", "grandes")
TILED_SUFFIX = "+mosaico"
FLOOR = 0.02  # confianza mínima con la que se recogen las detecciones: de ahí se barren los umbrales
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
FP_TARGETS = (0.1, 0.5, 1.0)  # falsos positivos por foto con los que se compara el recall
TOLERANCE = 0.03  # RF-DETR "queda cerca" si pierde menos que esto de recall contra la referencia
NMS_THRESHOLD = 0.4

# commercial: la columna del manifest. `in_service`: si puede entrar al servicio si gana la comparación.
CANDIDATE_INFO: Mapping[str, Mapping[str, Any]] = {
    "rfdetr": {
        "nombre": "RF-DETR (Human face, Open Images)",
        "commercial": "yes",
        "puede_entrar_al_servicio": True,
    },
    "scrfd": {
        "nombre": "SCRFD",
        "commercial": "escalable",
        "puede_entrar_al_servicio": "solo si gana y InsightFace confirma el uso comercial",
    },
    "servicio": {
        "nombre": "Adaptador del servicio (puerto face)",
        "commercial": "según el manifest",
        "puede_entrar_al_servicio": True,
    },
    "retinaface_r50": {
        "nombre": "RetinaFace-R50 (biubug6)",
        "commercial": "no",
        "puede_entrar_al_servicio": False,
    },
    "yunet": {"nombre": "YuNet", "commercial": "no", "puede_entrar_al_servicio": False},
}
REFERENCE = "retinaface_r50"


class FaceDetector(Protocol):
    """Un candidato: devuelve las cajas de rostro de una foto RGB, con su confianza."""

    def detect(self, img: Pixels) -> list[Box] | None: ...


# --- utilidades de cajas -------------------------------------------------------------------------------


def nms(dets: NDArray[np.float32], threshold: float = NMS_THRESHOLD) -> list[int]:
    """NMS de los repositorios originales (con el `+ 1` de los píxeles) sobre `x1, y1, x2, y2, score`."""
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


def letterbox(img: Pixels, size: int, fill: Sequence[float] = (0, 0, 0)) -> tuple[NDArray[Any], float]:
    """La foto achicada para que entre en `size × size`, pegada arriba a la izquierda; y la escala usada."""
    height, width = img.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(img, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.empty((size, size, 3), np.float32)
    canvas[:] = np.array(fill, np.float32)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, scale


def to_boxes(dets: NDArray[np.float32], width: int, height: int) -> list[Box]:
    boxes = []
    for x1, y1, x2, y2, score in dets:
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
        if x2 > x1 and y2 > y1:
            boxes.append(Box(float(x1), float(y1), float(x2), float(y2), conf=float(score), label="face"))
    return sorted(boxes, key=lambda box: -box.conf)


# --- candidatos ------------------------------------------------------------------------------------------


class YuNet:
    """YuNet con `cv2.FaceDetectorYN`. Cada fila de `faces`: `x, y, ancho, alto`, 10 puntos y la confianza."""

    def __init__(self, model: Path) -> None:
        self._detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), FLOOR, NMS_THRESHOLD, 5000)

    def detect(self, img: Pixels) -> list[Box] | None:
        height, width = img.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        dets = np.array(
            [[x, y, x + w, y + h, row[14]] for row in faces for x, y, w, h in [row[:4]]], np.float32
        )
        return to_boxes(dets, width, height)


class Scrfd:
    """SCRFD (InsightFace) sobre onnxruntime, con la misma decodificación que el adaptador del servicio.

    Ver `adapters/models/scrfd_face.py`: la foto va achicada arriba a la izquierda de un lienzo cuadrado,
    normalizada con `(x - 127,5) / 128` en orden RGB; los modelos de 6 o 9 salidas traen, por cada stride
    (8, 16 y 32), las confianzas, las distancias a los cuatro bordes y (los de 9) los puntos, con
    dos anclas por posición.
    """

    def __init__(self, model: Path, size: int = 640) -> None:
        self._session = OnnxSession(model.read_bytes(), 0)
        self._size = size
        outputs = len(self._session.output_names)
        if outputs not in (6, 9):
            raise ValueError(f"se esperaba un SCRFD de 6 o 9 salidas, el modelo tiene {outputs}")

    def decode(
        self, outputs: Sequence[NDArray[Any]], scale: float, threshold: float = FLOOR
    ) -> NDArray[np.float32]:
        return scrfd_face.decode(outputs, scale, self._size, threshold)

    def detect(self, img: Pixels) -> list[Box] | None:
        tensor, scale = scrfd_face.preprocess(img, self._size)
        outputs = self._session.run(None, {self._session.input_names[0]: tensor})
        return to_boxes(self.decode(outputs, scale), img.shape[1], img.shape[0])


class RetinaFace:
    """RetinaFace-R50 (biubug6) exportado a ONNX con `convert_to_onnx.py`, de entrada cuadrada fija.

    Como en `detect.py` del repositorio: BGR menos `(104, 117, 123)`; las cajas se decodifican contra anclas
    de tamaños `[16, 32]`, `[64, 128]` y `[256, 512]` en los strides 8, 16 y 32, con varianzas `0,1` y `0,2`;
    la confianza es la segunda columna de `conf`. Las salidas se reconocen por su último eje (4, 2 y 10).
    """

    STEPS = (8, 16, 32)
    MIN_SIZES = ((16, 32), (64, 128), (256, 512))
    VARIANCES = (0.1, 0.2)
    MEAN_BGR = (104, 117, 123)

    def __init__(self, model: Path, size: int = 640) -> None:
        self._session = OnnxSession(model.read_bytes(), 0)
        self._size = size
        self._priors = self.priors(size)

    @classmethod
    def priors(cls, size: int) -> NDArray[np.float32]:
        """Las anclas `[cx, cy, ancho, alto]` normalizadas, en el orden de `PriorBox` del repositorio."""
        anchors = []
        for step, min_sizes in zip(cls.STEPS, cls.MIN_SIZES, strict=True):
            cells = ceil(size / step)
            for row in range(cells):
                for column in range(cells):
                    for min_size in min_sizes:
                        anchors.append(
                            [
                                (column + 0.5) * step / size,
                                (row + 0.5) * step / size,
                                min_size / size,
                                min_size / size,
                            ]
                        )
        return np.array(anchors, np.float32)

    def decode(
        self, loc: NDArray[Any], conf: NDArray[Any], scale: float, threshold: float = FLOOR
    ) -> NDArray[np.float32]:
        priors = self._priors
        center = priors[:, :2] + loc[:, :2] * self.VARIANCES[0] * priors[:, 2:]
        size = priors[:, 2:] * np.exp(loc[:, 2:] * self.VARIANCES[1])
        boxes = np.hstack([center - size / 2, center + size / 2]) * self._size / scale
        scores = conf[:, 1]
        keep = np.where(scores > threshold)[0]
        dets = np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32)
        dets = dets[np.argsort(-dets[:, 4], kind="stable")]
        return dets[nms(dets)] if len(dets) else dets

    def detect(self, img: Pixels) -> list[Box] | None:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.float32) - np.array(self.MEAN_BGR, np.float32)
        canvas, scale = letterbox(bgr, self._size)  # el relleno en cero equivale a la media
        outputs = self._session.run(
            None, {self._session.input_names[0]: np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])}
        )
        by_axis = {int(o.shape[-1]): np.asarray(o)[0] for o in outputs}
        if not {4, 2} <= set(by_axis):
            raise ValueError("el modelo no tiene las salidas de RetinaFace (cajas de 4 y confianzas de 2)")
        return to_boxes(self.decode(by_axis[4], by_axis[2], scale), img.shape[1], img.shape[0])


class Tiled:
    """Un detector de rostros corrido también por mosaico: la foto completa más tiles, fusionados con NMS."""

    def __init__(
        self, inner: FaceDetector, tile: int = DEFAULT_TILE_PX, overlap: float = DEFAULT_OVERLAP
    ) -> None:
        self._inner = inner
        self._tile = tile
        self._overlap = overlap

    def detect(self, img: Pixels) -> list[Box] | None:
        boxes = detect_tiled(self._inner.detect, img, {"face"}, self._tile, self._overlap)
        return None if boxes is None else sorted(boxes, key=lambda box: -box.conf)


class AdapterFace:
    """RF-DETR a través del adaptador real del servicio (puerto `face`), con el umbral casi en cero."""

    def __init__(self, models: Any) -> None:
        self._models = models

    def detect(self, img: Pixels) -> list[Box] | None:
        faces = self._models.face.detect(img)
        return (
            None if faces is None else [Box(f.x1, f.y1, f.x2, f.y2, conf=f.conf, label="face") for f in faces]
        )


# --- métricas --------------------------------------------------------------------------------------------


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(truths: Sequence[Sequence[float]], boxes: Sequence[Box]) -> tuple[list[bool], set[int]]:
    """Empareja cada caja (de mayor a menor confianza) con una cara real que nadie más usó."""
    used: set[int] = set()
    hits: list[bool] = []
    for box in sorted(boxes, key=lambda item: -item.conf):
        best, best_iou = -1, IOU
        for index, truth in enumerate(truths):
            overlap = iou((box.x1, box.y1, box.x2, box.y2), truth)
            if index not in used and overlap >= best_iou:
                best, best_iou = index, overlap
        if best >= 0:
            used.add(best)
        hits.append(best >= 0)
    return hits, used


def rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def size_of(truth: Sequence[float]) -> str:
    """`chicas`, `medianas` o `grandes` según el lado menor de la cara en la foto original."""
    side = min(truth[2] - truth[0], truth[3] - truth[1])
    return SIZES[0] if side < SMALL_SIDE_PX else SIZES[1] if side <= MEDIUM_SIDE_PX else SIZES[2]


def is_small(truth: Sequence[float], small_px: float = SMALL_SIDE_PX) -> bool:
    return min(truth[2] - truth[0], truth[3] - truth[1]) < small_px


def average_precision(
    truths: Mapping[Any, Sequence[Sequence[float]]], predictions: Mapping[Any, Sequence[Box]]
) -> float | None:
    """AP con IoU 0,5 (área bajo la curva de precisión y recall, con la precisión monótona)."""
    total = sum(len(v) for v in truths.values())
    if total == 0:
        return None
    ranked = []
    for image, boxes in predictions.items():
        hits, _ = match(truths.get(image, []), boxes)
        ranked.extend(zip((b.conf for b in sorted(boxes, key=lambda item: -item.conf)), hits, strict=True))
    ranked.sort(key=lambda item: -item[0])
    hits = np.array([hit for _, hit in ranked], float)
    if hits.size == 0:
        return 0.0
    true_positives = np.cumsum(hits)
    precision = np.maximum.accumulate((true_positives / np.arange(1, hits.size + 1))[::-1])[::-1]
    recall = true_positives / total
    return round(float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision)), 4)


def operating_points(
    truths: Mapping[Any, Sequence[Sequence[float]]],
    predictions: Mapping[Any, Sequence[Box]],
    small_px: float = SMALL_SIDE_PX,
) -> list[dict[str, Any]]:
    """Recall, recall en chicas, precisión y falsos positivos por foto con cada umbral."""
    images = max(len(truths), 1)
    total = sum(len(v) for v in truths.values())
    small_total = sum(is_small(t, small_px) for v in truths.values() for t in v)
    size_totals = Counter(size_of(t) for v in truths.values() for t in v)
    rows = []
    for threshold in THRESHOLDS:
        found = small_found = positives = false_positives = 0
        size_found: Counter[str] = Counter()
        for image, faces in truths.items():
            kept = [b for b in predictions.get(image, []) if b.conf >= threshold]
            hits, used = match(faces, kept)
            positives += sum(hits)
            false_positives += len(hits) - sum(hits)
            found += len(used)
            small_found += sum(is_small(faces[i], small_px) for i in used)
            size_found.update(size_of(faces[i]) for i in used)
        rows.append(
            {
                "umbral": threshold,
                "recall": rate(found, total),
                "recall_chicas": rate(small_found, small_total),
                "recall_por_tamano": {size: rate(size_found[size], size_totals[size]) for size in SIZES},
                "precision": rate(positives, positives + false_positives),
                "fp_por_foto": round(false_positives / images, 4),
            }
        )
    return rows


def recall_at(rows: Sequence[Mapping[str, Any]], max_fp: float) -> dict[str, Any] | None:
    """El punto de mayor recall que no pasa de `max_fp` falsos positivos por foto."""
    valid = [r for r in rows if r["fp_por_foto"] <= max_fp and r["recall"] is not None]
    return max(valid, key=lambda r: (r["recall"], -r["umbral"]), default=None)


def verdict(
    report: Mapping[str, Mapping[str, Any]], target: float = FP_TARGETS[0], tolerance: float = TOLERANCE
) -> dict[str, Any]:
    """Aplica el criterio de `docs/modelos.md` con el recall a `target` falsos positivos por foto."""
    recalls = {}
    for name, data in report.items():
        point = recall_at(data["puntos"], target)
        recalls[name] = point["recall"] if point else None
    result: dict[str, Any] = {"falsos_positivos_por_foto": target, "recall": recalls}
    rfdetr, reference = recalls.get("rfdetr"), recalls.get(REFERENCE)
    if rfdetr is None or reference is None:
        result["rfdetr_queda"] = None
        result["motivo"] = "hacen falta RF-DETR y RetinaFace-R50 para aplicar el criterio"
    else:
        result["rfdetr_queda"] = rfdetr >= reference - tolerance
        result["diferencia_con_la_referencia"] = round(rfdetr - reference, 4)
    scrfd = recalls.get("scrfd")
    if scrfd is not None and rfdetr is not None and scrfd - rfdetr > tolerance:
        result["scrfd_gana"] = True
        result["nota_scrfd"] = (
            "SCRFD es escalable: solo puede entrar si InsightFace confirma el uso comercial"
        )
    return result


def evaluate(
    truths: Mapping[Any, Sequence[Sequence[float]]], candidates: Mapping[str, Mapping[Any, Sequence[Box]]]
) -> dict[str, Any]:
    report = {}
    for name, predictions in candidates.items():
        points = operating_points(truths, predictions)
        base = CANDIDATE_INFO[name.removesuffix(TILED_SUFFIX)]
        report[name] = {
            **base,
            **({"nombre": f"{base['nombre']} + mosaico"} if name.endswith(TILED_SUFFIX) else {}),
            "ap50": average_precision(truths, predictions),
            "puntos": points,
            "recall_a_fp": {str(t): recall_at(points, t) for t in FP_TARGETS},
        }
    return report


# --- línea de comandos -----------------------------------------------------------------------------------


def load_truths(
    coco: Mapping[str, Any],
) -> tuple[dict[int, list[tuple[float, float, float, float]]], dict[int, str]]:
    """Todas las anotaciones del COCO se tratan como rostros."""
    truths: dict[int, list[tuple[float, float, float, float]]] = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        x, y, w, h = annotation["bbox"]
        truths[annotation["image_id"]].append((x, y, x + w, y + h))
    return truths, {image["id"]: image["file_name"] for image in coco["images"]}


def sweep_rows(points: Sequence[Mapping[str, Any]], low: float, high: float) -> list[Mapping[str, Any]]:
    """Los puntos de operación con umbral entre `low` y `high` (inclusive)."""
    return [row for row in points if low - 1e-9 <= row["umbral"] <= high + 1e-9]


def print_sweep(candidates: Mapping[str, Mapping[str, Any]], low: float, high: float) -> None:
    """Barrido del umbral de confianza: recall total y por tamaño frente a falsos positivos por foto."""
    for name, data in candidates.items():
        print(
            f"Barrido de {name} (umbral · recall · chicas · medianas · grandes · falsos positivos por foto):"
        )
        for row in sweep_rows(data["puntos"], low, high):
            sizes = row["recall_por_tamano"]
            print(
                f"  {row['umbral']:<5} {row['recall']!s:>7} {sizes['chicas']!s:>7} {sizes['medianas']!s:>7} "
                f"{sizes['grandes']!s:>7}   {row['fp_por_foto']}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, help=f"export COCO con las caras (por defecto <data-dir>/{FACE_FOLDER})"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument(
        "--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="pesos (MODELS_DIR)"
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument(
        "--rfdetr-model", help="modelo del puerto face del manifest (p. ej. rfdetr_openimages)"
    )
    parser.add_argument(
        "--service-model",
        help="modelo del puerto face del manifest corrido con el adaptador del servicio (p. ej. scrfd_10g)",
    )
    parser.add_argument(
        "--service-tiling", action="store_true", help="suma el adaptador del servicio con FACE_TILING=true"
    )
    parser.add_argument("--scrfd-model", type=Path, help="ONNX de SCRFD de 6 o 9 salidas")
    parser.add_argument(
        "--retinaface-model", type=Path, help="ONNX de RetinaFace-R50 (convert_to_onnx.py de biubug6)"
    )
    parser.add_argument("--yunet-model", type=Path, help="ONNX de YuNet (face_detection_yunet_*.onnx)")
    parser.add_argument("--size", type=int, default=640, help="lado de la entrada de SCRFD y RetinaFace")
    parser.add_argument(
        "--tiling",
        nargs="+",
        choices=tuple(CANDIDATE_INFO),
        default=[],
        help="detectores a correr también por mosaico (se suman como <nombre>+mosaico)",
    )
    parser.add_argument(
        "--sweep-from", type=float, default=0.20, help="umbral inicial del barrido que se imprime"
    )
    parser.add_argument(
        "--sweep-to", type=float, default=0.50, help="umbral final del barrido que se imprime"
    )
    parser.add_argument("--tile-px", type=int, default=DEFAULT_TILE_PX, help="lado de cada tile")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="solape entre tiles (0 a 0,9)")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        parser.error("indicá --out-dir o definí EVAL_OUT")
    if args.dataset_dir is None:
        if args.data_dir is None:
            parser.error("indicá --dataset-dir o --data-dir (EVAL_DATA_DIR)")
        args.dataset_dir = Path(args.data_dir) / FACE_FOLDER
    if not any(
        (args.rfdetr_model, args.service_model, args.scrfd_model, args.retinaface_model, args.yunet_model)
    ):
        parser.error(
            "indicá al menos un detector: --rfdetr-model, --scrfd-model, --retinaface-model o --yunet-model"
        )
    if (args.rfdetr_model or args.service_model) and args.models_dir is None:
        parser.error("--rfdetr-model y --service-model necesitan --models-dir (MODELS_DIR)")
    return args


def build_candidates(args: argparse.Namespace) -> dict[str, FaceDetector]:
    candidates: dict[str, FaceDetector] = {}
    if args.rfdetr_model:
        models = build_models(
            {"face": args.rfdetr_model}, args.manifest, args.models_dir, params={"redaction_min_conf": FLOOR}
        )
        candidates["rfdetr"] = AdapterFace(models)
    if args.service_model:
        for suffix, tiling in (("", 0.0), *((TILED_SUFFIX, 1.0),) * args.service_tiling):
            models = build_models(
                {"face": args.service_model},
                args.manifest,
                args.models_dir,
                params={
                    "redaction_min_conf": FLOOR,
                    "face_tiling": tiling,
                    "tiling_tile_px": float(args.tile_px),
                    "tiling_overlap": args.overlap,
                },
            )
            candidates["servicio" + suffix] = AdapterFace(models)
    if args.scrfd_model:
        candidates["scrfd"] = Scrfd(args.scrfd_model, args.size)
    if args.retinaface_model:
        candidates["retinaface_r50"] = RetinaFace(args.retinaface_model, args.size)
    if args.yunet_model:
        candidates["yunet"] = YuNet(args.yunet_model)
    for name in args.tiling:
        if name not in candidates:
            raise ValueError(f"--tiling pide {name}, pero no se pasó su modelo")
        candidates[name + TILED_SUFFIX] = Tiled(candidates[name], args.tile_px, args.overlap)
    return candidates


def main(
    argv: list[str] | None = None,
    factory: Callable[[argparse.Namespace], Mapping[str, FaceDetector]] = build_candidates,
) -> int:
    args = parse_args(argv)
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (args.dataset_dir, args.out_dir)):
        print("Las fotos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    annotations = Path(args.dataset_dir) / ANNOTATIONS
    if not annotations.is_file():
        print(f"Falta {ANNOTATIONS} en {args.dataset_dir}.", file=sys.stderr)
        return 1
    try:
        candidates = factory(args)
    except (ModelSetupError, OSError, ValueError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    truths, names = load_truths(json.loads(annotations.read_text(encoding="utf-8")))
    decoder = CvImageDecoder()
    predictions: dict[str, dict[int, list[Box]]] = {name: {} for name in candidates}
    times: dict[str, list[float]] = {name: [] for name in candidates}
    for image_id, file_name in names.items():
        pixels = decoder.decode((Path(args.dataset_dir) / file_name).read_bytes())
        for name, detector in candidates.items():
            start = time.perf_counter()
            boxes = detector.detect(pixels)
            times[name].append((time.perf_counter() - start) * 1000)
            predictions[name][image_id] = boxes or []
    report = {
        "fotos": len(truths),
        "caras": sum(len(v) for v in truths.values()),
        "caras_chicas": sum(is_small(t) for v in truths.values() for t in v),
        "candidatos": evaluate(truths, predictions),
    }
    for name in candidates:
        ordered = sorted(times[name])
        report["candidatos"][name]["latencia_ms_mediana"] = round(statistics.median(ordered))
        report["candidatos"][name]["latencia_ms_p95"] = round(ordered[round(0.95 * (len(ordered) - 1))])
    report["caras_por_tamano"] = dict(Counter(size_of(t) for v in truths.values() for t in v))
    report["criterio"] = verdict(report["candidatos"])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "face_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{report['fotos']} fotos · {report['caras']} caras "
        f"({report['caras_chicas']} chicas, menos de {SMALL_SIDE_PX} px)"
    )
    header = f"{'detector':<16}{'commercial':<11}{'AP50':>7}"
    header += "".join(f"{f'rec@{t}fp':>11}" for t in FP_TARGETS) + f"{'rec.chicas@0.1':>16}{'ms':>7}"
    print(header)
    for name, data in report["candidatos"].items():
        cells = []
        for target in FP_TARGETS:
            point = data["recall_a_fp"][str(target)]
            cells.append(f"{point['recall'] if point else '-':>11}")
        small = data["recall_a_fp"][str(FP_TARGETS[0])]
        print(
            f"{name:<16}{data['commercial']:<11}{data['ap50']!s:>7}{''.join(cells)}"
            f"{small['recall_chicas'] if small else '-':>16}{data['latencia_ms_mediana']:>7}"
        )
    print(f"Caras por tamaño: {report['caras_por_tamano']}")
    print_sweep(report["candidatos"], args.sweep_from, args.sweep_to)
    for target in FP_TARGETS:
        print(
            f"Recall por tamaño con a lo sumo {target} falsos positivos por foto (y falsos positivos reales):"
        )
        print(
            f"  {'detector':<20}"
            + "".join(f"{size:>10}" for size in SIZES)
            + f"{'FP/foto':>9}{'ms med.':>9}{'ms p95':>8}"
        )
        for name, data in report["candidatos"].items():
            point = data["recall_a_fp"][str(target)]
            sizes = point["recall_por_tamano"] if point else {}
            print(
                f"  {name:<20}"
                + "".join(f"{sizes.get(size, '-')!s:>10}" for size in SIZES)
                + f"{point['fp_por_foto'] if point else '-':>9}"
                + f"{data['latencia_ms_mediana']:>9}{data['latencia_ms_p95']:>8}"
            )
    print(f"Criterio: {report['criterio']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
