"""Evalúa el detector (puertos `detector` y `face`) sobre un split COCO: AP50 y recall por clase.

Corre el adaptador real (la misma sesión ONNX y el mismo preproceso que el servicio) con los umbrales
casi en cero para calcular el AP50, y con los umbrales del servicio para el recall y la precisión: lo que
el servicio realmente vería. También mide el recall en objetos chicos y la latencia en CPU. La consola
y `report.json` tienen solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.adapters.models.rfdetr_detector import FACE, LABELS, MODEL_NAME
from vvi_ia.adapters.models.tiling import DEFAULT_OVERLAP, DEFAULT_TILE_PX
from vvi_ia.domain.model import Box, BoxLabel
from vvi_ia.service.validate_evidence import reduce_for_models, to_original

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = "_annotations.coco.json"
DATASET_NAME = "openimages-detector"
IOU = 0.5
SMALL_SIDE_PX = 32  # un objeto con algún lado menor a esto, en la foto original, es chico
FLOOR = 0.01
CLASS_IDS = {name: index + 1 for index, name in enumerate(LABELS[column] for column in sorted(LABELS))}
DEFAULT_THRESHOLDS = {BoxLabel.CAR: 0.40, BoxLabel.WHEEL: 0.40, BoxLabel.PLATE: 0.20, FACE: 0.30}
SERVICE_MAX_SIDE_PX = 1024  # MODEL_MAX_SIDE_PX: el servicio reduce cada foto a este lado antes de los modelos
COMBINED = (str(BoxLabel.CAR), str(BoxLabel.PLATE))  # lo que se presenta: autos y patentes (sin ruedas)
RECALL_POINTS = 101  # interpolación del AP, igual que COCO


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU de dos cajas `(x1, y1, x2, y2)`."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(
    truths: Sequence[Sequence[float]], predictions: Sequence[tuple[float, Sequence[float]]]
) -> tuple[list[bool], set[int]]:
    """Empareja cada predicción (de mayor a menor confianza) con una caja real que nadie más usó.

    Devuelve si cada predicción acertó y qué cajas reales quedaron encontradas.
    """
    used: set[int] = set()
    hits: list[bool] = []
    for _, box in sorted(predictions, key=lambda item: -item[0]):
        best, best_iou = -1, IOU
        for index, truth in enumerate(truths):
            overlap = iou(box, truth)
            if index not in used and overlap >= best_iou:
                best, best_iou = index, overlap
        if best >= 0:
            used.add(best)
        hits.append(best >= 0)
    return hits, used


def scored_hits(
    truths: Sequence[Sequence[float]], predictions: Sequence[tuple[float, Sequence[float]]]
) -> list[tuple[float, bool]]:
    """`(confianza, acertó)` de cada predicción, de mayor a menor confianza."""
    hits, _ = match(truths, predictions)
    return list(zip(sorted((conf for conf, _ in predictions), reverse=True), hits, strict=True))


def average_precision(ranked: Sequence[tuple[float, bool]], truth_count: int) -> float:
    """AP con 101 puntos de recall (como COCO) de una lista `(confianza, acertó)`."""
    if truth_count == 0 or not ranked:
        return 0.0
    ordered = sorted(ranked, key=lambda item: -item[0])
    hits = np.array([hit for _, hit in ordered], dtype=float)
    tp = np.cumsum(hits)
    fp = np.cumsum(1 - hits)
    recall = tp / truth_count
    precision = tp / np.maximum(tp + fp, 1e-12)
    for index in range(len(precision) - 2, -1, -1):  # envolvente: la precisión no sube al avanzar
        precision[index] = max(precision[index], precision[index + 1])
    total = 0.0
    for point in np.linspace(0.0, 1.0, RECALL_POINTS):
        found = int(np.searchsorted(recall, point, side="left"))
        total += float(precision[found]) if found < len(precision) else 0.0
    return total / RECALL_POINTS


def truth_boxes(coco_gt: COCO, image_id: int, category_id: int) -> list[tuple[float, float, float, float]]:
    return [
        (a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
        for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[image_id], catIds=[category_id]))
    ]


def combined(
    coco_gt: COCO,
    predictions: dict[int, list[Box]],
    thresholds: Mapping[str, float],
    class_ids: Mapping[str, int] = CLASS_IDS,
    names: Sequence[str] = COMBINED,
) -> dict[str, Any]:
    """Micro-promedio de varias clases: las cajas de todas juntas (el acierto sigue siendo por clase).

    Recall y precisión con el umbral de cada clase; el AP50 con todas las cajas (umbral casi en cero)
    ordenadas por confianza en una sola curva.
    """
    tp = fp = truth_count = 0
    ranked: list[tuple[float, bool]] = []
    for image_id in coco_gt.getImgIds():
        for name in names:
            truths = truth_boxes(coco_gt, image_id, class_ids[name])
            guesses = [
                (box.conf, (box.x1, box.y1, box.x2, box.y2))
                for box in predictions.get(image_id, [])
                if box.label == name
            ]
            truth_count += len(truths)
            ranked += scored_hits(truths, guesses)
            hits, _ = match(truths, [g for g in guesses if g[0] >= thresholds[name]])
            tp += sum(hits)
            fp += len(hits) - sum(hits)
    return {
        "clases": list(names),
        "cajas_reales": truth_count,
        "recall": rate(tp, truth_count),
        "precision": rate(tp, tp + fp),
        "falsos_positivos": fp,
        "ap50": round(average_precision(ranked, truth_count), 4),
    }


def ap50(coco_gt: COCO, detections: list[dict[str, Any]], category_id: int) -> float | None:
    """AP con IoU 0,5 de una clase; `None` si no hay cajas reales de esa clase."""
    if not coco_gt.getAnnIds(catIds=[category_id]):
        return None
    if not detections:
        return 0.0
    evaluator = COCOeval(coco_gt, coco_gt.loadRes(detections), "bbox")
    evaluator.params.catIds = [category_id]
    evaluator.params.iouThrs = np.array([IOU])
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"][
        0, :, 0, 0, -1
    ]  # IoU 0,5 · todos los recalls · área "all" · 100 dets
    valid = precision[precision > -1]
    return round(float(valid.mean()), 4) if valid.size else 0.0


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate(
    coco_gt: COCO,
    predictions: dict[int, list[Box]],
    thresholds: dict[str, float] | None = None,
    class_ids: Mapping[str, int] = CLASS_IDS,
) -> dict[str, Any]:
    """AP50, recall y precisión por clase, y recall de los objetos chicos."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    detections = [
        {
            "image_id": image_id,
            "category_id": class_ids[str(box.label)],
            "bbox": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
            "score": box.conf,
        }
        for image_id, boxes in predictions.items()
        for box in boxes
    ]
    report: dict[str, Any] = {}
    for name, category_id in class_ids.items():
        tp = fp = truth_count = small_found = small_total = 0
        for image_id in coco_gt.getImgIds():
            truths = [
                (a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
                for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[image_id], catIds=[category_id]))
            ]
            guesses = [
                (box.conf, (box.x1, box.y1, box.x2, box.y2))
                for box in predictions.get(image_id, [])
                if box.label == name and box.conf >= thresholds[name]
            ]
            hits, found = match(truths, guesses)
            tp += sum(hits)
            fp += len(hits) - sum(hits)
            truth_count += len(truths)
            small = {i for i, t in enumerate(truths) if min(t[2] - t[0], t[3] - t[1]) < SMALL_SIDE_PX}
            small_total += len(small)
            small_found += len(small & found)
        report[name] = {
            "cajas_reales": truth_count,
            "umbral": thresholds[name],
            "recall": rate(tp, truth_count),
            "precision": rate(tp, tp + fp),
            "falsos_positivos": fp,
            "cajas_chicas": small_total,
            "recall_chicas": rate(small_found, small_total),
            "ap50": ap50(coco_gt, [d for d in detections if d["category_id"] == category_id], category_id),
        }
    return report


def cpu_name() -> str:
    """Nombre de la CPU (solo para documentar dónde se midió la latencia)."""
    try:
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor() or platform.machine()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, help=f"split COCO (por defecto <data-dir>/{DATASET_NAME}/<split>)"
    )
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument(
        "--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="pesos (MODELS_DIR)"
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--detector-min-conf", type=float, default=DEFAULT_THRESHOLDS[BoxLabel.CAR])
    parser.add_argument("--plate-min-conf", type=float, default=DEFAULT_THRESHOLDS[BoxLabel.PLATE])
    parser.add_argument("--face-min-conf", type=float, default=DEFAULT_THRESHOLDS[FACE])
    parser.add_argument(
        "--face-tiling", action="store_true", help="rostros también por mosaico (FACE_TILING)"
    )
    parser.add_argument(
        "--plate-tiling", action="store_true", help="patentes también por mosaico (PLATE_TILING)"
    )
    parser.add_argument(
        "--model-max-side",
        type=int,
        default=SERVICE_MAX_SIDE_PX,
        help="lado al que se reduce cada foto antes del detector (MODEL_MAX_SIDE_PX; 0 = sin reducir)",
    )
    parser.add_argument("--tile-px", type=int, default=DEFAULT_TILE_PX, help="lado de cada tile")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="solape entre tiles")
    args = parser.parse_args(argv)
    for flag, value in (("--out-dir", args.out_dir), ("--models-dir", args.models_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    if args.dataset_dir is None:
        if args.data_dir is None:
            parser.error("indicá --dataset-dir o --data-dir (TRAIN_DATA_DIR)")
        args.dataset_dir = Path(args.data_dir) / DATASET_NAME / args.split
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if Path(args.out_dir).resolve().is_relative_to(REPO_ROOT) or Path(
        args.dataset_dir
    ).resolve().is_relative_to(REPO_ROOT):
        print("El dataset y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    annotations = Path(args.dataset_dir) / ANNOTATIONS
    if not annotations.is_file():
        print(f"Falta {annotations.name} en {args.dataset_dir}.", file=sys.stderr)
        return 1
    try:
        models = build_models(
            {"detector": args.model, "face": args.model},
            args.manifest,
            args.models_dir,
            params={
                "detector_min_conf": FLOOR,
                "redaction_min_conf": FLOOR,
                "face_tiling": float(args.face_tiling),
                "plate_tiling": float(args.plate_tiling),
                "tiling_tile_px": float(args.tile_px),
                "tiling_overlap": args.overlap,
            },
        )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    coco_gt = COCO(str(annotations))
    decoder = CvImageDecoder()
    predictions: dict[int, list[Box]] = {}
    times: list[float] = []
    for image_id, info in coco_gt.imgs.items():
        pixels = decoder.decode((Path(args.dataset_dir) / info["file_name"]).read_bytes())
        view = reduce_for_models(pixels, args.model_max_side) if args.model_max_side else None
        shown = pixels if view is None else view
        ratio = pixels.shape[1] / shown.shape[1]
        start = time.perf_counter()
        # Lo que corre el servicio: un pase para decidir la presencia o, con mosaico, el pase más los tiles
        vehicles = models.vehicle.detect_plates(shown) if args.plate_tiling else models.vehicle.detect(shown)
        times.append((time.perf_counter() - start) * 1000)
        faces = models.face.detect(shown)  # reutiliza el pase (memoria del adaptador): no cuenta en el tiempo
        predictions[image_id] = to_original((vehicles or []) + (faces or []), ratio) or []
    thresholds = {
        BoxLabel.CAR: args.detector_min_conf,
        BoxLabel.WHEEL: args.detector_min_conf,
        BoxLabel.PLATE: args.plate_min_conf,
        FACE: args.face_min_conf,
    }
    report = {
        "modelo": models.versions()["detector"],
        "split": args.split,
        "fotos": len(predictions),
        "mosaico": {"rostros": args.face_tiling, "patentes": args.plate_tiling, "tile_px": args.tile_px}
        if args.face_tiling or args.plate_tiling
        else None,
        "cpu": cpu_name(),
        "lado_maximo_px": args.model_max_side or None,
        "latencia_ms_media": round(statistics.mean(times)),
        "latencia_ms_mediana": round(statistics.median(times)),
        "latencia_ms_p95": round(sorted(times)[round(0.95 * (len(times) - 1))]),
        "clases": evaluate(coco_gt, predictions, thresholds),
        "auto_y_patente": combined(coco_gt, predictions, thresholds),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "detector_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Modelo {report['modelo']} · {report['split']} · {report['fotos']} fotos · "
        f"{report['latencia_ms_media']} ms por foto (media; mediana {report['latencia_ms_mediana']}, "
        f"p95 {report['latencia_ms_p95']}) en {report['cpu']} · "
        f"mosaico {report['mosaico'] or 'apagado'}"
    )
    print(
        f"{'clase':<8}{'reales':>8}{'umbral':>8}{'recall':>8}{'prec.':>8}{'FP':>6}{'chicas':>8}{'rec.ch.':>9}{'AP50':>8}"
    )
    for name, row in report["clases"].items():
        print(
            f"{name:<8}{row['cajas_reales']:>8}{row['umbral']:>8}{row['recall']!s:>8}{row['precision']!s:>8}"
            f"{row['falsos_positivos']:>6}{row['cajas_chicas']:>8}{row['recall_chicas']!s:>9}{row['ap50']!s:>8}"
        )
    print(f"auto + patente (micro): {report['auto_y_patente']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
