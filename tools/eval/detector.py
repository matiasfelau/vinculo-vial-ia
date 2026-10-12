"""Evalúa el detector (puertos `detector` y `face`) sobre un split COCO: AP50 y recall por clase.

Corre el adaptador real (la misma sesión ONNX y el mismo preproceso que el servicio) con los umbrales
casi en cero para calcular el AP50, y con los umbrales del servicio para el recall y la precisión: lo que
el servicio realmente vería. También mide el recall en objetos chicos y la latencia en CPU. La consola
y `report.json` tienen solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
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

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = "_annotations.coco.json"
DATASET_NAME = "openimages-detector"
IOU = 0.5
SMALL_SIDE_PX = 32  # un objeto con algún lado menor a esto, en la foto original, es chico
FLOOR = 0.01
CLASS_IDS = {name: index + 1 for index, name in enumerate(LABELS[column] for column in sorted(LABELS))}
DEFAULT_THRESHOLDS = {BoxLabel.CAR: 0.40, BoxLabel.WHEEL: 0.40, BoxLabel.PLATE: 0.20, FACE: 0.30}


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
        start = time.perf_counter()
        vehicles, faces = models.vehicle.detect(pixels), models.face.detect(pixels)
        times.append((time.perf_counter() - start) * 1000)
        predictions[image_id] = (vehicles or []) + (faces or [])
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
        "latencia_ms_mediana": round(statistics.median(times)),
        "latencia_ms_p95": round(sorted(times)[round(0.95 * (len(times) - 1))]),
        "clases": evaluate(coco_gt, predictions, thresholds),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "detector_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Modelo {report['modelo']} · {report['split']} · {report['fotos']} fotos · "
        f"{report['latencia_ms_mediana']} ms por foto (CPU, p95 {report['latencia_ms_p95']}) · "
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
