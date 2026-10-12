"""Evalúa el detector de daños (puerto `damage`) sobre un split de CarDD: AP50 y recall por clase.

Corre el adaptador real (la misma sesión ONNX y el mismo preproceso que el servicio) con el umbral casi en
cero para calcular el AP50 y con `DAMAGE_MIN_CONF` para el recall y la precisión: lo que el servicio
realmente vería. La consola y `damage_report.json` tienen solo cifras. CarDD no se redistribuye: el dataset y
la salida se quedan fuera del repo. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from detector import ANNOTATIONS, FLOOR, REPO_ROOT, evaluate
from pycocotools.coco import COCO

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.adapters.models.rfdetr_damage import DAMAGE_LABELS, MODEL_NAME
from vvi_ia.domain.model import Box

DATASET_NAME = "cardd-detector"
DEFAULT_MIN_CONF = 0.70
CLASS_IDS = {
    name: index + 1 for index, name in enumerate(DAMAGE_LABELS[column] for column in sorted(DAMAGE_LABELS))
}


def photo_level(coco_gt: Any, predictions: dict[int, list[Box]], min_conf: float) -> dict[str, Any]:
    """Qué fracción de las fotos con algún daño anotado tiene al menos una caja con confianza suficiente.

    Es lo que decide el servicio: la foto de detalle pasa si el detector encuentra algún daño. Todas las
    fotos de CarDD tienen daño, así que esto es un recall por foto; no mide falsos positivos.
    """
    with_damage = [i for i in coco_gt.getImgIds() if coco_gt.getAnnIds(imgIds=[i])]
    found = sum(any(b.conf >= min_conf for b in predictions.get(i, [])) for i in with_damage)
    return {
        "fotos_con_dano": len(with_damage),
        "con_alguna_caja": found,
        "recall_por_foto": round(found / len(with_damage), 4) if with_damage else None,
    }


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
    parser.add_argument(
        "--damage-min-conf", type=float, default=float(os.environ.get("DAMAGE_MIN_CONF", DEFAULT_MIN_CONF))
    )
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
        print("CarDD y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    annotations = Path(args.dataset_dir) / ANNOTATIONS
    if not annotations.is_file():
        print(f"Falta {annotations.name} en {args.dataset_dir}. Corré antes prepare.py.", file=sys.stderr)
        return 1
    try:
        models = build_models(
            {"damage": args.model}, args.manifest, args.models_dir, params={"damage_min_conf": FLOOR}
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
        boxes = models.damage.detect(pixels)
        times.append((time.perf_counter() - start) * 1000)
        predictions[image_id] = boxes or []
    thresholds = dict.fromkeys(CLASS_IDS, args.damage_min_conf)
    report = {
        "modelo": models.versions()["damage"],
        "split": args.split,
        "fotos": len(predictions),
        "latencia_ms_mediana": round(statistics.median(times)),
        "por_foto": photo_level(coco_gt, predictions, args.damage_min_conf),
        "clases": evaluate(coco_gt, predictions, thresholds, CLASS_IDS),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "damage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Modelo {report['modelo']} · {report['split']} · {report['fotos']} fotos · "
        f"{report['latencia_ms_mediana']} ms por foto (CPU) · umbral {args.damage_min_conf}"
    )
    print(f"Por foto (¿hay algún daño?): {report['por_foto']}")
    print(
        f"{'clase':<14}{'reales':>8}{'recall':>8}{'prec.':>8}{'FP':>6}{'chicas':>8}{'rec.ch.':>9}{'AP50':>8}"
    )
    for name, row in report["clases"].items():
        print(
            f"{name:<14}{row['cajas_reales']:>8}{row['recall']!s:>8}{row['precision']!s:>8}"
            f"{row['falsos_positivos']:>6}{row['cajas_chicas']:>8}{row['recall_chicas']!s:>9}{row['ap50']!s:>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
