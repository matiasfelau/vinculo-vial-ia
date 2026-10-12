"""Entrena RF-DETR-Seg Small con el dataset propio de rebajes (clases `curb` y `curb_cut`).

Parte de los pesos COCO que publica Roboflow. FP32 por defecto, lote 2 con acumulación de 8 y parada
temprana sobre el split de validación; deja `training_summary.json` con tiempos, memoria y métricas.
Se reentrena a medida que crece el dataset. Uso: ver `tools/training/curb/README.md`.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from prepare import CLASSES, CURB_FOLDER, DATASET_FOLDER  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument(
        "--out-dir", type=Path, help="carpeta de la corrida (por defecto <curb>/runs/<fecha>)"
    )
    common.add_training_options(parser, epochs=50, patience=10)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí EVAL_DATA_DIR")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    curb = Path(args.data_dir) / CURB_FOLDER
    dataset_dir = curb / DATASET_FOLDER
    args.out_dir = args.out_dir or curb / "runs" / "rfdetr-seg-curb"
    if common.inside_repo(dataset_dir, args.out_dir):
        print("Los datos y los resultados no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    for split in ("train", "valid"):
        if not (dataset_dir / split / common.ANNOTATIONS).is_file():
            print(
                f"Falta {split}/{common.ANNOTATIONS} en {dataset_dir}. Corré primero prepare.py.",
                file=sys.stderr,
            )
            return 1
    gpu, problem = common.check_device(args.device)
    if problem:
        print(problem, file=sys.stderr)
        return 1
    from rfdetr import RFDETRSegSmall  # se importa acá: tarda y no hace falta para --help

    common.train_model(
        RFDETRSegSmall(device=args.device),
        args,
        name="RF-DETR-Seg Small (rebajes)",
        dataset_dir=dataset_dir,
        out_dir=args.out_dir,
        classes=CLASSES,
        gpu_before=gpu,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
