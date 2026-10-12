"""Entrena RF-DETR Small con CarDD para detectar daños en la carrocería.

Parte de los pesos COCO que publica Roboflow. FP32 por defecto, lote chico con acumulación de gradiente y
parada temprana sobre el split de validación; deja `training_summary.json` con tiempos, memoria y métricas.
Los pesos entrenados con CarDD no se publican. Uso: ver `tools/training/damage/README.md`.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from prepare import CLASSES, DATASET_NAME  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--out-dir", type=Path, help="carpeta de la corrida (por defecto <data-dir>/runs/damage)"
    )
    common.add_training_options(parser)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR")
    args.out_dir = args.out_dir or Path(args.data_dir) / "runs" / "damage"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.data_dir) / DATASET_NAME
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
    from rfdetr import RFDETRSmall  # se importa acá: tarda y no hace falta para --help

    common.train_model(
        RFDETRSmall(device=args.device),
        args,
        name="RF-DETR Small (daños, CarDD)",
        dataset_dir=dataset_dir,
        out_dir=args.out_dir,
        classes=CLASSES,
        gpu_before=gpu,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
