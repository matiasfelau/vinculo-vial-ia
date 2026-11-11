"""Entrena RF-DETR Small con el subconjunto de Open Images del detector v2 (`v2_download.py`).

**Mismo procedimiento que v1** (`tools/training/detector/train.py`): parte de los pesos COCO de Roboflow,
FP32, lote 2 con acumulación de 8, hasta 20 épocas con parada temprana de 5 sobre la validación, 512 px. Lo
único distinto son los datos y las clases (doce, en el orden de `v2_classes.CLASSES`). Las fotos de fondo
van sin anotaciones. Uso: ver `tools/training/detector_v2/README.md`.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from v2_classes import CLASSES  # noqa: E402
from v2_download import DATASET_NAME  # noqa: E402

RUN_NAME = "detector-v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--out-dir", type=Path, help=f"carpeta de la corrida (por defecto <data-dir>/runs/{RUN_NAME})"
    )
    common.add_training_options(parser)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("indicá --data-dir o definí TRAIN_DATA_DIR")
    args.out_dir = args.out_dir or Path(args.data_dir) / "runs" / RUN_NAME
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
                f"Falta {split}/{common.ANNOTATIONS} en {dataset_dir}. Corré primero v2_download.py.",
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
        name="RF-DETR Small (detector v2)",
        dataset_dir=dataset_dir,
        out_dir=args.out_dir,
        classes=CLASSES,
        gpu_before=gpu,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
