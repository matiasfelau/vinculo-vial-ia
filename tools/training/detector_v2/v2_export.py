"""Exporta a ONNX el detector v2 entrenado con `v2_train.py`, listo para `MODELS_DIR`.

Compara las salidas del ONNX con las de PyTorch, comprueba que las clases sean las de `v2_classes.CLASSES` y
muestra el sha256 que va en `models/manifest.json`. **No lo activa**: el modelo entra al manifest como
`rfdetr_openimages_v2` y solo se usa si una configuración lo elige (`MODEL_DETECTOR=rfdetr_openimages_v2`).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from v2_classes import CLASSES  # noqa: E402
from v2_train import RUN_NAME  # noqa: E402

from vvi_ia.adapters.models.rfdetr_detector import MODEL_NAME_V2, OUTPUT_NAMES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument("--checkpoint", type=Path, help="checkpoint a exportar (por defecto el mejor)")
    parser.add_argument(
        "--out-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="carpeta de pesos (MODELS_DIR)"
    )
    parser.add_argument("--force", action="store_true", help="pisar el archivo si ya existe")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        parser.error("indicá --out-dir o definí MODELS_DIR")
    if args.checkpoint is None:
        if args.data_dir is None:
            parser.error("indicá --checkpoint o --data-dir (TRAIN_DATA_DIR)")
        args.checkpoint = common.find_checkpoint(Path(args.data_dir) / "runs" / RUN_NAME)
        if args.checkpoint is None:
            parser.error("no encontré ningún checkpoint de la corrida: indicá --checkpoint")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from rfdetr import RFDETRSmall  # se importa acá: tarda y no hace falta para --help

    return common.export_checkpoint(
        RFDETRSmall,
        args.checkpoint,
        classes=CLASSES,
        outputs=OUTPUT_NAMES,
        target=Path(args.out_dir) / f"{MODEL_NAME_V2}.onnx",
        force=args.force,
        model_name=MODEL_NAME_V2,
    )


if __name__ == "__main__":
    sys.exit(main())
