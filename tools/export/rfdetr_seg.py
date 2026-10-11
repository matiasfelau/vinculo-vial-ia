"""Exporta RF-DETR-Seg preentrenado con COCO a ONNX para el puerto `segmenter`.

Usa los pesos que publica Roboflow para RF-DETR-Seg Small (o Medium) entrenados con COCO, sin
reentrenar nada. Solo se permiten esos dos tamaños: XL y 2XL tienen otra licencia (PML 1.0). Después
compara las salidas del ONNX con las de PyTorch y muestra el sha256 que va en `models/manifest.json`.

Los pesos se bajan de `storage.googleapis.com/rfdetr` a `~/.roboflow/models` (lo hace la librería,
con verificación de MD5). Uso: ver `tools/export/README.md`.
"""

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from rfdetr import RFDETRSegMedium, RFDETRSegSmall

from vvi_ia.adapters.models.onnx_session import OnnxSession
from vvi_ia.adapters.models.rfdetr_seg import MODEL_NAME, OUTPUT_NAMES

SIZES = {"small": RFDETRSegSmall, "medium": RFDETRSegMedium}
OPSET = 17
MAX_ABS_DIFF = 5e-2  # sobre logits de magnitud ~10: lo observado es 1e-2 o menos
SEED = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=os.environ.get("MODELS_DIR"),
        help="carpeta de pesos (por defecto, la variable MODELS_DIR)",
    )
    parser.add_argument("--size", choices=sorted(SIZES), default="small", help="tamaño del modelo")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--force", action="store_true", help="pisar el archivo si ya existe")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        parser.error("indicá --out-dir o definí MODELS_DIR")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    target = out_dir / f"{MODEL_NAME}.onnx"
    if target.exists() and not args.force:
        print(f"Ya existe {target}. Usá --force para pisarlo.", file=sys.stderr)
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Pediste --device cuda pero PyTorch no ve ninguna GPU. Usá --device cpu.", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando RF-DETR-Seg {args.size} (COCO)...")
    model = SIZES[args.size](device=args.device)
    resolution = int(model.model_config.resolution)

    print(f"Exportando a ONNX (opset {OPSET}, FP32, entrada fija 1×3×{resolution}×{resolution})...")
    with tempfile.TemporaryDirectory() as tmp:
        exported = Path(model.export(output_dir=tmp, opset_version=OPSET, verbose=False))
        data = exported.read_bytes()

    session = OnnxSession(data, intra_op_threads=0)
    if session.output_names != OUTPUT_NAMES:
        print(f"Salidas inesperadas del ONNX: {session.output_names}", file=sys.stderr)
        return 1
    sample = torch.randn(
        1, 3, resolution, resolution, generator=torch.Generator().manual_seed(SEED), dtype=torch.float32
    )
    actual = session.run(None, {session.input_names[0]: sample.numpy()})
    with torch.no_grad():
        reference = model.model.model.eval().to(args.device)(sample.to(args.device))
    expected = [reference["pred_boxes"], reference["pred_logits"], reference["pred_masks"]]
    diffs = [float(np.max(np.abs(a - e.cpu().numpy()))) for a, e in zip(actual, expected, strict=True)]
    print(f"Diferencia máxima ONNX (CPU) vs PyTorch ({args.device}) en {OUTPUT_NAMES}: {diffs}")
    if max(diffs) > MAX_ABS_DIFF:
        print("El ONNX no reproduce a PyTorch: no se guardó. No lo uses.", file=sys.stderr)
        return 1

    target.write_bytes(data)
    print(f"\nListo: {target} ({len(data) / 1e6:.1f} MB)")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    print(f'Copiá ese valor en el campo "sha256" de {MODEL_NAME} en models/manifest.json.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
