"""Exporta DINOv2 ViT-S/14 a ONNX para el puerto `embedder`.

Baja los pesos publicados por Meta (`facebook/dinov2-small`, Apache 2.0) de Hugging Face, en una
revisión fija, y los convierte sin entrenar ni ajustar nada. Después compara la salida del ONNX con
la de PyTorch y muestra el sha256 que va en `models/manifest.json`.

Uso: ver `tools/export/README.md`.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Dinov2Model

from vvi_ia.adapters.models.dinov2 import INPUT_SIDE, MODEL_NAME
from vvi_ia.adapters.models.onnx_session import OnnxSession

HF_REPO = "facebook/dinov2-small"
HF_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
OPSET = 17
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "last_hidden_state"
MAX_ABS_DIFF = 1e-3
SEED = 0


class Tokens(torch.nn.Module):
    """Deja como única salida los tokens de la última capa (CLS primero), ya con la norma final."""

    def __init__(self, model: Dinov2Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=os.environ.get("MODELS_DIR"),
        help="carpeta de pesos (por defecto, la variable MODELS_DIR)",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--revision", default=HF_REVISION, help="commit de Hugging Face a exportar")
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

    print(f"Bajando {HF_REPO} (revisión {args.revision[:12]})...")
    model = Tokens(Dinov2Model.from_pretrained(HF_REPO, revision=args.revision)).eval().to(args.device)
    sample = torch.randn(
        1, 3, INPUT_SIDE, INPUT_SIDE, generator=torch.Generator().manual_seed(SEED), dtype=torch.float32
    ).to(args.device)

    print(f"Exportando a ONNX (opset {OPSET}, FP32, entrada fija 1×3×{INPUT_SIDE}×{INPUT_SIDE})...")
    with torch.no_grad():
        expected = model(sample).cpu().numpy()
        torch.onnx.export(
            model,
            (sample,),
            str(target),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=OPSET,
            dynamo=False,
        )

    data = target.read_bytes()
    actual = OnnxSession(data, intra_op_threads=0).run(None, {INPUT_NAME: sample.cpu().numpy()})[0]
    diff = float(np.max(np.abs(actual - expected)))
    print(f"Diferencia máxima entre ONNX (CPU) y PyTorch ({args.device}): {diff:.2e}")
    if actual.shape != expected.shape or diff > MAX_ABS_DIFF:
        target.unlink()
        print("El ONNX no reproduce a PyTorch: se borró el archivo. No lo uses.", file=sys.stderr)
        return 1

    print(f"\nListo: {target} ({len(data) / 1e6:.1f} MB), salida {actual.shape}")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    print(f'Copiá ese valor en el campo "sha256" de {MODEL_NAME} en models/manifest.json.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
