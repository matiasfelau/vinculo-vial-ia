"""Exporta el DINOv2 ajustado con 3DRealCar a ONNX y verifica que reproduce a PyTorch.

El ONNX tiene la misma interfaz que el del modelo base (`tools/export/dinov2.py`): entrada fija
`1 x 3 x 224 x 224`, FP32, opset 17, salida `last_hidden_state` con el token CLS primero. El adaptador del
servicio
lo usa tal cual (toma el CLS y lo normaliza). La verificación compara, sobre recortes reales de validación, el
embedding de ONNX Runtime (CPU) contra el de PyTorch: diferencia máxima por componente y coseno mínimo.

Imprime el sha256 que va en `models/manifest.json`. Los pesos quedan en `MODELS_DIR`, nunca en git.
Uso: ver `tools/training/reid/README.md`.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from train import HF_REPO, HF_REVISION, CropSet, Embedder, load_items
from transformers import Dinov2Model

from vvi_ia.adapters.models.dinov2 import INPUT_SIDE
from vvi_ia.adapters.models.onnx_session import OnnxSession

REPO_ROOT = Path(__file__).resolve().parents[3]
OPSET = 17
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "last_hidden_state"
MAX_ABS_DIFF = 1e-3  # por componente del embedding normalizado
MIN_COSINE = 0.9999
SAMPLES = 64
DEFAULT_NAME = "dinov2_vits14_reid"


class Tokens(torch.nn.Module):
    """Deja como única salida los tokens de la última capa (CLS primero), ya con la norma final."""

    def __init__(self, backbone: Dinov2Model) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=pixel_values).last_hidden_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt de train.py")
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--force", action="store_true", help="pisar el archivo si ya existe")
    args = parser.parse_args(argv)
    for flag, value in (("--root", args.root), ("--out-dir", args.out_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = Path(args.out_dir) / f"{args.name}.onnx"
    if target.exists() and not args.force:
        print(f"Ya existe {target}. Usá --force para pisarlo.", file=sys.stderr)
        return 1
    backbone = Dinov2Model.from_pretrained(HF_REPO, revision=HF_REVISION)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()
    tokens = Tokens(backbone).eval()
    sample = torch.randn(1, 3, INPUT_SIDE, INPUT_SIDE, generator=torch.Generator().manual_seed(0))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            tokens,
            (sample,),
            str(target),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=OPSET,
            dynamo=False,
        )
    session = OnnxSession(target.read_bytes(), intra_op_threads=0)
    items = load_items(Path(args.root))["val"][:SAMPLES]
    dataset = CropSet(Path(args.root), items, {}, train=False)
    reference = Embedder(backbone).eval()
    worst_diff, worst_cosine = 0.0, 1.0
    for index in range(len(dataset)):
        pixels, _ = dataset[index]
        batch = pixels.unsqueeze(0)
        with torch.no_grad():
            expected = reference(batch).numpy()[0]
        out: Any = session.run(None, {INPUT_NAME: batch.numpy()})[0]
        cls = np.asarray(out[0, 0], np.float32)
        got = cls / np.linalg.norm(cls)
        worst_diff = max(worst_diff, float(np.abs(got - expected).max()))
        worst_cosine = min(worst_cosine, float(got @ expected))
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    report = {
        "archivo": str(target),
        "sha256": sha,
        "recortes_verificados": len(dataset),
        "diferencia_maxima_por_componente": worst_diff,
        "coseno_minimo_onnx_vs_pytorch": worst_cosine,
        "epoca": checkpoint.get("epoca"),
    }
    print(json.dumps(report, indent=2))
    if worst_diff > MAX_ABS_DIFF or worst_cosine < MIN_COSINE:
        print("El ONNX NO reproduce a PyTorch: no se usa.", file=sys.stderr)
        target.unlink()
        return 1
    print("El ONNX reproduce a PyTorch. Registrá el sha256 en models/manifest.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
