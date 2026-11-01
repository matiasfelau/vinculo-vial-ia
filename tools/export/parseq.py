"""Exporta PARSeq preentrenado a ONNX para el puerto `ocr`.

Baja de GitHub el código de `baudm/parseq` (en un commit fijo, con `torch.hub`) y los pesos
preentrenados de su release v1.0.0 (PyTorch verifica el hash del archivo). No se reentrena nada.
El ONNX resultante es la decodificación greedy autorregresiva con una iteración de refinamiento,
con los 26 pasos desenrollados y un lote fijo de 1. Después compara sus logits con los del
`forward` original de PARSeq y muestra el sha256 que va en `models/manifest.json`.

Uso: ver `tools/export/README.md`.
"""

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vvi_ia.adapters.models.onnx_session import OnnxSession
from vvi_ia.adapters.models.parseq import CHARSET, MODEL_NAME, OUTPUT_NAMES

HUB_REPO = "baudm/parseq"
HUB_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
HUB_MODEL = "parseq"
INPUT_NAME = "image"
OPSET = 17
MAX_ABS_DIFF = 1e-3
CHECK_SEEDS = (0, 1, 2)


class GreedyDecoder(nn.Module):
    """El `forward` de PARSeq con `max_length` fijo: mismo algoritmo, sin asignaciones in-place.

    Las máscaras booleanas son constantes. El `forward` original las modifica en el lugar y eso se
    exporta como un `Where` sobre booleanos, que ONNX Runtime no implementa en CPU.
    """

    def __init__(self, model: nn.Module, bos_id: int, eos_id: int, steps: int) -> None:
        super().__init__()
        self.model = model
        self.bos_id, self.eos_id, self.steps = bos_id, eos_id, steps
        cloze = torch.zeros(steps, steps, dtype=torch.bool)
        for i in range(steps - 1):
            cloze[i, i + 1] = True  # cada posición ve todo el contexto menos su propio carácter
        self.register_buffer("causal", torch.triu(torch.ones(steps, steps, dtype=torch.bool), 1))
        self.register_buffer("cloze", cloze)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        m, steps = self.model, self.steps
        memory = m.encode(image)
        positions = m.pos_queries[:, :steps].expand(1, -1, -1)
        bos = torch.full((1, 1), self.bos_id, dtype=torch.long, device=image.device)
        context = bos
        logits = []
        for i in range(steps):
            j = i + 1
            out = m.decode(
                context,
                memory,
                self.causal[:j, :j],
                tgt_query=positions[:, i:j],
                tgt_query_mask=self.causal[i:j, :j],
            )
            step = m.head(out)
            logits.append(step)
            if j < steps:
                context = torch.cat([context, step[:, 0].argmax(-1, keepdim=True)], dim=1)
        first = torch.cat(logits, dim=1)
        context = torch.cat([bos, first[:, :-1].argmax(-1)], dim=1)
        padding = (context == self.eos_id).int().cumsum(-1) > 0
        out = m.decode(context, memory, self.cloze, padding, positions, self.cloze[:, : context.shape[1]])
        refined: torch.Tensor = m.head(out)
        return refined


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=os.environ.get("MODELS_DIR"),
        help="carpeta de pesos (por defecto, la variable MODELS_DIR)",
    )
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
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bajando PARSeq (commit {HUB_COMMIT[:12]}) y sus pesos preentrenados...")
    system = torch.hub.load(f"{HUB_REPO}:{HUB_COMMIT}", HUB_MODEL, pretrained=True, trust_repo=True)
    system = system.eval()
    tokenizer, model = system.tokenizer, system.model
    height, width = (int(side) for side in system.hparams.img_size)
    steps = int(system.hparams.max_label_length) + 1
    if system.hparams.charset_train != CHARSET or tokenizer.eos_id != 0 or len(tokenizer) != len(CHARSET) + 3:
        print("El modelo no usa el juego de caracteres que espera el adaptador.", file=sys.stderr)
        return 1

    wrapper = GreedyDecoder(model, tokenizer.bos_id, tokenizer.eos_id, steps).eval()
    print(f"Exportando a ONNX (opset {OPSET}, FP32, entrada fija 1×3×{height}×{width}, {steps} pasos)...")
    sample = torch.randn(1, 3, height, width, generator=torch.Generator().manual_seed(0))
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        exported = Path(tmp) / "parseq.onnx"
        torch.onnx.export(
            wrapper,
            (sample,),
            str(exported),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            opset_version=OPSET,
            dynamo=False,
        )
        data = exported.read_bytes()

    session = OnnxSession(data, intra_op_threads=0)
    worst = 0.0
    for seed in CHECK_SEEDS:
        image = torch.randn(1, 3, height, width, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            expected = model(tokenizer, image, steps - 1).numpy()
        (actual,) = session.run(None, {INPUT_NAME: image.numpy()})
        worst = max(worst, float(np.max(np.abs(actual - expected))))
        if not np.array_equal(actual.argmax(-1), expected.argmax(-1)):
            worst = float("inf")
    print(f"Diferencia máxima entre ONNX (CPU) y el forward original de PARSeq: {worst:.2e}")
    if worst > MAX_ABS_DIFF:
        print("El ONNX no reproduce a PyTorch: no se guardó. No lo uses.", file=sys.stderr)
        return 1

    target.write_bytes(data)
    print(f"\nListo: {target} ({len(data) / 1e6:.1f} MB)")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    print(f'Copiá ese valor en el campo "sha256" de {MODEL_NAME} en models/manifest.json.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
