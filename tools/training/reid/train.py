"""Ajusta DINOv2 ViT-S/14 con 3DRealCar para re-identificar autos (R2: ¿GENERAL y PLATE son el mismo auto?).

Pérdida: **CosFace** (margen aditivo en el coseno) sobre un prototipo por auto de entrenamiento. El servicio
decide
con un umbral sobre el coseno (`REID_T`), así que se entrena justo eso: que las vistas del mismo auto queden
por
encima del coseno de cualquier otro auto con un margen. Al ser una pérdida con prototipos, cada paso ve todos
los
autos como negativos: el lote chico y la acumulación de gradiente (los 6 GB de la GTX 1660, en FP32) no
cambian lo
que aprende, algo que una pérdida por pares dentro del lote (triplet, contrastiva) no garantiza.

Los prototipos arrancan en el promedio de los embeddings del modelo base, y el embedding es el token CLS de la
última capa normalizado L2, igual que el servicio: el ONNX ajustado es un reemplazo directo del base.

Parada temprana sobre validación: el S4 (autos distintos detectados) al umbral que deja el U1 bajo 1 % con
cota del 95 %. Los datos y los pesos quedan fuera del repositorio. Uso: ver `tools/training/reid/README.md`.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from dataset import build_pairs
from metrics import auc, calibrate
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Dinov2Model

from vvi_ia.adapters.models.dinov2 import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIDE, preprocess

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_REPO = "facebook/dinov2-small"
HF_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"  # la misma que `tools/export/dinov2.py`
SCALE = 30.0  # CosFace: s
MARGIN = 0.30  # CosFace: m (en coseno)
MIN_FREE_GPU_MB = 3000
EVAL_BATCH = 64
MAX_PIXEL = 255.0
AREA_RANGE = (
    0.80,
    1.0,
)  # lo que queda del recorte tras el recorte aleatorio (la caja del detector no es exacta)
ASPECT_JITTER = 0.10
COLOR_JITTER = 0.20
FLIP_PROBABILITY = 0.5
EARLY_STOP_MIN_GAIN = 1e-4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - semilla global para las librerías que la usan
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_square(crop: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """El achique del adaptador: a 224 x 224 (sin respetar la proporción), con su interpolación."""
    shrinks = max(crop.shape[:2]) > INPUT_SIDE
    return np.asarray(
        cv2.resize(
            crop, (INPUT_SIDE, INPUT_SIDE), interpolation=cv2.INTER_AREA if shrinks else cv2.INTER_CUBIC
        ),
        np.uint8,
    )


def normalize(image: np.ndarray[Any, Any]) -> torch.Tensor:
    scaled = (image.astype(np.float32) / MAX_PIXEL - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1)))


def eval_tensor(crop: np.ndarray[Any, Any]) -> torch.Tensor:
    """Exactamente el preproceso del servicio (`dinov2.preprocess`) sobre el recorte completo."""
    mask = np.ones(crop.shape[:2], np.bool_)
    tensor = preprocess(crop, mask)
    assert tensor is not None
    return torch.from_numpy(tensor[0])


def augment(crop: np.ndarray[Any, Any], rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """Recorte aleatorio, brillo, contraste, saturación y espejo: lo que varía entre fotos del mismo auto."""
    height, width = crop.shape[:2]
    area = rng.uniform(*AREA_RANGE)
    aspect = math.exp(rng.uniform(-ASPECT_JITTER, ASPECT_JITTER))
    new_w = min(width, max(8, int(round(width * math.sqrt(area * aspect)))))
    new_h = min(height, max(8, int(round(height * math.sqrt(area / aspect)))))
    x = int(rng.integers(0, width - new_w + 1))
    y = int(rng.integers(0, height - new_h + 1))
    out = resize_square(crop[y : y + new_h, x : x + new_w]).astype(np.float32)
    brightness = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)
    contrast = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)
    saturation = 1.0 + rng.uniform(-COLOR_JITTER, COLOR_JITTER)
    mean = out.mean()
    out = (out - mean) * contrast + mean
    gray = out.mean(axis=2, keepdims=True)
    out = (out - gray) * saturation + gray
    out = np.clip(out * brightness, 0, 255).astype(np.uint8)
    if rng.random() < FLIP_PROBABILITY:
        out = np.ascontiguousarray(out[:, ::-1])
    return out


class CropSet(Dataset[tuple[torch.Tensor, int]]):
    """Recortes de auto en PNG. `train` aplica aumentos; si no, el preproceso exacto del servicio."""

    def __init__(
        self,
        root: Path,
        items: Sequence[Mapping[str, Any]],
        labels: Mapping[str, int],
        train: bool,
        crops_dir: str = "crops",
    ):
        self.root = root / crops_dir
        self.items = list(items)
        self.labels = labels
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.items[index]
        data = np.fromfile(str(self.root / item["archivo"]), np.uint8)
        crop = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if self.train:
            rng = np.random.default_rng((torch.initial_seed() + index) % 2**32)
            tensor = normalize(augment(crop, rng))
        else:
            tensor = eval_tensor(crop)
        return tensor, self.labels.get(item["auto"], -1)


class Embedder(nn.Module):
    """DINOv2 ViT-S/14: el token CLS de la última capa normalizado L2 (lo que lee el servicio)."""

    def __init__(self, backbone: Dinov2Model) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(pixel_values=pixels).last_hidden_state[:, 0], dim=-1)


class CosFace(nn.Module):
    """Un prototipo por auto; `logits = s · (cos θ − m · uno_caliente)`."""

    def __init__(self, prototypes: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(prototypes.clone())

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = embeddings @ F.normalize(self.weight, dim=-1).T
        logits = SCALE * (cosine - MARGIN * F.one_hot(labels, cosine.shape[1]))
        return F.cross_entropy(logits, labels)


def load_items(root: Path, index: str = "crops.json") -> dict[str, list[dict[str, Any]]]:
    items = json.loads((root / index).read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for item in items:
        out[item["particion"]].append(item)
    return out


@torch.no_grad()
def embed(model: Embedder, loader: DataLoader[Any], device: str) -> np.ndarray[Any, Any]:
    model.eval()
    parts = [model(pixels.to(device)).cpu().numpy() for pixels, _ in loader]
    return np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)


def validation_metrics(matrix: np.ndarray[Any, Any], pairs: Any) -> dict[str, Any]:
    """AUC y S4 al umbral que deja el U1 bajo 1 % (con cota del 95 %), sobre los pares de validación."""

    def cosines(index_pairs: Sequence[tuple[int, ...]]) -> np.ndarray[Any, Any]:
        if not index_pairs:
            return np.zeros(0)
        i, j = np.array([p[0] for p in index_pairs]), np.array([p[1] for p in index_pairs])
        return np.asarray((matrix[i] * matrix[j]).sum(axis=1), np.float64)

    same, different, hard = cosines(pairs.same), cosines(pairs.different), cosines(pairs.hard)
    result = calibrate(same, different)
    hard_s4 = (
        float((hard < result["umbral"]).mean()) if result.get("umbral") is not None and hard.size else None
    )
    return {
        "auc": auc(same, different),
        "auc_dificiles": auc(same, hard),
        "umbral": result.get("umbral"),
        "U1": result.get("U1"),
        "S4": result.get("S4"),
        "S4_dificiles": hard_s4,
    }


def free_gpu_mb() -> int:
    free, _ = torch.cuda.mem_get_info()
    return int(free / (1024 * 1024))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--index", default="crops_v2.json", help="índice de recortes dentro de --root")
    parser.add_argument("--crops-dir", default="crops-v2", help="carpeta de recortes dentro de --root")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=12, help="máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4, help="lote efectivo = lote × acumulación")
    parser.add_argument("--lr", type=float, default=2e-5, help="DINOv2 (los prototipos usan 50 veces más)")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3, help="épocas sin mejora antes de parar")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="recortes de entrenamiento (para probar)")
    parser.add_argument(
        "--resume", action="store_true", help="seguir desde `last.pt` (el de la última época)"
    )
    args = parser.parse_args(argv)
    for flag, value in (("--root", args.root), ("--out-dir", args.out_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root, out_dir = Path(args.root), Path(args.out_dir)
    if any(p.resolve().is_relative_to(REPO_ROOT) for p in (root, out_dir)):
        print("Los datos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("PyTorch no ve ninguna GPU. Usá --device cpu.", file=sys.stderr)
            return 1
        if free_gpu_mb() < MIN_FREE_GPU_MB:
            print(
                f"Hay {free_gpu_mb()} MB de video libres y hacen falta {MIN_FREE_GPU_MB}. "
                "Cerrá lo que use la GPU (nvidia-smi).",
                file=sys.stderr,
            )
            return 1
    seed_everything(args.seed)
    device = args.device
    parts = load_items(root, args.index)
    train_items = parts["train"][: args.limit] if args.limit else parts["train"]
    cars = sorted({item["auto"] for item in train_items})
    labels = {car: index for index, car in enumerate(cars)}
    print(
        f"entrenamiento: {len(train_items)} recortes de {len(cars)} autos · "
        f"validación: {len(parts['val'])} recortes"
    )

    model = Embedder(Dinov2Model.from_pretrained(HF_REPO, revision=HF_REVISION)).to(device)
    plain = DataLoader(
        CropSet(root, train_items, labels, train=False, crops_dir=args.crops_dir),
        batch_size=EVAL_BATCH,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        CropSet(root, parts["val"], labels, train=False, crops_dir=args.crops_dir),
        batch_size=EVAL_BATCH,
        num_workers=args.workers,
    )
    val_pairs = build_pairs(parts["val"], seed=args.seed)
    print(f"pares de validación: {len(val_pairs.same)} del mismo auto, {len(val_pairs.different)} distintos")

    # Prototipos: el promedio de los embeddings del modelo base de cada auto de entrenamiento
    base = embed(model, plain, device)
    prototypes = np.zeros((len(cars), base.shape[1]), np.float32)
    for index, item in enumerate(train_items):
        prototypes[labels[item["auto"]]] += base[index]
    head = CosFace(torch.from_numpy(prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True))).to(
        device
    )

    history: list[dict[str, Any]] = []
    last = out_dir / "last.pt"
    resumed = args.resume and last.is_file()
    if not resumed:
        baseline = validation_metrics(embed(model, val_loader, device), val_pairs)
        history.append({"epoca": 0, "perdida": None, **baseline})
        print(f"época 0 (sin ajustar): {baseline}")

    loader = DataLoader(
        CropSet(root, train_items, labels, train=True, crops_dir=args.crops_dir),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": head.parameters(), "lr": args.lr * 50, "weight_decay": 0.0},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * (len(loader) // args.grad_accum)

    def schedule(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, stale, first_epoch = float("-inf"), 0, 1
    if resumed:  # una corrida cortada (p. ej. por falta de memoria de la máquina) sigue donde quedó
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.backbone.load_state_dict(state["backbone"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, stale, first_epoch = (
            state["historial"],
            state["mejor"],
            state["sin_mejora"],
            state["epoca"] + 1,
        )
        print(f"se retoma desde la época {state['epoca']}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        started, running, steps = time.perf_counter(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (pixels, targets) in enumerate(loader, 1):
            loss = head(model(pixels.to(device)), targets.to(device))
            (loss / args.grad_accum).backward()
            running, steps = running + loss.item(), steps + 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = validation_metrics(embed(model, val_loader, device), val_pairs)
        row = {
            "epoca": epoch,
            "perdida": round(running / max(steps, 1), 4),
            "segundos": round(time.perf_counter() - started),
            **metrics,
        }
        history.append(row)
        print(f"época {epoch}: {row}")
        score = (metrics["S4"] if metrics["S4"] is not None else -1.0) + 1e-3 * (metrics["auc"] or 0.0)
        if score > best + EARLY_STOP_MIN_GAIN:
            best, stale = score, 0
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "epoca": epoch,
                    "metricas": metrics,
                    "semilla": args.seed,
                },
                out_dir / "best.pt",
            )
        else:
            stale += 1
        torch.save(
            {
                "backbone": model.backbone.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "historial": history,
                "mejor": best,
                "sin_mejora": stale,
                "epoca": epoch,
            },
            last,
        )
        if stale >= args.patience:
            print(f"parada temprana en la época {epoch} (sin mejora en {args.patience})")
            break
    summary = {
        "perdida": f"CosFace (s={SCALE}, m={MARGIN}) con un prototipo por auto",
        "autos_entrenamiento": len(cars),
        "recortes_entrenamiento": len(train_items),
        "lote": args.batch_size,
        "acumulacion": args.grad_accum,
        "lr": args.lr,
        "semilla": args.seed,
        "hf": {"repo": HF_REPO, "revision": HF_REVISION},
        "historial": history,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Listo: {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
