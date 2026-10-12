"""Lo que comparten los scripts de entrenamiento de `tools/training/`: GPU, entrenamiento y export.

Reglas del proyecto: FP32 por defecto (la GTX 1660 no tiene tensor cores), lote chico con acumulación de
gradiente, y nunca bajar de modelo ante la falta de memoria: primero se ajusta la memoria, después se
usa CPU o una GPU alquilada. Los datos y los resultados nunca van dentro del repositorio.
"""

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vvi_ia.adapters.models.onnx_session import OnnxSession

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = "_annotations.coco.json"
SUMMARY = "training_summary.json"
MB = 1024 * 1024
MIN_FREE_GPU_MB = 3500  # RF-DETR Small a 512 px con lote 2 en FP32 necesita unos 3 GB
OPSET = 17
CHECKPOINTS = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth")
ROW_TOLERANCE = 1e-2  # sobre cajas normalizadas y logits de magnitud ~10
MIN_MATCHING_ROWS = 0.95
CHECK_SEEDS = (0, 1, 2)
# Salida del ONNX → salida del módulo de PyTorch con la que se compara
REFERENCE_KEYS: Mapping[str, str] = {"dets": "pred_boxes", "labels": "pred_logits", "masks": "pred_masks"}


def inside_repo(*paths: Path) -> bool:
    return any(Path(path).resolve().is_relative_to(REPO_ROOT) for path in paths)


def add_training_options(
    parser: argparse.ArgumentParser,
    *,
    epochs: int = 20,
    batch_size: int = 2,
    grad_accum: int = 8,
    patience: int = 5,
) -> None:
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=epochs, help="máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=grad_accum,
        help="pasos de acumulación (lote efectivo = lote × acum)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=patience, help="épocas sin mejorar antes de parar")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="precisión mixta FP16 (apagada por defecto)")
    parser.add_argument("--resume", type=Path, help="checkpoint desde el que retomar")


def gpu_report(device: str) -> dict[str, Any]:
    if device != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "memoria_libre_mb": free // MB,
        "memoria_total_mb": total // MB,
    }


def check_device(device: str) -> tuple[dict[str, Any], str | None]:
    """Estado de la GPU antes de entrenar y, si no se puede empezar, el motivo."""
    if device != "cuda":
        return {}, None
    if not torch.cuda.is_available():
        return {}, "Pediste --device cuda pero PyTorch no ve ninguna GPU. Usá --device cpu."
    report = gpu_report("cuda")
    if report["memoria_libre_mb"] < MIN_FREE_GPU_MB:
        return report, (
            f"La GPU tiene solo {report['memoria_libre_mb']} MB libres y hacen falta {MIN_FREE_GPU_MB}. "
            "Cerrá los programas que la usan (juegos, navegadores, OBS) y volvé a probar."
        )
    return report, None


def train_model(
    model: Any,
    args: argparse.Namespace,
    *,
    name: str,
    dataset_dir: Path,
    out_dir: Path,
    classes: Sequence[str],
    gpu_before: Mapping[str, Any],
) -> dict[str, Any]:
    """Entrena `model` con las opciones de `add_training_options` y deja `training_summary.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    has_test = (dataset_dir / "test" / ANNOTATIONS).is_file()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(
        f"Entrenando {name}: {args.epochs} épocas como máximo, lote {args.batch_size}×{args.grad_accum}, "
        f"{'FP16' if args.amp else 'FP32'}, {args.device}"
    )
    started = time.perf_counter()
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        device=args.device,
        amp_dtype="fp16" if args.amp else None,
        num_workers=args.num_workers,
        early_stopping=True,
        early_stopping_patience=args.patience,
        run_test=has_test,
        tensorboard=False,
        resume=str(args.resume) if args.resume else None,
    )
    summary: dict[str, Any] = {
        "modelo": name,
        "clases": list(classes),
        "precision": "FP16 (amp)" if args.amp else "FP32",
        "epocas_maximas": args.epochs,
        "lote": args.batch_size,
        "acumulacion": args.grad_accum,
        "segundos": round(time.perf_counter() - started),
        "dispositivo": args.device,
        "versiones": {package: metadata.version(package) for package in ("rfdetr", "torch")},
        **gpu_before,
    }
    if args.device == "cuda":
        summary["pico_memoria_asignada_mb"] = torch.cuda.max_memory_allocated() // MB
        summary["pico_memoria_reservada_mb"] = torch.cuda.max_memory_reserved() // MB
    summary.update(read_metrics(out_dir))
    (out_dir / SUMMARY).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def read_metrics(run_dir: Path) -> dict[str, Any]:
    """Épocas hechas y las mejores métricas de validación, leídas de `metrics.csv`."""
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    validation = [row for row in rows if row.get("val/mAP_50")]
    if not validation:
        return {}
    # RF-DETR elige el mejor checkpoint por mAP50:95, no por mAP50
    best = max(validation, key=lambda row: float(row["val/mAP_50_95"]))

    def number(row: Mapping[str, str], key: str) -> float | None:
        return round(float(row[key]), 4) if row.get(key) else None

    summary: dict[str, Any] = {
        "epocas_hechas": len(validation),
        "mejor_epoca": int(best["epoch"]),
        "validacion_mejor": {
            "mAP50": number(best, "val/mAP_50"),
            "mAP50_95": number(best, "val/mAP_50_95"),
            "recall": number(best, "val/recall"),
            "precision": number(best, "val/precision"),
        },
    }
    test = [row for row in rows if row.get("test/mAP_50")]
    if test:
        summary["prueba"] = {
            "mAP50": number(test[-1], "test/mAP_50"),
            "mAP50_95": number(test[-1], "test/mAP_50_95"),
            "recall": number(test[-1], "test/recall"),
            "precision": number(test[-1], "test/precision"),
        }
    return summary


def find_checkpoint(run_dir: Path) -> Path | None:
    return next((run_dir / name for name in CHECKPOINTS if (run_dir / name).is_file()), None)


def matching_rows(actual: Sequence[Any], expected: Sequence[Any], tolerance: float = ROW_TOLERANCE) -> float:
    """Fracción de consultas del ONNX que tienen una igual en PyTorch (todas las salidas juntas).

    El ONNX devuelve las consultas ordenadas por la selección de dos etapas, así que no se pueden
    comparar posición por posición: cada fila del ONNX se busca entre las de PyTorch. Un puñado de filas
    del borde de la selección puede cambiar cuando los puntajes están casi empatados, algo normal en un
    modelo poco entrenado; por eso se pide una fracción y no todas.
    """
    rows_actual = np.concatenate([part[0].reshape(part.shape[1], -1) for part in actual], axis=1)
    rows_expected = np.concatenate([part[0].reshape(part.shape[1], -1) for part in expected], axis=1)
    best = np.empty(len(rows_actual))
    for index, row in enumerate(rows_actual):
        best[index] = np.abs(rows_expected - row).max(axis=1).min()
    return float((best <= tolerance).mean())


def export_checkpoint(
    model_class: Any,
    checkpoint: Path,
    *,
    classes: Sequence[str],
    outputs: Sequence[str],
    target: Path,
    force: bool,
    model_name: str,
) -> int:
    """Exporta un checkpoint a ONNX en `target` si pasa los controles. Devuelve el código de salida."""
    if target.exists() and not force:
        print(f"Ya existe {target}. Usá --force para pisarlo.", file=sys.stderr)
        return 1
    model = model_class(pretrain_weights=str(checkpoint), device="cpu")
    if list(model.class_names) != list(classes):
        print(f"El checkpoint no tiene las clases esperadas: {list(classes)}.", file=sys.stderr)
        return 1
    resolution = int(model.model_config.resolution)
    print(
        f"Exportando {checkpoint.name} a ONNX (opset {OPSET}, FP32, "
        f"entrada fija {resolution}×{resolution})..."
    )
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(model.export(output_dir=tmp, opset_version=OPSET, verbose=False)).read_bytes()
    session = OnnxSession(data, intra_op_threads=0)
    if tuple(session.output_names) != tuple(outputs):
        print(f"Salidas inesperadas del ONNX: {session.output_names}", file=sys.stderr)
        return 1
    worst = 1.0
    torch_model = model.model.model.eval()
    for seed in CHECK_SEEDS:
        image = torch.randn(1, 3, resolution, resolution, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            reference = torch_model(image)
        actual = session.run(None, {session.input_names[0]: image.numpy()})
        expected = [reference[REFERENCE_KEYS[name]].numpy() for name in outputs]
        worst = min(worst, matching_rows(actual, expected))
    print(f"Consultas del ONNX (CPU) que coinciden con PyTorch, en el peor caso: {worst:.1%}")
    if worst < MIN_MATCHING_ROWS:
        print("El ONNX no reproduce a PyTorch: no se guardó. No lo uses.", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"\nListo: {target} ({len(data) / 1e6:.1f} MB), columnas de clases: {dict(enumerate(classes))}")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")
    print(f'Copiá ese valor en el campo "sha256" de {model_name} en models/manifest.json.')
    return 0
