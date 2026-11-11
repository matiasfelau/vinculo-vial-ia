"""Negativas nuevas para la regla de presencia: fotos de Open Images sin auto que el detector nunca vio.

`presence.py` mide las "negativas" con fotos del subconjunto de Open Images (`openimages-detector`),
incluida la partición de entrenamiento del detector. Este script baja fotos **distintas**: de otras clases
(rostros, personas, edificios, perros, árboles, flores), de la partición `test` de Open Images V7 y que no
estén en ninguna partición del subconjunto del detector. Una foto vale como negativa si ninguna de sus
cajas es `Car`, `Vehicle registration plate` ni `Wheel` (la misma definición de `presence.py`).

Las fotos quedan fuera del repo (`CC BY 2.0`, no se redistribuyen). Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET = "openimages-detector"
EXCLUDED_LABELS = frozenset({"Car", "Vehicle registration plate", "Wheel"})
DEFAULT_CLASSES: Mapping[str, int] = {
    "Human face": 500,
    "Person": 350,
    "Building": 300,
    "Dog": 200,
    "Tree": 200,
    "Flower": 150,
}
MIN_SIDE_PX = 200  # una foto muy chica no dice nada del detector


def seen_files(data_dir: Path) -> set[str]:
    """Nombres de las fotos del subconjunto del detector, en las tres particiones."""
    seen: set[str] = set()
    for split in ("train", "valid", "test"):
        annotations = data_dir / SUBSET / split / "_annotations.coco.json"
        if annotations.is_file():
            coco = json.loads(annotations.read_text(encoding="utf-8"))
            seen.update(image["file_name"] for image in coco["images"])
    return seen


def is_negative(labels: Iterable[str]) -> bool:
    """La foto no tiene ninguna caja de `Car`, `Vehicle registration plate` ni `Wheel`."""
    return not (set(labels) & EXCLUDED_LABELS)


def pick(
    candidates: Sequence[tuple[str, Sequence[str]]], seen: set[str], taken: set[str], limit: int
) -> list[str]:
    """Nombres de fotos negativas, no vistas y no repetidas (hasta `limit`)."""
    chosen: list[str] = []
    for name, labels in candidates:
        if len(chosen) >= limit:
            break
        if name in seen or name in taken or not is_negative(labels):
            continue
        chosen.append(name)
    return chosen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--eval-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument(
        "--out-dir", type=Path, help="destino (por defecto <eval-dir>/presence-negativas-nuevas)"
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="partición de Open Images (la de `train` carga más de 1 GB de anotaciones en memoria)",
    )
    parser.add_argument("--seed", type=int, default=7, help="distinta de la del subconjunto (0)")
    parser.add_argument("--scale", type=float, default=1.0, help="multiplica la cantidad de fotos por clase")
    args = parser.parse_args(argv)
    for flag, value in (("--data-dir", args.data_dir), ("--eval-dir", args.eval_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    args.out_dir = args.out_dir or Path(args.eval_dir) / "presence-negativas-nuevas"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (args.data_dir, args.eval_dir, args.out_dir)):
        print("Las fotos no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    import fiftyone as fo  # pesado: recién acá
    import fiftyone.zoo as foz
    from PIL import Image

    fo.config.dataset_zoo_dir = str(Path(args.data_dir) / "fiftyone-cache")
    seen = seen_files(Path(args.data_dir))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    report: dict[str, dict[str, int]] = {}
    for oi_class, count in DEFAULT_CLASSES.items():
        wanted = round(count * args.scale)
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=args.split,
            label_types=["detections"],
            classes=[oi_class],
            max_samples=wanted * 2,  # sobra: algunas tienen autos o ya las vio el detector
            seed=args.seed,
            shuffle=True,
            dataset_name=f"vvi-negatives-{oi_class.lower().replace(' ', '-')}",
            drop_existing_dataset=True,
        )
        candidates = [
            (
                Path(sample.filepath).name,
                [d.label for d in (sample.ground_truth.detections if sample.ground_truth else [])],
            )
            for sample in dataset
        ]
        paths = {Path(sample.filepath).name: Path(sample.filepath) for sample in dataset}
        names = pick(candidates, seen, taken, wanted)
        kept = 0
        for name in names:
            with Image.open(paths[name]) as image:
                if min(image.size) < MIN_SIDE_PX:
                    continue
            shutil.copy2(paths[name], out / name)
            taken.add(name)
            kept += 1
        report[oi_class] = {"pedidas": wanted, "candidatas": len(candidates), "guardadas": kept}
        print(f"{oi_class}: {report[oi_class]}")
    (out / "negativas.json").write_text(
        json.dumps({"semilla": args.seed, "clases": report, "fotos": sorted(taken)}, indent=2),
        encoding="utf-8",
    )
    print(f"{len(taken)} negativas nuevas en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
