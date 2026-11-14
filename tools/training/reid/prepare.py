"""Arma los recortes de auto de 3DRealCar tal como los ve el embedder del servicio, y la partición.

Para cada foto bajada con `fetch.py pull` hace lo que hace la cascada sin segmentador: reduce la foto a
`MODEL_MAX_SIDE_PX` (1024), corre el detector (`rfdetr_openimages`, con su confianza mínima), se queda con la
caja
`car` más grande y recorta esa caja (el embedder la achica a 224 x 224 con el mismo preproceso del servicio).
Solo usa el RGB. Guarda cada recorte como PNG (sin pérdida) en `crops/<auto>/` y escribe `crops.json` con el
ángulo de vista de cada foto (a partir de la pose de la cámara), las anotaciones del auto y su partición
(agrupada por auto, estratificada por luz, tipo y color, con semilla fija).

Los datos quedan fuera del repositorio. Uso: ver `tools/training/reid/README.md`.
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dataset import Car, azimuths, group_split, load_frozen, save_split
from fetch import LIGHTING, is_complete, load_annotations

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import build_models
from vvi_ia.config import Settings
from vvi_ia.domain.model import BoxLabel
from vvi_ia.main import adapter_params
from vvi_ia.service.validate_evidence import reduce_for_models

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_MAX_SIDE_PX = 1024  # el del servicio
DETECTOR = "rfdetr_openimages"

_worker: dict[str, Any] = {}


def init_worker(manifest: str, models_dir: str, threads: int) -> None:
    settings = Settings.model_construct(plate_tiling=False, face_tiling=False)
    _worker["models"] = build_models(
        {"detector": DETECTOR}, Path(manifest), Path(models_dir), threads, adapter_params(settings)
    )
    _worker["decoder"] = CvImageDecoder()


def largest_car_crop(pixels: np.ndarray[Any, Any], models: Any) -> tuple[np.ndarray[Any, Any], float] | None:
    """Recorte de la caja `car` más grande en la foto reducida (como el servicio), con su confianza."""
    view = reduce_for_models(pixels, MODEL_MAX_SIDE_PX)
    shown = pixels if view is None else view
    boxes = [b for b in (models.vehicle.detect(shown) or ()) if b.label == BoxLabel.CAR]
    if not boxes:
        return None
    box = max(boxes, key=lambda b: b.area)
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(shown.shape[1], round(box.x2)), min(shown.shape[0], round(box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return shown[y1:y2, x1:x2].copy(), box.conf


def process_car(job: tuple[str, str, str]) -> dict[str, Any]:
    """Recortes de todas las fotos bajadas de un auto."""
    car, raw_root, crop_root = job
    models, decoder = _worker["models"], _worker["decoder"]
    folder = Path(crop_root) / car
    folder.mkdir(parents=True, exist_ok=True)
    frames, dropped = [], 0
    for jpg in sorted((Path(raw_root) / car).glob("frame_*.jpg")):
        number = int(jpg.stem.split("_")[1])
        pose_file = jpg.with_suffix(".json")
        if not (
            is_complete(jpg) and is_complete(pose_file)
        ):  # una bajada interrumpida deja archivos a medias
            continue
        pose = json.loads(pose_file.read_text(encoding="utf-8"))["cameraPoseARFrame"]
        target = folder / f"{number:05d}.png"
        if target.is_file():  # ya recortada en una corrida anterior (la confianza no se guardó)
            crop_size = cv2.imdecode(np.fromfile(str(target), np.uint8), cv2.IMREAD_COLOR).shape
            frames.append(
                {
                    "frame": number,
                    "pose": pose,
                    "file": f"{car}/{number:05d}.png",
                    "ancho": int(crop_size[1]),
                    "alto": int(crop_size[0]),
                    "confianza": None,
                }
            )
            continue
        found = largest_car_crop(decoder.decode(jpg.read_bytes()), models)
        if found is None:
            dropped += 1
            frames.append({"frame": number, "pose": pose, "file": None})
            continue
        crop, conf = found
        cv2.imencode(".png", crop)[1].tofile(str(target))
        frames.append(
            {
                "frame": number,
                "pose": pose,
                "file": f"{car}/{number:05d}.png",
                "ancho": int(crop.shape[1]),
                "alto": int(crop.shape[0]),
                "confianza": round(float(conf), 4),
            }
        )
    return {"auto": car, "fotos": frames, "sin_auto": dropped}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument(
        "--workers", type=int, default=3, help="procesos (cada uno con su sesión del detector)"
    )
    parser.add_argument("--threads", type=int, default=2, help="hilos de ONNX Runtime por proceso")
    parser.add_argument("--seed", type=int, default=0, help="semilla de la partición")
    parser.add_argument("--limit", type=int, help="solo los primeros N autos (para probar)")
    args = parser.parse_args(argv)
    for flag, value in (("--root", args.root), ("--models-dir", args.models_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if root.resolve().is_relative_to(REPO_ROOT):
        print("Los datos no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    annotations = load_annotations(root / "annotations.json")
    names = [car for car in sorted(selection) if (root / "raw" / car).is_dir()]
    names = names[: args.limit] if args.limit else names
    jobs = [(car, str(root / "raw"), str(root / "crops")) for car in names]
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(args.manifest), str(args.models_dir), args.threads),
    ) as pool:
        for count, result in enumerate(pool.map(process_car, jobs), 1):
            results[result["auto"]] = result
            if count % 50 == 0:
                print(f"{count}/{len(jobs)} autos")
    cars = [
        Car(name, a["marca"], a["modelo"], a["luz"], a["tipo"], a["color"])
        for name in results
        if any(f["file"] for f in results[name]["fotos"]) and (a := annotations.get(name))
    ]
    # Los autos ya particionados no se mueven (`split.json`): solo se sortean los nuevos
    split = group_split(cars, seed=args.seed, frozen=load_frozen(root))
    save_split(root, {**load_frozen(root), **split})
    out: list[dict[str, Any]] = []
    for car in cars:
        frames = results[car.name]["fotos"]
        if not frames:
            continue
        angles = azimuths([f["pose"] for f in frames])
        for frame, angle in zip(frames, angles, strict=True):
            if frame["file"] is None:
                continue
            out.append(
                {
                    "auto": car.name,
                    "marca": car.brand,
                    "modelo": car.model,
                    "luz": car.lighting,
                    "tipo": car.kind,
                    "color": car.color,
                    "particion": split[car.name],
                    "foto": frame["frame"],
                    "archivo": frame["file"],
                    "azimut": round(angle, 2),
                    "ancho": frame["ancho"],
                    "alto": frame["alto"],
                    "confianza": frame["confianza"],
                }
            )
    (root / "crops.json").write_text(json.dumps(out), encoding="utf-8")
    dropped = sum(r["sin_auto"] for r in results.values())
    total = dropped + len(out)
    per_split = Counter(item["particion"] for item in out)
    cars_per_split = Counter(split[car.name] for car in cars)
    by_light = Counter((split[car.name], car.lighting) for car in cars)
    print(f"{len(out)} recortes de {len(cars)} autos; {dropped} de {total} fotos sin auto detectado")
    print(f"recortes por partición: {dict(per_split)} · autos: {dict(cars_per_split)}")
    for part in ("train", "val", "test"):
        print(f"  {part}: " + " · ".join(f"{light} {by_light[(part, light)]}" for light in LIGHTING))
    return 0


if __name__ == "__main__":
    sys.exit(main())
