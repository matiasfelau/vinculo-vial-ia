"""Re-ID con 3DRealCar: DINOv2 sin ajustar contra el ajustado, por luz, ángulo y pares difíciles (y VeRi-776).

Para cada modelo (puerto `embedder` del servicio, ONNX en CPU) embebe los recortes de auto que arma
`tools/training/reid/prepare.py` (la caja `car` del detector, sin segmentador) de las particiones `val` y
`test`
y compara por coseno, con la misma regla que el servicio (otro auto si cos < `REID_T`):

- **U1**: pares del mismo auto marcados como distintos. **S4**: pares de autos distintos detectados.
- Pares del mismo auto con una separación angular mínima de 30° (por la pose de la cámara) y por rango de
  ángulo.
- Pares de autos distintos: todos, y difíciles (misma marca y mismo color).
- Por condición de luz: estándar, reflectiva y oscura.
- El umbral se calibra sobre `val` (el mayor con U1 acotado a 1 % al 95 %) y se reporta sobre `test`.

Con `--veri-dir` mide también a los dos modelos sobre VeRi-776 (recortes de cámaras de vigilancia) con el
mismo
recorte del detector y el umbral calibrado en 3DRealCar. Todo sale en cifras; los datos quedan fuera del repo.
Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "reid"))

from dataset import ANGLE_BINS, PairSet, angle_bin, build_pairs  # noqa: E402
from metrics import at_threshold, auc, calibrate, histogram  # noqa: E402

from vvi_ia.adapters.imaging import CvImageDecoder  # noqa: E402
from vvi_ia.adapters.models.registry import ModelSetupError, build_models  # noqa: E402
from vvi_ia.config import Settings  # noqa: E402
from vvi_ia.domain.model import BoxLabel  # noqa: E402
from vvi_ia.main import adapter_params  # noqa: E402
from vvi_ia.service.ports import ImageDecodeError  # noqa: E402
from vvi_ia.service.validate_evidence import reduce_for_models  # noqa: E402

DETECTOR = "rfdetr_openimages"
BASELINE = "dinov2_vits14"
TUNED = "dinov2_vits14_reid"
SERVICE_REID_T = Settings.model_fields["reid_t"].default  # el umbral que hoy usa el servicio
LIGHTING = ("standard", "reflective", "dark")
MAX_U1 = 0.01
MODEL_MAX_SIDE_PX = 1024
Matrix = NDArray[np.float32]


def embed_crops(models: Any, root: Path, items: Sequence[Mapping[str, Any]], workers: int) -> Matrix:
    """Embedding de cada recorte con el adaptador del servicio (el recorte completo es la máscara)."""
    decoder = CvImageDecoder()

    def one(item: Mapping[str, Any]) -> NDArray[np.float32]:
        crop = cv2.imdecode(np.fromfile(str(root / "crops" / item["archivo"]), np.uint8), cv2.IMREAD_COLOR)
        embedding = models.embedder.embed(crop, np.ones(crop.shape[:2], np.bool_))
        if embedding is None:
            raise RuntimeError("el embedder no devolvió un vector")
        return np.asarray(embedding, np.float32)

    del decoder
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(one, items)))


def cosines(matrix: Matrix, pairs: Sequence[tuple[int, ...]]) -> NDArray[np.float64]:
    if not pairs:
        return np.zeros(0)
    i, j = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    return np.asarray((matrix[i] * matrix[j]).sum(axis=1), np.float64)


def summarize(
    same: NDArray[np.float64], different: NDArray[np.float64], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """AUC y U1 / S4 a cada umbral para un conjunto de pares."""
    return {
        "auc": auc(same, different),
        **{name: at_threshold(same, different, t) for name, t in thresholds.items()},
    }


def car_lighting(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(item["luz"]) for item in items]


def evaluate_split(
    matrix: Matrix, items: Sequence[Mapping[str, Any]], pairs: PairSet, thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Todas las tablas de una partición con un modelo: global, difíciles, por luz y por ángulo."""
    lighting = car_lighting(items)
    same = cosines(matrix, pairs.same)
    different = cosines(matrix, pairs.different)
    hard = cosines(matrix, pairs.hard)
    same_light = np.array([lighting[i] for i, _, _ in pairs.same])
    different_light = np.array(
        [lighting[i] if lighting[i] == lighting[j] else "mixta" for i, j in pairs.different]
    )
    hard_light = np.array([lighting[i] if lighting[i] == lighting[j] else "mixta" for i, j in pairs.hard])
    angles = np.array([angle for _, _, angle in pairs.same])
    bins = np.array([angle_bin(a) or "" for a in angles])
    out: dict[str, Any] = {
        "todos": summarize(same, different, thresholds),
        "dificiles": summarize(same, hard, thresholds),
        "por_luz": {},
        "por_luz_dificiles": {},
        "por_angulo": {},
        "similitudes_percentiles": {
            "mismo_auto": [round(float(v), 4) for v in np.percentile(same, [1, 5, 50, 95])],
            "autos_distintos": [round(float(v), 4) for v in np.percentile(different, [5, 50, 95, 99])],
            "dificiles": [round(float(v), 4) for v in np.percentile(hard, [5, 50, 95, 99])]
            if hard.size
            else None,
        },
        "histograma": {
            "mismo_auto": histogram(same),
            "autos_distintos": histogram(different),
            "dificiles": histogram(hard),
        },
    }
    for light in LIGHTING:
        out["por_luz"][light] = summarize(
            same[same_light == light], different[different_light == light], thresholds
        )
        out["por_luz_dificiles"][light] = summarize(
            same[same_light == light], hard[hard_light == light], thresholds
        )
    for low, _ in ANGLE_BINS:
        label = angle_bin(low) or ""
        select = bins == label
        out["por_angulo"][label] = {
            name: {
                k: v
                for k, v in at_threshold(same[select], different, t).items()
                if k.startswith("U1") or k == "pares_mismo_auto"
            }
            for name, t in thresholds.items()
        }
    return out


def detector_crop(pixels: NDArray[np.uint8], models: Any) -> NDArray[np.uint8] | None:
    """Recorte de la caja `car` más grande en la foto reducida (como la cascada sin segmentador)."""
    view = reduce_for_models(pixels, MODEL_MAX_SIDE_PX)
    shown = pixels if view is None else view
    boxes = [b for b in (models.vehicle.detect(shown) or ()) if b.label == BoxLabel.CAR]
    if not boxes:
        return None
    box = max(boxes, key=lambda b: b.area)
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(shown.shape[1], round(box.x2)), min(shown.shape[0], round(box.y2))
    return None if x2 <= x1 or y2 <= y1 else np.asarray(shown[y1:y2, x1:x2])


def veri_crops(
    veri: Path, per_track: int, seed: int, limit: int | None, models: Any, workers: int
) -> tuple[list[Any], list[NDArray[np.uint8]], dict[str, int]]:
    """Recortes de VeRi (como en `reid_veri.py`) pasados por el detector; descarta los sin auto."""
    from reid_veri import load_labels, read_names, select_crops  # noqa: PLC0415

    labels = load_labels(veri / "test_label.xml")
    crops = select_crops(
        labels,
        read_names(veri / "name_test.txt"),
        read_names(veri / "name_query.txt"),
        per_track,
        seed,
        limit,
    )
    decoder = CvImageDecoder()

    def one(crop: Any) -> NDArray[np.uint8] | str:
        try:
            pixels = decoder.decode((veri / "image_test" / crop.name).read_bytes())
        except (OSError, ImageDecodeError):
            return "ilegible"
        found = detector_crop(pixels, models)
        return "sin auto detectado" if found is None else found

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, crops))
    kept = [(c, r) for c, r in zip(crops, results, strict=True) if not isinstance(r, str)]
    discarded: dict[str, int] = {}
    for result in results:
        if isinstance(result, str):
            discarded[result] = discarded.get(result, 0) + 1
    return [c for c, _ in kept], [r for _, r in kept if not isinstance(r, str)], discarded


def veri_report(
    crops: Sequence[Any],
    images: Sequence[NDArray[np.uint8]],
    model: Any,
    thresholds: Mapping[str, float],
    workers: int,
) -> dict[str, Any]:
    from reid_veri import pair_similarities  # noqa: PLC0415

    def one(image: NDArray[np.uint8]) -> NDArray[np.float32]:
        embedding = model.embedder.embed(image, np.ones(image.shape[:2], np.bool_))
        if embedding is None:
            raise RuntimeError("el embedder no devolvió un vector")
        return np.asarray(embedding, np.float32)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        matrix = np.stack(list(pool.map(one, images)))
    same, different, hard, _ = pair_similarities(list(crops), matrix)
    same, different, hard = (a.astype(np.float64) for a in (same, different, hard))
    stride = max(1, different.size // 2_000_000)
    own = calibrate(same, different[::stride], MAX_U1)
    thresholds = (
        {**thresholds, "calibrado_en_VeRi": float(own["umbral"])}
        if own.get("umbral") is not None
        else dict(thresholds)
    )
    return {
        "pares": {
            "mismo_auto": int(same.size),
            "autos_distintos": int(different.size),
            "dificiles": int(hard.size),
        },
        "todos": summarize(same, different[::stride], thresholds),
        "dificiles": summarize(same, hard[:: max(1, hard.size // 2_000_000)], thresholds),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument(
        "--models", nargs="+", default=[BASELINE], help=f"embedders a medir (p. ej. {BASELINE} {TUNED})"
    )
    parser.add_argument(
        "--veri-dir", type=Path, default=os.environ.get("VERI_DIR"), help="VeRi-776 (VERI_DIR)"
    )
    parser.add_argument("--per-track", type=int, default=3, help="recortes de VeRi por (auto, cámara)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="recortes por partición (para probar)")
    args = parser.parse_args(argv)
    for flag, value in (
        ("--root", args.root),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root, out_dir = Path(args.root), Path(args.out_dir)
    if any(p.resolve().is_relative_to(REPO_ROOT) for p in (root, out_dir)):
        print("Los datos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    items = json.loads((root / "crops.json").read_text(encoding="utf-8"))
    parts = {name: [i for i in items if i["particion"] == name] for name in ("val", "test")}
    if args.limit:
        parts = {name: rows[: args.limit] for name, rows in parts.items()}
    pairs = {name: build_pairs(rows, seed=args.seed) for name, rows in parts.items()}
    print(
        {
            name: {
                "recortes": len(parts[name]),
                "mismo_auto": len(p.same),
                "distintos": len(p.different),
                "dificiles": len(p.hard),
            }
            for name, p in pairs.items()
        },
        flush=True,
    )
    settings = Settings.model_construct(plate_tiling=False, face_tiling=False)
    report: dict[str, Any] = {"REID_T_del_servicio": SERVICE_REID_T, "max_U1": MAX_U1, "modelos": {}}
    built: dict[str, Any] = {}
    try:
        for name in args.models:
            built[name] = build_models(
                {"embedder": name, "detector": DETECTOR} if args.veri_dir else {"embedder": name},
                args.manifest,
                args.models_dir,
                params=adapter_params(settings),
            )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    calibrated: dict[str, float | None] = {}
    for name, models in built.items():
        matrices = {part: embed_crops(models, root, parts[part], args.workers) for part in parts}
        val_same = cosines(matrices["val"], pairs["val"].same)
        val_different = cosines(matrices["val"], pairs["val"].different)
        calibration = calibrate(val_same, val_different, MAX_U1)
        calibrated[name] = calibration.get("umbral")
        thresholds = {"REID_T_del_servicio": float(SERVICE_REID_T)}
        if calibrated[name] is not None:
            thresholds["calibrado_en_val"] = float(calibrated[name])
        report["modelos"][name] = {
            "version": models.versions()["embedder"],
            "calibracion_en_val": calibration,
            "test": evaluate_split(matrices["test"], parts["test"], pairs["test"], thresholds),
            "val": evaluate_split(matrices["val"], parts["val"], pairs["val"], thresholds),
        }
        todos = report["modelos"][name]["test"]["todos"]
        print(f"[{name}] AUC test {todos['auc']:.4f} · umbral calibrado {calibrated[name]}", flush=True)
    if args.veri_dir:
        veri = Path(args.veri_dir)
        first = next(iter(built.values()))
        crops, images, discarded = veri_crops(
            veri, args.per_track, args.seed, args.limit, first, args.workers
        )
        report["veri"] = {"recortes_usados": len(crops), "descartados": discarded, "modelos": {}}
        for name, models in built.items():
            thresholds = {"REID_T_del_servicio": float(SERVICE_REID_T)}
            if calibrated[name] is not None:
                thresholds["calibrado_en_3DRealCar"] = float(calibrated[name])
            report["veri"]["modelos"][name] = veri_report(crops, images, models, thresholds, args.workers)
            print(f"[VeRi {name}] AUC {report['veri']['modelos'][name]['todos']['auc']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reid_3drealcar_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_tables(report)
    return 0


def pct(value: float | None) -> str:
    return "n/d" if value is None else f"{100 * value:.1f} %"


def print_tables(report: Mapping[str, Any]) -> None:
    for name, entry in report["modelos"].items():
        print(f"\n== {name} ({entry['version']}) — test ==")
        print(f"calibración en val: {entry['calibracion_en_val']}")
        test = entry["test"]
        for key, label in (
            ("todos", "todos"),
            ("dificiles", "difíciles"),
            *((f"luz:{light}", light) for light in LIGHTING),
        ):
            block = (
                test["todos"]
                if key == "todos"
                else test["dificiles"]
                if key == "dificiles"
                else test["por_luz"][key[4:]]
            )
            row = f"{label:<11} AUC {block['auc'] if block['auc'] is None else round(block['auc'], 4)}"
            for threshold_name in [k for k in block if k != "auc"]:
                cell = block[threshold_name]
                row += (
                    f" | {threshold_name}: U1 {pct(cell['U1'])} (cota {pct(cell['U1_cota_95'])})"
                    f" S4 {pct(cell['S4'])}"
                )
            print(row)
        for label, cell in test["por_angulo"].items():
            print(
                f"  ángulo {label}°: "
                + " | ".join(f"{k}: U1 {pct(v['U1'])} de {v['pares_mismo_auto']}" for k, v in cell.items())
            )
    if "veri" in report:
        print(
            "\n"
            f"== VeRi-776 ({report['veri']['recortes_usados']} recortes; "
            f"descartados {report['veri']['descartados']}) =="
        )
        for name, block in report["veri"]["modelos"].items():
            print(name, block["pares"])
            for scope in ("todos", "dificiles"):
                row = f"  {scope:<10} AUC {block[scope]['auc']}"
                for threshold_name in [k for k in block[scope] if k != "auc"]:
                    cell = block[scope][threshold_name]
                    row += f" | {threshold_name}: U1 {pct(cell['U1'])} S4 {pct(cell['S4'])}"
                print(row)


if __name__ == "__main__":
    sys.exit(main())
