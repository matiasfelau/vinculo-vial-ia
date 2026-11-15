"""Re-ID con 3DRealCar: DINOv2 sin ajustar contra el ajustado, por luz, ángulo y pares difíciles.

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

Todo sale en cifras; los datos quedan fuera del repo.
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
from vvi_ia.service.validate_evidence import reduce_for_models  # noqa: E402

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument(
        "--models", nargs="+", default=[BASELINE], help=f"embedders a medir (p. ej. {BASELINE} {TUNED})"
    )
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
    settings = Settings.model_construct(plate_tiling=False)
    report: dict[str, Any] = {"REID_T_del_servicio": SERVICE_REID_T, "max_U1": MAX_U1, "modelos": {}}
    built: dict[str, Any] = {}
    try:
        for name in args.models:
            built[name] = build_models(
                {"embedder": name},
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


if __name__ == "__main__":
    sys.exit(main())
