"""Re-ID con 3DRealCar y pares parciales: AUC, U1 y S4 por tipo de par, luz y pares difíciles (y VeRi-776).

Lo que decide R2 en el servicio no es solo "auto entero contra auto entero": la foto GENERAL se compara
contra la de PLATE (trompa o cola) y contra la de DETAIL (una zona del auto). Esta medición usa tres
tipos de pares sobre los recortes de `tools/training/reid/prepare.py` (vehículo dominante del detector
v2, 0,35 de confianza, sin segmentador) y los simulados por `tools/training/reid/partial.py`:

- **entero**: dos vistas del auto entero (separadas por al menos 30°).
- **general_patente**: una vista entera contra un recorte de trompa o cola con la patente al 8-15 % del ancho.
- **general_detalle**: una vista entera contra una zona del auto.

El umbral se calibra **en validación, con los pares de los tres tipos juntos** (el mayor umbral con la
cota del 95 % de U1 <= 1 %) y se reporta en prueba. Se mide el modelo ajustado contra DINOv2 sin ajustar
con los mismos recortes, y con `--veri-dir`, VeRi-776 como referencia (recortes de cámaras de
vigilancia, con el umbral calibrado acá). R2 solo rechaza con certeza: ante la duda, GREEN. Todo sale en
cifras; los datos quedan fuera del repo. Uso: ver `tools/eval/README.md`.
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

from dataset import ANGLE_BINS, angle_bin, build_pairs  # noqa: E402
from metrics import at_threshold, auc, calibrate  # noqa: E402
from partial import cross_different_pairs, cross_same_pairs  # noqa: E402

from vvi_ia.adapters.models.registry import ModelSetupError, build_models  # noqa: E402
from vvi_ia.config import Settings  # noqa: E402
from vvi_ia.main import adapter_params  # noqa: E402

DETECTOR = "rfdetr_openimages_v2"
BASELINE = "dinov2_vits14"
TUNED = "dinov2_vits14_reid_v2"
SERVICE_REID_T = Settings.model_fields["reid_t"].default  # el umbral que hoy usa el servicio
LIGHTING = ("standard", "reflective", "dark")
TYPES = ("entero", "general_patente", "general_detalle")
MAX_U1 = 0.01
DIFFERENT_LIMIT = 100_000
HARD_LIMIT = 30_000
Matrix = NDArray[np.float32]


class PairType:
    """Los pares de un tipo en una partición: cosenos del mismo auto, de distintos y de los difíciles."""

    def __init__(
        self,
        same: NDArray[np.float64],
        different: NDArray[np.float64],
        hard: NDArray[np.float64],
        same_light: NDArray[Any],
        different_light: NDArray[Any],
        hard_light: NDArray[Any],
        angles: NDArray[np.float64],
    ) -> None:
        self.same, self.different, self.hard = same, different, hard
        self.same_light, self.different_light, self.hard_light = same_light, different_light, hard_light
        self.angles = angles


def embed_files(models: Any, folder: Path, names: Sequence[str], workers: int) -> Matrix:
    """Embedding de cada recorte con el adaptador del servicio (el recorte completo es la máscara)."""

    def one(name: str) -> NDArray[np.float32]:
        crop = cv2.imdecode(np.fromfile(str(folder / name), np.uint8), cv2.IMREAD_COLOR)
        embedding = models.embedder.embed(crop, np.ones(crop.shape[:2], np.bool_))
        if embedding is None:
            raise RuntimeError("el embedder no devolvió un vector")
        return np.asarray(embedding, np.float32)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(one, names)))


def dots(a: Matrix, b: Matrix, pairs: Sequence[tuple[int, ...]]) -> NDArray[np.float64]:
    if not pairs:
        return np.zeros(0)
    i, j = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    return np.asarray((a[i] * b[j]).sum(axis=1), np.float64)


def light_of_pairs(
    lights_a: Sequence[str], lights_b: Sequence[str], pairs: Sequence[tuple[int, ...]]
) -> NDArray[Any]:
    """Luz de cada par; si los dos autos tienen luz distinta, `mixta`."""
    return np.array([lights_a[p[0]] if lights_a[p[0]] == lights_b[p[1]] else "mixta" for p in pairs])


def build_types(
    whole: Sequence[Mapping[str, Any]],
    partial: Mapping[str, Sequence[Mapping[str, Any]]],
    whole_matrix: Matrix,
    partial_matrix: Mapping[str, Matrix],
    seed: int,
) -> dict[str, PairType]:
    """Los tres tipos de pares de una partición."""
    pairs = build_pairs(whole, seed=seed, different_limit=DIFFERENT_LIMIT, hard_limit=HARD_LIMIT)
    lights = [str(w["luz"]) for w in whole]
    out: dict[str, PairType] = {
        "entero": PairType(
            dots(whole_matrix, whole_matrix, pairs.same),
            dots(whole_matrix, whole_matrix, pairs.different),
            dots(whole_matrix, whole_matrix, pairs.hard),
            np.array([lights[i] for i, _, _ in pairs.same]),
            light_of_pairs(lights, lights, pairs.different),
            light_of_pairs(lights, lights, pairs.hard),
            np.array([angle for _, _, angle in pairs.same]),
        )
    }
    cars_w = [str(w["auto"]) for w in whole]
    brand_color = {str(w["auto"]): (str(w["marca"]), str(w["color"])) for w in whole}
    for kind, name in (("patente", "general_patente"), ("detalle", "general_detalle")):
        part = list(partial.get(kind, []))
        if not part:
            continue
        brand_color.update({str(p["auto"]): (str(p["marca"]), str(p["color"])) for p in part})
        cars_p = [str(p["auto"]) for p in part]
        same = cross_same_pairs(
            [(str(w["auto"]), float(w["azimut"])) for w in whole],
            [(str(p["auto"]), float(p["azimut"])) for p in part],
        )
        different = cross_different_pairs(cars_w, cars_p, brand_color, DIFFERENT_LIMIT, seed)
        hard = cross_different_pairs(cars_w, cars_p, brand_color, HARD_LIMIT, seed, hard=True)
        lights_p = [str(p["luz"]) for p in part]
        out[name] = PairType(
            dots(whole_matrix, partial_matrix[kind], [(i, j) for i, j, _ in same]),
            dots(whole_matrix, partial_matrix[kind], different),
            dots(whole_matrix, partial_matrix[kind], hard),
            np.array([lights[i] for i, _, _ in same]),
            light_of_pairs(lights, lights_p, different),
            light_of_pairs(lights, lights_p, hard),
            np.array([angle for _, _, angle in same]),
        )
    return out


def pooled(types: Mapping[str, PairType]) -> PairType:
    """Todos los pares de todos los tipos juntos."""
    parts = list(types.values())
    cat = np.concatenate
    return PairType(
        cat([p.same for p in parts]),
        cat([p.different for p in parts]),
        cat([p.hard for p in parts]),
        cat([p.same_light for p in parts]),
        cat([p.different_light for p in parts]),
        cat([p.hard_light for p in parts]),
        cat([p.angles for p in parts]),
    )


def summarize(
    same: NDArray[np.float64], different: NDArray[np.float64], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """AUC y U1 / S4 a cada umbral para un conjunto de pares."""
    return {
        "auc": auc(same, different),
        **{n: at_threshold(same, different, t) for n, t in thresholds.items()},
    }


def block(pair: PairType, thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Todas las cifras de un conjunto de pares: total, difíciles, por luz y por ángulo."""
    out: dict[str, Any] = {
        "pares": {
            "mismo_auto": int(pair.same.size),
            "distintos": int(pair.different.size),
            "dificiles": int(pair.hard.size),
        },
        "todos": summarize(pair.same, pair.different, thresholds),
        "dificiles": summarize(pair.same, pair.hard, thresholds),
        "por_luz": {},
        "por_luz_dificiles": {},
        "por_angulo": {},
    }
    for light in LIGHTING:
        out["por_luz"][light] = summarize(
            pair.same[pair.same_light == light], pair.different[pair.different_light == light], thresholds
        )
        out["por_luz_dificiles"][light] = summarize(
            pair.same[pair.same_light == light], pair.hard[pair.hard_light == light], thresholds
        )
    bins = np.array([angle_bin(a) or "" for a in pair.angles])
    for low, _ in ANGLE_BINS:
        label = angle_bin(low) or ""
        out["por_angulo"][label] = {
            n: {
                k: v
                for k, v in at_threshold(pair.same[bins == label], pair.different, t).items()
                if k.startswith("U1") or k == "pares_mismo_auto"
            }
            for n, t in thresholds.items()
        }
    return out


def evaluate_model(
    models: Any,
    root: Path,
    parts: Mapping[str, Sequence[Mapping[str, Any]]],
    partial: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Un modelo: calibra en validación (los tres tipos juntos) y reporta en prueba y validación."""
    types: dict[str, dict[str, PairType]] = {}
    for split in ("val", "test"):
        whole_matrix = embed_files(
            models, root / "crops-v2", [str(w["archivo"]) for w in parts[split]], workers
        )
        partial_matrix = {
            kind: embed_files(models, root / "partial-v2", [str(p["archivo"]) for p in rows], workers)
            for kind, rows in partial[split].items()
            if rows
        }
        types[split] = build_types(parts[split], partial[split], whole_matrix, partial_matrix, seed)
    val_all = pooled(types["val"])
    calibration = calibrate(val_all.same, val_all.different, MAX_U1)
    per_type = {name: calibrate(p.same, p.different, MAX_U1) for name, p in types["val"].items()}
    thresholds = {"REID_T_del_servicio": float(SERVICE_REID_T)}
    if calibration.get("umbral") is not None:
        thresholds["calibrado_en_val"] = float(calibration["umbral"])
    strictest = [c["umbral"] for c in per_type.values() if c.get("umbral") is not None]
    if strictest:
        thresholds["calibrado_por_tipo_el_mas_bajo"] = float(min(strictest))
    report: dict[str, Any] = {
        "version": models.versions()["embedder"],
        "calibracion_en_val_con_los_tres_tipos": calibration,
        "calibracion_en_val_por_tipo": per_type,
    }
    for split in ("test", "val"):
        report[split] = {
            "total": block(pooled(types[split]), thresholds),
            **{name: block(p, thresholds) for name, p in types[split].items()},
        }
    return report


def detector_models(embedder: str, args: argparse.Namespace) -> Any:
    settings = Settings.model_construct(plate_tiling=False, face_tiling=False)
    return build_models(
        {"embedder": embedder, "detector": DETECTOR},
        args.manifest,
        args.models_dir,
        params=adapter_params(settings),
    )


def pct(value: float | None) -> str:
    return "n/d" if value is None else f"{100 * value:.1f} %"


def print_tables(report: Mapping[str, Any]) -> None:
    for name, entry in report["modelos"].items():
        print(f"\n== {name} ({entry['version']}) — prueba ==")
        print(f"calibración en val (tres tipos juntos): {entry['calibracion_en_val_con_los_tres_tipos']}")
        for scope in ("total", *TYPES):
            if scope not in entry["test"]:
                continue
            data = entry["test"][scope]
            for key in ("todos", "dificiles"):
                cell = data[key]
                auc_value = None if cell["auc"] is None else round(cell["auc"], 4)
                row = f"{scope:<16} {key:<10} AUC {auc_value}"
                for threshold_name in [k for k in cell if k != "auc"]:
                    c = cell[threshold_name]
                    row += f" | {threshold_name}: U1 {pct(c['U1'])} (cota {pct(c['U1_cota_95'])})"
                    row += f" S4 {pct(c['S4'])}"
                print(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--models", nargs="+", default=[BASELINE, TUNED], help="embedders a medir")
    parser.add_argument(
        "--veri-dir", type=Path, default=os.environ.get("VERI_DIR"), help="VeRi-776 (VERI_DIR)"
    )
    parser.add_argument("--per-track", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
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
    whole_items = json.loads((root / "crops_v2.json").read_text(encoding="utf-8"))
    partial_items = json.loads((root / "partial_v2.json").read_text(encoding="utf-8"))
    parts = {s: [i for i in whole_items if i["particion"] == s] for s in ("val", "test")}
    partial = {
        s: {
            k: [i for i in partial_items if i["particion"] == s and i["tipo"] == k]
            for k in ("patente", "detalle")
        }
        for s in ("val", "test")
    }
    if args.limit:
        parts = {s: rows[: args.limit] for s, rows in parts.items()}
        keep = {str(r["auto"]) for rows in parts.values() for r in rows}
        partial = {
            s: {k: [i for i in rows if i["auto"] in keep] for k, rows in d.items()}
            for s, d in partial.items()
        }
    print(
        {s: {"enteros": len(parts[s]), **{k: len(v) for k, v in partial[s].items()}} for s in parts},
        flush=True,
    )
    report: dict[str, Any] = {"REID_T_del_servicio": SERVICE_REID_T, "max_U1": MAX_U1, "modelos": {}}
    try:
        built = {name: detector_models(name, args) for name in args.models}
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    calibrated: dict[str, float | None] = {}
    for name, models in built.items():
        report["modelos"][name] = evaluate_model(models, root, parts, partial, args.workers, args.seed)
        calibrated[name] = report["modelos"][name]["calibracion_en_val_con_los_tres_tipos"].get("umbral")
        print(f"[{name}] umbral calibrado en val: {calibrated[name]}", flush=True)
    if args.veri_dir:
        from reid_3drealcar import veri_crops, veri_report  # noqa: PLC0415

        first = next(iter(built.values()))
        crops, images, discarded = veri_crops(
            Path(args.veri_dir), args.per_track, args.seed, args.limit, first, args.workers
        )
        report["veri"] = {"recortes_usados": len(crops), "descartados": discarded, "modelos": {}}
        for name, models in built.items():
            thresholds = {"REID_T_del_servicio": float(SERVICE_REID_T)}
            if calibrated[name] is not None:
                thresholds["calibrado_en_3DRealCar"] = float(calibrated[name])
            report["veri"]["modelos"][name] = veri_report(crops, images, models, thresholds, args.workers)
            print(f"[VeRi {name}] AUC {report['veri']['modelos'][name]['todos']['auc']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reid_parciales_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_tables(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
