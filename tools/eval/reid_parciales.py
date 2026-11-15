"""Re-ID con 3DRealCar: general contra patente, con un único umbral calibrado en validación.

R2 compara la foto GENERAL contra la de PLATE (la trompa o la cola, cortada entre el capot y el
parabrisas, con la patente al 8-15 % del ancho). Esta medición usa los recortes de
`tools/training/reid/prepare.py` (vehículo dominante del detector v2, 0,35 de confianza, sin
segmentador) y los simulados por `tools/training/reid/partial.py`:

- **general_patente** (el que decide R2): una vista entera contra un recorte de trompa o cola, de vistas
  separadas por 30° o más. **Un umbral único**, calibrado en validación: el menor con el que al menos
  `TARGET_S4` (70 %) de los pares de autos distintos queda como "otro auto" (S4 y U1 suben juntos con el
  umbral, así que es el de menor U1), y que además deja la cota del 95 % de U1 de validación en `MAX_U1`
  (2 %) o menos.
- **entero** (solo referencia, no entra al umbral): dos vistas del auto entero.

Hay una sola comparación por alerta, así que U1 de este par es también el U1 por alerta. En prueba se
reporta U1 con su cota, S4 total, en pares difíciles (misma marca y color) y por condición de luz, para
el modelo ajustado y para el DINOv2 sin ajustar con los mismos recortes. R2 solo rechaza con certeza:
ante la duda, GREEN. Todo sale en cifras; los datos quedan fuera del repo. Uso: ver
`tools/eval/README.md`.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "training" / "reid"))

from dataset import ANGLE_BINS, angle_bin, build_pairs  # noqa: E402
from metrics import auc, calibrate, upper_bound  # noqa: E402
from partial import cross_different_pairs, cross_same_pairs  # noqa: E402

from vvi_ia.adapters.models.manifest import load_manifest  # noqa: E402
from vvi_ia.adapters.models.registry import ModelSetupError, build_models  # noqa: E402
from vvi_ia.config import Settings  # noqa: E402
from vvi_ia.main import adapter_params  # noqa: E402

DETECTOR = "rfdetr_openimages_v2"
BASELINE = "dinov2_vits14"
TUNED = "dinov2_vits14_reid_v2"
SERVICE_REID_T = Settings.model_fields["reid_t"].default  # el umbral que hoy usa el servicio
LIGHTING = ("standard", "reflective", "dark")
TARGET_S4 = 0.70  # el S4 que se pide en validación para el umbral único del servicio (15/11/2026)
MAX_U1 = 0.02  # la cota del 95 % de U1 que se acepta (el umbral de U1 del proyecto, 1 %, no cambia)
PROJECT_U1 = 0.01
DIFFERENT_LIMIT = 100_000
HARD_LIMIT = 30_000
Matrix = NDArray[np.float32]


class PairSet:
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


class LazyModels:
    """Los modelos del servicio, cargados recién cuando hacen falta (con los embeddings en caché, no)."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._models: Any = None

    def get(self) -> Any:
        if self._models is None:
            self._models = self._factory()
        return self._models


def embed_files(
    models: LazyModels, folder: Path, names: Sequence[str], workers: int, cache: Path | None = None
) -> Matrix:
    """Embedding de cada recorte con el adaptador del servicio (el recorte completo es la máscara).

    Con `cache` (un `.npy`), guarda la matriz y la reutiliza si ya existe con la misma cantidad de recortes.
    """
    if cache is not None and cache.is_file():
        cached: Matrix = np.load(cache)
        if cached.shape[0] == len(names):
            return cached
    embedder = models.get().embedder

    def one(name: str) -> NDArray[np.float32]:
        crop = cv2.imdecode(np.fromfile(str(folder / name), np.uint8), cv2.IMREAD_COLOR)
        embedding = embedder.embed(crop, np.ones(crop.shape[:2], np.bool_))
        if embedding is None:
            raise RuntimeError("el embedder no devolvió un vector")
        return np.asarray(embedding, np.float32)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        matrix: Matrix = np.stack(list(pool.map(one, names)))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, matrix)
    return matrix


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


def build_sets(
    whole: Sequence[Mapping[str, Any]],
    plates: Sequence[Mapping[str, Any]],
    whole_matrix: Matrix,
    plate_matrix: Matrix | None,
    seed: int,
) -> dict[str, PairSet]:
    """Los pares de una partición: `entero` (referencia) y `general_patente` (el que decide R2)."""
    pairs = build_pairs(whole, seed=seed, different_limit=DIFFERENT_LIMIT, hard_limit=HARD_LIMIT)
    lights = [str(w["luz"]) for w in whole]
    out: dict[str, PairSet] = {
        "entero": PairSet(
            dots(whole_matrix, whole_matrix, pairs.same),
            dots(whole_matrix, whole_matrix, pairs.different),
            dots(whole_matrix, whole_matrix, pairs.hard),
            np.array([lights[i] for i, _, _ in pairs.same]),
            light_of_pairs(lights, lights, pairs.different),
            light_of_pairs(lights, lights, pairs.hard),
            np.array([angle for _, _, angle in pairs.same]),
        )
    }
    if plate_matrix is not None and len(plates):
        cars_w = [str(w["auto"]) for w in whole]
        cars_p = [str(p["auto"]) for p in plates]
        brand_color = {str(r["auto"]): (str(r["marca"]), str(r["color"])) for r in (*whole, *plates)}
        same = cross_same_pairs(
            [(str(w["auto"]), float(w["azimut"])) for w in whole],
            [(str(p["auto"]), float(p["azimut"])) for p in plates],
        )
        different = cross_different_pairs(cars_w, cars_p, brand_color, DIFFERENT_LIMIT, seed)
        hard = cross_different_pairs(cars_w, cars_p, brand_color, HARD_LIMIT, seed, hard=True)
        lights_p = [str(p["luz"]) for p in plates]
        out["general_patente"] = PairSet(
            dots(whole_matrix, plate_matrix, [(i, j) for i, j, _ in same]),
            dots(whole_matrix, plate_matrix, different),
            dots(whole_matrix, plate_matrix, hard),
            np.array([lights[i] for i, _, _ in same]),
            light_of_pairs(lights, lights_p, different),
            light_of_pairs(lights, lights_p, hard),
            np.array([angle for _, _, angle in same]),
        )
    return out


def threshold_for_s4(different: NDArray[np.float64], target: float = TARGET_S4) -> float:
    """El menor umbral con el que al menos `target` de los pares de autos distintos tienen coseno menor."""
    ordered = np.sort(different)
    index = int(np.ceil(target * ordered.size)) - 1
    return float(np.nextafter(ordered[index], np.inf))


def cell(
    same: NDArray[np.float64], different: NDArray[np.float64], hard: NDArray[np.float64], t: float
) -> dict[str, Any]:
    """U1 con su cota, S4 total y en difíciles de un conjunto de pares a un umbral."""
    marked = int((same < t).sum())
    return {
        "mismo_auto": int(same.size),
        "U1": marked / same.size if same.size else None,
        "U1_pares": marked,
        "U1_cota_95": upper_bound(marked, int(same.size)),
        "S4": float((different < t).mean()) if different.size else None,
        "S4_dificiles": float((hard < t).mean()) if hard.size else None,
        "distintos": int(different.size),
        "dificiles": int(hard.size),
    }


def report_at(pair: PairSet, t: float) -> dict[str, Any]:
    """Todas las cifras de un tipo de par a un umbral: total, por luz y por rango de ángulo."""
    out: dict[str, Any] = {
        "auc": auc(pair.same, pair.different),
        "auc_dificiles": auc(pair.same, pair.hard),
        "total": cell(pair.same, pair.different, pair.hard, t),
        "por_luz": {},
        "por_angulo": {},
    }
    for light in LIGHTING:
        out["por_luz"][light] = cell(
            pair.same[pair.same_light == light],
            pair.different[pair.different_light == light],
            pair.hard[pair.hard_light == light],
            t,
        )
    bins = np.array([angle_bin(a) or "" for a in pair.angles])
    for low, _ in ANGLE_BINS:
        label = angle_bin(low) or ""
        marked = pair.same[bins == label]
        out["por_angulo"][label] = {
            "mismo_auto": int(marked.size),
            "U1": float((marked < t).mean()) if marked.size else None,
        }
    return out


def choose_threshold(val: PairSet) -> dict[str, Any]:
    """El umbral único de R2 con los pares general-patente de validación.

    El menor con S4 >= `TARGET_S4`. Se acepta solo si la cota del 95 % de U1 de validación es `MAX_U1` o
    menos.
    """
    t = threshold_for_s4(val.different)
    chosen = cell(val.same, val.different, val.hard, t)
    return {
        "umbral": t,
        "validacion": chosen,
        "cota_de_u1_en_validacion_aceptable": chosen["U1_cota_95"] is not None
        and chosen["U1_cota_95"] <= MAX_U1,
        "criterio": {"S4_minimo": TARGET_S4, "cota_de_U1_maxima": MAX_U1},
        "referencia_cota_de_u1_1pct": calibrate(val.same, val.different, PROJECT_U1),
    }


def evaluate_model(
    models: LazyModels,
    version: str,
    root: Path,
    whole_items: Mapping[str, Sequence[Mapping[str, Any]]],
    plate_items: Mapping[str, Sequence[Mapping[str, Any]]],
    workers: int,
    seed: int,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Un modelo: elige el umbral en validación (general-patente) y reporta en prueba y en validación."""
    key = version.replace("@", "_").replace("/", "_")

    def cache_of(split: str, kind: str) -> Path | None:
        return None if cache_dir is None else cache_dir / f"{key}_{split}_{kind}.npy"

    sets: dict[str, dict[str, PairSet]] = {}
    for split in ("val", "test"):
        whole = embed_files(
            models,
            root / "crops-v2",
            [str(w["archivo"]) for w in whole_items[split]],
            workers,
            cache_of(split, "entero"),
        )
        plate = None
        if plate_items[split]:
            plate = embed_files(
                models,
                root / "partial-v2",
                [str(p["archivo"]) for p in plate_items[split]],
                workers,
                cache_of(split, "patente"),
            )
        sets[split] = build_sets(whole_items[split], plate_items[split], whole, plate, seed)
    choice = choose_threshold(sets["val"]["general_patente"])
    t = float(choice["umbral"])
    report: dict[str, Any] = {"version": version, "eleccion_en_validacion": choice}
    for split in ("test", "val"):
        report[split] = {name: report_at(pair, t) for name, pair in sets[split].items()}
    test = report["test"]["general_patente"]["total"]
    report["cumple_la_cota_de_u1_en_prueba"] = test["U1_cota_95"] is not None and test["U1_cota_95"] <= MAX_U1
    ref = choice["referencia_cota_de_u1_1pct"].get("umbral")
    if (
        ref is not None
    ):  # el mayor umbral con la cota de U1 <= 1 % en validación (el del servicio desde el 15/11/2026)
        report["al_umbral_de_cota_u1_1pct"] = {
            "umbral": float(ref),
            "test": {name: report_at(pair, float(ref)) for name, pair in sets["test"].items()},
            "val": {name: report_at(pair, float(ref)) for name, pair in sets["val"].items()},
        }
    report["a_los_umbrales_de_referencia"] = {
        "REID_T_del_servicio": {
            name: cell(p.same, p.different, p.hard, float(SERVICE_REID_T)) for name, p in sets["test"].items()
        }
    }
    return report


def pct(value: float | None) -> str:
    return "n/d" if value is None else f"{100 * value:.2f} %"


def print_tables(report: Mapping[str, Any]) -> None:
    for name, entry in report["modelos"].items():
        choice = entry["eleccion_en_validacion"]
        print(f"\n== {name} ({entry['version']}) ==")
        print(
            f"umbral {choice['umbral']:.4f} (validación: U1 {pct(choice['validacion']['U1'])}, cota "
            f"{pct(choice['validacion']['U1_cota_95'])}, S4 {pct(choice['validacion']['S4'])})"
        )
        for scope in ("general_patente", "entero"):
            data = entry["test"].get(scope)
            if data is None:
                continue
            total = data["total"]
            print(
                f"prueba {scope:<16} AUC {data['auc']:.4f} | U1 {pct(total['U1'])} "
                f"(cota {pct(total['U1_cota_95'])}, "
                f"{total['U1_pares']} de {total['mismo_auto']}) | S4 {pct(total['S4'])} "
                f"| difíciles {pct(total['S4_dificiles'])}"
            )
            for light, c in data["por_luz"].items():
                print(
                    f"    {light:<11} U1 {pct(c['U1'])} (cota {pct(c['U1_cota_95'])}, n {c['mismo_auto']}) "
                    f"S4 {pct(c['S4'])} difíciles {pct(c['S4_dificiles'])}"
                )
        print(f"cumple la cota de U1 <= {MAX_U1:.0%} en prueba: {entry['cumple_la_cota_de_u1_en_prueba']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--models", nargs="+", default=[BASELINE, TUNED], help="embedders a medir")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, help="guardar y reutilizar los embeddings (fuera del repo)")
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
    whole_all = json.loads((root / "crops_v2.json").read_text(encoding="utf-8"))
    plate_all = [
        i
        for i in json.loads((root / "partial_v2.json").read_text(encoding="utf-8"))
        if i["tipo"] == "patente"
    ]
    whole = {s: [i for i in whole_all if i["particion"] == s] for s in ("val", "test")}
    plates = {s: [i for i in plate_all if i["particion"] == s] for s in ("val", "test")}
    if args.limit:
        whole = {s: rows[: args.limit] for s, rows in whole.items()}
        keep = {str(r["auto"]) for rows in whole.values() for r in rows}
        plates = {s: [i for i in rows if i["auto"] in keep] for s, rows in plates.items()}
    print({s: {"enteros": len(whole[s]), "patente": len(plates[s])} for s in whole}, flush=True)
    manifest = load_manifest(args.manifest)
    settings = Settings.model_construct(plate_tiling=False)

    def factory(embedder: str) -> Callable[[], Any]:
        return lambda: build_models(
            {"embedder": embedder, "detector": DETECTOR},
            args.manifest,
            args.models_dir,
            params=adapter_params(settings),
        )

    report: dict[str, Any] = {
        "REID_T_del_servicio": SERVICE_REID_T,
        "criterio": {"S4": TARGET_S4, "cota_U1": MAX_U1},
    }
    report["modelos"] = {}
    lazy: dict[str, LazyModels] = {}
    try:
        for name in args.models:
            entry = manifest.get(name)
            if entry is None:
                raise ModelSetupError(f"{name} no está en el manifest")
            lazy[name] = LazyModels(factory(name))
            version = f"{name}@{entry.sha256[:12]}"
            report["modelos"][name] = evaluate_model(
                lazy[name], version, root, whole, plates, args.workers, args.seed, args.cache_dir
            )
            print(
                f"[{name}] umbral {report['modelos'][name]['eleccion_en_validacion']['umbral']:.4f}",
                flush=True,
            )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reid_parciales_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_tables(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
