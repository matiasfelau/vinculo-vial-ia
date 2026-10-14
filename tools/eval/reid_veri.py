"""Calibra `REID_T` con VeRi-776: ¿a partir de qué coseno es seguro declarar que es otro auto (R2)?

Para cada recorte de VeRi corre lo mismo que la cascada: segmenta los autos, se queda con el más grande y
calcula su embedding con la máscara (`segmenter` + `embedder`). Después compara por coseno:

- **pares del mismo auto**: mismo ID y cámaras distintas (la cascada nunca debe marcarlos `BURST_MISMATCH`);
- **pares de autos distintos**: IDs distintos; aparte los **difíciles**, con el mismo color y el mismo tipo
  según los atributos de VeRi.

R2 pide 0 pares del mismo auto por debajo de `REID_T` (cota del 95 % = 3/N). Si ese umbral exacto deja de
detectar casi todo, el informe trae la curva completa y propone un valor con una tasa de falsos rojos acotada.

VeRi son recortes de cámaras de vigilancia de baja resolución y su licencia es solo de investigación: se usa
únicamente para validar (no se entrena nada ni se copia al repo). Las fotos de la app son de celular, así
que el resultado es provisorio hasta validarlo con fotos propias. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from reid import DEFAULT_EMBEDDER, DEFAULT_SEGMENTER, HISTOGRAM_BINS, auc, describe, histogram, text_histogram

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.service.ports import ImageDecodeError
from vvi_ia.service.validate_evidence import largest_car

REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_REID_T = 0.50  # el valor por defecto de REID_T antes de calibrarlo con VeRi
DEFAULT_PER_TRACK = 3  # imágenes por (auto, cámara): acota el costo sin perder autos ni cámaras
DEFAULT_MAX_FALSE_RED = 0.001  # tasa de pares del mismo auto por debajo del umbral propuesto
MIN_USEFUL_DETECTION = 0.05  # con menos detección de autos distintos, el umbral exacto no sirve de nada
CURVE_STEP = 0.05
PROPOSAL_STEP = 0.005
LABEL_ITEM = re.compile(
    r'imageName="([^"]+)"\s+vehicleID="(\d+)"\s+cameraID="(c\d+)"\s+colorID="(\d+)"\s+typeID="(\d+)"'
)
BLOCK = 256


class DataError(ValueError):
    """VeRi no tiene la forma esperada."""


@dataclass(frozen=True)
class Crop:
    """Un recorte de VeRi con lo que hace falta para armar los pares."""

    name: str
    vehicle: str
    camera: str
    color: str
    kind: str
    is_query: bool


def load_labels(xml_path: Path) -> dict[str, tuple[str, str, str, str]]:
    """`imageName -> (vehicleID, cameraID, colorID, typeID)` de `test_label.xml` (codificado en gb2312)."""
    if not xml_path.is_file():
        raise DataError(f"no existe {xml_path.name}")
    text = xml_path.read_bytes().decode("gb2312", errors="replace")
    labels = {m[0]: (m[1], m[2], m[3], m[4]) for m in LABEL_ITEM.findall(text)}
    if not labels:
        raise DataError(f"{xml_path.name} no tiene imágenes etiquetadas")
    return labels


def read_names(path: Path) -> list[str]:
    if not path.is_file():
        raise DataError(f"no existe {path.name}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_crops(
    labels: dict[str, tuple[str, str, str, str]],
    test_names: Sequence[str],
    query_names: Sequence[str],
    per_track: int,
    seed: int,
    limit: int | None = None,
) -> list[Crop]:
    """Hasta `per_track` recortes por (auto, cámara), con prioridad para los de la consulta (`query`)."""
    queries = set(query_names)
    tracks: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in test_names:
        if name in labels:
            tracks[(labels[name][0], labels[name][1])].append(name)
    rng = random.Random(seed)  # noqa: S311 - muestreo de fotos, no criptografía
    chosen: list[str] = []
    for key in sorted(tracks):
        names = tracks[key]
        rng.shuffle(names)
        names.sort(key=lambda n: n not in queries)  # estable: las de la consulta primero
        chosen += names[:per_track]
    crops = [
        Crop(n, labels[n][0], labels[n][1], labels[n][2], labels[n][3], n in queries) for n in sorted(chosen)
    ]
    return crops[:limit] if limit else crops


def embed_crop(path: Path, models: Any, decoder: CvImageDecoder) -> tuple[Any, str | None]:
    """Embedding del auto más grande del recorte, o el motivo por el que no se pudo (como en la cascada)."""
    try:
        pixels = decoder.decode(path.read_bytes())
    except (OSError, ImageDecodeError):
        return None, "recorte ilegible"
    cars = models.cars.cars(pixels)
    if cars is None:
        return None, "falla del segmentador"
    if not cars:
        return None, "sin auto segmentado"
    embedding = models.embedder.embed(pixels, largest_car(cars).mask)
    return (embedding, None) if embedding is not None else (None, "sin embedding")


def embed_all(
    crops: Sequence[Crop], folder: Path, models: Any, workers: int
) -> tuple[list[Crop], NDArray[np.float32], Counter[str]]:
    """Recortes con embedding (en orden), la matriz normalizada y por qué se descartó el resto."""
    decoder = CvImageDecoder()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: embed_crop(folder / c.name, models, decoder), crops))
    kept = [(c, e) for c, (e, _) in zip(crops, results, strict=True) if e is not None]
    discarded = Counter(reason for _, reason in results if reason)
    if not kept:
        return [], np.zeros((0, 0), np.float32), discarded
    matrix = np.stack([e for _, e in kept]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.where(norms == 0, 1.0, norms)
    return [c for c, _ in kept], matrix, discarded


def pair_similarities(
    crops: Sequence[Crop], matrix: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.str_]]:
    """Cosenos de los pares: mismo auto (cámaras distintas), autos distintos y autos distintos difíciles.

    Devuelve también el auto de cada par del mismo auto (para contar en cuántos autos hay un caso).
    """
    vehicles = np.array([c.vehicle for c in crops])
    cameras = np.array([c.camera for c in crops])
    looks = np.array([f"{c.color}-{c.kind}" for c in crops])
    same: list[NDArray[np.float32]] = []
    same_ids: list[NDArray[np.str_]] = []
    different: list[NDArray[np.float32]] = []
    hard: list[NDArray[np.float32]] = []
    n = len(crops)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        sims = matrix[start:stop] @ matrix.T
        rows = np.arange(start, stop)[:, None]
        upper = np.arange(n)[None, :] > rows
        same_vehicle = vehicles[start:stop, None] == vehicles[None, :]
        is_same = upper & same_vehicle & (cameras[start:stop, None] != cameras[None, :])
        is_different = upper & ~same_vehicle
        is_hard = is_different & (looks[start:stop, None] == looks[None, :])
        same.append(sims[is_same])
        same_ids.append(np.broadcast_to(vehicles[start:stop, None], sims.shape)[is_same])
        different.append(sims[is_different])
        hard.append(sims[is_hard])

    def join(parts: list[Any], dtype: Any) -> Any:
        return np.concatenate(parts) if parts else np.zeros(0, dtype)

    return join(same, np.float32), join(different, np.float32), join(hard, np.float32), join(same_ids, str)


def rate_below(values: NDArray[np.float32], threshold: float) -> float | None:
    return round(float((values < threshold).mean()), 5) if values.size else None


def curve(
    same: NDArray[np.float32],
    different: NDArray[np.float32],
    hard: NDArray[np.float32],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Por umbral: falsos rojos (mismo auto debajo) y autos distintos que salen `BURST_MISMATCH`."""
    return [
        {
            "REID_T": round(float(t), 4),
            "mismo_auto_debajo": int((same < t).sum()),
            "falsos_rojos": rate_below(same, t),
            "autos_distintos_detectados": rate_below(different, t),
            "dificiles_detectados": rate_below(hard, t),
        }
        for t in thresholds
    ]


def calibrate(
    same: NDArray[np.float32],
    different: NDArray[np.float32],
    hard: NDArray[np.float32],
    max_false_red: float = DEFAULT_MAX_FALSE_RED,
) -> dict[str, Any]:
    """El umbral de R2 (0 pares del mismo auto debajo) y, si no sirve, un valor con falsos rojos acotados."""
    if same.size == 0 or different.size == 0:
        return {"posible": False, "motivo": "hacen falta pares del mismo auto y de autos distintos"}
    exact = float(
        np.floor(round(float(same.min()) * 1000, 3)) / 1000
    )  # `sim < exact` no deja ningún par igual debajo
    detection_at_exact = rate_below(different, exact) or 0.0
    useful = detection_at_exact >= MIN_USEFUL_DETECTION
    allowed = int(np.floor(max_false_red * same.size))
    ordered = np.sort(same)
    proposal = float(np.floor(float(ordered[allowed]) / PROPOSAL_STEP) * PROPOSAL_STEP)
    chosen = exact if useful else proposal
    return {
        "posible": True,
        "umbral_exacto": round(exact, 4),
        "cota_95_del_exacto": round(min(1.0, 3 / same.size), 6),
        "detecta_con_el_exacto": {
            "autos_distintos": detection_at_exact,
            "dificiles": rate_below(hard, exact),
        },
        "el_exacto_sirve": useful,
        "umbral_propuesto": round(chosen, 4),
        "criterio_de_la_propuesta": (
            "el exacto (0 falsos rojos)"
            if useful
            else f"el mayor valor con <= {max_false_red:.2%} de los pares del mismo auto debajo"
        ),
        "con_el_propuesto": {
            "falsos_rojos": rate_below(same, chosen),
            "pares_del_mismo_auto_debajo": int((same < chosen).sum()),
            "autos_distintos_detectados": rate_below(different, chosen),
            "dificiles_detectados": rate_below(hard, chosen),
        },
        "con_el_anterior": {
            "REID_T": PREVIOUS_REID_T,
            "falsos_rojos": rate_below(same, PREVIOUS_REID_T),
            "pares_del_mismo_auto_debajo": int((same < PREVIOUS_REID_T).sum()),
            "autos_distintos_detectados": rate_below(different, PREVIOUS_REID_T),
            "dificiles_detectados": rate_below(hard, PREVIOUS_REID_T),
        },
    }


def vehicles_with_low_pair(same: NDArray[np.float32], ids: NDArray[np.str_], threshold: float) -> int:
    """En cuántos autos hay al menos un par del mismo auto debajo del umbral."""
    return int(len(set(ids[same < threshold].tolist())))


def build_report(
    crops: Sequence[Crop],
    matrix: NDArray[np.float32],
    discarded: Counter[str],
    n_selected: int,
    max_false_red: float,
) -> dict[str, Any]:
    same, different, hard, ids = pair_similarities(crops, matrix)
    calibration = calibrate(same, different, hard, max_false_red)
    thresholds = sorted({*np.round(np.arange(0.0, 1.0001, CURVE_STEP), 2).tolist(), PREVIOUS_REID_T})
    chosen = calibration.get("umbral_propuesto")
    return {
        "recortes_elegidos": n_selected,
        "recortes_usados": len(crops),
        "descartados": dict(discarded),
        "autos": len({c.vehicle for c in crops}),
        "camaras": len({c.camera for c in crops}),
        "pares": {
            "mismo_auto": int(same.size),
            "autos_distintos": int(different.size),
            "dificiles": int(hard.size),
        },
        "similitudes": {
            "mismo_auto": describe(same.astype(np.float64)),
            "autos_distintos": describe(different.astype(np.float64)),
            "dificiles": describe(hard.astype(np.float64)),
        },
        "auc": {
            "todos": auc(
                same.astype(np.float64), different[:: max(1, different.size // 2_000_000)].astype(np.float64)
            ),
            "dificiles": auc(
                same.astype(np.float64), hard[:: max(1, hard.size // 2_000_000)].astype(np.float64)
            ),
        },
        "calibracion": calibration,
        "autos_con_un_par_igual_debajo_del_propuesto": (
            vehicles_with_low_pair(same, ids, chosen) if chosen is not None else None
        ),
        "curva": curve(same, different, hard, thresholds),
        "histograma": {
            "limites": [float(b) for b in HISTOGRAM_BINS],
            "mismo_auto": histogram(same.astype(np.float64)),
            "autos_distintos": histogram(different.astype(np.float64)),
            "dificiles": histogram(hard.astype(np.float64)),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--veri-dir",
        type=Path,
        default=os.environ.get("VERI_DIR"),
        help="carpeta VeRi (VERI_DIR) con image_test/",
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--segmenter", default=DEFAULT_SEGMENTER)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument(
        "--segmenter-min-conf", type=float, default=float(os.environ.get("SEGMENTER_MIN_CONF", 0.5))
    )
    parser.add_argument(
        "--per-track", type=int, default=DEFAULT_PER_TRACK, help="recortes por (auto, cámara)"
    )
    parser.add_argument("--workers", type=int, default=4, help="recortes que se embeben a la vez")
    parser.add_argument("--limit", type=int, help="recortes en total (para probar rápido)")
    parser.add_argument("--max-false-red", type=float, default=DEFAULT_MAX_FALSE_RED)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for flag, value in (
        ("--veri-dir", args.veri_dir),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    if not 0.0 <= args.max_false_red < 1.0 or args.per_track < 1 or args.workers < 1:
        parser.error("--max-false-red entre 0 y 1; --per-track y --workers de 1 en adelante")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    veri = Path(args.veri_dir)
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (veri, args.out_dir)):
        print("VeRi y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    try:
        labels = load_labels(veri / "test_label.xml")
        crops = select_crops(
            labels,
            read_names(veri / "name_test.txt"),
            read_names(veri / "name_query.txt"),
            args.per_track,
            args.seed,
            args.limit,
        )
        models = build_models(
            {"segmenter": args.segmenter, "embedder": args.embedder},
            args.manifest,
            args.models_dir,
            params={"segmenter_min_conf": args.segmenter_min_conf},
        )
    except (DataError, ModelSetupError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    print(f"{len(crops)} recortes elegidos; embebiendo…", flush=True)
    used, matrix, discarded = embed_all(crops, veri / "image_test", models, args.workers)
    report = build_report(used, matrix, discarded, len(crops), args.max_false_red)
    report["modelos"] = {k: v for k, v in models.versions().items() if k in ("segmenter", "embedder")}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "reid_veri_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Modelos {report['modelos']} · {report['recortes_usados']} de "
        f"{report['recortes_elegidos']} recortes, "
        f"{report['autos']} autos, {report['camaras']} cámaras"
    )
    print(f"Descartados (no comparables en la cascada): {report['descartados'] or 'ninguno'}")
    print(f"Pares: {report['pares']} · AUC {report['auc']}")
    for key in ("mismo_auto", "autos_distintos", "dificiles"):
        print(f"{key}: {report['similitudes'][key]}")
    print(f"Calibración: {report['calibracion']}")
    print("REID_T · mismo auto debajo (falsos rojos) · autos distintos detectados · difíciles detectados")
    for row in report["curva"]:
        print(
            f"  {row['REID_T']:.2f}  {row['mismo_auto_debajo']:>7} ({row['falsos_rojos']})  "
            f"{row['autos_distintos_detectados']}  {row['dificiles_detectados']}"
        )
    for key in ("mismo_auto", "autos_distintos", "dificiles"):
        print(f"Histograma, {key}:")
        print("\n".join(text_histogram(report["histograma"][key])) or "  (sin pares)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
