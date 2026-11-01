"""Evalúa la re-identificación: ¿se parecen más dos fotos del mismo auto que las de autos distintos?

Para cada foto corre lo mismo que el servicio: segmenta los autos (`segmenter`), se queda con el más
grande y calcula su embedding (`embedder`). Después compara por coseno todas las fotos del mismo auto
(pares iguales) y las de autos distintos (pares distintos), muestra las dos distribuciones y propone
`REID_T_LOW` y `REID_T_HIGH`.

Entrada: `<EVAL_DATA_DIR>/reid/fotos.csv` con las columnas `archivo` y `auto` (el mismo valor en `auto`
significa el mismo auto físico) y las fotos en `<EVAL_DATA_DIR>/reid/fotos/`. Las fotos tienen patentes y
rostros de terceros: el dataset y la salida van fuera del repo y la consola muestra solo cifras.
Uso: ver `tools/eval/README.md`.
"""

import argparse
import csv
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import Vector
from vvi_ia.domain.reid import cosine
from vvi_ia.service.ports import ImageDecodeError
from vvi_ia.service.validate_evidence import largest_car

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = "reid"
PHOTOS_FOLDER = "fotos"
CSV_NAME = "fotos.csv"
COLUMNS = ("archivo", "auto")
DEFAULT_SEGMENTER = "rfdetr_seg_coco"
DEFAULT_EMBEDDER = "dinov2_vits14"
CURRENT_T_LOW = (
    0.075  # el valor por defecto de REID_T en `config.py` (un solo umbral: otro auto si cos < REID_T)
)
CURRENT_T_HIGH = 0.70  # el límite alto que usaba el modelo anterior; solo informativo
DEFAULT_MAX_FALSE_FRAUD = 0.01
DEFAULT_MAX_FALSE_ACCEPT = 0.01
DEFAULT_MAX_DIFFERENT = 20_000
HISTOGRAM_BINS = np.round(np.arange(-1.0, 1.0001, 0.05), 2)
PERCENTILES = (1, 5, 50, 95, 99)
THRESHOLD_STEP = 0.01


class DataError(ValueError):
    """El CSV de fotos no es válido."""


@dataclass(frozen=True)
class Photo:
    """Una foto del CSV: su archivo y a qué auto físico pertenece."""

    file: str
    car: str


def load_photos(csv_path: Path) -> list[Photo]:
    """Fotos del CSV. Lanza `DataError` indicando el número de fila de cada problema."""
    if not csv_path.is_file():
        raise DataError(f"no existe {csv_path.name}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(COLUMNS) - set(reader.fieldnames):
            raise DataError(f"{csv_path.name} necesita las columnas {', '.join(COLUMNS)}")
        photos: list[Photo] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            file, car = (row["archivo"] or "").strip(), (row["auto"] or "").strip()
            if not file or not car:
                raise DataError(f"{csv_path.name}, fila {row_number}: faltan el archivo o el auto")
            if file in seen:
                raise DataError(f"{csv_path.name}, fila {row_number}: la foto está repetida")
            seen.add(file)
            photos.append(Photo(file, car))
    return photos


def embed_photos(
    photos: Sequence[Photo], folder: Path, models: Any
) -> tuple[dict[str, list[Vector]], Counter[str]]:
    """Embedding del auto más grande de cada foto, agrupado por auto, y cuántas fotos se descartaron."""
    decoder = CvImageDecoder()
    by_car: dict[str, list[Vector]] = defaultdict(list)
    discarded: Counter[str] = Counter()
    for photo in photos:
        try:
            pixels = decoder.decode((folder / photo.file).read_bytes())
        except (OSError, ImageDecodeError):
            discarded["foto ilegible"] += 1
            continue
        cars = models.cars.cars(pixels)
        if cars is None:
            discarded["falla del segmentador"] += 1
            continue
        if not cars:
            discarded["sin auto"] += 1
            continue
        embedding = models.embedder.embed(pixels, largest_car(cars).mask)
        if embedding is None:
            discarded["sin embedding"] += 1
            continue
        by_car[photo.car].append(embedding)
    return dict(by_car), discarded


def similarities(
    by_car: Mapping[str, Sequence[Vector]], max_different: int = DEFAULT_MAX_DIFFERENT, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cosenos de los pares del mismo auto y de los de autos distintos (estos, sampleados si son muchos)."""
    same = [cosine(a, b) for vectors in by_car.values() for a, b in itertools.combinations(vectors, 2)]
    pairs = [
        (a, b)
        for (_, group_a), (_, group_b) in itertools.combinations(sorted(by_car.items()), 2)
        for a in group_a
        for b in group_b
    ]
    if len(pairs) > max_different:
        pairs = random.Random(seed).sample(pairs, max_different)  # noqa: S311 (muestreo, no criptografía)
    different = [cosine(a, b) for a, b in pairs]
    return np.array(same, np.float64), np.array(different, np.float64)


def describe(values: NDArray[np.float64]) -> dict[str, Any]:
    if values.size == 0:
        return {"pares": 0}
    return {
        "pares": int(values.size),
        "media": round(float(values.mean()), 4),
        "desvio": round(float(values.std()), 4),
        "minimo": round(float(values.min()), 4),
        **{f"p{q}": round(float(np.percentile(values, q)), 4) for q in PERCENTILES},
        "maximo": round(float(values.max()), 4),
    }


def histogram(values: NDArray[np.float64]) -> list[int]:
    return [int(count) for count in np.histogram(values, bins=HISTOGRAM_BINS)[0]]


def auc(same: NDArray[np.float64], different: NDArray[np.float64]) -> float | None:
    """Probabilidad de que un par igual tenga más similitud que uno distinto (1,0 = se separan del todo)."""
    if same.size == 0 or different.size == 0:
        return None
    values = np.concatenate([same, different])
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size)
    start = 0
    for end in range(1, values.size + 1):
        if end == values.size or ordered[end] != ordered[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2  # los empates comparten el rango promedio
            start = end
    rank_sum = ranks[: same.size].sum()
    return round(float((rank_sum - same.size * (same.size + 1) / 2) / (same.size * different.size)), 4)


def decide_shares(values: NDArray[np.float64], t_low: float, t_high: float) -> dict[str, float | None]:
    """Qué fracción de los pares se clasificaría SAME, UNCERTAIN y DIFFERENT con estos umbrales."""
    if values.size == 0:
        return {"SAME": None, "UNCERTAIN": None, "DIFFERENT": None}
    same = float((values >= t_high).mean())
    different = float((values < t_low).mean())
    return {
        "SAME": round(same, 4),
        "UNCERTAIN": round(1.0 - same - different, 4),
        "DIFFERENT": round(different, 4),
    }


def floor_step(value: float) -> float:
    return round(float(np.floor(round(value / THRESHOLD_STEP, 6)) * THRESHOLD_STEP), 2)


def ceil_step(value: float) -> float:
    return round(float(np.ceil(round(value / THRESHOLD_STEP, 6)) * THRESHOLD_STEP), 2)


def propose_thresholds(
    same: NDArray[np.float64],
    different: NDArray[np.float64],
    max_false_fraud: float = DEFAULT_MAX_FALSE_FRAUD,
    max_false_accept: float = DEFAULT_MAX_FALSE_ACCEPT,
) -> dict[str, Any]:
    """`REID_T_LOW` y `REID_T_HIGH` que cumplen los dos límites de error.

    - `REID_T_LOW`: por debajo se declara DIFERENTE (alerta roja de fraude). Un par del mismo auto que cae
      ahí es un falso fraude: el umbral no puede dejar debajo más de `max_false_fraud` de los pares iguales
      (tope: `tope_bajo`).
    - `REID_T_HIGH`: desde ahí se declara IGUAL. Un par de autos distintos que llega ahí es un fraude que
      pasa: el umbral no puede ser menor que el valor que superan `max_false_accept` de los pares distintos
      (piso: `piso_alto`).

    Los dos límites se cumplen siempre. Si las distribuciones se separan (`tope_bajo` por encima de
    `piso_alto`), la banda entre los umbrales cae en el hueco y casi ningún par queda ahí; si se solapan, la
    banda se ensancha y más pares quedan en UNCERTAIN (amarillo), que es lo honesto.
    """
    if same.size == 0 or different.size == 0:
        return {"posible": False, "motivo": "hacen falta pares iguales y pares distintos"}
    ceiling_low = float(np.quantile(same, max_false_fraud, method="lower"))
    allowed = int(np.floor(max_false_accept * different.size))
    ordered = np.sort(different)[::-1]
    floor_high = (
        float(np.nextafter(ordered[allowed], np.inf)) if allowed < ordered.size else float(ordered[-1])
    )
    low = floor_step(min(ceiling_low, floor_high))
    high = ceil_step(max(ceiling_low, floor_high))
    if high <= low:
        high = round(low + THRESHOLD_STEP, 2)
    return {
        "REID_T_LOW": low,
        "REID_T_HIGH": high,
        "se_separan": ceiling_low > floor_high,
        "tope_bajo": round(ceiling_low, 4),
        "piso_alto": round(floor_high, 4),
        "limites": {"falso_fraude_max": max_false_fraud, "falso_igual_max": max_false_accept},
        "pares_iguales": decide_shares(same, low, high),
        "pares_distintos": decide_shares(different, low, high),
    }


def build_report(
    by_car: Mapping[str, Sequence[Vector]],
    discarded: Counter[str],
    n_photos: int,
    max_false_fraud: float,
    max_false_accept: float,
    max_different: int,
    seed: int,
) -> dict[str, Any]:
    same, different = similarities(by_car, max_different, seed)
    return {
        "fotos_en_el_csv": n_photos,
        "fotos_usadas": sum(len(v) for v in by_car.values()),
        "descartadas": dict(discarded),
        "autos": len(by_car),
        "autos_con_mas_de_una_foto": sum(len(v) > 1 for v in by_car.values()),
        "pares_iguales": describe(same),
        "pares_distintos": describe(different),
        "auc": auc(same, different),
        "histograma": {
            "limites": [float(b) for b in HISTOGRAM_BINS],
            "iguales": histogram(same),
            "distintos": histogram(different),
        },
        "valores_actuales": {
            "REID_T_LOW": CURRENT_T_LOW,
            "REID_T_HIGH": CURRENT_T_HIGH,
            "pares_iguales": decide_shares(same, CURRENT_T_LOW, CURRENT_T_HIGH),
            "pares_distintos": decide_shares(different, CURRENT_T_LOW, CURRENT_T_HIGH),
        },
        "propuesta": propose_thresholds(same, different, max_false_fraud, max_false_accept),
    }


def text_histogram(counts: Sequence[int], width: int = 40) -> list[str]:
    """Barras de texto de los bins que tienen pares."""
    peak = max(counts, default=0) or 1
    return [
        f"  {HISTOGRAM_BINS[i]:>5.2f} a {HISTOGRAM_BINS[i + 1]:>5.2f} {count:>6}  "
        f"{'#' * max(1, round(width * count / peak))}"
        for i, count in enumerate(counts)
        if count
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("EVAL_DATA_DIR"),
        help=f"carpeta de datos de evaluación (EVAL_DATA_DIR); usa {DATA_FOLDER}/",
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument(
        "--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="pesos (MODELS_DIR)"
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--segmenter", default=DEFAULT_SEGMENTER)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument(
        "--segmenter-min-conf", type=float, default=float(os.environ.get("SEGMENTER_MIN_CONF", 0.5))
    )
    parser.add_argument("--max-false-fraud", type=float, default=DEFAULT_MAX_FALSE_FRAUD)
    parser.add_argument("--max-false-accept", type=float, default=DEFAULT_MAX_FALSE_ACCEPT)
    parser.add_argument(
        "--max-different", type=int, default=DEFAULT_MAX_DIFFERENT, help="tope de pares distintos"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for flag, value in (
        ("--data-dir", args.data_dir),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    for flag, value in (
        ("--max-false-fraud", args.max_false_fraud),
        ("--max-false-accept", args.max_false_accept),
    ):
        if not 0.0 <= value < 1.0:
            parser.error(f"{flag} debe estar entre 0 y 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    folder = Path(args.data_dir) / DATA_FOLDER
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (folder, args.out_dir)):
        print("El dataset y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    try:
        photos = load_photos(folder / CSV_NAME)
        models = build_models(
            {"segmenter": args.segmenter, "embedder": args.embedder},
            args.manifest,
            args.models_dir,
            params={"segmenter_min_conf": args.segmenter_min_conf},
        )
    except (DataError, ModelSetupError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    by_car, discarded = embed_photos(photos, folder / PHOTOS_FOLDER, models)
    report = build_report(
        by_car,
        discarded,
        len(photos),
        args.max_false_fraud,
        args.max_false_accept,
        args.max_different,
        args.seed,
    )
    report["modelos"] = {
        "segmenter": models.versions()["segmenter"],
        "embedder": models.versions()["embedder"],
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "reid_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Modelos {report['modelos']} · {report['fotos_usadas']} de {report['fotos_en_el_csv']} fotos, "
        f"{report['autos']} autos"
    )
    print(f"Descartadas: {report['descartadas'] or 'ninguna'}")
    for title, key in (("Mismo auto", "pares_iguales"), ("Autos distintos", "pares_distintos")):
        print(f"{title}: {report[key]}")
    print(f"AUC (1,0 = se separan del todo): {report['auc']}")
    print("Histograma, mismo auto:")
    print("\n".join(text_histogram(report["histograma"]["iguales"])) or "  (sin pares)")
    print("Histograma, autos distintos:")
    print("\n".join(text_histogram(report["histograma"]["distintos"])) or "  (sin pares)")
    current = report["valores_actuales"]
    print(
        f"Con los valores actuales {current['REID_T_LOW']}/{current['REID_T_HIGH']}: "
        f"iguales {current['pares_iguales']} · distintos {current['pares_distintos']}"
    )
    proposal = report["propuesta"]
    if "REID_T_LOW" in proposal:
        print(
            f"Propuesta: REID_T_LOW={proposal['REID_T_LOW']} REID_T_HIGH={proposal['REID_T_HIGH']} "
            f"(las distribuciones {'se separan' if proposal['se_separan'] else 'se solapan'})"
        )
    else:
        print(f"Sin propuesta: {proposal['motivo']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
