"""R1 (presencia de vehículo) sobre los conjuntos con el encuadre de 1-2 m de `capture_sets.py`.

R1 es la misma regla para todas las fotos (general, patente, detalle): hay un vehículo si el detector da una
caja `car` (en v2, cualquier clase de presencia: auto, camión, van o colectivo) con confianza >= `t_car`, o el
segmentador una instancia con confianza >= `t_seg`, y ocupa al menos `PRESENCE_MIN_AREA` de la foto
(det ∨ seg). La patente, la rueda y el daño no cuentan, y detalle no tiene regla propia.

- `measure`: corre el detector y el segmentador sobre cada foto (con un piso de confianza de 0,01) y guarda,
  por foto, los hallazgos `[confianza, fracción de la foto]`. Es la única parte que usa los modelos.
- `report`: desde esas señales barre `t_car` y `t_seg`, elige los umbrales en la mitad A de las negativas y
  de las válidas y reporta en la mitad B (semilla fija, la de `capture_sets.py`): S1 (negativas aceptadas,
  con cota), el rechazo de válidas por tipo de foto (con cota), U3 por alerta de 3 fotos (general + patente +
  detalle) y el rechazo de vehículos pesados.

U3 por alerta = 1 − (1 − pG)(1 − pP)(1 − pD): supone que los rechazos de las tres fotos son independientes.
Es una estimación: no hay alertas reales, y una correlación entre fotos (el mismo auto, la misma luz) haría
que la unión real sea menor. Las cotas son Clopper-Pearson de una cola al 95 % (3/N sin fallas); la cota de U3
usa las cotas de cada tipo, así que es conservadora. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from capture_sets import SEED, halves
from presence import MissingSignalError, upper_bound

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import BoxLabel, Pixels
from vvi_ia.service.validate_evidence import reduce_for_models

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOOR = 0.01
MEASURE_ATTEMPTS = 3
MODEL_MAX_SIDE_PX = 1024  # MODEL_MAX_SIDE_PX del servicio
MIN_AREA = 0.05  # PRESENCE_MIN_AREA del servicio
GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0,30 a 0,95
CURRENT = {"car": 0.40, "seg": 0.50}  # DETECTOR_MIN_CONF y SEGMENTER_MIN_CONF del servicio
S1_BOUND = 0.01  # cota máxima de S1 al elegir los umbrales
VALID = ("general", "plate", "detail")
HEAVY = ("heavy_Truck", "heavy_Van", "heavy_Bus")
NEGATIVES = "neg"
ALL_SETS = (*VALID, *HEAVY, *(f"{h}_whole" for h in HEAVY), NEGATIVES)

Signals = Mapping[str, Sequence[Sequence[float]]]


def accepted(signals: Signals, t_car: float | None, t_seg: float | None, min_area: float = MIN_AREA) -> bool:
    """R1: ¿hay un vehículo? `None` apaga esa señal (solo detector o solo segmentador)."""
    for name, threshold in (("car", t_car), ("seg", t_seg)):
        if threshold is None:
            continue
        if any(conf >= threshold and area >= min_area for conf, area in signals.get(name, ())):
            return True
    return False


def measure_photo(models: Any, image: Pixels) -> dict[str, list[list[float]]]:
    """Hallazgos del detector y del segmentador sobre la foto reducida como la reduce el servicio."""
    reduced = reduce_for_models(image, MODEL_MAX_SIDE_PX)
    image = image if reduced is None else reduced
    total = float(image.shape[0] * image.shape[1])
    detections = models.vehicle.detect(image)
    cars = models.cars.cars(image)
    missing = [n for n, found in (("detector", detections), ("segmentador", cars)) if found is None]
    if missing:  # un modelo que no responde no es "no encontré nada"
        raise MissingSignalError(", ".join(missing))
    return {
        "car": [[b.conf, b.area / total] for b in detections or () if b.label == BoxLabel.CAR],
        "seg": [[c.conf, float(c.mask.sum()) / total] for c in cars or ()],
    }


def measure(args: argparse.Namespace) -> int:
    try:
        models = build_models(
            {"detector": args.detector, "segmenter": "rfdetr_seg_coco"},
            args.manifest,
            args.models_dir,
            params={
                "detector_min_conf": FLOOR,
                "redaction_min_conf": FLOOR,
                "segmenter_min_conf": FLOOR,
            },
        )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    decoder = CvImageDecoder()
    signals: dict[str, list[dict[str, list[list[float]]]]] = {}
    missing_total = 0
    for name in ALL_SETS:
        files = sorted((Path(args.sets_dir) / name).glob("*.jpg"))
        rows: list[dict[str, list[list[float]]]] = []
        for path in files:
            image = decoder.decode(path.read_bytes())
            for _ in range(MEASURE_ATTEMPTS):
                try:
                    rows.append(measure_photo(models, image))
                    break
                except MissingSignalError:
                    continue
            else:
                missing_total += 1
                rows.append({"car": [], "seg": []})
        signals[name] = rows
        print(f"{name}: {len(rows)} fotos medidas", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "signals.json").write_text(
        json.dumps(
            {"detector": models.versions()["detector"], "senales": signals, "faltantes": missing_total}
        ),
        encoding="utf-8",
    )
    if missing_total:
        print(f"ATENCIÓN: {missing_total} fotos con señales faltantes: la medición NO es válida, repetila.")
        return 2
    return 0


# --- análisis -------------------------------------------------------------------------------------


def rate(failures: int, trials: int) -> dict[str, Any]:
    return {
        "n": trials,
        "fallas": failures,
        "tasa": round(failures / trials, 4) if trials else None,
        "cota_95": upper_bound(failures, trials),
    }


def u3(rates: Mapping[str, Mapping[str, Any]], key: str) -> float | None:
    """Rechazo de una alerta de 3 fotos con rechazos independientes (`key`: `tasa` o `cota_95`)."""
    values = [rates[name][key] for name in VALID]
    if any(v is None for v in values):
        return None
    ok = 1.0
    for value in values:
        ok *= 1.0 - float(value)
    return round(1.0 - ok, 4)


def split(rows: Sequence[Any], seed: int = SEED) -> dict[str, list[Any]]:
    labels = halves(len(rows), seed)
    return {
        "A": [r for r, h in zip(rows, labels, strict=True) if h == "A"],
        "B": [r for r, h in zip(rows, labels, strict=True) if h == "B"],
        "todas": list(rows),
    }


def evaluate(
    signals: Mapping[str, Sequence[Signals]],
    family: Sequence[bool],
    t_car: float | None,
    t_seg: float | None,
) -> dict[str, Any]:
    """Todas las cifras de una configuración de umbrales."""

    def rejected(rows: Sequence[Signals]) -> int:
        return sum(not accepted(r, t_car, t_seg) for r in rows)

    result: dict[str, Any] = {"t_car": t_car, "t_seg": t_seg}
    negatives = split(signals[NEGATIVES])
    result["negativas"] = {
        half: rate(sum(accepted(r, t_car, t_seg) for r in rows), len(rows))
        for half, rows in negatives.items()
    }
    labels = halves(len(signals[NEGATIVES]), SEED)
    for half in ("A", "B"):  # las negativas sin ninguna parte de vehículo anotada (rueda, patente, moto...)
        kept = [
            r for r, h, fam in zip(signals[NEGATIVES], labels, family, strict=True) if h == half and not fam
        ]
        result["negativas"][f"{half}_sin_familia_vehiculo"] = rate(
            sum(accepted(r, t_car, t_seg) for r in kept), len(kept)
        )
    valid = {name: split(signals[name]) for name in VALID}
    result["validas"] = {
        name: {half: rate(rejected(rows), len(rows)) for half, rows in halves_.items()}
        for name, halves_ in valid.items()
    }
    result["u3"] = {
        half: {
            "estimada": u3({n: result["validas"][n][half] for n in VALID}, "tasa"),
            "cota_95": u3({n: result["validas"][n][half] for n in VALID}, "cota_95"),
        }
        for half in ("A", "B", "todas")
    }
    result["pesados"] = {
        name: rate(rejected(signals[name]), len(signals[name]))
        for name in (*HEAVY, *(f"{h}_whole" for h in HEAVY))
    }
    for suffix in ("", "_whole"):
        rows = [r for h in HEAVY for r in signals[f"{h}{suffix}"]]
        result["pesados"][f"total{suffix}"] = rate(rejected(rows), len(rows))
    return result


def sweep(
    signals: Mapping[str, Sequence[Signals]], family: Sequence[bool], seg_off: bool = True
) -> list[dict[str, Any]]:
    seg_values: tuple[float | None, ...] = (*GRID, None) if seg_off else GRID
    return [
        evaluate(signals, family, t_car, t_seg)
        for t_car in (*GRID, None)
        for t_seg in seg_values
        if not (t_car is None and t_seg is None)
    ]


def best(
    configs: Sequence[Mapping[str, Any]],
    seg: float | None | str = "free",
    bound: float = S1_BOUND,
    negatives: str = "A",
) -> Mapping[str, Any] | None:
    """La configuración det ∨ seg con menor U3 en la mitad A y cota de S1 <= `bound` en la mitad A.

    `seg="free"` barre `t_seg`; con un número, lo fija (por ejemplo 0,50, el del servicio).
    """
    candidates = [
        c
        for c in configs
        if c["t_car"] is not None
        and c["t_seg"] is not None
        and (seg == "free" or c["t_seg"] == seg)
        and (c["negativas"][negatives]["cota_95"] or 1.0) <= bound
        and c["u3"]["A"]["estimada"] is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            c["u3"]["A"]["estimada"],
            c["negativas"][negatives]["fallas"],
            -c["t_car"],
            -c["t_seg"],
        ),
    )


def compact(config: Mapping[str, Any]) -> dict[str, Any]:
    """Una fila para la tabla completa: lo necesario para repasar el barrido sin las cotas."""
    return {
        "t_car": config["t_car"],
        "t_seg": config["t_seg"],
        "S1_A": config["negativas"]["A"]["fallas"],
        "S1_B": config["negativas"]["B"]["fallas"],
        "U3_A": config["u3"]["A"]["estimada"],
        "U3_B": config["u3"]["B"]["estimada"],
    }


def report(args: argparse.Namespace) -> int:
    data = json.loads((Path(args.out_dir) / "signals.json").read_text(encoding="utf-8"))
    signals = data["senales"]
    specs = json.loads((Path(args.sets_dir) / "specs.json").read_text(encoding="utf-8"))
    family = [
        bool(s["detail"]["family"]) for s in specs["specs"][NEGATIVES]
    ]  # delatan un vehículo (rueda...)
    configs = sweep(signals, family)
    current = evaluate(signals, family, CURRENT["car"], CURRENT["seg"])
    current_without_seg = evaluate(signals, family, CURRENT["car"], None)
    chosen = {
        "umbrales_actuales": current,
        "actuales_sin_segmentador": current_without_seg,
        "mejor_con_barrido_del_segmentador": best(configs),
        "mejor_con_segmentador_en_0_50": best(configs, seg=CURRENT["seg"]),
        # lectura aparte: S1 <= 1 % medido solo en las negativas sin parte de vehículo anotada
        "mejor_con_barrido_del_segmentador_sin_familia": best(configs, negatives="A_sin_familia_vehiculo"),
        "mejor_con_segmentador_en_0_50_sin_familia": best(
            configs, seg=CURRENT["seg"], negatives="A_sin_familia_vehiculo"
        ),
    }
    out = Path(args.out_dir)
    (out / "r1_capture.json").write_text(
        json.dumps(
            {
                "detector": data["detector"],
                "minimo_de_area": MIN_AREA,
                "cota_de_s1_al_elegir": S1_BOUND,
                "elegidas": chosen,
                "barrido": [compact(c) for c in configs],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Detector {data['detector']}")
    for name, config in chosen.items():
        if config is None:
            print(f"{name}: ninguna configuración cumple S1 <= {S1_BOUND:.0%} (cota) en la mitad A")
            continue
        neg, val, heavy, u = config["negativas"], config["validas"], config["pesados"], config["u3"]
        print(f"[{name}] car >= {config['t_car']} · seg >= {config['t_seg']}")
        print(
            f"  S1 A {neg['A']['fallas']}/{neg['A']['n']} (cota {neg['A']['cota_95']})"
            f" · B {neg['B']['fallas']}/{neg['B']['n']} (cota {neg['B']['cota_95']})"
            f" · sin familia A {neg['A_sin_familia_vehiculo']['fallas']}/{neg['A_sin_familia_vehiculo']['n']}"
            f" B {neg['B_sin_familia_vehiculo']['fallas']}/{neg['B_sin_familia_vehiculo']['n']}"
            f" (cota {neg['B_sin_familia_vehiculo']['cota_95']})"
        )
        for kind in VALID:
            b = val[kind]["B"]
            whole = val[kind]["todas"]
            print(
                f"  {kind:<8} rechazo B {b['fallas']}/{b['n']} (cota {b['cota_95']})"
                f" · todas {whole['fallas']}/{whole['n']}"
            )
        print(f"  U3 B {u['B']['estimada']} (cota {u['B']['cota_95']}) · todas {u['todas']['estimada']}")
        print(
            f"  pesados {heavy['total']['fallas']}/{heavy['total']['n']} (cota {heavy['total']['cota_95']})"
            f" · enteros {heavy['total_whole']['fallas']}/{heavy['total_whole']['n']}"
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure", "report"))
    parser.add_argument("--sets-dir", type=Path, required=True, help="salida de capture_sets.py build")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--detector", default="rfdetr_openimages_v2")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        parser.error("indicá --out-dir o definí EVAL_OUT")
    if args.command == "measure" and args.models_dir is None:
        parser.error("`measure` pide --models-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (args.sets_dir, args.out_dir)):
        print("Las fotos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    return {"measure": measure, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
