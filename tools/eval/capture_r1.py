"""R1 (presencia de vehículo) sobre los conjuntos con el encuadre de 1-2 m de `capture_sets.py`.

R1 es la misma regla para todas las fotos (general, patente, detalle): hay un vehículo si el detector da
una caja de una clase que cuenta (auto, van o camión; el colectivo no cuenta) con confianza >= `t_car`, o
el segmentador una instancia con confianza >= `t_seg`, y ocupa al menos `PRESENCE_MIN_AREA` de la foto
(det ∨ seg). La patente, la rueda y el daño no cuentan, y detalle no tiene regla propia.

- `measure`: corre el detector y el segmentador sobre cada foto (con un piso de confianza de 0,01) y
  guarda, por foto, los hallazgos `[confianza, fracción de la foto]` **de cada clase del detector por
  separado** y del segmentador. Es la única parte que usa los modelos; así, qué clases cuentan se decide
  después, sin volver a medir.
- `report`: desde esas señales barre `t_car` y `t_seg`, elige los umbrales en la mitad A y reporta en la
  mitad B (semilla fija, la de `capture_sets.py`): S1 (negativas aceptadas, con cota, con y sin las
  ambiguas), el rechazo de válidas por tipo de foto (con cota), U3 por alerta de 3 fotos, el rechazo de
  vehículos pesados (camión y van) y de pickups, el de colectivos y ambulancias, y qué dejó pasar a cada
  negativa aceptada (detector o segmentador, y la clase).

U3 por alerta = 1 − (1 − pG)(1 − pP)(1 − pD): supone que los rechazos de las tres fotos son
independientes. Es una estimación: no hay alertas reales, y una correlación entre fotos (el mismo auto,
la misma luz) haría que la unión real sea menor. Las cotas son Clopper-Pearson de una cola al 95 % (3/N
sin fallas); la cota de U3 usa las cotas de cada tipo, así que es conservadora. Uso: ver
`tools/eval/README.md`.
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
from capture_sets import SEED, halves
from presence import MissingSignalError, upper_bound

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import Pixels
from vvi_ia.service.validate_evidence import reduce_for_models

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOOR = 0.01
MEASURE_ATTEMPTS = 10
CHECKPOINT_EVERY = 100  # fotos entre guardados de `signals.json` (`--resume` sigue donde quedó)
MODEL_MAX_SIDE_PX = 1024  # MODEL_MAX_SIDE_PX del servicio
MIN_AREA = 0.05  # PRESENCE_MIN_AREA del servicio
GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0,30 a 0,95
CURRENT = {"car": 0.40, "seg": 0.50}  # los umbrales hasta el 14/11/2026 (el servicio usa ahora `LOWER`)
LOWER = {"car": 0.35, "seg": 0.50}  # DETECTOR_MIN_CONF y SEGMENTER_MIN_CONF del servicio desde el 14/11/2026
S1_BOUND = 0.01  # cota máxima de S1 al elegir los umbrales
COUNTED = ("car", "truck", "van")  # clases del detector que cuentan como vehículo (`ROLES` del adaptador)
VALID = ("general", "plate", "detail")
ALERT_2 = ("general", "plate")  # una alerta de 2 fotos (sin la de detalle, que probablemente se elimine)
HEAVY = ("heavy_Truck", "heavy_Van")  # camiones y vans válidos
PICKUP = "pickup"
NOT_VEHICLES = ("colectivo", "ambulancia")  # negativas: se mide cuántas se rechazan
NEGATIVES = "sinvehiculo"
WHOLE = tuple(f"{n}_whole" for n in (*HEAVY, PICKUP, *NOT_VEHICLES))
ALL_SETS = (*VALID, *HEAVY, PICKUP, *NOT_VEHICLES, *WHOLE, NEGATIVES)
AMBIGUOUS = (
    "ambigua"  # `tipo` de una negativa cuya única señal de vehículo es una rueda, una patente o un `Vehicle`
)

Signals = Mapping[str, Any]  # {"det": {clase: [[conf, área]]}, "seg": [[conf, área]]}


def passing(
    signals: Signals,
    t_car: float | None,
    t_seg: float | None,
    min_area: float = MIN_AREA,
    classes: Sequence[str] = COUNTED,
) -> list[tuple[str, str, float, float]]:
    """Las detecciones que dejan pasar la foto: `(fuente, clase, confianza, área)`. `None` apaga esa señal."""
    found: list[tuple[str, str, float, float]] = []
    if t_car is not None:
        for name in classes:
            found += [
                ("detector", name, conf, area)
                for conf, area in signals["det"].get(name, ())
                if conf >= t_car and area >= min_area
            ]
    if t_seg is not None:
        found += [
            ("segmentador", "car", conf, area)
            for conf, area in signals["seg"]
            if conf >= t_seg and area >= min_area
        ]
    return found


def accepted(
    signals: Signals,
    t_car: float | None,
    t_seg: float | None,
    min_area: float = MIN_AREA,
    classes: Sequence[str] = COUNTED,
) -> bool:
    """R1: ¿hay un vehículo? `None` apaga esa señal (solo detector o solo segmentador)."""
    if t_car is not None and any(
        conf >= t_car and area >= min_area for name in classes for conf, area in signals["det"].get(name, ())
    ):
        return True
    return t_seg is not None and any(conf >= t_seg and area >= min_area for conf, area in signals["seg"])


def measure_photo(models: Any, image: Pixels) -> dict[str, Any]:
    """Hallazgos del detector (por clase) y del segmentador, sobre la foto ya reducida."""
    reduced = reduce_for_models(image, MODEL_MAX_SIDE_PX)
    image = image if reduced is None else reduced
    total = float(image.shape[0] * image.shape[1])
    by_class = models.vehicle.detect_by_class(image)
    cars = models.cars.cars(image)
    missing = [n for n, found in (("detector", by_class), ("segmentador", cars)) if found is None]
    if missing:  # un modelo que no responde no es "no encontré nada"
        raise MissingSignalError(", ".join(missing))
    return {
        "det": {name: [[b.conf, b.area / total] for b in (by_class or {}).get(name, ())] for name in COUNTED},
        "seg": [[c.conf, float(c.mask.sum()) / total] for c in cars or ()],
    }


def measure_with_retries(models: Any, decoder: Any, path: Path) -> dict[str, Any] | None:
    """Mide una foto; ante un modelo que no responde o falta de memoria de la máquina, espera y reintenta.

    `None` si no se pudo: esa foto cuenta como señal faltante y la medición no es válida.
    """
    for attempt in range(MEASURE_ATTEMPTS):
        try:
            return measure_photo(models, decoder.decode(path.read_bytes()))
        except (MissingSignalError, MemoryError, cv2.error, OSError):
            gc.collect()
            time.sleep(min(5.0 * (attempt + 1), 30.0))
    return None


def measure(args: argparse.Namespace) -> int:
    try:
        models = build_models(
            {"detector": args.detector, "segmenter": "rfdetr_seg_coco"},
            args.manifest,
            args.models_dir,
            params={"detector_min_conf": FLOOR, "redaction_min_conf": FLOOR, "segmenter_min_conf": FLOOR},
        )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    decoder = CvImageDecoder()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "signals.json"
    version = models.versions()["detector"]
    signals: dict[str, list[Any]] = {}
    missing_by_set: dict[str, int] = {}
    done: list[str] = []

    def save() -> None:
        target.write_text(
            json.dumps(
                {
                    "detector": version,
                    "senales": signals,
                    "faltantes_por_conjunto": missing_by_set,
                    "completos": done,
                }
            ),
            encoding="utf-8",
        )

    if args.resume and target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("detector") == version:
            signals = previous["senales"]
            missing_by_set = previous.get("faltantes_por_conjunto", {})
            done = previous.get("completos", [])
    for name in ALL_SETS:
        if name in done and not missing_by_set.get(name):
            print(f"{name}: {len(signals[name])} fotos (ya medidas)", flush=True)
            continue
        files = sorted((Path(args.sets_dir) / name).glob("*.jpg"))
        # un conjunto a medias sigue donde quedó; uno completo con señales faltantes se repite entero
        rows: list[Any] = [] if name in done else list(signals.get(name, []))
        missing = 0 if name in done else missing_by_set.get(name, 0)
        done = [d for d in done if d != name]
        for path in files[len(rows) :]:
            result = measure_with_retries(models, decoder, path)
            if result is None:
                missing += 1
                result = {"det": {n: [] for n in COUNTED}, "seg": []}
            rows.append(result)
            if len(rows) % CHECKPOINT_EVERY == 0:
                signals[name], missing_by_set[name] = rows, missing
                save()
        signals[name], missing_by_set[name] = rows, missing
        done.append(name)
        save()
        print(f"{name}: {len(rows)} fotos medidas, {missing} con señales faltantes", flush=True)
    missing_total = sum(missing_by_set.values())
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


def u3(rates: Mapping[str, Mapping[str, Any]], key: str, names: Sequence[str] = VALID) -> float | None:
    """Rechazo de una alerta con rechazos independientes entre sus fotos (`key`: `tasa` o `cota_95`).

    Con las fotos de `VALID` (general, patente y detalle) es U3; con `ALERT_2` (general y patente), U2.
    """
    values = [rates[name][key] for name in names]
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
    ambiguous: Sequence[bool],
    t_car: float | None,
    t_seg: float | None,
) -> dict[str, Any]:
    """Todas las cifras de una configuración de umbrales. `ambiguous` marca las negativas ambiguas."""

    def rejected(rows: Sequence[Signals]) -> int:
        return sum(not accepted(r, t_car, t_seg) for r in rows)

    def by_half(name: str, count: Any) -> dict[str, dict[str, Any]]:
        return {half: rate(count(rows), len(rows)) for half, rows in split(signals[name]).items()}

    result: dict[str, Any] = {"t_car": t_car, "t_seg": t_seg}
    negatives = signals[NEGATIVES]
    result["negativas"] = by_half(NEGATIVES, lambda rows: sum(accepted(r, t_car, t_seg) for r in rows))
    labels = halves(len(negatives), SEED)
    for half, key in (("A", "A_sin_ambiguas"), ("B", "B_sin_ambiguas"), (None, "todas_sin_ambiguas")):
        kept = [
            r
            for r, h, amb in zip(negatives, labels, ambiguous, strict=True)
            if not amb and (half is None or h == half)
        ]
        result["negativas"][key] = rate(sum(accepted(r, t_car, t_seg) for r in kept), len(kept))
    result["validas"] = {name: by_half(name, rejected) for name in VALID}
    result["u2"] = {
        half: {
            "estimada": u3({n: result["validas"][n][half] for n in ALERT_2}, "tasa", ALERT_2),
            "cota_95": u3({n: result["validas"][n][half] for n in ALERT_2}, "cota_95", ALERT_2),
        }
        for half in ("A", "B", "todas")
    }
    result["u3"] = {
        half: {
            "estimada": u3({n: result["validas"][n][half] for n in VALID}, "tasa"),
            "cota_95": u3({n: result["validas"][n][half] for n in VALID}, "cota_95"),
        }
        for half in ("A", "B", "todas")
    }
    # vehículos válidos pesados: rechazo (cada conjunto en sus mitades; el total suma las mitades de cada uno)
    heavy_names = (PICKUP, *HEAVY, f"{PICKUP}_whole", *(f"{h}_whole" for h in HEAVY))
    heavy = {name: by_half(name, rejected) for name in heavy_names}
    for suffix in ("", "_whole"):
        names = [f"{h}{suffix}" for h in HEAVY]
        heavy[f"camion_van{suffix}"] = {
            half: rate(sum(heavy[n][half]["fallas"] for n in names), sum(heavy[n][half]["n"] for n in names))
            for half in ("A", "B", "todas")
        }
    result["pesados"] = heavy
    # colectivos y ambulancias: negativas; "fallas" son las rechazadas (lo que se espera)
    result["otros_vehiculos"] = {
        name: by_half(name, rejected) for name in (*NOT_VEHICLES, *(f"{n}_whole" for n in NOT_VEHICLES))
    }
    return result


def accepted_details(
    signals: Sequence[Signals],
    specs: Sequence[Mapping[str, Any]],
    half: str,
    t_car: float | None,
    t_seg: float | None,
) -> dict[str, Any]:
    """Qué dejó pasar a cada negativa aceptada de una mitad: detector o segmentador, y la clase."""
    labels = halves(len(signals), SEED)
    rows: list[dict[str, Any]] = []
    for index, (row, spec, which) in enumerate(zip(signals, specs, labels, strict=True)):
        found = passing(row, t_car, t_seg) if which == half else []
        if not found:
            continue
        sources = {f[0] for f in found}
        rows.append(
            {
                "indice": index,
                "tipo": spec["detail"]["tipo"],
                "etiquetas": spec["detail"]["etiquetas"],
                "decide": "ambos" if len(sources) == 2 else next(iter(sources)),
                "detector": [
                    {"clase": c, "conf": round(conf, 3), "area": round(area, 3)}
                    for s, c, conf, area in found
                    if s == "detector"
                ],
                "segmentador": [
                    {"conf": round(conf, 3), "area": round(area, 3)}
                    for s, _c, conf, area in found
                    if s == "segmentador"
                ],
            }
        )
    by_decider = Counter(r["decide"] for r in rows)
    by_class = Counter(d["clase"] for r in rows for d in r["detector"])
    return {
        "aceptadas": len(rows),
        "por_decisor": dict(by_decider),
        "clases_del_detector": dict(by_class),
        "ambiguas": sum(r["tipo"] == AMBIGUOUS for r in rows),
        "fotos": rows,
    }


def sweep(
    signals: Mapping[str, Sequence[Signals]], ambiguous: Sequence[bool], seg_off: bool = True
) -> list[dict[str, Any]]:
    seg_values: tuple[float | None, ...] = (*GRID, None) if seg_off else GRID
    return [
        evaluate(signals, ambiguous, t_car, t_seg)
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

    `seg="free"` barre `t_seg`; con un número, lo fija (por ejemplo 0,50, el del servicio). `negatives`
    dice qué negativas de la mitad A cuentan: `A` (todas) o `A_sin_ambiguas`.
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


def chosen_configs(
    signals: Mapping[str, Sequence[Signals]], ambiguous: Sequence[bool]
) -> tuple[dict[str, Mapping[str, Any] | None], list[dict[str, Any]]]:
    """Las configuraciones que se reportan y el barrido completo."""
    configs = sweep(signals, ambiguous)
    fixed = CURRENT["seg"]
    chosen: dict[str, Mapping[str, Any] | None] = {
        "car_0_40_actual": evaluate(signals, ambiguous, CURRENT["car"], CURRENT["seg"]),
        "car_0_35": evaluate(signals, ambiguous, LOWER["car"], LOWER["seg"]),
        "mejor_con_barrido_del_segmentador": best(configs),
        "mejor_con_segmentador_en_0_50": best(configs, seg=fixed),
        # lectura aparte: el S1 <= 1 % se exige solo sobre las negativas sin las ambiguas
        "mejor_sin_ambiguas_con_barrido_del_segmentador": best(configs, negatives="A_sin_ambiguas"),
        "mejor_sin_ambiguas_con_segmentador_en_0_50": best(configs, seg=fixed, negatives="A_sin_ambiguas"),
    }
    return chosen, configs


def report(args: argparse.Namespace) -> int:
    data = json.loads((Path(args.out_dir) / "signals.json").read_text(encoding="utf-8"))
    signals = data["senales"]
    specs = json.loads((Path(args.sets_dir) / "specs.json").read_text(encoding="utf-8"))["specs"]
    negative_specs = specs[NEGATIVES]
    ambiguous = [s["detail"]["tipo"] == AMBIGUOUS for s in negative_specs]
    chosen, configs = chosen_configs(signals, ambiguous)
    details = {  # qué dejó pasar a cada negativa aceptada en la mitad B
        name: (
            None
            if config is None
            else accepted_details(signals[NEGATIVES], negative_specs, "B", config["t_car"], config["t_seg"])
        )
        for name, config in chosen.items()
    }
    out = Path(args.out_dir)
    (out / "r1_capture.json").write_text(
        json.dumps(
            {
                "detector": data["detector"],
                "minimo_de_area": MIN_AREA,
                "clases_que_cuentan": list(COUNTED),
                "cota_de_s1_al_elegir": S1_BOUND,
                "elegidas": chosen,
                "negativas_aceptadas_en_B": details,
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
        neg, val, heavy, other, u = (
            config["negativas"],
            config["validas"],
            config["pesados"],
            config["otros_vehiculos"],
            config["u3"],
        )
        print(f"[{name}] car >= {config['t_car']} · seg >= {config['t_seg']}")
        print(
            f"  S1 A {neg['A']['fallas']}/{neg['A']['n']} (cota {neg['A']['cota_95']})"
            f" · B {neg['B']['fallas']}/{neg['B']['n']} (cota {neg['B']['cota_95']})"
            f" · B sin ambiguas {neg['B_sin_ambiguas']['fallas']}/{neg['B_sin_ambiguas']['n']}"
            f" (cota {neg['B_sin_ambiguas']['cota_95']})"
        )
        for kind in VALID:
            b = val[kind]["B"]
            print(f"  {kind:<8} rechazo B {b['fallas']}/{b['n']} (cota {b['cota_95']})")
        two = config["u2"]
        print(f"  U3 B {u['B']['estimada']} (cota {u['B']['cota_95']})")
        print(
            f"  U2 (general + patente) B {two['B']['estimada']} (cota {two['B']['cota_95']})"
            f" · todas {two['todas']['estimada']} (cota {two['todas']['cota_95']})"
        )
        for kind in ("camion_van", PICKUP, "heavy_Truck", "heavy_Van"):
            b = heavy[kind]["B"]
            print(f"  {kind:<12} rechazo B {b['fallas']}/{b['n']} (cota {b['cota_95']})")
        for kind in NOT_VEHICLES:
            b = other[kind]["B"]
            everything = other[kind]["todas"]
            print(
                f"  {kind:<12} rechazadas B {b['fallas']}/{b['n']} ({b['tasa']})"
                f" · todas {everything['fallas']}/{everything['n']}"
            )
        d = details[name]
        if d is not None:
            print(
                f"  aceptadas en B: {d['aceptadas']} · decide {d['por_decisor']}"
                f" · clases {d['clases_del_detector']}"
            )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure", "report"))
    parser.add_argument(
        "--sets-dir", type=Path, required=True, help="salida de capture_sets.py build y extras"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--detector", default="rfdetr_openimages_v2")
    parser.add_argument(
        "--resume", action="store_true", help="no repite los conjuntos ya medidos y completos"
    )
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
