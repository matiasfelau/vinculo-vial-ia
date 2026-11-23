"""Presencia de auto en cada foto: ¿qué regla no deja pasar un no-auto (A) ni rechaza un primer plano (C)?

**Histórico (12/11/2026):** mide una regla de presencia anterior (la patente o la rueda como señal, CarDD como
foto de detalle) con exclusiones manuales de negativas. La vigente es `capture_sets.py` y `capture_r1.py`.


Mide, con los adaptadores reales, las señales que puede usar el servicio para decidir que en una foto hay un
auto: `car`, `wheel` y `plate` del detector, una instancia de auto del segmentador y una caja de daño. Con
esas señales evalúa tres reglas, de la más simple a la más amplia, contra cuatro conjuntos:

- **general** y **context** (hace falta `car`): fotos de autos enteros de Open Images (las de `damage_fp.py`).
- **plate**: recortes de patentes del dataset de Roboflow, con un margen alrededor.
- **detail**: fotos de prueba de CarDD (primeros planos de la carrocería).
- **negativas**: fotos de Open Images **sin** `Car`, `Vehicle registration plate` ni `Wheel` (regla A: ninguna
  debe pasar).

Reglas para plate y detail: `R1` = car, wheel o plate del detector; `R2` = R1 o un auto segmentado; `R3` = R2
o una caja de daño. Las fotos y la salida quedan fuera del repo; la consola muestra solo cifras. Uso: ver
`tools/eval/README.md`.
"""

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import BoxLabel, Pixels
from vvi_ia.service.validate_evidence import reduce_for_models

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = "_annotations.coco.json"
FLOOR = 0.01
CROP_MARGIN = 0.5  # los recortes de patentes llevan este margen (de su ancho y alto) alrededor
MIN_CROP_SIDE = 16
SIGNALS = ("car", "wheel", "plate", "seg", "damage")
RULES: Mapping[str, tuple[str, ...]] = {
    "R1": ("car", "wheel", "plate"),
    "R2": ("car", "wheel", "plate", "seg"),
    "R3": ("car", "wheel", "plate", "seg", "damage"),
}
DEFAULT_THRESHOLDS = {"car": 0.40, "wheel": 0.40, "plate": 0.40, "seg": 0.50, "damage": 0.70}
AREA_SWEEP = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)  # fracción mínima de la foto que ocupa la señal
MEASURE_ATTEMPTS = 3  # veces que se reintenta una foto si un modelo no responde
SETS = ("general", "plate", "detail", "negativas")
# La regla R1 del servicio, por rol (`_has_car`): GENERAL, CONTEXT y DETAIL piden un auto (detector o
# segmentador); solo PLATE acepta también una patente. La rueda salió de la regla y la patente salió de
# GENERAL y DETAIL (una patente sola no valida una foto: alcanza una arrancada o impresa). DETAIL suma una
# caja de daño con `DAMAGE_PRESENCE_T` (sin área mínima). El área mínima es `PRESENCE_MIN_AREA`.
SERVICE_RULE: Mapping[str, tuple[str, ...]] = {
    "general": ("car", "seg"),
    "plate": ("car", "plate", "seg"),
    "detail": ("car", "seg"),
}
SERVICE_MIN_AREA = 0.05
OPEN_IMAGES = "openimages-detector"
NEGATIVE_EXCLUDED = ("car", "plate", "wheel")  # la foto no puede tener ninguna de estas clases anotadas


Signals = Mapping[
    str, Sequence[Sequence[float]]
]  # por señal, cada hallazgo es [confianza, fracción de la foto]


def passes(
    signals: Signals, rule: Sequence[str], thresholds: Mapping[str, float], min_area: float = 0.0
) -> bool:
    """¿Alguna señal de la regla llega a su umbral de confianza y ocupa al menos `min_area` de la foto?"""
    return any(
        conf >= thresholds[name] and area >= min_area for name in rule for conf, area in signals.get(name, ())
    )


def rates(
    scores: Mapping[str, Sequence[Signals]],
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
    min_area: float = 0.0,
) -> dict[str, Any]:
    """Por conjunto, qué fracción pasa como auto con la regla de `general` (solo `car`) y con R1, R2 y R3."""
    out: dict[str, Any] = {}
    for name, rows in scores.items():
        total = len(rows)
        row: dict[str, Any] = {"fotos": total}
        row["solo_car"] = sum(passes(s, ("car",), thresholds, min_area) for s in rows)
        for rule, signals in RULES.items():
            row[rule] = sum(passes(s, signals, thresholds, min_area) for s in rows)
        out[name] = row
    return out


def bound(n: int) -> float | None:
    """Cota superior del 95 % con 0 fallas en `n` casos (regla del 3)."""
    return round(min(1.0, 3 / n), 4) if n else None


def calibrate_damage_presence(
    negatives: Sequence[Signals], rejected_details: Sequence[Signals]
) -> dict[str, Any]:
    """`DAMAGE_PRESENCE_T`: el primer valor (a 3 decimales) sobre la mayor confianza de daño sin auto.

    Con ese umbral ninguna foto sin auto pasa por presencia de daño (regla absoluta: cota 95 % = 3/N).
    Después se cuenta cuántos primeros planos que las otras reglas rechazaban se recuperan con él.
    """

    def best(signals: Signals) -> float:
        return max((conf for conf, _ in signals.get("damage", ())), default=0.0)

    highest = max((best(s) for s in negatives), default=0.0)
    threshold = math.ceil((highest + 1e-9) * 1000) / 1000
    return {
        "DAMAGE_PRESENCE_T": threshold if threshold <= 1.0 else None,
        "negativas": len(negatives),
        "negativas_que_pasan": sum(best(s) >= threshold for s in negatives),
        "cota_95": bound(len(negatives)),
        "mayor_confianza_de_una_negativa": round(highest, 4),
        "detalles_rechazados_por_las_otras_reglas": len(rejected_details),
        "detalles_recuperados": sum(best(s) >= threshold for s in rejected_details),
    }


def upper_bound(failures: int, trials: int, confidence: float = 0.95) -> float | None:
    """Cota superior de la tasa de fallas: 3/N con 0 fallas (regla del 3); con más, Clopper-Pearson exacta."""
    if trials <= 0:
        return None
    if failures == 0:
        return round(min(1.0, 3 / trials), 4)
    if failures >= trials:
        return 1.0
    alpha = 1.0 - confidence
    low, high = failures / trials, 1.0
    for _ in range(60):  # bisección: la p más alta cuya cola binomial P(X <= fallas) todavía es >= alpha
        mid = (low + high) / 2
        tail = sum(math.comb(trials, i) * mid**i * (1 - mid) ** (trials - i) for i in range(failures + 1))
        low, high = (mid, high) if tail >= alpha else (low, mid)
    return round((low + high) / 2, 4)


def service_rules(
    segmenter: bool = True, wheel: bool = False, detail_plate: bool = False
) -> dict[str, tuple[str, ...]]:
    """La regla `SERVICE_RULE`, con o sin la señal del segmentador.

    `wheel=True` le suma la clase rueda en PLATE y DETAIL y `detail_plate=True` la patente en DETAIL: juntas
    reconstruyen la regla anterior a las decisiones del 11/11/2026 (para comparar con las tablas viejas).
    """
    rules = {
        role: tuple(name for name in signals if segmenter or name != "seg")
        for role, signals in SERVICE_RULE.items()
    }
    if detail_plate:
        rules["detail"] = ("car", "plate", *rules["detail"][1:])
    if wheel:
        for role in ("plate", "detail"):
            rules[role] = ("car", "wheel", *rules[role][1:])
    return rules


def service_rule(
    scores: Mapping[str, Sequence[Signals]],
    damage_t: float,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
    min_area: float = SERVICE_MIN_AREA,
    segmenter: bool = True,
    damage: bool = True,
    wheel: bool = False,
    detail_plate: bool = False,
) -> dict[str, Any]:
    """R1 tal como la aplica el servicio, por rol: fotos válidas rechazadas (C) y negativas que pasan (A).

    `segmenter=False` quita la señal del segmentador, `damage=False` la caja de daño en DETAIL y
    `wheel=True` le suma la clase rueda y `detail_plate=True` la patente en DETAIL (la regla anterior). Las
    negativas se miden en dos tandas: las de siempre y, si las hay, las que el detector nunca vio.
    """
    rules = service_rules(segmenter, wheel, detail_plate)

    def accepts(role: str, signals: Signals) -> bool:
        if passes(signals, rules[role], thresholds, min_area):
            return True
        return damage and role == "detail" and any(conf >= damage_t for conf, _ in signals.get("damage", ()))

    out: dict[str, Any] = {}
    for role in rules:
        rows = scores.get(role, ())
        rejected = sum(not accepts(role, signals) for signals in rows)
        entry: dict[str, Any] = {
            "fotos": len(rows),
            "rechazadas": rejected,
            "tasa_de_rechazo": round(rejected / len(rows), 4) if rows else None,
        }
        negatives = scores.get("negativas", ())
        passing = sum(accepts(role, signals) for signals in negatives)
        entry["negativas"] = len(negatives)
        entry["negativas_que_pasan"] = passing
        entry["cota_95_negativas"] = upper_bound(passing, len(negatives))
        fresh = scores.get("negativas_nuevas")
        if fresh is not None:  # las que el detector nunca vio al entrenar
            fresh_passing = sum(accepts(role, signals) for signals in fresh)
            entry["negativas_nuevas"] = {
                "fotos": len(fresh),
                "que_pasan": fresh_passing,
                "cota_95": upper_bound(fresh_passing, len(fresh)),
            }
        out[role] = entry
    return out


def service_comparison(scores: Mapping[str, Sequence[Signals]], damage_t: float) -> dict[str, Any]:
    """La regla del servicio con el segmentador (como estaba), sin él, y sin él ni el daño."""
    return {
        "con_segmentador": service_rule(scores, damage_t),
        "sin_segmentador": service_rule(scores, damage_t, segmenter=False),
        "sin_segmentador_ni_dano": service_rule(scores, damage_t, segmenter=False, damage=False),
    }


def verdict(table: Mapping[str, Any]) -> dict[str, Any]:
    """Qué reglas cumplen A (ninguna negativa pasa) y C (ninguna foto válida se rechaza)."""
    result: dict[str, Any] = {}
    negatives = table["negativas"]
    for rule in ("solo_car", *RULES):
        a_fail = negatives[rule]
        roles = ("general",) if rule == "solo_car" else ("plate", "detail")
        c_fail = {role: table[role]["fotos"] - table[role][rule] for role in roles if role in table}
        result[rule] = {
            "negativas_que_pasan": a_fail,
            "validas_rechazadas_por_rol": c_fail,
            "cumple_A": a_fail == 0,
            "cumple_C": all(v == 0 for v in c_fail.values()),
        }
    return result


class MissingSignalError(RuntimeError):
    """Un modelo no respondió (devolvió `None`) sobre una foto: la medición quedaría incompleta."""


def measure(
    models: Any, decoder: CvImageDecoder, image: Pixels, max_side: int = 0
) -> dict[str, list[list[float]]]:
    """Todos los hallazgos de cada señal en la foto: `[confianza, fracción de la foto que ocupa]`.

    Con `max_side` la foto se reduce a ese lado mayor antes de los modelos, como hace el servicio
    (`MODEL_MAX_SIDE_PX`); las fracciones de área no dependen de la escala.
    """
    if max_side:
        reduced = reduce_for_models(image, max_side)
        image = image if reduced is None else reduced
    total = float(image.shape[0] * image.shape[1])
    signals: dict[str, list[list[float]]] = {name: [] for name in SIGNALS}
    detections = models.vehicle.detect(image)
    cars = models.cars.cars(image)
    damage = models.damage.detect(image)
    missing = [
        name
        for name, found in (("detector", detections), ("segmentador", cars), ("daño", damage))
        if found is None
    ]
    if missing:  # un modelo que no responde no es "no encontré nada": la señal faltaría en silencio
        raise MissingSignalError(", ".join(missing))
    for box in detections or ():
        if box.label in (BoxLabel.CAR, BoxLabel.WHEEL, BoxLabel.PLATE):
            signals[str(box.label)].append([box.conf, box.area / total])
    for car in cars or ():
        signals["seg"].append([car.conf, float(car.mask.sum()) / total])
    for box in damage or ():
        signals["damage"].append([box.conf, box.area / total])
    return signals


def plate_crops(folder: Path, limit: int | None = None) -> list[NDArray[np.uint8]]:
    """Recortes de las patentes anotadas (con margen), como si fueran la foto de rol PLATE."""
    decoder = CvImageDecoder()
    crops: list[Pixels] = []
    for split in sorted(p for p in folder.iterdir() if (p / ANNOTATIONS).is_file()):
        coco = json.loads((split / ANNOTATIONS).read_text(encoding="utf-8"))
        files = {i["id"]: i["file_name"] for i in coco["images"]}
        cache: dict[int, Pixels] = {}
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            if min(w, h) < MIN_CROP_SIDE:
                continue
            if ann["image_id"] not in cache:
                cache = {ann["image_id"]: decoder.decode((split / files[ann["image_id"]]).read_bytes())}
            image = cache[ann["image_id"]]
            x1, y1 = max(0, int(x - CROP_MARGIN * w)), max(0, int(y - CROP_MARGIN * h))
            x2, y2 = (
                min(image.shape[1], int(x + w * (1 + CROP_MARGIN))),
                min(image.shape[0], int(y + h * (1 + CROP_MARGIN))),
            )
            crops.append(image[y1:y2, x1:x2].copy())
    return crops[:limit]


def negative_photos(root: Path) -> list[Path]:
    """Fotos de Open Images sin ninguna de las clases de `NEGATIVE_EXCLUDED` anotada."""
    out: list[Path] = []
    for split in ("train", "valid", "test"):
        path = root / OPEN_IMAGES / split
        if not (path / ANNOTATIONS).is_file():
            continue
        coco = json.loads((path / ANNOTATIONS).read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in coco["categories"]}
        present: dict[int, set[str]] = {}
        for ann in coco["annotations"]:
            present.setdefault(ann["image_id"], set()).add(names[ann["category_id"]])
        out += [
            path / i["file_name"]
            for i in coco["images"]
            if not present.get(i["id"], set()) & set(NEGATIVE_EXCLUDED)
        ]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--eval-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument(
        "--detector", default="rfdetr_openimages", help="modelo del puerto detector (v1 o v2)"
    )
    parser.add_argument("--limit", type=int, help="fotos por conjunto (para probar rápido)")
    parser.add_argument(
        "--model-max-side",
        type=int,
        default=1024,
        help="lado al que se reduce cada foto antes de los modelos (MODEL_MAX_SIDE_PX; 0 = sin reducir)",
    )
    parser.add_argument("--fresh-limit", type=int, help="máximo de negativas nuevas")
    parser.add_argument(
        "--fresh-negatives",
        type=Path,
        help="carpeta con fotos de Open Images sin auto que el detector nunca vio (negativas_nuevas)",
    )
    args = parser.parse_args(argv)
    for flag, value in (
        ("--data-dir", args.data_dir),
        ("--eval-dir", args.eval_dir),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (args.data_dir, args.eval_dir, args.out_dir)):
        print("Las fotos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    try:
        models = build_models(
            {"detector": args.detector, "segmenter": "rfdetr_seg_coco", "damage": "rfdetr_cardd"},
            args.manifest,
            args.models_dir,
            params={
                "detector_min_conf": FLOOR,
                "redaction_min_conf": FLOOR,
                "segmenter_min_conf": FLOOR,
                "damage_min_conf": FLOOR,
            },
        )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    decoder = CvImageDecoder()
    data = Path(args.data_dir)
    sources: dict[str, list[Any]] = {
        "general": sorted((data / "damage-fp" / "images").glob("*.jpg")),
        "plate": plate_crops(Path(args.eval_dir) / "ocr-roboflow"),
        "detail": sorted((data / "cardd-detector" / "test").glob("*.jpg")),
        "negativas": negative_photos(data),
    }
    if args.fresh_negatives:
        fresh = sorted(Path(args.fresh_negatives).glob("*.jpg"))
        sources["negativas_nuevas"] = fresh[: args.fresh_limit] if args.fresh_limit else fresh
    scores: dict[str, list[dict[str, list[list[float]]]]] = {}

    def measure_item(item: Any) -> dict[str, list[list[float]]]:
        pixels = item if isinstance(item, np.ndarray) else decoder.decode(Path(item).read_bytes())
        return measure(models, decoder, pixels, args.model_max_side)

    failures: dict[str, int] = {}

    def measure_retrying(name: str, item: Any) -> dict[str, list[list[float]]] | None:
        for _ in range(MEASURE_ATTEMPTS):
            try:
                return measure_item(item)
            except MissingSignalError:
                continue
        failures[name] = failures.get(name, 0) + 1
        return None

    for name, items in sources.items():
        items = items[: args.limit] if args.limit else items
        measured = [measure_retrying(name, item) for item in items]
        scores[name] = [m if m is not None else {signal: [] for signal in SIGNALS} for m in measured]
        print(
            f"{name}: {len(scores[name])} fotos medidas, {failures.get(name, 0)} con señales faltantes",
            flush=True,
        )
    table = rates(scores)
    rejected = [s for s in scores["detail"] if not passes(s, RULES["R2"], DEFAULT_THRESHOLDS, 0.05)]
    damage_presence = calibrate_damage_presence(scores["negativas"], rejected)
    sweep = {str(area): rates(scores, min_area=area) for area in AREA_SWEEP}
    service = service_comparison(scores, damage_presence["DAMAGE_PRESENCE_T"] or 1.0)
    report = {
        "umbrales": DEFAULT_THRESHOLDS,
        "tabla": table,
        "reglas": verdict(table),
        "barrido_de_area": {area: {"tabla": rows, "reglas": verdict(rows)} for area, rows in sweep.items()},
        "presencia_por_dano": damage_presence,
        "regla_del_servicio": {
            "DAMAGE_PRESENCE_T": damage_presence["DAMAGE_PRESENCE_T"],
            "configuraciones": service,
        },
        "senales_faltantes": failures,
        "senales": scores,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "presence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{'conjunto':<11}{'fotos':>6}{'solo car':>10}{'R1':>6}{'R2':>6}{'R3':>6}")
    for name, row in table.items():
        print(f"{name:<11}{row['fotos']:>6}{row['solo_car']:>10}{row['R1']:>6}{row['R2']:>6}{row['R3']:>6}")
    for rule, row in report["reglas"].items():
        print(f"{rule}: {row}")
    print(f"Presencia por daño en DETAIL: {damage_presence}")
    print("R1 del servicio por rol (área >= 5 %, daño >= DAMAGE_PRESENCE_T en DETAIL):")
    for config, roles in service.items():
        print(f"  [{config}]")
        for role, row in roles.items():
            fresh = row.get("negativas_nuevas")
            extra = (
                f"vistas {row['negativas_que_pasan']}/{row['negativas']} (cota {row['cota_95_negativas']})"
            )
            if fresh:
                extra += f" · nuevas {fresh['que_pasan']}/{fresh['fotos']} (cota {fresh['cota_95']})"
            print(f"    {role:<8} válidas rechazadas {row['rechazadas']}/{row['fotos']} · negativas {extra}")
    print("Barrido del área mínima (fracción de la foto): negativas que pasan / válidas rechazadas")
    for area, rows in sweep.items():
        verdicts = verdict(rows)
        cells = [
            f"{rule} {row['negativas_que_pasan']}/{sum(row['validas_rechazadas_por_rol'].values())}"
            for rule, row in verdicts.items()
        ]
        print(f"  area >= {area}: " + " · ".join(cells))
    if failures:
        print(f"ATENCIÓN: fotos con señales faltantes {failures}: la medición NO es válida, repetila.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
