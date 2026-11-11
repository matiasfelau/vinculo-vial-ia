"""R1 con la regla nueva de patente: P1 contra P2, por nivel de zoom, y U3 por alerta de tres fotos.

Decisión del 11/11/2026: una patente sola **no valida** una foto (alcanza una patente arrancada o impresa para
hacer pasar cualquier imagen). En GENERAL y DETAIL la patente deja de contar como presencia: se pide un auto
(detector, segmentador o los dos). En PLATE hay dos variantes, que este script mide sin aplicar ninguna:

- **P1** (la actual): un auto **o** una patente.
- **P2**: una patente **y** un auto.

Como en `presence_grid.py`, no vuelve a inferir: lee las señales ya medidas (`presence.py` para las fotos
válidas de siempre y las negativas, `plate_zoom.py` para los primeros planos de patente a 25, 40 y 60 %) y
evalúa cualquier combinación de decisor (`det`, `seg`, `or` = det ∨ seg, `and`), variante y umbrales. Los
umbrales de auto, de segmentador y de patente se barren desde 0,30 de a 0,05 hasta 0,95; el área mínima
de cada hallazgo es la del servicio (5 % de la foto). Son umbrales **solo de esta medición**: no se aplican.

**U3 por alerta.** Una alerta trae GENERAL + PLATE + un DETAIL. Se estima la probabilidad de que R1 rechace
alguna de las tres como `1 - (1 - pG)(1 - pP)(1 - pD)`, **suponiendo que las fotos de una alerta fallan de
forma independiente**. Es una estimación: las fotos salen de conjuntos distintos (no hay alertas reales) y, en
la práctica, las fallas de una misma alerta pueden estar correlacionadas (el mismo auto, la misma luz), lo que
haría que la unión real sea menor. La cota usa las cotas de cada foto (Clopper-Pearson 95 %) en la misma
fórmula, así que es conservadora, no un intervalo exacto.
"""

import argparse
import csv
import itertools
import json
import os
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import presence
import presence_grid as grid
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES = ("P1", "P2")
DECIDERS = grid.DECIDERS
THRESHOLDS = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0,30 … 0,95
CURRENT = {"t_car": 0.40, "t_plate": 0.40, "t_seg": 0.50}
ZOOM_SETS = ("plate_z25", "plate_z40", "plate_z60")
PLATE_SETS = ("plate_tight", *ZOOM_SETS)  # `plate_tight`: la patente ocupa el 50 % (los recortes de siempre)
ZOOM_LABELS = {
    "plate_tight": "50 % (ajustado, el de siempre)",
    "plate_z25": "25 %",
    "plate_z40": "40 %",
    "plate_z60": "60 %",
}
NEGATIVES = grid.NEGATIVE_SETS
S1_MAX_NEW = grid.S1_MAX_NEW
VALID_ROLES = ("general", "detail")


def alert_rejection(p_general: float, p_plate: float, p_detail: float) -> float:
    """Probabilidad de que R1 rechace alguna de las tres fotos de una alerta (fallas independientes)."""
    return 1.0 - (1.0 - p_general) * (1.0 - p_plate) * (1.0 - p_detail)


def car_vector(feats: NDArray[Any], decider: str, t_car: float, t_seg: float) -> NDArray[Any]:
    det = feats[:, 0] >= t_car
    seg = feats[:, 2] >= t_seg
    return {"det": det, "seg": seg, "or": det | seg, "and": det & seg}[decider]


def accept(
    feats: NDArray[Any], role: str, decider: str, rule: str, t_car: float, t_plate: float, t_seg: float
) -> NDArray[Any]:
    """Qué fotos acepta R1: GENERAL y DETAIL piden un auto; PLATE, un auto o una patente (P1) o ambos (P2)."""
    car = car_vector(feats, decider, t_car, t_seg)
    if role != "plate":
        return car
    plate = feats[:, 1] >= t_plate
    return (car | plate) if rule == "P1" else (car & plate)


def grid_points(decider: str) -> list[tuple[float, float, float]]:
    """`(t_car, t_plate, t_seg)` que importan para el decisor; los demás quedan en el valor actual."""
    cars = THRESHOLDS if decider in ("det", "or", "and") else (CURRENT["t_car"],)
    segs = THRESHOLDS if decider in ("seg", "or", "and") else (CURRENT["t_seg"],)
    return list(itertools.product(cars, THRESHOLDS, segs))


def evaluate(
    valid: Mapping[str, NDArray[Any]],
    negatives: Mapping[str, NDArray[Any]],
    *,
    decider: str,
    rule: str,
    point: tuple[float, float, float],
    excluded: Mapping[str, NDArray[Any]],
) -> dict[str, Any]:
    """Rechazo de válidas por tipo y zoom, negativas aceptadas por rol y U3 por alerta, con cotas."""
    t_car, t_plate, t_seg = point
    rejected: dict[str, int] = {}
    totals: dict[str, int] = {}
    for name, feats in valid.items():
        role = "plate" if name in PLATE_SETS else name
        rejected[name] = int((~accept(feats, role, decider, rule, t_car, t_plate, t_seg)).sum())
        totals[name] = len(feats)
    accepted: dict[str, dict[str, int]] = {}
    for role in ("general", "plate", "detail"):
        seen = accept(negatives["negativas"], role, decider, rule, t_car, t_plate, t_seg)
        fresh = accept(negatives["negativas_nuevas"], role, decider, rule, t_car, t_plate, t_seg)
        accepted[role] = {
            "vistas": int(seen.sum()),
            "nuevas": int(fresh.sum()),
            "vistas_sin_excluidas": int((seen & ~excluded["negativas"]).sum()),
            "nuevas_sin_fondo": int((fresh & ~excluded["negativas_nuevas"]).sum()),  # sin las excluidas
        }
    rates = {n: rejected[n] / totals[n] for n in rejected}
    bounds = {n: grid.bound(rejected[n], totals[n]) or 0.0 for n in rejected}
    u3 = {z: alert_rejection(rates["general"], rates[z], rates["detail"]) for z in PLATE_SETS if z in rates}
    u3_bound = {
        z: alert_rejection(bounds["general"], bounds[z], bounds["detail"]) for z in PLATE_SETS if z in rates
    }
    zooms = [z for z in ZOOM_SETS if z in rates]
    return {
        "decisor": decider,
        "regla": rule,
        "t_car": t_car,
        "t_plate": t_plate,
        "t_seg": t_seg,
        "rechazadas": rejected,
        "validas": totals,
        "cota_rechazo": {n: grid.bound(rejected[n], totals[n]) for n in rejected},
        "negativas_aceptadas": accepted,
        "cumple_s1": all(accepted[r]["nuevas"] <= S1_MAX_NEW for r in accepted),
        "cumple_s1_sin_fondo": all(accepted[r]["nuevas_sin_fondo"] <= S1_MAX_NEW for r in accepted),
        "u3_por_alerta": {z: round(v, 4) for z, v in u3.items()},
        "u3_cota": {z: round(v, 4) for z, v in u3_bound.items()},
        "u3_medio_zooms": round(float(np.mean([u3[z] for z in zooms])), 4) if zooms else None,
    }


def sweep(
    scores: Mapping[str, Sequence[presence.Signals]],
    zoom_scores: Mapping[str, Sequence[presence.Signals]],
    names: Mapping[str, Sequence[str]],
    excluded_stems: Collection[str] = grid.BACKGROUND_CAR,
) -> list[dict[str, Any]]:
    """Todas las combinaciones.

    `excluded_stems`: fotos "sin auto" que en verdad muestran un vehículo (revisadas a mano):
    `nuevas_sin_fondo`, `vistas_sin_excluidas` y `cumple_s1_sin_fondo` las dejan afuera.
    """
    valid = {"general": grid.features(scores["general"]), "detail": grid.features(scores["detail"])}
    valid["plate_tight"] = grid.features(scores["plate"])
    for key in ZOOM_SETS:
        if key in zoom_scores:
            valid[key] = grid.features(zoom_scores[key])
    negatives = {group: grid.features(scores[group]) for group in NEGATIVES}
    banned = set(excluded_stems)
    excluded = {
        group: np.array([Path(n).stem in banned for n in names[group]], dtype=bool) for group in NEGATIVES
    }
    rows = []
    for decider, rule in itertools.product(DECIDERS, RULES):
        for point in grid_points(decider):
            rows.append(
                evaluate(valid, negatives, decider=decider, rule=rule, point=point, excluded=excluded)
            )
    return rows


def is_current(row: Mapping[str, Any]) -> bool:
    return (row["t_car"], row["t_plate"], row["t_seg"]) == (
        CURRENT["t_car"],
        CURRENT["t_plate"],
        CURRENT["t_seg"],
    )


def objective(row: Mapping[str, Any], target: str) -> float:
    """Lo que se minimiza: U3 por alerta con el zoom `target` (`plate_z25`…) o el promedio de los tres."""
    value = row["u3_medio_zooms"] if target == "promedio" else row["u3_por_alerta"][target]
    return float(value)


def best(rows: Sequence[Mapping[str, Any]], target: str, key: str = "cumple_s1") -> dict[str, Any] | None:
    """De las que cumplen S1, la de menor U3 (empate: menos negativas aceptadas, umbrales más bajos)."""
    ok = [r for r in rows if r[key]]
    if not ok:
        return None
    return dict(
        min(
            ok,
            key=lambda r: (
                objective(r, target),
                sum(a["nuevas"] for a in r["negativas_aceptadas"].values()),
                r["t_car"],
                r["t_plate"],
                r["t_seg"],
            ),
        )
    )


def best_table(rows: Sequence[Mapping[str, Any]], key: str = "cumple_s1") -> list[dict[str, Any]]:
    """Por decisor, variante y objetivo (cada zoom y el promedio), la mejor configuración que cumple S1."""
    out = []
    targets = [*ZOOM_SETS, "promedio"]
    for decider, rule, target in itertools.product(DECIDERS, RULES, targets):
        mine = [r for r in rows if r["decisor"] == decider and r["regla"] == rule]
        out.append({"decisor": decider, "regla": rule, "objetivo": target, "mejor": best(mine, target, key)})
    return out


def area_sensitivity(
    scores: Mapping[str, Sequence[presence.Signals]],
    zoom_scores: Mapping[str, Sequence[presence.Signals]],
    floors: Sequence[float] = (0.05, 0.01, 0.0),
) -> list[dict[str, Any]]:
    """Con los umbrales actuales y det ∨ seg: rechazo de PLATE por zoom según el área mínima de la patente.

    Una patente que ocupa el 25 % del ancho de una foto 4:3 ocupa cerca del 3 % de su área: con el mínimo del
    servicio (5 %) no cuenta, aunque el detector la vea. Es un diagnóstico, no una propuesta.
    """
    out = []
    sets = {"plate_tight": scores["plate"], **{k: v for k, v in zoom_scores.items() if k in ZOOM_SETS}}
    for floor in floors:
        for rule in RULES:
            row: dict[str, Any] = {"area_minima_de_la_patente": floor, "regla": rule, "rechazadas": {}}
            for name, photos in sets.items():
                feats = np.array(
                    [
                        [
                            grid.best_conf(p, "car", presence.SERVICE_MIN_AREA),
                            grid.best_conf(p, "plate", floor),
                            grid.best_conf(p, "seg", presence.SERVICE_MIN_AREA),
                        ]
                        for p in photos
                    ]
                ).reshape(len(photos), 3)
                ok = accept(
                    feats, "plate", "or", rule, CURRENT["t_car"], CURRENT["t_plate"], CURRENT["t_seg"]
                )
                row["rechazadas"][name] = {"rechazadas": int((~ok).sum()), "de": len(photos)}
            out.append(row)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="presence_report.json de presence.py")
    parser.add_argument(
        "--zoom-report", type=Path, required=True, help="plate_zoom_report.json de plate_zoom.py"
    )
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"))
    parser.add_argument("--fresh-negatives", type=Path, required=True)
    parser.add_argument(
        "--exclude-negatives",
        type=Path,
        help="JSON: lista de identificadores de fotos 'sin auto' que muestran un vehículo (revisadas a mano)",
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"))
    args = parser.parse_args(argv)
    for flag, value in (("--data-dir", args.data_dir), ("--out-dir", args.out_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(argv: list[str] | None = None) -> int:
    import presence_configs

    args = parse_args(argv)
    out = Path(args.out_dir)
    if out.resolve().is_relative_to(REPO_ROOT):
        print("La salida no puede estar dentro del repositorio.", file=sys.stderr)
        return 1
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    zoom_report = json.loads(Path(args.zoom_report).read_text(encoding="utf-8"))
    if report.get("senales_faltantes") or zoom_report.get("senales_faltantes"):
        print("Hay señales faltantes: la medición no vale.", file=sys.stderr)
        return 2
    scores = report["senales"]
    zoom_scores = zoom_report["senales"]
    names = presence_configs.reconstruct_names(scores, Path(args.data_dir), args.fresh_negatives)
    extra = (
        json.loads(Path(args.exclude_negatives).read_text(encoding="utf-8")) if args.exclude_negatives else []
    )
    excluded_stems = sorted({*grid.BACKGROUND_CAR, *extra})
    rows = sweep(scores, zoom_scores, names, excluded_stems)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "detector": zoom_report.get("detector"),
        "configuraciones": len(rows),
        "patentes_por_zoom": {k: len(v) for k, v in zoom_scores.items()},
        "negativas_excluidas": excluded_stems,
        "punto_actual": [r for r in rows if is_current(r)],
        "mejor_que_cumple_s1": best_table(rows, "cumple_s1"),
        "mejor_que_cumple_s1_sin_fondo": best_table(rows, "cumple_s1_sin_fondo"),
        "sensibilidad_del_area_de_la_patente": area_sensitivity(scores, zoom_scores),
    }
    (out / "plate_rule.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "plate_rule.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [
            {
                "decisor": r["decisor"],
                "regla": r["regla"],
                "t_car": r["t_car"],
                "t_plate": r["t_plate"],
                "t_seg": r["t_seg"],
                "cumple_s1": r["cumple_s1"],
                "u3_medio_zooms": r["u3_medio_zooms"],
                **{f"rech_{k}": v for k, v in r["rechazadas"].items()},
                **{f"nuevas_{k}": v["nuevas"] for k, v in r["negativas_aceptadas"].items()},
            }
            for r in rows
        ]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f"{len(rows)} configuraciones; {sum(r['cumple_s1'] for r in rows)} cumplen S1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
