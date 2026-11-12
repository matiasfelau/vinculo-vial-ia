"""Barrido de la regla R1 sin la clase rueda, a partir de las señales que midió `presence.py`.

**Histórico (12/11/2026):** mide una regla de presencia anterior (la patente o la rueda como señal, CarDD como
foto de detalle) con exclusiones manuales de negativas. La vigente es `capture_sets.py` y `capture_r1.py`.


No vuelve a inferir: cada foto trae, por señal, la lista `[confianza, fracción de la foto]`, así que cualquier
combinación de decisor, patente y umbrales se evalúa sin volver a correr un modelo.

Dimensiones:

- **decisor** de "hay un auto": `det` (solo la clase `car` del detector), `seg` (solo el segmentador), `or`
  (cualquiera de los dos), `and` (los dos, sobre la misma foto).
- **patente** como señal de presencia (la clase `plate` del detector, que se suma al decisor con un OR):
  `todas` (GENERAL, PLATE y DETAIL), `partes` (PLATE y DETAIL: la regla actual del servicio), `solo_plate`
  (solo en la foto de patente) y `ninguna`.
- **umbrales solo de R1** (no tocan los de redacción): `t_car` y `t_plate` del detector y `t_seg` del
  segmentador, desde el valor actual hasta 0,95 de a 0,05. El área mínima es la del servicio (5 %).
- **mosaico**: R1 llama a `VehicleDetector.detect` (un pase, sin mosaico); el mosaico solo se usa para
  redactar. Por eso no es una dimensión del barrido (ver el README).

Un "falso" de S1 es una foto sin auto que la configuración acepta, en cada rol (GENERAL, PLATE, DETAIL). Las
cotas son las de `presence.upper_bound` (Clopper-Pearson al 95 %; 3/N con 0 fallas).
"""

import argparse
import csv
import functools
import itertools
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import presence
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("general", "plate", "detail")
DECIDERS = ("det", "seg", "or", "and")
PLATE_MODES = ("todas", "partes", "solo_plate", "ninguna")
PLATE_ROLES: Mapping[str, tuple[str, ...]] = {
    "todas": ROLES,
    "partes": ("plate", "detail"),
    "solo_plate": ("plate",),
    "ninguna": (),
}
NEGATIVE_SETS = ("negativas", "negativas_nuevas")
T_CAR = tuple(round(0.40 + 0.05 * i, 2) for i in range(12))  # 0,40 … 0,95
T_SEG = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))  # 0,50 … 0,95
T_PLATE = T_CAR
CURRENT = {"t_car": 0.40, "t_plate": 0.40, "t_seg": 0.50}
S1_MAX_NEW = 5  # "como máximo 5 de 1.200": la cota del 95 % queda en torno al 1 %
BACKGROUND_CAR = ("381d3671020dca70", "52491dfdb5474272", "6aea64be3ea593ec")  # auto real al fondo


@functools.cache
def bound(failures: int, trials: int) -> float | None:
    """`presence.upper_bound`, con memoria: el barrido repite las mismas cuentas miles de veces."""
    return presence.upper_bound(failures, trials)


def best_conf(photo: presence.Signals, name: str, min_area: float) -> float:
    """Mayor confianza de una señal entre las cajas que ocupan al menos `min_area` (0 si no hay ninguna)."""
    return max((conf for conf, area in photo.get(name, ()) if area >= min_area), default=0.0)


def features(photos: Sequence[presence.Signals], min_area: float = presence.SERVICE_MIN_AREA) -> NDArray[Any]:
    """Una fila por foto: mejor confianza de `car`, `plate` y `seg` (con el área mínima)."""
    return np.array(
        [[best_conf(p, name, min_area) for name in ("car", "plate", "seg")] for p in photos], dtype=float
    ).reshape(len(photos), 3)


def accept(
    feats: NDArray[Any], role: str, decider: str, plate_mode: str, t_car: float, t_plate: float, t_seg: float
) -> NDArray[Any]:
    """Qué fotos acepta R1 con esa configuración (vector booleano)."""
    det = feats[:, 0] >= t_car
    seg = feats[:, 2] >= t_seg
    car = {"det": det, "seg": seg, "or": det | seg, "and": det & seg}[decider]
    if role in PLATE_ROLES[plate_mode]:
        car = car | (feats[:, 1] >= t_plate)
    return car


def grid_points(decider: str, plate_mode: str) -> list[tuple[float, float, float]]:
    """Los umbrales que importan para la configuración (los demás quedan en el valor actual)."""
    cars = T_CAR if decider in ("det", "or", "and") else (CURRENT["t_car"],)
    segs = T_SEG if decider in ("seg", "or", "and") else (CURRENT["t_seg"],)
    plates = T_PLATE if plate_mode != "ninguna" else (CURRENT["t_plate"],)
    return list(itertools.product(cars, plates, segs))


def evaluate(
    valid: Mapping[str, NDArray[Any]],
    negatives: Mapping[str, NDArray[Any]],
    *,
    decider: str,
    plate_mode: str,
    thresholds: tuple[float, float, float],
    excluded: NDArray[Any] | None = None,
) -> dict[str, Any]:
    """Rechazo de válidas y negativas aceptadas por rol, con cotas.

    `excluded` marca fotos de `negativas_nuevas` que se dejan afuera (las de auto real al fondo).
    """
    t_car, t_plate, t_seg = thresholds
    out: dict[str, Any] = {}
    for role in ROLES:
        ok = accept(valid[role], role, decider, plate_mode, t_car, t_plate, t_seg)
        rejected = int((~ok).sum())
        seen = int(accept(negatives["negativas"], role, decider, plate_mode, t_car, t_plate, t_seg).sum())
        fresh_mask = accept(negatives["negativas_nuevas"], role, decider, plate_mode, t_car, t_plate, t_seg)
        fresh = int(fresh_mask.sum())
        kept = len(fresh_mask) if excluded is None else int((~excluded).sum())
        fresh_kept = fresh if excluded is None else int((fresh_mask & ~excluded).sum())
        out[role] = {
            "validas": len(ok),
            "rechazadas": rejected,
            "rechazo_cota_95": bound(rejected, len(ok)),
            "vistas": seen,
            "vistas_de": len(negatives["negativas"]),
            "vistas_cota_95": bound(seen, len(negatives["negativas"])),
            "nuevas": fresh,
            "nuevas_de": len(fresh_mask),
            "nuevas_cota_95": bound(fresh, len(fresh_mask)),
            "nuevas_sin_fondo": fresh_kept,
            "nuevas_sin_fondo_de": kept,
            "nuevas_sin_fondo_cota_95": bound(fresh_kept, kept),
        }
    out["cumple_s1"] = all(out[r]["nuevas"] <= S1_MAX_NEW for r in ROLES)
    out["cumple_s1_sin_fondo"] = all(out[r]["nuevas_sin_fondo"] <= S1_MAX_NEW for r in ROLES)
    out["validas_rechazadas"] = sum(out[r]["rechazadas"] for r in ROLES)
    out["u3"] = out["plate"]["rechazadas"] + out["detail"]["rechazadas"]
    out["nuevas_aceptadas"] = sum(out[r]["nuevas"] for r in ROLES)
    return out


def sweep(
    scores: Mapping[str, Sequence[presence.Signals]], names: Mapping[str, Sequence[str]]
) -> list[dict[str, Any]]:
    """Todas las combinaciones; cada fila es una configuración con sus umbrales."""
    valid = {role: features(scores[role]) for role in ROLES}
    negatives = {group: features(scores[group]) for group in NEGATIVE_SETS}
    background = np.array([Path(n).stem in BACKGROUND_CAR for n in names["negativas_nuevas"]], dtype=bool)
    rows = []
    for decider, mode in itertools.product(DECIDERS, PLATE_MODES):
        for point in grid_points(decider, mode):
            result = evaluate(
                valid, negatives, decider=decider, plate_mode=mode, thresholds=point, excluded=background
            )
            rows.append(
                {
                    "decisor": decider,
                    "patente": mode,
                    "t_car": point[0],
                    "t_plate": point[1],
                    "t_seg": point[2],
                    **result,
                }
            )
    return rows


def is_current(row: Mapping[str, Any]) -> bool:
    """El punto de operación actual (umbrales del servicio) de esa configuración."""
    return (row["t_car"], row["t_plate"], row["t_seg"]) == (
        CURRENT["t_car"],
        CURRENT["t_plate"],
        CURRENT["t_seg"],
    )


def cheapest(rows: Sequence[Mapping[str, Any]], key: str = "cumple_s1") -> dict[str, Any] | None:
    """De las que cumplen S1, la que rechaza menos válidas (empate: menos aceptadas, umbrales más bajos)."""
    ok = [r for r in rows if r[key]]
    if not ok:
        return None
    return min(
        ok,
        key=lambda r: (r["validas_rechazadas"], r["nuevas_aceptadas"], r["t_car"], r["t_plate"], r["t_seg"]),
    )


def cheapest_by_config(rows: Sequence[Mapping[str, Any]], key: str = "cumple_s1") -> list[dict[str, Any]]:
    """Por decisor y modo de patente, la mejor que cumple S1 (o ninguna)."""
    found = []
    for decider, mode in itertools.product(DECIDERS, PLATE_MODES):
        mine = [r for r in rows if r["decisor"] == decider and r["patente"] == mode]
        best = cheapest(mine, key)
        found.append({"decisor": decider, "patente": mode, "mejor": best})
    return found


def frontier(rows: Sequence[Mapping[str, Any]], decider: str, plate_mode: str) -> list[dict[str, Any]]:
    """Puntos no dominados: fotos sin auto aceptadas (el peor rol) contra válidas rechazadas."""
    mine = sorted(
        (r for r in rows if r["decisor"] == decider and r["patente"] == plate_mode),
        key=lambda r: (max(r[x]["nuevas"] for x in ROLES), r["validas_rechazadas"]),
    )
    best = None
    out = []
    for r in mine:
        if best is None or r["validas_rechazadas"] < best:
            best = r["validas_rechazadas"]
            out.append(
                {
                    "t_car": r["t_car"],
                    "t_plate": r["t_plate"],
                    "t_seg": r["t_seg"],
                    "peor_rol_nuevas": max(r[x]["nuevas"] for x in ROLES),
                    "validas_rechazadas": r["validas_rechazadas"],
                    "por_rol": {x: r[x]["rechazadas"] for x in ROLES},
                }
            )
    return out


def accepted_negatives(
    scores: Mapping[str, Sequence[presence.Signals]], names: Mapping[str, Sequence[str]]
) -> list[dict[str, Any]]:
    """Fotos sin auto que acepta el detector (auto o patente, umbrales actuales) y qué vio el segmentador.

    El segmentador se informa con su mejor instancia en cualquier tamaño y con la que cuenta para R1.
    """
    found = []
    for group in NEGATIVE_SETS:
        for index, photo in enumerate(scores[group]):
            car = best_conf(photo, "car", presence.SERVICE_MIN_AREA)
            plate = best_conf(photo, "plate", presence.SERVICE_MIN_AREA)
            if car < CURRENT["t_car"] and plate < CURRENT["t_plate"]:
                continue
            seg_any = max((c for c, _ in photo.get("seg", ())), default=0.0)
            seg_counts = best_conf(photo, "seg", presence.SERVICE_MIN_AREA)
            found.append(
                {
                    "conjunto": group,
                    "foto": names[group][index],
                    "senal": "auto" if car >= CURRENT["t_car"] else "patente",
                    "conf_auto": round(car, 3),
                    "conf_patente": round(plate, 3),
                    "segmentador_vio_auto": seg_counts >= CURRENT["t_seg"],
                    "segmentador_mejor_conf_en_cualquier_tamano": round(seg_any, 3),
                    "segmentador_conf_con_area_minima": round(seg_counts, 3),
                    "auto_real_al_fondo": Path(names[group][index]).stem in BACKGROUND_CAR,
                }
            )
    return found


def per_photo(
    scores: Mapping[str, Sequence[presence.Signals]],
    names: Mapping[str, Sequence[str]],
    min_area: float = presence.SERVICE_MIN_AREA,
) -> list[dict[str, Any]]:
    """Qué se encendió en cada foto válida y sin auto (con los umbrales actuales) y con qué confianza."""
    rows = []
    for group in (*ROLES, *NEGATIVE_SETS):
        for index, photo in enumerate(scores[group]):
            car, plate, seg = (best_conf(photo, s, min_area) for s in ("car", "plate", "seg"))
            lit = {
                "auto_detector": car >= CURRENT["t_car"],
                "patente_detector": plate >= CURRENT["t_plate"],
                "auto_segmentador": seg >= CURRENT["t_seg"],
            }
            rows.append(
                {
                    "conjunto": "validas" if group in ROLES else "sin_auto",
                    "grupo": group,
                    "foto": names[group][index],
                    **lit,
                    "encendido": [k for k, v in lit.items() if v] or ["nada"],
                    "conf_auto_detector": round(car, 3),
                    "conf_patente_detector": round(plate, 3),
                    "conf_auto_segmentador": round(seg, 3),
                }
            )
    return rows


def cross_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Detector sí/no contra segmentador sí/no, separado en válidas y sin auto (y por grupo).

    Para el detector cuenta el auto; en las fotos de partes (PLATE y DETAIL) también cuenta la patente.
    """
    out: dict[str, Any] = {}
    for group in (*ROLES, *NEGATIVE_SETS):
        mine = [r for r in rows if r["grupo"] == group]
        part = group in ("plate", "detail", "negativas", "negativas_nuevas")
        cell = {"det_si_seg_si": 0, "det_si_seg_no": 0, "det_no_seg_si": 0, "det_no_seg_no": 0}
        only_plate = 0
        for r in mine:
            det = r["auto_detector"] or (part and r["patente_detector"])
            only_plate += bool(part and r["patente_detector"] and not r["auto_detector"])
            cell["det_" + ("si" if det else "no") + "_seg_" + ("si" if r["auto_segmentador"] else "no")] += 1
        out[group] = {"fotos": len(mine), **cell, "det_solo_por_patente": only_plate}
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="presence_report.json de presence.py")
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"))
    parser.add_argument("--fresh-negatives", type=Path, required=True, help="carpeta de las negativas nuevas")
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
    if report.get("senales_faltantes"):
        print("El informe tiene señales faltantes: no vale.", file=sys.stderr)
        return 2
    scores = report["senales"]
    names = presence_configs.reconstruct_names(scores, Path(args.data_dir), args.fresh_negatives)
    rows = sweep(scores, names)
    photos = per_photo(scores, names)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "configuraciones": len(rows),
        "punto_actual": [r for r in rows if is_current(r)],
        "mejor_que_cumple_s1": cheapest_by_config(rows, "cumple_s1"),
        "mejor_que_cumple_s1_sin_fondo": cheapest_by_config(rows, "cumple_s1_sin_fondo"),
        "tabla_cruzada": cross_table(photos),
        "frontera": {f"{d}/{m}": frontier(rows, d, m) for d, m in itertools.product(DECIDERS, PLATE_MODES)},
        "negativas_aceptadas_por_el_detector": accepted_negatives(scores, names),
    }
    (out / "r1_grid.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "registro_por_foto.json").write_text(
        json.dumps(photos, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    with (out / "r1_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [
            {
                "decisor": r["decisor"],
                "patente": r["patente"],
                "t_car": r["t_car"],
                "t_plate": r["t_plate"],
                "t_seg": r["t_seg"],
                "cumple_s1": r["cumple_s1"],
                "validas_rechazadas": r["validas_rechazadas"],
                **{f"{role}_{k}": r[role][k] for role in ROLES for k in ("rechazadas", "vistas", "nuevas")},
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
