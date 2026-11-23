"""R1 en varias configuraciones, a partir de las señales que midió `presence.py` (sin volver a inferir).

**Histórico (12/11/2026):** mide una regla de presencia anterior (la patente o la rueda como señal, CarDD como
foto de detalle) con exclusiones manuales de negativas. La vigente es `capture_sets.py` y `capture_r1.py`.


Con las señales de cada foto (`presence_report.json`) evalúa la regla R1 del servicio, por rol, en:

- **a** sin segmentador (la oficial) y **b** con segmentador, ambas sin la caja de daño;
- **la mejor de las dos** sin la clase rueda como señal de presencia, y con la rueda exigiendo una confianza
  más alta (barrido).

Para cada foto sin auto que una configuración acepta dice qué señal (clase del detector o segmentador) la
dejó pasar. **S1**: fotos sin auto aceptadas, con su cota del 95 %. **U3**: primeros planos válidos
(PLATE y DETAIL) rechazados. `--verify` vuelve a medir esas fotos y comprueba que el orden de los nombres
coincide con el de las señales. `--sheets` arma hojas de contacto (fuera del repo) para mirarlas.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import presence

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import build_models

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("general", "plate", "detail")
CLOSE_UPS = ("plate", "detail")
NEGATIVE_SETS = ("negativas", "negativas_nuevas")
WHEEL_SWEEP = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MODEL_OF = {
    "car": "detector (clase car)",
    "wheel": "detector (clase wheel)",
    "plate": "detector (clase plate)",
    "seg": "segmentador (instancia de auto)",
}
SIGNAL_TOLERANCE = 1e-3


def thresholds_with(wheel: float) -> dict[str, float]:
    return {**presence.DEFAULT_THRESHOLDS, "wheel": wheel}


def run_config(
    scores: Mapping[str, Sequence[presence.Signals]], *, segmenter: bool, wheel: bool, wheel_t: float
) -> dict[str, Any]:
    """R1 sin la caja de daño y con la patente en DETAIL, como en las tablas a, b, c1 y c2."""
    roles = presence.service_rule(
        scores,
        1.0,
        thresholds_with(wheel_t),
        segmenter=segmenter,
        damage=False,
        wheel=wheel,
        detail_plate=True,
    )
    rejected = sum(roles[r]["rechazadas"] for r in CLOSE_UPS)
    total = sum(roles[r]["fotos"] for r in CLOSE_UPS)
    return {
        "segmentador": segmenter,
        "rueda": wheel,
        "umbral_rueda": wheel_t if wheel else None,
        "reglas": {role: list(rule) for role, rule in presence.service_rules(segmenter, wheel, True).items()},
        "roles": roles,
        "U3": {
            "rechazadas": rejected,
            "fotos": total,
            "tasa": round(rejected / total, 4),
            "cota_95": presence.upper_bound(rejected, total) if rejected else None,
        },
        "S1": {role: s1(roles[role]) for role in ROLES},
    }


def s1(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Fotos sin auto aceptadas (vistas, nuevas y todas juntas) con la cota del 95 %."""
    seen, fresh = entry["negativas_que_pasan"], entry["negativas_nuevas"]["que_pasan"]
    trials = entry["negativas"] + entry["negativas_nuevas"]["fotos"]
    return {
        "vistas": f"{seen}/{entry['negativas']}",
        "nuevas": f"{fresh}/{entry['negativas_nuevas']['fotos']}",
        "nuevas_cota_95": entry["negativas_nuevas"]["cota_95"],
        "todas": f"{seen + fresh}/{trials}",
        "todas_cota_95": presence.upper_bound(seen + fresh, trials),
    }


def badness(config: Mapping[str, Any]) -> tuple[int, int]:
    """Orden para elegir la mejor: primero S1 (nuevas aceptadas, sumadas por rol) y después U3."""
    accepted = sum(config["roles"][r]["negativas_nuevas"]["que_pasan"] for r in ROLES)
    return accepted, config["U3"]["rechazadas"]


def triggers(
    signals: presence.Signals,
    rule: Sequence[str],
    thresholds: Mapping[str, float],
    min_area: float = presence.SERVICE_MIN_AREA,
) -> list[dict[str, Any]]:
    """Las señales de `rule` que dejan pasar la foto: clase o modelo, confianza y fracción de la foto."""
    return [
        {"senal": name, "modelo": MODEL_OF[name], "confianza": round(conf, 3), "area": round(area, 3)}
        for name in rule
        for conf, area in signals.get(name, ())
        if conf >= thresholds[name] and area >= min_area
    ]


def accepted_negatives(
    scores: Mapping[str, Sequence[presence.Signals]],
    names: Mapping[str, Sequence[str]],
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Por rol, cada foto sin auto que acepta la configuración y qué señal la dejó pasar."""
    thresholds = thresholds_with(config["umbral_rueda"] or presence.DEFAULT_THRESHOLDS["wheel"])
    out: dict[str, list[dict[str, Any]]] = {}
    for role in ROLES:
        found = []
        for group in NEGATIVE_SETS:
            for index, signals in enumerate(scores.get(group, ())):
                hits = triggers(signals, config["reglas"][role], thresholds)
                if hits:
                    found.append({"conjunto": group, "foto": names[group][index], "senales": hits})
        out[role] = found
    return out


def compare(
    scores: Mapping[str, Sequence[presence.Signals]], names: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    default_t = presence.DEFAULT_THRESHOLDS["wheel"]
    a = run_config(scores, segmenter=False, wheel=True, wheel_t=default_t)
    b = run_config(scores, segmenter=True, wheel=True, wheel_t=default_t)
    best_key = "a_sin_segmentador" if badness(a) <= badness(b) else "b_con_segmentador"
    use_seg = best_key.startswith("b")
    configs: dict[str, dict[str, Any]] = {"a_sin_segmentador": a, "b_con_segmentador": b}
    configs["c1_" + best_key[0] + "_sin_rueda"] = run_config(
        scores, segmenter=use_seg, wheel=False, wheel_t=default_t
    )
    configs["ref_a_sin_rueda"] = run_config(scores, segmenter=False, wheel=False, wheel_t=default_t)
    sweep = {
        f"{t:.2f}": run_config(scores, segmenter=use_seg, wheel=True, wheel_t=t)
        for t in WHEEL_SWEEP
        if t > default_t
    }
    for config in configs.values():
        config["aceptadas"] = accepted_negatives(scores, names, config)
    for config in sweep.values():
        config["aceptadas"] = accepted_negatives(scores, names, config)
    best_wheel = min(sweep, key=lambda key: badness(sweep[key]))
    configs["c2_" + best_key[0] + "_rueda_" + best_wheel] = sweep[best_wheel]
    return {
        "criterio_de_la_mejor": "menos fotos sin auto nuevas aceptadas (suma de roles); empate: menos U3",
        "mejor_de_a_y_b": best_key,
        "configuraciones": configs,
        "barrido_de_la_rueda": sweep,
    }


def reconstruct_names(
    scores: Mapping[str, Sequence[presence.Signals]], data_dir: Path, fresh_dir: Path | None
) -> dict[str, list[str]]:
    """Nombre de cada foto, en el mismo orden en que `presence.py` las midió (se comprueba con `--verify`)."""
    listed: dict[str, list[Path]] = {
        "general": sorted((data_dir / "damage-fp" / "images").glob("*.jpg")),
        "detail": sorted((data_dir / "cardd-detector" / "test").glob("*.jpg")),
        "negativas": presence.negative_photos(data_dir),
    }
    if fresh_dir is not None:
        listed["negativas_nuevas"] = sorted(fresh_dir.glob("*.jpg"))[
            : len(scores.get("negativas_nuevas", ()))
        ]
    names = {group: [str(p) for p in paths] for group, paths in listed.items()}
    names["plate"] = [f"recorte-{i}" for i in range(len(scores.get("plate", ())))]
    for group, paths in names.items():
        if group in scores and len(paths) != len(scores[group]):
            raise SystemExit(f"{group}: {len(paths)} fotos en disco y {len(scores[group])} en el informe")
    return names


def verify_order(
    scores: Mapping[str, Sequence[presence.Signals]],
    names: Mapping[str, Sequence[str]],
    accepted: Mapping[str, Sequence[tuple[str, int]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Vuelve a medir las fotos aceptadas y compara con las señales guardadas."""
    models = build_models(
        {"detector": "rfdetr_openimages", "segmenter": "rfdetr_seg_coco", "damage": "rfdetr_cardd"},
        args.manifest,
        args.models_dir,
        params={
            "detector_min_conf": presence.FLOOR,
            "redaction_min_conf": presence.FLOOR,
            "segmenter_min_conf": presence.FLOOR,
            "damage_min_conf": presence.FLOOR,
        },
    )
    decoder = CvImageDecoder()
    checked = mismatched = 0
    for group, items in accepted.items():
        for _, index in items:
            fresh = presence.measure(
                models, decoder, decoder.decode(Path(names[group][index]).read_bytes()), 1024
            )
            stored = scores[group][index]
            checked += 1
            same = all(
                len(fresh[s]) == len(stored.get(s, ()))
                and all(
                    abs(x - y) < SIGNAL_TOLERANCE
                    for pair, other in zip(sorted(fresh[s]), sorted(stored.get(s, ())), strict=True)
                    for x, y in zip(pair, other, strict=True)
                )
                for s in presence.SIGNALS
            )
            mismatched += not same
    return {"fotos_comprobadas": checked, "no_coinciden": mismatched}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="presence_report.json de presence.py")
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"))
    parser.add_argument("--fresh-negatives", type=Path, help="carpeta de las negativas nuevas")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"))
    parser.add_argument("--verify", action="store_true", help="vuelve a medir las fotos aceptadas")
    parser.add_argument("--sheets", action="store_true", help="hojas de contacto de las aceptadas")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"))
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    args = parser.parse_args(argv)
    for flag, value in (("--data-dir", args.data_dir), ("--out-dir", args.out_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def print_table(result: Mapping[str, Any]) -> None:
    print(f"La mejor de a y b: {result['mejor_de_a_y_b']} ({result['criterio_de_la_mejor']})")
    header = (
        f"{'configuración':<34}{'U3 (plate+detail)':>20}   S1 nuevas general / plate / detail   S1 vistas"
    )
    print(header)
    rows = {
        **result["configuraciones"],
        **{f"  barrido rueda >= {k}": v for k, v in result["barrido_de_la_rueda"].items()},
    }
    for name, config in rows.items():
        u3 = config["U3"]
        fresh = " / ".join(
            f"{config['S1'][r]['nuevas']} ({config['S1'][r]['nuevas_cota_95']})" for r in ROLES
        )
        seen = " / ".join(config["S1"][r]["vistas"] for r in ROLES)
        print(f"{name:<34}{u3['rechazadas']:>4}/{u3['fotos']:<4} ({u3['tasa']:.1%}){'':>4}{fresh}   {seen}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if Path(args.out_dir).resolve().is_relative_to(REPO_ROOT):
        print("La salida no puede estar dentro del repositorio.", file=sys.stderr)
        return 1
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if report.get("senales_faltantes"):
        print("El informe tiene señales faltantes: no vale.", file=sys.stderr)
        return 2
    scores = report["senales"]
    names = reconstruct_names(scores, Path(args.data_dir), args.fresh_negatives)
    result = compare(scores, names)
    print_table(result)
    accepted: dict[str, set[tuple[str, int]]] = {}
    for config in result["configuraciones"].values():
        for role in ROLES:
            for item in config["aceptadas"][role]:
                accepted.setdefault(item["conjunto"], set()).add(
                    (item["conjunto"], names[item["conjunto"]].index(item["foto"]))
                )
    union = {group: sorted(items) for group, items in accepted.items()}
    result["fotos_aceptadas_por_alguna"] = sum(len(v) for v in union.values())
    print(f"Fotos sin auto aceptadas por alguna configuración: {result['fotos_aceptadas_por_alguna']}")
    if args.verify:
        result["verificacion_del_orden"] = verify_order(scores, names, union, args)
        print(f"Verificación del orden de los nombres: {result['verificacion_del_orden']}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "presence_configs.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.sheets:
        write_sheets(result, names, Path(args.out_dir) / "hojas-aceptadas")
    return 0


def write_sheets(result: Mapping[str, Any], names: Mapping[str, Sequence[str]], folder: Path) -> None:
    """Una hoja de contacto con cada foto sin auto aceptada por alguna configuración (para mirarla a ojo)."""
    from audit_sheets import Panel, sheets

    seen: dict[str, list[str]] = {}
    for key, config in result["configuraciones"].items():
        for role in ROLES:
            for item in config["aceptadas"][role]:
                label = f"{key[0:2]}:{role[:3]}:" + "+".join(h["senal"] for h in item["senales"])
                seen.setdefault(item["foto"], []).append(label)
    panels = [
        Panel(Path(photo), f"{number:02d} " + " ".join(sorted(set(labels)))[:42])
        for number, (photo, labels) in enumerate(sorted(seen.items()), 1)
    ]
    index = {photo: number for number, (photo, _) in enumerate(sorted(seen.items()), 1)}
    (folder.parent / "hojas-aceptadas-indice.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(sheets(panels, folder, columns=4, rows=3, cell=420))} hojas en {folder}")


if __name__ == "__main__":
    sys.exit(main())
