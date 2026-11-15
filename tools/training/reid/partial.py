"""Pares parciales de 3DRealCar: "vista general contra la foto de patente" para validación y prueba.

R2 compara la foto GENERAL contra la de PLATE: la trompa o la cola, cortada entre el capot y el
parabrisas, con la patente al 8-15 % del ancho. 3DRealCar solo tiene vistas del auto entero; esta
herramienta **simula** ese encuadre a partir de ellas, con la misma ventana que los conjuntos de R1
(`tools/eval/capture_sets.py`). (La foto de detalle no se simula: R2 no la compara y probablemente se
elimine del contrato.)

**Cómo se arman** (por auto de validación y de prueba, con semilla fija):

1. **Vistas frontales y traseras por la pose.** El auto es una caja más ancha de costado que de frente.
   Con el azimut de cada vista (de la pose de la cámara de ARKit, ver `dataset.azimuths`) y la relación
   ancho/alto de la caja del vehículo en cada vista, se ajusta `aspecto ≈ c + a·cos(2·az) +
   b·sin(2·az)`: el máximo cae en el azimut de la vista de costado y las vistas frontales y traseras
   están a ±90° de él. Se aceptan las vistas a menos de `FRONT_REAR_TOLERANCE_DEG` de ese eje. Si el
   ajuste es débil (el auto casi no cambia de proporción), el auto no aporta pares de patente.
2. **Patente.** De esas vistas, las que el detector (v2, 0,35 de auto) ve con una patente (confianza >=
   `PLATE_MIN_CONF`) dentro de la caja del vehículo dominante y con un ancho de 10 a 30 % del ancho del
   vehículo. Se toman hasta `PLATE_PER_CAR` (las de mayor confianza, separadas al menos 30° entre sí) y
   se recorta la ventana 16:9 de `capture_sets.plate_window`: la patente al 8-15 % del ancho, el
   vehículo entero a lo ancho y cortado a la altura del capot.
3. Cada recorte simulado se lleva a 1600 px de lado mayor y JPEG de <= 900 KB (lo que envía la app) y
   pasa por el mismo camino que una foto real: se reduce a 1024 px, el detector busca el vehículo
   dominante y se recorta su caja (**sin máscaras del segmentador**). Si el detector no encuentra un
   vehículo, el servicio no tendría embedding (duda, GREEN): se cuenta como "sin vehículo" y no entra a
   los pares.

Salida (`--root`, fuera del repo): `partial-v2/` con los recortes en PNG y `partial_v2.json` con un
elemento por recorte (auto, vista de origen, azimut, ventana) y `partial_v2_stats.json` con los conteos.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dataset import angle_difference

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "eval"))

DETECTOR = "rfdetr_openimages_v2"
CROPS_INDEX = "crops_v2.json"
PARTIAL_DIR = "partial-v2"
PARTIAL_INDEX = "partial_v2.json"
STATS_FILE = "partial_v2_stats.json"
MODEL_MAX_SIDE_PX = 1024
FRONT_REAR_TOLERANCE_DEG = 30.0
MIN_ASPECT_STRENGTH = 0.20  # amplitud del ajuste sobre el aspecto medio: debajo no hay un eje claro
MIN_ASPECT_FIT = 0.50  # R² mínimo del ajuste
MIN_FRAMES_FOR_AXIS = 6
PLATE_MIN_CONF = 0.30
PLATE_PER_CAR = 2
MIN_SEPARATION_DEG = 30.0
PLATE_TO_CAR_WIDTH = (0.10, 0.30)  # igual que `capture_sets.PLATE_TO_CAR_WIDTH`
PARTITIONS = ("val", "test")
KINDS = ("patente",)


def side_axis(azimuths: Sequence[float], aspects: Sequence[float]) -> tuple[float, float, float] | None:
    """Azimut (grados, 0-180) de la vista de costado, fuerza del ajuste y R², o `None` si no hay un eje claro.

    Ajusta `aspecto ≈ c + a·cos(2·az) + b·sin(2·az)` (el auto es más ancho que alto de costado y más
    angosto de frente o de atrás) por mínimos cuadrados. Las vistas de frente y de atrás están a +-90°
    de ese azimut.
    """
    if len(azimuths) != len(aspects) or len(azimuths) < MIN_FRAMES_FOR_AXIS:
        return None
    az = np.radians(np.asarray(azimuths, np.float64))
    y = np.asarray(aspects, np.float64)
    design = np.column_stack([np.ones_like(az), np.cos(2 * az), np.sin(2 * az)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    total = float(((y - y.mean()) ** 2).sum())
    fit = 1.0 - float(((y - fitted) ** 2).sum()) / total if total > 0 else 0.0
    amplitude = float(math.hypot(coef[1], coef[2]))
    strength = amplitude / float(coef[0]) if coef[0] > 0 else 0.0
    if strength < MIN_ASPECT_STRENGTH or fit < MIN_ASPECT_FIT:
        return None
    side = math.degrees(math.atan2(coef[2], coef[1]) / 2.0) % 180.0
    return side, strength, fit


def is_front_or_rear(azimuth: float, side: float, tolerance: float = FRONT_REAR_TOLERANCE_DEG) -> bool:
    """¿La vista está a menos de `tolerance` grados de la frontal o la trasera (a +-90° de la de costado)?"""
    return any(angle_difference(azimuth, side + offset) <= tolerance for offset in (90.0, 270.0))


def pick_separated(candidates: Sequence[tuple[float, float]], count: int, min_separation: float) -> list[int]:
    """Índices de hasta `count` candidatos `(azimut, puntaje)`: los de mayor puntaje separados entre sí."""
    chosen: list[int] = []
    for index in sorted(range(len(candidates)), key=lambda i: -candidates[i][1]):
        if all(angle_difference(candidates[index][0], candidates[c][0]) >= min_separation for c in chosen):
            chosen.append(index)
        if len(chosen) == count:
            break
    return chosen


def plate_inside_car(
    plate: tuple[float, float, float, float], car: tuple[float, float, float, float]
) -> float | None:
    """Ancho de la patente sobre el del vehículo, si su centro cae dentro del vehículo y está en el rango."""
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    if not (car[0] <= cx <= car[2] and car[1] <= cy <= car[3]):
        return None
    ratio = (plate[2] - plate[0]) / max(car[2] - car[0], 1e-9)
    return ratio if PLATE_TO_CAR_WIDTH[0] <= ratio <= PLATE_TO_CAR_WIDTH[1] else None


def largest_car_crop(pixels: np.ndarray[Any, Any], models: Any) -> tuple[np.ndarray[Any, Any], float] | None:
    """Recorte de la caja `car` más grande en la foto reducida (como el servicio), con su confianza."""
    from vvi_ia.domain.model import BoxLabel
    from vvi_ia.service.validate_evidence import reduce_for_models

    view = reduce_for_models(pixels, MODEL_MAX_SIDE_PX)
    shown = pixels if view is None else view
    boxes = [b for b in (models.vehicle.detect(shown) or ()) if b.label == BoxLabel.CAR]
    if not boxes:
        return None
    box = max(boxes, key=lambda b: b.area)
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(shown.shape[1], round(box.x2)), min(shown.shape[0], round(box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return shown[y1:y2, x1:x2].copy(), box.conf


# --- armado de pares (puros) ---------------------------------------------------------------------------


def cross_same_pairs(
    general: Sequence[tuple[str, float]],
    partial: Sequence[tuple[str, float]],
    min_separation: float = MIN_SEPARATION_DEG,
) -> list[tuple[int, int, float]]:
    """Pares del mismo auto: `(i en general, j en parcial, ángulo)`, con una separación mínima entre vistas.

    `general` y `parcial` son listas de `(auto, azimut)`. Un recorte parcial no se compara con la vista
    de la que salió (ni con las que están a menos de `min_separation`).
    """
    by_car: dict[str, list[int]] = {}
    for index, (car, _) in enumerate(general):
        by_car.setdefault(car, []).append(index)
    pairs = []
    for j, (car, azimuth) in enumerate(partial):
        for i in by_car.get(car, []):
            angle = angle_difference(general[i][1], azimuth)
            if angle >= min_separation:
                pairs.append((i, j, angle))
    return pairs


def cross_different_pairs(
    general: Sequence[str],
    partial: Sequence[str],
    brand_color: Mapping[str, tuple[str, str]],
    limit: int,
    seed: int = 0,
    hard: bool = False,
) -> list[tuple[int, int]]:
    """Hasta `limit` pares `(i en general, j en parcial)` de autos distintos, sorteados con `seed`.

    Con `hard`, solo de autos de la misma marca y el mismo color. `general` y `parcial` son listas con
    el auto de cada elemento.
    """
    rng = random.Random(seed)  # noqa: S311 - muestreo de pares, no criptografía
    g_by_car: dict[str, list[int]] = {}
    p_by_car: dict[str, list[int]] = {}
    for index, car in enumerate(general):
        g_by_car.setdefault(car, []).append(index)
    for index, car in enumerate(partial):
        p_by_car.setdefault(car, []).append(index)
    cars_g, cars_p = sorted(g_by_car), sorted(p_by_car)
    candidates = [
        (a, b) for a in cars_g for b in cars_p if a != b and (not hard or (brand_color[a] == brand_color[b]))
    ]
    if not candidates:
        return []
    pairs: set[tuple[int, int]] = set()
    total = sum(len(g_by_car[a]) * len(p_by_car[b]) for a, b in candidates)
    if total <= limit:  # pocas combinaciones: todas
        return [(i, j) for a, b in candidates for i in g_by_car[a] for j in p_by_car[b]]
    attempts = 0
    while len(pairs) < limit and attempts < limit * 6:
        attempts += 1
        a, b = rng.choice(candidates)
        pairs.add((rng.choice(g_by_car[a]), rng.choice(p_by_car[b])))
    return sorted(pairs)


# --- simulación (usa el detector) ------------------------------------------------------------------------


def detect_frame(models: Any, pixels: np.ndarray[Any, Any]) -> dict[str, Any] | None:
    """Vehículo dominante y patentes de una vista, en píxeles de la foto original; `None` sin vehículo."""
    from vvi_ia.domain.model import BoxLabel
    from vvi_ia.service.validate_evidence import reduce_for_models

    reduced = reduce_for_models(pixels, MODEL_MAX_SIDE_PX)
    shown = pixels if reduced is None else reduced
    boxes = models.vehicle.detect(shown)
    if boxes is None:
        raise RuntimeError("el detector no respondió")
    scale = pixels.shape[1] / shown.shape[1]
    cars = [b for b in boxes if b.label == BoxLabel.CAR]
    if not cars:
        return None
    car = max(cars, key=lambda b: b.area)
    plates = [b for b in boxes if b.label == BoxLabel.PLATE and b.conf >= PLATE_MIN_CONF]
    return {
        "auto": (car.x1 * scale, car.y1 * scale, car.x2 * scale, car.y2 * scale),
        "patentes": [
            {"caja": (b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale), "conf": b.conf} for b in plates
        ],
    }


def simulate_car(
    car: str,
    frames: Sequence[Mapping[str, Any]],
    raw_root: Path,
    out_root: Path,
    models: Any,
    decoder: Any,
    seed: int,
) -> dict[str, Any]:
    """Recortes de patente de un auto (vistas frontales y traseras). Devuelve los elementos y los conteos."""
    from capture_sets import crop_to_jpeg, plate_window

    rng = random.Random(f"{seed}|{car}")  # noqa: S311 - muestreo de ventanas, no criptografía
    stats: Counter[str] = Counter()
    side = side_axis([f["azimut"] for f in frames], [f["ancho"] / f["alto"] for f in frames])
    stats["con_eje"] += side is not None
    detections: dict[int, dict[str, Any] | None] = {}
    pixels_by_frame: dict[int, np.ndarray[Any, Any]] = {}

    def view(frame: Mapping[str, Any]) -> tuple[np.ndarray[Any, Any], dict[str, Any] | None]:
        number = int(frame["foto"])
        if number not in pixels_by_frame:
            pixels_by_frame[number] = decoder.decode(
                (raw_root / car / f"frame_{number:05d}.jpg").read_bytes()
            )
            detections[number] = detect_frame(models, pixels_by_frame[number])
        return pixels_by_frame[number], detections[number]

    items: list[dict[str, Any]] = []

    def emit(frame: Mapping[str, Any], window: tuple[int, int, int, int], extra: dict[str, Any]) -> None:
        pixels, _ = view(frame)
        photo = decoder.decode(crop_to_jpeg(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), window))
        found = largest_car_crop(photo, models)
        if found is None:
            stats["patente_sin_vehiculo"] += 1
            return
        crop, conf = found
        name = f"{int(frame['foto']):05d}_patente_{len(items)}.png"
        target = out_root / car / name
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", crop)[1].tofile(str(target))  # igual que prepare.py: el arreglo RGB tal cual
        stats["patente_con_recorte"] += 1
        items.append(
            {
                "auto": car,
                "tipo": "patente",
                "foto": int(frame["foto"]),
                "azimut": float(frame["azimut"]),
                "archivo": f"{car}/{name}",
                "ancho": int(crop.shape[1]),
                "alto": int(crop.shape[0]),
                "confianza": round(float(conf), 4),
                "ventana": list(window),
                **extra,
            }
        )

    # patente: vistas frontales y traseras (por la pose) donde el detector ve una patente
    if side is not None:
        candidates: list[tuple[Mapping[str, Any], dict[str, Any], float, float]] = []
        for frame in frames:
            if not is_front_or_rear(float(frame["azimut"]), side[0]):
                continue
            stats["vistas_frontales_o_traseras"] += 1
            _, found = view(frame)
            if found is None:
                continue
            best = None
            for plate in sorted(found["patentes"], key=lambda p: -p["conf"]):
                ratio = plate_inside_car(plate["caja"], found["auto"])
                if ratio is not None:
                    best = (plate, ratio)
                    break
            if best is not None:
                stats["vistas_frontales_o_traseras_con_patente"] += 1
                candidates.append((frame, best[0], best[1], float(best[0]["conf"])))
        for index in pick_separated(
            [(float(c[0]["azimut"]), c[3]) for c in candidates], PLATE_PER_CAR, MIN_SEPARATION_DEG
        ):
            frame, plate, ratio, conf = candidates[index]
            pixels, found = view(frame)
            assert found is not None
            size = (pixels.shape[1], pixels.shape[0])
            window = plate_window(plate["caja"], found["auto"], size, rng)
            if window is None:
                stats["patente_sin_ventana"] += 1
                continue
            emit(
                frame, window[0], {"patente_sobre_vehiculo": round(ratio, 3), "patente_conf": round(conf, 3)}
            )
    return {"items": items, "stats": dict(stats)}


# --- línea de comandos ------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=os.environ.get("REID_DATA_DIR"), help="REID_DATA_DIR")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--detector", default=DETECTOR)
    parser.add_argument("--index", default=CROPS_INDEX, help="índice de recortes enteros (de prepare.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="solo los primeros N autos (para probar)")
    args = parser.parse_args(argv)
    for flag, value in (("--root", args.root), ("--models-dir", args.models_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(argv: list[str] | None = None) -> int:
    from vvi_ia.adapters.imaging import CvImageDecoder
    from vvi_ia.adapters.models.registry import build_models
    from vvi_ia.config import Settings
    from vvi_ia.main import adapter_params

    args = parse_args(argv)
    root = Path(args.root)
    if root.resolve().is_relative_to(REPO_ROOT):
        print("Los datos no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    rows = json.loads((root / args.index).read_text(encoding="utf-8"))
    by_car: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["particion"] in PARTITIONS:
            by_car.setdefault(row["auto"], []).append(row)
    names = sorted(by_car)[: args.limit] if args.limit else sorted(by_car)
    out_root = root / PARTIAL_DIR
    cache = out_root / "_por_auto"
    cache.mkdir(parents=True, exist_ok=True)
    pending = [car for car in names if not (cache / f"{car}.json").is_file()]
    models = decoder = None
    if pending:  # los modelos solo se cargan si falta simular algún auto
        settings = Settings.model_construct(plate_tiling=False, face_tiling=False)
        models = build_models(
            {"detector": args.detector}, args.manifest, args.models_dir, 2, adapter_params(settings)
        )
        decoder = CvImageDecoder()
    items: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for count, car in enumerate(names, 1):
        cached = cache / f"{car}.json"
        if cached.is_file():  # `--resume` implícito: un auto ya simulado no se repite
            result = json.loads(cached.read_text(encoding="utf-8"))
        else:
            result = simulate_car(car, by_car[car], root / "raw", out_root, models, decoder, args.seed)
            cached.write_text(json.dumps(result), encoding="utf-8")
        info = by_car[car][0]
        for item in result["items"]:
            if item["tipo"] != "patente":  # las cachés viejas pueden traer detalles: se descartan
                continue
            items.append(
                {
                    **item,
                    "particion": info["particion"],
                    "marca": info["marca"],
                    "modelo": info["modelo"],
                    "luz": info["luz"],
                    "tipo_auto": info["tipo"],
                    "color": info["color"],
                }
            )
        totals.update({k: v for k, v in result["stats"].items() if not k.startswith("detalle")})
        totals["autos"] += 1
        if count % 25 == 0:
            print(f"{count}/{len(names)} autos", flush=True)
    (root / PARTIAL_INDEX).write_text(json.dumps(items), encoding="utf-8")
    per = Counter((i["particion"], i["luz"]) for i in items)
    stats = {
        "totales": dict(totals),
        "recortes_de_patente": {f"{p}|{light}": n for (p, light), n in sorted(per.items())},
        "autos_con_algun_recorte": len({i["auto"] for i in items}),
    }
    (root / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
