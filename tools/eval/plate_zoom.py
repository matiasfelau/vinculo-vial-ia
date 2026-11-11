"""Primeros planos de patente realistas: la misma patente a tres niveles de zoom, y qué señales encienden.

Los recortes de `presence.py` (patente con medio ancho de margen: la patente ocupa el 50 % del ancho) son más
ajustados que la foto que saca la app. Acá se arma otro set a partir de las **imágenes completas**:

- las del dataset de Roboflow "patentes" (todas las particiones, `ocr-roboflow/`);
- las fotos con patente de la **partición de prueba de Open Images** que usa v1 (`openimages-detector/test`).

Por cada patente anotada se recorta una **ventana 4:3 horizontal centrada en la patente** cuyo ancho es
`ancho_de_la_patente / zoom`, con zoom 25 %, 40 % y 60 % (la patente ocupa ese porcentaje del ancho de la
foto). Si la ventana se sale de la imagen se la corre hasta que entre; si no cabe (más ancha o más alta que la
imagen) o queda de menos de `MIN_CROP_WIDTH` px, esa patente no entra. **Se usan solo las patentes que
admiten los tres zooms**, para que la comparación entre niveles sea sobre las mismas patentes.

Las fotos se arman en memoria y no se escriben a disco. De cada una se guardan las señales de
`presence.measure` (`[confianza, fracción de la foto]` por señal), que es lo que usa `presence_grid.py`.
Los identificadores no llevan nombres de archivo (podrían traer la patente). Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import presence

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import Pixels

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = "_annotations.coco.json"
ZOOMS = (0.25, 0.40, 0.60)
ASPECT = 0.75  # alto / ancho de la foto: 4:3 horizontal
MIN_CROP_WIDTH = 128  # una ventana más angosta que esto sería una foto de baja resolución, no de la app
MIN_PLATE_SIDE = 16
OPEN_IMAGES_PLATE = "plate"


@dataclass(frozen=True)
class PlateRef:
    """Una patente anotada: de dónde sale, su imagen y su caja `(x, y, ancho, alto)` en píxeles."""

    source: str
    ident: str
    image: Path
    image_size: tuple[int, int]  # (ancho, alto)
    box: tuple[float, float, float, float]


def zoom_key(zoom: float) -> str:
    return f"plate_z{round(zoom * 100)}"


def crop_window(
    ref: PlateRef, zoom: float, aspect: float = ASPECT, min_width: int = MIN_CROP_WIDTH
) -> tuple[int, int, int, int] | None:
    """Ventana `(x1, y1, x2, y2)` donde la patente ocupa `zoom` del ancho, o `None` si no cabe."""
    width, height = ref.image_size
    x, y, plate_w, plate_h = ref.box
    crop_w = round(plate_w / zoom)
    crop_h = round(crop_w * aspect)
    if crop_w < min_width or crop_w > width or crop_h > height:
        return None
    x1 = round(min(max(x + plate_w / 2 - crop_w / 2, 0), width - crop_w))
    y1 = round(min(max(y + plate_h / 2 - crop_h / 2, 0), height - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def common_refs(refs: Sequence[PlateRef], zooms: Sequence[float] = ZOOMS) -> list[PlateRef]:
    """Las patentes que admiten todos los zooms."""
    return [r for r in refs if all(crop_window(r, z) is not None for z in zooms)]


def coco_refs(folder: Path, source: str, plate_category: str | None) -> list[PlateRef]:
    """Patentes anotadas de un split COCO. `plate_category` filtra por nombre; `None` toma todas."""
    path = folder / ANNOTATIONS
    if not path.is_file():
        return []
    coco = json.loads(path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    refs = []
    for annotation in coco["annotations"]:
        if plate_category is not None and names[annotation["category_id"]] != plate_category:
            continue
        x, y, w, h = annotation["bbox"]
        if min(w, h) < MIN_PLATE_SIDE:
            continue
        image = images[annotation["image_id"]]
        refs.append(
            PlateRef(
                source,
                f"{source}-{folder.name}-{annotation['id']}",
                folder / image["file_name"],
                (image["width"], image["height"]),
                (x, y, w, h),
            )
        )
    return refs


def all_refs(eval_dir: Path, data_dir: Path) -> list[PlateRef]:
    refs: list[PlateRef] = []
    roboflow = eval_dir / "ocr-roboflow"
    for split in sorted(p for p in roboflow.iterdir() if (p / ANNOTATIONS).is_file()):
        refs += coco_refs(split, "roboflow", None)
    refs += coco_refs(data_dir / "openimages-detector" / "test", "openimages-test", OPEN_IMAGES_PLATE)
    return refs


def crop(image: Pixels, window: tuple[int, int, int, int]) -> Pixels:
    x1, y1, x2, y2 = window
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"))
    parser.add_argument("--eval-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"))
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"))
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"))
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--detector", default="rfdetr_openimages", help="modelo del puerto detector")
    parser.add_argument("--model-max-side", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="patentes (para probar rápido)")
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
    refs = all_refs(Path(args.eval_dir), Path(args.data_dir))
    usable = common_refs(refs)
    usable = usable[: args.limit] if args.limit else usable
    per_source = {s: sum(r.source == s for r in usable) for s in sorted({r.source for r in refs})}
    print(f"{len(refs)} patentes anotadas, {len(usable)} admiten los tres zooms {per_source}", flush=True)
    try:
        models = build_models(
            {"detector": args.detector, "segmenter": "rfdetr_seg_coco", "damage": "rfdetr_cardd"},
            args.manifest,
            args.models_dir,
            params={
                "detector_min_conf": presence.FLOOR,
                "redaction_min_conf": presence.FLOOR,
                "segmenter_min_conf": presence.FLOOR,
                "damage_min_conf": presence.FLOOR,
            },
        )
    except ModelSetupError as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    decoder = CvImageDecoder()
    signals: dict[str, list[dict[str, list[list[float]]]]] = {zoom_key(z): [] for z in ZOOMS}
    sources: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    cache: tuple[Path, Pixels] | None = None
    for index, ref in enumerate(usable):
        if cache is None or cache[0] != ref.image:
            cache = (ref.image, decoder.decode(ref.image.read_bytes()))
        windows: dict[str, list[int]] = {}
        for zoom in ZOOMS:
            window = crop_window(ref, zoom)
            assert window is not None
            windows[zoom_key(zoom)] = list(window)
            photo = crop(cache[1], window)
            measured = None
            for _ in range(presence.MEASURE_ATTEMPTS):
                try:
                    measured = presence.measure(models, decoder, photo, args.model_max_side)
                    break
                except presence.MissingSignalError:
                    continue
            if measured is None:
                missing[zoom_key(zoom)] = missing.get(zoom_key(zoom), 0) + 1
                measured = {name: [] for name in presence.SIGNALS}
            signals[zoom_key(zoom)].append(measured)
        sources.append({"id": ref.ident, "fuente": ref.source, "ventanas": windows})
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(usable)}", flush=True)
    report = {
        "detector": args.detector,
        "zooms": list(ZOOMS),
        "aspecto": ASPECT,
        "ancho_minimo_de_la_foto_px": MIN_CROP_WIDTH,
        "patentes_anotadas": len(refs),
        "patentes_usadas": len(usable),
        "por_fuente": per_source,
        "senales_faltantes": missing,
        "fuentes": sources,
        "senales": signals,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plate_zoom_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Listo: {out / 'plate_zoom_report.json'} · señales faltantes: {missing}")
    return 2 if missing else 0


def summarize(report: Mapping[str, Any]) -> dict[str, int]:
    """Cuántas patentes usó cada zoom (todas lo mismo: son las mismas patentes)."""
    return {key: len(rows) for key, rows in report["senales"].items()}


if __name__ == "__main__":
    sys.exit(main())
