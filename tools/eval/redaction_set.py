"""Mide cuántas veces la redacción tapa la patente denunciada, sobre el set etiquetado de OCR.

Para cada patente legible de auto del set (`ocr_set.py`, etiquetas en `etiquetas.xlsx` o `etiquetas.csv`) toma
la foto original, usa el texto etiquetado como la patente denunciada y corre el mismo camino que el
servicio en el paso 8: el detector de patentes (con mosaico), el OCR sobre cada caja detectada (las que
coinciden con la denunciada, `MATCH` o `FUZZY`, se exceptúan) y `plan_redaction`. La patente denunciada de
la foto es la caja anotada del set. Se mide **antes** (solo se exceptúan las cajas que el OCR dio por
coincidentes) y **después** (tampoco se tapan las cajas contenidas, ≥ 80 % de su área, en una exceptuada),
con las mismas detecciones y lecturas: una sola pasada del OCR por caja.

"Tapa" se mide por la fracción del área de la caja anotada que queda bajo el relleno de alguna caja del plan
(con el margen del 10 %): total si cubre al menos `TOTAL_FRACTION`, parcial si cubre al menos
`PARTIAL_FRACTION` y menos que eso, y ninguna en otro caso. `tapa_algo` cuenta cualquier superposición (como
la medición de S3 del README). Si el OCR no lee la denunciada no hay excepción y
se tapa a propósito (es el costo de redactar con fail-closed): se informa aparte (`sin_exencion`).

Corre el OCR sobre el set: hacerlo recién con las etiquetas completas. Escribe en `--out-dir` `redaction.json`
(solo cifras) y `redaction_rows.csv` (id, luz y fracciones: nunca el texto). Uso: ver `tools/eval/README.md`.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ocr_labels
from cascade import MemoryStore, PassthroughDecryptor
from ocr import IN_SCOPE_FORMATS, PHOTOS_DIR, Sample, inside_repo, load_set_samples, sample_format

from vvi_ia.adapters.imaging import CvImageDecoder, JpegRenderer
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.config import Settings
from vvi_ia.domain.model import Box, BoxLabel, EvidenceImage, EvidenceRole
from vvi_ia.domain.plate import canonicalize
from vvi_ia.domain.redaction import EXEMPT_CONTAINED, RedactionThresholds, plan_redaction
from vvi_ia.main import adapter_params, build_redaction, build_thresholds
from vvi_ia.service.ports import ImageDecodeError
from vvi_ia.service.validate_evidence import ValidateEvidence, _Run, reduce_for_models

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DETECTOR = "rfdetr_openimages_v2"  # el del servicio (MODEL_DETECTOR)
TOTAL_FRACTION = 0.95  # el relleno cubre al menos esto de la caja anotada: tapa total
PARTIAL_FRACTION = 0.02  # y al menos esto: tapa parcial (menos es un borde que roza, no una tapada)
THRESHOLDS = (0.0, PARTIAL_FRACTION, 0.10, 0.50, TOTAL_FRACTION)  # cortes de la cobertura que se informan
ROW_COLUMNS = ("id", "luz", "patentes_detectadas", "exceptuadas", "cobertura_antes", "cobertura_despues")
NONE, PARTIAL, TOTAL = "ninguna", "parcial", "total"


def coverage(target: Box, plan: Sequence[Box]) -> float:
    """Fracción del área de `target` que queda bajo el relleno de las cajas de `plan`.

    Usa la misma grilla de píxeles que `fill_boxes` (piso en el origen, techo en el final).
    """
    x1, y1, x2, y2 = math.floor(target.x1), math.floor(target.y1), math.ceil(target.x2), math.ceil(target.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    covered = np.zeros((y2 - y1, x2 - x1), bool)
    for box in plan:
        bx1, by1 = max(x1, math.floor(box.x1)), max(y1, math.floor(box.y1))
        bx2, by2 = min(x2, math.ceil(box.x2)), min(y2, math.ceil(box.y2))
        if bx2 > bx1 and by2 > by1:
            covered[by1 - y1 : by2 - y1, bx1 - x1 : bx2 - x1] = True
    return float(covered.mean())


def classify(fraction: float) -> str:
    if fraction >= TOTAL_FRACTION:
        return TOTAL
    return PARTIAL if fraction >= PARTIAL_FRACTION else NONE


@dataclass(frozen=True)
class Measure:
    """Lo que se midió en una patente denunciada."""

    ident: str
    light: str
    plates: int  # cajas de patente que devolvió el detector con confianza de redacción
    exempt: int  # las que el OCR dio por la denunciada (MATCH o FUZZY)
    before: float
    after: float
    unavailable: bool = False  # el detector de patentes no respondió (fail-closed: no hay redacción)


def measure_one(
    use_case: ValidateEvidence,
    run: _Run,
    image: EvidenceImage,
    reported: str,
    target: Box,
    thresholds: RedactionThresholds,
) -> tuple[int, int, float, float] | None:
    """`(patentes, exceptuadas, cobertura antes, cobertura después)`, o `None` si no hay detector de patentes.

    Es el camino del paso 8 del servicio: las mismas detecciones, las mismas lecturas y `plan_redaction`.
    """
    detections = run.detect_plates(image)
    if detections is None:
        return None
    exempt = use_case._matching_plates(run, image, detections, reported)  # el mismo camino que el servicio
    before = plan_redaction(detections, thresholds, image.width, image.height, exempt, exempt_contained=None)
    after = plan_redaction(
        detections, thresholds, image.width, image.height, exempt, exempt_contained=EXEMPT_CONTAINED
    )
    plates = sum(box.label == BoxLabel.PLATE and box.conf >= thresholds.plate_min_conf for box in detections)
    return plates, len(exempt), coverage(target, before or []), coverage(target, after or [])


def measure_samples(
    use_case: ValidateEvidence,
    samples: Iterable[Sample],
    photos_dir: Path,
    model_max_side_px: int,
    thresholds: RedactionThresholds,
) -> tuple[list[Measure], Counter[str]]:
    """Mide cada patente legible de auto. Las lecturas del OCR se reutilizan entre filas de la misma foto."""
    decoder = CvImageDecoder()
    runs: dict[tuple[str, str], tuple[EvidenceImage, _Run] | None] = {}
    measures: list[Measure] = []
    skipped: Counter[str] = Counter()
    for sample in samples:
        if sample_format(sample) not in IN_SCOPE_FORMATS:
            continue
        key = (sample.split, sample.file)
        if key not in runs:
            try:
                pixels = decoder.decode((photos_dir / sample.split / sample.file).read_bytes())
            except (OSError, ImageDecodeError):
                runs[key] = None
            else:
                view = reduce_for_models(pixels, model_max_side_px)
                runs[key] = (
                    EvidenceImage(role=EvidenceRole.GENERAL, index=0, pixels=pixels, view=view),
                    _Run(models=use_case._models),  # el mismo conjunto de modelos
                )
        loaded = runs[key]
        if loaded is None:
            skipped["foto ilegible"] += 1
            continue
        image, run = loaded
        result = measure_one(use_case, run, image, canonicalize(sample.label), sample.box, thresholds)
        if result is None:
            measures.append(Measure(sample.ident, sample.light, 0, 0, 0.0, 0.0, unavailable=True))
        else:
            measures.append(Measure(sample.ident, sample.light, *result))
    return measures, skipped


def rate(count: int, total: int) -> float | None:
    return round(100 * count / total, 1) if total else None


def side_summary(fractions: Sequence[float]) -> dict[str, Any]:
    kinds = Counter(classify(fraction) for fraction in fractions)
    total = len(fractions)
    anything = sum(fraction > 0 for fraction in fractions)
    return {
        # Como en la medición de S3: "algo" es cualquier superposición con el relleno.
        "tapa_algo": anything,
        "tapa_algo_pct": rate(anything, total),
        "tapa_total": kinds[TOTAL],
        "tapa_parcial": kinds[PARTIAL],
        "tapa_total_o_parcial": kinds[TOTAL] + kinds[PARTIAL],
        "tapa_total_o_parcial_pct": rate(kinds[TOTAL] + kinds[PARTIAL], total),
        "sin_tapar": kinds[NONE],
        "cobertura_mayor_que": {
            f"{threshold:g}": sum(fraction > threshold for fraction in fractions) for threshold in THRESHOLDS
        },
    }


def group_summary(measures: Sequence[Measure]) -> dict[str, Any]:
    available = [m for m in measures if not m.unavailable]
    with_exemption = [m for m in available if m.exempt]
    without = [m for m in available if not m.exempt]
    changed = [m for m in available if classify(m.before) != classify(m.after)]
    return {
        "fotos": len(measures),
        "sin_detector_de_patentes": sum(m.unavailable for m in measures),
        "con_exencion": len(with_exemption),
        "sin_exencion": len(without),
        "antes": side_summary([m.before for m in available]),
        "despues": side_summary([m.after for m in available]),
        # La exención es lo que protege a la denunciada: sin ella (el OCR no la leyó) se tapa a propósito.
        "con_exencion_antes": side_summary([m.before for m in with_exemption]),
        "con_exencion_despues": side_summary([m.after for m in with_exemption]),
        "sin_exencion_antes": side_summary([m.before for m in without]),
        "sin_exencion_despues": side_summary([m.after for m in without]),
        "cambian_de_clase": len(changed),
        "empeoran": sum(m.after > m.before + 1e-9 for m in available),
    }


def build_report(
    measures: Sequence[Measure], thresholds: RedactionThresholds, versions: dict[str, str]
) -> dict[str, Any]:
    by_light: dict[str, list[Measure]] = {}
    for measure in measures:
        by_light.setdefault(measure.light or "sin dato", []).append(measure)
    return {
        "modelos": versions,
        "umbrales": {
            "PLATE_REDACT_MIN_CONF": thresholds.plate_min_conf,
            "contenida_minima": EXEMPT_CONTAINED,
            "tapa_total_desde": TOTAL_FRACTION,
            "tapa_parcial_desde": PARTIAL_FRACTION,
        },
        **group_summary(measures),
        "por_luz": {light: group_summary(items) for light, items in sorted(by_light.items())},
    }


def write_rows(path: Path, measures: Sequence[Measure]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_COLUMNS))
        writer.writeheader()
        for m in measures:
            writer.writerow(
                {
                    "id": m.ident,
                    "luz": m.light,
                    "patentes_detectadas": m.plates,
                    "exceptuadas": m.exempt,
                    "cobertura_antes": f"{m.before:.3f}",
                    "cobertura_despues": f"{m.after:.3f}",
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument("--labeled-set", type=Path, help="set etiquetado de ocr_set.py (xlsx o csv)")
    parser.add_argument(
        "--photos-dir", type=Path, help=f"export con las fotos (por defecto <data-dir>/{PHOTOS_DIR})"
    )
    parser.add_argument("--incomplete", action="store_true", help="saltar las filas sin etiquetar")
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, help="MODEL_DETECTOR")
    parser.add_argument("--ocr", default="parseq", help="MODEL_OCR")
    parser.add_argument("--plate-tiling", choices=("on", "off"), default="on", help="PLATE_TILING")
    parser.add_argument("--threads", type=int, default=0, help="hilos de ONNX Runtime (0: los que elija ORT)")
    args = parser.parse_args(argv)
    for flag, value in (
        ("--labeled-set", args.labeled_set),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    if args.photos_dir is None:
        if args.data_dir is None:
            parser.error("indicá --photos-dir o --data-dir")
        args.photos_dir = Path(args.data_dir) / PHOTOS_DIR
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name, path in (
        ("el set etiquetado", args.labeled_set),
        ("las fotos", args.photos_dir),
        ("la salida", args.out_dir),
    ):
        if inside_repo(path):
            print(
                f"{name} no puede estar dentro del repositorio: tiene patentes de terceros.", file=sys.stderr
            )
            return 1
    settings = Settings.model_construct(redaction_required=True, plate_tiling=args.plate_tiling == "on")
    try:
        loaded = load_set_samples(args.labeled_set, args.incomplete)
        models = build_models(
            {"detector": args.detector, "ocr": args.ocr},
            args.manifest,
            args.models_dir,
            args.threads,
            adapter_params(settings),
        )
    except (FileNotFoundError, ModelSetupError, ocr_labels.LabelsError, ImportError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    store = MemoryStore()
    use_case = ValidateEvidence(
        store=store,
        writer=store,
        decryptor=PassthroughDecryptor(),
        decoder=CvImageDecoder(),
        renderer=JpegRenderer(),
        models=models,
        thresholds=build_thresholds(settings),
        redaction=build_redaction(settings),
    )
    thresholds = RedactionThresholds(settings.plate_redact_min_conf)
    measures, skipped = measure_samples(
        use_case, loaded.samples, args.photos_dir, settings.model_max_side_px, thresholds
    )
    report = build_report(measures, thresholds, models.versions())
    report["salteadas"] = dict(skipped)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "redaction.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_rows(args.out_dir / "redaction_rows.csv", measures)
    before, after = report["antes"], report["despues"]
    print(
        f"Patentes denunciadas medidas: {report['fotos']} · "
        f"sin detector de patentes: {report['sin_detector_de_patentes']}"
    )
    print(
        f"Con exención (el OCR la leyó como la denunciada): {report['con_exencion']} · "
        f"sin: {report['sin_exencion']}"
    )
    for name, side in (("antes", before), ("después", after)):
        print(
            f"Redacción {name}: tapa algo {side['tapa_algo']} ({side['tapa_algo_pct']} %) · "
            f"total {side['tapa_total']} · parcial {side['tapa_parcial']} · sin tapar {side['sin_tapar']}"
        )
    print(f"Cambian de clase: {report['cambian_de_clase']} · empeoran: {report['empeoran']}")
    print(f"Salida: {args.out_dir / 'redaction.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
