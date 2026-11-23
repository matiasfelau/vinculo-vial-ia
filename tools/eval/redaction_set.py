"""Mide cuántas veces la redacción tapa la patente denunciada, sobre el set etiquetado de OCR.

Para cada patente legible de auto del set (`ocr_set.py`, etiquetas en `etiquetas.xlsx` o `etiquetas.csv`) toma
la foto original como foto de patente, usa el texto etiquetado como la patente denunciada y corre el mismo
camino que el servicio en el paso 8: el detector de patentes (con mosaico), el OCR sobre cada caja detectada
(las que coinciden con la denunciada, `MATCH` o `FUZZY`, se exceptúan) y `plan_redaction`. La patente
denunciada de la foto es la caja anotada del set. Se mide **antes** (el servicio con la patente principal
de la foto de patente, la excepción por lectura y las cajas contenidas en la exceptuada) y **después**
(además, no se tapan las cajas que contienen entera a una exceptuada y el margen no pisa a una exceptuada),
con las mismas detecciones y lecturas: una sola pasada del OCR por caja.

"Tapa" se mide por la fracción del área de la caja anotada que queda bajo el relleno de alguna caja del plan
(con el margen del 10 %): total si cubre al menos `TOTAL_FRACTION`, parcial si cubre al menos
`PARTIAL_FRACTION` y menos que eso, y ninguna en otro caso. `tapa_algo` cuenta cualquier superposición (como
la medición de S3 del README). Si el OCR no lee la denunciada no hay excepción y
se tapa a propósito (es el costo de redactar con fail-closed): se informa aparte (`sin_exencion`).

`principal` cuenta cuántas veces la patente principal elegida no es la denunciada (IoU con la caja anotada
menor que `MAIN_TARGET_IOU`): en esos casos queda a la vista otra patente y la denunciada puede quedar tapada.
`casos_tapados_con_exencion` lista las fotos donde el OCR leyó la denunciada y el plan igual la tapa, con qué
caja la cubre (mayor que la patente, vecina que solo llega con su margen, o solapada).
`ajenas` mide si las dos reglas nuevas dejan expuesta alguna otra patente de la misma foto (las demás cajas
anotadas del export): cuántas estaban tapadas por completo antes y dejan de estarlo después.

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ocr_labels
from ocr import IN_SCOPE_FORMATS, PHOTOS_DIR, Sample, inside_repo, load_set_samples, sample_format
from ocr_set import invalid_boxes, load_annotations

from vvi_ia.adapters.imaging import CvImageDecoder, JpegRenderer
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.config import Settings
from vvi_ia.domain.model import Box, BoxLabel, EvidenceImage, EvidenceRole
from vvi_ia.domain.plate import canonicalize
from vvi_ia.domain.redaction import (
    EXEMPT_CONTAINED,
    EXEMPT_CONTAINING,
    REDACTION_MARGIN,
    RedactionThresholds,
    contained_fraction,
    main_plate,
    painted_boxes,
    plan_redaction,
)
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
MAIN_TARGET_IOU = 0.5  # la principal es la denunciada si su caja se parece a la anotada al menos esto
BIGGER, NEIGHBOUR, OVERLAP = "mayor que la patente", "vecina (solo llega con el margen)", "solapada"


class PassthroughDecryptor:
    """Sin cifrado: el descifrado HPKE no entra en esta medición."""

    def open(self, ciphertext: bytes, info: bytes) -> bytes:
        return ciphertext


class MemoryStore:
    """Almacenamiento en memoria: el caso de uso no se ejecuta, solo se usan sus piezas del paso 8."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def put(self, key: str, body: bytes, content_type: str) -> None:
        self.objects[key] = body


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
    before: float  # cobertura de la denunciada con el servicio hasta la patente principal
    after: float  # y además sin tapar las cajas que contienen a la exceptuada ni su margen
    unavailable: bool = False  # el detector de patentes no respondió (fail-closed: no hay redacción)
    has_main: bool = False  # hay una patente principal (alguna caja con confianza de redacción)
    main_is_target: bool = False  # la principal es la denunciada (IoU con la caja anotada)
    main_read_as_target: bool = False  # y el OCR la lee como la denunciada
    covers: tuple[dict[str, Any], ...] = ()  # qué cajas cubren la denunciada en el plan de después
    others: tuple[tuple[str, float, float], ...] = ()  # otras patentes anotadas: (clave, antes, después)


def iou(a: Box, b: Box) -> float:
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = width * height
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def intersection(a: Box, b: Box) -> float:
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return width * height


def covering_boxes(
    detections: Sequence[Box],
    exempt: Sequence[Box],
    thresholds: RedactionThresholds,
    width: int,
    height: int,
    target: Box,
) -> tuple[dict[str, Any], ...]:
    """Las cajas del plan (con las dos reglas nuevas) que tocan la denunciada, con qué son (sin texto)."""
    found: list[dict[str, Any]] = []
    for box in detections:
        if box.label != BoxLabel.PLATE or box.conf < thresholds.plate_min_conf or box in exempt:
            continue
        if any(contained_fraction(box, other) >= EXEMPT_CONTAINED for other in exempt):
            continue
        if any(contained_fraction(other, box) >= EXEMPT_CONTAINING for other in exempt):
            continue
        share = coverage(target, painted_boxes(box, REDACTION_MARGIN, width, height, exempt, True))
        if share <= 0:
            continue
        holds_exempt = max((contained_fraction(other, box) for other in exempt), default=0.0)
        if holds_exempt >= EXEMPT_CONTAINED:
            kind = BIGGER
        elif intersection(box, target) <= 0:
            kind = NEIGHBOUR
        else:
            kind = OVERLAP
        found.append(
            {
                "tipo": kind,
                "confianza": round(box.conf, 2),
                "cubre_de_la_denunciada": round(share, 3),
                "area_sobre_la_denunciada": round(box.area / target.area, 2) if target.area > 0 else None,
            }
        )
    return tuple(found)


def measure_one(
    use_case: ValidateEvidence,
    run: _Run,
    image: EvidenceImage,
    sample: Sample,
    thresholds: RedactionThresholds,
    others: Sequence[tuple[str, Box]] = (),
) -> Measure:
    """La cobertura de la denunciada (y de las otras patentes anotadas) antes y después de las dos reglas.

    Es el camino del paso 8 del servicio: las mismas detecciones, las mismas lecturas y `plan_redaction`.
    """
    detections = run.detect_plates(image)
    if detections is None:
        return Measure(sample.ident, sample.light, 0, 0, 0.0, 0.0, unavailable=True)
    reported = canonicalize(sample.label)
    exempt = use_case._matching_plates(run, image, detections, reported)  # el mismo camino que el servicio
    main = main_plate(detections, thresholds.plate_min_conf)
    size = (image.width, image.height)
    before = plan_redaction(
        detections, thresholds, *size, exempt, main_plate=main, exempt_containing=None, clip_margin=False
    )
    after = plan_redaction(detections, thresholds, *size, exempt, main_plate=main)
    plates = sum(box.label == BoxLabel.PLATE and box.conf >= thresholds.plate_min_conf for box in detections)
    target = sample.box
    cover_after = coverage(target, after or [])
    exempt_after = [*exempt, *([main] if main is not None else [])]
    return Measure(
        sample.ident,
        sample.light,
        plates,
        len(exempt),
        coverage(target, before or []),
        cover_after,
        has_main=main is not None,
        main_is_target=main is not None and iou(main, target) >= MAIN_TARGET_IOU,
        main_read_as_target=main is not None
        and bool(use_case._matching_plates(run, image, [main], reported)),
        covers=covering_boxes(detections, exempt_after, thresholds, *size, target) if cover_after > 0 else (),
        others=tuple((key, coverage(box, before or []), coverage(box, after or [])) for key, box in others),
    )


def measure_samples(
    use_case: ValidateEvidence,
    samples: Iterable[Sample],
    photos_dir: Path,
    model_max_side_px: int,
    thresholds: RedactionThresholds,
    annotated: Mapping[tuple[str, str], Sequence[Box]] | None = None,
) -> tuple[list[Measure], Counter[str]]:
    """Mide cada patente legible de auto. Las lecturas del OCR se reutilizan entre filas de la misma foto."""
    decoder = CvImageDecoder()
    by_photo: dict[tuple[str, str], list[tuple[int, Sample]]] = {}
    for position, sample in enumerate(samples):
        if sample_format(sample) in IN_SCOPE_FORMATS:
            by_photo.setdefault((sample.split, sample.file), []).append((position, sample))
    measured: list[tuple[int, Measure]] = []
    skipped: Counter[str] = Counter()
    for (split, file), items in by_photo.items():  # una foto por vez: la decodificada pesa decenas de MB
        try:
            pixels = decoder.decode((photos_dir / split / file).read_bytes())
        except (OSError, ImageDecodeError):
            skipped["foto ilegible"] += len(items)
            continue
        image = EvidenceImage(
            role=EvidenceRole.PLATE,
            index=0,
            pixels=pixels,
            view=reduce_for_models(pixels, model_max_side_px),
        )
        run = _Run(models=use_case._models)  # el mismo conjunto de modelos
        for position, sample in items:
            others = [
                (f"{split}/{file}#{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}", box)
                for box in (annotated or {}).get((split, file), ())
                if iou(box, sample.box) < MAIN_TARGET_IOU  # otra patente, no la misma con otra caja
            ]
            measured.append((position, measure_one(use_case, run, image, sample, thresholds, others)))
    return [measure for _, measure in sorted(measured, key=lambda item: item[0])], skipped


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


def principal_summary(measures: Sequence[Measure], stage: str = "after") -> dict[str, Any]:
    """Cuántas veces la principal elegida no es la denunciada y qué pasa entonces con la denunciada."""

    def covered(m: Measure) -> float:
        return m.after if stage == "after" else m.before

    with_main = [m for m in measures if m.has_main]
    other = [m for m in with_main if not m.main_is_target]
    return {
        "sin_principal": len(measures) - len(with_main),
        "es_la_denunciada": sum(m.main_is_target for m in with_main),
        "no_es_la_denunciada": len(other),
        "no_es_la_denunciada_pct": rate(len(other), len(measures)),
        "es_la_denunciada_y_el_ocr_la_lee": sum(
            m.main_is_target and m.main_read_as_target for m in with_main
        ),
        "no_es_la_denunciada_y_la_denunciada_queda_tapada_algo": sum(covered(m) > 0 for m in other),
        "no_es_la_denunciada_y_la_denunciada_queda_tapada_total": sum(
            classify(covered(m)) == TOTAL for m in other
        ),
        "no_es_la_denunciada_y_el_ocr_la_lee_como_la_denunciada": sum(m.main_read_as_target for m in other),
    }


def others_summary(measures: Sequence[Measure]) -> dict[str, Any]:
    """Si las dos reglas nuevas dejan expuesta alguna otra patente anotada de la misma foto.

    Cada otra patente cuenta una vez (la fila donde más se destapa). "Expuesta" es que estaba tapada por
    completo antes (`TOTAL_FRACTION`) y deja de estarlo; `menos_tapadas`, cualquier caída de más de 5 puntos.
    """
    worst: dict[str, tuple[str, float, float]] = {}
    for m in measures:
        for key, before, after in m.others:
            if key not in worst or before - after > worst[key][1] - worst[key][2]:
                worst[key] = (m.ident, before, after)
    exposed = [(i, b, a) for i, b, a in worst.values() if b >= TOTAL_FRACTION > a]
    less = [(i, b, a) for i, b, a in worst.values() if a < b - 0.05]
    return {
        "patentes_ajenas_anotadas": len(worst),
        "tapadas_por_completo_antes": sum(b >= TOTAL_FRACTION for _, b, _ in worst.values()),
        "tapadas_por_completo_despues": sum(a >= TOTAL_FRACTION for _, _, a in worst.values()),
        "destapadas_por_las_reglas": len(exposed),
        "menos_tapadas": len(less),
        "casos": [
            {"id": ident, "antes": round(b, 3), "despues": round(a, 3)} for ident, b, a in sorted(less)
        ],
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
        "principal": principal_summary(available),
        "principal_antes_de_las_reglas": principal_summary(available, "before"),
        "ajenas": others_summary(available),
        "casos_tapados_con_exencion": [
            {
                "id": m.ident,
                "luz": m.light,
                "cobertura": round(m.after, 3),
                "cubren": list(m.covers),
            }
            for m in with_exemption
            if m.after > 0
        ],
        "tipos_de_caja_que_cubre_con_exencion": dict(
            Counter(cover["tipo"] for m in with_exemption if m.after > 0 for cover in m.covers)
        ),
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
            "contiene_entera_desde": EXEMPT_CONTAINING,
            "principal_es_la_denunciada_con_iou_desde": MAIN_TARGET_IOU,
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


def annotated_plates(photos_dir: Path) -> dict[tuple[str, str], list[Box]]:
    """Todas las patentes anotadas del export por foto (sin las cajas inválidas: autos, pilas de patentes)."""
    annotations = load_annotations(photos_dir)
    invalid = invalid_boxes(annotations)
    found: dict[tuple[str, str], list[Box]] = {}
    for annotation in annotations:
        if annotation in invalid:
            continue
        x, y, width, height = annotation.box
        found.setdefault((annotation.split, annotation.file), []).append(Box(x, y, x + width, y + height))
    return found


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
        use_case,
        loaded.samples,
        args.photos_dir,
        settings.model_max_side_px,
        thresholds,
        annotated_plates(args.photos_dir),
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
    main = report["principal"]
    print(
        f"Patente principal: no es la denunciada {main['no_es_la_denunciada']} "
        f"({main['no_es_la_denunciada_pct']} %) · es la denunciada {main['es_la_denunciada']} · "
        f"sin principal {main['sin_principal']}"
    )
    others = report["ajenas"]
    print(
        f"Otras patentes anotadas: {others['patentes_ajenas_anotadas']} · tapadas por completo antes "
        f"{others['tapadas_por_completo_antes']} y después {others['tapadas_por_completo_despues']} · "
        f"destapadas por las reglas {others['destapadas_por_las_reglas']} · "
        f"menos tapadas {others['menos_tapadas']}"
    )
    print(
        f"Con exención y tapada igual: {len(report['casos_tapados_con_exencion'])} · "
        f"cajas que la cubren: {report['tipos_de_caja_que_cubre_con_exencion']}"
    )
    print(f"Salida: {args.out_dir / 'redaction.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
