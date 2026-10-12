"""Evalúa cómo decide el servicio si un auto bloquea un rebaje, sobre el split `test` de las fotos propias.

Compara tres sistemas sobre las mismas fotos, con el auto de `rfdetr_seg_coco` y la misma función de
decisión del dominio (`driveway_check`):

- **sam3**: SAM 3 sin entrenar (con prompts de texto), la referencia;
- **entrenado**: el RF-DETR-Seg entrenado con tus fotos (`rfdetr_seg_curb`);
- **oraculo**: las máscaras anotadas a mano. No es un modelo: aísla el error de la *decisión* del error de las
  *máscaras*.

La decisión esperada de cada tipo de foto (la lista está en `EXPECTED`) permite medir la precisión y el recall
de BLOCKING y qué parte de las fotos cae en BLOCKING_MARGINAL. Cada falla se clasifica en "máscara mala" (la
máscara del sistema no se parece a la anotada: hace falta mejor modelo o más datos) o "decisión mala con
máscara buena" (las máscaras están bien y falla el criterio: se ajusta `RAMP_MARGIN`, no se reentrena).

Fotos con patentes y rostros de terceros: todo queda fuera del repo y la consola muestra solo cifras.
Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import Mask
from vvi_ia.domain.ramp import DrivewayCheck, driveway_check, wheel_diameter
from vvi_ia.service.validate_evidence import largest_car

REPO_ROOT = Path(__file__).resolve().parents[2]
CURB_FOLDER = "curb"
ANNOTATIONS = "_annotations.coco.json"
DEFAULT_MARGIN = 0.08  # los valores por defecto de RAMP_MARGIN y RAMP_SCALE_FALLBACK en `config.py`
DEFAULT_SCALE_FALLBACK = 0.45
GOOD_MASK_IOU = 0.50  # una máscara con este IoU contra la anotada se considera buena
MARGIN_SWEEP = tuple(round(0.02 * step, 2) for step in range(0, 16))  # 0,00 a 0,30
SYSTEMS = ("sam3", "entrenado", "oraculo")
NO_MODEL = "none"
MASK_BAD = "mascara_mala"
DECISION_BAD = "decision_mala_con_mascara_buena"
SAM_PROMPTS = {"curb": ("curb",), "curb_cut": ("dropped curb", "curb ramp")}
FAILURES_FOLDER = "failures"
GT_COLOR = (0, 158, 115)  # verde (RGB): lo anotado a mano
PRED_COLOR = (213, 94, 0)  # bermellón: lo que predijo el sistema

BLOCKING = DrivewayCheck.BLOCKING
MARGINAL = DrivewayCheck.BLOCKING_MARGINAL
# Decisiones que se aceptan para cada tipo de foto (los nombres son los del `tipo` de fotos.csv)
EXPECTED: Mapping[str, frozenset[DrivewayCheck]] = {
    "positivo_claro": frozenset({BLOCKING}),
    "positivo_marginal": frozenset({BLOCKING, MARGINAL}),
    "negativo_auto_junto_rebaje": frozenset({DrivewayCheck.NOT_BLOCKING}),
    "negativo_auto_sin_rebaje": frozenset({DrivewayCheck.NO_DRIVEWAY}),
    "negativo_parecido": frozenset(
        {DrivewayCheck.NO_DRIVEWAY, DrivewayCheck.NO_CAR}
    ),  # NO_CAR si no hay auto
    "negativo_rebaje_sin_auto": frozenset({DrivewayCheck.NO_CAR}),
}
POSITIVE_TYPES = ("positivo_claro", "positivo_marginal")
MUST_FIND_CUT = "negativo_rebaje_sin_auto"  # además de NO_CAR, tiene que detectar el curb_cut


@dataclass(frozen=True)
class Sample:
    """Una foto del test con todo lo necesario para decidir: máscaras anotadas, auto y rueda."""

    name: str
    kind: str
    image: NDArray[np.uint8]
    gt_curb: Mask
    gt_cut: Mask
    car: Mask | None  # el auto más grande, o `None` si el segmentador no encontró ninguno
    wheel_diam: float | None


# Un sistema recibe la foto (y lo anotado, solo el oráculo lo usa) y devuelve `(curb, curb_cut)` o `None`
Predictor = Callable[["Sample"], "tuple[Mask, Mask] | None"]


# --- máscaras anotadas y calidad -------------------------------------------------------------------------


def polygons_mask(polygons: Sequence[Sequence[float]], shape: tuple[int, int]) -> Mask:
    canvas = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(canvas, [np.round(np.array(polygon)).astype(np.int32).reshape(-1, 2)], 1)
    return canvas.astype(bool)


def ground_truth(coco: Mapping[str, Any], file_name: str, shape: tuple[int, int]) -> tuple[Mask, Mask]:
    """Máscaras `(curb, curb_cut)` anotadas para una foto del COCO."""
    names = {category["id"]: category["name"] for category in coco["categories"]}
    image_id = next(image["id"] for image in coco["images"] if image["file_name"] == file_name)
    masks = {"curb": np.zeros(shape, bool), "curb_cut": np.zeros(shape, bool)}
    for annotation in coco["annotations"]:
        if annotation["image_id"] == image_id and names[annotation["category_id"]] in masks:
            masks[names[annotation["category_id"]]] |= polygons_mask(annotation["segmentation"], shape)
    return masks["curb"], masks["curb_cut"]


def iou(a: Mask, b: Mask) -> float | None:
    """IoU de dos máscaras; `None` si las dos están vacías (no hay nada que comparar)."""
    union = int(np.logical_or(a, b).sum())
    return None if union == 0 else float(np.logical_and(a, b).sum() / union)


def is_good(prediction: Mask, truth: Mask) -> bool:
    """Una máscara es buena si se parece a la anotada, o si las dos están vacías."""
    value = iou(prediction, truth)
    return True if value is None else value >= GOOD_MASK_IOU


# --- decisión ------------------------------------------------------------------------------------------


def decide(
    curb: Mask,
    cut: Mask,
    car: Mask | None,
    wheel_diam: float | None,
    margin: float = DEFAULT_MARGIN,
    scale_fallback: float = DEFAULT_SCALE_FALLBACK,
) -> DrivewayCheck:
    """La decisión del dominio. Sin auto no se llama a `driveway_check`, como en el servicio."""
    if car is None or not car.any():
        return DrivewayCheck.NO_CAR
    return driveway_check(curb, cut, car, wheel_diam=wheel_diam, margin=margin, scale_fallback=scale_fallback)


def judge(sample: Sample, prediction: tuple[Mask, Mask], decision: DrivewayCheck) -> dict[str, Any]:
    """Si la decisión es la esperada y, si no, por qué falló."""
    ok = decision in EXPECTED[sample.kind]
    curb_ok, cut_ok = is_good(prediction[0], sample.gt_curb), is_good(prediction[1], sample.gt_cut)
    cut_found = iou(prediction[1], sample.gt_cut)
    if sample.kind == MUST_FIND_CUT:
        ok = ok and cut_found is not None and cut_found >= GOOD_MASK_IOU
    reason = None if ok else (DECISION_BAD if curb_ok and cut_ok else MASK_BAD)
    return {"ok": ok, "decision": decision, "mascara_buena": curb_ok and cut_ok, "causa": reason}


# --- métricas ------------------------------------------------------------------------------------------


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(judged: Sequence[tuple[Sample, dict[str, Any]]]) -> dict[str, Any]:
    """Precisión y recall de BLOCKING, % MARGINAL, resultados por tipo y por qué fallan."""
    predicted = [(s, j) for s, j in judged if j["decision"] == BLOCKING]
    positives = [(s, j) for s, j in judged if s.kind in POSITIVE_TYPES]
    true_positives = sum(s.kind in POSITIVE_TYPES for s, _ in predicted)
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in EXPECTED:
        rows = [(s, j) for s, j in judged if s.kind == kind]
        if rows:
            by_kind[kind] = {
                "fotos": len(rows),
                "aciertos": sum(j["ok"] for _, j in rows),
                "decisiones": dict(Counter(j["decision"].value for _, j in rows)),
            }
    failures = Counter(j["causa"] for _, j in judged if not j["ok"])
    return {
        "fotos": len(judged),
        "aciertos": sum(j["ok"] for _, j in judged),
        "precision_blocking": rate(true_positives, len(predicted)),
        "recall_blocking": rate(true_positives, len(positives)),
        "pct_marginal": rate(sum(j["decision"] == MARGINAL for _, j in judged), len(judged)),
        "pct_marginal_en_positivas": rate(
            sum(j["decision"] == MARGINAL for _, j in positives), len(positives)
        ),
        "por_tipo": by_kind,
        "fallas": {MASK_BAD: failures.get(MASK_BAD, 0), DECISION_BAD: failures.get(DECISION_BAD, 0)},
    }


def margin_sweep(samples: Sequence[Sample], margins: Sequence[float] = MARGIN_SWEEP) -> list[dict[str, Any]]:
    """Con las máscaras anotadas (el oráculo): qué `RAMP_MARGIN` acierta más decisiones."""
    rows = []
    for margin in margins:
        results = [
            decide(s.gt_curb, s.gt_cut, s.car, s.wheel_diam, margin) in EXPECTED[s.kind]
            for s in samples
            if not (s.kind == MUST_FIND_CUT and not s.gt_cut.any())
        ]
        rows.append(
            {
                "margen": margin,
                "aciertos": sum(results),
                "fotos": len(results),
                "tasa": rate(sum(results), len(results)),
            }
        )
    return rows


def recommend_margin(sweep: Sequence[Mapping[str, Any]], default: float = DEFAULT_MARGIN) -> float | None:
    """El margen con más aciertos; ante un empate, el más cercano al valor por defecto."""
    valid = [row for row in sweep if row["tasa"] is not None]
    if not valid:
        return None
    best = max(row["tasa"] for row in valid)
    return float(min((row["margen"] for row in valid if row["tasa"] == best), key=lambda m: abs(m - default)))


def build_report(
    predictors: Mapping[str, Predictor], samples: Sequence[Sample], margin: float, scale_fallback: float
) -> tuple[dict[str, Any], dict[str, list[tuple[Sample, tuple[Mask, Mask], dict[str, Any]]]]]:
    """Reporte de cifras y el detalle por foto de cada sistema (para dibujar las fallas)."""
    detail: dict[str, list[tuple[Sample, tuple[Mask, Mask], dict[str, Any]]]] = {}
    report: dict[str, Any] = {
        "fotos_del_test": len(samples),
        "tipos": dict(Counter(s.kind for s in samples)),
        "sistemas": {},
    }
    for name, predictor in predictors.items():
        rows = []
        failed = 0
        for sample in samples:
            prediction = predictor(sample)
            if prediction is None:
                failed += 1
                prediction = (np.zeros_like(sample.gt_curb), np.zeros_like(sample.gt_cut))
            decision = decide(
                prediction[0], prediction[1], sample.car, sample.wheel_diam, margin, scale_fallback
            )
            rows.append((sample, prediction, judge(sample, prediction, decision)))
        detail[name] = rows
        report["sistemas"][name] = {**summarize([(s, j) for s, _, j in rows]), "fallas_del_modelo": failed}
    sweep = margin_sweep(samples)
    report["margen"] = {
        "actual": margin,
        "barrido_con_oraculo": sweep,
        "recomendado": recommend_margin(sweep, margin),
    }
    return report, detail


def draw_failure(
    sample: Sample, prediction: tuple[Mask, Mask], judged: Mapping[str, Any]
) -> NDArray[np.uint8]:
    """La foto con lo anotado (verde) y lo predicho (bermellón) y la decisión."""
    out = sample.image.copy()
    for color, masks in ((GT_COLOR, (sample.gt_curb, sample.gt_cut)), (PRED_COLOR, prediction)):
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)
    text = f"{sample.kind}: {judged['decision'].value} ({judged['causa']})"
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def write_failures(
    detail: Mapping[str, Sequence[tuple[Sample, tuple[Mask, Mask], dict[str, Any]]]], out_dir: Path
) -> dict[str, int]:
    written: dict[str, int] = {}
    for system, rows in detail.items():
        folder = out_dir / FAILURES_FOLDER / system
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.jpg"):
            old.unlink()
        count = 0
        for sample, prediction, judged in rows:
            if not judged["ok"]:
                canvas = draw_failure(sample, prediction, judged)
                cv2.imwrite(
                    str(folder / f"{Path(sample.name).stem}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                )
                count += 1
        written[system] = count
    return written


# --- armado de las muestras y de los sistemas ---------------------------------------------------------------


def load_samples(
    split_dir: Path,
    kinds: Mapping[str, str],
    cars: Callable[[Any], Any],
    wheels: Callable[[Any, Mask], float | None] | None = None,
) -> list[Sample]:
    """Fotos del split con su tipo, sus máscaras anotadas, el auto más grande y la rueda."""
    coco = json.loads((split_dir / ANNOTATIONS).read_text(encoding="utf-8"))
    decoder = CvImageDecoder()
    samples: list[Sample] = []
    for image in coco["images"]:
        name = image["file_name"]
        kind = kinds.get(name)
        if kind not in EXPECTED:
            continue
        pixels = decoder.decode((split_dir / name).read_bytes())
        gt_curb, gt_cut = ground_truth(coco, name, pixels.shape[:2])
        found = cars(pixels)
        car = largest_car(found).mask if found else None
        diameter = wheels(pixels, car) if wheels is not None and car is not None else None
        samples.append(Sample(name, kind, pixels, gt_curb, gt_cut, car, diameter))
    return samples


def annotated(sample: Sample) -> tuple[Mask, Mask]:
    """El "sistema" oráculo: devuelve las máscaras anotadas a mano de la foto."""
    return sample.gt_curb, sample.gt_cut


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument(
        "--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="pesos (MODELS_DIR)"
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--segmenter", default="rfdetr_seg_coco", help="modelo de autos")
    parser.add_argument(
        "--curb-model",
        default="rfdetr_seg_curb",
        help="modelo de rebajes entrenado; 'none' para no evaluarlo",
    )
    parser.add_argument("--detector", help="modelo del puerto detector, para medir la rueda (opcional)")
    parser.add_argument(
        "--no-sam3", action="store_true", help="no correr SAM 3 (no hace falta la GPU ni el acceso)"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="dispositivo de SAM 3")
    parser.add_argument("--margin", type=float, default=float(os.environ.get("RAMP_MARGIN", DEFAULT_MARGIN)))
    parser.add_argument(
        "--scale-fallback",
        type=float,
        default=float(os.environ.get("RAMP_SCALE_FALLBACK", DEFAULT_SCALE_FALLBACK)),
    )
    args = parser.parse_args(argv)
    for flag, value in (
        ("--data-dir", args.data_dir),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    return args


def main(
    argv: list[str] | None = None, sam3_factory: Callable[[argparse.Namespace], Any] | None = None
) -> int:
    args = parse_args(argv)
    curb = Path(args.data_dir) / CURB_FOLDER
    if any(Path(p).resolve().is_relative_to(REPO_ROOT) for p in (curb, args.out_dir)):
        print("Las fotos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    split_dir = curb / "dataset" / "test"
    if not (split_dir / ANNOTATIONS).is_file():
        print(f"Falta el split test en {split_dir}. Corré antes prepare.py.", file=sys.stderr)
        return 1
    trained = args.curb_model.lower() != NO_MODEL
    selection = {"segmenter": args.segmenter, **({"curb": args.curb_model} if trained else {})}
    if args.detector:
        selection["detector"] = args.detector
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "curb"))
        from photos import PhotosError, load_photos

        kinds = {p.file: p.kind for p in load_photos(curb / "fotos.csv")}
        models = build_models(selection, args.manifest, args.models_dir)
    except (PhotosError, ModelSetupError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    measure = None
    if args.detector:
        from vvi_ia.domain.model import BoxLabel

        def measure(image: NDArray[np.uint8], car: Mask) -> float | None:
            found = [box for box in models.vehicle.detect(image) or () if box.label == BoxLabel.WHEEL]
            return wheel_diameter(found, car)

    samples = load_samples(split_dir, kinds, models.cars.cars, measure)
    predictors: dict[str, Predictor] = {"oraculo": annotated}
    if trained:
        predictors = {"entrenado": lambda sample: models.curb.curb(sample.image), **predictors}
    if not args.no_sam3:
        factory = sam3_factory or _default_sam3
        predictors = {"sam3": factory(args), **predictors}
    report, detail = build_report(predictors, samples, args.margin, args.scale_fallback)
    report["modelos"] = {
        "segmenter": models.versions()["segmenter"],
        "curb": models.versions()["curb"] if trained else None,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "curb_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failures = write_failures(detail, Path(args.out_dir))
    print(f"{report['fotos_del_test']} fotos del test · tipos {report['tipos']}")
    for name, row in report["sistemas"].items():
        print(
            f"{name:<10} aciertos {row['aciertos']}/{row['fotos']} · "
            f"BLOCKING precisión {row['precision_blocking']} "
            f"recall {row['recall_blocking']} · MARGINAL {row['pct_marginal']} · fallas {row['fallas']}"
        )
    print(
        f"RAMP_MARGIN: actual {report['margen']['actual']} · "
        f"recomendado con las máscaras anotadas {report['margen']['recomendado']}"
    )
    print(f"Fallas dibujadas en {Path(args.out_dir) / FAILURES_FOLDER}: {failures}")
    return 0


def _default_sam3(args: argparse.Namespace) -> Predictor:
    """SAM 3 sin entrenar como predictor: une las instancias de cada clase en una máscara."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
    import prelabel_curb

    backend = prelabel_curb.Sam3Backend(
        args.device, prelabel_curb.DEFAULT_THRESHOLD, prelabel_curb.DEFAULT_MASK_THRESHOLD
    )

    def predict(sample: Sample) -> tuple[Mask, Mask]:
        image = sample.image
        instances = prelabel_curb.process_photo(
            image, backend, SAM_PROMPTS, prelabel_curb.DEFAULT_SIMPLIFY_PX, prelabel_curb.DEFAULT_MIN_AREA
        )
        masks = {name: np.zeros(image.shape[:2], bool) for name in SAM_PROMPTS}
        for instance in instances:
            masks[instance.label] |= polygons_mask(instance.polygons, image.shape[:2])
        return masks["curb"], masks["curb_cut"]

    return predict


if __name__ == "__main__":
    sys.exit(main())
