"""R3 con confusiones del OCR: ¿cuándo una diferencia entre la lectura y la patente reportada es una duda?

Compara tres criterios para declarar `PLATE_MISMATCH` (RED, fraude) sobre las lecturas de PARSeq del dataset
"patentes" de Roboflow, con la misma carga y recorte que `ocr.py`:

- **V0, la actual**: lectura segura (confianza >= `OCR_MIN_CONF`) y diferencia en 2 o más caracteres;
  una sola diferencia es duda (GREEN).
- **V1, la actual + confusiones**: igual que V0, pero una diferencia entre caracteres que el OCR confunde con
  frecuencia no cuenta (es duda, no una patente distinta).
- **V2, solo confusiones y baja confianza**: sin la regla de "un solo carácter"; hay MISMATCH si algún
  carácter difiere, no es una confusión frecuente y PARSeq lo leyó con confianza >= `t_char`. No hay un
  umbral global sobre la lectura: la duda por baja confianza es por carácter.

Las diferencias se miden después de la normalización por posición del dominio (`plate_check`): con el mismo
largo, un carácter se lee como dígito o como letra según la patente reportada (O/0, I/1, B/8, S/5, Z/2,
G/6...). Con largos distintos se alinea con la distancia de edición.

**Lista de confusiones.** Sale de los errores reales de PARSeq en este dataset: las diferencias entre la
lectura y la etiqueta en las lecturas del mismo largo con 3 diferencias o menos (más que eso es un recorte
roto, no una confusión). Una pareja entra si aparece en al menos `MIN_PAIR_COUNT` lecturas. Como la lista y la
evaluación usan las mismas lecturas, **se evalúa con "dejar una afuera"**: para cada lectura la lista se arma
sin sus propios errores, así que U2 no premia memorizar. También se evalúa la lista a priori (C/G, O/D/Q, M/N,
E/F, P/R, U/V, 3/8, 1/7, 5/6), que no depende del dataset.

**Métricas.** S5: de los pares cruzados (la lectura de una foto contra la etiqueta de otra patente) cuántos
salen MISMATCH. U2: de las lecturas contra su propia etiqueta, cuántas salen MISMATCH (una patente correcta
marcada como equivocada), con la cota del 95 %. Las lecturas y los textos quedan en RAM; `report.json` tiene
solo cifras y parejas de caracteres, nunca patentes. Uso: ver `tools/eval/README.md`.
"""

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr import (
    OTHER_FORMAT,
    OVERRIDES_PATH,
    SPLITS,
    label_format,
    load_overrides,
    load_samples,
)
from presence import upper_bound
from rapidfuzz.distance import Levenshtein

from vvi_ia.adapters.imaging import CvImageDecoder
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.domain.model import EvidenceImage, EvidenceRole
from vvi_ia.domain.plate import TO_DIGIT, TO_LETTER, canonicalize
from vvi_ia.service.ports import ImageDecodeError
from vvi_ia.service.validate_evidence import crop_box

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = "ocr-roboflow"
CURRENT_MIN_CONF = 0.78  # OCR_MIN_CONF del servicio
GATES = (0.60, 0.78)
T_CHARS = (0.50, 0.60, 0.70, 0.78, 0.90, 0.95)
MIN_PAIR_COUNT = 2
MAX_ERRORS_FOR_LIST = 3
APRIORI = (
    "CG",
    "ODQ",
    "MN",
    "EF",
    "PR",
    "UV",
    "38",
    "17",
    "56",
)  # familias del pedido, parejas dentro de cada una

Pair = tuple[str, str]


def pairs_of(families: Sequence[str]) -> frozenset[Pair]:
    return frozenset((min(a, b), max(a, b)) for family in families for a in family for b in family if a != b)


APRIORI_PAIRS = pairs_of(APRIORI)


@dataclass(frozen=True)
class Reading:
    """Una lectura: etiqueta, texto canónico, probabilidad por carácter y formato. Solo en RAM."""

    label: str
    text: str
    probs: tuple[float, ...]
    in_scope: bool

    @property
    def conf(self) -> float:
        return min(self.probs, default=0.0)


@dataclass(frozen=True)
class Diff:
    """Una diferencia contra la patente reportada: caracteres y probabilidad del leído (1,0 si no hay)."""

    reported: str | None  # None: sobra un carácter en la lectura
    read: str | None  # None: falta un carácter en la lectura
    prob: float

    @property
    def pair(self) -> Pair | None:
        if self.reported is None or self.read is None:
            return None
        return (min(self.reported, self.read), max(self.reported, self.read))


def normalize(read: str, reported: str) -> str:
    """El carácter leído como dígito o letra según el de la patente reportada en la misma posición."""
    return read.translate(TO_DIGIT if reported.isdigit() else TO_LETTER)


def differences(reported: str, reading: Reading) -> list[Diff]:
    """Diferencias entre la patente reportada y la lectura, después de la normalización por posición."""
    text, probs = reading.text, reading.probs
    if len(text) == len(reported):
        return [
            Diff(r, normalize(o, r), p)
            for r, o, p in zip(reported, text, probs, strict=True)
            if normalize(o, r) != r
        ]
    found: list[Diff] = []
    for op in Levenshtein.editops(reported, text):
        if op.tag == "replace":
            r, o = reported[op.src_pos], text[op.dest_pos]
            if normalize(o, r) != r:
                found.append(Diff(r, normalize(o, r), probs[op.dest_pos]))
        elif op.tag == "insert":  # sobra un carácter en la lectura
            found.append(Diff(None, text[op.dest_pos], probs[op.dest_pos]))
        else:  # falta un carácter: la confianza es la del vecino menos seguro
            near = [probs[i] for i in (op.src_pos - 1, op.src_pos) if 0 <= i < len(probs)]
            found.append(Diff(reported[op.src_pos], None, min(near, default=1.0)))
    return found


def is_confusion(diff: Diff, confusions: frozenset[Pair]) -> bool:
    return diff.pair is not None and diff.pair in confusions


def v0(diffs: Sequence[Diff], reading: Reading, _c: frozenset[Pair], gate: float, _t: float) -> bool:
    return reading.conf >= gate and len(diffs) >= 2  # noqa: PLR2004


def v1(diffs: Sequence[Diff], reading: Reading, c: frozenset[Pair], gate: float, _t: float) -> bool:
    return reading.conf >= gate and sum(not is_confusion(d, c) for d in diffs) >= 2  # noqa: PLR2004


def v2(diffs: Sequence[Diff], reading: Reading, c: frozenset[Pair], _gate: float, t_char: float) -> bool:
    return bool(reading.text) and any(not is_confusion(d, c) and d.prob >= t_char for d in diffs)


Variant = Callable[[Sequence[Diff], Reading, frozenset[Pair], float, float], bool]


def error_pairs(reading: Reading) -> Counter[Pair]:
    """Parejas de caracteres que el OCR confundió en esta lectura (vacío si el recorte está roto)."""
    if not reading.in_scope or len(reading.text) != len(reading.label):
        return Counter()
    diffs = differences(reading.label, reading)
    if not diffs or len(diffs) > MAX_ERRORS_FOR_LIST:
        return Counter()
    return Counter(d.pair for d in diffs if d.pair is not None)  # una vez por lectura y pareja


def pair_counts(readings: Sequence[Reading]) -> Counter[Pair]:
    total: Counter[Pair] = Counter()
    for reading in readings:
        total.update(set(error_pairs(reading)))
    return total


def leave_one_out(
    readings: Sequence[Reading],
) -> list[frozenset[Pair]]:
    """Por lectura, las parejas frecuentes contando solo las demás lecturas."""
    total = pair_counts(readings)
    lists = []
    for reading in readings:
        own = set(error_pairs(reading))
        lists.append(frozenset(pair for pair, n in total.items() if n - (pair in own) >= MIN_PAIR_COUNT))
    return lists


def evaluate(
    readings: Sequence[Reading],
    lists: Sequence[frozenset[Pair]],
    variant: Variant,
    gate: float,
    t_char: float,
) -> dict[str, Any]:
    """S5 (pares cruzados) y U2 (lecturas contra su propia etiqueta) de una variante."""
    scope = [(i, r) for i, r in enumerate(readings) if r.in_scope and r.text]
    labels = sorted({r.label for _, r in scope})
    detected = pairs = 0
    for i, reading in scope:
        for label in labels:
            if label == reading.label:
                continue
            pairs += 1
            detected += variant(differences(label, reading), reading, lists[i], gate, t_char)
    flagged = sum(
        variant(differences(r.label, r), r, lists[i], gate, t_char) for i, r in enumerate(readings) if r.text
    )
    total = sum(1 for r in readings if r.text)
    return {
        "S5_detectadas": detected,
        "S5_pares": pairs,
        "S5": round(detected / pairs, 4) if pairs else None,
        "U2_marcadas": flagged,
        "U2_lecturas": total,
        "U2": round(flagged / total, 4) if total else None,
        "U2_cota_95": upper_bound(flagged, total),
    }


def read_all(args: argparse.Namespace) -> list[Reading]:
    """Todas las lecturas con la probabilidad de cada carácter (recorte igual que el servicio)."""
    loaded = load_samples(args.dataset_dir, args.splits, load_overrides(args.overrides))
    models = build_models({"ocr": args.model}, args.manifest, args.models_dir)
    reader = models.plate
    read_chars = getattr(reader, "read_chars", None)
    if read_chars is None:
        raise ModelSetupError("el adaptador de OCR no ofrece read_chars")
    decoder = CvImageDecoder()
    images: dict[tuple[str, str], EvidenceImage | None] = {}
    readings: list[Reading] = []
    for sample in loaded.samples:
        key = (sample.split, sample.file)
        if key not in images:
            try:
                pixels = decoder.decode((args.dataset_dir / sample.split / sample.file).read_bytes())
                images[key] = EvidenceImage(role=EvidenceRole.PLATE, index=0, pixels=pixels)
            except (OSError, ImageDecodeError):
                images[key] = None
        image = images[key]
        crop = None if image is None else crop_box(image, sample.box)
        if crop is None:
            continue
        found = read_chars(crop)
        if found is None:
            continue
        raw, probs = found
        kept = [i for i, ch in enumerate(raw) if ch.isascii() and ch.isalnum()]
        text = canonicalize(raw)
        readings.append(
            Reading(
                label=sample.label,
                text=text,
                probs=tuple(probs[i] for i in kept) if len(kept) == len(text) else (),
                in_scope=label_format(sample.label) != OTHER_FORMAT,
            )
        )
    return [r for r in readings if len(r.probs) == len(r.text)]


def build_report(readings: Sequence[Reading]) -> dict[str, Any]:
    empty: list[frozenset[Pair]] = [frozenset()] * len(readings)
    learned = leave_one_out(readings)
    apriori: list[frozenset[Pair]] = [APRIORI_PAIRS] * len(readings)
    rows: list[dict[str, Any]] = []

    def add(
        name: str, variant: Variant, lists: Sequence[frozenset[Pair]], gate: float, t_char: float
    ) -> None:
        rows.append(
            {
                "variante": name,
                "OCR_MIN_CONF": gate if variant is not v2 else None,
                "t_char": t_char if variant is v2 else None,
                **evaluate(readings, lists, variant, gate, t_char),
            }
        )

    for gate in GATES:
        add("V0 actual (difiere en 2 o más)", v0, empty, gate, 0.0)
        add("V1 actual + confusiones (aprendidas, dejar una afuera)", v1, learned, gate, 0.0)
        add("V1 actual + confusiones (lista a priori)", v1, apriori, gate, 0.0)
    for t_char in T_CHARS:
        add("V2 confusiones (aprendidas) y baja confianza", v2, learned, 0.0, t_char)
        add("V2 confusiones (lista a priori) y baja confianza", v2, apriori, 0.0, t_char)
        add("V2 sin lista de confusiones, solo baja confianza", v2, empty, 0.0, t_char)
    total = pair_counts(readings)
    in_scope = [r for r in readings if r.in_scope]
    return {
        "lecturas": len(readings),
        "lecturas_en_alcance": len(in_scope),
        "lecturas_exactas": sum(r.text == r.label for r in in_scope),
        "lecturas_con_errores_para_la_lista": sum(bool(error_pairs(r)) for r in readings),
        "lista_aprendida": {
            f"{a}/{b}": n for (a, b), n in sorted(total.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 1
        },
        "lista_aprendida_minimo": MIN_PAIR_COUNT,
        "lista_a_priori": sorted(f"{a}/{b}" for a, b in APRIORI_PAIRS),
        "variantes": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"))
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"))
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"))
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--model", default="parseq")
    args = parser.parse_args(argv)
    for flag, value in (
        ("--data-dir", args.data_dir),
        ("--out-dir", args.out_dir),
        ("--models-dir", args.models_dir),
    ):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    args.dataset_dir = args.data_dir / DATASET_DIR
    args.overrides = args.data_dir / OVERRIDES_PATH
    args.splits = [s for s in SPLITS if (args.dataset_dir / s / "_annotations.coco.json").is_file()]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(p.resolve().is_relative_to(REPO_ROOT) for p in (args.data_dir, args.out_dir)):
        print("Los datos y la salida no pueden estar dentro del repositorio.", file=sys.stderr)
        return 1
    try:
        readings = read_all(args)
    except (FileNotFoundError, ModelSetupError) as exc:
        print(f"No se pudo empezar: {exc}", file=sys.stderr)
        return 1
    report = build_report(readings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ocr_confusions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Lecturas {report['lecturas']} (en alcance {report['lecturas_en_alcance']}, "
        f"exactas {report['lecturas_exactas']})"
    )
    print(f"Lista aprendida (parejas y lecturas en que aparecen): {report['lista_aprendida']}")
    for row in report["variantes"]:
        knob = f"gate {row['OCR_MIN_CONF']}" if row["OCR_MIN_CONF"] is not None else f"t_char {row['t_char']}"
        print(
            f"{row['variante']:<62}{knob:<14} S5 {row['S5_detectadas']}/{row['S5_pares']} = {row['S5']}"
            f" · U2 {row['U2_marcadas']}/{row['U2_lecturas']} (cota {row['U2_cota_95']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
