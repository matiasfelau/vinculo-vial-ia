"""Evaluación offline de PARSeq (tiny y small, FP32 e INT8) como lector de patentes en el dispositivo.

Es la alternativa a ML Kit que se evaluó y se descartó (ver "Alternativa evaluada: PARSeq en el
dispositivo" en `tools/eval/README.md`). No toca el servicio ni la app. Cinco subcomandos:

- `export`: baja de Hugging Face los pesos del autor (`baudm/parseq-tiny`, `baudm/parseq-small`,
  Apache-2.0), los exporta a ONNX FP32 con el mismo exportador del servicio (`tools/export/parseq.py`) y
  los cuantiza a INT8 con la cuantización dinámica de ONNX Runtime. Necesita `torch` (extra `tools`).
- `eval`: lee los 300 recortes etiquetados, con margen (carpeta `mlkit`) y sin margen
  (`mlkit-sin-margen`), con los cuatro modelos. Aplica lo mismo que la app: normalización por
  posición (letra o dígito según el formato) y estructura válida (anterior o Mercosur). Informa lectura
  exacta, sin lectura y lectura equivocada, en total y por luz, sin umbral de confianza y con umbrales.
- `latency`: tiempo por recorte en la CPU de esta PC (1 hilo, 4 hilos y todos), solo como referencia.
- `aar`: cuánto pesa ONNX Runtime para Android por ABI (sin comprimir y comprimido), a partir de los
  `.aar` de Maven Central.
- `report`: junta `results.json`, `latency.json` y el tamaño de los ONNX en las tablas del README.

Los textos de las patentes solo viven en RAM: a disco van cifras y, en `rows-*.csv`, la lectura y su
confianza por id (son patentes de terceros: la salida no puede estar dentro del repositorio).
Uso: ver `tools/eval/README.md`.
"""

import argparse
import csv
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import ocr_labels
import onnxruntime as ort
from rapidfuzz.distance import Levenshtein

from vvi_ia.adapters.models.parseq import (
    MERCOSUR,
    OLD_FORMAT,
    _substitutions,
    decode_greedy,
    extract_plate_chars,
    preprocess,
)
from vvi_ia.domain.plate import TO_DIGIT, TO_LETTER, canonicalize

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "tools" / "export" / "parseq.py"
SIZES = ("tiny", "small")
QUANTS = ("fp32", "int8")
HUB_NAMES = {"tiny": "parseq_tiny", "small": "parseq"}  # nombres del hub de baudm/parseq
HF_REPOS = {"tiny": "baudm/parseq-tiny", "small": "baudm/parseq-small"}
WEIGHTS_FILE = "pytorch_model.bin"
CROPS = {
    "con_margen": "mlkit",
    "sin_margen": "mlkit-sin-margen",
}  # carpeta del set de cada variante de recorte
LIGHTS = ("dia", "noche")
GROUPS = ("total", *LIGHTS)
THRESHOLDS = (0.0, 0.5, 0.78, 0.9)  # 0 = sin umbral; 0,78 es OCR_MIN_CONF del servicio
INPUT_SIZE = (32, 128)  # alto y ancho de PARSeq
THREADS = (1, 4, 0)  # 0 = todos los núcleos
WARMUP = 20
ROUNDS = 3  # pasadas completas sobre los recortes
PERCENTILE = 0.95
EXACT, NONE, WRONG = "exacta", "sin_lectura", "equivocada"
RESULTS_FILE = "results.json"
LATENCY_FILE = "latency.json"
MS = 1000.0
MB = 1e6
ABIS = ("arm64-v8a", "armeabi-v7a", "x86", "x86_64")


def model_name(size: str, quant: str) -> str:
    return f"parseq-{size}-{quant}"


def all_models() -> list[str]:
    return [model_name(size, quant) for size in SIZES for quant in QUANTS]


@dataclass(frozen=True)
class Reading:
    """Lo que leyó un modelo en un recorte. Los textos solo viven en RAM."""

    ident: str
    light: str
    label: str  # patente canónica de la etiqueta
    text: str | None  # lectura normalizada, `None` si no hay una patente válida
    conf: float


def normalize(text: str, probs: Sequence[float]) -> tuple[str, float] | None:
    """Patente en formato válido con cada carácter leído como letra o dígito según su posición.

    Es lo que hace la app: busca una ventana Mercosur (`LLDDDLL`) o anterior (`LLLDDD`) tolerando confusiones
    letra/dígito, y las corrige por posición (O/0, I/1, B/8...). La confianza es la menor de sus caracteres.
    `None` si el texto no tiene la estructura de una patente: no hay lectura.
    """
    window, window_probs = extract_plate_chars(text, probs)
    for pattern in (MERCOSUR, OLD_FORMAT):
        if len(window) == len(pattern) and _substitutions(window, pattern) is not None:
            chars = []
            for char, kind in zip(window, pattern, strict=True):
                if kind == "L" and char.isdigit():
                    char = char.translate(TO_LETTER)
                elif kind == "D" and char.isalpha():
                    char = char.translate(TO_DIGIT)
                chars.append(char)
            return "".join(chars), min(window_probs, default=0.0)
    return None


def outcome(reading: Reading, threshold: float) -> str:
    """`sin_lectura` si no hay patente válida o no llega al umbral; si no, exacta o equivocada."""
    if reading.text is None or reading.conf < threshold:
        return NONE
    return EXACT if reading.text == reading.label else WRONG


def pct(count: int, total: int) -> float:
    return round(100 * count / total, 1) if total else 0.0


def block(rows: Sequence[Reading], threshold: float) -> dict[str, int | float]:
    """Cuántas lecturas exactas, sin lectura y equivocadas (y cuántas, a un carácter) hay en `rows`."""
    counts: Counter[str] = Counter()
    one_char = 0
    for row in rows:
        result = outcome(row, threshold)
        counts[result] += 1
        if result == WRONG and row.text is not None:
            one_char += Levenshtein.distance(row.text, row.label) == 1
    total = len(rows)
    return {
        "recortes": total,
        "exacta": counts[EXACT],
        "exacta_pct": pct(counts[EXACT], total),
        "sin_lectura": counts[NONE],
        "sin_lectura_pct": pct(counts[NONE], total),
        "equivocada": counts[WRONG],
        "equivocada_pct": pct(counts[WRONG], total),
        "equivocada_a_un_caracter": one_char,
    }


def threshold_key(threshold: float) -> str:
    return "sin_umbral" if threshold == 0.0 else f"umbral_{threshold}"


def summarize(rows: Sequence[Reading]) -> dict[str, dict[str, dict[str, int | float]]]:
    """`{umbral: {total|dia|noche: block}}` de las lecturas de un modelo sobre una variante de recorte."""
    return {
        threshold_key(threshold): {
            group: block([r for r in rows if group == "total" or r.light == group], threshold)
            for group in GROUPS
        }
        for threshold in THRESHOLDS
    }


def load_crops(folder: Path) -> dict[str, np.ndarray]:
    """Recortes RGB de la carpeta, por id (el nombre del archivo sin extensión)."""
    crops = {}
    for path in sorted(folder.glob("*.png")):
        bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            crops[path.stem] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return crops


def load_labels(set_dir: Path) -> dict[str, dict[str, str]]:
    """Las patentes legibles de auto del set etiquetado, por id."""
    return {row["id"]: row for row in ocr_labels.read_label_rows(set_dir) if ocr_labels.is_legible_car(row)}


def open_session(path: Path, threads: int = 0) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def read_crops(
    session: Any, crops: Mapping[str, np.ndarray], labels: Mapping[str, Mapping[str, str]]
) -> list[Reading]:
    """Corre el modelo sobre cada recorte etiquetado, en orden de id."""
    input_name = session.get_inputs()[0].name
    readings = []
    for ident in sorted(labels):
        (logits,) = session.run(None, {input_name: preprocess(crops[ident], *INPUT_SIZE)})
        found = normalize(*decode_greedy(logits))
        readings.append(
            Reading(
                ident=ident,
                light=labels[ident]["luz"],
                label=canonicalize(labels[ident]["texto"]),
                text=found[0] if found else None,
                conf=found[1] if found else 0.0,
            )
        )
    return readings


def write_readings(path: Path, readings: Sequence[Reading]) -> None:
    """id, luz, lectura y confianza por recorte. Tiene patentes de terceros: nunca va al repo."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "luz", "lectura", "confianza"])
        for row in readings:
            writer.writerow([row.ident, row.light, row.text or "", f"{row.conf:.4f}"])


def percentile(values: Sequence[float], q: float) -> float:
    """El valor en la posición `floor(q * n)` de los ordenados (la misma regla que `ocr_mlkit.py`)."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def timing(values_ms: Sequence[float]) -> dict[str, float]:
    return {
        "mediana_ms": round(statistics.median(values_ms), 2),
        "p95_ms": round(percentile(values_ms, PERCENTILE), 2),
        "media_ms": round(statistics.fmean(values_ms), 2),
        "minimo_ms": round(min(values_ms), 2),
        "maximo_ms": round(max(values_ms), 2),
    }


def time_model(
    session: Any, tensors: Sequence[np.ndarray], *, warmup: int = WARMUP, rounds: int = ROUNDS
) -> dict[str, dict[str, float]]:
    """Latencia de la inferencia y de la decodificación por recorte, con calentamiento y varias pasadas."""
    input_name = session.get_inputs()[0].name
    for tensor in tensors[:warmup]:
        session.run(None, {input_name: tensor})
    run_ms, decode_ms = [], []
    for _ in range(rounds):
        for tensor in tensors:
            start = time.perf_counter()
            (logits,) = session.run(None, {input_name: tensor})
            middle = time.perf_counter()
            decode_greedy(logits)
            end = time.perf_counter()
            run_ms.append((middle - start) * MS)
            decode_ms.append((end - middle) * MS)
    return {"inferencia": timing(run_ms), "decodificacion": timing(decode_ms)}


def time_preprocess(crops: Sequence[np.ndarray]) -> dict[str, float]:
    values = []
    for crop in crops:
        start = time.perf_counter()
        preprocess(crop, *INPUT_SIZE)
        values.append((time.perf_counter() - start) * MS)
    return timing(values)


def file_sizes(path: Path) -> dict[str, float]:
    """Tamaño del archivo, sin comprimir y comprimido con deflate (lo que viaja en un APK o AAB), en MB."""
    data = path.read_bytes()
    return {
        "mb": round(len(data) / MB, 2),
        "comprimido_mb": round(len(zlib.compress(data, 6)) / MB, 2),
    }


def aar_report(aar: Path) -> dict[str, Any]:
    """Peso de las bibliotecas nativas de un `.aar` por ABI: sin comprimir y comprimido (deflate)."""
    abis: dict[str, dict[str, float]] = {}
    with zipfile.ZipFile(aar) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) == 3 and parts[0] == "jni" and parts[2].endswith(".so"):  # noqa: PLR2004
                entry = abis.setdefault(parts[1], {"mb": 0.0, "comprimido_mb": 0.0})
                entry["mb"] += info.file_size / MB
                entry["comprimido_mb"] += info.compress_size / MB
    abis = {abi: {k: round(v, 2) for k, v in values.items()} for abi, values in sorted(abis.items())}
    return {
        "aar_mb": round(aar.stat().st_size / MB, 2),
        "abis": abis,
        "todas_las_abis": {
            key: round(sum(values[key] for values in abis.values()), 2) for key in ("mb", "comprimido_mb")
        },
    }


def apk_estimate(runtime: Mapping[str, float], model: Mapping[str, float]) -> dict[str, float]:
    """Una ABI de ONNX Runtime más un modelo: en disco (sin comprimir) y de descarga (comprimido), en MB."""
    return {
        "en_disco_mb": round(runtime["mb"] + model["mb"], 1),
        "descarga_mb": round(runtime["comprimido_mb"] + model["comprimido_mb"], 1),
    }


def quantize(source: Path, target: Path) -> None:
    """Cuantización dinámica de ONNX Runtime: pesos en INT8 (con signo), activaciones cuantizadas al vuelo."""
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)


def load_export_module() -> Any:
    spec = importlib.util.spec_from_file_location("export_parseq", EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"no se pudo cargar {EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_weights(size: str, weights_dir: Path) -> Path:
    """Los pesos del autor en Hugging Face (Apache-2.0 según su ficha); los baja si todavía no están."""
    target = weights_dir / f"parseq-{size}" / WEIGHTS_FILE
    if not target.is_file():
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        snapshot_download(repo_id=HF_REPOS[size], local_dir=str(target.parent))
    return target


def export_models(weights_dir: Path, onnx_dir: Path) -> None:  # pragma: no cover - necesita torch y la red
    """Exporta tiny y small a ONNX FP32 (verificado contra PyTorch) y los cuantiza a INT8."""
    import hashlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from vvi_ia.adapters.models.onnx_session import OnnxSession  # noqa: PLC0415
    from vvi_ia.adapters.models.parseq import CHARSET  # noqa: PLC0415

    mod = load_export_module()
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        system = torch.hub.load(
            f"{mod.HUB_REPO}:{mod.HUB_COMMIT}", HUB_NAMES[size], pretrained=False, trust_repo=True
        )
        state = torch.load(fetch_weights(size, weights_dir), map_location="cpu", weights_only=True)
        system.model.load_state_dict(state, strict=True)  # los pesos de HF no llevan el prefijo `model.`
        system.eval()
        tokenizer, model = system.tokenizer, system.model
        height, width = (int(side) for side in system.hparams.img_size)
        steps = int(system.hparams.max_label_length) + 1
        assert system.hparams.charset_train == CHARSET and tokenizer.eos_id == 0
        wrapper = mod.GreedyDecoder(model, tokenizer.bos_id, tokenizer.eos_id, steps).eval()
        sample = torch.randn(1, 3, height, width, generator=torch.Generator().manual_seed(0))
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            exported = Path(tmp) / "m.onnx"
            torch.onnx.export(
                wrapper, (sample,), str(exported), input_names=["image"], output_names=["logits"],
                opset_version=mod.OPSET, dynamo=False,
            )  # fmt: skip
            data = exported.read_bytes()
        session = OnnxSession(data, intra_op_threads=0)
        worst = 0.0
        for seed in mod.CHECK_SEEDS:
            image = torch.randn(1, 3, height, width, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                expected = model(tokenizer, image, steps - 1).numpy()
            (actual,) = session.run(None, {"image": image.numpy()})
            worst = max(worst, float(np.max(np.abs(actual - expected))))
            assert np.array_equal(actual.argmax(-1), expected.argmax(-1))
        fp32 = onnx_dir / f"{model_name(size, 'fp32')}.onnx"
        fp32.write_bytes(data)
        int8 = onnx_dir / f"{model_name(size, 'int8')}.onnx"
        quantize(fp32, int8)
        print(
            f"{size}: FP32 {len(data) / MB:.1f} MB (diferencia máxima con PyTorch {worst:.1e}, "
            f"sha256 {hashlib.sha256(data).hexdigest()[:12]}), INT8 {int8.stat().st_size / MB:.1f} MB"
        )


def evaluate_all(
    set_dir: Path, onnx_dir: Path, out_dir: Path, *, session_factory: Callable[[Path], Any] = open_session
) -> dict[str, Any]:
    """Los cuatro modelos sobre las dos variantes de recorte; escribe `results.json` y los `rows-*.csv`."""
    labels = load_labels(set_dir)
    results: dict[str, Any] = {}
    for crop_name, folder in CROPS.items():
        crops = load_crops(set_dir / folder)
        if set(crops) != set(labels):
            raise ValueError(
                f"los recortes de {folder}\\ no son los mismos que las etiquetas legibles de auto"
            )
        for name in all_models():
            readings = read_crops(session_factory(onnx_dir / f"{name}.onnx"), crops, labels)
            results[f"{name}|{crop_name}"] = summarize(readings)
            write_readings(out_dir / f"rows-{name}-{crop_name}.csv", readings)
    (out_dir / RESULTS_FILE).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def latency_all(
    set_dir: Path, onnx_dir: Path, out_dir: Path, *, rounds: int = ROUNDS, warmup: int = WARMUP
) -> dict[str, Any]:
    """Latencia por recorte de los cuatro modelos, sin margen, con 1, 4 y todos los hilos."""
    crops = list(load_crops(set_dir / CROPS["sin_margen"]).values())
    tensors = [preprocess(crop, *INPUT_SIZE) for crop in crops]
    results: dict[str, Any] = {
        "cpu": platform.processor(),
        "nucleos_logicos": os.cpu_count(),
        "ort": ort.__version__,
        "preproceso": time_preprocess(crops),
        "modelos": {},
    }
    for name in all_models():
        for threads in THREADS:
            session = open_session(onnx_dir / f"{name}.onnx", threads)
            key = f"{name}|{'todos' if threads == 0 else threads}"
            results["modelos"][key] = time_model(session, tensors, warmup=warmup, rounds=rounds)
            print(f"{key:24} inferencia {results['modelos'][key]['inferencia']}", flush=True)
    (out_dir / LATENCY_FILE).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def cell(block_: Mapping[str, Any]) -> str:
    """`exacta / sin lectura / equivocada` en porcentaje, con coma decimal."""
    values = (block_["exacta_pct"], block_["sin_lectura_pct"], block_["equivocada_pct"])
    return " / ".join(f"{value:.1f}".replace(".", ",") for value in values)


def report_tables(
    results: Mapping[str, Any], latency: Mapping[str, Any] | None, onnx_dir: Path | None
) -> str:
    """Tablas en Markdown: exacta / sin lectura / equivocada (%), en total y por luz, tamaño y latencia."""
    lines = []
    for crop_name in CROPS:
        lines += [
            f"**Recorte {crop_name.replace('_', ' ')}**",
            "",
            "| Modelo | Total | Día | Noche | ONNX (MB) | Latencia, 1 hilo: mediana (p95) |",
            "|---|---|---|---|---|---|",
        ]
        for name in all_models():
            entry = results[f"{name}|{crop_name}"]["sin_umbral"]
            size = ""
            if onnx_dir is not None and (onnx_dir / f"{name}.onnx").is_file():
                size = f"{file_sizes(onnx_dir / f'{name}.onnx')['mb']:.1f}".replace(".", ",")
            speed = ""
            if latency is not None and crop_name == "sin_margen":
                run = latency["modelos"][f"{name}|1"]["inferencia"]
                speed = f"{run['mediana_ms']:.1f} ms ({run['p95_ms']:.0f})".replace(".", ",")
            groups = " | ".join(cell(entry[group]) for group in GROUPS)
            lines.append(f"| {name} | {groups} | {size} | {speed} |")
        lines.append("")
    return "\n".join(lines)


def check_outside_repo(*named: tuple[str, Path | None]) -> str | None:
    for name, path in named:
        if path is not None and path.resolve().is_relative_to(REPO_ROOT):
            return f"{name} no puede estar dentro del repositorio: tiene patentes de terceros."
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="pesos de Hugging Face a ONNX FP32 e INT8")
    export.add_argument("--weights-dir", type=Path, required=True, help="donde están (o se bajan) los pesos")
    export.add_argument("--onnx-dir", type=Path, required=True, help="salida de los ONNX")
    for name in ("eval", "latency"):
        run = sub.add_parser(name, help="medir sobre el set etiquetado")
        run.add_argument(
            "--labeled-set",
            type=Path,
            required=True,
            help="set de ocr_set.py (con mlkit\\ y mlkit-sin-margen\\)",
        )
        run.add_argument("--onnx-dir", type=Path, required=True)
        run.add_argument("--out-dir", type=Path, required=True)
    aar = sub.add_parser("aar", help="peso de ONNX Runtime para Android, por ABI")
    aar.add_argument("aars", nargs="+", type=Path, help="archivos .aar de Maven Central")
    report = sub.add_parser("report", help="tablas del README")
    report.add_argument("--out-dir", type=Path, required=True, help="donde están results.json y latency.json")
    report.add_argument("--onnx-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "aar":
        for aar in args.aars:
            print(f"{aar.name}: {json.dumps(aar_report(aar), indent=1)}")
        return 0
    error = check_outside_repo(
        ("el set etiquetado", getattr(args, "labeled_set", None)),
        ("la salida", getattr(args, "out_dir", None)),
        ("los ONNX", getattr(args, "onnx_dir", None)),
        ("los pesos", getattr(args, "weights_dir", None)),
    )
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    try:
        if args.command == "export":
            export_models(args.weights_dir, args.onnx_dir)
        elif args.command == "eval":
            args.out_dir.mkdir(parents=True, exist_ok=True)
            print(
                report_tables(
                    evaluate_all(args.labeled_set, args.onnx_dir, args.out_dir), None, args.onnx_dir
                )
            )
        elif args.command == "latency":
            args.out_dir.mkdir(parents=True, exist_ok=True)
            latency_all(args.labeled_set, args.onnx_dir, args.out_dir)
        else:
            results = json.loads((args.out_dir / RESULTS_FILE).read_text(encoding="utf-8"))
            latency_path = args.out_dir / LATENCY_FILE
            latency = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.is_file() else None
            print(report_tables(results, latency, args.onnx_dir))
    except (OSError, ValueError, KeyError, ocr_labels.LabelsError) as exc:
        print(f"No se pudo: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
