"""Latencia de la cascada completa por alerta, en CPU y con los modelos reales.

Arma alertas con fotos reales de tus datasets (autos con patente de Open Images y daños de CarDD) y corre el
caso de uso `ValidateEvidence` de punta a punta: la misma cascada y los mismos adaptadores que el servicio,
de a una alerta por vez. La foto GENERAL y la PLATE son la misma (un auto con patente visible) y la patente
reportada es la que lee el OCR, para que la cascada llegue a todas las etapas, incluida la redacción, en
lugar de cortarse en un RED.

Lo que NO mide es la red (S3) ni el descifrado HPKE: el almacenamiento y el descifrado son en memoria. Los
tiempos por etapa salen del propio log del caso de uso.

El chequeo de rebajes salió de la cascada, así que no hay nada que simular.
La consola y
`cascade_report.json` tienen solo cifras. Uso: ver `tools/eval/README.md`.
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import random
import statistics
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from detector import REPO_ROOT

from vvi_ia.adapters.imaging import CvImageDecoder, JpegRenderer
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.config import Settings
from vvi_ia.domain.model import AlertType, BoxLabel, EvidenceRole, Pixels
from vvi_ia.domain.plate import canonicalize
from vvi_ia.main import adapter_params, build_redaction, build_thresholds
from vvi_ia.service.payload import EvidenceItem, EvidenceUploaded
from vvi_ia.service.validate_evidence import ValidateEvidence

SELECTION = {
    "detector": "rfdetr_openimages",
    "face": "scrfd_10g",
    "segmenter": "rfdetr_seg_coco",
    "embedder": "dinov2_vits14",
    "ocr": "parseq",
    "damage": "rfdetr_cardd",
}
RELAXED = {"reid_t": -1.0, "damage_presence_t": 0.0}
PLATE_READ_CONF = 0.9  # la patente reportada tiene que ser la que lee el OCR con seguridad
MIN_READABLE = 12
STAGES = ("load", "vehicle", "burst", "plate", "context", "redaction")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PassthroughDecryptor:
    """Sin cifrado: el descifrado HPKE no entra en esta medición."""

    def open(self, ciphertext: bytes, info: bytes) -> bytes:
        return ciphertext


class MemoryStore:
    """Almacenamiento en memoria: lectura y escritura sin red."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def put(self, key: str, body: bytes, content_type: str) -> None:
        self.objects[key] = body


class DurationCapture(logging.Handler):
    """Se queda con `durations_ms` del log "validación completada"."""

    def __init__(self) -> None:
        super().__init__()
        self.last: dict[str, float] = {}
        self.verdict = ""

    def emit(self, record: logging.LogRecord) -> None:
        durations = getattr(record, "durations_ms", None)
        if durations is not None:
            self.last = dict(durations)
            self.verdict = str(getattr(record, "verdict", ""))


def resized_copy(path: Path, long_side: int, folder: Path) -> Path:
    """Copia de la foto con el lado mayor en `long_side` px (en `folder`), para simular fotos de celular."""
    target = folder / f"{path.parent.name}_{path.stem}_{long_side}.jpg"
    if not target.is_file():
        pixels = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        scale = long_side / max(pixels.shape[:2])
        size = (round(pixels.shape[1] * scale), round(pixels.shape[0] * scale))
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", cv2.resize(pixels, size, interpolation=cv2.INTER_CUBIC))[1].tofile(str(target))
    return target


def photos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass(frozen=True)
class Layout:
    """Fotos de una alerta por rol y la patente reportada."""

    photos: dict[EvidenceRole, list[Path]]
    plate: bytes


def build_alert(alert_type: AlertType, layout: Layout, store: MemoryStore) -> EvidenceUploaded:
    """Alerta con las fotos dadas por rol; las sube a `store` tal cual."""
    alert_id = str(uuid.uuid4())
    items = []
    for role, paths in layout.photos.items():
        for index, path in enumerate(paths):
            data = path.read_bytes()
            key = f"alerts/{alert_id}/original/{role.lower()}-{index}.bin"
            store.objects[key] = data
            items.append(
                EvidenceItem(role=role, index=index, key=key, sha256=hashlib.sha256(data).hexdigest())
            )
    return EvidenceUploaded(
        alert_id=alert_id,
        alert_type=alert_type,
        pat_enc_ia=base64.b64encode(layout.plate).decode(),
        bucket="bench",
        evidence=tuple(items),
    )


def summarize(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "mediana": round(statistics.median(ordered)),
        "media": round(statistics.fmean(ordered)),
        "p95": round(p95),
        "maximo": round(ordered[-1]),
    }


def has_part(models: Any, image: Pixels, min_area: float) -> bool:
    """Lo mismo que la presencia del servicio para PLATE y DETAIL: un auto, una rueda o una patente."""
    minimum = min_area * image.shape[0] * image.shape[1]
    labels = {BoxLabel.CAR, BoxLabel.WHEEL, BoxLabel.PLATE}
    boxes = models.vehicle.detect(image) or ()
    if any(
        b.label in labels and b.area >= minimum and (b.label != BoxLabel.PLATE or b.conf >= 0.4)
        for b in boxes
    ):
        return True
    return any(int(car.mask.sum()) >= minimum for car in models.cars.cars(image) or ())


def readable_plates(
    models: Any,
    pool: Sequence[Path],
    rng: random.Random,
    wanted: int,
    prepare: Callable[[Path], Path] = lambda path: path,
) -> list[tuple[Path, bytes]]:
    """Fotos de autos cuya patente lee el OCR, con la lectura como patente reportada (sin medir)."""
    decoder = CvImageDecoder()
    found: list[tuple[Path, bytes]] = []
    for original in rng.sample(list(pool), len(pool)):
        path = prepare(original)
        pixels = decoder.decode(path.read_bytes())
        boxes = models.vehicle.detect(pixels) or ()
        plates = [b for b in boxes if b.label == BoxLabel.PLATE]
        if not plates or not any(b.label == BoxLabel.CAR for b in boxes):
            continue
        box = max(plates, key=lambda b: b.conf)
        crop = pixels[max(0, int(box.y1)) : int(box.y2), max(0, int(box.x1)) : int(box.x2)]
        read = models.plate.read(crop) if crop.size else None
        text = canonicalize(read[0]) if read else ""
        if text and read and read[1] >= PLATE_READ_CONF:
            found.append((path, text.encode()))
        if len(found) >= max(wanted, MIN_READABLE):
            break
    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TRAIN_DATA_DIR"), help="TRAIN_DATA_DIR"
    )
    parser.add_argument(
        "--eval-dir", type=Path, default=os.environ.get("EVAL_DATA_DIR"), help="EVAL_DATA_DIR"
    )
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("EVAL_OUT"), help="salida (EVAL_OUT)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("MODELS_DIR"), help="MODELS_DIR")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "models" / "manifest.json")
    parser.add_argument("--alerts", type=int, default=20, help="alertas por escenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="hilos de ONNX Runtime (0: los que elija ORT)")
    parser.add_argument("--no-redaction", action="store_true", help="REDACTION_REQUIRED=false (solo local)")
    parser.add_argument(
        "--plate-tiling",
        choices=("on", "off"),
        default="on",
        help="PLATE_TILING (por defecto on, como el servicio)",
    )
    parser.add_argument(
        "--real-thresholds",
        action="store_true",
        help="usar los umbrales reales (con fotos de prueba casi todas terminan en RED y saltan etapas)",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        help="reescala las fotos a este lado mayor en px (p. ej. 4000 para fotos de celular de 4000 x 3000)",
    )
    parser.add_argument("--general-dir", type=Path, help="autos (por defecto <data-dir>/damage-fp/images)")
    parser.add_argument("--detail-dir", type=Path, help="daños (por defecto <data-dir>/cardd-detector/test)")
    args = parser.parse_args(argv)
    for flag, value in (("--out-dir", args.out_dir), ("--models-dir", args.models_dir)):
        if value is None:
            parser.error(f"indicá {flag} o definí su variable de entorno")
    if args.data_dir is None and (args.general_dir is None or args.detail_dir is None):
        parser.error("indicá --data-dir (TRAIN_DATA_DIR) o --general-dir y --detail-dir")
    args.general_dir = args.general_dir or Path(args.data_dir) / "damage-fp" / "images"
    args.detail_dir = args.detail_dir or Path(args.data_dir) / "cardd-detector" / "test"
    return args


def run_scenario(
    use_case: ValidateEvidence,
    store: MemoryStore,
    capture: DurationCapture,
    alert_type: AlertType,
    layouts: list[Layout],
) -> dict[str, Any]:
    """Una alerta de calentamiento y luego las medidas; devuelve el total y el tiempo por etapa."""
    use_case.execute(build_alert(alert_type, layouts[0], store))
    totals: list[float] = []
    stages: dict[str, list[float]] = {stage: [] for stage in STAGES}
    redacted = 0
    verdicts: Counter[str] = Counter()
    for layout in layouts:
        payload = build_alert(alert_type, layout, store)
        start = time.perf_counter()
        use_case.execute(payload)
        totals.append((time.perf_counter() - start) * 1000)
        for stage in STAGES:
            stages[stage].append(capture.last.get(stage, 0.0))
        redacted += "redaction" in capture.last
        verdicts[capture.verdict] += 1
        store.objects.clear()
    return {
        "alertas": len(totals),
        "con_redaccion": redacted,
        "veredictos": dict(verdicts),
        "ms_por_alerta": summarize(totals),
        "ms_medianos_por_etapa": {s: round(statistics.median(v)) for s, v in stages.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if Path(args.out_dir).resolve().is_relative_to(REPO_ROOT):
        print("La salida no puede estar dentro del repositorio.", file=sys.stderr)
        return 1
    folders = {"autos": args.general_dir, "daños": args.detail_dir}
    pools = {name: photos_in(Path(folder)) for name, folder in folders.items()}
    for name, pool in pools.items():
        if not pool or Path(folders[name]).resolve().is_relative_to(REPO_ROOT):
            print(f"No hay fotos de {name} en {folders[name]} (o está dentro del repo).", file=sys.stderr)
            return 1
    # Salvo con --real-thresholds, los umbrales se aflojan para que la cascada recorra todas las etapas
    # (incluida la redacción): se mide el camino completo, no los veredictos.
    relaxed = {} if args.real_thresholds else RELAXED
    settings = Settings.model_construct(
        redaction_required=not args.no_redaction, plate_tiling=args.plate_tiling == "on", **relaxed
    )
    selection = dict(SELECTION)
    try:
        models = build_models(
            selection, args.manifest, args.models_dir, args.threads, adapter_params(settings)
        )
    except ModelSetupError as exc:
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
    capture = DurationCapture()
    decoder = CvImageDecoder()
    logging.getLogger("vvi_ia.service.validate_evidence").addHandler(capture)
    logging.getLogger("vvi_ia.service.validate_evidence").setLevel(logging.INFO)
    rng = random.Random(args.seed)  # noqa: S311 - muestreo de fotos, no criptografía
    photo_folder = Path(args.out_dir) / "cascade-photos"

    def prepare(path: Path) -> Path:
        return resized_copy(path, args.long_side, photo_folder) if args.long_side else path

    readable = readable_plates(models, pools["autos"], rng, args.alerts, prepare)

    present: dict[Path, bool] = {}

    def shows_a_part(path: Path) -> bool:
        """La foto de detalle pasa la presencia de auto (si no, la cascada se cortaría antes de tiempo)."""
        if path not in present:
            present[path] = has_part(models, decoder.decode(path.read_bytes()), settings.presence_min_area)
        return present[path]

    def pick_detail() -> Path:
        while True:
            path = prepare(rng.choice(pools["daños"]))
            if shows_a_part(path):
                return path

    def layouts(kind: str, details: int = 1) -> list[Layout]:
        out = []
        for _ in range(args.alerts):
            car, plate = rng.choice(readable)
            photos = {EvidenceRole.GENERAL: [car], EvidenceRole.PLATE: [car]}
            if kind == "crashed":
                photos[EvidenceRole.DETAIL] = [pick_detail() for _ in range(details)]
            else:
                photos[EvidenceRole.CONTEXT] = [car]  # el mismo auto: así la re-ID no corta la cascada
            out.append(Layout(photos, plate))
        return out

    scenarios = {
        "CRASHED de 3 fotos (general, patente, 1 detalle)": (AlertType.CRASHED, layouts("crashed", 1)),
        "CRASHED de 5 fotos (general, patente, 3 detalles)": (AlertType.CRASHED, layouts("crashed", 3)),
        "PARKED_ON_GARAGE de 3 fotos (general, patente, contexto)": (
            AlertType.PARKED_ON_GARAGE,
            layouts("garage"),
        ),
    }
    report: dict[str, Any] = {
        "modelos": models.versions(),
        "hilos_onnx": args.threads,
        "redaccion": not args.no_redaction,
        "mosaico_patentes": args.plate_tiling == "on",
        "lado_mayor_px": args.long_side,
        "escenarios": {},
    }
    for name, (alert_type, alert_layouts) in scenarios.items():
        report["escenarios"][name] = run_scenario(use_case, store, capture, alert_type, alert_layouts)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "cascade_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modelos: {report['modelos']}")
    print(
        f"Hilos ONNX: {args.threads or 'por defecto'} · redacción: {report['redaccion']} · "
        f"mosaico de patentes: {report['mosaico_patentes']} · lado mayor: {args.long_side or 'original'} · "
    )
    header = "".join(f"{s:>10}" for s in STAGES)
    print(f"{'escenario':<60}{'mediana':>9}{'p95':>8}{'máx':>8}   etapas medianas (ms): {header}")
    for name, row in report["escenarios"].items():
        stages = "".join(f"{row['ms_medianos_por_etapa'][s]:>10}" for s in STAGES)
        total = row["ms_por_alerta"]
        print(f"{name:<60}{total['mediana']:>9}{total['p95']:>8}{total['maximo']:>8}   {' ' * 26}{stages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
