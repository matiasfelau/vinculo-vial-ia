"""Caso de uso: orquesta la cascada de validación y devuelve un `ValidationResult`.

Todo el contenido sensible (patente, texto del OCR, embeddings, imágenes) vive solo en RAM
mientras dura `execute`. Se loguean únicamente `alert_id`, veredicto, códigos, duraciones y versiones.
"""

import base64
import hashlib
import hmac
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from vvi_ia.domain.model import (
    AlertType,
    Box,
    BoxLabel,
    CarInstance,
    EvidenceImage,
    EvidenceRole,
    Pixels,
    ReasonCode,
    Scores,
    ValidationResult,
    Vector,
)
from vvi_ia.domain.plate import PlateCheck, canonicalize, plate_check
from vvi_ia.domain.ports import (
    CarSegmenter,
    CurbSegmenter,
    DamageDetector,
    Embedder,
    FaceDetector,
    PlateReader,
    VehicleDetector,
)
from vvi_ia.domain.ramp import DrivewayCheck, driveway_check, wheel_diameter
from vvi_ia.domain.reid import ReidOutcome, best_match, same_vehicle
from vvi_ia.domain.verdict import aggregate
from vvi_ia.service.payload import EvidenceUploaded
from vvi_ia.service.ports import (
    DecryptionError,
    EvidenceStore,
    ImageDecodeError,
    ImageDecoder,
    PayloadDecryptor,
    SystemDecryptionError,
)

logger = logging.getLogger(__name__)

MS_PER_SECOND = 1000.0

Codes = list[ReasonCode]


@dataclass(frozen=True)
class Models:
    """Los siete puertos de inferencia, cada uno con su `version`."""

    vehicle: VehicleDetector
    cars: CarSegmenter
    curb: CurbSegmenter
    embedder: Embedder
    plate: PlateReader
    damage: DamageDetector
    face: FaceDetector

    def versions(self) -> dict[str, str]:
        """Versiones tal como viajan en `model_versions`."""
        return {
            "detector": self.vehicle.version,
            "segmenter": self.cars.version,
            "embedder": self.embedder.version,
            "ocr": self.plate.version,
            "damage": self.damage.version,
            "curb": self.curb.version,
            "face": self.face.version,
        }


@dataclass(frozen=True)
class Thresholds:
    """Umbrales del dominio (los de cada modelo viven en su adaptador)."""

    reid_t_low: float
    reid_t_high: float
    ocr_min_conf: float
    damage_min_conf: float
    ramp_margin: float
    ramp_scale_fallback: float


@dataclass(frozen=True)
class PlateRead:
    """Lectura de patente; solo en RAM."""

    text: str
    conf: float


@dataclass
class _Run:
    """Estado de una ejecución: cachés por foto para no repetir inferencias."""

    models: Models
    detections: dict[tuple[EvidenceRole, int], list[Box] | None] = field(default_factory=dict)
    cars: dict[tuple[EvidenceRole, int], list[CarInstance] | None] = field(default_factory=dict)
    reads: dict[tuple[EvidenceRole, int], PlateRead | None] = field(default_factory=dict)
    reference_embeddings: list[Vector] = field(default_factory=list)

    def detect(self, image: EvidenceImage) -> list[Box] | None:
        key = (image.role, image.index)
        if key not in self.detections:
            self.detections[key] = self.models.vehicle.detect(image.pixels)
        return self.detections[key]

    def segment_cars(self, image: EvidenceImage) -> list[CarInstance] | None:
        key = (image.role, image.index)
        if key not in self.cars:
            self.cars[key] = self.models.cars.cars(image.pixels)
        return self.cars[key]

    def read_plate(self, image: EvidenceImage) -> PlateRead | None:
        """Lee la caja de patente de mayor confianza, con recorte ajustado y sin padding."""
        key = (image.role, image.index)
        if key not in self.reads:
            self.reads[key] = self._read_best_plate(image)
        return self.reads[key]

    def _read_best_plate(self, image: EvidenceImage) -> PlateRead | None:
        box = best_plate_box(self.detect(image))
        if box is None:
            return None
        crop = crop_box(image, box)
        if crop is None:
            return None
        read = self.models.plate.read(crop)
        return None if read is None else PlateRead(text=read[0], conf=read[1])


def best_plate_box(detections: Sequence[Box] | None) -> Box | None:
    """Caja de patente de mayor confianza."""
    plates = [box for box in detections or () if box.label == BoxLabel.PLATE]
    return max(plates, key=lambda box: box.conf, default=None)


def crop_box(image: EvidenceImage, box: Box) -> Pixels | None:
    """Recorte de la caja sin padding, acotado a la imagen; `None` si queda vacío."""
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(image.width, int(round(box.x2))), min(image.height, int(round(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.pixels[y1:y2, x1:x2]


def largest_car(cars: Sequence[CarInstance]) -> CarInstance:
    """Auto de mayor área: el sujeto de la foto."""
    return max(cars, key=lambda car: int(car.mask.sum()))


@contextmanager
def _timed(durations: dict[str, float], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        durations[stage] = round((time.perf_counter() - start) * MS_PER_SECOND, 1)


def _add(codes: Codes, code: ReasonCode) -> None:
    if code not in codes:
        codes.append(code)


class ValidateEvidence:
    """Cascada de validación de evidencia fotográfica (síncrona; corre en el pool de workers)."""

    def __init__(
        self,
        store: EvidenceStore,
        decryptor: PayloadDecryptor,
        decoder: ImageDecoder,
        models: Models,
        thresholds: Thresholds,
    ) -> None:
        self._store = store
        self._decryptor = decryptor
        self._decoder = decoder
        self._models = models
        self._t = thresholds

    def execute(self, payload: EvidenceUploaded) -> ValidationResult:
        """Valida la evidencia.

        Lanza `TransientError` ante fallas recuperables, y `ObjectNotFoundError` o
        `SystemDecryptionError` ante errores del sistema; el llamador decide reintento o DLQ.
        """
        durations: dict[str, float] = {}
        with _timed(durations, "load"):
            images = self._load_images(payload)
        if images is None:
            return self._finish(payload, [ReasonCode.EVIDENCE_UNREADABLE], Scores(), durations)

        with _timed(durations, "plate_secret"):
            reported = self._reported_plate(payload)

        run = _Run(models=self._models)
        by_role = {role: [i for i in images if i.role == role] for role in EvidenceRole}
        general, plate_img = by_role[EvidenceRole.GENERAL][0], by_role[EvidenceRole.PLATE][0]
        codes: Codes = []

        with _timed(durations, "vehicle"):
            if not self._vehicle_present(run, general, codes):
                return self._finish(payload, codes, Scores(), durations)
        with _timed(durations, "burst"):
            similarity = self._burst(run, general, plate_img, codes)
        with _timed(durations, "plate"):
            ocr_conf = self._plate(run, plate_img, reported, codes)
        with _timed(durations, "context"):
            context_conf = self._context(run, payload.alert_type, by_role, codes)

        scores = Scores(burst_similarity=similarity, ocr_confidence=ocr_conf, context_confidence=context_conf)
        return self._finish(payload, codes, scores, durations)

    # --- carga y descifrado -------------------------------------------------------------

    def _load_images(self, payload: EvidenceUploaded) -> list[EvidenceImage] | None:
        """Descarga, verifica sha256, descifra y decodifica. `None` si alguna foto no es legible."""
        info = payload.alert_id.encode()
        images: list[EvidenceImage] = []
        for item in payload.evidence:
            ciphertext = self._store.get(item.key)
            if not hmac.compare_digest(hashlib.sha256(ciphertext).hexdigest(), item.sha256):
                return None
            try:
                pixels = self._decoder.decode(self._decryptor.open(ciphertext, info))
            except (DecryptionError, ImageDecodeError):
                return None
            images.append(EvidenceImage(role=item.role, index=item.index, pixels=pixels))
        return images

    def _reported_plate(self, payload: EvidenceUploaded) -> str:
        try:
            raw = self._decryptor.open(base64.b64decode(payload.pat_enc_ia), payload.alert_id.encode())
            plate = canonicalize(raw.decode("utf-8"))
        except (DecryptionError, UnicodeDecodeError):
            raise SystemDecryptionError("pat_enc_ia no se pudo descifrar") from None
        if not plate:
            raise SystemDecryptionError("pat_enc_ia no contiene una patente")
        return plate

    # --- etapas -------------------------------------------------------------------------

    def _vehicle_present(self, run: _Run, general: EvidenceImage, codes: Codes) -> bool:
        """E1. `False` si no hay auto (RED): el resto de la cascada no tiene sentido."""
        detections = run.detect(general)
        if detections is None:
            _add(codes, ReasonCode.BURST_UNCERTAIN)
            return True
        if not any(box.label == BoxLabel.CAR for box in detections):
            _add(codes, ReasonCode.NO_VEHICLE)
            return False
        return True

    def _burst(
        self, run: _Run, general: EvidenceImage, plate_img: EvidenceImage, codes: Codes
    ) -> float | None:
        """E1b: ¿GENERAL y PLATE muestran el mismo auto? Devuelve la similitud si se calculó."""
        general_cars = run.segment_cars(general)
        plate_cars = run.segment_cars(plate_img)
        if not general_cars or plate_cars is None:
            _add(codes, ReasonCode.BURST_UNCERTAIN)
            return None
        if not plate_cars:
            _add_all(codes, self._burst_by_ocr(run, general, plate_img))
            return None
        emb_general = self._models.embedder.embed(general.pixels, largest_car(general_cars).mask)
        emb_plate = self._models.embedder.embed(plate_img.pixels, largest_car(plate_cars).mask)
        if emb_general is None or emb_plate is None:
            _add(codes, ReasonCode.BURST_UNCERTAIN)
            return None
        run.reference_embeddings.extend((emb_general, emb_plate))
        result = same_vehicle(emb_general, emb_plate, self._t.reid_t_low, self._t.reid_t_high)
        if result.outcome == ReidOutcome.UNCERTAIN:
            _add(codes, ReasonCode.BURST_UNCERTAIN)
        elif result.outcome == ReidOutcome.DIFFERENT:
            _add(codes, ReasonCode.BURST_MISMATCH)
        return result.similarity

    def _burst_by_ocr(self, run: _Run, general: EvidenceImage, plate_img: EvidenceImage) -> Codes:
        """PLATE sin auto detectable: se cruzan las lecturas de patente de ambas fotos."""
        first, second = run.read_plate(general), run.read_plate(plate_img)
        if first is None or second is None:
            return [ReasonCode.BURST_UNCERTAIN]
        text = canonicalize(first.text)
        if not text or first.conf < self._t.ocr_min_conf:
            return [ReasonCode.BURST_UNCERTAIN]
        check = plate_check(text, second.text, second.conf, self._t.ocr_min_conf)
        if check == PlateCheck.UNREADABLE:
            return [ReasonCode.BURST_UNCERTAIN]
        return [ReasonCode.BURST_MISMATCH] if check == PlateCheck.MISMATCH else []

    def _plate(self, run: _Run, plate_img: EvidenceImage, reported: str, codes: Codes) -> float | None:
        """E2: OCR de la patente de PLATE contra la patente reportada."""
        read = run.read_plate(plate_img)
        if read is None:
            _add(codes, ReasonCode.PLATE_UNREADABLE)
            return None
        check = plate_check(reported, read.text, read.conf, self._t.ocr_min_conf)
        if check == PlateCheck.FUZZY:
            _add(codes, ReasonCode.PLATE_FUZZY)
        elif check == PlateCheck.MISMATCH:
            _add(codes, ReasonCode.PLATE_MISMATCH)
        elif check == PlateCheck.UNREADABLE:
            _add(codes, ReasonCode.PLATE_UNREADABLE)
        return read.conf

    def _context(
        self,
        run: _Run,
        alert_type: AlertType,
        by_role: dict[EvidenceRole, list[EvidenceImage]],
        codes: Codes,
    ) -> float | None:
        """E3: el daño (CRASHED) o el auto frente al rebaje (PARKED_ON_GARAGE)."""
        if alert_type == AlertType.CRASHED:
            return self._damage(by_role[EvidenceRole.DETAIL], codes)
        return self._driveway(run, by_role[EvidenceRole.CONTEXT][0], codes)

    def _damage(self, details: Sequence[EvidenceImage], codes: Codes) -> float | None:
        best: float | None = None
        any_unavailable = False
        for image in details:
            boxes = self._models.damage.detect(image.pixels)
            if boxes is None:
                any_unavailable = True
                continue
            for box in boxes:
                if box.conf >= self._t.damage_min_conf and (best is None or box.conf > best):
                    best = box.conf
        if best is None:
            _add(codes, ReasonCode.CONTEXT_UNCERTAIN if any_unavailable else ReasonCode.CONTEXT_NOT_DETECTED)
        return best

    def _driveway(self, run: _Run, context: EvidenceImage, codes: Codes) -> float | None:
        curb = self._models.curb.curb(context.pixels)
        cars = run.segment_cars(context)
        if curb is None or not cars:
            _add(codes, ReasonCode.CONTEXT_UNCERTAIN)
            return None
        target, similarity = self._target_car(run, context, cars)
        if target is None:
            _add(codes, ReasonCode.CONTEXT_UNCERTAIN)
            return similarity
        wheels = [box for box in run.detect(context) or () if box.label == BoxLabel.WHEEL]
        check = driveway_check(
            curb[0],
            curb[1],
            target.mask,
            wheel_diam=wheel_diameter(wheels, target.mask),
            margin=self._t.ramp_margin,
            scale_fallback=self._t.ramp_scale_fallback,
        )
        if check == DrivewayCheck.BLOCKING_MARGINAL or check == DrivewayCheck.NO_CAR:
            _add(codes, ReasonCode.CONTEXT_UNCERTAIN)
        elif check in (DrivewayCheck.NOT_BLOCKING, DrivewayCheck.NO_DRIVEWAY):
            _add(codes, ReasonCode.CONTEXT_NOT_DETECTED)
        return similarity

    def _target_car(
        self, run: _Run, context: EvidenceImage, cars: Sequence[CarInstance]
    ) -> tuple[CarInstance | None, float | None]:
        """Auto objetivo: la instancia con mayor coseno contra el embedding de PLATE o GENERAL."""
        if run.reference_embeddings:
            embeddings = [self._models.embedder.embed(context.pixels, car.mask) for car in cars]
            if all(embedding is not None for embedding in embeddings):
                match = best_match([e for e in embeddings if e is not None], run.reference_embeddings)
                if match is None:
                    return None, None
                index, similarity = match
                return (cars[index] if similarity >= self._t.reid_t_low else None), similarity
        if len(cars) == 1:
            return cars[0], None
        return None, None

    # --- cierre -------------------------------------------------------------------------

    def _finish(
        self, payload: EvidenceUploaded, codes: Codes, scores: Scores, durations: dict[str, float]
    ) -> ValidationResult:
        verdict = aggregate(codes)
        versions = self._models.versions()
        logger.info(
            "validación completada",
            extra={
                "alert_id": payload.alert_id,
                "verdict": verdict.value,
                "reason_codes": [code.value for code in codes],
                "durations_ms": durations,
                "model_versions": versions,
            },
        )
        return ValidationResult(
            verdict=verdict,
            reason_codes=tuple(codes),
            scores=scores,
            model_versions=versions,
        )


def _add_all(codes: Codes, new: Sequence[ReasonCode]) -> None:
    for code in new:
        _add(codes, code)
