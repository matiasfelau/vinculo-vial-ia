"""Caso de uso: orquesta la cascada de validación y devuelve un `ValidationResult`.

El veredicto es solo GREEN o RED. RED únicamente con evidencia firme: falta el auto en alguna foto (R1), otro
auto confirmado por la re-ID (R2), una patente leída con seguridad que no coincide (R3), evidencia ilegible o
redacción no disponible (R4). Todo lo dudoso o que no se puede comparar es GREEN (R5).

Todo el contenido sensible (patente, texto del OCR, embeddings, imágenes) vive solo en RAM
mientras dura `execute`. Se loguean únicamente `alert_id`, veredicto, códigos, duraciones y versiones.
"""

import base64
import hashlib
import hmac
import logging
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2

from vvi_ia.domain.model import (
    Box,
    BoxLabel,
    CarInstance,
    EvidenceImage,
    EvidenceRole,
    Pixels,
    ReasonCode,
    RedactedRef,
    Scores,
    ValidationResult,
    Vector,
    Verdict,
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
from vvi_ia.domain.redaction import RedactionThresholds, plan_redaction
from vvi_ia.domain.reid import ReidOutcome, best_match, same_vehicle
from vvi_ia.domain.verdict import aggregate
from vvi_ia.service.payload import EvidenceUploaded
from vvi_ia.service.ports import (
    DecryptionError,
    EvidenceStore,
    EvidenceTooLargeError,
    EvidenceWriter,
    ImageDecodeError,
    ImageDecoder,
    PayloadDecryptor,
    RedactionRenderer,
    SystemDecryptionError,
)

logger = logging.getLogger(__name__)

MS_PER_SECOND = 1000.0
JPEG_CONTENT_TYPE = "image/jpeg"

Codes = list[ReasonCode]


@dataclass(frozen=True)
class Models:
    """Los siete puertos de inferencia, cada uno con su `version`."""

    vehicle: VehicleDetector
    cars: CarSegmenter
    curb: CurbSegmenter  # EXPERIMENTAL: fuera de la cascada; se arma solo si MODEL_CURB no está vacío
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
            "face": self.face.version,
        }


@dataclass(frozen=True)
class Thresholds:
    """Umbrales del dominio (los de cada modelo viven en su adaptador)."""

    reid_t: float  # coseno bajo el cual la re-ID confirma que es otro auto (BURST_MISMATCH)
    ocr_min_conf: float  # lectura segura de patente: con menos confianza no hay veredicto de patente
    damage_presence_t: float  # confianza de una caja de daño para contar como presencia de auto en DETAIL
    presence_min_area: (
        float  # fracción mínima de la foto que ocupa el auto (o la parte) para contar como presente
    )
    model_max_side_px: (
        int  # lado mayor al que se reduce cada foto antes de los modelos (las cajas vuelven al original)
    )
    presence_plate_min_conf: float  # confianza mínima de una patente para contar como parte de un auto


@dataclass(frozen=True)
class RedactionSettings:
    """Parámetros de la redacción (paso 8)."""

    required: bool
    face_min_conf: float
    plate_min_conf: float
    max_side_px: int
    workers: int = 1  # fotos que se planifican a la vez (1: una por una)


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
    plate_detections: dict[tuple[EvidenceRole, int], list[Box] | None] = field(default_factory=dict)
    cars: dict[tuple[EvidenceRole, int], list[CarInstance] | None] = field(default_factory=dict)
    reads: dict[tuple[EvidenceRole, int, Box], PlateRead | None] = field(default_factory=dict)
    reference_embeddings: list[Vector] = field(default_factory=list)
    embeddings: dict[tuple[EvidenceRole, int, int], Vector | None] = field(default_factory=dict)
    general_embedding: Vector | None = None

    def detect(self, image: EvidenceImage) -> list[Box] | None:
        """El único pase del detector sobre la foto reducida (sin mosaico), con las cajas en la original."""
        key = (image.role, image.index)
        if key not in self.detections:
            self.detections[key] = to_original(self.models.vehicle.detect(image.model_pixels), image.ratio)
        return self.detections[key]

    def detect_plates(self, image: EvidenceImage) -> list[Box] | None:
        """El pase anterior más las patentes chicas del mosaico: solo para redactar."""
        key = (image.role, image.index)
        if key not in self.plate_detections:
            self.plate_detections[key] = to_original(
                self.models.vehicle.detect_plates(image.model_pixels), image.ratio
            )
        return self.plate_detections[key]

    def segment_cars(self, image: EvidenceImage) -> list[CarInstance] | None:
        """Los autos de la foto reducida (las máscaras son del tamaño de la reducida)."""
        key = (image.role, image.index)
        if key not in self.cars:
            self.cars[key] = self.models.cars.cars(image.model_pixels)
        return self.cars[key]

    def embed(self, image: EvidenceImage, car_index: int, car: CarInstance) -> Vector | None:
        """Embedding de una instancia de auto (con caché: la re-ID y la rampa comparten el trabajo)."""
        key = (image.role, image.index, car_index)
        if key not in self.embeddings:
            self.embeddings[key] = self.models.embedder.embed(image.model_pixels, car.mask)
        return self.embeddings[key]

    def read_plate(self, image: EvidenceImage) -> PlateRead | None:
        """Lee la caja de patente de mayor confianza, con recorte ajustado y sin padding."""
        box = best_plate_box(self.detect(image))
        return None if box is None else self.read_box(image, box)

    def read_box(self, image: EvidenceImage, box: Box) -> PlateRead | None:
        """Lee el recorte de una caja de patente (con caché); `None` si no hay modelo o recorte."""
        key = (image.role, image.index, box)
        if key not in self.reads:
            crop = crop_box(image, box)
            read = None if crop is None else self.models.plate.read(crop)
            self.reads[key] = None if read is None else PlateRead(text=read[0], conf=read[1])
        return self.reads[key]


def reduce_for_models(pixels: Pixels, max_side: int) -> Pixels | None:
    """La foto reducida para que su lado mayor sea `max_side`, o `None` si ya entra (no se agranda nunca)."""
    height, width = pixels.shape[:2]
    if max(height, width) <= max_side:
        return None
    scale = max_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    reduced: Pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    return reduced


def to_original(boxes: Sequence[Box] | None, ratio: float) -> list[Box] | None:
    """Las cajas de la foto reducida, en las coordenadas de la original."""
    if boxes is None or ratio == 1.0:
        return None if boxes is None else list(boxes)
    return [
        Box(b.x1 * ratio, b.y1 * ratio, b.x2 * ratio, b.y2 * ratio, conf=b.conf, label=b.label) for b in boxes
    ]


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
        writer: EvidenceWriter,
        decryptor: PayloadDecryptor,
        decoder: ImageDecoder,
        renderer: RedactionRenderer,
        models: Models,
        thresholds: Thresholds,
        redaction: RedactionSettings,
    ) -> None:
        self._store = store
        self._writer = writer
        self._decryptor = decryptor
        self._decoder = decoder
        self._renderer = renderer
        self._models = models
        self._t = thresholds
        self._redaction = redaction

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
            ordered = [general, plate_img, *by_role[EvidenceRole.DETAIL], *by_role[EvidenceRole.CONTEXT]]
            if not self._vehicle_present(run, ordered, codes):
                return self._finish(payload, codes, Scores(), durations)
        with _timed(durations, "burst"):
            similarity = self._burst(run, general, plate_img, codes)
            self._reid_others(
                run, general, by_role[EvidenceRole.DETAIL] + by_role[EvidenceRole.CONTEXT], codes
            )
        with _timed(durations, "plate"):
            ocr_conf = self._plate(run, plate_img, reported, codes)

        scores = Scores(burst_similarity=similarity, ocr_confidence=ocr_conf)
        redacted: tuple[RedactedRef, ...] = ()
        if aggregate(codes) != Verdict.RED:
            with _timed(durations, "redaction"):
                available = self._redact(run, payload, images, reported)
            if available is None:  # fail-closed: sin redacción de TODAS las fotos no hay veredicto aprobable
                _add(codes, ReasonCode.REDACTION_UNAVAILABLE)
            else:
                redacted = available
        return self._finish(payload, codes, scores, durations, redacted)

    # --- carga y descifrado -------------------------------------------------------------

    def _load_images(self, payload: EvidenceUploaded) -> list[EvidenceImage] | None:
        """Descarga, verifica sha256, descifra y decodifica. `None` si alguna foto no es legible."""
        info = payload.alert_id.encode()
        images: list[EvidenceImage] = []
        for item in payload.evidence:
            try:
                ciphertext = self._store.get(item.key)
            except EvidenceTooLargeError:
                return None
            if not hmac.compare_digest(hashlib.sha256(ciphertext).hexdigest(), item.sha256):
                return None
            try:
                pixels = self._decoder.decode(self._decryptor.open(ciphertext, info))
            except (DecryptionError, ImageDecodeError):
                return None
            view = reduce_for_models(pixels, self._t.model_max_side_px)
            images.append(EvidenceImage(role=item.role, index=item.index, pixels=pixels, view=view))
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

    def _vehicle_present(self, run: _Run, images: Sequence[EvidenceImage], codes: Codes) -> bool:
        """R1. `False` si en alguna foto no hay evidencia de un auto (RED, fraude): se corta la cascada.

        GENERAL y CONTEXT tienen que mostrar un auto; PLATE y DETAIL, un auto o una parte (rueda o patente) y,
        en DETAIL, también cuenta una caja de daño con confianza suficiente. Solo hay NO_VEHICLE si todos los
        modelos consultados respondieron y ninguno encontró nada; si no se pudo decidir, no se penaliza.
        """
        for image in images:
            if self._has_car(run, image) is False:
                _add(codes, ReasonCode.NO_VEHICLE)
                return False
        return True

    def _has_car(self, run: _Run, image: EvidenceImage) -> bool | None:
        """¿Hay un auto (o una parte, en PLATE y DETAIL) que ocupe al menos `presence_min_area` de la foto?

        Cuenta una caja de `car` del detector, una de `wheel` o `plate` si la foto es de una parte, una
        instancia de auto del segmentador o, en DETAIL, una caja de daño. `None` si no se pudo decidir (sin
        hallazgos y algún modelo sin respuesta).
        """
        part = image.role in (EvidenceRole.PLATE, EvidenceRole.DETAIL)
        labels = {BoxLabel.CAR} | ({BoxLabel.WHEEL, BoxLabel.PLATE} if part else set())
        minimum = self._t.presence_min_area * image.width * image.height
        mask_minimum = self._t.presence_min_area * image.model_pixels.shape[0] * image.model_pixels.shape[1]
        detections = run.detect(image)
        for box in detections or ():
            if box.label in labels and box.area >= minimum:
                if box.label != BoxLabel.PLATE or box.conf >= self._t.presence_plate_min_conf:
                    return True
        cars = run.segment_cars(image)
        if any(int(car.mask.sum()) >= mask_minimum for car in cars or ()):
            return True
        answered = detections is not None and cars is not None
        if image.role == EvidenceRole.DETAIL:
            damage = self._models.damage.detect(image.model_pixels)
            if any(box.conf >= self._t.damage_presence_t for box in damage or ()):
                return True
            answered = answered and damage is not None
        return False if answered else None

    def _burst(
        self, run: _Run, general: EvidenceImage, plate_img: EvidenceImage, codes: Codes
    ) -> float | None:
        """R2: ¿GENERAL y PLATE muestran el mismo auto? Devuelve la similitud si se calculó.

        Solo hay BURST_MISMATCH si la re-ID confirma que es otro auto; si no se puede comparar (falta el auto
        segmentado o el embedding), no hay código.
        """
        general_cars = run.segment_cars(general)
        plate_cars = run.segment_cars(plate_img)
        if not general_cars or not plate_cars:
            return None
        emb_general = run.embed(general, 0, largest_car(general_cars))
        emb_plate = run.embed(plate_img, 0, largest_car(plate_cars))
        run.general_embedding = emb_general
        if emb_general is None or emb_plate is None:
            return None
        run.reference_embeddings.extend((emb_general, emb_plate))
        result = same_vehicle(emb_general, emb_plate, self._t.reid_t, self._t.reid_t)
        if result.outcome == ReidOutcome.DIFFERENT:
            _add(codes, ReasonCode.BURST_MISMATCH)
        return result.similarity

    def _reid_others(
        self, run: _Run, general: EvidenceImage, others: Sequence[EvidenceImage], codes: Codes
    ) -> None:
        """R2: cada foto de DETAIL y CONTEXT contra GENERAL (el mismo auto en todas las fotos).

        De las instancias de auto de la foto cuenta la que más se parece a GENERAL. Una foto en la que no se
        segmenta ningún auto (un primer plano de la carrocería) o sin embedding no se puede comparar.
        """
        if run.general_embedding is None:
            return
        for image in others:
            candidates = [
                embedding
                for index, car in enumerate(run.segment_cars(image) or ())
                if (embedding := run.embed(image, index, car)) is not None
            ]
            match = best_match(candidates, [run.general_embedding])
            if match is None:
                continue
            outcome = same_vehicle(
                run.general_embedding, candidates[match[0]], self._t.reid_t, self._t.reid_t
            ).outcome
            if outcome == ReidOutcome.DIFFERENT:
                _add(codes, ReasonCode.BURST_MISMATCH)

    def _plate(self, run: _Run, plate_img: EvidenceImage, reported: str, codes: Codes) -> float | None:
        """R3: OCR de la patente de PLATE contra la reportada.

        Solo hay PLATE_MISMATCH con una lectura segura (confianza >= OCR_MIN_CONF) que no coincide. Una
        lectura casi igual o ilegible no penaliza.
        """
        read = run.read_plate(plate_img)
        if read is None:
            return None
        check = plate_check(reported, read.text, read.conf, self._t.ocr_min_conf)
        if check == PlateCheck.MISMATCH:
            _add(codes, ReasonCode.PLATE_MISMATCH)
        return read.conf

    # --- redacción (paso 8) --------------------------------------------------------------

    def _redact(
        self, run: _Run, payload: EvidenceUploaded, images: Sequence[EvidenceImage], reported: str
    ) -> tuple[RedactedRef, ...] | None:
        """Genera y sube las versiones redactadas de TODAS las fotos, o `None` si no se puede.

        Fail-closed: si algún modelo de rostros o de patentes no responde (`None`) o falla la reescritura de
        alguna foto, no se sube ninguna (nunca quedan redactadas parciales). Una foto sin rostros ni patentes
        (`[]`) cuenta como redactada. Un error al subir a S3 no es un veredicto: se propaga para reintento.
        Con `REDACTION_REQUIRED=false` (solo `ENV=local`) la foto re-encodeada sin tapar cuenta como lista.
        """
        plans: dict[tuple[EvidenceRole, int], list[Box]] | None
        if self._redaction.required:
            plans = self._plan_all(run, images, reported)
            if plans is None:
                return None
        else:
            logger.warning(
                "redacción desactivada: se sube la evidencia sin redactar",
                extra={"alert_id": payload.alert_id},
            )
            plans = {(image.role, image.index): [] for image in images}
        rendered: list[tuple[RedactedRef, bytes]] = []
        for image in images:
            try:
                body = self._renderer.render(
                    image.pixels, plans[(image.role, image.index)], self._redaction.max_side_px
                )
            except Exception as exc:  # noqa: BLE001 - cualquier falla de la reescritura deja la foto sin redactar
                # Solo el tipo de error: el mensaje podría arrastrar contenido de la imagen.
                logger.warning(
                    "falló la reescritura de la foto",
                    extra={"alert_id": payload.alert_id, "error": type(exc).__name__},
                )
                return None
            key = f"alerts/{payload.alert_id}/redacted/{image.role.value.lower()}-{image.index}.jpg"
            rendered.append((RedactedRef(role=image.role, index=image.index, key=key), body))
        for ref, body in rendered:  # recién con todas listas se sube: los errores de S3 se propagan
            self._writer.put(ref.key, body, JPEG_CONTENT_TYPE)
        return tuple(ref for ref, _ in rendered)

    def _plan_all(
        self, run: _Run, images: Sequence[EvidenceImage], reported: str
    ) -> dict[tuple[EvidenceRole, int], list[Box]] | None:
        """Plan de todas las fotos, o `None` si falta algún modelo (no se redacta ninguna)."""
        thresholds = RedactionThresholds(self._redaction.face_min_conf, self._redaction.plate_min_conf)

        def plan_one(image: EvidenceImage) -> list[Box] | None:
            faces = to_original(self._models.face.detect(image.model_pixels), image.ratio)
            detections = run.detect_plates(image)
            exempt = self._matching_plates(run, image, detections, reported)
            if faces is None or detections is None:
                return None
            return plan_redaction(faces, detections, thresholds, image.width, image.height, exempt)

        workers = min(self._redaction.workers, len(images))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                found = list(pool.map(plan_one, images))
        else:
            found = [plan_one(image) for image in images]
        if any(plan is None for plan in found):
            return None
        return {
            (image.role, image.index): plan
            for image, plan in zip(images, found, strict=True)
            if plan is not None
        }

    def _matching_plates(
        self, run: _Run, image: EvidenceImage, detections: Sequence[Box] | None, reported: str
    ) -> list[Box]:
        """Patentes que coinciden con la reportada (no se tapan); las que el OCR no lee se tapan."""
        exempt: list[Box] = []
        for box in detections or ():
            if box.label != BoxLabel.PLATE or box.conf < self._redaction.plate_min_conf:
                continue
            read = run.read_box(image, box)
            if read is None:
                continue  # sin lectura (o caja degenerada) no se puede saber si coincide: se tapa
            check = plate_check(reported, read.text, read.conf, self._t.ocr_min_conf)
            if check in (PlateCheck.MATCH, PlateCheck.FUZZY):
                exempt.append(box)
        return exempt

    # --- cierre -------------------------------------------------------------------------

    def _finish(
        self,
        payload: EvidenceUploaded,
        codes: Codes,
        scores: Scores,
        durations: dict[str, float],
        redacted: tuple[RedactedRef, ...] = (),
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
            redacted=redacted,
        )
