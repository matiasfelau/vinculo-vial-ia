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
from vvi_ia.domain.ramp import driveway_overlap, ramp_reason, wheel_diameter
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
    damage_t_low: float  # confianza de daño bajo la cual el daño no está (RED)
    damage_t_high: float  # confianza desde la cual el daño está con claridad
    presence_min_area: (
        float  # fracción mínima de la foto que ocupa el auto (o la parte) para contar como presente
    )
    presence_plate_min_conf: float  # confianza mínima de una patente para contar como parte de un auto
    ramp_t_low: float  # solape (en escalas de rueda) debajo del cual el auto no bloquea: RED
    ramp_t_high: float  # solape desde el cual el auto bloquea con claridad: sin observación
    ramp_scale_fallback: float


@dataclass(frozen=True)
class RedactionSettings:
    """Parámetros de la redacción (paso 8)."""

    required: bool
    face_min_conf: float
    plate_min_conf: float
    max_side_px: int


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
    reads: dict[tuple[EvidenceRole, int, Box], PlateRead | None] = field(default_factory=dict)
    reference_embeddings: list[Vector] = field(default_factory=list)
    embeddings: dict[tuple[EvidenceRole, int, int], Vector | None] = field(default_factory=dict)
    general_embedding: Vector | None = None

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

    def embed(self, image: EvidenceImage, car_index: int, car: CarInstance) -> Vector | None:
        """Embedding de una instancia de auto (con caché: la re-ID y la rampa comparten el trabajo)."""
        key = (image.role, image.index, car_index)
        if key not in self.embeddings:
            self.embeddings[key] = self.models.embedder.embed(image.pixels, car.mask)
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
        with _timed(durations, "context"):
            context_conf = self._context(run, payload.alert_type, by_role, codes)

        scores = Scores(burst_similarity=similarity, ocr_confidence=ocr_conf, context_confidence=context_conf)
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

    def _vehicle_present(self, run: _Run, images: Sequence[EvidenceImage], codes: Codes) -> bool:
        """E1. `False` si falta el auto en alguna foto (RED): el resto de la cascada no tiene sentido.

        GENERAL y CONTEXT tienen que mostrar un auto; PLATE y DETAIL, un auto o una parte (rueda o patente).
        Si los modelos no responden no se puede decidir y la foto suma `BURST_UNCERTAIN`.
        """
        for image in images:
            found = self._has_car(run, image)
            if found is None:
                _add(codes, ReasonCode.BURST_UNCERTAIN)
            elif not found:
                _add(codes, ReasonCode.NO_VEHICLE)
                return False
        return True

    def _has_car(self, run: _Run, image: EvidenceImage) -> bool | None:
        """¿Hay un auto (o una parte, en PLATE y DETAIL) que ocupe al menos `presence_min_area` de la foto?

        Cuenta una caja de `car` del detector, una de `wheel` o `plate` si la foto es de una parte, o una
        instancia de auto del segmentador. `None` si no se pudo decidir (sin hallazgos y sin un modelo).
        """
        labels = {BoxLabel.CAR}
        if image.role in (EvidenceRole.PLATE, EvidenceRole.DETAIL):
            labels |= {BoxLabel.WHEEL, BoxLabel.PLATE}
        minimum = self._t.presence_min_area * image.width * image.height
        detections = run.detect(image)
        for box in detections or ():
            if box.label in labels and box.area >= minimum:
                if box.label != BoxLabel.PLATE or box.conf >= self._t.presence_plate_min_conf:
                    return True
        cars = run.segment_cars(image)
        if any(int(car.mask.sum()) >= minimum for car in cars or ()):
            return True
        return False if detections is not None and cars is not None else None

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
        emb_general = run.embed(general, 0, largest_car(general_cars))
        emb_plate = run.embed(plate_img, 0, largest_car(plate_cars))
        run.general_embedding = emb_general
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

    def _reid_others(
        self, run: _Run, general: EvidenceImage, others: Sequence[EvidenceImage], codes: Codes
    ) -> None:
        """E1c: cada foto de DETAIL y CONTEXT contra GENERAL (el mismo auto en todas las fotos).

        De las instancias de auto de la foto cuenta la que más se parece a GENERAL. Una foto en la que no se
        segmenta ningún auto (un primer plano de la carrocería) no se puede comparar y no suma ningún código.
        """
        if run.general_embedding is None:
            return
        for image in others:
            cars = run.segment_cars(image)
            if cars is None:
                _add(codes, ReasonCode.BURST_UNCERTAIN)
                continue
            candidates = [run.embed(image, index, car) for index, car in enumerate(cars)]
            if any(candidate is None for candidate in candidates):
                _add(codes, ReasonCode.BURST_UNCERTAIN)
                continue
            match = best_match([c for c in candidates if c is not None], [run.general_embedding])
            if match is None:
                continue
            outcome = same_vehicle(
                run.general_embedding,
                [c for c in candidates if c is not None][match[0]],
                self._t.reid_t_low,
                self._t.reid_t_high,
            ).outcome
            if outcome == ReidOutcome.DIFFERENT:
                _add(codes, ReasonCode.BURST_MISMATCH)
            elif outcome == ReidOutcome.UNCERTAIN:
                _add(codes, ReasonCode.BURST_UNCERTAIN)

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
                if box.conf >= self._t.damage_t_low and (best is None or box.conf > best):
                    best = box.conf
        if best is None:
            _add(codes, ReasonCode.CONTEXT_UNCERTAIN if any_unavailable else ReasonCode.CONTEXT_NOT_DETECTED)
        elif best < self._t.damage_t_high:
            _add(codes, ReasonCode.CONTEXT_UNCERTAIN)  # entre los dos umbrales: dudoso
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
        measure = driveway_overlap(
            curb[0],
            curb[1],
            target.mask,
            wheel_diam=wheel_diameter(wheels, target.mask),
            scale_fallback=self._t.ramp_scale_fallback,
        )
        reason = ramp_reason(measure, self._t.ramp_t_low, self._t.ramp_t_high)
        if reason is not None:
            _add(codes, reason)
        return similarity

    def _target_car(
        self, run: _Run, context: EvidenceImage, cars: Sequence[CarInstance]
    ) -> tuple[CarInstance | None, float | None]:
        """Auto objetivo: la instancia con mayor coseno contra el embedding de PLATE o GENERAL."""
        if run.reference_embeddings:
            embeddings = [run.embed(context, index, car) for index, car in enumerate(cars)]
            if all(embedding is not None for embedding in embeddings):
                match = best_match([e for e in embeddings if e is not None], run.reference_embeddings)
                if match is None:
                    return None, None
                index, similarity = match
                return (cars[index] if similarity >= self._t.reid_t_low else None), similarity
        if len(cars) == 1:
            return cars[0], None
        return None, None

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
        plans: dict[tuple[EvidenceRole, int], list[Box]] = {}
        for image in images:
            faces = self._models.face.detect(image.pixels)
            detections = run.detect(image)
            exempt = self._matching_plates(run, image, detections, reported)
            if faces is None or detections is None:
                return None
            plan = plan_redaction(faces, detections, thresholds, image.width, image.height, exempt)
            if plan is None:
                return None
            plans[(image.role, image.index)] = plan
        return plans

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


def _add_all(codes: Codes, new: Sequence[ReasonCode]) -> None:
    for code in new:
        _add(codes, code)
