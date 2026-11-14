"""Fakes de los puertos para probar la cascada sin modelos reales.

Cada foto sintética es un arreglo constante cuyo valor (marcador) la identifica; los fakes
responden según ese marcador. No hay patentes ni imágenes reales en los fixtures.
"""

import base64
import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from vvi_ia.domain.model import (
    AlertType,
    Box,
    BoxLabel,
    CarInstance,
    EvidenceRole,
    Mask,
    Pixels,
    Vector,
)
from vvi_ia.service.payload import EvidenceItem, EvidenceUploaded
from vvi_ia.service.ports import DecryptionError, ImageDecodeError, ObjectNotFoundError
from vvi_ia.service.validate_evidence import (
    Models,
    RedactionSettings,
    Thresholds,
)

SIZE = 100
REPORTED_PLATE = "TST123"
BUCKET = "bucket-de-prueba"

GENERAL, PLATE, CONTEXT = 10, 20, 40


def detail_marker(index: int) -> int:
    return 30 + index


def marker_of(img: Pixels) -> int:
    return int(img[0, 0, 0])


BRIGHT = 200


def photo(marker: int) -> Pixels:
    """Foto sintética: el canal R lleva el marcador y G/B son brillantes (el relleno opaco los apaga)."""
    img = np.full((SIZE, SIZE, 3), BRIGHT, np.uint8)
    img[..., 0] = marker
    return img


def rect(r1: int, r2: int, c1: int, c2: int) -> Mask:
    mask = np.zeros((SIZE, SIZE), bool)
    mask[r1:r2, c1:c2] = True
    return mask


def vec(angle_deg: float) -> Vector:
    rad = np.deg2rad(angle_deg)
    return np.array([np.cos(rad), np.sin(rad)])


def car_box() -> Box:
    return Box(5, 5, 95, 95, 0.9, BoxLabel.CAR)


def plate_box(conf: float = 0.9) -> Box:
    return Box(10, 10, 60, 30, conf, BoxLabel.PLATE)


THRESHOLDS = Thresholds(
    reid_t=0.50,
    ocr_min_conf=0.60,
    presence_min_area=0.05,
    model_max_side_px=1024,
)


# --- modelos ---------------------------------------------------------------------------


class FakeVehicleDetector:
    version = "fake-detector"

    def __init__(self, by_marker: dict[int, list[Box] | None]) -> None:
        self.by_marker = by_marker
        self.calls = 0
        self.plate_calls = 0

    def detect(self, img: Pixels) -> list[Box] | None:
        self.calls += 1
        return self.by_marker.get(marker_of(img))

    def detect_plates(self, img: Pixels) -> list[Box] | None:
        self.plate_calls += 1
        return self.by_marker.get(marker_of(img))


class FakeCarSegmenter:
    version = "fake-segmenter"

    def __init__(self, by_marker: dict[int, list[CarInstance] | None]) -> None:
        self.by_marker = by_marker

    def cars(self, img: Pixels) -> list[CarInstance] | None:
        return self.by_marker.get(marker_of(img))


class FakeCurbSegmenter:
    version = "fake-curb"

    def __init__(self, by_marker: dict[int, tuple[Mask, Mask] | None]) -> None:
        self.by_marker = by_marker

    def curb(self, img: Pixels) -> tuple[Mask, Mask] | None:
        return self.by_marker.get(marker_of(img))


class FakeEmbedder:
    version = "fake-embedder"

    def __init__(self) -> None:
        self.by_marker_area: dict[tuple[int, int], Vector | None] = {}
        self.by_marker: dict[int, Vector | None] = {}

    def embed(self, img: Pixels, mask: Mask) -> Vector | None:
        marker, area = marker_of(img), int(mask.sum())
        if (marker, area) in self.by_marker_area:
            return self.by_marker_area[(marker, area)]
        return self.by_marker.get(marker)


class FakePlateReader:
    version = "fake-ocr"

    def __init__(self, by_marker: dict[int, tuple[str, float] | None]) -> None:
        self.by_marker = by_marker
        self.override: Callable[[Pixels], tuple[str, float] | None] | None = None
        self.calls = 0

    def read(self, crop: Pixels) -> tuple[str, float] | None:
        self.calls += 1
        if self.override is not None:
            return self.override(crop)
        return self.by_marker.get(marker_of(crop))


class FakeDamageDetector:
    version = "fake-damage"

    def __init__(self, by_marker: dict[int, list[Box] | None]) -> None:
        self.by_marker = by_marker

    def detect(self, img: Pixels) -> list[Box] | None:
        return self.by_marker.get(marker_of(img))


class FakeFaceDetector:
    version = "fake-face"

    def __init__(self, by_marker: dict[int, list[Box] | None] | None = None) -> None:
        self.by_marker = by_marker or {}

    def detect(self, img: Pixels) -> list[Box] | None:
        return self.by_marker.get(marker_of(img))


@dataclass
class FakeModels:
    """Conjunto de fakes con acceso a cada uno para ajustarlos desde el test."""

    vehicle: FakeVehicleDetector
    cars: FakeCarSegmenter
    curb: FakeCurbSegmenter
    embedder: FakeEmbedder
    plate: FakePlateReader
    damage: FakeDamageDetector
    face: FakeFaceDetector

    def as_models(self) -> Models:
        return Models(self.vehicle, self.cars, self.curb, self.embedder, self.plate, self.damage, self.face)


def happy_models() -> FakeModels:
    """Modelos que producen GREEN en CRASHED y en PARKED_ON_GARAGE (patente = REPORTED_PLATE)."""
    curb = rect(80, 86, 0, 20) | rect(80, 86, 81, 100)
    cut = rect(80, 86, 20, 81)
    embedder = FakeEmbedder()
    embedder.by_marker_area = {
        (GENERAL, 6400): vec(0),
        (PLATE, 6400): vec(10),
        (CONTEXT, 1600): vec(5),
        (CONTEXT, 400): vec(90),
    }
    return FakeModels(
        vehicle=FakeVehicleDetector(
            {
                GENERAL: [car_box(), plate_box()],
                PLATE: [car_box(), plate_box(0.95)],
                CONTEXT: [car_box(), Box(35, 60, 45, 80, 0.9, BoxLabel.WHEEL)],
                **{detail_marker(i): [Box(0, 0, 80, 80, 0.9, BoxLabel.CAR)] for i in range(3)},
            }
        ),
        cars=FakeCarSegmenter(
            {
                GENERAL: [CarInstance(rect(10, 90, 10, 90))],
                PLATE: [CarInstance(rect(10, 90, 10, 90))],
                CONTEXT: [CarInstance(rect(40, 80, 30, 70)), CarInstance(rect(20, 40, 0, 20))],
                **{detail_marker(i): [] for i in range(3)},  # en un primer plano no se segmenta ningún auto
            }
        ),
        curb=FakeCurbSegmenter({CONTEXT: (curb, cut)}),
        embedder=embedder,
        plate=FakePlateReader({GENERAL: (REPORTED_PLATE, 0.9), PLATE: (REPORTED_PLATE, 0.95)}),
        damage=FakeDamageDetector({detail_marker(i): [Box(10, 10, 50, 50, 0.8)] for i in range(3)}),
        face=FakeFaceDetector({m: [] for m in (GENERAL, PLATE, CONTEXT, *map(detail_marker, range(3)))}),
    )


REDACTION = RedactionSettings(required=True, face_min_conf=0.30, plate_min_conf=0.20, max_side_px=1600)


class FakeWriter:
    """Registra lo que se sube a `redacted/`."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.puts: list[str] = []
        self.error: Exception | None = None

    def put(self, key: str, body: bytes, content_type: str) -> None:
        if self.error is not None:
            raise self.error
        self.puts.append(key)
        self.objects[key] = (body, content_type)


# --- almacenamiento, HPKE y decodificación ---------------------------------------------


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.error: Exception | None = None
        self.errors_by_key: dict[str, Exception] = {}
        self.gets: list[str] = []

    def get(self, key: str) -> bytes:
        self.gets.append(key)
        if key in self.errors_by_key:
            raise self.errors_by_key[key]
        if self.error is not None:
            raise self.error
        if key not in self.objects:
            raise ObjectNotFoundError("objeto inexistente")
        return self.objects[key]


class FakeDecryptor:
    """Cifrado de juguete: `ENC|info|plaintext`; abrir exige el mismo `info`."""

    @staticmethod
    def seal(plaintext: bytes, info: str) -> bytes:
        return b"ENC|" + info.encode() + b"|" + plaintext

    def open(self, ciphertext: bytes, info: bytes) -> bytes:
        prefix = b"ENC|" + info + b"|"
        if not ciphertext.startswith(prefix):
            raise DecryptionError("no descifra")
        return ciphertext[len(prefix) :]


class FakeDecoder:
    """Decodifica `IMG<marcador>` a una foto sintética."""

    def decode(self, data: bytes) -> Pixels:
        if not data.startswith(b"IMG"):
            raise ImageDecodeError("no es una imagen")
        return photo(int(data[3:]))


@dataclass
class Scenario:
    """Alerta completa: payload, almacenamiento y fakes de infraestructura."""

    payload: EvidenceUploaded
    store: FakeStore
    alert_id: str
    decryptor: FakeDecryptor = field(default_factory=FakeDecryptor)
    decoder: FakeDecoder = field(default_factory=FakeDecoder)

    def corrupt_sha(self, role: EvidenceRole, index: int = 0) -> None:
        items = tuple(
            item.model_copy(update={"sha256": "0" * 64}) if (item.role, item.index) == (role, index) else item
            for item in self.payload.evidence
        )
        self.payload = self.payload.model_copy(update={"evidence": items})

    def replace_object(self, role: EvidenceRole, data: bytes, index: int = 0) -> None:
        """Cambia el objeto en S3 y su sha256 en el evento (el hash coincide)."""
        key = next(i.key for i in self.payload.evidence if (i.role, i.index) == (role, index))
        self.store.objects[key] = data
        items = tuple(
            item.model_copy(update={"sha256": hashlib.sha256(data).hexdigest()})
            if (item.role, item.index) == (role, index)
            else item
            for item in self.payload.evidence
        )
        self.payload = self.payload.model_copy(update={"evidence": items})

    def with_pat_enc_ia(self, raw: bytes) -> None:
        self.payload = self.payload.model_copy(update={"pat_enc_ia": base64.b64encode(raw).decode()})


def build_scenario(alert_type: AlertType = AlertType.CRASHED, details: int = 1) -> Scenario:
    alert_id = str(uuid.uuid4())
    store = FakeStore()
    roles: list[tuple[EvidenceRole, int, int]] = [
        (EvidenceRole.GENERAL, 0, GENERAL),
        (EvidenceRole.PLATE, 0, PLATE),
    ]
    if alert_type == AlertType.CRASHED:
        roles += [(EvidenceRole.DETAIL, i, detail_marker(i)) for i in range(details)]
    else:
        roles.append((EvidenceRole.CONTEXT, 0, CONTEXT))
    items: list[EvidenceItem] = []
    for role, index, marker in roles:
        ciphertext = FakeDecryptor.seal(f"IMG{marker}".encode(), alert_id)
        key = f"alerts/{alert_id}/original/{role.lower()}-{index}.bin"
        store.objects[key] = ciphertext
        items.append(
            EvidenceItem(role=role, index=index, key=key, sha256=hashlib.sha256(ciphertext).hexdigest())
        )
    pat_enc_ia = base64.b64encode(FakeDecryptor.seal(REPORTED_PLATE.encode(), alert_id)).decode()
    payload = EvidenceUploaded(
        alert_id=alert_id,
        alert_type=alert_type,
        pat_enc_ia=pat_enc_ia,
        bucket=BUCKET,
        evidence=tuple(items),
    )
    return Scenario(payload=payload, store=store, alert_id=alert_id)
