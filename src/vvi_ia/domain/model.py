"""Tipos del dominio: cajas, imágenes de evidencia, códigos de razón y resultado."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

Pixels = NDArray[np.uint8]
Mask = NDArray[np.bool_]
Vector = NDArray[np.floating[Any]]


class AlertType(StrEnum):
    """Tipos de alerta con validación."""

    PARKED_ON_GARAGE = "PARKED_ON_GARAGE"
    CRASHED = "CRASHED"


class EvidenceRole(StrEnum):
    """Rol de cada foto dentro de la alerta."""

    GENERAL = "GENERAL"
    PLATE = "PLATE"
    DETAIL = "DETAIL"
    CONTEXT = "CONTEXT"


class Verdict(StrEnum):
    """Veredicto de la IA: solo GREEN o RED. RED requiere evidencia firme; lo dudoso es GREEN."""

    GREEN = "GREEN"
    RED = "RED"


class ReasonCode(StrEnum):
    """Códigos de razón de un veredicto RED (contrato compartido). GREEN no lleva códigos."""

    NO_VEHICLE = "NO_VEHICLE"  # R1: en alguna foto no hay evidencia de un auto (fraude)
    BURST_MISMATCH = "BURST_MISMATCH"  # R2: la re-ID confirma que es otro auto (fraude)
    PLATE_MISMATCH = "PLATE_MISMATCH"  # R3: lectura segura de una patente que no coincide (fraude)
    EVIDENCE_UNREADABLE = "EVIDENCE_UNREADABLE"  # R4: foto ilegible, alterada o demasiado grande
    REDACTION_UNAVAILABLE = "REDACTION_UNAVAILABLE"  # R4: no se pudo redactar todas las fotos


RED_CODES: frozenset[ReasonCode] = frozenset(ReasonCode)
FRAUD_CODES: frozenset[ReasonCode] = frozenset(
    {ReasonCode.NO_VEHICLE, ReasonCode.BURST_MISMATCH, ReasonCode.PLATE_MISMATCH}
)


class BoxLabel(StrEnum):
    """Clases que devuelve un detector de vehículos."""

    CAR = "car"
    WHEEL = "wheel"
    PLATE = "plate"


@dataclass(frozen=True)
class Box:
    """Caja en píxeles (x1, y1) esquina superior izquierda, (x2, y2) inferior derecha."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    label: str = ""

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(frozen=True)
class CarInstance:
    """Un auto segmentado: máscara booleana del tamaño de la imagen."""

    mask: Mask
    conf: float = 1.0


@dataclass(frozen=True)
class EvidenceImage:
    """Foto de evidencia ya descifrada y decodificada (solo vive en RAM)."""

    role: EvidenceRole
    index: int
    pixels: Pixels
    view: Pixels | None = (
        None  # la misma foto reducida: es la que ven los modelos (los píxeles son los originales)
    )

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def model_pixels(self) -> Pixels:
        """La foto que ven los modelos: la reducida, o la original si no hizo falta reducir."""
        return self.pixels if self.view is None else self.view

    @property
    def ratio(self) -> float:
        """Cuántas veces es más grande la original que la reducida (para volver a sus coordenadas)."""
        return self.width / int(self.model_pixels.shape[1])


@dataclass(frozen=True)
class Scores:
    """Puntajes que viajan en el evento para el reentrenamiento futuro."""

    burst_similarity: float | None = None
    ocr_confidence: float | None = None
    context_confidence: float | None = None


@dataclass(frozen=True)
class RedactedRef:
    """Referencia a una versión redactada ya subida."""

    role: EvidenceRole
    index: int
    key: str


@dataclass(frozen=True)
class ValidationResult:
    """Resultado de la cascada de validación."""

    verdict: Verdict
    reason_codes: tuple[ReasonCode, ...]
    scores: Scores = field(default_factory=Scores)
    model_versions: Mapping[str, str] = field(default_factory=dict)
    redacted: tuple[RedactedRef, ...] = ()
