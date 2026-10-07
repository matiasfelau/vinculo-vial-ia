"""Payload del evento `alert.evidence.uploaded`: validación estricta de esquema (paso 0)."""

import base64
import binascii
import re
import uuid
from collections import Counter
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from vvi_ia.domain.model import AlertType, EvidenceRole

MAX_PAT_ENC_IA_CHARS = 1024
MAX_DETAIL_PHOTOS = 3
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

_ROLE_COUNTS: Mapping[AlertType, Mapping[EvidenceRole, range]] = {
    AlertType.CRASHED: {
        EvidenceRole.GENERAL: range(1, 2),
        EvidenceRole.PLATE: range(1, 2),
        EvidenceRole.DETAIL: range(1, MAX_DETAIL_PHOTOS + 1),
    },
    AlertType.PARKED_ON_GARAGE: {
        EvidenceRole.GENERAL: range(1, 2),
        EvidenceRole.PLATE: range(1, 2),
        EvidenceRole.CONTEXT: range(1, 2),
    },
}


class InvalidPayloadError(Exception):
    """El evento no cumple el esquema: va directo a la DLQ, sin reintento."""


class EvidenceItem(BaseModel):
    """Una foto original cifrada en S3."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    role: EvidenceRole
    index: int = Field(ge=0)
    key: str = Field(min_length=1, max_length=512)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _sha256_is_hex(cls, value: str) -> str:
        if not _SHA256_HEX.fullmatch(value):
            raise ValueError("sha256 debe ser hexadecimal en minúsculas de 64 caracteres")
        return value


class EvidenceUploaded(BaseModel):
    """Payload de `alert.evidence.uploaded`."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    alert_id: str
    alert_type: AlertType
    pat_enc_ia: str = Field(min_length=1, max_length=MAX_PAT_ENC_IA_CHARS)
    bucket: str = Field(min_length=1)
    evidence: tuple[EvidenceItem, ...]

    @field_validator("alert_id")
    @classmethod
    def _alert_id_is_uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value

    @field_validator("pat_enc_ia")
    @classmethod
    def _pat_enc_ia_is_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("pat_enc_ia no es base64 estándar") from exc
        return value

    @model_validator(mode="after")
    def _evidence_matches_alert_type(self) -> Self:
        allowed = _ROLE_COUNTS[self.alert_type]
        counts = Counter(item.role for item in self.evidence)
        for role in counts:
            if role not in allowed:
                raise ValueError(f"el rol {role} no corresponde a {self.alert_type}")
        for role, valid in allowed.items():
            if counts[role] not in valid:
                raise ValueError(f"cantidad inválida de fotos con rol {role}")
        for role in counts:
            indexes = sorted(item.index for item in self.evidence if item.role == role)
            if indexes != list(range(len(indexes))):
                raise ValueError(f"los índices del rol {role} deben ser 0..n-1 sin repetir")
        prefix = f"alerts/{self.alert_id}/original/"
        for item in self.evidence:
            if not item.key.startswith(prefix) or ".." in item.key:
                raise ValueError("la clave del objeto no pertenece a la alerta")
        return self

    def items(self, role: EvidenceRole) -> list[EvidenceItem]:
        """Fotos de un rol, ordenadas por índice."""
        return sorted((item for item in self.evidence if item.role == role), key=lambda i: i.index)


def parse_evidence_uploaded(raw: Mapping[str, Any], expected_bucket: str) -> EvidenceUploaded:
    """Valida el payload y que apunte al bucket configurado (lista de permitidos).

    Lanza `InvalidPayloadError` sin incluir los valores recibidos.
    """
    try:
        payload = EvidenceUploaded.model_validate(raw)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise InvalidPayloadError(f"payload inválido en: {', '.join(fields) or 'raíz'}") from None
    if payload.bucket != expected_bucket:
        raise InvalidPayloadError("el bucket del evento no es el configurado")
    return payload
