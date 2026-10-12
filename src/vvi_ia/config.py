"""Configuración del servicio: se lee del entorno y falla al arrancar si es inválida."""

import base64
import binascii
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HPKE_KEY_BYTES = 32
NULL_MODEL = "null"
MODEL_NAME_PATTERN = r"^[a-z0-9][a-z0-9_]*$"
LOCAL_ENV = "local"


class Settings(BaseSettings):
    """Variables de entorno del servicio. Las obligatorias no tienen valor por defecto."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False, protected_namespaces=()
    )

    # Broker
    rabbitmq_url: SecretStr
    rabbitmq_exchange: str = "vvi.topic.exchange"
    rabbitmq_dlq_exchange: str = "vvi.dlq.exchange"
    input_queue: str = "ia.alert-evidence-uploaded.q"
    max_retries: int = Field(default=5, ge=0)
    workers: int = Field(default=2, ge=1)
    publish_confirm_timeout_ms: int = Field(default=5000, ge=1)

    # S3
    s3_region: str = Field(min_length=1)
    evidence_bucket: str = Field(min_length=1)
    evidence_max_part_bytes: int = Field(default=1_048_576, ge=1)
    s3_timeout_ms: int = Field(default=5000, ge=1)
    s3_breaker_failure_threshold: int = Field(default=5, ge=1)
    s3_breaker_window_ms: int = Field(default=60000, ge=1)
    s3_breaker_open_duration_ms: int = Field(default=30000, ge=1)
    s3_breaker_half_open_max_calls: int = Field(default=1, ge=1)

    # Criptografía
    ia_hpke_private_key: SecretStr

    # Entorno y redacción
    env: str = Field(min_length=1)
    redaction_required: bool = True

    # Umbrales del dominio
    reid_t_low: float = Field(default=0.50, ge=0.0, le=1.0)
    reid_t_high: float = Field(default=0.70, ge=0.0, le=1.0)
    ocr_min_conf: float = Field(default=0.60, ge=0.0, le=1.0)
    damage_min_conf: float = Field(default=0.40, ge=0.0, le=1.0)
    face_min_conf: float = Field(default=0.30, ge=0.0, le=1.0)
    plate_redact_min_conf: float = Field(default=0.20, ge=0.0, le=1.0)
    ramp_margin: float = Field(default=0.08, ge=0.0)
    ramp_scale_fallback: float = Field(default=0.45, gt=0.0)
    redact_max_side_px: int = Field(default=1600, ge=1)
    segmenter_min_conf: float = Field(default=0.50, ge=0.0, le=1.0)
    detector_min_conf: float = Field(default=0.40, ge=0.0, le=1.0)
    curb_min_conf: float = Field(default=0.50, ge=0.0, le=1.0)

    # Modelos: `null` (default) o el nombre de un modelo de models/manifest.json
    model_detector: str = Field(default=NULL_MODEL, pattern=MODEL_NAME_PATTERN)
    model_face: str = Field(default=NULL_MODEL, pattern=MODEL_NAME_PATTERN)
    model_segmenter: str = Field(default=NULL_MODEL, pattern=MODEL_NAME_PATTERN)
    model_curb: str = Field(default=NULL_MODEL, pattern=MODEL_NAME_PATTERN)
    model_embedder: str = Field(default=NULL_MODEL, pattern=MODEL_NAME_PATTERN)
    model_ocr: str = Field(default=NULL_MODEL, pattern=MODEL_NAME_PATTERN)
    model_damage: str = Field(default=NULL_MODEL, pattern=MODEL_NAME_PATTERN)
    models_dir: Path | None = None
    models_manifest: Path = Path("models/manifest.json")
    onnx_intra_op_threads: int = Field(default=0, ge=0)

    health_port: int = Field(default=8080, ge=1, le=65535)

    @field_validator("ia_hpke_private_key")
    @classmethod
    def _hpke_key_is_x25519(cls, value: SecretStr) -> SecretStr:
        try:
            raw = base64.b64decode(value.get_secret_value(), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("IA_HPKE_PRIVATE_KEY no es base64 estándar") from exc
        if len(raw) != HPKE_KEY_BYTES:
            raise ValueError(f"IA_HPKE_PRIVATE_KEY debe tener {HPKE_KEY_BYTES} bytes")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.reid_t_low >= self.reid_t_high:
            raise ValueError("REID_T_LOW debe ser menor que REID_T_HIGH")
        if not self.redaction_required and self.env != LOCAL_ENV:
            raise ValueError(f"REDACTION_REQUIRED=false solo se acepta con ENV={LOCAL_ENV}")
        if self.models_dir is None and any(name != NULL_MODEL for name in self.model_selection().values()):
            raise ValueError("MODELS_DIR es obligatorio cuando algún MODEL_* no es null")
        return self

    def model_selection(self) -> dict[str, str]:
        """Selección `{puerto: nombre}` tal como la espera el registro de modelos."""
        return {
            "detector": self.model_detector,
            "face": self.model_face,
            "segmenter": self.model_segmenter,
            "curb": self.model_curb,
            "embedder": self.model_embedder,
            "ocr": self.model_ocr,
            "damage": self.model_damage,
        }


def load_settings() -> Settings:
    """Construye la configuración leyendo el entorno; lanza ValidationError si es inválida."""
    return Settings()  # type: ignore[call-arg]
