"""Manifest de modelos (`models/manifest.json`): qué pesos existen, con qué licencia y con qué hash.

Los pesos nunca van a git ni a la imagen: viven en `MODELS_DIR`, montado de solo lectura. El manifest
sí se versiona y es lo que permite verificar al arrancar que el archivo montado es el esperado.
"""

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SHA256_HEX = re.compile(r"[0-9a-f]{64}")
SAFE_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
VERSION_HASH_CHARS = 12
PORTS = ("detector", "face", "segmenter", "curb", "embedder", "ocr", "damage")


class ManifestError(Exception):
    """El manifest no existe o no cumple el esquema."""


class Commercial(StrEnum):
    """Camino a uso comercial de un modelo."""

    YES = "yes"  # licencia comercial
    ESCALABLE = "escalable"  # académica, con autorización comercial posible
    NO = "no"  # solo vive en tools para comparar; la única excepción es el puerto `face` (ver registry.py)


class Paper(BaseModel):
    """Publicación con revisión de pares que respalda el modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    year: int = Field(ge=1990, le=2100)


class ModelEntry(BaseModel):
    """Un modelo del manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_]*$")
    version: str = Field(min_length=1)
    ports: tuple[str, ...] = Field(min_length=1)
    file: str
    sha256: str | None
    license: str = Field(min_length=1)
    paper: Paper
    source: str = Field(min_length=1)
    commercial: Commercial
    notes: str = ""
    # Clases del modelo, en el orden de las columnas de salida, para los que las declaran (el detector v2).
    # Un modelo sin `classes` usa las de su adaptador.
    classes: tuple[str, ...] | None = Field(default=None, min_length=1)

    @field_validator("ports")
    @classmethod
    def _known_ports(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(PORTS))
        if unknown:
            raise ValueError(f"puertos desconocidos: {', '.join(unknown)}")
        return value

    @field_validator("file")
    @classmethod
    def _plain_file_name(cls, value: str) -> str:
        # Solo un nombre de archivo: nada de rutas, para que no se pueda salir de MODELS_DIR.
        if not SAFE_FILE_NAME.fullmatch(value):
            raise ValueError("file debe ser un nombre de archivo simple, sin directorios")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_is_hex(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_HEX.fullmatch(value):
            raise ValueError("sha256 debe ser hexadecimal en minúsculas de 64 caracteres")
        return value

    def version_label(self, sha256: str) -> str:
        """`<modelo>@<sha256[:12]>`: lo que viaja en `model_versions`."""
        return f"{self.name}@{sha256[:VERSION_HASH_CHARS]}"


class Manifest(BaseModel):
    """Contenido completo de `models/manifest.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1, le=1)
    models: tuple[ModelEntry, ...]

    @model_validator(mode="after")
    def _unique_names(self) -> "Manifest":
        names = [entry.name for entry in self.models]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"modelos repetidos: {', '.join(repeated)}")
        return self

    def get(self, name: str) -> ModelEntry | None:
        return next((entry for entry in self.models if entry.name == name), None)


def load_manifest(path: Path) -> Manifest:
    """Lee y valida el manifest. Lanza `ManifestError`."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"no existe el manifest {path.name}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"no se pudo leer el manifest: {type(exc).__name__}") from None
    try:
        return Manifest.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_input=False)
        )
        raise ManifestError(f"manifest inválido: {details}") from None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
