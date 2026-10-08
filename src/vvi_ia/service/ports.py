"""Puertos de infraestructura del caso de uso y los errores que cruzan esa frontera.

Los mensajes de las excepciones nunca llevan contenido (patente, texto del OCR, imágenes).
"""

from collections.abc import Sequence
from typing import Protocol

from vvi_ia.domain.model import Box, Pixels


class TransientError(Exception):
    """Falla recuperable (S3 caído, timeout, circuito abierto): el mensaje se reintenta."""


class ObjectNotFoundError(Exception):
    """El objeto de evidencia no existe en el almacenamiento: error del sistema, no reintentable."""


class DecryptionError(Exception):
    """El criptograma HPKE no se pudo abrir (clave, `info` o contenido inválidos)."""


class ImageDecodeError(Exception):
    """Los bytes no son una imagen JPEG válida o superan el límite de píxeles."""


class SystemDecryptionError(Exception):
    """`pat_enc_ia` no descifra: error del sistema, el mensaje va a la DLQ."""


class EvidenceStore(Protocol):
    """Lectura de los originales cifrados (S3)."""

    def get(self, key: str) -> bytes:
        """Descarga un objeto. Lanza `ObjectNotFoundError` o `TransientError`."""
        ...


class PayloadDecryptor(Protocol):
    """Apertura de criptogramas HPKE con la clave privada de IA."""

    def open(self, ciphertext: bytes, info: bytes) -> bytes:
        """Descifra `enc‖ct`. Lanza `DecryptionError`."""
        ...


class ImageDecoder(Protocol):
    """Decodificación segura de JPEG a píxeles RGB."""

    def decode(self, data: bytes) -> Pixels:
        """Decodifica con límite de píxeles. Lanza `ImageDecodeError`."""
        ...


class EvidenceWriter(Protocol):
    """Escritura de las versiones redactadas (S3, solo `redacted/`)."""

    def put(self, key: str, body: bytes, content_type: str) -> None:
        """Sube un objeto. Lanza `TransientError`."""
        ...


class RedactionRenderer(Protocol):
    """Tapa las cajas con relleno opaco, limita el lado mayor y codifica a JPEG sin EXIF."""

    def render(self, pixels: Pixels, boxes: Sequence[Box], max_side: int) -> bytes:
        """Devuelve el JPEG final."""
        ...
