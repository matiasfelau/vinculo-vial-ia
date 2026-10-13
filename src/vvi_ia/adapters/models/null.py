"""Adaptadores nulos: todos los puertos devuelven `None` ("no hay modelo").

Es el único adaptador de US-08. Con ellos no hay evidencia que penalizar, pero sin redacción disponible el
veredicto es RED (REDACTION_UNAVAILABLE).
"""

from vvi_ia.domain.model import Box, CarInstance, Mask, Pixels, Vector

NULL_VERSION = "null"


class NullVehicleDetector:
    version = NULL_VERSION

    def detect(self, img: Pixels) -> list[Box] | None:
        return None


class NullCarSegmenter:
    version = NULL_VERSION

    def cars(self, img: Pixels) -> list[CarInstance] | None:
        return None


class NullCurbSegmenter:
    version = NULL_VERSION

    def curb(self, img: Pixels) -> tuple[Mask, Mask] | None:
        return None


class NullEmbedder:
    version = NULL_VERSION

    def embed(self, img: Pixels, mask: Mask) -> Vector | None:
        return None


class NullPlateReader:
    version = NULL_VERSION

    def read(self, crop: Pixels) -> tuple[str, float] | None:
        return None


class NullDamageDetector:
    version = NULL_VERSION

    def detect(self, img: Pixels) -> list[Box] | None:
        return None


class NullFaceDetector:
    version = NULL_VERSION

    def detect(self, img: Pixels) -> list[Box] | None:
        return None
