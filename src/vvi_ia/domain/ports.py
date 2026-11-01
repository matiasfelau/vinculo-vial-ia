"""Puertos de los modelos de inferencia.

`None` significa "no hay modelo o falló la inferencia"; una lista vacía significa
"miré y no encontré nada". Un adaptador no escribe a disco, no hace red ni loguea contenido,
y lee sus umbrales de su propia configuración.
"""

from typing import Protocol

from vvi_ia.domain.model import Box, CarInstance, Mask, Pixels, Vector


class VehicleDetector(Protocol):
    """Detecta autos, ruedas y patentes (clases `car`, `wheel`, `plate`)."""

    version: str

    def detect(self, img: Pixels) -> list[Box] | None:
        """Un solo pase sobre la foto entera: el que sirve para la presencia y la lectura de la patente."""
        ...

    def detect_plates(self, img: Pixels) -> list[Box] | None:
        """Lo mismo, más las patentes chicas del mosaico (si está activo). Es más caro: solo para redactar."""
        ...


class CarSegmenter(Protocol):
    """Una máscara booleana por auto, del tamaño de la imagen."""

    version: str

    def cars(self, img: Pixels) -> list[CarInstance] | None: ...


class CurbSegmenter(Protocol):
    """Máscaras `(curb, curb_cut)`; garage y accesibilidad son la misma clase."""

    version: str

    def curb(self, img: Pixels) -> tuple[Mask, Mask] | None: ...


class Embedder(Protocol):
    """Vector 1-D normalizado L2. El adaptador aplica la máscara (fondo gris)."""

    version: str

    def embed(self, img: Pixels, mask: Mask) -> Vector | None: ...


class PlateReader(Protocol):
    """Texto crudo y confianza en [0, 1]; el dominio canonicaliza."""

    version: str

    def read(self, crop: Pixels) -> tuple[str, float] | None: ...


class DamageDetector(Protocol):
    """Cajas de daño, con confianza."""

    version: str

    def detect(self, img: Pixels) -> list[Box] | None: ...


class FaceDetector(Protocol):
    """Cajas de rostro, con confianza."""

    version: str

    def detect(self, img: Pixels) -> list[Box] | None: ...
