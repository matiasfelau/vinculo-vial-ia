"""Plan de redacción: qué cajas se tapan en una foto.

El producto tapa solo las patentes de terceros. Los rostros no se redactan (riesgo aceptado: las fotos se
toman a 1-2 m centradas en el vehículo): el detector de rostros es experimental, fuera de la cascada.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from vvi_ia.domain.model import Box, BoxLabel

REDACTION_MARGIN = 0.10
# Una detección de patente con al menos esta parte de su área adentro de la caja exceptuada (la de la patente
# denunciada) es la misma patente o un pedazo de ella (el mosaico de patentes chicas la detecta dos veces):
# tapar esa detección taparía la denunciada.
EXEMPT_CONTAINED = 0.8


@dataclass(frozen=True)
class RedactionThresholds:
    """Confianza mínima para redactar una patente (baja a propósito: ante la duda, tapar)."""

    plate_min_conf: float


def expand(box: Box, margin: float, width: int, height: int) -> Box:
    """Agranda la caja `margin` de su ancho y alto en cada lado, recortada a la imagen."""
    dx, dy = box.width * margin, box.height * margin
    return Box(
        x1=max(0.0, box.x1 - dx),
        y1=max(0.0, box.y1 - dy),
        x2=min(float(width), box.x2 + dx),
        y2=min(float(height), box.y2 + dy),
        conf=box.conf,
        label=box.label,
    )


def contained_fraction(inner: Box, outer: Box) -> float:
    """Qué parte del área de `inner` queda adentro de `outer` (0 si `inner` no tiene área)."""
    if inner.area <= 0:
        return 0.0
    overlap = Box(
        x1=max(inner.x1, outer.x1),
        y1=max(inner.y1, outer.y1),
        x2=min(inner.x2, outer.x2),
        y2=min(inner.y2, outer.y2),
    )
    return overlap.area / inner.area


def plan_redaction(
    detections: Sequence[Box] | None,
    thresholds: RedactionThresholds,
    width: int,
    height: int,
    exempt_plates: Collection[Box] = (),
    margin: float = REDACTION_MARGIN,
    exempt_contained: float | None = EXEMPT_CONTAINED,
) -> list[Box] | None:
    """Cajas a tapar con relleno opaco, o `None` si no se puede redactar (fail-closed).

    `detections` (las del detector de vehículos, de donde salen las patentes) es `None` cuando no hay
    modelo o falló la inferencia: en ese caso no hay plan. Es lo único de lo que depende la disponibilidad de
    la redacción. Las patentes de `exempt_plates` (la que coincide con la reportada) no se tapan, ni tampoco
    las cajas de patente que tienen adentro al menos `exempt_contained` de su área (la misma patente
    detectada otra vez, o un pedazo): taparlas taparía la denunciada. Con `None` solo se exceptúan las cajas
    de `exempt_plates` (el comportamiento anterior, para comparar en las mediciones).
    """
    if detections is None:
        return None
    boxes = [
        plate
        for plate in detections
        if plate.label == BoxLabel.PLATE
        and plate.conf >= thresholds.plate_min_conf
        and plate not in exempt_plates
        and not (
            exempt_contained is not None
            and any(contained_fraction(plate, exempt) >= exempt_contained for exempt in exempt_plates)
        )
    ]
    return [expand(box, margin, width, height) for box in boxes]
