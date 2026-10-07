"""Plan de redacción: qué cajas se tapan en una foto."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from vvi_ia.domain.model import Box, BoxLabel

REDACTION_MARGIN = 0.10


@dataclass(frozen=True)
class RedactionThresholds:
    """Confianza mínima para redactar rostros y patentes (bajas a propósito: ante la duda, tapar)."""

    face_min_conf: float
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


def plan_redaction(
    faces: Sequence[Box] | None,
    detections: Sequence[Box] | None,
    thresholds: RedactionThresholds,
    width: int,
    height: int,
    exempt_plates: Collection[Box] = (),
    margin: float = REDACTION_MARGIN,
) -> list[Box] | None:
    """Cajas a tapar con relleno opaco, o `None` si no se puede redactar (fail-closed).

    `faces` y `detections` (las detecciones de vehículo, de donde salen las patentes) son
    `None` cuando no hay modelo: en ese caso no hay plan. Las patentes de `exempt_plates`
    (la que coincide con la reportada) no se tapan.
    """
    if faces is None or detections is None:
        return None
    boxes = [face for face in faces if face.conf >= thresholds.face_min_conf]
    boxes += [
        plate
        for plate in detections
        if plate.label == BoxLabel.PLATE
        and plate.conf >= thresholds.plate_min_conf
        and plate not in exempt_plates
    ]
    return [expand(box, margin, width, height) for box in boxes]
