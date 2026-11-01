"""EXPERIMENTAL, EN CONSTRUCCIÓN y fuera de la cascada: ¿el auto bloquea el tramo rebajado del cordón?

No lo usa `ValidateEvidence`: el chequeo de rampa se sacó del veredicto. Queda para retomarlo cuando haya un
modelo de rebajes entrenado con fotos propias.

Sin chequeo de distancia al cordón (decisión explícita: el auto de enfrente es un riesgo aceptado).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.domain.model import Box, Mask

_KERNEL_SIZE = 3
_FIT_LINE_EPS = 0.01
_MIN_COMPONENTS = 2


class DrivewayCheck(StrEnum):
    """Resultado de `driveway_check`: la decisión es binaria (bloquea o no)."""

    BLOCKING = "BLOCKING"
    NOT_BLOCKING = "NOT_BLOCKING"
    NO_DRIVEWAY = "NO_DRIVEWAY"
    NO_CAR = "NO_CAR"


@dataclass(frozen=True)
class DrivewayMeasure:
    """Qué se pudo medir: `status` si no hay auto o rebaje, o el `score` del solape (en escalas de rueda).

    `score > 0` es que el auto se mete en el rebaje; `score < 0`, que queda a esa distancia.
    """

    status: DrivewayCheck | None
    score: float | None


def _largest(mask: Mask) -> Mask:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < _MIN_COMPONENTS:
        return mask
    return np.asarray(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _lower_envelope(mask: Mask) -> NDArray[np.float32]:
    cols = np.flatnonzero(mask.any(axis=0))
    rows = mask.shape[0] - 1 - np.argmax(mask[::-1, cols], axis=0)
    return np.column_stack([cols, rows]).astype(np.float32)


def _open(mask: Mask) -> Mask:
    kernel = np.ones((_KERNEL_SIZE, _KERNEL_SIZE), np.uint8)
    return np.asarray(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool))


def wheel_diameter(wheels: Sequence[Box], car: Mask) -> float | None:
    """Mediana de la altura de las ruedas cuyo centro cae dentro de la máscara del auto."""
    height, width = car.shape
    heights = [
        wheel.height
        for wheel in wheels
        if 0 <= int(wheel.center[0]) < width
        and 0 <= int(wheel.center[1]) < height
        and car[int(wheel.center[1]), int(wheel.center[0])]
    ]
    return float(np.median(heights)) if heights else None


def driveway_overlap(
    curb: Mask,
    cut: Mask,
    car: Mask,
    wheel_diam: float | None = None,
    scale_fallback: float = 0.45,
    min_cut: float = 0.002,
) -> DrivewayMeasure:
    """Mide cuánto del tramo rebajado del cordón queda cubierto por el auto.

    Proyecta la envolvente inferior del auto y el rebaje sobre la recta del cordón. El solape se expresa en
    escalas: diámetros de rueda (o, sin rueda, una fracción de la altura del auto).
    """
    cut = _open(cut)
    car = _open(car)
    if not car.any():
        return DrivewayMeasure(DrivewayCheck.NO_CAR, None)
    if cut.sum() < min_cut * cut.size:
        return DrivewayMeasure(DrivewayCheck.NO_DRIVEWAY, None)
    pts = np.column_stack(np.nonzero(_largest(curb | cut))[::-1]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, _FIT_LINE_EPS, _FIT_LINE_EPS).ravel()
    direction, origin = np.array([vx, vy]), np.array([x0, y0])
    scale = wheel_diam or scale_fallback * np.ptp(np.nonzero(car)[0])
    t_car = (_lower_envelope(car) - origin) @ direction
    t_cut = (np.column_stack(np.nonzero(cut)[::-1]).astype(np.float32) - origin) @ direction
    overlap = min(t_car.max(), t_cut.max()) - max(t_car.min(), t_cut.min())
    return DrivewayMeasure(None, float(overlap / scale))


class RampOutcome(StrEnum):
    """Lectura del solape con dos umbrales (experimental: ya no es parte del veredicto)."""

    NOT_DETECTED = "NOT_DETECTED"  # sin rebaje, o el auto claramente aparte
    UNCERTAIN = "UNCERTAIN"  # sin auto que medir, o entre los dos umbrales


def ramp_reason(measure: DrivewayMeasure, t_low: float, t_high: float) -> RampOutcome | None:
    """Lectura del solape: `None` si el auto bloquea con claridad.

    Sin auto no se puede medir (dudoso). Sin rebaje, o con el auto claramente aparte (`score < t_low`), el
    problema no está. Entre `t_low` y `t_high` es dudoso.
    """
    if measure.status == DrivewayCheck.NO_CAR:
        return RampOutcome.UNCERTAIN
    if measure.status == DrivewayCheck.NO_DRIVEWAY or (measure.score is not None and measure.score < t_low):
        return RampOutcome.NOT_DETECTED
    if measure.score is not None and measure.score < t_high:
        return RampOutcome.UNCERTAIN
    return None


def driveway_check(
    curb: Mask,
    cut: Mask,
    car: Mask,
    wheel_diam: float | None = None,
    margin: float = 0.08,
    scale_fallback: float = 0.45,
    min_cut: float = 0.002,
) -> DrivewayCheck:
    """Decisión binaria: BLOCKING si el solape supera `-margin` escalas (casi tocando también bloquea)."""
    measure = driveway_overlap(curb, cut, car, wheel_diam, scale_fallback, min_cut)
    if measure.status is not None:
        return measure.status
    assert measure.score is not None
    return DrivewayCheck.BLOCKING if measure.score > -margin else DrivewayCheck.NOT_BLOCKING
