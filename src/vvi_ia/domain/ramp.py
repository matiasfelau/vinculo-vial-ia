"""Chequeo de rampa o rebaje: ¿el auto bloquea el tramo rebajado del cordón?

Sin chequeo de distancia al cordón (decisión explícita: el auto de enfrente es un riesgo aceptado).
"""

from collections.abc import Sequence
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from vvi_ia.domain.model import Box, Mask

_KERNEL_SIZE = 3
_FIT_LINE_EPS = 0.01
_MIN_COMPONENTS = 2


class DrivewayCheck(StrEnum):
    """Resultado de `driveway_check`."""

    BLOCKING = "BLOCKING"
    BLOCKING_MARGINAL = "BLOCKING_MARGINAL"
    NOT_BLOCKING = "NOT_BLOCKING"
    NO_DRIVEWAY = "NO_DRIVEWAY"
    NO_CAR = "NO_CAR"


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


def driveway_check(
    curb: Mask,
    cut: Mask,
    car: Mask,
    wheel_diam: float | None = None,
    margin: float = 0.08,
    scale_fallback: float = 0.45,
    min_cut: float = 0.002,
) -> DrivewayCheck:
    """Mide cuánto del tramo rebajado del cordón queda cubierto por el auto.

    Proyecta la envolvente inferior del auto y el rebaje sobre la recta del cordón. El solape
    se compara contra `margin * escala`, con la escala en diámetros de rueda (o, sin rueda,
    una fracción de la altura del auto).
    """
    cut = _open(cut)
    car = _open(car)
    if not car.any():
        return DrivewayCheck.NO_CAR
    if cut.sum() < min_cut * cut.size:
        return DrivewayCheck.NO_DRIVEWAY
    pts = np.column_stack(np.nonzero(_largest(curb | cut))[::-1]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, _FIT_LINE_EPS, _FIT_LINE_EPS).ravel()
    direction, origin = np.array([vx, vy]), np.array([x0, y0])
    scale = wheel_diam or scale_fallback * np.ptp(np.nonzero(car)[0])
    t_car = (_lower_envelope(car) - origin) @ direction
    t_cut = (np.column_stack(np.nonzero(cut)[::-1]).astype(np.float32) - origin) @ direction
    overlap = min(t_car.max(), t_cut.max()) - max(t_car.min(), t_cut.min())
    if overlap > margin * scale:
        return DrivewayCheck.BLOCKING
    if overlap > -margin * scale:
        return DrivewayCheck.BLOCKING_MARGINAL
    return DrivewayCheck.NOT_BLOCKING
