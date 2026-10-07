"""Re-identificación: ¿dos fotos muestran el mismo auto?"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from vvi_ia.domain.model import Mask, Pixels, Vector

GRAY_FILL = 128


class ReidOutcome(StrEnum):
    """Decisión según los umbrales de similitud."""

    SAME = "SAME"
    UNCERTAIN = "UNCERTAIN"
    DIFFERENT = "DIFFERENT"


@dataclass(frozen=True)
class ReidResult:
    """Similitud coseno y la decisión que se desprende de ella."""

    similarity: float
    outcome: ReidOutcome


def masked_crop(img: Pixels, mask: Mask, fill: int = GRAY_FILL) -> Pixels | None:
    """Recorte de la caja de la máscara, con el fondo (fuera de la máscara) en gris.

    Devuelve `None` si la máscara está vacía.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y1:y2, x1:x2].copy()
    crop[~mask[y1:y2, x1:x2]] = fill
    return crop


def cosine(a: Vector, b: Vector) -> float:
    """Similitud coseno; 0.0 si alguno de los vectores es nulo."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("los embeddings deben ser vectores 1-D del mismo largo")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))


def same_vehicle(a: Vector, b: Vector, t_low: float, t_high: float) -> ReidResult:
    """cos ≥ t_high → SAME · entre los umbrales → UNCERTAIN · cos < t_low → DIFFERENT."""
    similarity = cosine(a, b)
    if similarity >= t_high:
        return ReidResult(similarity, ReidOutcome.SAME)
    if similarity < t_low:
        return ReidResult(similarity, ReidOutcome.DIFFERENT)
    return ReidResult(similarity, ReidOutcome.UNCERTAIN)


def best_match(candidates: Sequence[Vector], references: Sequence[Vector]) -> tuple[int, float] | None:
    """Índice del candidato con mayor coseno contra alguna referencia, y ese coseno."""
    best: tuple[int, float] | None = None
    for index, candidate in enumerate(candidates):
        for reference in references:
            score = cosine(candidate, reference)
            if best is None or score > best[1]:
                best = (index, score)
    return best
