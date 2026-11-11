"""Métricas de verificación de re-ID: AUC, U1, S4, cotas del 95 % y calibración del umbral.

Se decide con el mismo criterio que el servicio: dos fotos son **otro auto** si su coseno es menor que
`REID_T`.

- **U1**: pares del *mismo* auto marcados como distintos (coseno < T). Es el error grave: penaliza a un
  usuario
  honesto. Se acota con Clopper-Pearson (con 0 fallas, 3/N).
- **S4**: pares de autos *distintos* detectados como distintos (coseno < T). Es lo que se gana con el umbral.

Todo es puro (NumPy): nada lee datos ni modelos.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Scores = NDArray[np.float64]
DEFAULT_MAX_U1 = 0.01
CONFIDENCE = 0.95
BISECTION_STEPS = 60


def auc(same: Sequence[float] | Scores, different: Sequence[float] | Scores) -> float | None:
    """Probabilidad de que un par del mismo auto tenga más coseno que uno de distintos (empates = 1/2)."""
    positives, negatives = np.asarray(same, np.float64), np.asarray(different, np.float64)
    if positives.size == 0 or negatives.size == 0:
        return None
    ordered = np.sort(negatives)
    below = np.searchsorted(ordered, positives, side="left")
    equal = np.searchsorted(ordered, positives, side="right") - below
    return float((below + 0.5 * equal).sum() / (positives.size * negatives.size))


def upper_bound(failures: int, trials: int, confidence: float = CONFIDENCE) -> float | None:
    """Cota superior de la tasa de fallas: 3/N con 0 fallas; con más, Clopper-Pearson exacta (una cola)."""
    if trials <= 0:
        return None
    if failures == 0:
        return min(1.0, 3 / trials)
    if failures >= trials:
        return 1.0
    alpha = 1.0 - confidence
    low, high = failures / trials, 1.0
    for _ in range(BISECTION_STEPS):  # la p más alta cuya cola P(X <= fallas) todavía es >= alpha
        mid = (low + high) / 2
        log_q, log_p = math.log1p(-mid), math.log(mid)
        tail = sum(
            math.exp(
                math.lgamma(trials + 1)
                - math.lgamma(i + 1)
                - math.lgamma(trials - i + 1)
                + i * log_p
                + (trials - i) * log_q
            )
            for i in range(failures + 1)
        )
        low, high = (mid, high) if tail >= alpha else (low, mid)
    return (low + high) / 2


def rate(count: int, total: int) -> float | None:
    return count / total if total else None


def at_threshold(same: Scores, different: Scores, threshold: float) -> dict[str, Any]:
    """U1 y S4 con su cota (U1) a un umbral dado."""
    u1_count = int((same < threshold).sum())
    s4_count = int((different < threshold).sum())
    return {
        "pares_mismo_auto": int(same.size),
        "pares_autos_distintos": int(different.size),
        "U1": rate(u1_count, int(same.size)),
        "U1_cota_95": upper_bound(u1_count, int(same.size)),
        "U1_pares": u1_count,
        "S4": rate(s4_count, int(different.size)),
        "S4_pares": s4_count,
    }


def calibrate(same: Scores, different: Scores, max_u1: float = DEFAULT_MAX_U1) -> dict[str, Any]:
    """El mayor umbral cuya cota del 95 % de U1 (sobre estos pares) no pasa de `max_u1`, y el S4 que deja.

    Con `n` pares del mismo auto se admiten hasta `k` fallas, la mayor cantidad con `upper_bound(k, n) <=
    max_u1`;
    el umbral es el `k`-ésimo coseno más bajo (el que deja afuera a lo sumo `k` pares). `None` si ni con 0
    fallas alcanza (hacen falta al menos 3 / `max_u1` pares).
    """
    n = int(same.size)
    if n == 0 or different.size == 0:
        return {"umbral": None, "motivo": "faltan pares"}
    allowed = -1
    for k in range(0, n):
        bound = upper_bound(k, n)
        if bound is None or bound > max_u1:
            break
        allowed = k
    if allowed < 0:
        return {"umbral": None, "motivo": f"con {n} pares la cota con 0 fallas ({3 / n:.4f}) supera {max_u1}"}
    ordered = np.sort(same)
    threshold = float(ordered[allowed])
    return {"umbral": threshold, "fallas_admitidas": allowed, **at_threshold(same, different, threshold)}


def histogram(values: Scores, bins: int = 20, low: float = -1.0, high: float = 1.0) -> list[int]:
    return [int(x) for x in np.histogram(values, bins=bins, range=(low, high))[0]]
