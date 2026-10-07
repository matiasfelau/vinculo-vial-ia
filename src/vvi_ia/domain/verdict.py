"""Agregación de códigos de razón en el semáforo."""

from collections.abc import Iterable

from vvi_ia.domain.model import FRAUD_CODES, RED_CODES, YELLOW_CODES, ReasonCode, Verdict


def aggregate(codes: Iterable[ReasonCode]) -> Verdict:
    """RED si hay algún código RED; si no, YELLOW si hay alguno YELLOW; si no, GREEN."""
    present = set(codes)
    if present & RED_CODES:
        return Verdict.RED
    if present & YELLOW_CODES:
        return Verdict.YELLOW
    return Verdict.GREEN


def fraud_codes(codes: Iterable[ReasonCode]) -> frozenset[ReasonCode]:
    """Subconjunto de los códigos que penalizan al emisor."""
    return frozenset(codes) & FRAUD_CODES


def is_fraud(codes: Iterable[ReasonCode]) -> bool:
    """True si algún código pertenece al conjunto FRAUD."""
    return bool(fraud_codes(codes))
