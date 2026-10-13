"""Agregación de códigos de razón en el veredicto (solo GREEN o RED)."""

from collections.abc import Iterable

from vvi_ia.domain.model import FRAUD_CODES, RED_CODES, ReasonCode, Verdict


def aggregate(codes: Iterable[ReasonCode]) -> Verdict:
    """RED si hay algún código; sin códigos, GREEN."""
    return Verdict.RED if set(codes) & RED_CODES else Verdict.GREEN


def fraud_codes(codes: Iterable[ReasonCode]) -> frozenset[ReasonCode]:
    """Subconjunto de los códigos que penalizan al emisor."""
    return frozenset(codes) & FRAUD_CODES


def is_fraud(codes: Iterable[ReasonCode]) -> bool:
    """True si algún código pertenece al conjunto FRAUD."""
    return bool(fraud_codes(codes))
