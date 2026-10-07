"""Canonicalización y comparación de patentes."""

import re
from enum import StrEnum

from rapidfuzz.distance import Levenshtein

TO_DIGIT = str.maketrans("OQDIZSGB", "00012568")
TO_LETTER = str.maketrans("012568", "OIZSGB")

FUZZY_DISTANCE = 1
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


class PlateCheck(StrEnum):
    """Resultado de comparar la patente reportada con la leída por OCR."""

    MATCH = "PLATE_MATCH"
    FUZZY = "PLATE_FUZZY"
    MISMATCH = "PLATE_MISMATCH"
    UNREADABLE = "PLATE_UNREADABLE"


def canonicalize(raw: str) -> str:
    """Mayúsculas y solo letras y dígitos."""
    return _NON_ALNUM.sub("", raw.upper())


def plate_check(reported: str, ocr: str | None, ocr_conf: float, min_conf: float) -> PlateCheck:
    """Compara la patente reportada (canónica) con el texto del OCR.

    Si el OCR tiene el mismo largo, corrige las confusiones típicas (O/0, I/1, B/8...) según
    la posición: dígito o letra en la patente reportada.
    """
    ocr = canonicalize(ocr or "")
    if not ocr or ocr_conf < min_conf:
        return PlateCheck.UNREADABLE
    if len(ocr) == len(reported):
        ocr = "".join(
            o.translate(TO_DIGIT if r.isdigit() else TO_LETTER) for o, r in zip(ocr, reported, strict=True)
        )
    distance = Levenshtein.distance(reported, ocr)
    if distance == 0:
        return PlateCheck.MATCH
    if distance <= FUZZY_DISTANCE:
        return PlateCheck.FUZZY
    return PlateCheck.MISMATCH
