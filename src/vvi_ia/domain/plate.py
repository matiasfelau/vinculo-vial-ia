"""Canonicalización y comparación de patentes."""

import re
from enum import StrEnum

from rapidfuzz.distance import Levenshtein

TO_DIGIT = str.maketrans("OQDIZSGB", "00012568")
TO_LETTER = str.maketrans("012568", "OIZSGB")

FUZZY_DISTANCE = 1
# Formatos en alcance: ABC123 (anterior) y AB123CD (Mercosur). L = letra, D = dígito.
SHAPES = ("LLLDDD", "LLDDDLL")
_AS_LETTER = frozenset("012568")  # dígitos que el OCR confunde con una letra
_AS_DIGIT = frozenset("OQDIZSGB")  # letras que el OCR confunde con un dígito
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


def _fits(text: str, shape: str) -> bool:
    """¿`text` puede ser una patente de esa forma, tolerando las confusiones letra/dígito del OCR?"""
    if len(text) != len(shape):
        return False
    for char, kind in zip(text, shape, strict=True):
        if kind == "L":
            ok = char.isalpha() or char in _AS_LETTER
        else:
            ok = char.isdigit() or char in _AS_DIGIT
        if not ok:
            return False
    return True


def has_known_structure(text: str) -> bool:
    """¿La lectura tiene el largo y la estructura de algún formato en alcance (ABC123 o AB123CD)?

    Tolera las confusiones típicas: en una posición de letra acepta un dígito que se confunde con una letra
    (0, 1, 2, 5, 6, 8), y en una de dígito, una letra que se confunde con un dígito (O, Q, D, I, Z, S, G, B).
    """
    return any(_fits(canonicalize(text), shape) for shape in SHAPES)


def plate_check(reported: str, ocr: str | None, ocr_conf: float, min_conf: float) -> PlateCheck:
    """Compara la patente reportada (canónica) con el texto del OCR.

    Si el OCR tiene el mismo largo, corrige las confusiones típicas (O/0, I/1, B/8...) según
    la posición: dígito o letra en la patente reportada. Una lectura que no tiene el largo ni la
    estructura de ningún formato en alcance (ABC123 o AB123CD) es ilegible: duda, nunca una patente
    distinta (así una patente de dos líneas leída a medias no puede dar `MISMATCH`).
    """
    ocr = canonicalize(ocr or "")
    if not ocr or ocr_conf < min_conf or not has_known_structure(ocr):
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
