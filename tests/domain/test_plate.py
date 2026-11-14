import pytest

from vvi_ia.domain.plate import PlateCheck, canonicalize, plate_check

MIN_CONF = 0.6


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("ab 123-cd", "AB123CD"), ("  a.b/1\n2 ", "AB12"), ("", ""), ("---", "")],
)
def test_canonicalize(raw: str, expected: str) -> None:
    assert canonicalize(raw) == expected


def test_coincidencia_exacta() -> None:
    assert plate_check("AB123CD", "AB123CD", 0.9, MIN_CONF) == PlateCheck.MATCH


def test_canonicaliza_el_texto_del_ocr() -> None:
    assert plate_check("AB123CD", "ab-123 cd", 0.9, MIN_CONF) == PlateCheck.MATCH


@pytest.mark.parametrize(
    ("reported", "ocr"),
    [
        ("AB012CD", "ABO12CD"),  # O leída donde va un 0
        ("AO123CD", "A0123CD"),  # 0 leído donde va una O
        ("AB123CD", "ABI23CD"),  # I leída donde va un 1
        ("AB123CD", "A8123CD"),  # 8 leído donde va una B
        ("AB812CD", "ABB12CD"),  # B leída donde va un 8
        ("AB123CD", "AB1Z3CD"),  # Z leída donde va un 2
    ],
)
def test_confusiones_de_ocr_se_corrigen_segun_la_posicion(reported: str, ocr: str) -> None:
    assert plate_check(reported, ocr, 0.9, MIN_CONF) == PlateCheck.MATCH


def test_confusion_sin_correccion_posible_queda_fuzzy() -> None:
    # TO_LETTER no mapea 0 → D: la diferencia de un carácter queda como FUZZY
    assert plate_check("AB123CD", "AB123C0", 0.9, MIN_CONF) == PlateCheck.FUZZY


def test_distancia_uno_es_fuzzy() -> None:
    assert plate_check("AB123CD", "AB129CD", 0.9, MIN_CONF) == PlateCheck.FUZZY


def test_distancia_uno_por_largo_distinto_es_fuzzy() -> None:
    assert plate_check("AB123CD", "AB123C", 0.9, MIN_CONF) == PlateCheck.FUZZY


def test_distancia_dos_es_mismatch() -> None:
    assert plate_check("AB123CD", "AB199CD", 0.9, MIN_CONF) == PlateCheck.MISMATCH


def test_patente_totalmente_distinta_es_mismatch() -> None:
    assert plate_check("AB123CD", "ZZ999ZZ", 0.9, MIN_CONF) == PlateCheck.MISMATCH


def test_confianza_bajo_el_umbral_es_ilegible() -> None:
    assert plate_check("AB123CD", "AB123CD", 0.59, MIN_CONF) == PlateCheck.UNREADABLE


def test_confianza_en_el_umbral_se_acepta() -> None:
    assert plate_check("AB123CD", "AB123CD", MIN_CONF, MIN_CONF) == PlateCheck.MATCH


@pytest.mark.parametrize("ocr", [None, "", "  --  "])
def test_ocr_vacio_es_ilegible(ocr: str | None) -> None:
    assert plate_check("AB123CD", ocr, 0.99, MIN_CONF) == PlateCheck.UNREADABLE


@pytest.mark.parametrize(
    ("reported", "ocr"),
    [
        ("ABC123", "A8C123"),  # formato anterior: 8 donde va una B
        ("ABC123", "ABCI23"),  # I donde va un 1
        ("ABC123", "ABC1Z3"),  # Z donde va un 2
        ("ABC125", "ABC12S"),  # S donde va un 5
        ("ABC126", "ABC12G"),  # G donde va un 6
        ("ABC120", "ABC12O"),  # O donde va un 0
        ("AB125CD", "AB12SCD"),  # Mercosur: S donde va un 5
        ("AB126CD", "AB12GCD"),  # Mercosur: G donde va un 6
        ("AG125CD", "A6125CD"),  # Mercosur: 6 donde va una G
        ("S5C123", "5SC123"),  # el mismo par intercambiado, cada uno según su posición
    ],
)
def test_la_normalizacion_por_posicion_vale_para_los_dos_formatos(reported: str, ocr: str) -> None:
    assert plate_check(reported, ocr, 0.9, MIN_CONF) == PlateCheck.MATCH


def test_la_normalizacion_necesita_el_mismo_largo() -> None:
    """Con un carácter de menos no hay posiciones que comparar: la confusión cuenta como otra diferencia."""
    assert plate_check("AB012CD", "ABO1CD", 0.9, MIN_CONF) == PlateCheck.MISMATCH
