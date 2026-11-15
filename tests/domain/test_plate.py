import pytest

from vvi_ia.domain.plate import PlateCheck, canonicalize, has_known_structure, plate_check

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


def test_distancia_uno_por_un_caracter_distinto_es_fuzzy() -> None:
    assert plate_check("AB123CD", "AB123CE", 0.9, MIN_CONF) == PlateCheck.FUZZY


def test_una_lectura_sin_el_largo_de_ningun_formato_es_ilegible_no_fuzzy() -> None:
    """Falta un carácter: no es ABC123 ni AB123CD, así que es duda (ilegible), no una patente casi igual."""
    assert plate_check("AB123CD", "AB123C", 0.9, MIN_CONF) == PlateCheck.UNREADABLE


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
    """Con un carácter de menos no hay posiciones que comparar ni formato: la lectura es ilegible (duda)."""
    assert plate_check("AB012CD", "ABO1CD", 0.9, MIN_CONF) == PlateCheck.UNREADABLE


@pytest.mark.parametrize("text", ["ABC123", "AB123CD", "A8C1Z3", "AB1Z3CD", "ABO123", "AB12SCD"])
def test_has_known_structure_acepta_los_dos_formatos_y_sus_confusiones(text: str) -> None:
    assert has_known_structure(text)


@pytest.mark.parametrize(
    "text",
    ["", "870", "A07", "870EPB", "ABC12", "ABC1234", "AB123CDE", "1234567", "ABCDEFG", "AB-3CD", "ABC12C3"],
)
def test_has_known_structure_rechaza_lo_que_no_es_abc123_ni_ab123cd(text: str) -> None:
    assert not has_known_structure(text)


def test_una_patente_de_dos_lineas_leida_a_medias_no_puede_ser_mismatch() -> None:
    """Moto (formato fuera de alcance): PARSeq lee solo una línea, con confianza alta. Antes salía RED."""
    assert plate_check("870EPB", "870", 0.977, MIN_CONF) == PlateCheck.UNREADABLE
    assert plate_check("A077BEH", "A07", 0.97, MIN_CONF) == PlateCheck.UNREADABLE


def test_una_lectura_completa_de_otra_patente_sigue_siendo_mismatch() -> None:
    assert plate_check("AB123CD", "XY987ZW", 0.95, MIN_CONF) == PlateCheck.MISMATCH
    assert plate_check("ABC123", "XYZ987", 0.95, MIN_CONF) == PlateCheck.MISMATCH
