import pytest

from vvi_ia.domain.model import FRAUD_CODES, RED_CODES, YELLOW_CODES, ReasonCode, Verdict
from vvi_ia.domain.verdict import aggregate, fraud_codes, is_fraud

R = ReasonCode


def test_sin_codigos_es_green() -> None:
    assert aggregate([]) == Verdict.GREEN


@pytest.mark.parametrize("code", sorted(YELLOW_CODES))
def test_un_codigo_yellow_da_yellow(code: ReasonCode) -> None:
    assert aggregate([code]) == Verdict.YELLOW


@pytest.mark.parametrize("code", sorted(RED_CODES))
def test_un_codigo_red_da_red(code: ReasonCode) -> None:
    assert aggregate([code]) == Verdict.RED


def test_red_tiene_precedencia_sobre_yellow() -> None:
    assert aggregate([R.PLATE_FUZZY, R.BURST_UNCERTAIN, R.NO_VEHICLE]) == Verdict.RED


def test_evidence_unreadable_es_red_aunque_no_sea_fraude() -> None:
    assert aggregate([R.EVIDENCE_UNREADABLE]) == Verdict.RED
    assert not is_fraud([R.EVIDENCE_UNREADABLE])


def test_los_codigos_se_reparten_en_red_y_yellow_sin_solaparse() -> None:
    assert RED_CODES | YELLOW_CODES == set(ReasonCode)
    assert not RED_CODES & YELLOW_CODES


def test_el_mapeo_de_los_bloques_del_veredicto() -> None:
    assert {R.NO_VEHICLE, R.PLATE_MISMATCH, R.BURST_MISMATCH} == FRAUD_CODES  # RED por fraude
    assert {R.CONTEXT_NOT_DETECTED, R.EVIDENCE_UNREADABLE, R.REDACTION_UNAVAILABLE} == RED_CODES - FRAUD_CODES
    assert YELLOW_CODES == {R.PLATE_FUZZY, R.PLATE_UNREADABLE, R.BURST_UNCERTAIN, R.CONTEXT_UNCERTAIN}


@pytest.mark.parametrize("code", [R.CONTEXT_NOT_DETECTED, R.EVIDENCE_UNREADABLE, R.REDACTION_UNAVAILABLE])
def test_los_rojos_que_no_son_fraude_no_penalizan(code: ReasonCode) -> None:
    assert aggregate([code]) == Verdict.RED
    assert not is_fraud([code])


def test_fraude_son_exactamente_tres_codigos() -> None:
    assert FRAUD_CODES == {R.NO_VEHICLE, R.BURST_MISMATCH, R.PLATE_MISMATCH}
    assert FRAUD_CODES <= RED_CODES


@pytest.mark.parametrize("code", sorted(FRAUD_CODES))
def test_cada_codigo_de_fraude_penaliza(code: ReasonCode) -> None:
    assert is_fraud([R.PLATE_FUZZY, code])
    assert fraud_codes([R.PLATE_FUZZY, code]) == {code}


@pytest.mark.parametrize("code", sorted(set(ReasonCode) - FRAUD_CODES))
def test_el_resto_no_penaliza(code: ReasonCode) -> None:
    assert not is_fraud([code])
