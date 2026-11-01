import pytest

from vvi_ia.domain.model import FRAUD_CODES, RED_CODES, ReasonCode, Verdict
from vvi_ia.domain.verdict import aggregate, fraud_codes, is_fraud

R = ReasonCode


def test_el_veredicto_es_solo_green_o_red() -> None:
    assert set(Verdict) == {Verdict.GREEN, Verdict.RED}


def test_sin_codigos_es_green() -> None:
    assert aggregate([]) == Verdict.GREEN


@pytest.mark.parametrize("code", sorted(ReasonCode))
def test_cualquier_codigo_da_red(code: ReasonCode) -> None:
    assert aggregate([code]) == Verdict.RED


def test_los_codigos_son_exactamente_los_seis_de_red() -> None:
    assert set(ReasonCode) == {
        R.NO_VEHICLE,
        R.BURST_MISMATCH,
        R.PLATE_MISMATCH,
        R.EVIDENCE_UNREADABLE,
        R.REDACTION_UNAVAILABLE,
        R.VALIDATION_FAILED,
    }
    assert RED_CODES == set(ReasonCode)


@pytest.mark.parametrize(
    "old", ["BURST_UNCERTAIN", "PLATE_FUZZY", "PLATE_UNREADABLE", "CONTEXT_NOT_DETECTED", "CONTEXT_UNCERTAIN"]
)
def test_los_codigos_dudosos_ya_no_existen(old: str) -> None:
    assert old not in ReasonCode.__members__


def test_fraude_son_exactamente_tres_codigos() -> None:
    assert FRAUD_CODES == {R.NO_VEHICLE, R.BURST_MISMATCH, R.PLATE_MISMATCH}
    assert FRAUD_CODES <= RED_CODES


@pytest.mark.parametrize("code", [R.EVIDENCE_UNREADABLE, R.REDACTION_UNAVAILABLE, R.VALIDATION_FAILED])
def test_los_rojos_que_no_son_fraude_no_penalizan(code: ReasonCode) -> None:
    assert aggregate([code]) == Verdict.RED
    assert not is_fraud([code])


@pytest.mark.parametrize("code", sorted(FRAUD_CODES))
def test_cada_codigo_de_fraude_penaliza(code: ReasonCode) -> None:
    assert is_fraud([R.REDACTION_UNAVAILABLE, code])
    assert fraud_codes([R.REDACTION_UNAVAILABLE, code]) == {code}
