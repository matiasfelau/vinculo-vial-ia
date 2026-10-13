"""Un escenario por cada código de veredicto y por cada caso que ya no pesa: el veredicto es solo GREEN o RED.

RED únicamente con evidencia firme (R1 a R4); lo dudoso o lo que no se puede comparar es GREEN (R5).
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from tests.fakes import (
    GENERAL,
    PLATE,
    REDACTION,
    REPORTED_PLATE,
    THRESHOLDS,
    FakeModels,
    FakeWriter,
    Scenario,
    build_scenario,
    detail_marker,
    happy_models,
    vec,
)
from vvi_ia.adapters.imaging import JpegRenderer
from vvi_ia.domain.model import (
    FRAUD_CODES,
    AlertType,
    EvidenceRole,
    ReasonCode,
    ValidationResult,
    Verdict,
)
from vvi_ia.domain.verdict import is_fraud
from vvi_ia.service.validate_evidence import ValidateEvidence

R = ReasonCode
GARAGE = AlertType.PARKED_ON_GARAGE
CRASHED = AlertType.CRASHED


def run(scenario: Scenario, models: FakeModels) -> ValidationResult:
    use_case = ValidateEvidence(
        store=scenario.store,
        writer=FakeWriter(),
        decryptor=scenario.decryptor,
        decoder=scenario.decoder,
        renderer=JpegRenderer(),
        models=models.as_models(),
        thresholds=THRESHOLDS,
        redaction=REDACTION,
    )
    return use_case.execute(scenario.payload)


Arrange = Callable[[Scenario, FakeModels], None]


@dataclass(frozen=True)
class Red:
    alert: AlertType
    arrange: Arrange
    code: ReasonCode


def _no_vehicle(_: Scenario, models: FakeModels) -> None:
    models.vehicle.by_marker[GENERAL] = []
    models.cars.by_marker[GENERAL] = []


def _burst_mismatch(_: Scenario, models: FakeModels) -> None:
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(85)


def _plate_mismatch(_: Scenario, models: FakeModels) -> None:
    models.plate.by_marker[PLATE] = ("ZZ999ZZ", 0.95)


def _unreadable(scenario: Scenario, _: FakeModels) -> None:
    scenario.corrupt_sha(EvidenceRole.GENERAL)


def _redaction_unavailable(_: Scenario, models: FakeModels) -> None:
    models.face.by_marker[GENERAL] = None


RED = {
    "NO_VEHICLE": Red(CRASHED, _no_vehicle, R.NO_VEHICLE),
    "NO_VEHICLE (garage)": Red(GARAGE, _no_vehicle, R.NO_VEHICLE),
    "BURST_MISMATCH": Red(CRASHED, _burst_mismatch, R.BURST_MISMATCH),
    "PLATE_MISMATCH": Red(CRASHED, _plate_mismatch, R.PLATE_MISMATCH),
    "EVIDENCE_UNREADABLE": Red(CRASHED, _unreadable, R.EVIDENCE_UNREADABLE),
    "REDACTION_UNAVAILABLE": Red(CRASHED, _redaction_unavailable, R.REDACTION_UNAVAILABLE),
}


@pytest.mark.parametrize("name", sorted(RED))
def test_cada_codigo_rojo_se_produce_y_penaliza_solo_si_es_fraude(name: str) -> None:
    case = RED[name]
    scenario, models = build_scenario(case.alert), happy_models()
    case.arrange(scenario, models)

    result = run(scenario, models)

    assert result.reason_codes == (case.code,)
    assert result.verdict == Verdict.RED  # un caso RED nunca termina GREEN (regla A)
    assert is_fraud(result.reason_codes) == (case.code in FRAUD_CODES)


def test_todos_los_codigos_tienen_un_escenario() -> None:
    assert {case.code for case in RED.values()} == set(ReasonCode)


# --- R5: lo dudoso o lo que no se puede comparar es GREEN ---------------------------------------------


def _plate_fuzzy(_: Scenario, models: FakeModels) -> None:
    models.plate.by_marker[PLATE] = (REPORTED_PLATE[:-1] + "9", 0.95)


def _plate_unreadable(_: Scenario, models: FakeModels) -> None:
    models.plate.by_marker[PLATE] = (REPORTED_PLATE, 0.2)


def _plate_without_reading(_: Scenario, models: FakeModels) -> None:
    models.plate.by_marker[PLATE] = None


def _burst_in_between(_: Scenario, models: FakeModels) -> None:
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(55)


def _burst_not_comparable(_: Scenario, models: FakeModels) -> None:
    models.embedder.by_marker_area[(PLATE, 6400)] = None


def _no_damage(_: Scenario, models: FakeModels) -> None:
    models.damage.by_marker[detail_marker(0)] = []


def _no_curb_model(_: Scenario, models: FakeModels) -> None:
    models.curb.by_marker.clear()


GREEN = {
    "PLATE_FUZZY": _plate_fuzzy,
    "PLATE_UNREADABLE": _plate_unreadable,
    "OCR sin lectura": _plate_without_reading,
    "BURST_UNCERTAIN": _burst_in_between,
    "ráfaga que no se puede comparar": _burst_not_comparable,
    "daño ausente": _no_damage,
    "contexto sin modelo de rebajes": _no_curb_model,
}


@pytest.mark.parametrize("name", sorted(GREEN))
@pytest.mark.parametrize("alert", [CRASHED, GARAGE])
def test_lo_dudoso_o_incomparable_es_green_sin_codigos(name: str, alert: AlertType) -> None:
    scenario, models = build_scenario(alert), happy_models()
    GREEN[name](scenario, models)

    result = run(scenario, models)

    assert (
        result.verdict == Verdict.GREEN and result.reason_codes == ()
    )  # regla C: el caso válido no termina rojo
    assert len(result.redacted) == 3


@pytest.mark.parametrize("alert", [CRASHED, GARAGE])
def test_el_caso_valido_es_green(alert: AlertType) -> None:
    result = run(build_scenario(alert), happy_models())

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()


def test_nunca_hay_yellow() -> None:
    assert {verdict.value for verdict in Verdict} == {"GREEN", "RED"}
    for name, arrange in GREEN.items():
        scenario, models = build_scenario(CRASHED), happy_models()
        arrange(scenario, models)

        assert run(scenario, models).verdict in (Verdict.GREEN, Verdict.RED), name


def test_varios_rojos_se_acumulan_sin_perder_el_fraude() -> None:
    scenario, models = build_scenario(CRASHED), happy_models()
    _plate_mismatch(scenario, models)
    _burst_mismatch(scenario, models)

    result = run(scenario, models)

    assert set(result.reason_codes) == {R.PLATE_MISMATCH, R.BURST_MISMATCH}
    assert result.verdict == Verdict.RED and is_fraud(result.reason_codes)
