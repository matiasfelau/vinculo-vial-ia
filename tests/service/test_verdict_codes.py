"""Un escenario por cada código de veredicto: qué lo produce, de qué color es y si penaliza al emisor.

Es la matriz de las reglas de decisión: un caso RED no termina verde ni amarillo, un caso amarillo no termina
verde, y un caso válido (el feliz) no termina rojo.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from tests.fakes import (
    CONTEXT,
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
    rect,
    vec,
)
from vvi_ia.adapters.imaging import JpegRenderer
from vvi_ia.domain.model import (
    FRAUD_CODES,
    AlertType,
    CarInstance,
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


@dataclass(frozen=True)
class Case:
    """Cómo provocar un código: el tipo de alerta y cómo arreglar los modelos o la evidencia."""

    alert: AlertType
    arrange: Callable[[Scenario, FakeModels], None]
    code: ReasonCode
    verdict: Verdict


def _no_vehicle(_: Scenario, models: FakeModels) -> None:
    models.vehicle.by_marker[GENERAL] = []
    models.cars.by_marker[GENERAL] = []


def _burst_mismatch(_: Scenario, models: FakeModels) -> None:
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(85)


def _plate_mismatch(_: Scenario, models: FakeModels) -> None:
    models.plate.by_marker[PLATE] = ("ZZ999ZZ", 0.95)


def _unreadable(scenario: Scenario, _: FakeModels) -> None:
    scenario.corrupt_sha(EvidenceRole.GENERAL)


def _context_not_detected_damage(_: Scenario, models: FakeModels) -> None:
    models.damage.by_marker[detail_marker(0)] = []


def _context_not_detected_ramp(_: Scenario, models: FakeModels) -> None:
    models.curb.by_marker[CONTEXT] = (rect(80, 86, 0, 100), rect(0, 0, 0, 0))  # sin tramo rebajado


def _redaction_unavailable(_: Scenario, models: FakeModels) -> None:
    models.face.by_marker[GENERAL] = None


def _plate_fuzzy(_: Scenario, models: FakeModels) -> None:
    models.plate.by_marker[PLATE] = (REPORTED_PLATE[:-1] + "9", 0.95)


def _plate_unreadable(_: Scenario, models: FakeModels) -> None:
    models.plate.by_marker[PLATE] = (REPORTED_PLATE, 0.2)


def _burst_uncertain(_: Scenario, models: FakeModels) -> None:
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(55)


def _context_uncertain_damage(_: Scenario, models: FakeModels) -> None:
    models.damage.by_marker[detail_marker(0)] = None  # el detector de daños no responde


def _context_uncertain_ramp(_: Scenario, models: FakeModels) -> None:
    models.cars.by_marker[CONTEXT] = [CarInstance(rect(40, 80, 80, 98)), CarInstance(rect(20, 40, 0, 20))]
    models.embedder.by_marker_area[(CONTEXT, 40 * 18)] = vec(5)  # el auto apenas toca el rebaje


CASES = {
    "NO_VEHICLE": Case(CRASHED, _no_vehicle, R.NO_VEHICLE, Verdict.RED),
    "BURST_MISMATCH": Case(CRASHED, _burst_mismatch, R.BURST_MISMATCH, Verdict.RED),
    "PLATE_MISMATCH": Case(CRASHED, _plate_mismatch, R.PLATE_MISMATCH, Verdict.RED),
    "EVIDENCE_UNREADABLE": Case(CRASHED, _unreadable, R.EVIDENCE_UNREADABLE, Verdict.RED),
    "CONTEXT_NOT_DETECTED (daño)": Case(
        CRASHED, _context_not_detected_damage, R.CONTEXT_NOT_DETECTED, Verdict.RED
    ),
    "CONTEXT_NOT_DETECTED (rampa)": Case(
        GARAGE, _context_not_detected_ramp, R.CONTEXT_NOT_DETECTED, Verdict.RED
    ),
    "REDACTION_UNAVAILABLE": Case(CRASHED, _redaction_unavailable, R.REDACTION_UNAVAILABLE, Verdict.RED),
    "PLATE_FUZZY": Case(CRASHED, _plate_fuzzy, R.PLATE_FUZZY, Verdict.YELLOW),
    "PLATE_UNREADABLE": Case(CRASHED, _plate_unreadable, R.PLATE_UNREADABLE, Verdict.YELLOW),
    "BURST_UNCERTAIN": Case(CRASHED, _burst_uncertain, R.BURST_UNCERTAIN, Verdict.YELLOW),
    "CONTEXT_UNCERTAIN (daño)": Case(CRASHED, _context_uncertain_damage, R.CONTEXT_UNCERTAIN, Verdict.YELLOW),
    "CONTEXT_UNCERTAIN (rampa)": Case(GARAGE, _context_uncertain_ramp, R.CONTEXT_UNCERTAIN, Verdict.YELLOW),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_cada_codigo_se_produce_con_su_color(name: str) -> None:
    case = CASES[name]
    scenario, models = build_scenario(case.alert), happy_models()
    case.arrange(scenario, models)

    result = run(scenario, models)

    assert case.code in result.reason_codes
    assert result.verdict == case.verdict
    assert is_fraud(result.reason_codes) == (case.code in FRAUD_CODES)


def test_todos_los_codigos_tienen_un_escenario() -> None:
    assert {case.code for case in CASES.values()} == set(ReasonCode)


@pytest.mark.parametrize("alert", [CRASHED, GARAGE])
def test_el_caso_valido_no_termina_rojo_ni_amarillo(alert: AlertType) -> None:
    result = run(build_scenario(alert), happy_models())

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c.verdict == Verdict.RED))
def test_un_caso_red_nunca_termina_verde_ni_amarillo(name: str) -> None:
    case = CASES[name]
    scenario, models = build_scenario(case.alert), happy_models()
    case.arrange(scenario, models)

    assert run(scenario, models).verdict == Verdict.RED


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c.verdict == Verdict.YELLOW))
def test_un_caso_yellow_nunca_termina_verde(name: str) -> None:
    case = CASES[name]
    scenario, models = build_scenario(case.alert), happy_models()
    case.arrange(scenario, models)

    assert run(scenario, models).verdict != Verdict.GREEN


def test_red_gana_sobre_yellow_cuando_hay_los_dos() -> None:
    scenario, models = build_scenario(CRASHED), happy_models()
    _plate_fuzzy(scenario, models)
    _plate_mismatch(
        scenario, models
    )  # el último arreglo pisa al anterior: queda un mismatch y una ráfaga dudosa
    _burst_uncertain(scenario, models)

    result = run(scenario, models)

    assert R.BURST_UNCERTAIN in result.reason_codes and result.verdict == Verdict.RED
