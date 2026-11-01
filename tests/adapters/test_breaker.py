import pytest

from vvi_ia.adapters.breaker import CircuitBreaker, CircuitOpenError, CircuitState
from vvi_ia.service.ports import TransientError


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Upstream5xx(Exception):
    """Falla que cuenta para el breaker."""


class Upstream4xx(Exception):
    """Falla del invocante: no cuenta."""


def _counts(exc: Exception) -> bool:
    return isinstance(exc, Upstream5xx)


def _breaker(
    clock: Clock, threshold: int = 3, window: float = 60, open_for: float = 30, probes: int = 1
) -> CircuitBreaker:
    return CircuitBreaker("s3", threshold, window, open_for, probes, clock=clock)


def _fail(breaker: CircuitBreaker, exc: Exception) -> None:
    def operation() -> None:
        raise exc

    with pytest.raises(type(exc)):
        breaker.call(operation, _counts)


def _ok(breaker: CircuitBreaker) -> str:
    return breaker.call(lambda: "ok", _counts)


def test_cerrado_deja_pasar_y_devuelve_el_resultado() -> None:
    breaker = _breaker(Clock())

    assert _ok(breaker) == "ok"
    assert breaker.state == CircuitState.CLOSED


def test_abre_al_alcanzar_el_umbral_de_fallas() -> None:
    breaker = _breaker(Clock(), threshold=3)

    for _ in range(2):
        _fail(breaker, Upstream5xx())
    assert breaker.state == CircuitState.CLOSED
    _fail(breaker, Upstream5xx())

    assert breaker.state == CircuitState.OPEN


def test_con_el_circuito_abierto_se_rechaza_sin_ejecutar_la_operacion() -> None:
    breaker = _breaker(Clock(), threshold=1)
    _fail(breaker, Upstream5xx())
    calls: list[int] = []

    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: calls.append(1), _counts)

    assert calls == []


def test_circuit_open_error_es_transitorio() -> None:
    assert issubclass(CircuitOpenError, TransientError)


def test_los_errores_que_no_cuentan_no_abren_el_circuito() -> None:
    breaker = _breaker(Clock(), threshold=2)

    for _ in range(10):
        _fail(breaker, Upstream4xx())

    assert breaker.state == CircuitState.CLOSED


def test_las_fallas_fuera_de_la_ventana_no_se_acumulan() -> None:
    clock = Clock()
    breaker = _breaker(clock, threshold=2, window=10)

    _fail(breaker, Upstream5xx())
    clock.now += 11
    _fail(breaker, Upstream5xx())

    assert breaker.state == CircuitState.CLOSED


def test_pasado_el_tiempo_de_apertura_pasa_a_semiabierto_y_un_exito_cierra() -> None:
    clock = Clock()
    breaker = _breaker(clock, threshold=1, open_for=30)
    _fail(breaker, Upstream5xx())
    clock.now += 29
    assert breaker.state == CircuitState.OPEN

    clock.now += 1
    assert breaker.state == CircuitState.HALF_OPEN
    assert _ok(breaker) == "ok"

    assert breaker.state == CircuitState.CLOSED


def test_una_falla_en_el_sondeo_reabre_el_circuito() -> None:
    clock = Clock()
    breaker = _breaker(clock, threshold=1, open_for=30)
    _fail(breaker, Upstream5xx())
    clock.now += 30

    _fail(breaker, Upstream5xx())

    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        _ok(breaker)


def test_un_error_que_no_cuenta_en_el_sondeo_cierra_el_circuito() -> None:
    clock = Clock()
    breaker = _breaker(clock, threshold=1, open_for=30)
    _fail(breaker, Upstream5xx())
    clock.now += 30

    _fail(breaker, Upstream4xx())

    assert breaker.state == CircuitState.CLOSED


def test_semiabierto_limita_los_sondeos_concurrentes() -> None:
    clock = Clock()
    breaker = _breaker(clock, threshold=1, open_for=30, probes=1)
    _fail(breaker, Upstream5xx())
    clock.now += 30
    results: list[str] = []

    def probe() -> None:
        # mientras el sondeo está en vuelo, otra llamada es rechazada
        with pytest.raises(CircuitOpenError):
            _ok(breaker)
        results.append("rechazada")

    breaker.call(probe, _counts)

    assert results == ["rechazada"]
    assert breaker.state == CircuitState.CLOSED


def test_las_transiciones_se_loguean(caplog: pytest.LogCaptureFixture) -> None:
    breaker = _breaker(Clock(), threshold=1)

    with caplog.at_level("WARNING"):
        _fail(breaker, Upstream5xx())

    record = next(r for r in caplog.records if r.getMessage() == "circuit breaker cambió de estado")
    assert record.__dict__["breaker"] == "s3"
    assert (record.__dict__["from_state"], record.__dict__["to_state"]) == ("closed", "open")
