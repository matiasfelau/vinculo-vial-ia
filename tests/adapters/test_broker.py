import asyncio
import json
import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from aio_pika import DeliveryMode, Message
from aiormq.exceptions import ChannelInvalidStateError, DeliveryError
from botocore.exceptions import ClientError

from tests.fakes import REDACTION, THRESHOLDS, FakeWriter, build_scenario, happy_models
from vvi_ia.adapters.breaker import CircuitBreaker
from vvi_ia.adapters.broker import (
    DISCARD_REASON_HEADER,
    EVENT_ID_NAMESPACE,
    EVENT_TYPE_IN,
    AmqpPublisher,
    EvidenceMessageHandler,
    InvalidEnvelopeError,
    PublishFailedError,
    build_validated_event,
    build_validation_failed_event,
    death_count,
    parse_envelope,
)
from vvi_ia.adapters.imaging import JpegRenderer
from vvi_ia.adapters.storage_s3 import S3EvidenceStore
from vvi_ia.domain.model import (
    EvidenceRole,
    ReasonCode,
    RedactedRef,
    Scores,
    ValidationResult,
    Verdict,
)
from vvi_ia.domain.verdict import is_fraud
from vvi_ia.logging_setup import JsonFormatter
from vvi_ia.service.payload import EvidenceUploaded
from vvi_ia.service.ports import ObjectNotFoundError, SystemDecryptionError, TransientError
from vvi_ia.service.validate_evidence import ValidateEvidence

QUEUE = "ia.alert-evidence-uploaded.q"
MAX_RETRIES = 3
NOW = datetime(2026, 10, 7, 12, 0, 0, tzinfo=UTC)


# --- fakes -----------------------------------------------------------------------------


class FakeMessage:
    def __init__(
        self, body: bytes, headers: Mapping[str, Any] | None = None, correlation_id: str | None = None
    ):
        self.body = body
        self.headers: Mapping[str, Any] = headers or {}
        self.correlation_id = correlation_id
        self.log: list[str] = []
        self.nack_requeue: bool | None = None

    async def ack(self) -> None:
        self.log.append("ack")

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None:
        self.log.append("nack")
        self.nack_requeue = requeue


class FakePublisher:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.events: list[tuple[str, bytes, str, str | None]] = []
        self.dlq: list[tuple[str, bytes, str]] = []
        self.fail_events = False
        self.event_failures = 0  # las próximas N publicaciones de eventos no se confirman
        self.event_error: Exception | None = None  # una excepción que no es PublishFailedError
        self.fail_dlq = False

    async def publish_event(
        self, routing_key: str, body: bytes, message_id: str, correlation_id: str | None
    ) -> None:
        if self.event_error is not None:
            raise self.event_error
        if self.fail_events or self.event_failures > 0:
            self.event_failures = max(0, self.event_failures - 1)
            raise PublishFailedError("sin confirm")
        self.order.append("publish_event")
        self.events.append((routing_key, body, message_id, correlation_id))

    async def publish_dlq(self, queue: str, body: bytes, reason: str) -> None:
        if self.fail_dlq:
            raise PublishFailedError("sin confirm")
        self.order.append("publish_dlq")
        self.dlq.append((queue, body, reason))


class FakeUseCase:
    def __init__(self, result: ValidationResult | Exception) -> None:
        self.result = result
        self.calls: list[EvidenceUploaded] = []
        self.thread_ids: list[int] = []

    def execute(self, payload: EvidenceUploaded) -> ValidationResult:
        self.calls.append(payload)
        self.thread_ids.append(threading.get_ident())
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _result(
    verdict: Verdict = Verdict.RED, codes: tuple[ReasonCode, ...] = (ReasonCode.PLATE_MISMATCH,)
) -> ValidationResult:
    return ValidationResult(
        verdict=verdict,
        reason_codes=codes,
        scores=Scores(burst_similarity=0.91, ocr_confidence=0.93, context_confidence=None),
        model_versions={"detector": "null"},
    )


def _envelope_body(payload: EvidenceUploaded, event_id: uuid.UUID | None = None, **overrides: Any) -> bytes:
    envelope: dict[str, Any] = {
        "eventId": str(event_id or uuid.uuid4()),
        "eventType": EVENT_TYPE_IN,
        "timestamp": "2026-10-07T12:00:00Z",
        "producer": "vinculo-vial-alert",
        "actor": None,
        "payload": json.loads(payload.model_dump_json()),
    }
    envelope.update(overrides)
    return json.dumps(envelope).encode()


class Rig:
    def __init__(self, result: ValidationResult | Exception | None = None) -> None:
        self.order: list[str] = []
        self.publisher = FakePublisher(self.order)
        self.use_case = FakeUseCase(result if result is not None else _result())
        self.scenario = build_scenario()
        self.payload = self.scenario.payload
        self.handler = EvidenceMessageHandler(
            queue=QUEUE,
            max_retries=MAX_RETRIES,
            bucket=self.payload.bucket,
            use_case=self.use_case,
            publisher=self.publisher,
            run_blocking=self._run_blocking,
            clock=lambda: NOW,
        )

    async def _run_blocking(self, operation: Callable[[], ValidationResult]) -> ValidationResult:
        return await asyncio.to_thread(operation)

    def message(self, body: bytes | None = None, **kwargs: Any) -> FakeMessage:
        msg = FakeMessage(body if body is not None else _envelope_body(self.payload), **kwargs)
        original_ack, original_nack = msg.ack, msg.nack

        async def ack() -> None:
            self.order.append("ack")
            await original_ack()

        async def nack(multiple: bool = False, requeue: bool = True) -> None:
            self.order.append("nack")
            await original_nack(multiple, requeue)

        msg.ack = ack  # type: ignore[method-assign]
        msg.nack = nack  # type: ignore[method-assign]
        return msg


def _x_death(count: int, queue: str = QUEUE) -> dict[str, Any]:
    return {
        "x-death": [{"queue": queue, "count": count, "reason": "rejected", "exchange": "vvi.topic.exchange"}]
    }


def _causation(msg: FakeMessage) -> uuid.UUID:
    return uuid.UUID(json.loads(msg.body)["eventId"])


def _tampered(
    rig: Rig, mutate: Callable[[dict[str, Any]], object], event_id: uuid.UUID | None = None
) -> bytes:
    """El evento válido con el payload alterado por `mutate`: el sobre sigue siendo válido."""
    envelope = json.loads(_envelope_body(rig.payload, event_id))
    mutate(envelope["payload"])
    return json.dumps(envelope).encode()


def _only_validation_failed(rig: Rig, causation: uuid.UUID) -> dict[str, Any]:
    """El único evento publicado es el RED `VALIDATION_FAILED` de esta alerta y no lleva nada más."""
    ((routing_key, body, message_id, _),) = rig.publisher.events
    event: dict[str, Any] = json.loads(body)
    assert routing_key == "ia.evidence.validated.red"
    assert message_id == event["eventId"] == str(uuid.uuid5(EVENT_ID_NAMESPACE, str(causation)))
    assert event["eventType"] == "ia.evidence.validated"
    assert event["actor"] is None
    assert event["payload"] == {
        "alert_id": rig.payload.alert_id,
        "causation_event_id": str(causation),
        "verdict": "RED",
        "reason_codes": ["VALIDATION_FAILED"],
        "scores": {},
        "model_versions": {},
        "redacted": [],
    }
    assert not is_fraud(ReasonCode(code) for code in event["payload"]["reason_codes"])
    return event


# --- camino feliz: ack solo después del confirm ----------------------------------------


async def test_publica_con_confirm_y_recien_despues_hace_ack() -> None:
    rig = Rig()
    msg = rig.message(correlation_id="corr-1")

    await rig.handler.handle(msg)

    assert rig.order == ["publish_event", "ack"]
    routing_key, body, message_id, correlation_id = rig.publisher.events[0]
    assert routing_key == "ia.evidence.validated.red"
    assert correlation_id == "corr-1"
    assert json.loads(body)["eventId"] == message_id


@pytest.mark.parametrize("verdict", [Verdict.GREEN, Verdict.RED])
async def test_la_routing_key_lleva_el_veredicto_en_minusculas(verdict: Verdict) -> None:
    rig = Rig(_result(verdict, ()))

    await rig.handler.handle(rig.message())

    assert rig.publisher.events[0][0] == f"ia.evidence.validated.{verdict.value.lower()}"


async def test_el_evento_de_salida_respeta_el_contrato() -> None:
    result = ValidationResult(
        verdict=Verdict.RED,
        reason_codes=(ReasonCode.PLATE_MISMATCH,),
        scores=Scores(burst_similarity=0.91, ocr_confidence=0.93, context_confidence=None),
        model_versions={"detector": "d1", "ocr": "o1"},
        redacted=(RedactedRef(EvidenceRole.GENERAL, 0, "alerts/x/redacted/general-0.jpg"),),
    )
    rig = Rig(result)
    event_id = uuid.uuid4()

    await rig.handler.handle(rig.message(_envelope_body(rig.payload, event_id)))

    event = json.loads(rig.publisher.events[0][1])
    assert set(event) == {"eventId", "eventType", "timestamp", "producer", "actor", "payload"}
    assert event["eventType"] == "ia.evidence.validated"
    assert event["actor"] is None
    assert event["timestamp"] == "2026-10-07T12:00:00Z"
    assert event["payload"] == {
        "alert_id": rig.payload.alert_id,
        "causation_event_id": str(event_id),
        "verdict": "RED",
        "reason_codes": ["PLATE_MISMATCH"],
        "scores": {"burst_similarity": 0.91, "ocr_confidence": 0.93, "context_confidence": None},
        "model_versions": {"detector": "d1", "ocr": "o1"},
        "redacted": [{"role": "GENERAL", "index": 0, "key": "alerts/x/redacted/general-0.jpg"}],
    }


async def test_reprocesar_el_mismo_evento_emite_el_mismo_event_id() -> None:
    rig = Rig()
    body = _envelope_body(rig.payload)

    await rig.handler.handle(rig.message(body))
    await rig.handler.handle(rig.message(body))

    first, second = rig.publisher.events
    assert json.loads(first[1])["eventId"] == json.loads(second[1])["eventId"]
    assert first[1] == second[1]


def test_event_id_distinto_para_causas_distintas() -> None:
    rig = Rig()
    _, body_a, id_a = build_validated_event(rig.payload, _result(), uuid.uuid4(), NOW)
    _, body_b, id_b = build_validated_event(rig.payload, _result(), uuid.uuid4(), NOW)

    assert id_a != id_b
    assert body_a != body_b


async def test_la_cascada_corre_fuera_del_event_loop() -> None:
    rig = Rig()

    await rig.handler.handle(rig.message())

    assert rig.use_case.thread_ids[0] != threading.get_ident()


async def test_sin_correlation_id_se_genera_uno_y_se_propaga() -> None:
    rig = Rig()

    await rig.handler.handle(rig.message())

    assert uuid.UUID(rig.publisher.events[0][3] or "")


# --- errores transitorios y reintentos -------------------------------------------------


async def test_publicacion_sin_confirm_hace_nack_sin_requeue_y_no_ack() -> None:
    rig = Rig()
    rig.publisher.fail_events = True
    msg = rig.message()

    await rig.handler.handle(msg)

    assert msg.log == ["nack"]
    assert msg.nack_requeue is False


async def test_error_transitorio_hace_nack_sin_requeue() -> None:
    rig = Rig(TransientError("s3 caído"))
    msg = rig.message(headers=_x_death(1))

    await rig.handler.handle(msg)

    assert msg.log == ["nack"]
    assert msg.nack_requeue is False
    assert rig.publisher.dlq == []


async def test_al_superar_max_retries_va_a_la_dlq_con_la_cola_de_trabajo_y_luego_ack() -> None:
    rig = Rig(TransientError("s3 caído"))
    msg = rig.message(headers=_x_death(MAX_RETRIES))

    await rig.handler.handle(msg)

    assert rig.order == ["publish_event", "publish_dlq", "ack"]  # primero el aviso de validación fallida
    _only_validation_failed(rig, _causation(msg))
    queue, body, reason = rig.publisher.dlq[0]
    assert queue == QUEUE  # routing key = nombre de la cola de trabajo, no la del evento
    assert body == msg.body
    assert reason.startswith("max_retries_exceeded")


async def test_por_debajo_de_max_retries_todavia_reintenta() -> None:
    rig = Rig(TransientError("s3 caído"))
    msg = rig.message(headers=_x_death(MAX_RETRIES - 1))

    await rig.handler.handle(msg)

    assert msg.log == ["nack"]


async def test_publicacion_fallida_tras_max_retries_tambien_va_a_la_dlq() -> None:
    rig = Rig()
    rig.publisher.fail_events = True
    msg = rig.message(headers=_x_death(MAX_RETRIES))

    await rig.handler.handle(msg)

    assert rig.order == ["publish_dlq", "ack"]


async def test_si_la_dlq_no_confirma_no_se_hace_ack() -> None:
    rig = Rig(TransientError("s3 caído"))
    rig.publisher.fail_dlq = True
    msg = rig.message(headers=_x_death(MAX_RETRIES))

    await rig.handler.handle(msg)

    assert msg.log == ["nack"]


# --- mensajes inválidos: directo a la DLQ -----------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"no es json",
        b"[]",
        b'{"eventId": "no-uuid"}',
        json.dumps(
            {
                "eventId": str(uuid.uuid4()),
                "eventType": "otro.evento",
                "timestamp": "2026-10-07T12:00:00Z",
                "producer": "p",
                "actor": None,
                "payload": {},
            }
        ).encode(),
        json.dumps(
            {
                "eventId": str(uuid.uuid4()),
                "eventType": EVENT_TYPE_IN,
                "timestamp": "2026-10-07T12:00:00Z",
                "producer": "p",
                "actor": None,
                "payload": {"alert_id": "x"},
            }
        ).encode(),
    ],
)
async def test_mensaje_invalido_va_directo_a_la_dlq_sin_reintento_ni_cascada(body: bytes) -> None:
    rig = Rig()
    msg = rig.message(body)

    await rig.handler.handle(msg)

    assert rig.order == ["publish_dlq", "ack"]
    assert rig.publisher.dlq[0][0] == QUEUE
    assert rig.publisher.dlq[0][1] == body
    assert rig.use_case.calls == []
    assert rig.publisher.events == []


async def test_payload_de_otro_bucket_va_a_la_dlq() -> None:
    rig = Rig()
    body = _envelope_body(rig.payload)
    tampered = json.loads(body)
    tampered["payload"]["bucket"] = "otro-bucket"

    await rig.handler.handle(rig.message(json.dumps(tampered).encode()))

    assert rig.order == ["publish_event", "publish_dlq", "ack"]  # el alert_id es legible: se avisa antes
    assert rig.use_case.calls == []


async def test_el_motivo_de_descarte_no_incluye_valores_del_mensaje() -> None:
    rig = Rig()
    body = _envelope_body(rig.payload)
    tampered = json.loads(body)
    tampered["payload"]["pat_enc_ia"] = "valor-secreto-no-base64!!"

    await rig.handler.handle(rig.message(json.dumps(tampered).encode()))

    assert "valor-secreto" not in rig.publisher.dlq[0][2]


async def test_dlq_de_mensaje_invalido_que_no_confirma_hace_nack() -> None:
    rig = Rig()
    rig.publisher.fail_dlq = True
    msg = rig.message(b"basura")

    await rig.handler.handle(msg)

    assert msg.log == ["nack"]


# --- errores del sistema y panics ------------------------------------------------------


@pytest.mark.parametrize("error", [SystemDecryptionError("x"), ObjectNotFoundError("x")])
async def test_error_del_sistema_va_a_la_dlq_sin_reintento(error: Exception) -> None:
    rig = Rig(error)
    msg = rig.message(headers=_x_death(0))

    await rig.handler.handle(msg)

    assert rig.order == ["publish_event", "publish_dlq", "ack"]
    assert rig.publisher.dlq[0][2] == f"system_error:{type(error).__name__}"
    _only_validation_failed(rig, _causation(msg))


async def test_un_error_inesperado_no_termina_el_consumidor_y_va_a_la_dlq() -> None:
    rig = Rig(RuntimeError("bug"))
    msg = rig.message()

    await rig.handler.handle(msg)  # no lanza

    assert rig.order == ["publish_event", "publish_dlq", "ack"]
    assert rig.publisher.dlq[0][2] == "unexpected_error:RuntimeError"
    _only_validation_failed(rig, _causation(msg))


async def test_si_el_ack_falla_no_se_propaga_la_excepcion(caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig()
    msg = rig.message()

    async def broken_ack() -> None:
        raise ConnectionError("canal cerrado")

    msg.ack = broken_ack  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR):
        await rig.handler.handle(msg)

    assert any("no se pudo confirmar" in r.getMessage() for r in caplog.records)


# --- VALIDATION_FAILED: el aviso a Alert antes de derivar a la DLQ ------------------------


def _only_alert_id(payload: dict[str, Any]) -> None:
    for key in [k for k in payload if k != "alert_id"]:
        del payload[key]


def _uppercase_alert_id_and_other_bucket(payload: dict[str, Any]) -> None:
    payload["alert_id"] = payload["alert_id"].upper()
    payload["bucket"] = "otro-bucket"


INVALID_PAYLOADS = [
    pytest.param(lambda p: p.update(bucket="otro-bucket"), id="otro bucket"),
    pytest.param(lambda p: p.pop("evidence"), id="sin evidencia"),
    pytest.param(lambda p: p["evidence"][0].update(sha256="ABC"), id="sha256 inválido"),
    pytest.param(lambda p: p.update(pat_enc_ia="valor-secreto-no-base64!!"), id="pat_enc_ia no es base64"),
    pytest.param(lambda p: p.update(alert_type="PARKED_TOO_CLOSE"), id="tipo deshabilitado"),
    pytest.param(_only_alert_id, id="solo el alert_id"),
    pytest.param(_uppercase_alert_id_and_other_bucket, id="alert_id en mayúsculas"),
]

ILLEGIBLE_ALERT_IDS = [
    pytest.param(lambda p: p.pop("alert_id"), id="falta"),
    pytest.param(lambda p: p.update(alert_id=None), id="nulo"),
    pytest.param(lambda p: p.update(alert_id=""), id="vacío"),
    pytest.param(lambda p: p.update(alert_id="x"), id="no es un uuid"),
    pytest.param(lambda p: p.update(alert_id=12345), id="número"),
    pytest.param(lambda p: p.update(alert_id=["a"]), id="lista"),
]

DOOMED = ["payload inválido", "error del sistema", "error inesperado"]


def _doomed(kind: str) -> tuple[Rig, bytes]:
    """Un mensaje con alert_id legible que va a terminar en la DLQ por `kind`."""
    if kind == "payload inválido":
        rig = Rig()
        return rig, _tampered(rig, lambda p: p.update(bucket="otro-bucket"))
    rig = Rig(SystemDecryptionError("x") if kind == "error del sistema" else RuntimeError("bug"))
    return rig, _envelope_body(rig.payload)


@pytest.mark.parametrize("mutate", INVALID_PAYLOADS)
async def test_payload_invalido_con_alert_id_avisa_validation_failed_y_recien_despues_va_a_la_dlq(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    rig = Rig()
    causation = uuid.uuid4()
    msg = rig.message(_tampered(rig, mutate, causation), correlation_id="corr-9")

    await rig.handler.handle(msg)

    assert rig.order == ["publish_event", "publish_dlq", "ack"]
    _only_validation_failed(rig, causation)  # incluso con el alert_id en mayúsculas: sale canónico
    assert rig.publisher.events[0][3] == "corr-9"
    queue, body, reason = rig.publisher.dlq[0]
    assert (queue, body) == (QUEUE, msg.body)
    assert reason.startswith("invalid_payload")
    assert rig.use_case.calls == []


@pytest.mark.parametrize("mutate", ILLEGIBLE_ALERT_IDS)
async def test_sin_alert_id_legible_va_solo_a_la_dlq(mutate: Callable[[dict[str, Any]], object]) -> None:
    rig = Rig()
    msg = rig.message(_tampered(rig, mutate))

    await rig.handler.handle(msg)

    assert rig.order == ["publish_dlq", "ack"]
    assert rig.publisher.events == []
    assert rig.publisher.dlq[0][1] == msg.body
    assert rig.publisher.dlq[0][2].startswith("invalid_payload")


@pytest.mark.parametrize(
    "overrides",
    [{"eventId": "no-uuid"}, {"eventType": "otro.evento"}, {"timestamp": "ayer"}, {"producer": None}],
)
async def test_sobre_invalido_va_solo_a_la_dlq_aunque_el_cuerpo_tenga_un_alert_id(
    overrides: dict[str, Any],
) -> None:
    rig = Rig()
    msg = rig.message(_envelope_body(rig.payload, **overrides))

    await rig.handler.handle(msg)

    assert rig.order == ["publish_dlq", "ack"]
    assert rig.publisher.events == []
    assert rig.publisher.dlq[0][2].startswith("invalid_envelope")


async def test_reintentos_agotados_por_publicacion_fallida_avisan_si_el_broker_se_recupero() -> None:
    rig = Rig()
    rig.publisher.event_failures = 1  # solo falla la publicación del veredicto, no la del aviso
    msg = rig.message(headers=_x_death(MAX_RETRIES))

    await rig.handler.handle(msg)

    assert rig.order == ["publish_event", "publish_dlq", "ack"]
    _only_validation_failed(rig, _causation(msg))
    assert rig.publisher.dlq[0][2] == "max_retries_exceeded:publish_failed"


@pytest.mark.parametrize("kind", DOOMED)
async def test_si_el_aviso_no_se_confirma_se_reintenta_sin_derivar_a_la_dlq(kind: str) -> None:
    rig, body = _doomed(kind)
    rig.publisher.fail_events = True
    msg = rig.message(body, headers=_x_death(0))

    await rig.handler.handle(msg)

    assert msg.log == ["nack"]
    assert msg.nack_requeue is False  # a la cola .retry, por la vía normal
    assert rig.publisher.dlq == []
    assert rig.publisher.events == []

    rig.publisher.fail_events = False  # el broker se recupera: la siguiente entrega sí avisa
    again = rig.message(body, headers=_x_death(1))
    await rig.handler.handle(again)

    assert again.log == ["ack"]
    assert [event[0] for event in rig.publisher.events] == ["ia.evidence.validated.red"]
    assert len(rig.publisher.dlq) == 1


@pytest.mark.parametrize("kind", DOOMED)
async def test_agotados_los_reintentos_sin_poder_avisar_va_a_la_dlq_sin_aviso(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    rig, body = _doomed(kind)
    rig.publisher.fail_events = True
    msg = rig.message(body, headers=_x_death(MAX_RETRIES))

    with caplog.at_level(logging.ERROR):
        await rig.handler.handle(msg)

    assert rig.order == ["publish_dlq", "ack"]  # no queda en un ciclo: la DLQ es la salida
    assert rig.publisher.events == []
    assert rig.publisher.dlq[0][1] == body
    assert any("sin poder avisar VALIDATION_FAILED" in r.getMessage() for r in caplog.records)


async def test_si_la_dlq_no_confirma_despues_del_aviso_se_reintenta_y_el_aviso_se_repite_igual() -> None:
    rig = Rig(SystemDecryptionError("x"))
    rig.publisher.fail_dlq = True
    body = _envelope_body(rig.payload)
    msg = rig.message(body, headers=_x_death(0))

    await rig.handler.handle(msg)

    assert msg.log == ["nack"]
    assert len(rig.publisher.events) == 1
    assert rig.publisher.dlq == []

    rig.publisher.fail_dlq = False
    await rig.handler.handle(rig.message(body, headers=_x_death(1)))

    first, second = rig.publisher.events
    assert first[1] == second[1]  # mismo eventId y mismo cuerpo: Alert deduplica
    assert len(rig.publisher.dlq) == 1


async def test_una_excepcion_inesperada_al_avisar_no_escapa_y_cuenta_como_aviso_no_confirmado(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = Rig(SystemDecryptionError("x"))
    rig.publisher.event_error = RuntimeError("detalle-interno-sensible")
    msg = rig.message(headers=_x_death(0))

    with caplog.at_level(logging.DEBUG):
        await rig.handler.handle(msg)  # no lanza

    assert msg.log == ["nack"]
    assert rig.publisher.dlq == []
    output = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert "RuntimeError" in output  # solo el tipo del error
    assert "detalle-interno-sensible" not in output


@pytest.mark.parametrize("kind", DOOMED)
async def test_el_aviso_y_los_logs_no_contienen_el_criptograma_ni_datos_de_la_evidencia(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    rig, body = _doomed(kind)
    msg = rig.message(body, headers=_x_death(0))

    with caplog.at_level(logging.DEBUG):
        await rig.handler.handle(msg)
        rig.publisher.fail_events = True
        await rig.handler.handle(rig.message(body, headers=_x_death(0)))  # también cuando el aviso falla

    output = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    sent = rig.publisher.events[0][1].decode()
    reason = rig.publisher.dlq[0][2]
    secrets = [
        rig.payload.pat_enc_ia,
        "valor-secreto",
        rig.payload.evidence[0].key,
        rig.payload.evidence[0].sha256,
    ]
    assert caplog.records
    for text in (output, sent, reason):
        for secret in secrets:
            assert secret not in text


async def test_el_secreto_de_un_payload_invalido_no_llega_ni_al_aviso_ni_a_los_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = Rig()
    secret = "valor-secreto-no-base64!!"
    msg = rig.message(_tampered(rig, lambda p: p.update(pat_enc_ia=secret)))

    with caplog.at_level(logging.DEBUG):
        await rig.handler.handle(msg)

    output = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert secret not in output
    assert secret not in rig.publisher.events[0][1].decode()
    assert secret not in rig.publisher.dlq[0][2]


def test_el_evento_validation_failed_respeta_el_contrato_y_es_determinista() -> None:
    rig = Rig()
    alert_id, causation = rig.payload.alert_id, uuid.uuid4()

    routing_key, body, event_id = build_validation_failed_event(alert_id, causation, NOW)

    assert (routing_key, body, event_id) == build_validation_failed_event(alert_id, causation, NOW)
    event = json.loads(body)
    assert routing_key == "ia.evidence.validated.red"
    assert set(event) == {"eventId", "eventType", "timestamp", "producer", "actor", "payload"}
    assert event["eventId"] == str(event_id) == str(uuid.uuid5(EVENT_ID_NAMESPACE, str(causation)))
    assert event["timestamp"] == "2026-10-07T12:00:00Z"
    assert event["producer"] == "vinculo-vial-ia"
    # Mismo contrato que el veredicto de la cascada, y el mismo eventId para el mismo evento de entrada.
    _, normal_body, normal_id = build_validated_event(rig.payload, _result(), causation, NOW)
    assert set(event["payload"]) == set(json.loads(normal_body)["payload"])
    assert normal_id == event_id


# --- actor ------------------------------------------------------------------------------


async def test_un_actor_no_nulo_se_ignora_con_advertencia(caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig()
    actor = "9f1c7a52-aaaa-bbbb-cccc-000000000001"
    msg = rig.message(_envelope_body(rig.payload, actor=actor))

    with caplog.at_level(logging.WARNING):
        await rig.handler.handle(msg)

    assert rig.order == ["publish_event", "ack"]
    assert json.loads(rig.publisher.events[0][1])["actor"] is None
    warning = [r for r in caplog.records if r.levelno == logging.WARNING and "actor" in r.getMessage()]
    assert warning
    output = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert actor not in output
    assert actor not in rig.publisher.events[0][1].decode()


# --- privacidad en logs ------------------------------------------------------------------


async def test_los_logs_del_broker_no_contienen_el_criptograma_de_la_patente(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = Rig(TransientError("s3"))

    with caplog.at_level(logging.DEBUG):
        await rig.handler.handle(rig.message(headers=_x_death(MAX_RETRIES)))
        await rig.handler.handle(rig.message(b"basura"))

    output = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert caplog.records
    assert rig.payload.pat_enc_ia not in output


# --- x-death y sobre ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (None, 0),
        ({}, 0),
        ({"x-death": "no-es-lista"}, 0),
        ({"x-death": []}, 0),
        (_x_death(2), 2),
        (_x_death(2, queue="otra.q"), 0),
        ({"x-death": [{"queue": QUEUE, "count": 1}, {"queue": f"{QUEUE}.retry", "count": 9}]}, 1),
        ({"x-death": [{"queue": QUEUE.encode(), "count": 4}]}, 4),
        ({"x-death": [{"queue": QUEUE, "count": "3"}]}, 0),
        ({"x-death": [{"queue": QUEUE, "count": True}]}, 0),
        ({"x-death": ["raro", {"queue": QUEUE, "count": 5}]}, 5),
    ],
)
def test_death_count(headers: Mapping[str, Any] | None, expected: int) -> None:
    assert death_count(headers, QUEUE) == expected


def test_parse_envelope_valido() -> None:
    envelope = parse_envelope(_envelope_body(build_scenario().payload))

    assert envelope.eventType == EVENT_TYPE_IN
    assert envelope.actor is None


def test_parse_envelope_invalido_no_filtra_contenido() -> None:
    with pytest.raises(InvalidEnvelopeError) as exc:
        parse_envelope(b'{"eventId": "secreto-123"}')

    assert "secreto-123" not in str(exc.value)


# --- AmqpPublisher con un exchange falso -------------------------------------------------


class FakeExchange:
    name = "fake.exchange"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.published: list[tuple[Message, str, dict[str, Any]]] = []

    async def publish(self, message: Message, routing_key: str, **kwargs: Any) -> None:
        if self.error is not None:
            raise self.error
        self.published.append((message, routing_key, kwargs))


async def test_publica_persistente_mandatory_y_con_timeout() -> None:
    main, dlq = FakeExchange(), FakeExchange()
    publisher = AmqpPublisher(main, dlq, 2.5)  # type: ignore[arg-type]

    await publisher.publish_event("ia.evidence.validated.green", b"{}", "msg-1", "corr-1")

    message, routing_key, kwargs = main.published[0]
    assert routing_key == "ia.evidence.validated.green"
    assert message.delivery_mode == DeliveryMode.PERSISTENT
    assert message.message_id == "msg-1"
    assert message.correlation_id == "corr-1"
    assert kwargs == {"mandatory": True, "timeout": 2.5}
    assert dlq.published == []


async def test_dlq_se_publica_al_exchange_dlq_con_la_cola_como_routing_key() -> None:
    main, dlq = FakeExchange(), FakeExchange()
    publisher = AmqpPublisher(main, dlq, 2.5)  # type: ignore[arg-type]

    await publisher.publish_dlq(QUEUE, b"cuerpo", "invalid_payload")

    message, routing_key, kwargs = dlq.published[0]
    assert routing_key == QUEUE
    assert message.body == b"cuerpo"
    assert message.delivery_mode == DeliveryMode.PERSISTENT
    assert message.headers == {DISCARD_REASON_HEADER: "invalid_payload"}
    assert kwargs["mandatory"] is True
    assert main.published == []


@pytest.mark.parametrize(
    "error",
    [
        DeliveryError(None, None),  # type: ignore[arg-type]
        ChannelInvalidStateError("cerrado"),
        TimeoutError(),
        ConnectionResetError(),
    ],
)
async def test_los_fallos_de_publicacion_se_informan_como_transitorios(error: Exception) -> None:
    publisher = AmqpPublisher(FakeExchange(error), FakeExchange(error), 1.0)  # type: ignore[arg-type]

    with pytest.raises(PublishFailedError):
        await publisher.publish_event("rk", b"x", "id", None)
    with pytest.raises(PublishFailedError):
        await publisher.publish_dlq(QUEUE, b"x", "r")


def test_publish_failed_error_es_transitorio() -> None:
    assert issubclass(PublishFailedError, TransientError)


async def test_un_mensaje_devuelto_se_loguea_sin_el_cuerpo(caplog: pytest.LogCaptureFixture) -> None:
    publisher = AmqpPublisher(FakeExchange(DeliveryError(None, None)), FakeExchange(), 1.0)  # type: ignore[arg-type]

    with caplog.at_level(logging.ERROR), pytest.raises(PublishFailedError):
        await publisher.publish_event("ia.evidence.validated.red", b"cuerpo-sensible", "id", None)

    assert any("devolvió o rechazó" in r.getMessage() for r in caplog.records)
    assert "cuerpo-sensible" not in "\n".join(JsonFormatter().format(r) for r in caplog.records)


# --- NoSuchKey de un original: DLQ directo (adaptador S3 real + caso de uso real) ---------


class _MissingObjectClient:
    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        raise ClientError(
            {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "GetObject"
        )


async def test_no_such_key_de_un_original_va_directo_a_la_dlq_sin_reintentos() -> None:
    rig = Rig()
    scenario = rig.scenario
    store = S3EvidenceStore(
        _MissingObjectClient(), scenario.payload.bucket, CircuitBreaker("s3", 5, 60, 30, 1), 1024
    )
    use_case = ValidateEvidence(
        store=store,
        writer=FakeWriter(),
        decryptor=scenario.decryptor,
        decoder=scenario.decoder,
        renderer=JpegRenderer(),
        models=happy_models().as_models(),
        thresholds=THRESHOLDS,
        redaction=REDACTION,
    )
    rig.handler = EvidenceMessageHandler(
        queue=QUEUE,
        max_retries=MAX_RETRIES,
        bucket=scenario.payload.bucket,
        use_case=use_case,
        publisher=rig.publisher,
        run_blocking=rig._run_blocking,
        clock=lambda: NOW,
    )
    msg = rig.message(headers=_x_death(0))  # primer intento: todavía quedan reintentos

    await rig.handler.handle(msg)

    assert rig.order == ["publish_event", "publish_dlq", "ack"]
    assert rig.publisher.dlq[0][0] == QUEUE
    assert rig.publisher.dlq[0][2] == "system_error:ObjectNotFoundError"
    _only_validation_failed(rig, _causation(msg))
    assert msg.nack_requeue is None  # nunca se rechazó para reintentar
