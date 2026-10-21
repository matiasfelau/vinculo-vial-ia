"""Prueba de integración contra un RabbitMQ real y descartable.

Se omite salvo que `RABBITMQ_TEST_URL` apunte a un broker de prueba. Declara acá la topología que en
producción declara solo `vinculo-vial-core` (misma forma que `internal/topology/flow.go`).
"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

import aio_pika
import pytest
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage

from tests.fakes import build_scenario
from vvi_ia.adapters.broker import (
    EVENT_TYPE_IN,
    AmqpPublisher,
    EventPublisher,
    EvidenceConsumer,
    EvidenceMessageHandler,
    PublishFailedError,
)
from vvi_ia.domain.model import ReasonCode, ValidationResult, Verdict
from vvi_ia.service.payload import EvidenceUploaded
from vvi_ia.service.ports import TransientError

URL = os.environ.get("RABBITMQ_TEST_URL")
pytestmark = pytest.mark.skipif(URL is None, reason="RABBITMQ_TEST_URL no definido")

MAIN, RETRY, DLQ = "vvi.topic.exchange", "vvi.retry.exchange", "vvi.dlq.exchange"
QUEUE = "ia.alert-evidence-uploaded.q"
CAPTURE = "test.capture-validated.q"
RETRY_TTL_MS = 300
MAX_RETRIES = 2
WAIT_S = 10.0


class CountingUseCase:
    def __init__(self, behavior: Callable[[int], ValidationResult]) -> None:
        self.behavior = behavior
        self.calls = 0

    def execute(self, payload: EvidenceUploaded) -> ValidationResult:
        self.calls += 1
        return self.behavior(self.calls)


def _ok(_: int) -> ValidationResult:
    return ValidationResult(verdict=Verdict.RED, reason_codes=(ReasonCode.PLATE_MISMATCH,))


def _always_transient(_: int) -> ValidationResult:
    raise TransientError("s3 caído")


@pytest.fixture
async def channel() -> AsyncIterator[AbstractChannel]:
    assert URL is not None
    connection = await aio_pika.connect_robust(URL)
    async with connection:
        ch = await connection.channel()
        for name in (MAIN, RETRY, DLQ):
            await ch.declare_exchange(name, aio_pika.ExchangeType.TOPIC, durable=True)
        for name in (QUEUE, f"{QUEUE}.retry", f"{QUEUE}.dlq", CAPTURE):
            await (await ch.get_queue(name, ensure=False)).delete(if_unused=False, if_empty=False)
        work = await ch.declare_queue(
            QUEUE,
            durable=True,
            arguments={"x-dead-letter-exchange": RETRY, "x-dead-letter-routing-key": QUEUE},
        )
        retry = await ch.declare_queue(
            f"{QUEUE}.retry",
            durable=True,
            arguments={
                "x-message-ttl": RETRY_TTL_MS,
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": QUEUE,
            },
        )
        dlq = await ch.declare_queue(f"{QUEUE}.dlq", durable=True)
        capture = await ch.declare_queue(CAPTURE, durable=True)
        await work.bind(MAIN, EVENT_TYPE_IN)
        await retry.bind(RETRY, QUEUE)
        await dlq.bind(DLQ, QUEUE)
        await capture.bind(MAIN, "ia.evidence.validated.*")
        yield ch
        for q in (work, retry, dlq, capture):
            await q.delete(if_unused=False, if_empty=False)


def _consumer(use_case: CountingUseCase, bucket: str) -> EvidenceConsumer:
    async def run_blocking(operation: Callable[[], ValidationResult]) -> ValidationResult:
        return await asyncio.to_thread(operation)

    def factory(publisher: EventPublisher) -> EvidenceMessageHandler:
        return EvidenceMessageHandler(
            queue=QUEUE,
            max_retries=MAX_RETRIES,
            bucket=bucket,
            use_case=use_case,
            publisher=publisher,
            run_blocking=run_blocking,
        )

    assert URL is not None
    return EvidenceConsumer(
        url=URL,
        queue=QUEUE,
        main_exchange=MAIN,
        dlq_exchange=DLQ,
        prefetch=2,
        confirm_timeout_s=5.0,
        handler_factory=factory,
    )


def _event(payload: EvidenceUploaded) -> bytes:
    return json.dumps(
        {
            "eventId": str(uuid.uuid4()),
            "eventType": EVENT_TYPE_IN,
            "timestamp": datetime.now(UTC).isoformat(),
            "producer": "vinculo-vial-alert",
            "actor": None,
            "payload": json.loads(payload.model_dump_json()),
        }
    ).encode()


async def _publish(channel: AbstractChannel, body: bytes) -> None:
    exchange = await channel.get_exchange(MAIN)
    await exchange.publish(
        aio_pika.Message(body, delivery_mode=aio_pika.DeliveryMode.PERSISTENT), EVENT_TYPE_IN, mandatory=True
    )


async def _get(channel: AbstractChannel, name: str) -> AbstractIncomingMessage:
    queue = await channel.get_queue(name)
    async with asyncio.timeout(WAIT_S):
        while True:
            message = await queue.get(fail=False, no_ack=True)
            if message is not None:
                return message
            await asyncio.sleep(0.1)


async def test_evento_valido_publica_el_veredicto_y_vacia_la_cola(channel: AbstractChannel) -> None:
    payload = build_scenario().payload
    use_case = CountingUseCase(_ok)
    consumer = _consumer(use_case, payload.bucket)
    await consumer.start()
    try:
        assert consumer.consuming
        await _publish(channel, _event(payload))
        message = await _get(channel, CAPTURE)
    finally:
        await consumer.stop()

    event: dict[str, Any] = json.loads(message.body)
    assert message.routing_key == "ia.evidence.validated.red"
    assert message.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
    assert event["payload"]["alert_id"] == payload.alert_id
    assert event["payload"]["verdict"] == "RED"
    assert use_case.calls == 1
    assert (await (await channel.get_queue(QUEUE)).declare()).message_count == 0


async def test_mensaje_invalido_va_a_la_dlq_con_la_cola_de_trabajo_como_routing_key(
    channel: AbstractChannel,
) -> None:
    use_case = CountingUseCase(_ok)
    consumer = _consumer(use_case, "bucket-de-prueba")
    await consumer.start()
    try:
        exchange = await channel.get_exchange(MAIN)
        await exchange.publish(aio_pika.Message(b"no es json"), EVENT_TYPE_IN, mandatory=True)
        message = await _get(channel, f"{QUEUE}.dlq")
    finally:
        await consumer.stop()

    assert message.body == b"no es json"
    assert message.headers["x-discard-reason"].startswith("invalid_envelope")
    assert use_case.calls == 0
    # Sin sobre válido no hay alert_id legible: solo DLQ, ningún aviso.
    assert (await (await channel.get_queue(CAPTURE)).declare()).message_count == 0


async def test_falla_transitoria_reintenta_por_la_cola_retry_y_termina_en_la_dlq(
    channel: AbstractChannel,
) -> None:
    payload = build_scenario().payload
    use_case = CountingUseCase(_always_transient)
    consumer = _consumer(use_case, payload.bucket)
    await consumer.start()
    try:
        await _publish(channel, _event(payload))
        message = await _get(channel, f"{QUEUE}.dlq")
    finally:
        await consumer.stop()

    assert use_case.calls == MAX_RETRIES + 1  # el intento original más MAX_RETRIES reintentos
    assert message.headers["x-discard-reason"].startswith("max_retries_exceeded")
    # Antes de la DLQ se avisó a Alert que la validación falló (sin fraude).
    notice = await _get(channel, CAPTURE)
    assert notice.routing_key == "ia.evidence.validated.red"
    assert json.loads(notice.body)["payload"]["reason_codes"] == ["VALIDATION_FAILED"]
    assert json.loads(notice.body)["payload"]["alert_id"] == payload.alert_id


async def test_payload_invalido_con_alert_id_publica_validation_failed_y_va_a_la_dlq(
    channel: AbstractChannel,
) -> None:
    payload = build_scenario().payload
    use_case = CountingUseCase(_ok)
    consumer = _consumer(use_case, payload.bucket)
    envelope = json.loads(_event(payload))
    envelope["payload"]["evidence"] = []  # payload inválido con alert_id legible
    body = json.dumps(envelope).encode()
    await consumer.start()
    try:
        await _publish(channel, body)
        dead = await _get(channel, f"{QUEUE}.dlq")
        notice = await _get(channel, CAPTURE)
    finally:
        await consumer.stop()

    event: dict[str, Any] = json.loads(notice.body)
    assert notice.routing_key == "ia.evidence.validated.red"
    assert notice.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
    assert event["payload"] == {
        "alert_id": payload.alert_id,
        "causation_event_id": envelope["eventId"],
        "verdict": "RED",
        "reason_codes": ["VALIDATION_FAILED"],
        "scores": {},
        "model_versions": {},
        "redacted": [],
    }
    assert dead.body == body
    assert dead.headers["x-discard-reason"].startswith("invalid_payload")
    assert use_case.calls == 0
    assert (await (await channel.get_queue(QUEUE)).declare()).message_count == 0


async def test_si_el_aviso_no_tiene_ruta_reintenta_por_la_cola_retry_y_termina_en_la_dlq_sin_aviso(
    channel: AbstractChannel, caplog: pytest.LogCaptureFixture
) -> None:
    payload = build_scenario().payload
    use_case = CountingUseCase(_ok)
    consumer = _consumer(use_case, payload.bucket)
    envelope = json.loads(_event(payload))
    envelope["payload"]["evidence"] = []
    body = json.dumps(envelope).encode()
    # Sin el binding de `ia.evidence.validated.*` el aviso vuelve como mandatory: no se confirma.
    await (await channel.get_queue(CAPTURE)).unbind(MAIN, "ia.evidence.validated.*")
    await consumer.start()
    try:
        with caplog.at_level(logging.WARNING):
            await _publish(channel, body)
            dead = await _get(channel, f"{QUEUE}.dlq")
    finally:
        await consumer.stop()

    attempts = [r for r in caplog.records if "no se pudo publicar VALIDATION_FAILED" in r.getMessage()]
    assert len(attempts) == MAX_RETRIES + 1  # el intento original más MAX_RETRIES reintentos
    assert dead.body == body
    assert dead.headers["x-discard-reason"].startswith("invalid_payload")
    assert (await (await channel.get_queue(CAPTURE)).declare()).message_count == 0


async def test_publicar_sin_ruta_falla_porque_mandatory_devuelve_el_mensaje(
    channel: AbstractChannel,
) -> None:
    # Misma configuración de canal que usa EvidenceConsumer: confirms y returns que elevan.
    assert URL is not None
    connection = await aio_pika.connect_robust(URL)
    async with connection:
        publishing = await connection.channel(publisher_confirms=True, on_return_raises=True)
        publisher = AmqpPublisher(
            await publishing.get_exchange(MAIN), await publishing.get_exchange(DLQ), 5.0
        )

        with pytest.raises(PublishFailedError):
            await publisher.publish_event("sin.ruta.alguna", b"{}", "id", None)
