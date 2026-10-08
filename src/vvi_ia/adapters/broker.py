"""RabbitMQ con aio-pika: consumo con ack manual, reintentos por `x-death`, DLQ y publisher confirms.

La topología (exchanges, colas `.retry` y `.dlq`) la declara solo `vinculo-vial-core`; acá solo se
consume y se publica sobre topología existente. El retry y la DLQ se rutean por el **nombre de la
cola de trabajo**, nunca por la routing key del evento.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, TypeVar

import aio_pika
from aio_pika import DeliveryMode, Message
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection
from aiormq.exceptions import AMQPError, ChannelInvalidStateError, DeliveryError
from pydantic import BaseModel, ConfigDict, ValidationError

from vvi_ia.domain.model import ValidationResult
from vvi_ia.service.payload import EvidenceUploaded, InvalidPayloadError, parse_evidence_uploaded
from vvi_ia.service.ports import ObjectNotFoundError, SystemDecryptionError, TransientError

logger = logging.getLogger(__name__)

T = TypeVar("T")

PRODUCER = "vinculo-vial-ia"
EVENT_TYPE_IN = "alert.evidence.uploaded"
EVENT_TYPE_OUT = "ia.evidence.validated"
ROUTING_KEY_PREFIX = "ia.evidence.validated."
CONTENT_TYPE_JSON = "application/json"
DISCARD_REASON_HEADER = "x-discard-reason"
X_DEATH_HEADER = "x-death"
MS_PER_SECOND = 1000.0
SHUTDOWN_GRACE_SECONDS = 30.0
# Namespace fijo: el eventId de salida se deriva del evento de entrada, así un reintento o una
# redelivery emite el mismo eventId y el consumidor (Alert) puede deduplicar.
EVENT_ID_NAMESPACE = uuid.UUID("6f1b6a5e-3c1e-4f7e-9a53-1d6f7a0c2b11")


class PublishFailedError(TransientError):
    """El broker no confirmó la publicación (nack, mensaje devuelto, canal cerrado o timeout)."""


class InvalidEnvelopeError(Exception):
    """El sobre del evento no cumple el esquema: va directo a la DLQ."""


class Outcome(StrEnum):
    """Qué hacer con el mensaje de entrada tras procesarlo."""

    ACK = "ack"
    RETRY = "retry"


class Envelope(BaseModel):
    """Sobre estándar de eventos: `{ eventId, eventType, timestamp, producer, actor, payload }`."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    eventId: uuid.UUID  # noqa: N815 - nombre del contrato compartido
    eventType: str  # noqa: N815
    timestamp: datetime
    producer: str
    actor: Any = None
    payload: dict[str, Any]


def parse_envelope(body: bytes) -> Envelope:
    """Deserializa y valida el sobre. Lanza `InvalidEnvelopeError` sin incluir el contenido."""
    try:
        envelope = Envelope.model_validate_json(body)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise InvalidEnvelopeError(f"sobre inválido en: {', '.join(fields) or 'raíz'}") from None
    if envelope.eventType != EVENT_TYPE_IN:
        raise InvalidEnvelopeError("eventType inesperado")
    return envelope


def death_count(headers: Mapping[str, Any] | None, queue: str) -> int:
    """Veces que el mensaje murió en `queue` según el header `x-death` (0 si nunca fue rechazado)."""
    deaths = (headers or {}).get(X_DEATH_HEADER)
    if not isinstance(deaths, list):
        return 0
    count = 0
    for entry in deaths:
        if not isinstance(entry, Mapping):
            continue
        entry_queue = entry.get("queue")
        if isinstance(entry_queue, bytes):
            entry_queue = entry_queue.decode("utf-8", "replace")
        n = entry.get("count")
        if entry_queue == queue and isinstance(n, int) and not isinstance(n, bool):
            count = max(count, n)
    return count


def build_validated_event(
    payload: EvidenceUploaded,
    result: ValidationResult,
    causation_event_id: uuid.UUID,
    now: datetime,
) -> tuple[str, bytes, uuid.UUID]:
    """Arma `ia.evidence.validated.<verdict>`: (routing key, cuerpo JSON, eventId)."""
    event_id = uuid.uuid5(EVENT_ID_NAMESPACE, str(causation_event_id))
    envelope = {
        "eventId": str(event_id),
        "eventType": EVENT_TYPE_OUT,
        "timestamp": now.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "producer": PRODUCER,
        "actor": None,
        "payload": {
            "alert_id": payload.alert_id,
            "causation_event_id": str(causation_event_id),
            "verdict": result.verdict.value,
            "reason_codes": [code.value for code in result.reason_codes],
            "scores": {
                "burst_similarity": result.scores.burst_similarity,
                "ocr_confidence": result.scores.ocr_confidence,
                "context_confidence": result.scores.context_confidence,
            },
            "model_versions": dict(result.model_versions),
            "redacted": [
                {"role": ref.role.value, "index": ref.index, "key": ref.key} for ref in result.redacted
            ],
        },
    }
    routing_key = ROUTING_KEY_PREFIX + result.verdict.value.lower()
    return routing_key, json.dumps(envelope, separators=(",", ":")).encode(), event_id


class IncomingLike(Protocol):
    """Lo que el handler necesita de un mensaje entrante (lo cumple `aio_pika.IncomingMessage`)."""

    @property
    def body(self) -> bytes: ...

    @property
    def headers(self) -> Mapping[str, Any]: ...

    @property
    def correlation_id(self) -> str | None: ...

    async def ack(self) -> None: ...

    async def nack(self, multiple: bool = False, requeue: bool = True) -> None: ...


class EventPublisher(Protocol):
    """Publicación con confirm: retorna solo cuando el broker confirmó."""

    async def publish_event(
        self, routing_key: str, body: bytes, message_id: str, correlation_id: str | None
    ) -> None: ...

    async def publish_dlq(self, queue: str, body: bytes, reason: str) -> None: ...


class UseCase(Protocol):
    def execute(self, payload: EvidenceUploaded) -> ValidationResult: ...


RunBlocking = Callable[[Callable[[], T]], Awaitable[T]]


class AmqpPublisher:
    """Publica persistente, `mandatory=True` y con publisher confirm; loguea los mensajes devueltos."""

    def __init__(
        self,
        main_exchange: AbstractExchange,
        dlq_exchange: AbstractExchange,
        confirm_timeout_s: float,
    ) -> None:
        self._main = main_exchange
        self._dlq = dlq_exchange
        self._timeout = confirm_timeout_s

    async def publish_event(
        self, routing_key: str, body: bytes, message_id: str, correlation_id: str | None
    ) -> None:
        message = Message(
            body=body,
            content_type=CONTENT_TYPE_JSON,
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=message_id,
            correlation_id=correlation_id,
            type=EVENT_TYPE_OUT,
        )
        await self._publish(self._main, message, routing_key)

    async def publish_dlq(self, queue: str, body: bytes, reason: str) -> None:
        message = Message(
            body=body,
            content_type=CONTENT_TYPE_JSON,
            delivery_mode=DeliveryMode.PERSISTENT,
            headers={DISCARD_REASON_HEADER: reason},
        )
        await self._publish(self._dlq, message, queue)

    async def _publish(self, exchange: AbstractExchange, message: Message, routing_key: str) -> None:
        try:
            await exchange.publish(message, routing_key, mandatory=True, timeout=self._timeout)
        except DeliveryError as exc:
            # En modo confirm, un mensaje sin ruta vuelve como basic.return y se eleva acá.
            logger.error(
                "el broker devolvió o rechazó la publicación",
                extra={
                    "routing_key": routing_key,
                    "exchange": exchange.name,
                    "detail": _delivery_detail(exc),
                },
            )
            raise PublishFailedError("publicación no confirmada") from None
        except (AMQPError, ChannelInvalidStateError, TimeoutError, OSError) as exc:
            logger.error(
                "falló la publicación",
                extra={"routing_key": routing_key, "exchange": exchange.name, "error": type(exc).__name__},
            )
            raise PublishFailedError("publicación no confirmada") from None


def _delivery_detail(exc: DeliveryError) -> str:
    args = exc.args
    return str(args[0]) if args and isinstance(args[0], str) else type(exc).__name__


class EvidenceMessageHandler:
    """Procesa un mensaje: valida, corre la cascada en el pool, publica con confirm y recién ahí hace ack."""

    def __init__(
        self,
        *,
        queue: str,
        max_retries: int,
        bucket: str,
        use_case: UseCase,
        publisher: EventPublisher,
        run_blocking: RunBlocking[ValidationResult],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._queue = queue
        self._max_retries = max_retries
        self._bucket = bucket
        self._use_case = use_case
        self._publisher = publisher
        self._run_blocking = run_blocking
        self._clock = clock

    async def handle(self, message: IncomingLike) -> None:
        """Nunca lanza: un error inesperado deriva el mensaje a la DLQ."""
        correlation_id = message.correlation_id or str(uuid.uuid4())
        try:
            outcome = await self._process(message, correlation_id)
        except Exception as exc:  # noqa: BLE001 - el consumidor nunca debe terminar por un mensaje
            logger.error(
                "error inesperado procesando el mensaje",
                extra={"correlation_id": correlation_id, "error": type(exc).__name__},
            )
            outcome = await self._dead_letter(
                message, f"unexpected_error:{type(exc).__name__}", correlation_id
            )
        await self._settle(message, outcome, correlation_id)

    async def _settle(self, message: IncomingLike, outcome: Outcome, correlation_id: str) -> None:
        try:
            if outcome == Outcome.ACK:
                await message.ack()
            else:
                await message.nack(requeue=False)
        except Exception as exc:  # noqa: BLE001 - el canal pudo cerrarse; el broker re-entregará
            logger.error(
                "no se pudo confirmar o rechazar el mensaje",
                extra={
                    "correlation_id": correlation_id,
                    "outcome": outcome.value,
                    "error": type(exc).__name__,
                },
            )

    async def _process(self, message: IncomingLike, correlation_id: str) -> Outcome:
        try:
            envelope = parse_envelope(message.body)
        except InvalidEnvelopeError as exc:
            return await self._dead_letter(message, f"invalid_envelope: {exc}", correlation_id)
        if envelope.actor is not None:
            # IA nunca recibe identificadores de usuario: se ignora el valor (y no se loguea).
            logger.warning(
                "el evento trae actor no nulo; se ignora",
                extra={"event_id": str(envelope.eventId), "correlation_id": correlation_id},
            )
        try:
            payload = parse_evidence_uploaded(envelope.payload, self._bucket)
        except InvalidPayloadError as exc:
            return await self._dead_letter(message, f"invalid_payload: {exc}", correlation_id)

        try:
            result = await self._run_blocking(lambda: self._use_case.execute(payload))
        except TransientError as exc:
            logger.warning(
                "falla transitoria procesando la evidencia",
                extra={
                    "alert_id": payload.alert_id,
                    "correlation_id": correlation_id,
                    "error": type(exc).__name__,
                },
            )
            return await self._retry_or_dead_letter(message, "transient_error", correlation_id)
        except (SystemDecryptionError, ObjectNotFoundError) as exc:
            logger.error(
                "error del sistema procesando la evidencia",
                extra={
                    "alert_id": payload.alert_id,
                    "correlation_id": correlation_id,
                    "error": type(exc).__name__,
                },
            )
            return await self._dead_letter(message, f"system_error:{type(exc).__name__}", correlation_id)

        routing_key, body, event_id = build_validated_event(payload, result, envelope.eventId, self._clock())
        try:
            await self._publisher.publish_event(routing_key, body, str(event_id), correlation_id)
        except PublishFailedError:
            return await self._retry_or_dead_letter(message, "publish_failed", correlation_id)
        logger.info(
            "veredicto publicado",
            extra={
                "alert_id": payload.alert_id,
                "verdict": result.verdict.value,
                "reason_codes": [code.value for code in result.reason_codes],
                "correlation_id": correlation_id,
            },
        )
        return Outcome.ACK

    async def _retry_or_dead_letter(self, message: IncomingLike, reason: str, correlation_id: str) -> Outcome:
        retries = death_count(message.headers, self._queue)
        if retries >= self._max_retries:
            logger.error(
                "se superó MAX_RETRIES; el mensaje va a la DLQ",
                extra={"max_retries": self._max_retries, "correlation_id": correlation_id, "reason": reason},
            )
            return await self._dead_letter(message, f"max_retries_exceeded:{reason}", correlation_id)
        return Outcome.RETRY

    async def _dead_letter(self, message: IncomingLike, reason: str, correlation_id: str) -> Outcome:
        """Publica en la DLQ con la cola de trabajo como routing key; ack solo tras el confirm."""
        logger.error(
            "mensaje derivado a la DLQ",
            extra={"queue": self._queue, "reason": reason, "correlation_id": correlation_id},
        )
        try:
            await self._publisher.publish_dlq(self._queue, message.body, reason)
        except PublishFailedError:
            return Outcome.RETRY
        return Outcome.ACK


class EvidenceConsumer:
    """Conexión, consumo con `prefetch = WORKERS` y apagado ordenado."""

    def __init__(
        self,
        *,
        url: str,
        queue: str,
        main_exchange: str,
        dlq_exchange: str,
        prefetch: int,
        confirm_timeout_s: float,
        handler_factory: Callable[[EventPublisher], EvidenceMessageHandler],
    ) -> None:
        self._url = url
        self._queue_name = queue
        self._main_exchange = main_exchange
        self._dlq_exchange = dlq_exchange
        self._prefetch = prefetch
        self._confirm_timeout_s = confirm_timeout_s
        self._handler_factory = handler_factory
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._consumer_tag: str | None = None
        self._queue: aio_pika.abc.AbstractQueue | None = None
        self._handler: EvidenceMessageHandler | None = None
        self._inflight: set[asyncio.Task[None]] = set()

    @property
    def connected(self) -> bool:
        return self._connection is not None and not self._connection.is_closed

    @property
    def consuming(self) -> bool:
        return self.connected and self._consumer_tag is not None

    async def start(self) -> None:
        """Conecta (con reconexión automática), fija el QoS y empieza a consumir."""
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel(publisher_confirms=True, on_return_raises=True)
        await self._channel.set_qos(prefetch_count=self._prefetch)
        main = await self._channel.get_exchange(self._main_exchange)
        dlq = await self._channel.get_exchange(self._dlq_exchange)
        self._handler = self._handler_factory(AmqpPublisher(main, dlq, self._confirm_timeout_s))
        self._queue = await self._channel.get_queue(self._queue_name)
        self._consumer_tag = await self._queue.consume(self._on_message, no_ack=False)
        logger.info("consumo iniciado", extra={"queue": self._queue_name, "prefetch": self._prefetch})

    async def stop(self) -> None:
        """Deja de consumir, espera los mensajes en curso y cierra la conexión."""
        if self._queue is not None and self._consumer_tag is not None:
            try:
                await self._queue.cancel(self._consumer_tag)
            except Exception as exc:  # noqa: BLE001 - igual se sigue con el apagado
                logger.warning("no se pudo cancelar el consumidor", extra={"error": type(exc).__name__})
            self._consumer_tag = None
        if self._inflight:
            _, pending = await asyncio.wait(self._inflight, timeout=SHUTDOWN_GRACE_SECONDS)
            for task in pending:
                task.cancel()
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def _on_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        handler = self._handler
        if handler is None:
            return
        task = asyncio.current_task()
        if task is not None:
            self._inflight.add(task)
        try:
            await handler.handle(message)
        finally:
            if task is not None:
                self._inflight.discard(task)
