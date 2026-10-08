"""Almacenamiento de evidencia en S3 (AWS real): boto3 con timeouts, sin reintentos y con breaker.

Credenciales: cadena por defecto del SDK, sin endpoint custom. IA solo puede hacer Get sobre
`*/original/*` y Put sobre `*/redacted/*`.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from vvi_ia.adapters.breaker import CircuitBreaker
from vvi_ia.service.ports import ObjectNotFoundError, TransientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

T = TypeVar("T")

MS_PER_SECOND = 1000.0
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
HTTP_NOT_FOUND = 404
NOT_FOUND_CODES = frozenset({"NoSuchKey", "NotFound", "404"})
REDACTED_SEGMENT = "/redacted/"


def build_client(region: str, timeout_ms: int) -> "S3Client":
    """Cliente S3 con timeout de conexión y de lectura explícitos y sin reintentos internos."""
    timeout_s = timeout_ms / MS_PER_SECOND
    config = Config(
        connect_timeout=timeout_s,
        read_timeout=timeout_s,
        retries={"max_attempts": 0, "mode": "standard"},
    )
    return boto3.client("s3", region_name=region, config=config)


def _http_status(exc: ClientError) -> int:
    return int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))


def is_breaker_failure(exc: Exception) -> bool:
    """Cuentan: timeouts, errores de conexión, 5xx y 429. El resto de los 4xx no."""
    if isinstance(exc, ClientError):
        status = _http_status(exc)
        return status >= HTTP_SERVER_ERROR or status == HTTP_TOO_MANY_REQUESTS
    return isinstance(exc, BotoCoreError)


class S3EvidenceStore:
    """Implementa `EvidenceStore` (lectura de originales) y sube las versiones redactadas."""

    def __init__(self, client: Any, bucket: str, breaker: CircuitBreaker) -> None:
        self._client = client
        self._bucket = bucket
        self._breaker = breaker

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def get(self, key: str) -> bytes:
        """Descarga un objeto. Lanza `ObjectNotFoundError` o `TransientError`."""

        def operation() -> bytes:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            body: bytes = response["Body"].read()
            return body

        return self._guarded(operation)

    def put(self, key: str, body: bytes, content_type: str) -> None:
        """Sube una versión redactada. Solo acepta claves bajo `redacted/`."""
        if REDACTED_SEGMENT not in key:
            raise ValueError("IA solo escribe objetos redactados")

        def operation() -> None:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=body, ContentType=content_type)

        self._guarded(operation)

    def _guarded(self, operation: Callable[[], T]) -> T:
        try:
            return self._breaker.call(operation, is_breaker_failure)
        except TransientError:
            raise
        except ClientError as exc:
            if _is_not_found(exc):
                raise ObjectNotFoundError("el objeto no existe") from None
            raise TransientError(f"S3 respondió {_http_status(exc)}") from None
        except BotoCoreError as exc:
            raise TransientError(f"falla de conexión con S3: {type(exc).__name__}") from None


def _is_not_found(exc: ClientError) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return code in NOT_FOUND_CODES or _http_status(exc) == HTTP_NOT_FOUND
