from typing import Any

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

from vvi_ia.adapters.breaker import CircuitBreaker, CircuitOpenError, CircuitState
from vvi_ia.adapters.storage_s3 import S3EvidenceStore, build_client, is_breaker_failure
from vvi_ia.service.ports import ObjectNotFoundError, TransientError

KEY = "alerts/a/original/general-0.bin"
REDACTED_KEY = "alerts/a/redacted/general-0.jpg"


def _client_error(status: int, code: str = "Err", operation: str = "GetObject") -> ClientError:
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, operation)


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.objects: dict[str, bytes] = {}
        self.puts: list[dict[str, Any]] = []
        self.calls = 0

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"Body": _Body(self.objects[Key])}

    def put_object(self, **kwargs: Any) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        self.puts.append(kwargs)


def _store(client: FakeS3Client, threshold: int = 2) -> S3EvidenceStore:
    breaker = CircuitBreaker("s3", threshold, 60, 30, 1)
    return S3EvidenceStore(client, "bucket", breaker)


def test_get_devuelve_los_bytes() -> None:
    client = FakeS3Client()
    client.objects[KEY] = b"ciphertext"

    assert _store(client).get(KEY) == b"ciphertext"


def test_put_sube_con_content_type() -> None:
    client = FakeS3Client()

    _store(client).put(REDACTED_KEY, b"jpeg", "image/jpeg")

    assert client.puts == [
        {"Bucket": "bucket", "Key": REDACTED_KEY, "Body": b"jpeg", "ContentType": "image/jpeg"}
    ]


def test_put_rechaza_claves_fuera_de_redacted() -> None:
    client = FakeS3Client()

    with pytest.raises(ValueError, match="redactados"):
        _store(client).put(KEY, b"x", "image/jpeg")

    assert client.calls == 0


def test_no_such_key_es_objeto_inexistente_y_no_abre_el_circuito() -> None:
    client = FakeS3Client()
    client.error = _client_error(404, "NoSuchKey")
    store = _store(client, threshold=1)

    for _ in range(3):
        with pytest.raises(ObjectNotFoundError):
            store.get(KEY)

    assert store.breaker.state == CircuitState.CLOSED


def test_otros_4xx_son_transitorios_pero_no_abren_el_circuito() -> None:
    client = FakeS3Client()
    client.error = _client_error(403, "AccessDenied")
    store = _store(client, threshold=1)

    for _ in range(3):
        with pytest.raises(TransientError):
            store.get(KEY)

    assert store.breaker.state == CircuitState.CLOSED


@pytest.mark.parametrize(
    "error",
    [
        _client_error(500, "InternalError"),
        _client_error(503, "SlowDown"),
        _client_error(429, "Throttling"),
        ReadTimeoutError(endpoint_url="https://s3"),
        ConnectTimeoutError(endpoint_url="https://s3"),
        EndpointConnectionError(endpoint_url="https://s3"),
    ],
)
def test_timeouts_conexion_5xx_y_429_abren_el_circuito(error: Exception) -> None:
    client = FakeS3Client()
    client.error = error
    store = _store(client, threshold=2)

    for _ in range(2):
        with pytest.raises(TransientError):
            store.get(KEY)

    assert store.breaker.state == CircuitState.OPEN


def test_con_el_circuito_abierto_no_se_llama_a_s3() -> None:
    client = FakeS3Client()
    client.error = _client_error(500)
    store = _store(client, threshold=1)
    with pytest.raises(TransientError):
        store.get(KEY)
    calls = client.calls

    with pytest.raises(CircuitOpenError):
        store.get(KEY)
    with pytest.raises(CircuitOpenError):
        store.put(REDACTED_KEY, b"x", "image/jpeg")

    assert client.calls == calls


def test_los_mensajes_de_error_no_incluyen_la_clave() -> None:
    client = FakeS3Client()
    client.error = _client_error(500)

    with pytest.raises(TransientError) as exc:
        _store(client).get(KEY)

    assert KEY not in str(exc.value)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_client_error(500), True),
        (_client_error(429), True),
        (_client_error(404), False),
        (_client_error(403), False),
        (ReadTimeoutError(endpoint_url="x"), True),
        (ValueError("x"), False),
    ],
)
def test_clasificacion_de_fallas(error: Exception, expected: bool) -> None:
    assert is_breaker_failure(error) is expected


def test_el_cliente_tiene_timeouts_explicitos_y_no_reintenta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    client = build_client("us-east-1", 2500)

    config = client.meta.config
    assert config.connect_timeout == 2.5
    assert config.read_timeout == 2.5
    assert config.retries is not None
    assert config.retries["total_max_attempts"] == 1
    assert client.meta.endpoint_url.endswith("amazonaws.com")
