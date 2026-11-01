from aiohttp.test_utils import TestClient, TestServer

from vvi_ia.adapters.health import DOWN, UP, HealthServer

MODELS = {"detector": "null", "ocr": "parseq@0123456789ab"}


async def _client(ready: bool, deps: dict[str, str]) -> TestClient:  # type: ignore[type-arg]
    server = HealthServer(port=0, ready=lambda: ready, deps=lambda: deps, models=MODELS)
    client = TestClient(TestServer(server.build_app()))
    await client.start_server()
    return client


async def test_health_siempre_ok() -> None:
    client = await _client(False, {"rabbitmq": DOWN})
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        assert await resp.json() == {"status": "ok", "models": MODELS}
    finally:
        await client.close()


async def test_ready_503_si_el_consumidor_no_esta_activo() -> None:
    client = await _client(False, {})
    try:
        resp = await client.get("/ready")
        assert resp.status == 503
        assert (await resp.json())["status"] == "not_ready"
    finally:
        await client.close()


async def test_ready_200_con_el_consumidor_activo() -> None:
    client = await _client(True, {})
    try:
        resp = await client.get("/ready")
        assert resp.status == 200
    finally:
        await client.close()


async def test_deps_ok_cuando_todo_esta_up() -> None:
    client = await _client(True, {"rabbitmq": UP, "s3": UP})
    try:
        resp = await client.get("/health/deps")
        assert resp.status == 200
        assert await resp.json() == {"status": "ready", "checks": {"rabbitmq": UP, "s3": UP}}
    finally:
        await client.close()


async def test_deps_degraded_503_si_una_dependencia_esta_caida() -> None:
    client = await _client(True, {"rabbitmq": UP, "s3": DOWN})
    try:
        resp = await client.get("/health/deps")
        assert resp.status == 503
        assert (await resp.json())["status"] == "degraded"
    finally:
        await client.close()
