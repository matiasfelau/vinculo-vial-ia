"""Servidor HTTP interno de salud (nunca se expone en Kong)."""

import logging
from collections.abc import Callable, Mapping

from aiohttp import web

logger = logging.getLogger(__name__)

UP = "up"
DOWN = "down"
BIND_HOST = "0.0.0.0"  # noqa: S104 - escucha dentro del contenedor; el puerto no se publica al host


class HealthServer:
    """Expone /health (liveness), /ready y /health/deps.

    `ready` indica si el consumidor está activo; `deps` devuelve el estado `up`/`down`
    de cada dependencia (broker, circuit breaker de S3).
    """

    def __init__(
        self,
        port: int,
        ready: Callable[[], bool],
        deps: Callable[[], Mapping[str, str]],
    ) -> None:
        self._port = port
        self._ready = ready
        self._deps = deps
        self._runner: web.AppRunner | None = None

    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.get("/health", self._health),
                web.get("/ready", self._readiness),
                web.get("/health/deps", self._health_deps),
            ]
        )
        return app

    async def start(self) -> None:
        self._runner = web.AppRunner(self.build_app(), access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, BIND_HOST, self._port).start()
        logger.info("servidor de salud iniciado", extra={"port": self._port})

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _readiness(self, _request: web.Request) -> web.Response:
        if self._ready():
            return web.json_response({"status": "ready"})
        return web.json_response({"status": "not_ready"}, status=503)

    async def _health_deps(self, _request: web.Request) -> web.Response:
        checks = dict(self._deps())
        if all(state == UP for state in checks.values()):
            return web.json_response({"status": "ready", "checks": checks})
        return web.json_response({"status": "degraded", "checks": checks}, status=503)
