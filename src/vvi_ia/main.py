"""Arranque del servicio: config → adaptadores → consumidor + servidor de salud."""

import asyncio
import contextlib
import logging
import signal
import sys

from pydantic import ValidationError

from vvi_ia.adapters.health import DOWN, HealthServer
from vvi_ia.config import Settings, load_settings
from vvi_ia.logging_setup import configure_logging

logger = logging.getLogger(__name__)

EXIT_BAD_CONFIG = 2


async def run(settings: Settings) -> None:
    """Levanta el servicio y espera SIGINT/SIGTERM para apagar en orden."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows no soporta add_signal_handler
            loop.add_signal_handler(sig, stop.set)

    health = HealthServer(
        port=settings.health_port,
        ready=lambda: False,
        deps=lambda: {"rabbitmq": DOWN},
    )
    await health.start()
    logger.info("servicio iniciado", extra={"env": settings.env})
    try:
        await stop.wait()
    finally:
        await health.stop()
        logger.info("servicio detenido")


def main() -> int:
    """Punto de entrada: valida la configuración y corre el servicio."""
    configure_logging()
    try:
        settings = load_settings()
    except ValidationError as exc:
        # Solo campo y motivo: el valor de entrada puede ser un secreto.
        for error in exc.errors(include_input=False, include_url=False):
            logger.error(
                "configuración inválida",
                extra={"field": ".".join(str(p) for p in error["loc"]), "reason": error["msg"]},
            )
        return EXIT_BAD_CONFIG
    asyncio.run(run(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
