"""Arranque del servicio: config → adaptadores → consumidor + servidor de salud."""

import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError

from vvi_ia.adapters.breaker import CircuitBreaker, CircuitState
from vvi_ia.adapters.broker import EventPublisher, EvidenceConsumer, EvidenceMessageHandler
from vvi_ia.adapters.health import DOWN, UP, HealthServer
from vvi_ia.adapters.hpke import HpkeDecryptor
from vvi_ia.adapters.imaging import CvImageDecoder, JpegRenderer
from vvi_ia.adapters.models.registry import ModelSetupError, build_models
from vvi_ia.adapters.storage_s3 import S3EvidenceStore, build_client
from vvi_ia.config import Settings, load_settings
from vvi_ia.domain.model import ValidationResult
from vvi_ia.logging_setup import configure_logging
from vvi_ia.service.validate_evidence import (
    Models,
    RedactionSettings,
    Thresholds,
    ValidateEvidence,
)

logger = logging.getLogger(__name__)

EXIT_BAD_CONFIG = 2
EXIT_BAD_MODELS = 3
MS_PER_SECOND = 1000.0


def load_models(settings: Settings) -> Models:
    """Modelos de inferencia según `MODEL_*`. Verifica el sha256 de cada modelo seleccionado."""
    return build_models(
        selection=settings.model_selection(),
        manifest_path=settings.models_manifest,
        models_dir=settings.models_dir,
        intra_op_threads=settings.onnx_intra_op_threads,
        params=adapter_params(settings),
    )


def adapter_params(settings: Settings) -> dict[str, float]:
    """Parámetros que cada adaptador de modelo lee de su configuración."""
    return {
        "segmenter_min_conf": settings.segmenter_min_conf,
        "detector_min_conf": settings.detector_min_conf,
        "damage_min_conf": settings.damage_t_low,
        "curb_min_conf": settings.curb_min_conf,
        "face_tiling": float(settings.face_tiling),
        "plate_tiling": float(settings.plate_tiling),
        "tiling_tile_px": float(settings.tiling_tile_px),
        "tiling_overlap": settings.tiling_overlap,
        # Rostros y patentes salen desde el umbral de redacción más bajo: el dominio decide cuáles tapar
        "redaction_min_conf": min(settings.face_min_conf, settings.plate_redact_min_conf),
    }


def build_thresholds(settings: Settings) -> Thresholds:
    return Thresholds(
        reid_t_low=settings.reid_t_low,
        reid_t_high=settings.reid_t_high,
        ocr_min_conf=settings.ocr_min_conf,
        damage_t_low=settings.damage_t_low,
        damage_t_high=settings.damage_t_high,
        presence_min_area=settings.presence_min_area,
        presence_plate_min_conf=settings.detector_min_conf,
        ramp_t_low=settings.ramp_t_low,
        ramp_t_high=settings.ramp_t_high,
        ramp_scale_fallback=settings.ramp_scale_fallback,
    )


def build_redaction(settings: Settings) -> RedactionSettings:
    return RedactionSettings(
        required=settings.redaction_required,
        face_min_conf=settings.face_min_conf,
        plate_min_conf=settings.plate_redact_min_conf,
        max_side_px=settings.redact_max_side_px,
    )


async def run(settings: Settings, models: Models) -> None:
    """Levanta el servicio y espera SIGINT/SIGTERM para apagar en orden."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows no soporta add_signal_handler
            loop.add_signal_handler(sig, stop.set)

    pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="inference")
    breaker = CircuitBreaker(
        name="s3",
        failure_threshold=settings.s3_breaker_failure_threshold,
        window_s=settings.s3_breaker_window_ms / MS_PER_SECOND,
        open_duration_s=settings.s3_breaker_open_duration_ms / MS_PER_SECOND,
        half_open_max_calls=settings.s3_breaker_half_open_max_calls,
    )
    store = S3EvidenceStore(
        build_client(settings.s3_region, settings.s3_timeout_ms),
        settings.evidence_bucket,
        breaker,
        settings.evidence_max_part_bytes,
    )
    use_case = ValidateEvidence(
        store=store,
        writer=store,
        decryptor=HpkeDecryptor(settings.ia_hpke_private_key.get_secret_value()),
        decoder=CvImageDecoder(),
        renderer=JpegRenderer(),
        models=models,
        thresholds=build_thresholds(settings),
        redaction=build_redaction(settings),
    )

    async def run_blocking(operation: Callable[[], ValidationResult]) -> ValidationResult:
        return await loop.run_in_executor(pool, operation)

    def handler_factory(publisher: EventPublisher) -> EvidenceMessageHandler:
        return EvidenceMessageHandler(
            queue=settings.input_queue,
            max_retries=settings.max_retries,
            bucket=settings.evidence_bucket,
            use_case=use_case,
            publisher=publisher,
            run_blocking=run_blocking,
        )

    consumer = EvidenceConsumer(
        url=settings.rabbitmq_url.get_secret_value(),
        queue=settings.input_queue,
        main_exchange=settings.rabbitmq_exchange,
        dlq_exchange=settings.rabbitmq_dlq_exchange,
        prefetch=settings.workers,
        confirm_timeout_s=settings.publish_confirm_timeout_ms / MS_PER_SECOND,
        handler_factory=handler_factory,
    )
    health = HealthServer(
        port=settings.health_port,
        ready=lambda: consumer.consuming,
        deps=lambda: {
            "rabbitmq": UP if consumer.connected else DOWN,
            "s3": DOWN if breaker.state == CircuitState.OPEN else UP,
        },
        models=models.versions(),
    )
    await health.start()
    try:
        await consumer.start()
        logger.info("servicio iniciado", extra={"env": settings.env})
        await stop.wait()
    finally:
        logger.info("apagando el servicio")
        await consumer.stop()
        await health.stop()
        pool.shutdown(wait=True, cancel_futures=True)
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
    try:
        models = load_models(settings)
    except ModelSetupError as exc:
        logger.error("modelos inválidos: el servicio no arranca", extra={"reason": str(exc)})
        return EXIT_BAD_MODELS
    logger.info("modelos seleccionados", extra={"model_versions": models.versions()})
    asyncio.run(run(settings, models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
