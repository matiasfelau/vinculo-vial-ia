"""Paso 8: plan de redacción → relleno opaco → JPEG sin EXIF → subida a `redacted/`."""

import logging
from dataclasses import replace

import numpy as np
import pytest

from tests.fakes import (
    BRIGHT,
    GENERAL,
    PLATE,
    REDACTION,
    REPORTED_PLATE,
    THRESHOLDS,
    FakeModels,
    FakeWriter,
    Scenario,
    build_scenario,
    car_box,
    happy_models,
    plate_box,
)
from vvi_ia.adapters.imaging import CvImageDecoder, JpegRenderer
from vvi_ia.adapters.models.null import (
    NullCarSegmenter,
    NullCurbSegmenter,
    NullDamageDetector,
    NullEmbedder,
    NullFaceDetector,
    NullPlateReader,
    NullVehicleDetector,
)
from vvi_ia.domain.model import (
    AlertType,
    Box,
    BoxLabel,
    EvidenceRole,
    Pixels,
    ReasonCode,
    ValidationResult,
    Verdict,
)
from vvi_ia.domain.verdict import is_fraud
from vvi_ia.service.ports import TransientError
from vvi_ia.service.validate_evidence import Models, RedactionSettings, ValidateEvidence

DARK = 40  # el relleno opaco es negro: con JPEG queda muy por debajo del brillo original
LIGHT = 170
OTHER_TEXT = "ZZ999ZZ"

FACE = Box(70, 5, 90, 25, 0.9)
PLATE_A = Box(10, 10, 60, 30, 0.9, BoxLabel.PLATE)  # coincide con la reportada (recorte de 20x50)
PLATE_B = Box(10, 50, 70, 80, 0.9, BoxLabel.PLATE)  # otra patente (recorte de 30x60)


def _build(
    scenario: Scenario,
    models: FakeModels | Models,
    writer: FakeWriter | None = None,
    redaction: RedactionSettings = REDACTION,
) -> tuple[ValidateEvidence, FakeWriter]:
    writer = writer or FakeWriter()
    resolved = models.as_models() if isinstance(models, FakeModels) else models
    use_case = ValidateEvidence(
        store=scenario.store,
        writer=writer,
        decryptor=scenario.decryptor,
        decoder=scenario.decoder,
        renderer=JpegRenderer(),
        models=resolved,
        thresholds=THRESHOLDS,
        redaction=redaction,
    )
    return use_case, writer


def _run(
    scenario: Scenario,
    models: FakeModels | Models,
    writer: FakeWriter | None = None,
    redaction: RedactionSettings = REDACTION,
) -> tuple[ValidationResult, FakeWriter]:
    use_case, writer = _build(scenario, models, writer, redaction)
    return use_case.execute(scenario.payload), writer


def _uploaded(writer: FakeWriter, scenario: Scenario, name: str) -> Pixels:
    body, _ = writer.objects[f"alerts/{scenario.alert_id}/redacted/{name}.jpg"]
    return CvImageDecoder().decode(body)


def _by_text(crop: Pixels) -> tuple[str, float] | None:
    """Lector de prueba: la patente de 20x50 es la reportada; cualquier otra es ajena."""
    return (REPORTED_PLATE, 0.9) if crop.shape[:2] == (20, 50) else (OTHER_TEXT, 0.9)


def _models_with_two_plates() -> FakeModels:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = [
        Box(5, 5, 95, 95, 0.9, BoxLabel.CAR),
        PLATE_A,
        PLATE_B,
    ]
    models.plate.override = _by_text
    return models


STRAY = Box(0, 0, 20, 20, 0.5, BoxLabel.PLATE)  # una patente ajena de 20x20 que el OCR no lee


def _with_a_stray_plate(models: FakeModels, markers: tuple[int, ...]) -> None:
    """Agrega a cada foto una patente ajena en (0, 0, 20, 20): se tapa, y se nota en el píxel (10, 10)."""
    for marker in markers:
        models.vehicle.by_marker[marker] = [*(models.vehicle.by_marker[marker] or []), STRAY]
    models.plate.override = lambda crop: None if crop.shape[:2] == (20, 20) else (REPORTED_PLATE, 0.95)


def _g(img: Pixels, x: int, y: int) -> int:
    return int(img[y, x, 1])  # canal G: 200 en la foto original, 0 bajo el relleno


# --- cobertura -----------------------------------------------------------------------------


def test_cubre_las_patentes_ajenas_pero_no_la_que_coincide() -> None:
    scenario = build_scenario()

    result, writer = _run(scenario, _models_with_two_plates())

    assert result.verdict == Verdict.GREEN
    general = _uploaded(writer, scenario, "general-0")
    assert _g(general, 40, 65) < DARK  # patente ajena
    assert _g(general, 35, 20) > LIGHT  # patente que coincide: se deja ver
    assert _g(general, 95, 95) > LIGHT  # el resto de la foto queda intacto


def test_los_rostros_no_se_redactan_y_el_detector_de_rostros_no_se_llama() -> None:
    """Riesgo aceptado (16/11/2026): el producto tapa solo patentes; el modelo de rostros es experimental."""
    scenario = build_scenario()
    models = _models_with_two_plates()
    models.face.by_marker[GENERAL] = [FACE]  # aunque hubiera un detector de rostros que respondiera

    result, writer = _run(scenario, models)

    assert result.verdict == Verdict.GREEN
    general = _uploaded(writer, scenario, "general-0")
    assert _g(general, 80, 15) > LIGHT  # el rostro queda a la vista
    assert _g(general, 40, 65) < DARK  # y la patente ajena sigue tapada
    assert models.face.calls == 0


def test_la_redaccion_no_depende_del_modelo_de_rostros() -> None:
    """Con el detector de rostros nulo (None en todas las fotos) la redacción se genera igual."""
    models = happy_models()
    assert models.face.by_marker == {}  # None siempre: como NullFaceDetector

    result, writer = _run(build_scenario(), models)

    assert result.verdict == Verdict.GREEN
    assert ReasonCode.REDACTION_UNAVAILABLE not in result.reason_codes
    assert len(result.redacted) == 2 and len(writer.puts) == 2
    assert models.face.calls == 0


def test_el_relleno_incluye_el_margen_del_diez_por_ciento() -> None:
    scenario = build_scenario()

    _, writer = _run(scenario, _models_with_two_plates())

    general = _uploaded(writer, scenario, "general-0")
    # PLATE_B mide 60x30: el margen es 6 px a los lados y 3 px arriba/abajo (x 4..76, y 47..83)
    assert _g(general, 7, 65) < DARK
    assert _g(general, 40, 49) < DARK
    assert _g(general, 1, 65) > LIGHT


def test_patente_con_un_caracter_de_diferencia_tambien_se_considera_la_reportada() -> None:
    scenario = build_scenario()
    models = _models_with_two_plates()
    models.plate.override = lambda crop: (
        (REPORTED_PLATE[:-1] + "9", 0.9)
        if crop.shape[:2] == (20, 50)
        else (
            OTHER_TEXT,
            0.9,
        )
    )

    _, writer = _run(scenario, models)

    assert _g(_uploaded(writer, scenario, "general-0"), 35, 20) > LIGHT


def test_patente_con_lectura_de_baja_confianza_se_tapa() -> None:
    scenario = build_scenario()
    models = _models_with_two_plates()
    models.plate.override = lambda crop: (
        (REPORTED_PLATE, 0.2) if crop.shape[:2] == (20, 50) else (OTHER_TEXT, 0.9)
    )

    _, writer = _run(scenario, models)

    assert _g(_uploaded(writer, scenario, "general-0"), 35, 20) < DARK


@pytest.mark.parametrize("alert_type", [AlertType.CRASHED, AlertType.PARKED_ON_GARAGE])
def test_se_redactan_las_dos_fotos_de_la_alerta(alert_type: AlertType) -> None:
    scenario = build_scenario(alert_type)
    models = happy_models()
    _with_a_stray_plate(models, (GENERAL, PLATE))

    result, writer = _run(scenario, models)

    assert [(r.role, r.index) for r in result.redacted] == [
        (EvidenceRole.GENERAL, 0),
        (EvidenceRole.PLATE, 0),
    ]
    assert sorted(writer.objects) == sorted(
        f"alerts/{scenario.alert_id}/redacted/{name}.jpg" for name in ("general-0", "plate-0")
    )
    for name in ("general-0", "plate-0"):
        assert _g(_uploaded(writer, scenario, name), 10, 10) < DARK


def test_el_relleno_es_opaco_y_no_un_blur() -> None:
    scenario = build_scenario()

    _, writer = _run(scenario, _models_with_two_plates())

    region = _uploaded(writer, scenario, "general-0")[55:75, 15:65, 1]
    assert region.max() < DARK  # ningún píxel conserva información de la foto original


# --- fail-closed ---------------------------------------------------------------------------


def _nothing_uploaded(result: ValidationResult, writer: FakeWriter) -> None:
    assert result.redacted == ()
    assert writer.puts == []


def _unavailable(result: ValidationResult, writer: FakeWriter) -> None:
    """Fail-closed: RED por REDACTION_UNAVAILABLE (sin fraude) y ninguna foto subida."""
    assert result.verdict == Verdict.RED
    assert ReasonCode.REDACTION_UNAVAILABLE in result.reason_codes
    assert not is_fraud(result.reason_codes)
    _nothing_uploaded(result, writer)


def test_sin_detector_de_patentes_no_se_genera_ninguna_version() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = None  # la redacción necesita sus patentes; los rostros no cuentan

    result, writer = _run(build_scenario(), models)

    assert ReasonCode.REDACTION_UNAVAILABLE in result.reason_codes
    _unavailable(result, writer)
    assert models.face.calls == 0


def test_sin_detector_de_vehiculos_en_una_foto_no_se_redacta_ninguna() -> None:
    models = happy_models()
    models.vehicle.by_marker[PLATE] = None  # el detector (patentes) falla en una sola foto

    result, writer = _run(build_scenario(), models)

    _unavailable(result, writer)


def test_una_patente_que_el_ocr_no_lee_se_tapa_y_la_redaccion_sigue_disponible() -> None:
    scenario = build_scenario()
    models = _models_with_two_plates()
    models.plate.override = lambda crop: None  # hay cajas de patente en las fotos y el OCR no las lee

    result, writer = _run(scenario, models)

    assert len(result.redacted) == 2 and len(writer.puts) == 2
    assert ReasonCode.REDACTION_UNAVAILABLE not in result.reason_codes
    general = _uploaded(writer, scenario, "general-0")
    assert _g(general, 40, 15) < DARK  # la patente que no se pudo leer se tapa, aunque sea la reportada


def test_sin_patentes_cuenta_como_redactada() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = [car_box()]  # ninguna patente
    models.vehicle.by_marker[PLATE] = [car_box()]

    result, writer = _run(build_scenario(), models)

    assert len(result.redacted) == 2 and len(writer.puts) == 2
    assert ReasonCode.REDACTION_UNAVAILABLE not in result.reason_codes


def test_si_falla_la_reescritura_de_una_foto_no_se_sube_ninguna() -> None:
    class FailsOnSecond:
        def __init__(self) -> None:
            self.calls = 0

        def render(self, pixels: Pixels, boxes: object, max_side: int) -> bytes:
            self.calls += 1
            if self.calls == 2:
                raise ValueError("detalle interno con datos de la imagen")
            return b"jpeg"

    scenario = build_scenario()
    writer = FakeWriter()
    use_case, _ = _build(scenario, happy_models(), writer)
    use_case._renderer = FailsOnSecond()  # type: ignore[assignment]

    result = use_case.execute(scenario.payload)

    _unavailable(result, writer)


def test_con_todos_los_modelos_en_none_no_hay_version_redactada() -> None:
    null_models = Models(
        NullVehicleDetector(),
        NullCarSegmenter(),
        NullCurbSegmenter(),
        NullEmbedder(),
        NullPlateReader(),
        NullDamageDetector(),
        NullFaceDetector(),
    )

    result, writer = _run(build_scenario(), null_models)

    assert result.verdict == Verdict.RED  # sin redacción no se aprueba
    assert set(result.reason_codes) == {ReasonCode.REDACTION_UNAVAILABLE}
    _unavailable(result, writer)


def test_un_veredicto_red_no_genera_version_redactada() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = (OTHER_TEXT, 0.95)  # PLATE_MISMATCH

    result, writer = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED
    assert ReasonCode.PLATE_MISMATCH in result.reason_codes
    _nothing_uploaded(result, writer)


def test_evidencia_ilegible_no_genera_version_redactada() -> None:
    scenario = build_scenario()
    scenario.corrupt_sha(EvidenceRole.GENERAL)

    result, writer = _run(scenario, happy_models())

    assert result.reason_codes == (ReasonCode.EVIDENCE_UNREADABLE,)
    _nothing_uploaded(result, writer)


def test_una_patente_casi_igual_es_green_y_genera_version_redactada() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = (REPORTED_PLATE[:-1] + "9", 0.95)  # casi igual

    result, writer = _run(build_scenario(), models)

    assert result.verdict == Verdict.GREEN
    assert len(writer.puts) == 2


# --- salida: claves, JPEG, EXIF, tamaño ---------------------------------------------------


def test_sube_a_redacted_con_content_type_jpeg_y_devuelve_las_claves() -> None:
    scenario = build_scenario()

    result, writer = _run(scenario, happy_models())

    keys = [f"alerts/{scenario.alert_id}/redacted/{name}.jpg" for name in ("general-0", "plate-0")]
    assert [r.key for r in result.redacted] == keys
    assert sorted(writer.objects) == sorted(keys)
    for body, content_type in writer.objects.values():
        assert content_type == "image/jpeg"
        assert body.startswith(b"\xff\xd8\xff")


def test_el_jpeg_resultante_no_tiene_exif() -> None:
    scenario = build_scenario()

    _, writer = _run(scenario, happy_models())

    for body, _ in writer.objects.values():
        assert b"Exif" not in body
        assert b"\xff\xe1" not in body  # ningún segmento APP1


def test_el_lado_mayor_se_limita_a_redact_max_side_px() -> None:
    scenario = build_scenario()

    _, writer = _run(scenario, happy_models(), redaction=replace(REDACTION, max_side_px=50))

    for name in ("general-0", "plate-0"):
        assert max(_uploaded(writer, scenario, name).shape[:2]) == 50


def test_reprocesar_sobrescribe_las_mismas_claves() -> None:
    scenario = build_scenario()
    use_case, writer = _build(scenario, happy_models())

    first = use_case.execute(scenario.payload)
    second = use_case.execute(scenario.payload)

    assert first == second
    assert len(writer.objects) == 2
    assert len(writer.puts) == 4


def test_una_falla_transitoria_al_subir_se_propaga_para_reintentar() -> None:
    writer = FakeWriter()
    writer.error = TransientError("s3 caído")

    with pytest.raises(TransientError):
        _run(build_scenario(), happy_models(), writer)


# --- REDACTION_REQUIRED=false (solo ENV=local) ---------------------------------------------


def test_sin_redaccion_requerida_sube_la_imagen_re_encodeada_sin_tapar(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scenario = build_scenario()
    models = _models_with_two_plates()
    detections_before = models.vehicle.calls

    with caplog.at_level(logging.WARNING):
        result, writer = _run(scenario, models, redaction=replace(REDACTION, required=False))

    assert len(result.redacted) == 2
    general = _uploaded(writer, scenario, "general-0")
    assert _g(general, 40, 65) > LIGHT  # la patente ajena no se tapa
    assert b"Exif" not in writer.objects[result.redacted[0].key][0]
    warning = next(r for r in caplog.records if "redacción desactivada" in r.getMessage())
    assert warning.levelno == logging.WARNING
    assert warning.__dict__["alert_id"] == scenario.alert_id
    assert models.vehicle.calls >= detections_before


def test_sin_modelos_y_sin_redaccion_requerida_sube_todo_para_probar_el_visor() -> None:
    null_models = Models(
        NullVehicleDetector(),
        NullCarSegmenter(),
        NullCurbSegmenter(),
        NullEmbedder(),
        NullPlateReader(),
        NullDamageDetector(),
        NullFaceDetector(),
    )

    result, writer = _run(build_scenario(), null_models, redaction=replace(REDACTION, required=False))

    assert result.verdict == Verdict.GREEN
    assert len(result.redacted) == 2
    assert len(writer.puts) == 2


def test_sin_redaccion_requerida_un_red_tampoco_sube_nada() -> None:
    scenario = build_scenario()
    scenario.corrupt_sha(EvidenceRole.PLATE)

    result, writer = _run(scenario, happy_models(), redaction=replace(REDACTION, required=False))

    _nothing_uploaded(result, writer)


# --- privacidad ----------------------------------------------------------------------------


def test_los_logs_de_la_redaccion_no_contienen_la_patente_ni_el_texto_del_ocr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from vvi_ia.logging_setup import JsonFormatter

    with caplog.at_level(logging.DEBUG):
        _run(build_scenario(), _models_with_two_plates())
        _run(
            build_scenario(),
            _models_with_two_plates(),
            redaction=replace(REDACTION, required=False),
        )

    output = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert caplog.records
    assert REPORTED_PLATE not in output
    assert OTHER_TEXT not in output


def test_la_foto_original_en_memoria_no_se_modifica() -> None:
    scenario = build_scenario()
    seen: list[Pixels] = []
    decode = scenario.decoder.decode

    def spy(data: bytes) -> Pixels:
        pixels = decode(data)
        seen.append(pixels)
        return pixels

    scenario.decoder.decode = spy  # type: ignore[method-assign]

    _run(scenario, _models_with_two_plates())

    general = seen[0]
    assert np.all(general[..., 1] == BRIGHT)
    assert plate_box().label == BoxLabel.PLATE


# --- fotos planificadas en paralelo --------------------------------------------------------


def _in_parallel(workers: int) -> RedactionSettings:
    return RedactionSettings(required=True, plate_min_conf=0.20, max_side_px=1600, workers=workers)


def test_planificar_en_paralelo_sube_las_mismas_fotos_que_una_por_una() -> None:
    def uploaded(workers: int) -> dict[str, bytes]:
        scenario = build_scenario(AlertType.CRASHED)
        models = _models_with_two_plates()
        _with_a_stray_plate(models, (PLATE,))
        models.plate.override = lambda crop: (
            None if crop.shape[:2] == (20, 20) else _by_text(crop)
        )  # la ajena de 20x20 no se lee: se tapa
        result, writer = _run(scenario, models, redaction=_in_parallel(workers))
        assert [(r.role, r.index) for r in result.redacted][0] == (EvidenceRole.GENERAL, 0)
        return {key.rsplit("/", 1)[1]: body for key, (body, _) in writer.objects.items()}

    assert uploaded(3) == uploaded(1)


def test_en_paralelo_si_falla_una_sola_foto_no_se_sube_ninguna() -> None:
    models = happy_models()
    models.vehicle.by_marker[PLATE] = None  # falla el detector de patentes en esa foto

    result, writer = _run(build_scenario(), models, redaction=_in_parallel(3))

    _unavailable(result, writer)


# --- cajas contenidas en la patente exceptuada ------------------------------------------------------


NESTED = Box(20, 12, 50, 28, 0.6, BoxLabel.PLATE)  # un pedazo de PLATE_A (la reportada), de 16x30


def test_una_deteccion_contenida_en_la_patente_denunciada_no_la_tapa() -> None:
    """El mosaico detecta dos veces la misma patente: la caja de adentro no es una patente ajena."""
    scenario = build_scenario()
    models = _models_with_two_plates()
    models.vehicle.by_marker[GENERAL] = [*(models.vehicle.by_marker[GENERAL] or []), NESTED]

    _, writer = _run(scenario, models)

    general = _uploaded(writer, scenario, "general-0")
    assert _g(general, 35, 20) > LIGHT  # la patente denunciada queda a la vista, también bajo el pedazo
    assert _g(general, 40, 65) < DARK  # la ajena sigue tapada
