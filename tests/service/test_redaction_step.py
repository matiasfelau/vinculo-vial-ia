"""Paso 8: plan de redacción → relleno opaco → JPEG sin EXIF → subida a `redacted/`."""

import logging
from dataclasses import replace

import numpy as np
import pytest

from tests.fakes import (
    BRIGHT,
    CONTEXT,
    GENERAL,
    PLATE,
    REDACTION,
    REPORTED_PLATE,
    THRESHOLDS,
    FakeModels,
    FakeWriter,
    Scenario,
    build_scenario,
    detail_marker,
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


def _models_with_two_plates_and_a_face() -> FakeModels:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = [
        Box(5, 5, 95, 95, 0.9, BoxLabel.CAR),
        PLATE_A,
        PLATE_B,
    ]
    models.face.by_marker[GENERAL] = [FACE]
    models.plate.override = _by_text
    return models


def _g(img: Pixels, x: int, y: int) -> int:
    return int(img[y, x, 1])  # canal G: 200 en la foto original, 0 bajo el relleno


# --- cobertura -----------------------------------------------------------------------------


def test_cubre_rostros_y_patentes_ajenas_pero_no_la_que_coincide() -> None:
    scenario = build_scenario()

    result, writer = _run(scenario, _models_with_two_plates_and_a_face())

    assert result.verdict == Verdict.GREEN
    general = _uploaded(writer, scenario, "general-0")
    assert _g(general, 80, 15) < DARK  # rostro
    assert _g(general, 40, 65) < DARK  # patente ajena
    assert _g(general, 35, 20) > LIGHT  # patente que coincide: se deja ver
    assert _g(general, 95, 95) > LIGHT  # el resto de la foto queda intacto


def test_el_relleno_incluye_el_margen_del_diez_por_ciento() -> None:
    scenario = build_scenario()

    _, writer = _run(scenario, _models_with_two_plates_and_a_face())

    general = _uploaded(writer, scenario, "general-0")
    # PLATE_B mide 60x30: el margen es 6 px a los lados y 3 px arriba/abajo (x 4..76, y 47..83)
    assert _g(general, 7, 65) < DARK
    assert _g(general, 40, 49) < DARK
    assert _g(general, 1, 65) > LIGHT


def test_patente_con_un_caracter_de_diferencia_tambien_se_considera_la_reportada() -> None:
    scenario = build_scenario()
    models = _models_with_two_plates_and_a_face()
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
    models = _models_with_two_plates_and_a_face()
    models.plate.override = lambda crop: (
        (REPORTED_PLATE, 0.2) if crop.shape[:2] == (20, 50) else (OTHER_TEXT, 0.9)
    )

    _, writer = _run(scenario, models)

    assert _g(_uploaded(writer, scenario, "general-0"), 35, 20) < DARK


def test_se_redactan_todas_las_fotos_de_la_alerta() -> None:
    scenario = build_scenario(AlertType.CRASHED, details=2)
    models = happy_models()
    for marker in (GENERAL, PLATE, detail_marker(0), detail_marker(1)):
        models.face.by_marker[marker] = [Box(0, 0, 20, 20, 0.9)]
    models.damage.by_marker[detail_marker(1)] = [Box(0, 0, 9, 9, 0.8)]

    result, writer = _run(scenario, models)

    assert [(r.role, r.index) for r in result.redacted] == [
        (EvidenceRole.GENERAL, 0),
        (EvidenceRole.PLATE, 0),
        (EvidenceRole.DETAIL, 0),
        (EvidenceRole.DETAIL, 1),
    ]
    for name in ("general-0", "plate-0", "detail-0", "detail-1"):
        assert _g(_uploaded(writer, scenario, name), 10, 10) < DARK


def test_garage_redacta_la_foto_de_contexto() -> None:
    scenario = build_scenario(AlertType.PARKED_ON_GARAGE)
    models = happy_models()
    models.face.by_marker[CONTEXT] = [Box(0, 0, 20, 20, 0.9)]

    result, writer = _run(scenario, models)

    assert [r.role for r in result.redacted] == [
        EvidenceRole.GENERAL,
        EvidenceRole.PLATE,
        EvidenceRole.CONTEXT,
    ]
    assert _g(_uploaded(writer, scenario, "context-0"), 10, 10) < DARK


def test_el_relleno_es_opaco_y_no_un_blur() -> None:
    scenario = build_scenario()

    _, writer = _run(scenario, _models_with_two_plates_and_a_face())

    region = _uploaded(writer, scenario, "general-0")[55:75, 15:65, 1]
    assert region.max() < DARK  # ningún píxel conserva información de la foto original


# --- fail-closed ---------------------------------------------------------------------------


def _nothing_uploaded(result: ValidationResult, writer: FakeWriter) -> None:
    assert result.redacted == ()
    assert writer.puts == []


def test_sin_detector_de_rostros_no_se_genera_ninguna_version() -> None:
    models = happy_models()
    models.face.by_marker[GENERAL] = None

    result, writer = _run(build_scenario(), models)

    assert result.verdict == Verdict.GREEN
    _nothing_uploaded(result, writer)


def test_sin_detector_de_vehiculos_en_una_foto_no_se_redacta_ninguna() -> None:
    models = happy_models()
    models.vehicle.by_marker[detail_marker(0)] = None

    result, writer = _run(build_scenario(), models)

    _nothing_uploaded(result, writer)


def test_sin_lector_de_patentes_no_se_genera_ninguna_version() -> None:
    models = happy_models()
    models.plate.by_marker[GENERAL] = None  # hay una caja de patente en GENERAL y no se puede leer

    result, writer = _run(build_scenario(), models)

    _nothing_uploaded(result, writer)


def test_si_falla_una_sola_foto_tampoco_se_sube_ninguna_de_las_otras() -> None:
    models = happy_models()
    models.face.by_marker[detail_marker(0)] = None

    result, writer = _run(build_scenario(), models)

    _nothing_uploaded(result, writer)


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

    assert result.verdict == Verdict.YELLOW
    _nothing_uploaded(result, writer)


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


def test_yellow_si_genera_version_redactada() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = (REPORTED_PLATE[:-1] + "9", 0.95)  # PLATE_FUZZY

    result, writer = _run(build_scenario(), models)

    assert result.verdict == Verdict.YELLOW
    assert len(writer.puts) == 3


# --- salida: claves, JPEG, EXIF, tamaño ---------------------------------------------------


def test_sube_a_redacted_con_content_type_jpeg_y_devuelve_las_claves() -> None:
    scenario = build_scenario()

    result, writer = _run(scenario, happy_models())

    keys = [
        f"alerts/{scenario.alert_id}/redacted/{name}.jpg" for name in ("general-0", "plate-0", "detail-0")
    ]
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

    for name in ("general-0", "plate-0", "detail-0"):
        assert max(_uploaded(writer, scenario, name).shape[:2]) == 50


def test_reprocesar_sobrescribe_las_mismas_claves() -> None:
    scenario = build_scenario()
    use_case, writer = _build(scenario, happy_models())

    first = use_case.execute(scenario.payload)
    second = use_case.execute(scenario.payload)

    assert first == second
    assert len(writer.objects) == 3
    assert len(writer.puts) == 6


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
    models = _models_with_two_plates_and_a_face()
    detections_before = models.vehicle.calls

    with caplog.at_level(logging.WARNING):
        result, writer = _run(scenario, models, redaction=replace(REDACTION, required=False))

    assert len(result.redacted) == 3
    general = _uploaded(writer, scenario, "general-0")
    assert _g(general, 80, 15) > LIGHT  # el rostro no se tapa
    assert _g(general, 40, 65) > LIGHT  # la patente ajena tampoco
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

    assert result.verdict == Verdict.YELLOW
    assert len(result.redacted) == 3
    assert len(writer.puts) == 3


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
        _run(build_scenario(), _models_with_two_plates_and_a_face())
        _run(
            build_scenario(),
            _models_with_two_plates_and_a_face(),
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

    _run(scenario, _models_with_two_plates_and_a_face())

    general = seen[0]
    assert np.all(general[..., 1] == BRIGHT)
    assert plate_box().label == BoxLabel.PLATE
