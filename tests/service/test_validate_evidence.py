import logging
from dataclasses import replace

import pytest

from tests.fakes import (
    CONTEXT,
    GENERAL,
    PLATE,
    REDACTION,
    REPORTED_PLATE,
    THRESHOLDS,
    FakeDecryptor,
    FakeModels,
    FakeWriter,
    Scenario,
    build_scenario,
    car_box,
    detail_marker,
    happy_models,
    marker_of,
    plate_box,
    rect,
    vec,
)
from vvi_ia.adapters.imaging import JpegRenderer
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
    CarInstance,
    EvidenceRole,
    ReasonCode,
    ValidationResult,
    Verdict,
)
from vvi_ia.domain.verdict import is_fraud
from vvi_ia.logging_setup import JsonFormatter
from vvi_ia.service.ports import (
    EvidenceTooLargeError,
    ObjectNotFoundError,
    SystemDecryptionError,
    TransientError,
)
from vvi_ia.service.validate_evidence import Models, ValidateEvidence

R = ReasonCode
GARAGE = AlertType.PARKED_ON_GARAGE
CRASHED = AlertType.CRASHED


def _run(scenario: Scenario, models: FakeModels | Models | None = None) -> ValidationResult:
    resolved = models.as_models() if isinstance(models, FakeModels) else models
    use_case = ValidateEvidence(
        store=scenario.store,
        writer=FakeWriter(),
        decryptor=scenario.decryptor,
        decoder=scenario.decoder,
        renderer=JpegRenderer(),
        models=resolved or happy_models().as_models(),
        thresholds=THRESHOLDS,
        redaction=REDACTION,
    )
    return use_case.execute(scenario.payload)


def _codes(result: ValidationResult) -> set[ReasonCode]:
    return set(result.reason_codes)


def _null_models() -> Models:
    return Models(
        NullVehicleDetector(),
        NullCarSegmenter(),
        NullCurbSegmenter(),
        NullEmbedder(),
        NullPlateReader(),
        NullDamageDetector(),
        NullFaceDetector(),
    )


# --- camino feliz y adaptador null ------------------------------------------------------


@pytest.mark.parametrize("alert_type", [CRASHED, GARAGE])
def test_caso_feliz_es_green(alert_type: AlertType) -> None:
    result = _run(build_scenario(alert_type))

    assert result.verdict == Verdict.GREEN
    assert result.reason_codes == ()
    assert len(result.redacted) == 3
    assert result.scores.burst_similarity is not None
    assert result.scores.ocr_confidence == pytest.approx(0.95)
    assert result.scores.context_confidence is not None


@pytest.mark.parametrize("alert_type", [CRASHED, GARAGE])
def test_todo_none_da_red_por_redaccion_no_disponible(alert_type: AlertType) -> None:
    result = _run(build_scenario(alert_type), _null_models())

    assert result.verdict == Verdict.RED  # sin modelos de redacción no hay veredicto aprobable
    assert _codes(result) == {
        R.BURST_UNCERTAIN,
        R.PLATE_UNREADABLE,
        R.CONTEXT_UNCERTAIN,
        R.REDACTION_UNAVAILABLE,
    }
    assert result.redacted == ()
    assert set(result.model_versions) == {
        "detector",
        "segmenter",
        "embedder",
        "ocr",
        "damage",
        "curb",
        "face",
    }
    assert set(result.model_versions.values()) == {"null"}


def test_un_none_nunca_produce_fraude() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = None

    result = _run(build_scenario(), models)

    assert not is_fraud(result.reason_codes)
    assert result.verdict == Verdict.RED and R.REDACTION_UNAVAILABLE in result.reason_codes


def test_las_versiones_de_modelo_viajan_en_el_resultado() -> None:
    result = _run(build_scenario())

    assert result.model_versions["detector"] == "fake-detector"
    assert result.model_versions["ocr"] == "fake-ocr"


def test_reprocesar_es_idempotente() -> None:
    scenario = build_scenario(GARAGE)

    assert _run(scenario) == _run(scenario)


# --- E1: vehículo -----------------------------------------------------------------------


def test_ningun_auto_es_no_vehicle_y_corta_la_cascada() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = [plate_box()]

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED
    assert result.reason_codes == (R.NO_VEHICLE,)
    assert models.plate.calls == 0


def test_detector_sin_modelo_en_general_es_burst_uncertain() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = None

    assert R.BURST_UNCERTAIN in _codes(_run(build_scenario(), models))


# --- E1b: ráfaga ------------------------------------------------------------------------


def test_mismo_auto_en_ambas_fotos_no_agrega_codigos() -> None:
    assert R.BURST_MISMATCH not in _codes(_run(build_scenario()))


def test_autos_distintos_son_burst_mismatch() -> None:
    models = happy_models()
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(80)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED
    assert R.BURST_MISMATCH in _codes(result)
    assert result.scores.burst_similarity == pytest.approx(0.17, abs=0.01)


def test_similitud_intermedia_es_burst_uncertain() -> None:
    models = happy_models()
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(55)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.YELLOW
    assert _codes(result) == {R.BURST_UNCERTAIN}


@pytest.mark.parametrize("missing", ["embedder", "general_cars", "plate_cars", "general_empty"])
def test_cualquier_none_en_la_rafaga_es_uncertain(missing: str) -> None:
    models = happy_models()
    if missing == "embedder":
        models.embedder.by_marker_area[(GENERAL, 6400)] = None
    elif missing == "general_cars":
        models.cars.by_marker[GENERAL] = None
    elif missing == "plate_cars":
        models.cars.by_marker[PLATE] = None
    else:
        models.cars.by_marker[GENERAL] = []

    assert _codes(_run(build_scenario(), models)) == {R.BURST_UNCERTAIN}


def test_plate_sin_auto_cruza_el_ocr_y_coincide() -> None:
    models = happy_models()
    models.cars.by_marker[PLATE] = []

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


def test_plate_sin_auto_y_lecturas_distintas_es_burst_mismatch() -> None:
    models = happy_models()
    models.cars.by_marker[PLATE] = []
    models.plate.by_marker[GENERAL] = ("ZZ999ZZ", 0.9)

    assert R.BURST_MISMATCH in _codes(_run(build_scenario(), models))


@pytest.mark.parametrize(
    "general_read",
    [None, ("ZZ999ZZ", 0.3), ("", 0.9)],
)
def test_plate_sin_auto_y_lecturas_no_comparables_es_uncertain(
    general_read: tuple[str, float] | None,
) -> None:
    models = happy_models()
    models.cars.by_marker[PLATE] = []
    models.plate.by_marker[GENERAL] = general_read

    assert R.BURST_UNCERTAIN in _codes(_run(build_scenario(), models))
    assert R.BURST_MISMATCH not in _codes(_run(build_scenario(), models))


# --- E2: patente ------------------------------------------------------------------------


def test_patente_distinta_es_plate_mismatch() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = ("ZZ999ZZ", 0.95)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED
    assert _codes(result) == {R.PLATE_MISMATCH}


def test_patente_con_un_caracter_de_diferencia_es_plate_fuzzy() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = (REPORTED_PLATE[:-1] + "9", 0.95)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.YELLOW
    assert _codes(result) == {R.PLATE_FUZZY}


@pytest.mark.parametrize("read", [None, (REPORTED_PLATE, 0.2), ("", 0.9)])
def test_patente_ilegible(read: tuple[str, float] | None) -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = read

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.YELLOW
    assert _codes(result) == {R.PLATE_UNREADABLE}


def test_sin_caja_de_patente_en_plate_es_plate_unreadable() -> None:
    models = happy_models()
    models.vehicle.by_marker[PLATE] = []

    assert _codes(_run(build_scenario(), models)) == {R.PLATE_UNREADABLE}


def test_se_lee_la_caja_de_patente_de_mayor_confianza() -> None:
    models = happy_models()
    models.vehicle.by_marker[PLATE] = [Box(0, 0, 4, 4, 0.5, BoxLabel.PLATE), plate_box(0.99)]
    seen: list[tuple[int, ...]] = []
    original = models.plate.read

    def spy(crop):  # type: ignore[no-untyped-def]
        seen.append(crop.shape[:2])
        return original(crop)

    models.plate.read = spy  # type: ignore[method-assign]

    _run(build_scenario(), models)

    assert (20, 50) in seen  # recorte ajustado de la caja de 0.99, sin padding


def test_patente_y_rafaga_fallidas_acumulan_codigos() -> None:
    models = happy_models()
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(80)
    models.plate.by_marker[PLATE] = ("ZZ999ZZ", 0.95)

    assert _codes(_run(build_scenario(), models)) == {R.BURST_MISMATCH, R.PLATE_MISMATCH}


# --- E3: contexto (CRASHED) -------------------------------------------------------------


def test_dano_bajo_el_umbral_es_context_not_detected() -> None:
    models = happy_models()
    models.damage.by_marker[detail_marker(0)] = [Box(0, 0, 9, 9, 0.39)]

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED  # el problema ausente es RED, pero no es fraude
    assert _codes(result) == {R.CONTEXT_NOT_DETECTED}


def test_sin_cajas_de_dano_es_context_not_detected() -> None:
    models = happy_models()
    models.damage.by_marker[detail_marker(0)] = []

    assert _codes(_run(build_scenario(), models)) == {R.CONTEXT_NOT_DETECTED}


def test_detector_de_dano_sin_modelo_es_context_uncertain() -> None:
    models = happy_models()
    models.damage.by_marker[detail_marker(0)] = None

    assert _codes(_run(build_scenario(), models)) == {R.CONTEXT_UNCERTAIN}


def test_basta_una_foto_de_detalle_con_dano() -> None:
    models = happy_models()
    models.damage.by_marker[detail_marker(0)] = []
    models.damage.by_marker[detail_marker(1)] = [Box(0, 0, 9, 9, 0.2)]
    models.damage.by_marker[detail_marker(2)] = [Box(0, 0, 9, 9, 0.7)]

    result = _run(build_scenario(CRASHED, details=3), models)

    assert result.verdict == Verdict.GREEN
    assert result.scores.context_confidence == pytest.approx(0.7)


def test_un_detalle_sin_modelo_y_otro_vacio_es_uncertain() -> None:
    models = happy_models()
    models.damage.by_marker[detail_marker(0)] = None
    models.damage.by_marker[detail_marker(1)] = []

    assert _codes(_run(build_scenario(CRASHED, details=2), models)) == {R.CONTEXT_UNCERTAIN}


# --- E1c: re-identificación de DETAIL y CONTEXT contra GENERAL -----------------------------------------


def _detail_with_car(models: FakeModels, angle: float | None, index: int = 0) -> FakeModels:
    marker = detail_marker(index)
    models.cars.by_marker[marker] = [CarInstance(rect(10, 90, 10, 60))]
    models.embedder.by_marker_area[(marker, 80 * 50)] = None if angle is None else vec(angle)
    return models


def test_un_detalle_con_el_mismo_auto_no_suma_codigos() -> None:
    result = _run(build_scenario(), _detail_with_car(happy_models(), 5))

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()


def test_un_detalle_de_otro_auto_es_burst_mismatch() -> None:
    result = _run(build_scenario(), _detail_with_car(happy_models(), 85))

    assert _codes(result) == {R.BURST_MISMATCH}
    assert result.verdict == Verdict.RED and is_fraud(result.reason_codes)


def test_un_detalle_con_similitud_intermedia_es_burst_uncertain() -> None:
    result = _run(build_scenario(), _detail_with_car(happy_models(), 55))

    assert _codes(result) == {R.BURST_UNCERTAIN} and result.verdict == Verdict.YELLOW


def test_el_mismatch_se_detecta_en_cualquiera_de_los_detalles() -> None:
    models = _detail_with_car(_detail_with_car(happy_models(), 5, 0), 85, 2)

    assert _codes(_run(build_scenario(CRASHED, details=3), models)) == {R.BURST_MISMATCH}


def test_un_detalle_sin_auto_segmentado_no_se_compara() -> None:
    models = happy_models()  # en un primer plano no se segmenta ningún auto

    assert _codes(_run(build_scenario(), models)) == set()


def test_un_detalle_cuyo_segmentador_falla_es_burst_uncertain() -> None:
    models = happy_models()
    models.cars.by_marker[detail_marker(0)] = None

    assert _codes(_run(build_scenario(), models)) == {R.BURST_UNCERTAIN}


def test_un_detalle_sin_embedding_es_burst_uncertain() -> None:
    assert _codes(_run(build_scenario(), _detail_with_car(happy_models(), None))) == {R.BURST_UNCERTAIN}


def test_sin_embedding_de_general_no_se_comparan_los_detalles() -> None:
    models = _detail_with_car(happy_models(), 85)
    models.cars.by_marker[
        GENERAL
    ] = []  # sin auto en GENERAL: la ráfaga ya es dudosa y no hay con qué comparar

    assert _codes(_run(build_scenario(), models)) == {R.BURST_UNCERTAIN}


def test_el_contexto_de_otro_auto_es_burst_mismatch() -> None:
    models = happy_models()
    models.embedder.by_marker_area[(CONTEXT, 1600)] = vec(85)
    models.embedder.by_marker_area[(CONTEXT, 400)] = vec(80)

    assert R.BURST_MISMATCH in _codes(_run(build_scenario(GARAGE), models))


def test_el_embedding_del_contexto_se_calcula_una_sola_vez_por_auto() -> None:
    models = happy_models()
    calls: list[int] = []
    original = models.embedder.embed

    def counting(img: object, mask: object) -> object:
        calls.append(marker_of(img))  # type: ignore[arg-type]
        return original(img, mask)  # type: ignore[arg-type]

    models.embedder.embed = counting  # type: ignore[method-assign]

    _run(build_scenario(GARAGE), models)

    assert calls.count(CONTEXT) == 2  # los dos autos del contexto, una vez cada uno


# --- E3: contexto (PARKED_ON_GARAGE) ----------------------------------------------------


def _garage_models(target: CarInstance, area: int) -> FakeModels:
    models = happy_models()
    models.cars.by_marker[CONTEXT] = [target, CarInstance(rect(20, 40, 0, 20))]
    models.embedder.by_marker_area[(CONTEXT, area)] = vec(5)
    return models


def test_auto_sobre_el_rebaje_es_green() -> None:
    assert _run(build_scenario(GARAGE)).verdict == Verdict.GREEN


def test_contacto_exacto_es_context_uncertain() -> None:
    # el auto empieza justo donde termina el rebaje (80): solape 0 → marginal
    models = _garage_models(CarInstance(rect(40, 80, 80, 98)), 40 * 18)

    result = _run(build_scenario(GARAGE), models)

    assert _codes(result) == {R.CONTEXT_UNCERTAIN}


@pytest.mark.parametrize(
    ("t_low", "t_high", "expected"),
    [
        (-0.2, -0.1, set()),  # el solape (~ -0,05) llega a T_HIGH: bloquea con claridad
        (-0.08, 0.08, {R.CONTEXT_UNCERTAIN}),  # entre los dos umbrales: dudoso
        (0.01, 0.08, {R.CONTEXT_NOT_DETECTED}),  # bajo T_LOW: el auto no bloquea
    ],
)
def test_los_dos_umbrales_de_la_rampa_deciden_el_codigo(
    t_low: float, t_high: float, expected: set[R]
) -> None:
    models = _garage_models(CarInstance(rect(40, 80, 80, 98)), 40 * 18)  # solape 0
    scenario = build_scenario(GARAGE)
    use_case = ValidateEvidence(
        store=scenario.store,
        writer=FakeWriter(),
        decryptor=scenario.decryptor,
        decoder=scenario.decoder,
        renderer=JpegRenderer(),
        models=models.as_models(),
        thresholds=replace(THRESHOLDS, ramp_t_low=t_low, ramp_t_high=t_high),
        redaction=REDACTION,
    )

    assert _codes(use_case.execute(scenario.payload)) == expected


def test_auto_lejos_del_rebaje_es_context_not_detected() -> None:
    models = _garage_models(CarInstance(rect(40, 80, 90, 99)), 40 * 9)

    assert _codes(_run(build_scenario(GARAGE), models)) == {R.CONTEXT_NOT_DETECTED}


def test_sin_rebaje_es_context_not_detected() -> None:
    models = happy_models()
    models.curb.by_marker[CONTEXT] = (rect(80, 86, 0, 100), rect(0, 0, 0, 0))

    assert _codes(_run(build_scenario(GARAGE), models)) == {R.CONTEXT_NOT_DETECTED}


@pytest.mark.parametrize("missing", ["curb", "cars", "no_cars"])
def test_contexto_sin_modelo_o_sin_autos_es_uncertain(missing: str) -> None:
    models = happy_models()
    if missing == "curb":
        models.curb.by_marker[CONTEXT] = None
    elif missing == "cars":
        models.cars.by_marker[CONTEXT] = None
    else:
        models.cars.by_marker[CONTEXT] = []

    codes = _codes(_run(build_scenario(GARAGE), models))

    assert R.CONTEXT_UNCERTAIN in codes and not codes & {
        R.NO_VEHICLE,
        R.BURST_MISMATCH,
        R.CONTEXT_NOT_DETECTED,
    }


def test_el_objetivo_es_el_auto_que_mas_se_parece_a_plate_o_general() -> None:
    models = happy_models()
    # El auto chico (que no bloquea) se parece más a la referencia; el grande queda lejos
    models.embedder.by_marker_area[(CONTEXT, 1600)] = vec(70)
    models.embedder.by_marker_area[(CONTEXT, 400)] = vec(2)
    models.curb.by_marker[CONTEXT] = (rect(80, 86, 0, 20) | rect(80, 86, 81, 100), rect(80, 86, 20, 81))
    models.cars.by_marker[CONTEXT] = [CarInstance(rect(40, 80, 30, 70)), CarInstance(rect(60, 80, 0, 20))]
    models.embedder.by_marker_area[(CONTEXT, 20 * 20)] = vec(2)

    assert _codes(_run(build_scenario(GARAGE), models)) == {R.CONTEXT_NOT_DETECTED}


def test_mejor_coseno_bajo_reid_t_low_es_burst_mismatch_y_context_uncertain() -> None:
    models = happy_models()
    models.embedder.by_marker_area[(CONTEXT, 1600)] = vec(85)
    models.embedder.by_marker_area[(CONTEXT, 400)] = vec(90)

    result = _run(build_scenario(GARAGE), models)

    assert _codes(result) == {
        R.CONTEXT_UNCERTAIN,
        R.BURST_MISMATCH,
    }  # ningún auto del contexto es el reportado
    assert result.verdict == Verdict.RED


def test_sin_embedder_y_mas_de_un_auto_es_context_uncertain() -> None:
    models = happy_models()
    models.embedder.by_marker_area = {}

    assert R.CONTEXT_UNCERTAIN in _codes(_run(build_scenario(GARAGE), models))


def test_sin_embedder_y_un_solo_auto_se_evalua_ese_auto() -> None:
    models = happy_models()
    models.embedder.by_marker_area = {}
    models.cars.by_marker[CONTEXT] = [CarInstance(rect(40, 80, 30, 70))]

    result = _run(build_scenario(GARAGE), models)

    assert R.CONTEXT_UNCERTAIN not in _codes(result)
    assert R.CONTEXT_NOT_DETECTED not in _codes(result)


def test_la_escala_usa_las_ruedas_dentro_del_objetivo() -> None:
    # Solape de 1 px: con la escala de respaldo (margen 1.4 px) es marginal; con una rueda
    # de 4 px de alto dentro del objetivo (margen 0.32 px) es bloqueo.
    def models_with(wheel: Box | None) -> FakeModels:
        models = happy_models()
        models.vehicle.by_marker[CONTEXT] = [car_box(), *([wheel] if wheel else [])]
        models.cars.by_marker[CONTEXT] = [CarInstance(rect(40, 80, 79, 95))]
        models.embedder.by_marker_area[(CONTEXT, 40 * 16)] = vec(5)
        return models

    wheel = Box(85, 60, 89, 64, 0.9, BoxLabel.WHEEL)

    assert _codes(_run(build_scenario(GARAGE), models_with(None))) == {R.CONTEXT_UNCERTAIN}
    assert _run(build_scenario(GARAGE), models_with(wheel)).verdict == Verdict.GREEN


# --- evidencia ilegible -----------------------------------------------------------------


def test_sha256_distinto_es_evidence_unreadable() -> None:
    scenario = build_scenario()
    scenario.corrupt_sha(EvidenceRole.PLATE)
    models = happy_models()

    result = _run(scenario, models)

    assert result.verdict == Verdict.RED
    assert result.reason_codes == (R.EVIDENCE_UNREADABLE,)
    assert models.vehicle.calls == 0
    assert result.redacted == ()


def test_foto_que_no_descifra_es_evidence_unreadable() -> None:
    scenario = build_scenario()
    scenario.replace_object(EvidenceRole.GENERAL, b"ENC|otra-alerta|IMG10")

    assert _run(scenario).reason_codes == (R.EVIDENCE_UNREADABLE,)


def test_foto_cifrada_con_otro_info_es_evidence_unreadable() -> None:
    scenario = build_scenario()
    scenario.replace_object(EvidenceRole.DETAIL, FakeDecryptor.seal(b"IMG30", "otro-alert-id"))

    assert _run(scenario).reason_codes == (R.EVIDENCE_UNREADABLE,)


def test_foto_que_no_decodifica_es_evidence_unreadable() -> None:
    scenario = build_scenario()
    scenario.replace_object(EvidenceRole.PLATE, FakeDecryptor.seal(b"no-es-jpeg", scenario.alert_id))

    result = _run(scenario)

    assert result.verdict == Verdict.RED
    assert result.reason_codes == (R.EVIDENCE_UNREADABLE,)


def test_evidence_unreadable_no_es_fraude() -> None:
    scenario = build_scenario()
    scenario.corrupt_sha(EvidenceRole.GENERAL)

    assert not is_fraud(_run(scenario).reason_codes)


# --- errores del sistema y transitorios -------------------------------------------------


def test_pat_enc_ia_que_no_descifra_es_error_del_sistema() -> None:
    scenario = build_scenario()
    scenario.with_pat_enc_ia(b"ENC|otro-alert-id|" + REPORTED_PLATE.encode())

    with pytest.raises(SystemDecryptionError):
        _run(scenario)


def test_pat_enc_ia_con_contenido_no_utf8_es_error_del_sistema() -> None:
    scenario = build_scenario()
    scenario.with_pat_enc_ia(FakeDecryptor.seal(b"\xff\xfe", scenario.alert_id))

    with pytest.raises(SystemDecryptionError):
        _run(scenario)


def test_pat_enc_ia_sin_patente_es_error_del_sistema() -> None:
    scenario = build_scenario()
    scenario.with_pat_enc_ia(FakeDecryptor.seal(b" -- ", scenario.alert_id))

    with pytest.raises(SystemDecryptionError):
        _run(scenario)


def test_error_transitorio_de_s3_se_propaga() -> None:
    scenario = build_scenario()
    scenario.store.error = TransientError("breaker abierto")

    with pytest.raises(TransientError):
        _run(scenario)


def test_objeto_inexistente_se_propaga_como_error_del_sistema() -> None:
    scenario = build_scenario()
    scenario.store.objects.clear()

    with pytest.raises(ObjectNotFoundError):
        _run(scenario)


def test_la_patente_se_canonicaliza_antes_de_comparar() -> None:
    scenario = build_scenario()
    scenario.with_pat_enc_ia(FakeDecryptor.seal(b"tst-123", scenario.alert_id))

    assert _run(scenario).verdict == Verdict.GREEN


# --- privacidad en logs -----------------------------------------------------------------


def test_los_logs_no_contienen_la_patente_ni_el_texto_del_ocr(caplog: pytest.LogCaptureFixture) -> None:
    models = happy_models()
    ocr_text = "OCRSECRET9"
    models.plate.by_marker[PLATE] = (ocr_text, 0.95)
    models.plate.by_marker[GENERAL] = (ocr_text, 0.9)
    scenario = build_scenario(GARAGE)

    with caplog.at_level(logging.DEBUG):
        _run(scenario, models)
        scenario.corrupt_sha(EvidenceRole.GENERAL)
        _run(scenario, models)

    formatter = JsonFormatter()
    output = "\n".join(formatter.format(record) for record in caplog.records)
    assert caplog.records
    assert REPORTED_PLATE not in output
    assert ocr_text not in output
    assert scenario.payload.pat_enc_ia not in output


def test_el_log_final_trae_los_campos_permitidos(caplog: pytest.LogCaptureFixture) -> None:
    scenario = build_scenario()

    with caplog.at_level(logging.INFO):
        _run(scenario)

    record = next(r for r in caplog.records if r.getMessage() == "validación completada")
    assert record.__dict__["alert_id"] == scenario.alert_id
    assert record.__dict__["verdict"] == "GREEN"
    assert set(record.__dict__["durations_ms"]) >= {"load", "burst", "plate", "context"}
    assert record.__dict__["model_versions"]["detector"] == "fake-detector"


def test_objeto_que_supera_el_tope_es_evidence_unreadable_sin_correr_modelos() -> None:
    scenario = build_scenario()
    key = next(i.key for i in scenario.payload.evidence if i.role == EvidenceRole.PLATE)
    scenario.store.errors_by_key[key] = EvidenceTooLargeError("grande")
    models = happy_models()

    result = _run(scenario, models)

    assert result.verdict == Verdict.RED
    assert result.reason_codes == (R.EVIDENCE_UNREADABLE,)
    assert models.vehicle.calls == 0
    assert result.redacted == ()
    assert not is_fraud(result.reason_codes)
