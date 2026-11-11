import logging
from types import SimpleNamespace

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


def _only(
    models: FakeModels, marker: int, boxes: list[Box] | None, cars: list[CarInstance] | None
) -> FakeModels:
    models.vehicle.by_marker[marker] = boxes
    models.cars.by_marker[marker] = cars
    models.damage.by_marker[marker] = []  # el daño solo cuenta en DETAIL y acá se prueba la falta de partes
    return models


def test_una_foto_de_contexto_sin_auto_es_no_vehicle() -> None:
    models = _only(
        happy_models(), CONTEXT, [Box(0, 0, 10, 10, 0.9, BoxLabel.WHEEL)], []
    )  # una rueda no alcanza

    result = _run(build_scenario(GARAGE), models)

    assert result.reason_codes == (R.NO_VEHICLE,) and result.verdict == Verdict.RED


def test_una_foto_de_patente_sin_auto_ni_partes_es_no_vehicle() -> None:
    result = _run(build_scenario(), _only(happy_models(), PLATE, [], []))

    assert result.reason_codes == (R.NO_VEHICLE,) and is_fraud(result.reason_codes)


def test_un_detalle_sin_auto_ni_partes_es_no_vehicle_aunque_las_demas_esten_bien() -> None:
    models = _only(happy_models(), detail_marker(0), [], [])

    result = _run(build_scenario(), models)

    assert result.reason_codes == (R.NO_VEHICLE,) and result.redacted == ()


def test_el_mismo_defecto_en_cualquiera_de_los_detalles_alcanza() -> None:
    models = _only(happy_models(), detail_marker(2), [], [])

    assert _codes(_run(build_scenario(CRASHED, details=3), models)) == {R.NO_VEHICLE}


@pytest.mark.parametrize(
    ("boxes", "cars"),
    [
        ([Box(0, 0, 60, 60, 0.9, BoxLabel.WHEEL)], []),  # una rueda
        ([plate_box(0.95)], []),  # una patente
        ([], [CarInstance(rect(0, 60, 0, 60))]),  # un auto segmentado
        ([Box(0, 0, 90, 90, 0.9, BoxLabel.CAR)], None),  # un auto detectado, el segmentador sin modelo
    ],
)
@pytest.mark.parametrize("role_marker", [PLATE, detail_marker(0)])
def test_en_plate_y_detail_basta_una_parte_del_auto(
    boxes: list[Box], cars: list[CarInstance] | None, role_marker: int
) -> None:
    models = _only(happy_models(), role_marker, boxes, cars)

    assert R.NO_VEHICLE not in _codes(_run(build_scenario(), models))


def test_en_general_una_rueda_o_una_patente_no_alcanzan() -> None:
    models = _only(happy_models(), GENERAL, [Box(0, 0, 60, 60, 0.9, BoxLabel.WHEEL), plate_box()], [])

    assert _codes(_run(build_scenario(), models)) == {R.NO_VEHICLE}


def test_una_caja_chica_no_cuenta_como_presencia() -> None:
    small = Box(0, 0, 20, 20, 0.9, BoxLabel.CAR)  # 4 % de la foto de 100 x 100
    models = _only(happy_models(), GENERAL, [small], [CarInstance(rect(0, 20, 0, 20))])

    assert _codes(_run(build_scenario(), models)) == {R.NO_VEHICLE}


def test_una_patente_poco_confiable_no_cuenta_como_parte_del_auto() -> None:
    models = _only(happy_models(), PLATE, [plate_box(0.3)], [])

    assert _codes(_run(build_scenario(), models)) == {R.NO_VEHICLE}


def test_el_corte_por_falta_de_auto_no_corre_el_resto_de_la_cascada() -> None:
    models = _only(happy_models(), detail_marker(0), [], [])

    _run(build_scenario(), models)

    assert models.plate.calls == 0 and models.damage.by_marker  # sin lectura de patente ni redacción


def test_mismo_auto_en_ambas_fotos_no_agrega_codigos() -> None:
    assert R.BURST_MISMATCH not in _codes(_run(build_scenario()))


def test_autos_distintos_son_burst_mismatch() -> None:
    models = happy_models()
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(80)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED
    assert R.BURST_MISMATCH in _codes(result)
    assert result.scores.burst_similarity == pytest.approx(0.17, abs=0.01)


def test_patente_distinta_es_plate_mismatch() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = ("ZZ999ZZ", 0.95)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED
    assert _codes(result) == {R.PLATE_MISMATCH}


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


def test_el_mismatch_se_detecta_en_cualquiera_de_los_detalles() -> None:
    models = _detail_with_car(_detail_with_car(happy_models(), 5, 0), 85, 2)

    assert _codes(_run(build_scenario(CRASHED, details=3), models)) == {R.BURST_MISMATCH}


def test_un_detalle_sin_auto_segmentado_no_se_compara() -> None:
    models = happy_models()  # en un primer plano no se segmenta ningún auto

    assert _codes(_run(build_scenario(), models)) == set()


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
    assert set(record.__dict__["durations_ms"]) >= {"load", "vehicle", "burst", "plate"}
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


def test_las_versiones_de_modelo_viajan_en_el_resultado() -> None:
    result = _run(build_scenario())

    assert result.model_versions["detector"] == "fake-detector"
    assert result.model_versions["ocr"] == "fake-ocr"


def test_reprocesar_es_idempotente() -> None:
    scenario = build_scenario(GARAGE)

    assert _run(scenario) == _run(scenario)


def test_un_none_nunca_produce_fraude() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = None

    result = _run(build_scenario(), models)

    assert not is_fraud(result.reason_codes)
    assert result.verdict == Verdict.RED and R.REDACTION_UNAVAILABLE in result.reason_codes


# --- verdicto GREEN/RED: reglas R1 a R5 ------------------------------------------------------------


@pytest.mark.parametrize("alert_type", [CRASHED, GARAGE])
def test_caso_feliz_es_green_sin_codigos(alert_type: AlertType) -> None:
    result = _run(build_scenario(alert_type))

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()
    assert len(result.redacted) == 3
    assert result.scores.burst_similarity is not None
    assert result.scores.ocr_confidence == pytest.approx(0.95)
    assert result.scores.context_confidence is None  # ya no hay chequeo de contexto ni de daño


@pytest.mark.parametrize("alert_type", [CRASHED, GARAGE])
def test_sin_modelos_el_unico_rojo_es_redaccion_no_disponible(alert_type: AlertType) -> None:
    result = _run(build_scenario(alert_type), _null_models())

    assert result.verdict == Verdict.RED
    assert _codes(result) == {R.REDACTION_UNAVAILABLE}  # R4; nada de lo dudoso suma códigos
    assert result.redacted == () and not is_fraud(result.reason_codes)
    assert set(result.model_versions) == {"detector", "segmenter", "embedder", "ocr", "damage", "face"}
    assert set(result.model_versions.values()) == {"null"}


def test_los_codigos_dudosos_ya_no_existen_y_el_veredicto_nunca_es_amarillo() -> None:
    assert {v.value for v in Verdict} == {"GREEN", "RED"}
    models = happy_models()
    models.plate.by_marker[PLATE] = (REPORTED_PLATE[:-1] + "9", 0.95)  # casi igual

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


# --- R1: el daño solo cuenta como presencia en DETAIL --------------------------------------------------


def _detail_without_part(models: FakeModels, damage: float | None) -> FakeModels:
    marker = detail_marker(0)
    models.vehicle.by_marker[marker] = []
    models.cars.by_marker[marker] = []
    models.damage.by_marker[marker] = [] if damage is None else [Box(0, 0, 9, 9, damage)]
    return models


def test_un_detalle_sin_partes_pero_con_dano_seguro_cuenta_como_presente() -> None:
    models = _detail_without_part(happy_models(), 0.80)  # THRESHOLDS.damage_presence_t = 0,70

    assert _run(build_scenario(), models).reason_codes == ()


def test_un_dano_poco_confiable_no_alcanza_como_presencia() -> None:
    models = _detail_without_part(happy_models(), 0.60)

    assert _codes(_run(build_scenario(), models)) == {R.NO_VEHICLE}


def test_el_dano_no_cuenta_como_presencia_en_plate_ni_en_general() -> None:
    for marker in (PLATE, GENERAL):
        models = _only(happy_models(), marker, [], [])
        models.damage.by_marker[marker] = [Box(0, 0, 90, 90, 0.99)]

        assert _codes(_run(build_scenario(), models)) == {R.NO_VEHICLE}


def test_el_dano_no_hace_rojo_a_un_detalle_valido_y_su_ausencia_tampoco() -> None:
    models = happy_models()
    models.damage.by_marker[detail_marker(0)] = []  # el problema "ausente" ya no es un veredicto

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


def test_el_detector_de_danos_sin_modelo_no_penaliza() -> None:
    models = happy_models()
    models.damage.by_marker[detail_marker(0)] = None

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


# --- R1: modelos que no responden ---------------------------------------------------------------------


def test_si_ningun_modelo_responde_no_hay_no_vehicle() -> None:
    models = _only(happy_models(), detail_marker(0), None, None)
    models.damage.by_marker[detail_marker(0)] = None

    assert R.NO_VEHICLE not in _codes(_run(build_scenario(), models))


def test_si_un_modelo_falla_y_el_otro_no_ve_nada_tampoco_se_decide() -> None:
    models = _only(happy_models(), detail_marker(0), None, [])

    assert R.NO_VEHICLE not in _codes(_run(build_scenario(), models))


def test_detector_sin_modelo_en_general_no_penaliza() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = None
    models.cars.by_marker[GENERAL] = None

    assert R.NO_VEHICLE not in _codes(_run(build_scenario(), models))


def test_ningun_auto_es_no_vehicle_y_corta_la_cascada() -> None:
    models = happy_models()
    models.vehicle.by_marker[GENERAL] = [plate_box()]
    models.cars.by_marker[GENERAL] = []

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED and result.reason_codes == (R.NO_VEHICLE,)
    assert models.plate.calls == 0 and is_fraud(result.reason_codes)


# --- R2: re-ID, solo con confirmación ------------------------------------------------------------------


@pytest.mark.parametrize("angle", [55, 48])  # coseno 0,57 y 0,67: entre 0,50 y 0,70 (antes: dudoso)
def test_una_similitud_intermedia_es_green(angle: float) -> None:
    models = happy_models()
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(angle)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()


def test_sin_embedding_o_sin_auto_segmentado_no_se_compara_y_es_green() -> None:
    for marker in (PLATE, GENERAL):
        models = happy_models()
        models.embedder.by_marker_area[(marker, 6400)] = None

        assert _run(build_scenario(), models).verdict == Verdict.GREEN
    models = happy_models()
    models.cars.by_marker[PLATE] = []  # PLATE sin auto segmentado: no hay con qué comparar

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


def test_un_detalle_con_similitud_intermedia_es_green() -> None:
    assert _run(build_scenario(), _detail_with_car(happy_models(), 55)).verdict == Verdict.GREEN


def test_un_detalle_sin_embedding_o_con_segmentador_caido_es_green() -> None:
    assert _run(build_scenario(), _detail_with_car(happy_models(), None)).verdict == Verdict.GREEN
    models = happy_models()
    models.cars.by_marker[detail_marker(0)] = None

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


def test_el_umbral_de_reid_es_uno_solo() -> None:
    from dataclasses import replace

    models = happy_models()
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(55)  # coseno 0,57
    scenario = build_scenario()

    def verdict_with(t: float) -> Verdict:
        use_case = ValidateEvidence(
            store=scenario.store,
            writer=FakeWriter(),
            decryptor=scenario.decryptor,
            decoder=scenario.decoder,
            renderer=JpegRenderer(),
            models=models.as_models(),
            thresholds=replace(THRESHOLDS, reid_t=t),
            redaction=REDACTION,
        )
        return use_case.execute(scenario.payload).verdict

    assert verdict_with(0.50) == Verdict.GREEN and verdict_with(0.60) == Verdict.RED


# --- R3: patente, solo una lectura segura que no coincide ---------------------------------------------


def test_una_patente_casi_igual_es_green() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = (REPORTED_PLATE[:-1] + "9", 0.95)

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()


@pytest.mark.parametrize("read", [None, ("ZZ999ZZ", 0.30), (REPORTED_PLATE, 0.50)])
def test_una_patente_ilegible_o_poco_segura_es_green(read: tuple[str, float] | None) -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = read

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


def test_sin_caja_de_patente_en_plate_es_green() -> None:
    models = happy_models()
    models.vehicle.by_marker[PLATE] = [car_box()]  # el auto sin ninguna patente que leer

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.GREEN
    assert result.scores.ocr_confidence is None


def test_plate_sin_auto_segmentado_no_cruza_lecturas() -> None:
    models = happy_models()
    models.cars.by_marker[PLATE] = []
    models.plate.by_marker[GENERAL] = ("ZZ999ZZ", 0.95)  # antes, otra lectura en GENERAL daba BURST_MISMATCH

    assert _run(build_scenario(), models).verdict == Verdict.GREEN


def test_patente_mismatch_y_otro_auto_acumulan_codigos_de_fraude() -> None:
    models = happy_models()
    models.plate.by_marker[PLATE] = ("ZZ999ZZ", 0.95)
    models.embedder.by_marker_area[(PLATE, 6400)] = vec(85)

    result = _run(build_scenario(), models)

    assert _codes(result) == {R.PLATE_MISMATCH, R.BURST_MISMATCH} and is_fraud(result.reason_codes)


# --- el contexto ya no se evalúa --------------------------------------------------------------------


def test_el_modelo_de_rebajes_no_se_consulta_ni_hace_falta() -> None:
    models = happy_models()
    models.curb.by_marker[CONTEXT] = None  # sin rebaje, sin modelo, o con un auto lejos: da igual

    result = _run(build_scenario(GARAGE), models)

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()


def test_un_contexto_con_el_auto_lejos_del_rebaje_es_green() -> None:
    models = happy_models()
    models.cars.by_marker[CONTEXT] = [CarInstance(rect(40, 80, 90, 99)), CarInstance(rect(20, 40, 0, 20))]
    models.embedder.by_marker_area[(CONTEXT, 40 * 9)] = vec(5)

    assert _run(build_scenario(GARAGE), models).verdict == Verdict.GREEN


def test_un_contexto_sin_embedder_es_green() -> None:
    models = happy_models()
    models.embedder.by_marker_area[(CONTEXT, 1600)] = None
    models.embedder.by_marker_area[(CONTEXT, 400)] = None

    assert _run(build_scenario(GARAGE), models).verdict == Verdict.GREEN


# --- latencia: un solo pase por foto, reducción y mosaico solo al redactar -----------------------------


def test_reduce_for_models_reduce_el_lado_mayor_y_nunca_agranda() -> None:
    import numpy as np

    from vvi_ia.service.validate_evidence import reduce_for_models

    big = np.zeros((300, 400, 3), np.uint8)

    reduced = reduce_for_models(big, 200)

    assert reduced is not None and reduced.shape == (150, 200, 3)
    assert reduce_for_models(big, 400) is None and reduce_for_models(big, 1000) is None


def test_los_modelos_ven_la_foto_reducida_y_las_cajas_vuelven_al_original() -> None:
    import numpy as np

    from vvi_ia.domain.model import EvidenceImage
    from vvi_ia.service.validate_evidence import _Run, reduce_for_models

    seen: list[tuple[int, ...]] = []
    models = happy_models()

    def detect(img: np.ndarray) -> list[Box]:
        seen.append(img.shape)
        return [Box(10, 20, 30, 40, 0.9, BoxLabel.PLATE)]

    models.vehicle.detect = detect  # type: ignore[method-assign]
    pixels = np.zeros((200, 400, 3), np.uint8)
    image = EvidenceImage(EvidenceRole.GENERAL, 0, pixels, reduce_for_models(pixels, 100))

    boxes = _Run(models=models.as_models()).detect(image)

    assert seen == [(50, 100, 3)] and image.ratio == 4.0
    assert boxes == [Box(40, 80, 120, 160, 0.9, BoxLabel.PLATE)]


def test_el_detector_corre_un_solo_pase_por_foto_en_toda_la_cascada() -> None:
    models = happy_models()

    _run(build_scenario(), models)

    assert (
        models.vehicle.calls == 3
    )  # general, plate y detail: presencia, lectura y redacción comparten el pase


def test_el_mosaico_solo_corre_al_redactar_y_solo_si_el_veredicto_no_es_rojo() -> None:
    green = happy_models()
    _run(build_scenario(), green)
    red = happy_models()
    red.plate.by_marker[PLATE] = ("ZZ999ZZ", 0.95)  # PLATE_MISMATCH: no se redacta

    _run(build_scenario(), red)

    assert green.vehicle.plate_calls == 3  # una vez por foto
    assert red.vehicle.plate_calls == 0


# --- sin segmentador (MODEL_SEGMENTER=null): los autos salen de las cajas del detector ----------------


def _without_segmenter(models: FakeModels) -> FakeModels:
    models.cars = NullCarSegmenter()  # type: ignore[assignment]
    return models


def _boxed_models() -> FakeModels:
    """Sin segmentador, con un auto detectado en GENERAL y en PLATE y un embedding por caja (90 x 90 px)."""
    models = _without_segmenter(happy_models())
    models.vehicle.by_marker[PLATE] = [car_box(), plate_box(0.95)]
    models.embedder.by_marker_area = {(GENERAL, 8100): vec(0), (PLATE, 8100): vec(10)}
    return models


def test_sin_segmentador_la_presencia_la_decide_el_detector() -> None:
    models = _without_segmenter(happy_models())
    models.vehicle.by_marker[GENERAL] = [plate_box()]  # sin caja de auto

    result = _run(build_scenario(), models)

    assert result.verdict == Verdict.RED and result.reason_codes == (R.NO_VEHICLE,)


def test_sin_segmentador_un_auto_detectado_alcanza_y_la_cascada_sigue() -> None:
    result = _run(build_scenario(), _boxed_models())

    assert result.verdict == Verdict.GREEN and result.reason_codes == ()
    assert result.model_versions["segmenter"] == "null"


def test_sin_segmentador_la_reid_compara_el_recorte_de_la_caja_del_detector() -> None:
    same = _run(build_scenario(), _boxed_models())
    other = _boxed_models()
    other.embedder.by_marker_area[(PLATE, 8100)] = vec(80)

    different = _run(build_scenario(), other)

    assert same.reason_codes == () and same.scores.burst_similarity == pytest.approx(0.98, abs=0.01)
    assert R.BURST_MISMATCH in _codes(different) and different.verdict == Verdict.RED


def test_sin_segmentador_cada_detalle_se_compara_con_el_auto_de_su_caja() -> None:
    models = _boxed_models()
    marker = detail_marker(0)
    models.vehicle.by_marker[marker] = [Box(0, 0, 80, 80, 0.9, BoxLabel.CAR)]  # 6400 px
    models.embedder.by_marker_area[(marker, 6400)] = vec(85)  # otro auto

    assert R.BURST_MISMATCH in _codes(_run(build_scenario(), models))
    models.embedder.by_marker_area[(marker, 6400)] = vec(5)  # el mismo

    assert _run(build_scenario(), models).reason_codes == ()


def test_sin_segmentador_las_ruedas_y_patentes_no_son_autos_para_la_reid() -> None:
    models = _boxed_models()
    marker = detail_marker(0)  # un primer plano: solo una rueda, ningún auto que comparar
    models.embedder.by_marker = {marker: vec(85)}

    result = _run(build_scenario(), models)

    assert result.reason_codes == ()


def test_sin_segmentador_y_con_el_detector_caido_no_se_penaliza() -> None:
    models = _without_segmenter(happy_models())
    models.vehicle.by_marker[GENERAL] = None
    models.damage.by_marker = {}

    assert R.NO_VEHICLE not in _codes(_run(build_scenario(), models))


def test_con_segmentador_sus_instancias_mandan_aunque_no_haya_ninguna() -> None:
    """Un segmentador que responde `[]` no se completa con las cajas del detector (modo experimental)."""
    models = happy_models()
    models.vehicle.by_marker[PLATE] = [car_box(), plate_box(0.95)]
    models.cars.by_marker[PLATE] = []
    models.embedder.by_marker_area[(PLATE, 8100)] = vec(85)

    result = _run(build_scenario(), models)

    assert R.BURST_MISMATCH not in _codes(result)  # PLATE sin auto segmentado: no hay con qué comparar


def test_las_mascaras_de_las_cajas_estan_en_las_coordenadas_de_la_foto_reducida() -> None:
    import numpy as np

    from vvi_ia.domain.model import EvidenceImage
    from vvi_ia.service.validate_evidence import _Run

    original = np.zeros((200, 200, 3), np.uint8)
    reduced = np.zeros((100, 100, 3), np.uint8)
    image = EvidenceImage(role=EvidenceRole.GENERAL, index=0, pixels=original, view=reduced)
    models = _without_segmenter(happy_models())
    # el detector ve la foto reducida: un auto, una rueda (se ignora) y un auto fuera de la foto (se ignora)
    boxes = [
        Box(5, 10, 75, 60, 0.9, BoxLabel.CAR),
        Box(0, 0, 20, 20, 0.9, BoxLabel.WHEEL),
        Box(200, 200, 250, 250, 0.9, BoxLabel.CAR),
    ]
    models.vehicle = SimpleNamespace(version="fake", detect=lambda _: boxes)  # type: ignore[assignment]
    run = _Run(models=models.as_models())

    cars = run.car_instances(image)

    assert cars is not None and len(cars) == 1
    mask = cars[0].mask
    assert mask.shape == (100, 100) and mask.dtype == np.bool_
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) == (5, 75, 10, 60)
    assert cars[0].conf == pytest.approx(0.9)
    assert run.car_instances(image) is cars  # con caché
