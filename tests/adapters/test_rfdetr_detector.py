"""Detector RF-DETR: preproceso, decodificación por clase, núcleo compartido y registro con ONNX diminuto."""

import hashlib
import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from vvi_ia.adapters.models.manifest import Commercial, load_manifest
from vvi_ia.adapters.models.registry import ADAPTERS, ModelSetupError, build_models
from vvi_ia.adapters.models.rfdetr_detector import (
    FACE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABELS,
    MODEL_NAME,
    OUTPUT_NAMES,
    RfDetrDetections,
    RfDetrFaceDetector,
    RfDetrVehicleDetector,
    decode_boxes,
    preprocess,
)
from vvi_ia.config import load_settings
from vvi_ia.domain.model import BoxLabel
from vvi_ia.main import adapter_params

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = 5  # cuatro clases más la columna de fondo
QUERIES = 6
ON, OFF = 6.0, -6.0  # logits de confianza alta y baja (sigmoid ≈ 0.998 y 0.002)
THRESHOLDS = {BoxLabel.CAR: 0.4, BoxLabel.WHEEL: 0.4, BoxLabel.PLATE: 0.1, FACE: 0.1}
CAR, WHEEL, PLATE, FACE_COLUMN = 0, 1, 2, 3


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def outputs(*rows: tuple[int, float, tuple[float, float, float, float]]) -> tuple[NDArray[Any], NDArray[Any]]:
    """`dets` y `labels` de una imagen; cada fila es `(columna, confianza, (cx, cy, w, h) normalizado)`."""
    dets = np.zeros((1, QUERIES, 4), np.float32)
    dets[..., 2:] = 0.1
    labels = np.full((1, QUERIES, CLASSES), OFF, np.float32)
    for query, (column, confidence, box) in enumerate(rows):
        dets[0, query] = box
        labels[0, query, column] = logit(confidence)
    return dets, labels


class FakeSession:
    """Sesión falsa: devuelve las salidas configuradas y guarda lo que recibió."""

    def __init__(self, result: Sequence[NDArray[Any]] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[Sequence[str] | None, Mapping[str, NDArray[Any]]]] = []

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[Any]]:
        self.calls.append((output_names, input_feed))
        if isinstance(self.result, Exception):
            raise self.result
        return list(self.result)


def core(session: FakeSession, thresholds: Mapping[str, float] = THRESHOLDS) -> RfDetrDetections:
    return RfDetrDetections(session, "input", (16, 24), thresholds)


def photo(height: int = 40, width: int = 80) -> NDArray[np.uint8]:
    return np.random.default_rng(3).integers(0, 256, (height, width, 3), dtype=np.uint8)


FULL = (
    (CAR, 0.95, (0.5, 0.5, 0.8, 0.6)),
    (WHEEL, 0.70, (0.25, 0.75, 0.1, 0.2)),
    (PLATE, 0.15, (0.5, 0.8, 0.2, 0.1)),  # por debajo del umbral de vehículo, sobre el de redacción
    (FACE_COLUMN, 0.12, (0.1, 0.1, 0.05, 0.1)),
)


# --- preproceso --------------------------------------------------------------------------


def test_preproceso_devuelve_nchw_float32_a_la_resolucion_del_modelo() -> None:
    tensor = preprocess(photo(300, 500), 16, 24)

    assert tensor.shape == (1, 3, 16, 24)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]


def test_preproceso_normaliza_con_imagenet() -> None:
    color = (200, 40, 90)
    tensor = preprocess(np.full((20, 30, 3), color, np.uint8), 8, 8)

    expected = (np.array(color, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(tensor[0, :, 4, 4], expected, atol=1e-5)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (10, 10, 1), (0, 10, 3)])
def test_preproceso_rechaza_imagenes_que_no_son_rgb(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="RGB"):
        preprocess(np.zeros(shape, np.uint8), 8, 8)


# --- decodificación ----------------------------------------------------------------------


def test_las_columnas_siguen_el_orden_de_las_categorias_del_dataset() -> None:
    assert dict(LABELS) == {0: "car", 1: "wheel", 2: "plate", 3: "face"}


def test_decodifica_las_cajas_a_pixeles_de_la_foto() -> None:
    dets, labels = outputs((CAR, 0.95, (0.5, 0.5, 0.4, 0.5)))

    (box,) = decode_boxes(dets, labels, (40, 80), THRESHOLDS)

    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((24.0, 10.0, 56.0, 30.0), abs=1e-3)
    assert box.label == BoxLabel.CAR
    assert box.conf == pytest.approx(0.95, abs=1e-4)


def test_cada_clase_tiene_su_umbral() -> None:
    dets, labels = outputs(*FULL)

    kept = {box.label for box in decode_boxes(dets, labels, (40, 80), THRESHOLDS)}
    strict = {box.label for box in decode_boxes(dets, labels, (40, 80), dict.fromkeys(THRESHOLDS, 0.5))}

    assert kept == {BoxLabel.CAR, BoxLabel.WHEEL, BoxLabel.PLATE, FACE}
    assert strict == {BoxLabel.CAR, BoxLabel.WHEEL}


def test_ordena_de_mas_a_menos_confiable() -> None:
    dets, labels = outputs(*FULL)

    boxes = decode_boxes(dets, labels, (40, 80), THRESHOLDS)

    assert [round(box.conf, 2) for box in boxes] == [0.95, 0.70, 0.15, 0.12]


def test_ignora_la_columna_de_fondo_y_las_clases_desconocidas() -> None:
    dets, labels = outputs((4, 0.99, (0.5, 0.5, 0.2, 0.2)))  # la columna 4 no es ninguna clase

    assert decode_boxes(dets, labels, (40, 80), THRESHOLDS) == []


def test_recorta_las_cajas_a_la_foto_y_descarta_las_degeneradas() -> None:
    dets, labels = outputs((CAR, 0.9, (0.95, 0.5, 0.4, 0.5)), (WHEEL, 0.9, (0.5, 0.5, 0.0, 0.0)))

    (box,) = decode_boxes(dets, labels, (40, 80), THRESHOLDS)

    assert box.x2 == 80.0 and box.x1 > 0.0


def test_una_consulta_puede_dar_dos_clases() -> None:
    dets, labels = outputs((CAR, 0.9, (0.5, 0.5, 0.4, 0.4)))
    labels[0, 0, WHEEL] = logit(0.8)

    boxes = decode_boxes(dets, labels, (40, 80), THRESHOLDS)

    assert {box.label for box in boxes} == {BoxLabel.CAR, BoxLabel.WHEEL}


def test_decodifica_otro_mapeo_de_columnas() -> None:
    dets, labels = outputs((3, 0.9, (0.5, 0.5, 0.4, 0.4)))  # `car` en la columna 3, como en COCO

    (box,) = decode_boxes(dets, labels, (40, 80), THRESHOLDS, names={3: BoxLabel.CAR})

    assert box.label == BoxLabel.CAR


@pytest.mark.parametrize(
    ("dets", "labels"),
    [
        (np.zeros((QUERIES, 4), np.float32), np.zeros((1, QUERIES, CLASSES), np.float32)),
        (np.zeros((1, QUERIES, 5), np.float32), np.zeros((1, QUERIES, CLASSES), np.float32)),
        (np.zeros((1, QUERIES, 4), np.float32), np.zeros((1, QUERIES, 3), np.float32)),  # faltan clases
        (np.zeros((1, QUERIES, 4), np.float32), np.zeros((1, QUERIES + 1, CLASSES), np.float32)),
        (np.full((1, QUERIES, 4), np.nan, np.float32), np.zeros((1, QUERIES, CLASSES), np.float32)),
        (np.zeros((1, QUERIES, 4), np.float32), np.full((1, QUERIES, CLASSES), np.inf, np.float32)),
    ],
)
def test_salidas_invalidas(dets: NDArray[Any], labels: NDArray[Any]) -> None:
    with pytest.raises(ValueError, match="dets|labels|finita"):
        decode_boxes(dets, labels, (40, 80), THRESHOLDS)


# --- núcleo compartido y vistas ----------------------------------------------------------


def test_el_vehiculo_ve_auto_rueda_y_patente_y_el_rostro_solo_rostros() -> None:
    session = FakeSession(outputs(*FULL))
    shared = core(session)

    vehicles = RfDetrVehicleDetector(shared, "v").detect(photo())
    faces = RfDetrFaceDetector(shared, "v").detect(photo())

    assert vehicles is not None and {box.label for box in vehicles} == {"car", "wheel", "plate"}
    assert faces is not None and [box.label for box in faces] == ["face"]


def test_los_dos_puertos_con_la_misma_foto_corren_el_modelo_una_sola_vez() -> None:
    session = FakeSession(outputs(*FULL))
    shared = core(session)
    img = photo()

    RfDetrVehicleDetector(shared, "v").detect(img)
    RfDetrFaceDetector(shared, "v").detect(img)
    assert len(session.calls) == 1

    RfDetrFaceDetector(shared, "v").detect(photo())  # otra foto: vuelve a correr
    assert len(session.calls) == 2


def test_la_lista_devuelta_es_una_copia() -> None:
    shared = core(FakeSession(outputs(*FULL)))
    img = photo()
    first = RfDetrVehicleDetector(shared, "v").detect(img)
    assert first is not None
    first.clear()

    again = RfDetrVehicleDetector(shared, "v").detect(img)

    assert again is not None and len(again) == 3


def test_cada_hilo_recuerda_su_propia_foto() -> None:
    session = FakeSession(outputs(*FULL))
    shared = core(session)
    img = photo()
    shared.detect_all(img)
    seen: list[int] = []

    def other_thread() -> None:
        before = len(session.calls)
        shared.detect_all(img)  # en otro hilo no hay nada recordado
        seen.append(len(session.calls) - before)

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()

    assert seen == [1]


def test_los_dos_puertos_pasan_el_tensor_con_el_nombre_de_entrada() -> None:
    session = FakeSession(outputs(*FULL))

    core(session).detect_all(photo())

    ((names, feed),) = session.calls
    assert tuple(names or ()) == OUTPUT_NAMES
    assert feed["input"].shape == (1, 3, 16, 24)


def test_sin_detecciones_devuelve_lista_vacia_y_no_none() -> None:
    shared = core(FakeSession(outputs()))

    assert RfDetrVehicleDetector(shared, "v").detect(photo()) == []
    assert RfDetrFaceDetector(shared, "v").detect(photo()) == []


def test_si_la_sesion_falla_los_dos_devuelven_none_y_no_loguean_contenido(
    caplog: pytest.LogCaptureFixture,
) -> None:
    shared = core(FakeSession(RuntimeError("detalle interno con datos de la imagen")))

    with caplog.at_level(logging.WARNING):
        assert RfDetrVehicleDetector(shared, "v").detect(photo()) is None
        assert RfDetrFaceDetector(shared, "v").detect(photo()) is None

    assert {record.getMessage() for record in caplog.records} == {"falló la inferencia del detector"}
    assert "detalle interno" not in caplog.text


def test_un_fallo_no_se_recuerda() -> None:
    session = FakeSession(RuntimeError("x"))
    shared = core(session)
    img = photo()
    assert shared.detect_all(img) is None

    session.result = outputs(*FULL)

    assert shared.detect_all(img) is not None


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (0, 10, 3)])
def test_imagen_invalida_devuelve_none_sin_correr_el_modelo(shape: tuple[int, ...]) -> None:
    session = FakeSession(outputs(*FULL))

    assert core(session).detect_all(np.zeros(shape, np.uint8)) is None
    assert session.calls == []


# --- registro: un ONNX diminuto con la misma interfaz que RF-DETR ---------------------------


def tiny_detector(side: int = 32, names: Sequence[str] = OUTPUT_NAMES, classes: int = CLASSES) -> bytes:
    """`(1, 3, side, side)` → `dets` y `labels` constantes: un auto, una rueda, una patente y un rostro."""
    dets, labels = outputs(*FULL)
    labels = labels[..., :classes]
    constants = [dets, labels]
    graph = helper.make_graph(
        [helper.make_node("Identity", [f"const_{i}"], [name]) for i, name in enumerate(names)],
        "tiny_detector",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, side, side])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
            for name, value in zip(names, constants, strict=True)
        ],
        [numpy_helper.from_array(value, f"const_{i}") for i, value in enumerate(constants)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def wrong_model() -> bytes:
    """Un ONNX válido que no es RF-DETR."""
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "wrong",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return bytes(model.SerializeToString())


def write_env(tmp_path: Path, weights: bytes) -> tuple[Path, Path]:
    models_dir = tmp_path / "weights"
    models_dir.mkdir()
    (models_dir / f"{MODEL_NAME}.onnx").write_bytes(weights)
    manifest = tmp_path / "manifest.json"
    entry = {
        "name": MODEL_NAME,
        "version": "de-prueba",
        "ports": ["detector", "face"],
        "file": f"{MODEL_NAME}.onnx",
        "sha256": hashlib.sha256(weights).hexdigest(),
        "license": "Apache-2.0",
        "paper": {"title": "Un paper", "venue": "ICLR", "year": 2026},
        "source": "generado en el test",
        "commercial": "yes",
    }
    manifest.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
    return manifest, models_dir


def test_el_adaptador_esta_registrado_en_los_dos_puertos() -> None:
    assert ("detector", MODEL_NAME) in ADAPTERS and ("face", MODEL_NAME) in ADAPTERS


def test_seleccionado_en_los_dos_puertos_comparte_la_sesion_y_el_nucleo(tmp_path: Path) -> None:
    weights = tiny_detector()
    manifest, models_dir = write_env(tmp_path, weights)

    models = build_models({"detector": MODEL_NAME, "face": MODEL_NAME}, manifest, models_dir)

    sha = hashlib.sha256(weights).hexdigest()
    assert models.versions()["detector"] == models.versions()["face"] == f"{MODEL_NAME}@{sha[:12]}"
    assert models.vehicle._core is models.face._core  # type: ignore[attr-defined]
    img = photo(40, 80)
    vehicles, faces = models.vehicle.detect(img), models.face.detect(img)
    assert vehicles is not None and faces is not None
    assert {box.label for box in vehicles} == {"car", "wheel", "plate"}
    assert [box.label for box in faces] == ["face"]
    assert max(box.x2 for box in vehicles) <= 80 and max(box.y2 for box in vehicles) <= 40


def test_se_puede_seleccionar_en_un_solo_puerto(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_detector())

    models = build_models({"face": MODEL_NAME}, manifest, models_dir)

    assert models.versions()["detector"] == "null"
    faces = models.face.detect(photo())
    assert faces is not None and [box.label for box in faces] == ["face"]


def test_los_umbrales_llegan_por_los_parametros(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_detector())
    params = {"detector_min_conf": 0.99, "redaction_min_conf": 0.5}

    models = build_models({"detector": MODEL_NAME, "face": MODEL_NAME}, manifest, models_dir, params=params)

    assert models.vehicle.detect(photo()) == []  # el auto y la rueda no llegan a 0,99
    assert models.face.detect(photo()) == []  # el rostro (0,12) no llega a 0,5


@pytest.mark.parametrize("params", [{"detector_min_conf": 1.5}, {"redaction_min_conf": -0.1}])
def test_un_umbral_fuera_de_rango_no_deja_arrancar(tmp_path: Path, params: dict[str, float]) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_detector())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"detector": MODEL_NAME}, manifest, models_dir, params=params)


@pytest.mark.parametrize(
    "weights",
    [wrong_model(), tiny_detector(names=("dets", "otra")), tiny_detector(classes=3)],
    ids=["otro modelo", "otra salida", "faltan clases"],
)
def test_un_archivo_que_no_es_el_detector_no_deja_arrancar(tmp_path: Path, weights: bytes) -> None:
    manifest, models_dir = write_env(tmp_path, weights)

    with pytest.raises(ModelSetupError):
        build_models({"detector": MODEL_NAME, "face": MODEL_NAME}, manifest, models_dir)


# --- configuración -------------------------------------------------------------------------


def test_los_umbrales_por_defecto_y_el_de_redaccion_es_el_mas_bajo(base_env: pytest.MonkeyPatch) -> None:
    params = adapter_params(load_settings())

    assert params["detector_min_conf"] == 0.4
    assert (
        params["redaction_min_conf"] == 0.2
    )  # el menor entre FACE_MIN_CONF (0,30) y PLATE_REDACT_MIN_CONF (0,20)

    base_env.setenv("DETECTOR_MIN_CONF", "0.6")
    base_env.setenv("FACE_MIN_CONF", "0.05")
    params = adapter_params(load_settings())
    assert params["detector_min_conf"] == 0.6 and params["redaction_min_conf"] == 0.05


@pytest.mark.parametrize("value", ["-0.1", "1.1", "alto"])
def test_umbral_del_detector_invalido(base_env: pytest.MonkeyPatch, value: str) -> None:
    base_env.setenv("DETECTOR_MIN_CONF", value)

    with pytest.raises(ValueError, match="DETECTOR_MIN_CONF|detector_min_conf"):
        load_settings()


# --- manifest versionado -------------------------------------------------------------------


def test_el_manifest_del_repo_declara_el_detector() -> None:
    entry = load_manifest(REPO_ROOT / "models" / "manifest.json").get(MODEL_NAME)

    assert entry is not None
    assert entry.ports == ("detector", "face")
    assert entry.commercial == Commercial.YES
    assert entry.license == "Apache-2.0"
    assert (entry.paper.venue, entry.paper.year) == ("ICLR", 2026)
    assert entry.sha256 is not None
    assert "xlarge" not in entry.notes.lower().replace("nunca xl", "")


# --- integración con los pesos reales (opt-in: requiere MODELS_DIR) --------------------------


@pytest.mark.models
def test_pesos_reales_corren_sobre_una_foto_sin_nada() -> None:
    models = build_models(
        {"detector": MODEL_NAME, "face": MODEL_NAME},
        REPO_ROOT / "models" / "manifest.json",
        Path(os.environ["MODELS_DIR"]),
    )
    flat = np.full((240, 320, 3), 128, np.uint8)

    assert models.vehicle.detect(flat) == []
    assert models.face.detect(flat) == []
    assert models.versions()["detector"].startswith(f"{MODEL_NAME}@")
