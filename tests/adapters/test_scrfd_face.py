"""Detector de rostros SCRFD: preproceso, decodificación, mosaico y registro con un ONNX diminuto."""

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper

from vvi_ia.adapters.models.registry import ADAPTERS, ModelSetupError, build_models
from vvi_ia.adapters.models.scrfd_face import (
    INPUT_SIZE,
    MODEL_NAME,
    ScrfdFaceDetector,
    decode,
    letterbox,
    preprocess,
)
from vvi_ia.adapters.models.tiling import Tiling

CELLS = {8: 80, 16: 40, 32: 20}
ANCHORS = 2
Face = tuple[int, int, int, int, float, tuple[float, float, float, float]]


def outputs_with(*faces: Face) -> list[NDArray[Any]]:
    """Las 9 salidas de SCRFD en 640x640; cara: `(stride, fila, columna, ancla, confianza, distancias)`."""
    scores = {s: np.zeros((c * c * ANCHORS, 1), np.float32) for s, c in CELLS.items()}
    boxes = {s: np.zeros((c * c * ANCHORS, 4), np.float32) for s, c in CELLS.items()}
    for stride, row, column, anchor, confidence, distances in faces:
        index = (row * CELLS[stride] + column) * ANCHORS + anchor
        scores[stride][index, 0] = confidence
        boxes[stride][index] = distances
    points = {s: np.zeros((c * c * ANCHORS, 10), np.float32) for s, c in CELLS.items()}
    return [*scores.values(), *boxes.values(), *points.values()]


class FakeSession:
    input_names = ("input.1",)

    def __init__(self, result: Sequence[NDArray[Any]] | Exception, outputs: int = 9) -> None:
        self.result = result
        self.output_names = tuple(str(i) for i in range(outputs))
        self.calls: list[tuple[int, ...]] = []

    def run(self, output_names: Any, input_feed: Mapping[str, NDArray[Any]]) -> list[NDArray[Any]]:
        self.calls.append(input_feed["input.1"].shape)
        if isinstance(self.result, Exception):
            raise self.result
        return list(self.result)


def photo(height: int = 120, width: int = 200) -> NDArray[np.uint8]:
    return np.random.default_rng(1).integers(0, 256, (height, width, 3), dtype=np.uint8)


# --- preproceso --------------------------------------------------------------------------


def test_el_lienzo_deja_la_foto_arriba_a_la_izquierda_con_relleno_en_cero() -> None:
    canvas, scale = letterbox(np.full((100, 200, 3), 255, np.uint8), 64)

    assert scale == pytest.approx(0.32)
    assert canvas.shape == (64, 64, 3)
    assert canvas[:32, :64].min() == 255 and canvas[40:, :].max() == 0


def test_preproceso_normaliza_como_insightface() -> None:
    tensor, scale = preprocess(np.full((INPUT_SIZE, INPUT_SIZE, 3), 255, np.uint8))

    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE) and tensor.dtype == np.float32 and scale == 1.0
    np.testing.assert_allclose(tensor, (255 - 127.5) / 128.0, atol=1e-6)


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (0, 10, 3)])
def test_imagen_invalida(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="RGB"):
        preprocess(np.zeros(shape, np.uint8))


# --- decodificación ----------------------------------------------------------------------


def test_decodifica_la_caja_desde_el_centro_del_ancla_y_las_distancias() -> None:
    (det,) = decode(outputs_with((8, 10, 20, 0, 0.9, (2, 3, 4, 5))), scale=1.0, threshold=0.5)

    np.testing.assert_allclose(det, [144, 56, 192, 120, 0.9], atol=1e-4)  # centro (160, 80), strides de 8


def test_escala_las_cajas_a_la_foto_original() -> None:
    (det,) = decode(outputs_with((16, 5, 5, 1, 0.8, (1, 1, 1, 1))), scale=0.5, threshold=0.5)

    expected = [(80 - 16) / 0.5, (80 - 16) / 0.5, (80 + 16) / 0.5, (80 + 16) / 0.5, 0.8]
    np.testing.assert_allclose(det, expected, atol=1e-3)


def test_el_umbral_descarta_y_el_nms_une_cajas_que_se_pisan() -> None:
    faces = (
        (8, 10, 20, 0, 0.9, (2, 2, 2, 2)),
        (8, 10, 20, 1, 0.6, (2, 2, 2, 2)),  # la misma caja en la otra ancla
        (32, 2, 2, 0, 0.1, (1, 1, 1, 1)),
    )

    assert len(decode(outputs_with(*faces), 1.0, threshold=0.05)) == 2
    kept = decode(outputs_with(*faces), 1.0, threshold=0.5)
    assert kept.shape == (1, 5) and kept[0, 4] == pytest.approx(0.9)


def test_sin_caras_devuelve_una_matriz_vacia() -> None:
    assert decode(outputs_with(), 1.0, threshold=0.5).shape == (0, 5)


@pytest.mark.parametrize("breakage", ["salidas", "tamano", "nan"])
def test_salidas_invalidas(breakage: str) -> None:
    result = outputs_with()
    if breakage == "salidas":
        result = result[:5]
    elif breakage == "tamano":
        result[0] = result[0][:100]
    else:
        result[3][0, 0] = np.nan

    with pytest.raises(ValueError, match="salidas|finita"):
        decode(result, 1.0)


# --- adaptador ---------------------------------------------------------------------------


def detector(session: FakeSession, min_conf: float = 0.3, tiling: Tiling | None = None) -> ScrfdFaceDetector:
    return ScrfdFaceDetector(session, "input.1", min_conf, f"{MODEL_NAME}@abc", tiling=tiling)


def test_detect_devuelve_cajas_de_rostro_en_pixeles_de_la_foto() -> None:
    session = FakeSession(outputs_with((8, 10, 20, 0, 0.9, (2, 3, 4, 5))))

    boxes = detector(session).detect(photo(320, 640))  # el lado mayor mide 640: escala 1,0

    assert boxes is not None and len(boxes) == 1
    assert (boxes[0].label, round(boxes[0].conf, 2)) == ("face", 0.9)
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (144.0, 56.0, 192.0, 120.0)
    assert session.calls == [(1, 3, INPUT_SIZE, INPUT_SIZE)]


def test_las_cajas_se_recortan_a_la_foto() -> None:
    session = FakeSession(outputs_with((8, 0, 0, 0, 0.9, (5, 5, 5, 5))))

    (box,) = detector(session).detect(photo(100, 100)) or []

    assert (box.x1, box.y1) == (0.0, 0.0)


def test_el_umbral_es_el_del_adaptador() -> None:
    result = outputs_with((8, 10, 20, 0, 0.4, (2, 2, 2, 2)))

    assert len(detector(FakeSession(result), 0.3).detect(photo()) or []) == 1
    assert detector(FakeSession(result), 0.5).detect(photo()) == []


def test_si_la_sesion_falla_devuelve_none_y_no_loguea_contenido(caplog: pytest.LogCaptureFixture) -> None:
    session = FakeSession(RuntimeError("detalle interno con datos de la imagen"))

    with caplog.at_level(logging.WARNING):
        assert detector(session).detect(photo()) is None

    (record,) = caplog.records
    assert record.getMessage() == "falló la inferencia del detector de rostros"
    assert record.error == "RuntimeError"  # type: ignore[attr-defined]
    assert "detalle interno" not in caplog.text


def test_con_mosaico_corre_la_foto_completa_y_cada_tile() -> None:
    session = FakeSession(outputs_with((8, 10, 20, 0, 0.9, (2, 2, 2, 2))))
    tiling = Tiling(frozenset({"face"}), tile=640, overlap=0.25)

    boxes = detector(session, tiling=tiling).detect(photo(1000, 1000))

    assert boxes is not None and len(boxes) >= 1
    assert len(session.calls) == 1 + 4  # la foto completa y 4 tiles de 640 (la foto cabe en 2 x 2)


def test_el_mosaico_achica_las_fotos_enormes_antes_de_cortar() -> None:
    session = FakeSession(outputs_with())
    tiling = Tiling(frozenset({"face"}), tile=640, overlap=0.25, max_side=1000)

    detector(session, tiling=tiling).detect(photo(3000, 4000))  # 4000 x 3000 → 1000 x 750: 2 x 2 tiles

    assert len(session.calls) == 1 + 4


def test_una_foto_chica_no_se_mosaiquea() -> None:
    session = FakeSession(outputs_with())

    detector(session, tiling=Tiling(frozenset({"face"}))).detect(photo(300, 300))

    assert len(session.calls) == 1


# --- registro: un ONNX diminuto con las 9 salidas de SCRFD ---------------------------------


def tiny_scrfd(outputs: int = 9) -> bytes:
    values = outputs_with((8, 10, 20, 0, 0.9, (2, 3, 4, 5)))[:outputs]
    names = [f"out{i}" for i in range(len(values))]
    graph = helper.make_graph(
        [helper.make_node("Identity", [f"const_{i}"], [name]) for i, name in enumerate(names)],
        "tiny_scrfd",
        [helper.make_tensor_value_info("input.1", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, list(v.shape))
            for n, v in zip(names, values, strict=True)
        ],
        [numpy_helper.from_array(v, f"const_{i}") for i, v in enumerate(values)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def write_env(tmp_path: Path, weights: bytes, commercial: str = "escalable") -> tuple[Path, Path]:
    models_dir = tmp_path / "weights"
    models_dir.mkdir()
    (models_dir / f"{MODEL_NAME}.onnx").write_bytes(weights)
    manifest = tmp_path / "manifest.json"
    entry = {
        "name": MODEL_NAME,
        "version": "de-prueba",
        "ports": ["face"],
        "file": f"{MODEL_NAME}.onnx",
        "sha256": hashlib.sha256(weights).hexdigest(),
        "license": "InsightFace",
        "paper": {"title": "Un paper", "venue": "ICLR", "year": 2022},
        "source": "generado en el test",
        "commercial": commercial,
    }
    manifest.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
    return manifest, models_dir


def test_el_adaptador_esta_registrado_en_el_puerto_face() -> None:
    assert ("face", MODEL_NAME) in ADAPTERS


def test_seleccionado_por_entorno_carga_la_sesion_y_detecta(tmp_path: Path) -> None:
    weights = tiny_scrfd()
    manifest, models_dir = write_env(tmp_path, weights)

    models = build_models({"face": MODEL_NAME}, manifest, models_dir)

    assert models.versions()["face"] == f"{MODEL_NAME}@{hashlib.sha256(weights).hexdigest()[:12]}"
    boxes = models.face.detect(photo(640, 640))  # el ONNX diminuto devuelve siempre la misma cara
    assert boxes is not None and len(boxes) == 1 and boxes[0].label == "face"


def test_el_umbral_y_el_mosaico_llegan_por_los_parametros(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_scrfd())

    strict = build_models({"face": MODEL_NAME}, manifest, models_dir, params={"redaction_min_conf": 0.95})
    tiled = build_models({"face": MODEL_NAME}, manifest, models_dir, params={"face_tiling": 1.0})

    assert strict.face.detect(photo(640, 640)) == []
    assert tiled.face._tiling is not None  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_un_umbral_fuera_de_rango_no_deja_arrancar(tmp_path: Path, value: float) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_scrfd())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"face": MODEL_NAME}, manifest, models_dir, params={"redaction_min_conf": value})


def test_un_archivo_que_no_es_scrfd_no_deja_arrancar(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_scrfd(outputs=4))

    with pytest.raises(ModelSetupError):
        build_models({"face": MODEL_NAME}, manifest, models_dir)


def test_scrfd_se_puede_seleccionar_aunque_el_manifest_diga_commercial_no(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_scrfd(), commercial="no")

    assert build_models({"face": MODEL_NAME}, manifest, models_dir).versions()["face"].startswith(MODEL_NAME)
