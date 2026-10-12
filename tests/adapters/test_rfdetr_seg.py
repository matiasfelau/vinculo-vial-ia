"""Segmentador RF-DETR-Seg: preproceso, filtro de la clase car, máscaras y registro (con un ONNX diminuto)."""

import hashlib
import json
import logging
import os
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
from vvi_ia.adapters.models.rfdetr_seg import (
    CAR_CLASS_ID,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_NAME,
    OUTPUT_NAMES,
    RfDetrSegCarSegmenter,
    decode_cars,
    preprocess,
    sigmoid,
)
from vvi_ia.config import load_settings
from vvi_ia.main import adapter_params

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = 91  # columnas de labels, indexadas por id de COCO
TRUCK_CLASS_ID = 8
MASK_SIDE = 8
QUERIES = 4
ON, OFF = 6.0, -6.0  # logits de confianza alta y baja (sigmoid ≈ 0.998 y 0.002)


def logits(*rows: tuple[int, float]) -> NDArray[np.float32]:
    """`labels` de una imagen: cada fila es `(clase, logit)` de una consulta; el resto en OFF."""
    out = np.full((1, QUERIES, CLASSES), OFF, np.float32)
    for query, (cls, value) in enumerate(rows):
        out[0, query, cls] = value
    return out


def mask_logits(*halves: str) -> NDArray[np.float32]:
    """`masks`: por consulta, 'left', 'right', 'all' o 'none' (todo negativo)."""
    out = np.full((1, QUERIES, MASK_SIDE, MASK_SIDE), -1.0, np.float32)
    for query, half in enumerate(halves):
        if half in {"left", "all"}:
            out[0, query, :, : MASK_SIDE // 2] = 1.0
        if half in {"right", "all"}:
            out[0, query, :, MASK_SIDE // 2 :] = 1.0
    return out


class FakeSession:
    """Sesión falsa: devuelve las salidas configuradas y guarda lo que recibió."""

    def __init__(self, outputs: Sequence[NDArray[Any]] | Exception) -> None:
        self.outputs = outputs
        self.calls: list[tuple[Sequence[str] | None, Mapping[str, NDArray[Any]]]] = []

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[Any]]
    ) -> list[NDArray[Any]]:
        self.calls.append((output_names, input_feed))
        if isinstance(self.outputs, Exception):
            raise self.outputs
        return list(self.outputs)


def segmenter(
    labels: NDArray[Any], masks: NDArray[Any], min_conf: float = 0.5, size: tuple[int, int] = (16, 24)
) -> tuple[RfDetrSegCarSegmenter, FakeSession]:
    session = FakeSession([np.zeros((1, QUERIES, 4), np.float32), labels, masks])
    return RfDetrSegCarSegmenter(session, "input", size, min_conf, f"{MODEL_NAME}@abc"), session


def photo(height: int = 30, width: int = 50) -> NDArray[np.uint8]:
    return np.random.default_rng(3).integers(0, 256, (height, width, 3), dtype=np.uint8)


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


def test_sigmoid_no_desborda_con_logits_extremos() -> None:
    out = sigmoid(np.array([-1e9, 0.0, 1e9], np.float32))

    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


def test_solo_devuelve_autos_con_la_mascara_del_tamano_de_la_foto() -> None:
    adapter, session = segmenter(logits((CAR_CLASS_ID, ON), (CAR_CLASS_ID, OFF)), mask_logits("all", "all"))

    cars = adapter.cars(photo(30, 50))

    assert cars is not None and len(cars) == 1
    assert cars[0].mask.shape == (30, 50)
    assert cars[0].mask.dtype == np.bool_
    assert cars[0].mask.all()
    assert cars[0].conf == pytest.approx(float(sigmoid(np.float32(ON))), abs=1e-6)
    ((names, feed),) = session.calls
    assert tuple(names or ()) == OUTPUT_NAMES
    assert feed["input"].shape == (1, 3, 16, 24)


def test_ignora_las_otras_clases_aunque_tengan_confianza_alta() -> None:
    rows = logits((TRUCK_CLASS_ID, ON), (1, ON), (CAR_CLASS_ID, ON))
    adapter, _ = segmenter(rows, mask_logits("all", "all", "left"))

    cars = adapter.cars(photo())

    assert cars is not None and len(cars) == 1
    # Es la tercera consulta: su máscara cubre solo la mitad izquierda.
    assert cars[0].mask[:, :20].all() and not cars[0].mask[:, 30:].any()


def test_ordena_por_confianza_de_mayor_a_menor() -> None:
    adapter, _ = segmenter(
        logits((CAR_CLASS_ID, 1.0), (CAR_CLASS_ID, 4.0), (CAR_CLASS_ID, 2.0)),
        mask_logits("left", "right", "all"),
    )

    cars = adapter.cars(photo())

    assert cars is not None
    assert [round(c.conf, 2) for c in cars] == [0.98, 0.88, 0.73]
    assert cars[0].mask[:, 30:].all() and not cars[0].mask[:, :20].any()  # la de logit 4 es la derecha


def test_el_umbral_viene_de_la_configuracion_del_adaptador() -> None:
    rows = logits((CAR_CLASS_ID, 1.0), (CAR_CLASS_ID, 4.0))  # confianza 0.73 y 0.98
    masks = mask_logits("all", "all")

    for min_conf, expected in [(0.5, 2), (0.9, 1), (0.99, 0), (0.0, 2)]:
        adapter, _ = segmenter(rows, masks, min_conf=min_conf)
        cars = adapter.cars(photo())
        assert cars is not None and len(cars) == expected


def test_sin_autos_devuelve_lista_vacia_y_no_none() -> None:
    adapter, _ = segmenter(logits(), mask_logits())

    assert adapter.cars(photo()) == []


def test_un_auto_con_mascara_vacia_se_descarta() -> None:
    adapter, _ = segmenter(logits((CAR_CLASS_ID, ON), (CAR_CLASS_ID, ON)), mask_logits("none", "left"))

    cars = adapter.cars(photo())

    assert cars is not None and len(cars) == 1
    assert cars[0].mask.any()


def test_la_mascara_se_escala_sin_confundir_alto_y_ancho() -> None:
    adapter, _ = segmenter(logits((CAR_CLASS_ID, ON)), mask_logits("left"))

    cars = adapter.cars(photo(height=20, width=80))

    assert cars is not None
    mask = cars[0].mask
    assert mask.shape == (20, 80)
    assert mask[:, :35].all() and not mask[:, 45:].any()


def test_decode_cars_no_modifica_sus_entradas() -> None:
    labels, masks = logits((CAR_CLASS_ID, ON)), mask_logits("all")
    before = labels.copy(), masks.copy()

    decode_cars(labels, masks, (10, 10), 0.5)

    np.testing.assert_array_equal(labels, before[0])
    np.testing.assert_array_equal(masks, before[1])


# --- errores: None, nunca una excepción ni contenido en logs --------------------------------


@pytest.mark.parametrize(
    ("labels", "masks"),
    [
        (np.zeros((1, QUERIES, 3), np.float32), mask_logits()),  # sin la columna de car
        (np.zeros((QUERIES, CLASSES), np.float32), mask_logits()),  # sin eje de lote
        (logits(), np.zeros((1, QUERIES, MASK_SIDE), np.float32)),  # masks sin un eje
        (logits(), np.zeros((1, QUERIES + 1, MASK_SIDE, MASK_SIDE), np.float32)),  # otras consultas
        (np.full((1, QUERIES, CLASSES), np.nan, np.float32), mask_logits()),
        (logits(), np.full((1, QUERIES, MASK_SIDE, MASK_SIDE), np.inf, np.float32)),
    ],
)
def test_salida_invalida_devuelve_none(labels: NDArray[Any], masks: NDArray[Any]) -> None:
    adapter, _ = segmenter(labels, masks)

    assert adapter.cars(photo()) is None


@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 4), (0, 10, 3)])
def test_imagen_invalida_devuelve_none_sin_correr_el_modelo(shape: tuple[int, ...]) -> None:
    adapter, session = segmenter(logits(), mask_logits())

    assert adapter.cars(np.zeros(shape, np.uint8)) is None
    assert session.calls == []


def test_si_la_sesion_falla_devuelve_none_y_no_loguea_contenido(caplog: pytest.LogCaptureFixture) -> None:
    adapter = RfDetrSegCarSegmenter(
        FakeSession(RuntimeError("detalle interno con datos de la imagen")), "input", (16, 24), 0.5, "v"
    )

    with caplog.at_level(logging.WARNING):
        assert adapter.cars(photo()) is None

    (record,) = caplog.records
    assert record.getMessage() == "falló la inferencia del segmentador"
    assert record.error == "RuntimeError"  # type: ignore[attr-defined]
    assert "detalle interno" not in caplog.text


# --- registro: un ONNX diminuto con la misma interfaz que RF-DETR-Seg ---------------------


def tiny_rfdetr_seg(side: int = 32, outputs: Sequence[str] = OUTPUT_NAMES) -> bytes:
    """`(1, 3, side, side)` → `dets`, `labels` y `masks` constantes: una consulta auto (mitad izquierda)."""
    constants = [
        np.zeros((1, QUERIES, 4), np.float32),
        logits((CAR_CLASS_ID, ON)),
        mask_logits("left"),
    ]
    initializers = [numpy_helper.from_array(value, f"const_{index}") for index, value in enumerate(constants)]
    nodes = [helper.make_node("Identity", [f"const_{index}"], [name]) for index, name in enumerate(outputs)]
    graph = helper.make_graph(
        nodes,
        "tiny_rfdetr_seg",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, side, side])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
            for name, value in zip(outputs, constants, strict=True)
        ],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def wrong_model() -> bytes:
    """Un ONNX válido que no es RF-DETR-Seg."""
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
        "ports": ["segmenter"],
        "file": f"{MODEL_NAME}.onnx",
        "sha256": hashlib.sha256(weights).hexdigest(),
        "license": "Apache-2.0",
        "paper": {"title": "Un paper", "venue": "ICLR", "year": 2026},
        "source": "generado en el test",
        "commercial": "yes",
    }
    manifest.write_text(json.dumps({"schema_version": 1, "models": [entry]}), encoding="utf-8")
    return manifest, models_dir


def test_el_adaptador_esta_registrado_en_el_puerto_segmenter() -> None:
    assert ("segmenter", MODEL_NAME) in ADAPTERS


def test_seleccionado_por_entorno_carga_la_sesion_y_segmenta(tmp_path: Path) -> None:
    weights = tiny_rfdetr_seg()
    manifest, models_dir = write_env(tmp_path, weights)

    models = build_models({"segmenter": MODEL_NAME}, manifest, models_dir)

    sha = hashlib.sha256(weights).hexdigest()
    assert models.versions()["segmenter"] == f"{MODEL_NAME}@{sha[:12]}"
    cars = models.cars.cars(photo(40, 60))
    assert cars is not None and len(cars) == 1
    assert cars[0].mask.shape == (40, 60)
    assert cars[0].mask[:, :25].all() and not cars[0].mask[:, 35:].any()


def test_el_umbral_llega_al_adaptador_por_los_parametros(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_rfdetr_seg())

    params = {"segmenter_min_conf": 0.999}
    strict = build_models({"segmenter": MODEL_NAME}, manifest, models_dir, params=params)

    assert strict.cars.cars(photo()) == []  # la confianza del auto es ≈ 0.9975


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_un_umbral_fuera_de_rango_no_deja_arrancar(tmp_path: Path, value: float) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_rfdetr_seg())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"segmenter": MODEL_NAME}, manifest, models_dir, params={"segmenter_min_conf": value})


def test_un_archivo_que_no_es_rfdetr_seg_no_deja_arrancar(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, wrong_model())

    with pytest.raises(ModelSetupError, match="no se pudo crear"):
        build_models({"segmenter": MODEL_NAME}, manifest, models_dir)


def test_una_salida_con_otro_nombre_no_deja_arrancar(tmp_path: Path) -> None:
    manifest, models_dir = write_env(tmp_path, tiny_rfdetr_seg(outputs=("dets", "labels", "otra")))

    with pytest.raises(ModelSetupError):
        build_models({"segmenter": MODEL_NAME}, manifest, models_dir)


# --- configuración -------------------------------------------------------------------------


def test_el_umbral_por_defecto_es_0_5_y_se_pasa_a_los_adaptadores(base_env: pytest.MonkeyPatch) -> None:
    assert adapter_params(load_settings())["segmenter_min_conf"] == 0.5

    base_env.setenv("SEGMENTER_MIN_CONF", "0.8")
    assert adapter_params(load_settings())["segmenter_min_conf"] == 0.8


@pytest.mark.parametrize("value", ["-0.1", "1.1", "mucho"])
def test_umbral_del_segmentador_invalido(base_env: pytest.MonkeyPatch, value: str) -> None:
    base_env.setenv("SEGMENTER_MIN_CONF", value)

    with pytest.raises(ValueError):  # noqa: PT011
        load_settings()


# --- manifest versionado -------------------------------------------------------------------


def test_el_manifest_del_repo_declara_el_segmentador() -> None:
    entry = load_manifest(REPO_ROOT / "models" / "manifest.json").get(MODEL_NAME)

    assert entry is not None
    assert entry.ports == ("segmenter",)
    assert entry.commercial == Commercial.YES
    assert entry.license == "Apache-2.0"
    assert (entry.paper.venue, entry.paper.year) == ("ICLR", 2026)
    assert entry.sha256 is not None
    # Nunca XL ni 2XL (PML 1.0)
    assert "xlarge" not in entry.file.lower() and "2xl" not in entry.file.lower()


# --- integración con los pesos reales (opt-in: requiere MODELS_DIR) --------------------------


@pytest.mark.models
def test_pesos_reales_corren_sobre_una_foto_sin_autos() -> None:
    models = build_models(
        {"segmenter": MODEL_NAME}, REPO_ROOT / "models" / "manifest.json", Path(os.environ["MODELS_DIR"])
    )
    flat = np.full((240, 320, 3), 128, np.uint8)

    cars = models.cars.cars(flat)

    assert cars == []
    assert models.versions()["segmenter"].startswith(f"{MODEL_NAME}@")
